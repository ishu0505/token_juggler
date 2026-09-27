-- Pick a deployment AND reserve its quota, atomically, in one round trip.
--
-- Every limit is a GCRA token bucket: one number per bucket, the "theoretical
-- arrival time" (tat, ms). A bucket with capacity C over a window of P ms
-- emits one unit every T = P / C ms. Spending n units moves tat forward by
-- n * T; the spend is allowed while tat stays within P of now. This refills
-- continuously, so there is no window boundary to burst across.
--
-- ARGV[1] is JSON:
--   c    candidates, in preference order:
--        { cool = key index or 0,
--          b    = { bucket, ... }     required: ALL must have room
--          s    = { {bucket, ...}, ... } optional sources: the FIRST group
--                 with room pays (own reserve, shared pool, lent reserves)
--          conc = {key index, max} or nil }
--        where bucket = {key index, T ms/unit, units, P ms}
--   mode "first" (take the first that fits) | "least" (the least-utilised fit)
--   rr   key index of a round-robin counter, 0 for none
--   m    request id (concurrency lease member)
--   l    lease length in ms
--
-- Returns JSON {ok = candidate index, src = source index or 0, now = ms} on
-- success, else {ok = 0, wait = ms until something may fit (-1: never), why}.

local spec = cjson.decode(ARGV[1])
local t = redis.call('TIME')
local now = tonumber(t[1]) * 1000 + math.floor(tonumber(t[2]) / 1000)
local cands = spec.c
local n = #cands

local order = {}
for i = 1, n do order[i] = i end
if spec.rr and spec.rr > 0 and n > 1 then
  local offset = redis.call('INCR', KEYS[spec.rr])
  redis.call('PEXPIRE', KEYS[spec.rr], 86400000)
  local shift = (offset - 1) % n
  for i = 1, n do order[i] = ((i - 1 + shift) % n) + 1 end
end

-- Check a list of buckets. Returns (tats, score) when all fit, else
-- (nil, index of the first that doesn't, wait ms or -1 for never).
local function fit(buckets)
  local tats = {}
  local score = 0
  for j, b in ipairs(buckets) do
    local cost = b[3] * b[2]
    local window = b[4]
    if cost > window then return nil, j, -1 end
    local tat = tonumber(redis.call('GET', KEYS[b[1]]) or '0')
    if tat < now then tat = now end
    local new_tat = tat + cost
    if new_tat - now > window then return nil, j, new_tat - now - window end
    tats[j] = new_tat
    local utilisation = (new_tat - now) / window
    if utilisation > score then score = utilisation end
  end
  return tats, score
end

-- Returns (plan, score) when the candidate fits, else (nil, reason, wait_ms).
local function evaluate(ci)
  local c = cands[ci]
  if c.cool and c.cool > 0 then
    local ttl = redis.call('PTTL', KEYS[c.cool])
    if ttl > 0 then return nil, 'cooldown', ttl end
  end
  local tats, a, b = fit(c.b)
  if not tats then
    return nil, (b < 0 and 'too_large:' or 'full:') .. a, b
  end
  local score = a
  local src, src_tats = 0, nil
  if c.sr == 1 and (not c.s or #c.s == 0) then return nil, 'reserved', -1 end
  if c.s and #c.s > 0 then
    local best_wait, first_fail = -1, nil
    for g, group in ipairs(c.s) do
      local gt, ga, gb = fit(group)
      if gt then
        src, src_tats = g, gt
        if ga > score then score = ga end
        break
      end
      if not first_fail then first_fail = g .. ':' .. ga end
      if gb >= 0 and (best_wait < 0 or gb < best_wait) then best_wait = gb end
    end
    if src == 0 then return nil, 'no_source:' .. first_fail, best_wait end
  end
  if c.conc then
    local key = KEYS[c.conc[1]]
    redis.call('ZREMRANGEBYSCORE', key, '-inf', now)
    if redis.call('ZCARD', key) >= c.conc[2] then return nil, 'busy', 50 end
  end
  return {tats = tats, src = src, src_tats = src_tats}, score
end

local function write(buckets, tats)
  for j, b in ipairs(buckets) do
    local ttl = math.ceil(tats[j] - now) + 1000
    redis.call('SET', KEYS[b[1]], string.format('%.3f', tats[j]), 'PX', ttl)
  end
end

local function commit(ci, plan)
  local c = cands[ci]
  write(c.b, plan.tats)
  if plan.src > 0 then write(c.s[plan.src], plan.src_tats) end
  if c.conc then
    local key = KEYS[c.conc[1]]
    redis.call('ZADD', key, now + spec.l, spec.m)
    redis.call('PEXPIRE', key, spec.l + 60000)
  end
end

local why = {}
for i = 1, n do why[i] = '' end
local min_wait = -1
local best, best_plan, best_score = 0, nil, nil

for _, ci in ipairs(order) do
  local plan, a, b = evaluate(ci)
  if plan then
    if spec.mode ~= 'least' then
      commit(ci, plan)
      return cjson.encode({ok = ci, src = plan.src, now = now})
    end
    if best_score == nil or a < best_score then
      best, best_plan, best_score = ci, plan, a
    end
  else
    why[ci] = a
    if b >= 0 and (min_wait < 0 or b < min_wait) then min_wait = b end
  end
end

if best > 0 then
  commit(best, best_plan)
  return cjson.encode({ok = best, src = best_plan.src, now = now})
end
return cjson.encode({ok = 0, wait = min_wait, why = why})
