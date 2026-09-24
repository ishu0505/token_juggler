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
--          b    = { {key index, T ms/unit, units, P ms}, ... },
--          conc = {key index, max} or nil }
--   mode "first" (take the first that fits) | "least" (the least-utilised fit)
--   rr   key index of a round-robin counter, 0 for none
--   m    request id (concurrency lease member)
--   l    lease length in ms
--
-- Returns JSON {ok = candidate index, now = ms} on success, else
-- {ok = 0, wait = ms until something may fit (-1: never), why = {...}}.

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

-- Returns (tats, score) when the candidate fits, else (nil, reason, wait_ms).
local function evaluate(ci)
  local c = cands[ci]
  if c.cool and c.cool > 0 then
    local ttl = redis.call('PTTL', KEYS[c.cool])
    if ttl > 0 then return nil, 'cooldown', ttl end
  end
  local tats = {}
  local score = 0
  for j, b in ipairs(c.b) do
    local cost = b[3] * b[2]
    local window = b[4]
    if cost > window then return nil, 'too_large:' .. j, -1 end
    local tat = tonumber(redis.call('GET', KEYS[b[1]]) or '0')
    if tat < now then tat = now end
    local new_tat = tat + cost
    if new_tat - now > window then
      return nil, 'full:' .. j, new_tat - now - window
    end
    tats[j] = new_tat
    local utilisation = (new_tat - now) / window
    if utilisation > score then score = utilisation end
  end
  if c.conc then
    local key = KEYS[c.conc[1]]
    redis.call('ZREMRANGEBYSCORE', key, '-inf', now)
    if redis.call('ZCARD', key) >= c.conc[2] then return nil, 'busy', 50 end
  end
  return tats, score
end

local function commit(ci, tats)
  local c = cands[ci]
  for j, b in ipairs(c.b) do
    local ttl = math.ceil(tats[j] - now) + 1000
    redis.call('SET', KEYS[b[1]], string.format('%.3f', tats[j]), 'PX', ttl)
  end
  if c.conc then
    local key = KEYS[c.conc[1]]
    redis.call('ZADD', key, now + spec.l, spec.m)
    redis.call('PEXPIRE', key, spec.l + 60000)
  end
end

local why = {}
for i = 1, n do why[i] = '' end
local min_wait = -1
local best, best_tats, best_score = 0, nil, nil

for _, ci in ipairs(order) do
  local tats, a, b = evaluate(ci)
  if tats then
    if spec.mode ~= 'least' then
      commit(ci, tats)
      return cjson.encode({ok = ci, now = now})
    end
    if best_score == nil or a < best_score then
      best, best_tats, best_score = ci, tats, a
    end
  else
    why[ci] = a
    if b >= 0 and (min_wait < 0 or b < min_wait) then min_wait = b end
  end
end

if best > 0 then
  commit(best, best_tats)
  return cjson.encode({ok = best, now = now})
end
return cjson.encode({ok = 0, wait = min_wait, why = why})
