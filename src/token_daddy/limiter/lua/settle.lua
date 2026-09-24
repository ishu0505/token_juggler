-- Settle a finished call: correct its reservations, release its concurrency
-- lease, and record usage - one round trip, run after the caller has its answer.
--
-- ARGV[1] is JSON:
--   a   { {key index, T ms/unit, delta units}, ... }  delta = actual - reserved
--   cc  {key index, member} or nil                     concurrency lease to drop
--   u   { k = {key indexes}, ttl = {ms...}, f = {field = int} } or nil  rollups
--   ix  {key index, member} or nil                     usage index set entry
--   x   {key index, maxlen, {field, value, ...}} or nil  call-log stream entry

local spec = cjson.decode(ARGV[1])
local t = redis.call('TIME')
local now = tonumber(t[1]) * 1000 + math.floor(tonumber(t[2]) / 1000)

for _, adj in ipairs(spec.a or {}) do
  local key, per_unit, delta = KEYS[adj[1]], adj[2], adj[3]
  local tat = tonumber(redis.call('GET', key) or '0')
  if delta < 0 then
    -- A refund never pushes a bucket past full: clamp at now.
    tat = math.max(now, tat + delta * per_unit)
  else
    tat = math.max(tat, now) + delta * per_unit
  end
  if tat <= now then
    redis.call('DEL', key)
  else
    redis.call('SET', key, string.format('%.3f', tat), 'PX', math.ceil(tat - now) + 1000)
  end
end

if spec.cc then
  redis.call('ZREM', KEYS[spec.cc[1]], spec.cc[2])
end

if spec.u then
  for i, ki in ipairs(spec.u.k) do
    local key = KEYS[ki]
    for field, value in pairs(spec.u.f) do
      if value ~= 0 then redis.call('HINCRBY', key, field, value) end
    end
    redis.call('PEXPIRE', key, spec.u.ttl[i])
  end
end

if spec.ix then
  redis.call('SADD', KEYS[spec.ix[1]], spec.ix[2])
end

if spec.x then
  redis.call('XADD', KEYS[spec.x[1]], 'MAXLEN', '~', spec.x[2], '*', unpack(spec.x[3]))
end

return now
