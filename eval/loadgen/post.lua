-- wrk2 Lua script: POST a pre-built JSON body to /api/predict.
-- Usage:
--   PAYLOAD=/tmp/payload.json wrk2 -t4 -c16 -R200 -d30s \
--     -s eval/loadgen/post.lua http://localhost:11435/api/predict
--
-- -R sets the *constant arrival rate* (open-loop): wrk2 is coordinated-omission-correct, so the
-- reported p99/p99.9 include queueing delay (unlike ab/wrk). Read PAYLOAD from the env so the
-- same script serves any model/image (see eval/loadgen/make_payload.py).

local path = os.getenv("PAYLOAD")
local body = ""
if path ~= nil then
  local f = io.open(path, "rb")
  if f ~= nil then
    body = f:read("*all")
    f:close()
  end
end

wrk.method = "POST"
wrk.headers["Content-Type"] = "application/json"
wrk.body = body

-- Emit the full HdrHistogram latency distribution so eval/loadgen/parse_wrk2.py can extract
-- p50/p95/p99/p99.9 (wrk2's --latency prints the CO-correct distribution).
done = function(summary, latency, requests)
  io.write("----- wrk2 latency (us) -----\n")
  for _, p in pairs({ 50, 90, 95, 99, 99.9, 99.99, 100 }) do
    io.write(string.format("p%g %d\n", p, latency:percentile(p)))
  end
  io.write(string.format("requests %d\n", summary.requests))
  io.write(string.format("duration_us %d\n", summary.duration))
  io.write(string.format("bytes %d\n", summary.bytes))
  io.write(string.format("errors_connect %d\n", summary.errors.connect))
  io.write(string.format("errors_status %d\n", summary.errors.status))
  io.write(string.format("errors_timeout %d\n", summary.errors.timeout))
end
