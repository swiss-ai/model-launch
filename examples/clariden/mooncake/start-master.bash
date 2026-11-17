#!/bin/bash
# Sourced by the example's pre-launch commands. The worker owns the RAM pool.
mooncake_example_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export MOONCAKE_CONFIG_PATH="$mooncake_example_dir/mooncake.json"
export SGLANG_HICACHE_MOONCAKE_CONFIG_PATH="$MOONCAKE_CONFIG_PATH"

# Refuse occupied ports rather than connect this example to another service.
python3 - <<'PY'
import socket

for port in (15051, 19003):
    with socket.socket() as connection:
        connection.settimeout(1)
        if connection.connect_ex(("127.0.0.1", port)) == 0:
            raise RuntimeError(f"Mooncake example port {port} is already occupied")
PY

mooncake_master --rpc_port=15051 --rpc_address=127.0.0.1 \
  --metrics_port=19003 --metrics_host=127.0.0.1 \
  --enable_http_metadata_server=false --enable_offload=false --root_fs_dir= \
  --enable_oplog=false --enable_oplog_snapshot=false --enable_ha=false &
mooncake_master_pid=$!

stop_mooncake_master() {
  kill "$mooncake_master_pid" 2>/dev/null || true
  wait "$mooncake_master_pid" 2>/dev/null || true
}
trap stop_mooncake_master EXIT

mooncake_ready=false
for mooncake_attempt in {1..60}; do
  if ! kill -0 "$mooncake_master_pid" 2>/dev/null; then
    echo "Mooncake master exited during startup" >&2
    return 1
  fi
  if curl --noproxy '*' -fsS --max-time 1 http://127.0.0.1:19003/get_all_segments >/dev/null 2>&1; then
    mooncake_ready=true
    break
  fi
  sleep 1
done
if [[ "$mooncake_ready" != true ]]; then
  echo "Mooncake master did not become ready after $mooncake_attempt attempts" >&2
  return 1
fi
