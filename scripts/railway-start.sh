#!/bin/sh

set -eu

if [ "${EVERBENCH_SERVICE_ROLE:-combined}" != "combined" ]; then
    echo "EVERBENCH_SERVICE_ROLE must be 'combined'" >&2
    exit 2
fi

export EVERBENCH_DB_POOL_SIZE="${EVERBENCH_DB_POOL_SIZE:-10}"
.venv/bin/everbench migrate
if [ "${EVERBENCH_IMPORT_POSTGRES:-0}" = "1" ]; then
    .venv/bin/everbench import-postgres
fi

set --
for task_name in ${EVERBENCH_TASK_NAMES:-wiki-liftwing}; do
    set -- "$@" --task "$task_name"
done
.venv/bin/everbench register-tasks "$@"

.venv/bin/everbench worker-all --schedule-research "$@" &
worker_pid=$!
.venv/bin/gunicorn --bind "0.0.0.0:${PORT:-8000}" "everbench.api:create_app()" &
web_pid=$!

terminate() {
    trap - TERM INT
    kill "$worker_pid" "$web_pid" 2>/dev/null || true
    wait "$worker_pid" 2>/dev/null || true
    wait "$web_pid" 2>/dev/null || true
    exit 0
}
trap terminate TERM INT

while kill -0 "$worker_pid" 2>/dev/null && kill -0 "$web_pid" 2>/dev/null; do
    sleep 2
done
echo "web or worker exited unexpectedly" >&2
kill "$worker_pid" "$web_pid" 2>/dev/null || true
wait "$worker_pid" 2>/dev/null || true
wait "$web_pid" 2>/dev/null || true
exit 1
