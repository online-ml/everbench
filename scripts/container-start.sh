#!/bin/sh

set -eu

export EVERBENCH_DB_POOL_SIZE="${EVERBENCH_DB_POOL_SIZE:-10}"

everbench migrate

set --
for task_name in ${EVERBENCH_TASK_NAMES:-wiki-liftwing}; do
    set -- "$@" --task "$task_name"
done
everbench register-tasks "$@"

if [ "${EVERBENCH_SCHEDULE_RESEARCH:-0}" = "1" ]; then
    set -- "$@" --schedule-research
fi

everbench worker-all "$@" &
worker_pid=$!
gunicorn \
    --access-logfile - \
    --bind "0.0.0.0:${PORT:-8000}" \
    --error-logfile - \
    "everbench.api:create_app()" &
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
