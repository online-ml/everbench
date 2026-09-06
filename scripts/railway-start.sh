#!/bin/sh

set -eu

# Railway starts the web and worker services independently. Run migrations in
# the service network before either begins work; the command holds a Postgres
# advisory lock, so concurrent starts cannot race.
.venv/bin/everbench migrate

case "${EVERBENCH_SERVICE_ROLE:-web}" in
    web)
        export EVERBENCH_DB_POOL_SIZE="${EVERBENCH_DB_POOL_SIZE:-3}"
        exec .venv/bin/gunicorn --bind "0.0.0.0:${PORT:-8000}" "everbench.api:create_app()"
        ;;
    worker)
        export EVERBENCH_DB_POOL_SIZE="${EVERBENCH_DB_POOL_SIZE:-6}"
        exec .venv/bin/everbench worker-all --task wiki-liftwing
        ;;
    researcher)
        export EVERBENCH_DB_POOL_SIZE="${EVERBENCH_DB_POOL_SIZE:-2}"
        exec .venv/bin/everbench auto-worker-all --once
        ;;
    *)
        echo "EVERBENCH_SERVICE_ROLE must be 'web', 'worker', or 'researcher'" >&2
        exit 2
        ;;
esac
