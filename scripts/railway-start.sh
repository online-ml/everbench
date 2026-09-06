#!/bin/sh

set -eu

# Railway starts the web and worker services independently. Run migrations in
# the service network before either begins work; the command holds a Postgres
# advisory lock, so concurrent starts cannot race.
uv run everbench migrate

case "${EVERBENCH_SERVICE_ROLE:-web}" in
    web)
        exec uv run gunicorn --bind "0.0.0.0:${PORT:-8000}" "everbench.api:create_app()"
        ;;
    worker)
        exec uv run everbench worker-all
        ;;
    researcher)
        exec uv run everbench auto-worker-all
        ;;
    *)
        echo "EVERBENCH_SERVICE_ROLE must be 'web', 'worker', or 'researcher'" >&2
        exit 2
        ;;
esac
