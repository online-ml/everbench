#!/bin/sh

set -eu

case "${EVERBENCH_SERVICE_ROLE:-web}" in
    web)
        exec uv run gunicorn --bind "0.0.0.0:${PORT:-8000}" "everbench.api:create_app()"
        ;;
    worker)
        exec uv run everbench worker-all
        ;;
    *)
        echo "EVERBENCH_SERVICE_ROLE must be 'web' or 'worker'" >&2
        exit 2
        ;;
esac
