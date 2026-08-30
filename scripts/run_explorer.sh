#!/bin/sh
set -eu

task_cache_dir="${TMPDIR:-/tmp}/lexibeat-explorer-uv-cache"
export UV_CACHE_DIR="$task_cache_dir"

exec uv run --extra explorer python -m lexibeat.explorer_web "$@"
