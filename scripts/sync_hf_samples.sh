#!/bin/sh
set -eu

repo_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
source_dir="$repo_root/assets/production-core/v1"
destination=${LEXIBEAT_HF_BUCKET:-"hf://buckets/AntonDergunov/LexiBeatSamples/lexibeat-production-core/v1"}

if ! command -v hf >/dev/null 2>&1; then
  echo "The Hugging Face CLI is required: https://hf.co/cli" >&2
  exit 1
fi
if [ ! -f "$source_dir/catalog.sqlite3" ] || \
   [ ! -f "$source_dir/manifest.json" ]; then
  echo "Production bundle is incomplete. Run 'git lfs pull' first." >&2
  exit 1
fi

echo "Synchronizing $source_dir"
echo "Destination: $destination"
hf buckets sync "$source_dir" "$destination" "$@"
