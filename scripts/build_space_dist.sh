#!/bin/sh
set -eu

repo_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
destination="$repo_root/.space-dist"

rm -rf -- "$destination"
mkdir -p "$destination/lexibeat"

cp "$repo_root/app.py" "$destination/app.py"
cp "$repo_root/README.md" "$destination/README.md"
cp "$repo_root/NOTICE.md" "$destination/NOTICE.md"
cp "$repo_root/pyproject.toml" "$destination/pyproject.toml"
cp "$repo_root/requirements.txt" "$destination/requirements.txt"
cp "$repo_root/uv.lock" "$destination/uv.lock"
find "$repo_root/lexibeat" -maxdepth 1 -type f -name '*.py' \
  -exec cp '{}' "$destination/lexibeat/" ';'

test -f "$destination/lexibeat/__init__.py"
test ! -e "$destination/assets"
test ! -e "$destination/.venv"
if grep -Eq '^[[:space:]]*(-e[[:space:]]+)?(\.|file:)' \
    "$destination/requirements.txt"; then
  echo "Space requirements must not reference files installed after this step." >&2
  exit 1
fi

echo "Prepared code-only Space package at $destination"
du -sh "$destination"
