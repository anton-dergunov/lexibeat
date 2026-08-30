# LexiBeat Production Sample Bundle

`v1/` is the offline, checksum-locked sample set used by the
`production-v1` generation profile. Audio files and the compact catalog are
stored with Git LFS. `v1/manifest.json` records every logical asset ID, original
path, SHA-256 digest, source collection, license, attribution, and byte size.

The bundle contains only sources whose licenses allow redistribution. Most
assets are dedicated to the public domain under CC0 1.0. Salamander Grand Piano
is CC BY 3.0 and requires the attribution preserved in `NOTICE.md` and the
bundle manifest. Sources that prohibit sample repackaging are not supported.

Rebuild from the explicitly managed local and external libraries—without any
network access—with:

```bash
UV_CACHE_DIR=/tmp/lexibeat-uv-cache \
  uv run python scripts/build_production_bundle.py
```

Verify the installed bundle with:

```bash
UV_CACHE_DIR=/tmp/lexibeat-uv-cache uv run sample_bundle.py verify
```
