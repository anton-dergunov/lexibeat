# LexiBeat Production Sample Bundle

`v3/` is the offline, checksum-locked sample set used by the
`production-v1` generation profile: the listener-approved Wave 3 library,
2,440 assets and 3.1 GB. Audio files and the compact catalog are stored with
Git LFS. `v3/manifest.json` records every logical asset ID, original path,
SHA-256 digest, source collection, license, attribution, and byte size, and
carries the `expansion_policy` — the accepted banks, their audited registers
and gains, and the retained Wave 3 families — that the generator reads to
enable them. A bundle without that policy renders the plainer control
behaviour, which is how a host running the earlier 1.96 GB `v1` sounded.

The bundle contains only sources whose licenses allow redistribution. Most
assets are dedicated to the public domain under CC0 1.0. Salamander Grand Piano
is CC BY 3.0 and requires the attribution preserved in `NOTICE.md` and the
bundle manifest. Sources that prohibit sample repackaging are not supported.

It was built in three stages from the explicitly managed external library —
the core bundle (`scripts.samples.build_production_bundle`), then
`sample_library integrate-expansion` (candidate v2), then
`sample_library integrate-wave3 --keep-family … --reject-family …` (this
bundle); the README's sample-library section has the exact commands.

Verify the installed bundle, or ask which one is mounted, with:

```bash
uv run lexibeat-bundle verify
uv run lexibeat-bundle status
```
