# Superseded music-path cleanup

- Move the bare CLI and voice benchmark defaults to `MusicRequest` production generation.
- Remove `yoga`, `nocturne`, `lofi`, and `warm`, their pattern helpers, and the phrase-less renderer branch.
- Make resolved phrases mandatory and remove production-inactive legacy controls and explorer fields.
- Remove unused Euclidean helpers and audit Step 3B-only grammars, palettes, cyclic round-robin data, and single-sample replay paths.
- Update affected tests, examples, `AGENTS.md`, README/design references, and intentionally reject old phrase-less BedSpec JSON.
- Verify the API, CLI, demos, benchmark, unit suite, and a short synth-only render.
