"""The sample bundle: verify it, fetch it, publish it.

The engine works without it and offers only the sample-free ``electronic`` palette, reporting
``production_bundle: false``. With it, every family is available. It is 1.8 GB of content-addressed
audio, so it does not ride a release tarball of source and it does not enter a host's repository:
the host fetches it once into a volume and points ``LEXIBEAT_BUNDLE_ROOT`` at the mount.

Licensing is not why any of this is arranged this way, and an earlier draft said it was. The one
attribution-bearing source is CC-BY 3.0, which permits redistribution; the credit already travels
inside ``licenses/`` and ``NOTICE.md``. Size is the whole reason.

    lexibeat-bundle verify [--root DIR]
    lexibeat-bundle fetch --into DIR --from URL [--sha256 DIGEST]
    lexibeat-bundle publish --out DIR
"""

from __future__ import annotations

import hashlib
import json
import tarfile
import tempfile
from pathlib import Path
from typing import Any, Callable, Iterable

from .paths import REPOSITORY_BUNDLE_ROOT, configured_bundle_root

MANIFEST_NAME = "manifest.json"
CATALOG_NAME = "catalog.sqlite3"
CHECKSUMS_NAME = "SHA256SUMS"
BLOCK = 1024 * 1024


class BundleError(RuntimeError):
    """The bundle is absent, incomplete, or does not match its own manifest."""


def sha256(path: Path, block: int = BLOCK) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(block):
            digest.update(chunk)
    return digest.hexdigest()


def read_manifest(root: Path) -> dict[str, Any]:
    path = root / MANIFEST_NAME
    if not path.is_file():
        raise BundleError(f"No sample bundle at {root}: {MANIFEST_NAME} is missing.")
    return json.loads(path.read_text(encoding="utf-8"))


def manifest_entries(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    return [*manifest.get("catalog_assets", []), *manifest.get("named_pack_assets", [])]


def verify(root: Path | None = None, *,
           progress: Callable[[int, int, str], None] | None = None) -> dict[str, Any]:
    """Check every asset the manifest names against its recorded digest.

    A bundle that is present but wrong is worse than one that is absent: the engine would render
    with substituted audio and a seed would stop replaying. So this reports missing, altered and
    verified counts separately and refuses on either kind of failure.
    """
    root = Path(root) if root else configured_bundle_root()
    manifest = read_manifest(root)
    entries = manifest_entries(manifest)
    missing: list[str] = []
    altered: list[str] = []
    for index, entry in enumerate(entries, 1):
        relative = str(entry["bundle_path"])
        path = root / relative
        if progress:
            progress(index, len(entries), relative)
        if not path.is_file():
            missing.append(relative)
        elif sha256(path) != str(entry["sha256"]):
            altered.append(relative)
    catalog = root / CATALOG_NAME
    if not catalog.is_file():
        missing.append(CATALOG_NAME)
    report = {
        "root": str(root),
        "bundle": manifest.get("bundle"),
        "version": manifest.get("version"),
        "assets": len(entries),
        "verified": len(entries) - len(missing) - len(altered),
        "missing": missing,
        "altered": altered,
        "ok": not missing and not altered,
    }
    if not report["ok"]:
        raise BundleError(
            f"{len(missing)} missing and {len(altered)} altered of {len(entries)} "
            f"bundle assets under {root}.")
    return report


def checksums(paths: Iterable[Path], base: Path) -> str:
    """A `SHA256SUMS` in the format `shasum -c` reads."""
    return "".join(f"{sha256(path)}  {path.relative_to(base).as_posix()}\n"
                   for path in sorted(paths))


def publish(root: Path | None = None, *, out: Path) -> dict[str, Any]:
    """Pack a verified bundle into one archive with its digest, ready to attach to a release."""
    root = Path(root) if root else REPOSITORY_BUNDLE_ROOT
    report = verify(root)
    manifest = read_manifest(root)
    name = f"{manifest.get('bundle', 'lexibeat-bundle')}-v{manifest.get('version', 1)}"
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    archive = out / f"{name}.tar"
    temporary = archive.with_suffix(".partial.tar")
    try:
        with tarfile.open(temporary, "w") as tar:
            tar.add(root, arcname=name, recursive=True, filter=_reproducible)
        temporary.replace(archive)
    finally:
        temporary.unlink(missing_ok=True)
    (out / CHECKSUMS_NAME).write_text(checksums([archive], out), encoding="utf-8")
    return {**report, "archive": str(archive), "bytes": archive.stat().st_size,
            "sha256": sha256(archive)}


def _reproducible(info: tarfile.TarInfo) -> tarfile.TarInfo:
    """Same bytes from the same bundle on any machine, so a published digest means something."""
    info.uid = info.gid = 0
    info.uname = info.gname = ""
    info.mtime = 0
    info.mode = 0o755 if info.isdir() else 0o644
    return info


def fetch(url: str, *, into: Path, expected_sha256: str = "",
          progress: Callable[[int, int], None] | None = None) -> dict[str, Any]:
    """Download and verify a published bundle archive, then extract it into ``into``.

    Nothing is extracted before the digest matches. A bundle half-written by an interrupted
    download is the failure mode worth designing against, because the engine would then render with
    some samples missing rather than refusing.
    """
    import urllib.request

    into = Path(into)
    into.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="lexibeat-bundle-") as workspace:
        archive = Path(workspace) / "bundle.tar"
        with urllib.request.urlopen(url) as response, archive.open("wb") as handle:
            total = int(response.headers.get("content-length") or 0)
            read = 0
            while chunk := response.read(BLOCK):
                handle.write(chunk)
                read += len(chunk)
                if progress:
                    progress(read, total)
        digest = sha256(archive)
        if expected_sha256 and digest != expected_sha256:
            raise BundleError(
                f"The downloaded bundle is {digest}, not the expected {expected_sha256}.")
        with tarfile.open(archive) as tar:
            tar.extractall(into, filter="data")
    roots = [path for path in into.iterdir()
             if path.is_dir() and (path / MANIFEST_NAME).is_file()]
    root = roots[0] if len(roots) == 1 else into
    report = verify(root)
    return {**report, "sha256": digest, "url": url}


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(prog="lexibeat-bundle", description=__doc__.split("\n\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)

    check = commands.add_parser("verify", help="check a bundle against its own manifest")
    check.add_argument("--root", type=Path, default=None)

    get = commands.add_parser("fetch", help="download, verify and extract a published bundle")
    get.add_argument("--into", type=Path, required=True)
    get.add_argument("--from", dest="url", required=True)
    get.add_argument("--sha256", default="")

    put = commands.add_parser("publish", help="pack a bundle and write its SHA256SUMS")
    put.add_argument("--root", type=Path, default=None)
    put.add_argument("--out", type=Path, required=True)

    args = parser.parse_args(argv)
    try:
        if args.command == "verify":
            report = verify(args.root,
                            progress=lambda done, total, name:
                            print(f"\r  {done}/{total}  {name[:60]:<60}", end="", flush=True))
            print()
        elif args.command == "fetch":
            report = fetch(args.url, into=args.into, expected_sha256=args.sha256)
        else:
            report = publish(args.root, out=args.out)
    except BundleError as exc:
        print(f"lexibeat-bundle: {exc}")
        return 1
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
