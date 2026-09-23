"""The sample bundle: verify it, fetch it, publish it.

The engine works without it and offers only the sample-free ``electronic`` palette, reporting
``production_bundle: false``. With it, every family is available. It is 3.1 GB of content-addressed
audio, so it does not ride a release tarball of source and it does not enter a host's repository:
the host fetches it once into a volume and points ``LEXIBEAT_BUNDLE_ROOT`` at the mount.

Licensing is not why any of this is arranged this way, and an earlier draft said it was. The one
attribution-bearing source is CC-BY 3.0, which permits redistribution; the credit already travels
inside ``licenses/`` and ``NOTICE.md``. Size is the whole reason.

    lexibeat-bundle verify [--root DIR]
    lexibeat-bundle fetch --into DIR --from URL [--from URL …] [--sha256 DIGEST]
    lexibeat-bundle publish --out DIR
    lexibeat-bundle status [--root DIR]
"""

from __future__ import annotations

import hashlib
import json
import shutil
import tarfile
import tempfile
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

from .paths import REPOSITORY_BUNDLE_ROOT, bundle_present, configured_bundle_root

MANIFEST_NAME = "manifest.json"
CATALOG_NAME = "catalog.sqlite3"
CHECKSUMS_NAME = "SHA256SUMS"
BLOCK = 1024 * 1024
# GitHub refuses a release asset of 2 GiB or more, and the bundle is larger than that, so it is
# published as numbered parts under this size and joined again by `fetch`. The digest a host pins is
# the *whole* archive's, so how it was cut is not part of what is trusted.
PART_BYTES = 1900 * 1024 * 1024


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


_STATUS_CACHE: dict[tuple[str, float], dict[str, Any]] = {}


def status(root: Path | None = None) -> dict[str, Any]:
    """Which bundle is mounted, and whether every file its manifest names is on disk.

    A readable catalogue is not enough to answer either question. The engine rejects a candidate
    whose sample is missing rather than failing the render, so a half-installed bundle quietly
    favours the beds whose samples happen to exist; and a host that asks only "is there a bundle?"
    cannot tell the listener-approved library from the smaller one it replaced — which is exactly
    how a deployment ran the old one for weeks while sounding thinner than the music it was tuned
    against. This stats every file rather than hashing it, so it is cheap enough for `/schema`;
    `verify` is the digest check.
    """
    root = Path(root) if root else configured_bundle_root()
    manifest_path = root / MANIFEST_NAME
    try:
        key = (str(root), manifest_path.stat().st_mtime)
    except OSError:
        return {"present": False, "complete": False, "bundle": None, "version": None,
                "assets": 0, "missing": 0, "expanded": False}
    if key in _STATUS_CACHE:
        return _STATUS_CACHE[key]
    manifest = read_manifest(root)
    entries = manifest_entries(manifest)
    missing = sum(1 for entry in entries if not (root / str(entry["bundle_path"])).is_file())
    present = bundle_present(root)
    report = {
        "present": present,
        "complete": present and missing == 0,
        "bundle": manifest.get("bundle"),
        "version": str(manifest.get("version")),
        "assets": len(entries),
        "missing": missing,
        # The manifest's expansion policy is what switches on the Wave 2/3 instruments, their role
        # treatments and their gains; a bundle without one renders the control behaviour.
        "expanded": bool(manifest.get("expansion_policy")),
    }
    # Only a complete answer is remembered: an install still unpacking is asked again next time.
    if report["complete"]:
        _STATUS_CACHE[key] = report
    return report


def checksums(paths: Iterable[Path], base: Path) -> str:
    """A `SHA256SUMS` in the format `shasum -c` reads."""
    return "".join(f"{sha256(path)}  {path.relative_to(base).as_posix()}\n"
                   for path in sorted(paths))


def publish(root: Path | None = None, *, out: Path,
            part_bytes: int = PART_BYTES) -> dict[str, Any]:
    """Pack a verified bundle into numbered archive parts, ready to attach to a release.

    The parts are `NAME.tar.001`, `.002`, …, each under `part_bytes`; concatenated in order they are
    one reproducible tar, and `sha256` is that tar's digest — the one a host pins and `fetch`
    checks. `SHA256SUMS` lists the parts, so each upload can be checked on its own as well.
    """
    root = Path(root) if root else REPOSITORY_BUNDLE_ROOT
    report = verify(root)
    manifest = read_manifest(root)
    name = f"{manifest.get('bundle', 'lexibeat-bundle')}-v{manifest.get('version', 1)}"
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    for stale in out.glob(f"{name}.tar.*"):
        stale.unlink()
    with tempfile.TemporaryDirectory(prefix="lexibeat-publish-", dir=out) as workspace:
        whole = Path(workspace) / f"{name}.tar"
        with tarfile.open(whole, "w") as tar:
            tar.add(root, arcname=name, recursive=True, filter=_reproducible)
        digest = sha256(whole)
        size = whole.stat().st_size
        parts: list[Path] = []
        with whole.open("rb") as source:
            while True:
                part = out / f"{name}.tar.{len(parts) + 1:03d}"
                written = 0
                with part.open("wb") as handle:
                    while written < part_bytes and (chunk := source.read(
                            min(BLOCK, part_bytes - written))):
                        handle.write(chunk)
                        written += len(chunk)
                if not written:
                    part.unlink()
                    break
                parts.append(part)
    (out / CHECKSUMS_NAME).write_text(checksums(parts, out), encoding="utf-8")
    return {**report, "root_name": name, "parts": [part.name for part in parts],
            "bytes": size, "sha256": digest}


def _reproducible(info: tarfile.TarInfo) -> tarfile.TarInfo:
    """Same bytes from the same bundle on any machine, so a published digest means something."""
    info.uid = info.gid = 0
    info.uname = info.gname = ""
    info.mtime = 0
    info.mode = 0o755 if info.isdir() else 0o644
    return info


def fetch(urls: str | Sequence[str], *, into: Path, expected_sha256: str = "",
          progress: Callable[[int, int], None] | None = None) -> dict[str, Any]:
    """Download and verify a published bundle archive, then extract it into ``into``.

    ``urls`` are the archive's parts in order (see `publish`); they are joined into one tar and the
    digest checked is the whole one's.

    Nothing is extracted before the digest matches. A bundle half-written by an interrupted
    download is the failure mode worth designing against, because the engine would then render with
    some samples missing rather than refusing.

    ``into`` is a volume that holds *one* bundle. The root verified is the one this archive names,
    and once it verifies, any other bundle root beside it is removed: an upgrade then frees the
    gigabytes of the one it supersedes, and leaves nothing a host could mount by mistake. Nothing is
    removed before the new bundle has verified, so a failed upgrade keeps the old one.
    """
    import urllib.request

    urls = [urls] if isinstance(urls, str) else list(urls)
    into = Path(into)
    into.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="lexibeat-bundle-") as workspace:
        archive = Path(workspace) / "bundle.tar"
        read = 0
        with archive.open("wb") as handle:
            for url in urls:
                with urllib.request.urlopen(url) as response:
                    total = int(response.headers.get("content-length") or 0)
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
            names = {Path(member.name).parts[0] for member in tar.getmembers()
                     if Path(member.name).parts}
            if len(names) != 1:
                raise BundleError(f"The archive holds {len(names)} top-level entries, not one "
                                  "bundle root.")
            tar.extractall(into, filter="data")
    root = into / names.pop()
    report = verify(root)
    replaced = []
    for other in sorted(into.iterdir()):
        if other != root and other.is_dir() and (other / MANIFEST_NAME).is_file():
            shutil.rmtree(other)
            replaced.append(other.name)
    return {**report, "sha256": digest, "urls": urls, "replaced": replaced}


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(prog="lexibeat-bundle", description=__doc__.split("\n\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)

    check = commands.add_parser("verify", help="check a bundle against its own manifest")
    check.add_argument("--root", type=Path, default=None)

    get = commands.add_parser("fetch", help="download, verify and extract a published bundle")
    get.add_argument("--into", type=Path, required=True)
    get.add_argument("--from", dest="urls", action="append", required=True,
                     help="an archive part's URL; repeat for every part, in order")
    get.add_argument("--sha256", default="")

    state = commands.add_parser("status", help="which bundle is mounted, and is every file present")
    state.add_argument("--root", type=Path, default=None)

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
        elif args.command == "status":
            report = status(args.root)
        elif args.command == "fetch":
            report = fetch(args.urls, into=args.into, expected_sha256=args.sha256)
        else:
            report = publish(args.root, out=args.out)
    except BundleError as exc:
        print(f"lexibeat-bundle: {exc}")
        return 1
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
