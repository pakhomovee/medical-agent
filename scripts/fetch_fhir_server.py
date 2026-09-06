#!/usr/bin/env python3
"""Fetch and unpack the MedAgentBench FHIR server, with no Docker daemon.

Two things make the obvious route unavailable in practice:

* Container runtimes need privileges some GPU hosts do not grant. An AutoDL container
  drops ``cap_sys_admin``, so ``dockerd`` can never start, whatever the user id.
* Docker Hub is unreachable from some networks, and the public mirrors that proxy it are
  region-dependent, rate-limited and individually unreliable.

Neither actually matters, because the image is a Spring Boot HAPI FHIR war plus a
preloaded H2 database. Pull the layers over plain HTTPS, unpack them, run it on a JVM.

    python scripts/fetch_fhir_server.py --check          # which registry works from here?
    python scripts/fetch_fhir_server.py --out ~/fhir     # pull, unpack, write run.sh
    ~/fhir/run.sh

Design notes, each of which is a bug this script previously had:

* **Auth follows the WWW-Authenticate challenge.** Registries put their token endpoint in
  different places -- Docker Hub at auth.docker.io, daocloud at /auth/token on its own
  host. Guessing breaks on whichever one you did not test against.
* **The manifest is always resolved from the registry the blobs come from.** Mirrors are
  pull-through caches: they populate blobs only after a manifest request. Supplying a
  pinned manifest to skip that lookup leaves the cache cold and every blob 404s.
  A pinned manifest is therefore only meaningful for ``--offline``.
* **Transfers are abandoned on stall.** urllib's timeout is per socket read, so a mirror
  trickling bytes never trips it and the pull hangs forever.
* **Blobs fall through to the next registry.** Digest verification makes mixing sources
  safe, and a 1.8 GB pull should not die because one host is refusing today.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import tarfile
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

REPO = "jyxsu6/medagentbench"
TAG = "latest"

# Docker Hub only by default. Third-party pull-through mirrors are reachable with
# --registry when Hub is blocked, but they are not in the default trust path: they are
# operated by unknown parties and their availability varies by region. Whatever you use,
# the manifest is checked against the pinned copy before anything downloads, so a mirror
# cannot substitute content -- see verify_against_pin.
REGISTRIES = ["https://registry-1.docker.io"]

# Known public mirrors, for --check to probe when Docker Hub is unreachable. Listed as a
# convenience, not an endorsement.
KNOWN_MIRRORS = [
    "https://docker.m.daocloud.io",
    "https://docker.1ms.run",
    "https://docker.xuanyuan.me",
]

ACCEPT = ",".join([
    "application/vnd.docker.distribution.manifest.v2+json",
    "application/vnd.docker.distribution.manifest.list.v2+json",
    "application/vnd.oci.image.manifest.v1+json",
    "application/vnd.oci.image.index.v1+json",
])

MANIFEST_CACHE = "manifest.json"

# Trust anchor. Digest verification only proves a blob matches the manifest that named
# it -- and that manifest comes from the same registry as the blobs, so a hostile mirror
# could serve a poisoned manifest plus matching poisoned layers and every check would
# pass. The pinned copy was captured from Docker Hub and is compared against whatever a
# mirror returns, which is what actually makes pulling through a mirror safe. Refresh it
# only from registry-1.docker.io:
#     python scripts/fetch_fhir_server.py --pin-manifest
PINNED_MANIFEST = Path(__file__).resolve().parents[1] / "data" / "medagentbench_image_manifest.json"


# --- registry access ------------------------------------------------------------------

def _read(url: str, headers: dict, timeout: float) -> bytes:
    with urllib.request.urlopen(
        urllib.request.Request(url, headers=headers), timeout=timeout
    ) as response:
        return response.read()


def parse_challenge(header: str) -> dict:
    """``Bearer realm="https://h/auth/token",service="h"`` -> {"realm":..., "service":...}"""
    if not header.lower().startswith("bearer"):
        return {}
    out = {}
    for part in header[len("bearer"):].strip().split(","):
        key, _, value = part.partition("=")
        if value:
            out[key.strip()] = value.strip().strip('"')
    return out


def open_authed(url: str, repo: str, timeout: float, accept: str | None = None):
    """Open a registry URL, following a 401 challenge to wherever it points."""
    headers = {"Accept": accept or "*/*"}
    try:
        return urllib.request.urlopen(
            urllib.request.Request(url, headers=headers), timeout=timeout
        )
    except urllib.error.HTTPError as exc:
        if exc.code != 401:
            raise
        challenge = parse_challenge(exc.headers.get("WWW-Authenticate", ""))
        if not challenge.get("realm"):
            raise

    query = {"scope": f"repository:{repo}:pull"}
    if challenge.get("service"):
        query["service"] = challenge["service"]
    body = json.loads(
        _read(f"{challenge['realm']}?{urllib.parse.urlencode(query)}", {}, timeout)
    )
    token = body.get("token") or body.get("access_token")
    if not token:
        raise RuntimeError(f"no token from {challenge['realm']}")
    headers["Authorization"] = f"Bearer {token}"
    return urllib.request.urlopen(
        urllib.request.Request(url, headers=headers), timeout=timeout
    )


def manifest_identity(manifest: dict) -> tuple[str, frozenset]:
    """The parts that determine what content will be executed."""
    return manifest["config"]["digest"], frozenset(l["digest"] for l in manifest["layers"])


def verify_against_pin(manifest: dict, pin_path: Path = PINNED_MANIFEST) -> None:
    """Refuse a manifest whose content digests differ from the pinned copy.

    Without this, pulling through a mirror trusts that mirror completely.
    """
    if not pin_path.exists():
        print(f"    WARNING: no pinned manifest at {pin_path}; cannot verify the mirror")
        return
    pinned = json.loads(pin_path.read_text(encoding="utf-8"))
    got, expected = manifest_identity(manifest), manifest_identity(pinned)
    if got == expected:
        print(f"    verified against pin ({pin_path.name})")
        return
    config_differs = got[0] != expected[0]
    raise RuntimeError(
        "MANIFEST DOES NOT MATCH THE PINNED COPY -- refusing to download.\n"
        f"      config digest {'differs' if config_differs else 'matches'}; "
        f"{len(got[1] ^ expected[1])} layer digest(s) differ.\n"
        "      This registry is serving different content from Docker Hub. Do not use it.\n"
        "      If the upstream image genuinely changed, re-pin from Docker Hub with "
        "--pin-manifest and review the diff."
    )


def fetch_manifest(registry: str, repo: str, tag: str, timeout: float) -> dict:
    """Resolve the manifest. This also warms a pull-through mirror's blob cache."""
    with open_authed(f"{registry}/v2/{repo}/manifests/{tag}", repo, timeout, ACCEPT) as r:
        manifest = json.loads(r.read())
    if "manifests" in manifest:  # multi-arch index
        for entry in manifest["manifests"]:
            platform = entry.get("platform", {})
            if platform.get("os") == "linux" and platform.get("architecture") == "amd64":
                return fetch_manifest(registry, repo, entry["digest"], timeout)
        raise RuntimeError("no linux/amd64 manifest in the image index")
    return manifest


def fetch_blob(
    registry: str, repo: str, digest: str, dest: Path, timeout: float,
    min_bytes_per_s: float = 20_000.0,
) -> Path:
    """Fetch one blob from one registry, verifying its digest and abandoning stalls."""
    tmp = dest.with_suffix(".part")
    digester = hashlib.sha256()
    started = time.monotonic()

    with open_authed(f"{registry}/v2/{repo}/blobs/{digest}", repo, timeout) as response, \
            tmp.open("wb") as handle:
        total = int(response.headers.get("Content-Length") or 0)
        done = 0
        while chunk := response.read(1 << 20):
            handle.write(chunk)
            digester.update(chunk)
            done += len(chunk)
            elapsed = max(time.monotonic() - started, 1e-6)
            if elapsed > 30 and done / elapsed < min_bytes_per_s:
                tmp.unlink(missing_ok=True)
                raise TimeoutError(f"stalled at {done/1e6:.1f} MB ({done/elapsed/1e3:.1f} kB/s)")
            if total:
                rate = done / elapsed
                eta = (total - done) / rate if rate else 0
                print(f"\r    {digest[7:19]}  {done/1e6:7.1f}/{total/1e6:.1f} MB "
                      f"{done*100//total:3d}%  {rate/1e6:4.1f} MB/s  eta {eta/60:3.0f}m",
                      end="", flush=True)
    print()

    if digester.hexdigest() != digest.split(":", 1)[1]:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"digest mismatch for {digest}")
    tmp.rename(dest)
    return dest


def cached_or_fetch(
    registries: list[str], repo: str, digest: str, dest: Path, timeout: float
) -> Path:
    """Return a cached blob, else fetch from the first registry that serves it."""
    if dest.exists() and _sha256(dest) == digest.split(":", 1)[1]:
        return dest
    last = None
    for index, registry in enumerate(registries):
        try:
            return fetch_blob(registry, repo, digest, dest, timeout)
        except Exception as exc:
            last = exc
            print(f"    {registry} failed ({type(exc).__name__}); "
                  f"{len(registries) - index - 1} left")
    raise RuntimeError(f"all registries failed for {digest}: {last}")


def _sha256(path: Path) -> str:
    digester = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1 << 20):
            digester.update(chunk)
    return digester.hexdigest()


# --- preflight ------------------------------------------------------------------------

def check(registries: list[str], repo: str, tag: str, timeout: float) -> list[str]:
    """Report which registries serve both a manifest and a blob from here.

    Exists so that diagnosing a blocked network is one command rather than a session of
    ad-hoc curl. Manifest-only success is not enough: mirrors commonly resolve a manifest
    and then 404 the blobs.
    """
    print(f"checking {len(registries)} registries for {repo}:{tag}\n")
    working = []
    for registry in registries:
        print(f"  {registry}")
        try:
            manifest = fetch_manifest(registry, repo, tag, timeout)
            print(f"    manifest OK ({len(manifest['layers'])} layers)")
        except Exception as exc:
            print(f"    manifest FAILED ({type(exc).__name__}: {exc})")
            continue
        try:
            verify_against_pin(manifest)
        except RuntimeError as exc:
            print(f"    UNSAFE: {exc}".splitlines()[0])
            continue
        smallest = min(manifest["layers"], key=lambda layer: layer["size"])
        probe = Path(f"/tmp/.uqma_probe_{smallest['digest'][7:19]}")
        try:
            fetch_blob(registry, repo, smallest["digest"], probe, timeout)
            print("    blob OK")
            working.append(registry)
        except Exception as exc:
            print(f"    blob FAILED ({type(exc).__name__}: {exc})")
        finally:
            probe.unlink(missing_ok=True)

    print()
    if working:
        print("USE:  --registry " + " --registry ".join(working))
    else:
        print("No registry works from this network. Options:\n"
              "  - turn AutoDL network_turbo OFF (it proxies, and mirrors reject it)\n"
              "  - pull on another machine and copy the extracted tree over\n"
              "  - see RUNBOOK.md, section 'If no registry is reachable'")
    return working


# --- extraction -----------------------------------------------------------------------

def extract_layers(layers: list[Path], rootfs: Path) -> None:
    """Unpack lowest-first, honouring whiteouts.

    ``.wh.<name>`` deletes ``<name>``; ``.wh..wh..opq`` clears a directory's contents.
    """
    rootfs.mkdir(parents=True, exist_ok=True)
    for index, layer in enumerate(layers, 1):
        print(f"  [{index}/{len(layers)}] {layer.name[:19]}")
        with tarfile.open(layer, "r:*") as tar:
            members = []
            for member in tar.getmembers():
                name = Path(member.name).name
                parent = rootfs / Path(member.name).parent
                if name == ".wh..wh..opq":
                    if parent.is_dir():
                        for child in parent.iterdir():
                            _remove(child)
                    continue
                if name.startswith(".wh."):
                    _remove(parent / name[4:])
                    continue
                members.append(member)
            _extract(tar, members, rootfs)


def _remove(path: Path) -> None:
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path, ignore_errors=True)
    else:
        path.unlink(missing_ok=True)


def _extract(tar: tarfile.TarFile, members: list, rootfs: Path) -> None:
    root = rootfs.resolve()
    for member in members:
        if not str((rootfs / member.name).resolve()).startswith(str(root)):
            raise RuntimeError(f"refusing path traversal in layer: {member.name}")
        try:
            tar.extract(member, rootfs, filter="tar")
        except TypeError:  # Python < 3.12 has no filter=
            tar.extract(member, rootfs)
        except (OSError, tarfile.TarError):
            continue  # device nodes and similar; irrelevant to a JVM app


def write_launcher(rootfs: Path, out: Path, port: int, heap: str) -> Path:
    """Emit run.sh from the image entrypoint, repointed at the extracted tree.

    The packaged config puts the H2 database at an absolute ``/data`` path and
    SPRING_CONFIG_LOCATION at ``/configs``; both are rewritten so nothing has to exist at
    the filesystem root.
    """
    script = out / "run.sh"
    script.write_text(f"""#!/usr/bin/env bash
# Generated by scripts/fetch_fhir_server.py -- MedAgentBench FHIR server on a bare JVM.
set -euo pipefail
ROOTFS="{rootfs}"
cd "$ROOTFS/app"

JAVA="${{JAVA:-$(command -v java || echo /usr/lib/jvm/java-17-openjdk-amd64/bin/java)}}"
if [ ! -x "$JAVA" ]; then
  echo "java not found. apt-get install -y openjdk-17-jre-headless" >&2; exit 1
fi

if [ ! -f "$ROOTFS/configs/application.local.yaml" ]; then
  sed "s#/data#$ROOTFS/data#g" "$ROOTFS/configs/application.yaml" \\
      > "$ROOTFS/configs/application.local.yaml"
fi
export SPRING_CONFIG_LOCATION="file://$ROOTFS/configs/application.local.yaml"
export SERVER_PORT="{port}"

exec "$JAVA" -Xmx{heap} \\
  --class-path "$ROOTFS/app/main.war" \\
  '-Dloader.path=main.war!/WEB-INF/classes/,main.war!/WEB-INF/,{rootfs}/app/extra-classes' \\
  org.springframework.boot.loader.PropertiesLauncher
""", encoding="utf-8")
    script.chmod(0o755)
    return script


# --- entry point ----------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Fetch the MedAgentBench FHIR server")
    parser.add_argument("--out", type=Path, help="destination directory")
    parser.add_argument("--registry", action="append", default=None,
                        help="registry base URL; repeat to set fallback order")
    parser.add_argument("--repo", default=REPO)
    parser.add_argument("--tag", default=TAG)
    parser.add_argument("--check", action="store_true",
                        help="report which registries work from here, then exit")
    parser.add_argument("--try-mirrors", action="store_true",
                        help="also probe/use known third-party mirrors. Only needed where "
                             "Docker Hub is blocked; content is still pin-verified")
    parser.add_argument("--offline", action="store_true",
                        help="extract from cached blobs; never open a socket")
    parser.add_argument("--timeout", type=float, default=45.0)
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--heap", default="2g")
    parser.add_argument("--keep-blobs", action="store_true",
                        help="retain the ~1.8 GB layer cache after extracting")
    parser.add_argument("--allow-manifest-drift", action="store_true",
                        help="skip the pinned-manifest check. Only for a deliberate "
                             "upstream image change -- it disables the protection that "
                             "makes pulling through a mirror safe")
    parser.add_argument("--pin-manifest", action="store_true",
                        help="refresh data/medagentbench_image_manifest.json from Docker "
                             "Hub (never from a mirror), then exit")
    args = parser.parse_args(argv)

    registries = [r.rstrip("/") for r in (args.registry or REGISTRIES)]
    if args.try_mirrors and not args.registry:
        registries += KNOWN_MIRRORS

    if args.pin_manifest:
        manifest = fetch_manifest("https://registry-1.docker.io", args.repo, args.tag,
                                  args.timeout)
        PINNED_MANIFEST.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        print(f"pinned {len(manifest['layers'])} layers, config "
              f"{manifest['config']['digest']}\n  -> {PINNED_MANIFEST}")
        return 0

    if args.check:
        return 0 if check(registries, args.repo, args.tag, args.timeout) else 1
    if not args.out:
        parser.error("--out is required unless --check is given")

    out = args.out.expanduser().resolve()
    blobs, rootfs = out / "blobs", out / "rootfs"
    blobs.mkdir(parents=True, exist_ok=True)
    print(f"repo: {args.repo}:{args.tag}\nout:  {out}\n")

    if args.offline:
        cached = blobs / MANIFEST_CACHE
        if not cached.exists():
            print(f"--offline needs a cached manifest at {cached}; run online once first",
                  file=sys.stderr)
            return 2
        manifest = json.loads(cached.read_text(encoding="utf-8"))
        print(f"offline: cached manifest, {len(manifest['layers'])} layers")
        chosen: list[str] = []
    else:
        print("resolving manifest...")
        manifest, chosen = None, []
        for registry in registries:
            try:
                manifest = fetch_manifest(registry, args.repo, args.tag, args.timeout)
                print(f"  via {registry}: {len(manifest['layers'])} layers, "
                      f"{sum(l['size'] for l in manifest['layers'])/1e9:.2f} GB")
                if not args.allow_manifest_drift:
                    verify_against_pin(manifest)
                chosen = [registry] + [r for r in registries if r != registry]
                break
            except Exception as exc:
                print(f"  {registry} failed ({type(exc).__name__})")
        if manifest is None:
            print("\nNo registry served a manifest. Run --check for a diagnosis, and see\n"
                  "RUNBOOK.md. If AutoDL network_turbo is on, turn it OFF for registries.",
                  file=sys.stderr)
            return 2
        (blobs / MANIFEST_CACHE).write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    layers = manifest["layers"]

    if args.offline:
        missing = [l for l in layers if not (blobs / _blob_name(l["digest"])).exists()]
        if missing:
            print(f"--offline but {len(missing)} of {len(layers)} layers are not cached",
                  file=sys.stderr)
            return 2
        paths = [blobs / _blob_name(l["digest"]) for l in layers]
    else:
        print("\ndownloading layers (cached ones are skipped)...")
        paths = [
            cached_or_fetch(chosen, args.repo, layer["digest"],
                            blobs / _blob_name(layer["digest"]), args.timeout)
            for layer in layers
        ]

    print("\nextracting...")
    extract_layers(paths, rootfs)

    war, data = rootfs / "app" / "main.war", rootfs / "data"
    print(f"\n  {'OK ' if war.exists() else 'MISSING'} {war}")
    print(f"  {'OK ' if data.is_dir() else 'MISSING'} {data}")
    if not war.exists():
        print("\nmain.war absent; inspect rootfs/", file=sys.stderr)
        return 3

    script = write_launcher(rootfs, out, args.port, args.heap)
    if not args.keep_blobs and not args.offline:
        shutil.rmtree(blobs, ignore_errors=True)
        print("  layer cache removed (--keep-blobs to retain)")

    print(f"\nrun it:  {script}")
    if not shutil.which("java"):
        print("java:    NOT FOUND -- apt-get install -y openjdk-17-jre-headless")
    print(f"verify:  curl -s 'http://localhost:{args.port}/fhir/Patient?_count=1&_format=json'")
    return 0


def _blob_name(digest: str) -> str:
    return digest.replace(":", "_")


if __name__ == "__main__":
    raise SystemExit(main())
