#!/usr/bin/env python3
"""Fetch and unpack the MedAgentBench FHIR server image without a Docker daemon.

Needed because container runtimes are unavailable in some environments (an AutoDL
container drops ``cap_sys_admin``, so ``dockerd`` can never start) and because Docker Hub
is unreachable from some networks. Neither is a real obstacle: the image is a Spring Boot
HAPI FHIR war plus a preloaded H2 database, so it runs on a bare JVM.

    # default registry, straight from Docker Hub
    python scripts/fetch_fhir_server.py --out ~/fhir

    # via a mirror, when Hub is blocked
    python scripts/fetch_fhir_server.py --out ~/fhir \\
        --registry https://docker.m.daocloud.io

    # then, with openjdk-17-jre-headless installed
    ~/fhir/run.sh

Layers are cached by digest and verified against it, so an interrupted download resumes
rather than restarting -- the image is ~1.8 GB compressed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tarfile
import urllib.error
import urllib.request
from pathlib import Path

DEFAULT_REPO = "jyxsu6/medagentbench"
DEFAULT_TAG = "latest"
DEFAULT_REGISTRY = "https://registry-1.docker.io"
DEFAULT_AUTH = "https://auth.docker.io/token"
DEFAULT_SERVICE = "registry.docker.io"

ACCEPT = ",".join([
    "application/vnd.docker.distribution.manifest.v2+json",
    "application/vnd.docker.distribution.manifest.list.v2+json",
    "application/vnd.oci.image.manifest.v1+json",
    "application/vnd.oci.image.index.v1+json",
])


def _get(url: str, headers: dict, timeout: float = 60.0) -> bytes:
    request = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read()


def get_token(registry: str, repo: str, timeout: float) -> str | None:
    """Fetch a pull token. Mirrors often need none; failure here is not fatal."""
    if "docker.io" in registry:
        url = f"{DEFAULT_AUTH}?service={DEFAULT_SERVICE}&scope=repository:{repo}:pull"
    else:
        host = registry.split("//", 1)[-1]
        url = f"{registry}/token?service={host}&scope=repository:{repo}:pull"
    try:
        return json.loads(_get(url, {}, timeout)).get("token")
    except (urllib.error.URLError, ValueError, TimeoutError) as exc:
        print(f"  note: no token from {url.split('?')[0]} ({exc}); trying anonymously")
        return None


def fetch_manifest(registry: str, repo: str, tag: str, token: str | None, timeout: float) -> dict:
    headers = {"Accept": ACCEPT}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    raw = _get(f"{registry}/v2/{repo}/manifests/{tag}", headers, timeout)
    manifest = json.loads(raw)

    # Multi-arch index: pick linux/amd64.
    if "manifests" in manifest:
        for entry in manifest["manifests"]:
            platform = entry.get("platform", {})
            if platform.get("os") == "linux" and platform.get("architecture") == "amd64":
                return fetch_manifest(registry, repo, entry["digest"], token, timeout)
        raise RuntimeError("no linux/amd64 manifest in the image index")
    return manifest


def download_blob(
    registry: str, repo: str, digest: str, dest: Path, token: str | None, timeout: float
) -> Path:
    """Download one blob to ``dest``, verifying its digest. Cached and resumable."""
    if dest.exists() and _sha256(dest) == digest.split(":", 1)[1]:
        return dest

    headers = {"Accept": "*/*"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    request = urllib.request.Request(f"{registry}/v2/{repo}/blobs/{digest}", headers=headers)

    tmp = dest.with_suffix(".part")
    digester = hashlib.sha256()
    with urllib.request.urlopen(request, timeout=timeout) as response, tmp.open("wb") as handle:
        total = int(response.headers.get("Content-Length") or 0)
        done = 0
        while chunk := response.read(1 << 20):
            handle.write(chunk)
            digester.update(chunk)
            done += len(chunk)
            if total:
                pct = done * 100 // total
                print(f"\r    {digest[7:19]}  {done/1e6:7.1f}/{total/1e6:.1f} MB  {pct:3d}%",
                      end="", flush=True)
    print()

    actual = digester.hexdigest()
    expected = digest.split(":", 1)[1]
    if actual != expected:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"digest mismatch for {digest}: got sha256:{actual}")
    tmp.rename(dest)
    return dest


def _sha256(path: Path) -> str:
    digester = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1 << 20):
            digester.update(chunk)
    return digester.hexdigest()


def extract_layers(layers: list[Path], rootfs: Path) -> None:
    """Unpack layers in order, honouring whiteout markers.

    Layers are applied lowest-first; a ``.wh.<name>`` entry deletes ``<name>`` from the
    accumulated filesystem, and ``.wh..wh..opq`` clears a directory's contents.
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
            _safe_extract(tar, members, rootfs)


def _remove(path: Path) -> None:
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path, ignore_errors=True)
    else:
        path.unlink(missing_ok=True)


def _safe_extract(tar: tarfile.TarFile, members: list, rootfs: Path) -> None:
    """Extract, refusing paths that escape the destination."""
    root = rootfs.resolve()
    for member in members:
        target = (rootfs / member.name).resolve()
        if not str(target).startswith(str(root)):
            raise RuntimeError(f"refusing path traversal in layer: {member.name}")
        try:
            tar.extract(member, rootfs, filter="tar")
        except TypeError:  # Python < 3.12 has no filter=
            tar.extract(member, rootfs)
        except (OSError, tarfile.TarError):
            continue  # device nodes and the like; irrelevant to a JVM app


def write_launcher(rootfs: Path, out: Path, config: dict, port: int, heap: str) -> Path:
    """Emit run.sh, translating the image ENTRYPOINT to run against the extracted tree.

    The image sets SPRING_CONFIG_LOCATION to an absolute /configs path and the H2 database
    lives at an absolute /data path, so both are rewritten to point inside rootfs.
    """
    entrypoint = (config.get("config") or {}).get("Entrypoint") or []
    script = out / "run.sh"
    script.write_text(
        f"""#!/usr/bin/env bash
# Generated by scripts/fetch_fhir_server.py -- runs the MedAgentBench FHIR server
# directly on a JVM, with no container runtime.
set -euo pipefail
ROOTFS="{rootfs}"
cd "$ROOTFS/app"

export SPRING_CONFIG_LOCATION="file://$ROOTFS/configs/application.yaml"
export SERVER_PORT="{port}"

# The packaged application.yaml points the H2 database at an absolute /data path.
# Rewrite it to the extracted copy so no root-owned /data is needed.
if [ ! -f "$ROOTFS/configs/application.local.yaml" ]; then
  sed "s#/data#$ROOTFS/data#g" "$ROOTFS/configs/application.yaml" \\
      > "$ROOTFS/configs/application.local.yaml"
fi
export SPRING_CONFIG_LOCATION="file://$ROOTFS/configs/application.local.yaml"

exec java -Xmx{heap} \\
  --class-path "$ROOTFS/app/main.war" \\
  '-Dloader.path=main.war!/WEB-INF/classes/,main.war!/WEB-INF/,{rootfs}/app/extra-classes' \\
  org.springframework.boot.loader.PropertiesLauncher
""",
        encoding="utf-8",
    )
    script.chmod(0o755)
    if entrypoint:
        (out / "original_entrypoint.json").write_text(
            json.dumps(entrypoint, indent=2), encoding="utf-8"
        )
    return script


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Fetch the MedAgentBench FHIR server image")
    parser.add_argument("--out", type=Path, required=True, help="destination directory")
    parser.add_argument("--registry", default=DEFAULT_REGISTRY,
                        help="registry base URL; use a mirror if Docker Hub is blocked")
    parser.add_argument("--repo", default=DEFAULT_REPO)
    parser.add_argument("--tag", default=DEFAULT_TAG)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--heap", default="2g", help="JVM max heap, e.g. 2g")
    parser.add_argument("--keep-blobs", action="store_true",
                        help="keep the layer cache (~1.8 GB) after extracting")
    args = parser.parse_args(argv)

    out = args.out.expanduser().resolve()
    blobs = out / "blobs"
    rootfs = out / "rootfs"
    blobs.mkdir(parents=True, exist_ok=True)

    registry = args.registry.rstrip("/")
    print(f"registry: {registry}\nrepo:     {args.repo}:{args.tag}\nout:      {out}\n")

    print("resolving manifest...")
    token = get_token(registry, args.repo, args.timeout)
    try:
        manifest = fetch_manifest(registry, args.repo, args.tag, token, args.timeout)
    except (urllib.error.URLError, TimeoutError, RuntimeError, ValueError) as exc:
        print(f"\nFAILED to fetch the manifest: {exc}\n", file=sys.stderr)
        print("Most likely the registry is unreachable from this network. Try:", file=sys.stderr)
        print("  source /etc/network_turbo            # on AutoDL", file=sys.stderr)
        print("  --registry https://docker.m.daocloud.io", file=sys.stderr)
        return 2

    layers = manifest["layers"]
    total = sum(layer["size"] for layer in layers)
    print(f"  {len(layers)} layers, {total/1e9:.2f} GB compressed\n")

    print("downloading config...")
    config = json.loads(
        download_blob(registry, args.repo, manifest["config"]["digest"],
                      blobs / "config.json", token, args.timeout).read_text(encoding="utf-8")
    )

    print("downloading layers...")
    paths = []
    for layer in layers:
        paths.append(download_blob(registry, args.repo, layer["digest"],
                                   blobs / layer["digest"].replace(":", "_"),
                                   token, args.timeout))

    print("\nextracting...")
    extract_layers(paths, rootfs)

    war = rootfs / "app" / "main.war"
    data = rootfs / "data"
    print(f"\n  {'OK ' if war.exists() else 'MISSING'} {war}")
    print(f"  {'OK ' if data.is_dir() else 'MISSING'} {data}")
    if not war.exists():
        print("\nmain.war not found -- the image layout changed; inspect rootfs/", file=sys.stderr)
        return 3

    script = write_launcher(rootfs, out, config, args.port, args.heap)

    if not args.keep_blobs:
        shutil.rmtree(blobs, ignore_errors=True)
        print(f"  removed layer cache (pass --keep-blobs to retain)")

    java = shutil.which("java")
    print(f"\nlauncher: {script}")
    print(f"java:     {java or 'NOT FOUND -- apt-get install -y openjdk-17-jre-headless'}")
    print(f"\nrun it:   {script}")
    print(f"verify:   curl -s 'http://localhost:{args.port}/fhir/Patient?_count=1&_format=json'")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
