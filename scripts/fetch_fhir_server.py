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
import urllib.parse
import urllib.request
from pathlib import Path

DEFAULT_REPO = "jyxsu6/medagentbench"
DEFAULT_TAG = "latest"
DEFAULT_REGISTRY = "https://registry-1.docker.io"

# Tried in order for each blob. Registries fail independently and intermittently -- a
# mirror may 403 manifests but serve blobs, rate-limit after a partial pull, or simply be
# unreachable from a given network -- so falling through beats failing the whole run.
FALLBACK_REGISTRIES = [
    "https://registry-1.docker.io",
    "https://docker.m.daocloud.io",
    "https://docker.1ms.run",
    "https://hub.rat.dev",
]

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


def _parse_challenge(header: str) -> dict:
    """Parse a WWW-Authenticate Bearer challenge into its parameters.

    ``Bearer realm="https://host/auth/token",service="host"`` -> {"realm": ..., "service": ...}
    """
    if not header.lower().startswith("bearer"):
        return {}
    params = {}
    for part in header[len("bearer"):].strip().split(","):
        key, _, value = part.partition("=")
        if value:
            params[key.strip()] = value.strip().strip('"')
    return params


def _token_for(challenge: dict, repo: str, timeout: float) -> str:
    query = {"scope": f"repository:{repo}:pull"}
    if challenge.get("service"):
        query["service"] = challenge["service"]
    body = json.loads(
        _get(f"{challenge['realm']}?{urllib.parse.urlencode(query)}", {}, timeout)
    )
    token = body.get("token") or body.get("access_token")
    if not token:
        raise RuntimeError(f"no token in the response from {challenge['realm']}")
    return token


def open_authed(url: str, repo: str, timeout: float, accept: str | None = None):
    """Open a registry URL, following a 401 challenge if one comes back.

    Registries put their token endpoint in different places -- Docker Hub uses
    auth.docker.io, daocloud uses /auth/token on its own host -- and hardcoding either is
    exactly why the first version of this failed against a mirror. The spec already says
    where to look: the ``WWW-Authenticate`` header on the 401. Follow it and any mirror
    works without special-casing.
    """
    headers = {"Accept": accept} if accept else {"Accept": "*/*"}
    try:
        return urllib.request.urlopen(
            urllib.request.Request(url, headers=headers), timeout=timeout
        )
    except urllib.error.HTTPError as exc:
        if exc.code != 401:
            raise
        challenge = _parse_challenge(exc.headers.get("WWW-Authenticate", ""))
        if not challenge.get("realm"):
            raise
    headers["Authorization"] = f"Bearer {_token_for(challenge, repo, timeout)}"
    return urllib.request.urlopen(
        urllib.request.Request(url, headers=headers), timeout=timeout
    )


def fetch_manifest(registry: str, repo: str, tag: str, timeout: float) -> dict:
    with open_authed(f"{registry}/v2/{repo}/manifests/{tag}", repo, timeout, ACCEPT) as r:
        manifest = json.loads(r.read())

    # Multi-arch index: pick linux/amd64.
    if "manifests" in manifest:
        for entry in manifest["manifests"]:
            platform = entry.get("platform", {})
            if platform.get("os") == "linux" and platform.get("architecture") == "amd64":
                return fetch_manifest(registry, repo, entry["digest"], timeout)
        raise RuntimeError("no linux/amd64 manifest in the image index")
    return manifest


def download_blob(
    registries: list[str] | str, repo: str, digest: str, dest: Path, timeout: float
) -> Path:
    """Download one blob, trying each registry in turn. Cached and digest-verified.

    Registries fail independently and intermittently: a mirror may 403 manifests but
    serve blobs, rate-limit after a partial pull, or be unreachable from one network and
    fine from another. Falling through beats failing a 1.8 GB run on one bad host.
    """
    if isinstance(registries, str):
        registries = [registries]
    if dest.exists() and _sha256(dest) == digest.split(":", 1)[1]:
        return dest

    last = None
    for index, registry in enumerate(registries):
        try:
            return _download_blob_from(registry, repo, digest, dest, timeout)
        except Exception as exc:  # any transport or HTTP failure: try the next mirror
            last = exc
            remaining = len(registries) - index - 1
            print(f"    {registry} failed ({type(exc).__name__}); {remaining} mirror(s) left")
    raise RuntimeError(f"all registries failed for {digest}: {last}")


def _download_blob_from(
    registry: str, repo: str, digest: str, dest: Path, timeout: float
) -> Path:
    tmp = dest.with_suffix(".part")
    digester = hashlib.sha256()
    opened = open_authed(f"{registry}/v2/{repo}/blobs/{digest}", repo, timeout)
    with opened as response, tmp.open("wb") as handle:
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
    parser.add_argument("--registry", action="append", default=None,
                        help="registry base URL; repeat to set a fallback order. "
                             "Defaults to Docker Hub plus several mirrors, each tried "
                             "in turn per blob.")
    parser.add_argument("--repo", default=DEFAULT_REPO)
    parser.add_argument("--tag", default=DEFAULT_TAG)
    parser.add_argument("--manifest", type=Path, default=None,
                        help="use this manifest instead of resolving one from a registry; "
                             "data/medagentbench_image_manifest.json pins the exact image")
    parser.add_argument("--offline", action="store_true",
                        help="extract from already-downloaded blobs; never touch a registry")
    parser.add_argument("--timeout", type=float, default=45.0,
                        help="per-request timeout; a dead mirror is abandoned this fast")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--heap", default="2g", help="JVM max heap, e.g. 2g")
    parser.add_argument("--keep-blobs", action="store_true",
                        help="keep the layer cache (~1.8 GB) after extracting")
    args = parser.parse_args(argv)

    out = args.out.expanduser().resolve()
    blobs = out / "blobs"
    rootfs = out / "rootfs"
    blobs.mkdir(parents=True, exist_ok=True)

    registries = [r.rstrip("/") for r in (args.registry or FALLBACK_REGISTRIES)]
    registry = registries[0]
    print("registries:\n" + "\n".join(f"  {r}" for r in registries))
    print(f"repo:     {args.repo}:{args.tag}\nout:      {out}\n")

    cached_manifest = blobs / "manifest.json"
    manifest_source = args.manifest or (cached_manifest if args.offline else None)

    if manifest_source and manifest_source.exists():
        print(f"using manifest {manifest_source}")
        manifest = json.loads(manifest_source.read_text(encoding="utf-8"))
    elif args.offline:
        print(f"\n--offline needs a manifest: none at {cached_manifest}. Pass --manifest "
              "data/medagentbench_image_manifest.json\n", file=sys.stderr)
        return 2
    else:
        print("resolving manifest...")
        manifest, exc = None, None
        for candidate in registries:
            try:
                manifest = fetch_manifest(candidate, args.repo, args.tag, args.timeout)
                print(f"  resolved via {candidate}")
                break
            except Exception as err:
                exc = err
                print(f"  {candidate} failed ({type(err).__name__})")
        if manifest is None:
            print(f"\nFAILED to fetch the manifest: {exc}\n", file=sys.stderr)
            print("Options:", file=sys.stderr)
            print("  --manifest data/medagentbench_image_manifest.json   # pinned, no lookup",
                  file=sys.stderr)
            print("  --offline                     # extract from blobs already downloaded",
                  file=sys.stderr)
            print("  --registry https://docker.m.daocloud.io", file=sys.stderr)
            print("  turn AutoDL network_turbo OFF for mirrors, ON for HuggingFace",
                  file=sys.stderr)
            return 2
        cached_manifest.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    layers = manifest["layers"]
    total = sum(layer["size"] for layer in layers)
    print(f"  {len(layers)} layers, {total/1e9:.2f} GB compressed\n")

    if args.offline:
        missing = [l["digest"] for l in layers
                   if not (blobs / l["digest"].replace(":", "_")).exists()]
        if missing or not (blobs / "config.json").exists():
            print(f"\n--offline but {len(missing)} layer(s) are not cached in {blobs}.\n"
                  "Re-run online once to fetch them; cached layers are digest-verified and "
                  "skipped.\n", file=sys.stderr)
            return 2
        print("offline: using cached blobs")
        config = json.loads((blobs / "config.json").read_text(encoding="utf-8"))
        paths = [blobs / l["digest"].replace(":", "_") for l in layers]
    else:
        print("downloading config...")
        config = json.loads(
            download_blob(registries, args.repo, manifest["config"]["digest"],
                          blobs / "config.json", args.timeout).read_text(encoding="utf-8")
        )

        print("downloading layers...")
        paths = []
        for layer in layers:
            paths.append(download_blob(registries, args.repo, layer["digest"],
                                       blobs / layer["digest"].replace(":", "_"),
                                       args.timeout))

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

    if not args.keep_blobs and not args.offline:
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
