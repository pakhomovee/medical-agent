"""Registry auth and layer extraction for the no-Docker FHIR server fetch.

The auth logic is the part that broke in the field: the first version hardcoded Docker
Hub's token endpoint, which 403s on a mirror that puts its own at /auth/token. Following
the WWW-Authenticate challenge is what makes any mirror work, so it is pinned here.
"""

from __future__ import annotations

import io
import sys
import tarfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from fetch_fhir_server import _parse_challenge, extract_layers  # noqa: E402


# --- WWW-Authenticate parsing ---------------------------------------------------------

def test_parses_docker_hub_challenge():
    challenge = _parse_challenge(
        'Bearer realm="https://auth.docker.io/token",service="registry.docker.io"'
    )
    assert challenge["realm"] == "https://auth.docker.io/token"
    assert challenge["service"] == "registry.docker.io"


def test_parses_mirror_challenge_with_a_different_realm_path():
    """The daocloud shape -- the case the hardcoded /token guess got wrong."""
    challenge = _parse_challenge(
        'Bearer realm="https://docker.m.daocloud.io/auth/token",service="docker.m.daocloud.io"'
    )
    assert challenge["realm"] == "https://docker.m.daocloud.io/auth/token"
    assert challenge["service"] == "docker.m.daocloud.io"


def test_parses_challenge_carrying_a_scope():
    challenge = _parse_challenge(
        'Bearer realm="https://r/token",service="s",scope="repository:a/b:pull"'
    )
    assert challenge["scope"] == "repository:a/b:pull"


def test_challenge_without_service_is_still_usable():
    assert _parse_challenge('Bearer realm="https://r/token"') == {"realm": "https://r/token"}


def test_non_bearer_challenge_is_ignored():
    assert _parse_challenge('Basic realm="x"') == {}
    assert _parse_challenge("") == {}


def test_challenge_parsing_is_case_insensitive_on_the_scheme():
    assert _parse_challenge('bearer realm="https://r/t"')["realm"] == "https://r/t"


# --- layer extraction -----------------------------------------------------------------

def _layer(tmp_path: Path, name: str, entries: dict[str, bytes | None]) -> Path:
    """Build a gzipped layer. A None value means an empty marker file (whiteout)."""
    path = tmp_path / name
    with tarfile.open(path, "w:gz") as tar:
        for member_name, content in entries.items():
            data = content or b""
            info = tarfile.TarInfo(member_name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    return path


def test_layers_apply_lowest_first(tmp_path):
    lower = _layer(tmp_path, "a.tar.gz", {"app/x": b"old"})
    upper = _layer(tmp_path, "b.tar.gz", {"app/x": b"new"})
    rootfs = tmp_path / "rootfs"
    extract_layers([lower, upper], rootfs)
    assert (rootfs / "app/x").read_bytes() == b"new"


def test_whiteout_deletes_a_file_from_a_lower_layer(tmp_path):
    lower = _layer(tmp_path, "a.tar.gz", {"app/keep": b"k", "app/gone": b"g"})
    upper = _layer(tmp_path, "b.tar.gz", {"app/.wh.gone": None})
    rootfs = tmp_path / "rootfs"
    extract_layers([lower, upper], rootfs)
    assert (rootfs / "app/keep").exists()
    assert not (rootfs / "app/gone").exists()


def test_opaque_whiteout_clears_a_directory(tmp_path):
    lower = _layer(tmp_path, "a.tar.gz", {"app/a": b"1", "app/b": b"2"})
    upper = _layer(tmp_path, "b.tar.gz", {"app/.wh..wh..opq": None, "app/c": b"3"})
    rootfs = tmp_path / "rootfs"
    extract_layers([lower, upper], rootfs)
    assert not (rootfs / "app/a").exists()
    assert not (rootfs / "app/b").exists()
    assert (rootfs / "app/c").read_bytes() == b"3"


def test_path_traversal_is_refused(tmp_path):
    evil = _layer(tmp_path, "evil.tar.gz", {"../escaped": b"pwned"})
    with pytest.raises(RuntimeError, match="path traversal"):
        extract_layers([evil], tmp_path / "rootfs")
    assert not (tmp_path / "escaped").exists()


def test_extraction_is_idempotent(tmp_path):
    layer = _layer(tmp_path, "a.tar.gz", {"app/x": b"v"})
    rootfs = tmp_path / "rootfs"
    extract_layers([layer], rootfs)
    extract_layers([layer], rootfs)
    assert (rootfs / "app/x").read_bytes() == b"v"


# --- mirror rotation ------------------------------------------------------------------

def test_download_falls_through_to_a_working_mirror(tmp_path, monkeypatch):
    """A 1.8 GB pull must not die because one host is refusing today."""
    import fetch_fhir_server as F

    attempted = []

    def fake(registry, repo, digest, dest, timeout):
        attempted.append(registry)
        if registry != "https://good":
            raise RuntimeError("refused")
        dest.write_bytes(b"payload")
        return dest

    monkeypatch.setattr(F, "_download_blob_from", fake)
    out = F.download_blob(["https://bad1", "https://bad2", "https://good"],
                          "r/i", "sha256:x", tmp_path / "blob", 5)
    assert out.read_bytes() == b"payload"
    assert attempted == ["https://bad1", "https://bad2", "https://good"]


def test_download_reports_all_mirrors_failing(tmp_path, monkeypatch):
    import fetch_fhir_server as F

    monkeypatch.setattr(F, "_download_blob_from",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("nope")))
    with pytest.raises(RuntimeError, match="all registries failed"):
        F.download_blob(["https://a", "https://b"], "r/i", "sha256:x", tmp_path / "b", 5)


def test_a_bare_registry_string_still_works(tmp_path, monkeypatch):
    import fetch_fhir_server as F

    monkeypatch.setattr(F, "_download_blob_from",
                        lambda reg, repo, dig, dest, t: (dest.write_bytes(b"k"), dest)[1])
    assert F.download_blob("https://one", "r/i", "sha256:x", tmp_path / "b", 5).exists()


def test_cached_blob_skips_the_network_entirely(tmp_path, monkeypatch):
    import hashlib

    import fetch_fhir_server as F

    dest = tmp_path / "blob"
    dest.write_bytes(b"cached")
    digest = "sha256:" + hashlib.sha256(b"cached").hexdigest()
    monkeypatch.setattr(F, "_download_blob_from",
                        lambda *a, **k: pytest.fail("must not hit the network"))
    assert F.download_blob(["https://x"], "r/i", digest, dest, 5) == dest


def test_pinned_manifest_matches_the_live_image():
    """The committed manifest pins the exact image; layer count and digest are fixed."""
    import json

    manifest = json.loads(
        Path("data/medagentbench_image_manifest.json").read_text(encoding="utf-8")
    )
    assert len(manifest["layers"]) == 38
    assert manifest["config"]["digest"].startswith("sha256:a232b7b22b86")
    assert sum(layer["size"] for layer in manifest["layers"]) > 1.8e9
