"""Registry auth and layer extraction for the no-Docker FHIR server fetch.

The auth logic is the part that broke in the field: the first version hardcoded Docker
Hub's token endpoint, which 403s on a mirror that puts its own at /auth/token. Following
the WWW-Authenticate challenge is what makes any mirror work, so it is pinned here.
"""

from __future__ import annotations

import io
import json
import sys
import tarfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from fetch_fhir_server import extract_layers, parse_challenge  # noqa: E402


# --- WWW-Authenticate parsing ---------------------------------------------------------

def test_parses_docker_hub_challenge():
    challenge = parse_challenge(
        'Bearer realm="https://auth.docker.io/token",service="registry.docker.io"'
    )
    assert challenge["realm"] == "https://auth.docker.io/token"
    assert challenge["service"] == "registry.docker.io"


def test_parses_mirror_challenge_with_a_different_realm_path():
    """The daocloud shape -- the case the hardcoded /token guess got wrong."""
    challenge = parse_challenge(
        'Bearer realm="https://docker.m.daocloud.io/auth/token",service="docker.m.daocloud.io"'
    )
    assert challenge["realm"] == "https://docker.m.daocloud.io/auth/token"
    assert challenge["service"] == "docker.m.daocloud.io"


def test_parses_challenge_carrying_a_scope():
    challenge = parse_challenge(
        'Bearer realm="https://r/token",service="s",scope="repository:a/b:pull"'
    )
    assert challenge["scope"] == "repository:a/b:pull"


def test_challenge_without_service_is_still_usable():
    assert parse_challenge('Bearer realm="https://r/token"') == {"realm": "https://r/token"}


def test_non_bearer_challenge_is_ignored():
    assert parse_challenge('Basic realm="x"') == {}
    assert parse_challenge("") == {}


def test_challenge_parsing_is_case_insensitive_on_the_scheme():
    assert parse_challenge('bearer realm="https://r/t"')["realm"] == "https://r/t"


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

    monkeypatch.setattr(F, "fetch_blob", fake)
    out = F.cached_or_fetch(["https://bad1", "https://bad2", "https://good"],
                          "r/i", "sha256:x", tmp_path / "blob", 5)
    assert out.read_bytes() == b"payload"
    assert attempted == ["https://bad1", "https://bad2", "https://good"]


def test_download_reports_all_mirrors_failing(tmp_path, monkeypatch):
    import fetch_fhir_server as F

    monkeypatch.setattr(F, "fetch_blob",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("nope")))
    with pytest.raises(RuntimeError, match="all registries failed"):
        F.cached_or_fetch(["https://a", "https://b"], "r/i", "sha256:x", tmp_path / "b", 5)


def test_a_bare_registry_string_still_works(tmp_path, monkeypatch):
    import fetch_fhir_server as F

    monkeypatch.setattr(F, "fetch_blob",
                        lambda reg, repo, dig, dest, t: (dest.write_bytes(b"k"), dest)[1])
    assert F.cached_or_fetch("https://one", "r/i", "sha256:x", tmp_path / "b", 5).exists()


def test_cached_blob_skips_the_network_entirely(tmp_path, monkeypatch):
    import hashlib

    import fetch_fhir_server as F

    dest = tmp_path / "blob"
    dest.write_bytes(b"cached")
    digest = "sha256:" + hashlib.sha256(b"cached").hexdigest()
    monkeypatch.setattr(F, "fetch_blob",
                        lambda *a, **k: pytest.fail("must not hit the network"))
    assert F.cached_or_fetch(["https://x"], "r/i", digest, dest, 5) == dest

# --- preflight ------------------------------------------------------------------------

def test_check_requires_a_working_blob_not_just_a_manifest(monkeypatch, tmp_path):
    """Mirrors commonly resolve a manifest and then 404 the blobs, so both are probed."""
    import fetch_fhir_server as F

    manifest = {"config": {"digest": "sha256:cfg"},
                "layers": [{"digest": "sha256:aaa", "size": 10}]}
    monkeypatch.setattr(F, "fetch_manifest", lambda reg, *a: manifest)
    monkeypatch.setattr(F, "verify_against_pin", lambda *a, **k: None)

    def blob(registry, repo, digest, dest, timeout, **kw):
        if registry != "https://good":
            raise RuntimeError("404")
        dest.write_bytes(b"x")
        return dest

    monkeypatch.setattr(F, "fetch_blob", blob)
    assert F.check(["https://manifest-only", "https://good"], "r/i", "latest", 5) == \
        ["https://good"]


def test_check_returns_empty_when_nothing_works(monkeypatch):
    import fetch_fhir_server as F

    monkeypatch.setattr(F, "fetch_manifest",
                        lambda *a: (_ for _ in ()).throw(RuntimeError("unreachable")))
    monkeypatch.setattr(F, "verify_against_pin", lambda *a, **k: None)
    assert F.check(["https://a", "https://b"], "r/i", "latest", 5) == []


def test_launcher_repoints_absolute_data_and_config_paths(tmp_path):
    """The image hardcodes /data and /configs; nothing may need to exist at the root."""
    import fetch_fhir_server as F

    rootfs = tmp_path / "rootfs"
    script = F.write_launcher(rootfs, tmp_path, port=8080, heap="2g")
    text = script.read_text(encoding="utf-8")
    assert str(rootfs) in text
    assert "application.local.yaml" in text          # rewritten copy, not the packaged one
    assert "SERVER_PORT=\"8080\"" in text
    assert script.stat().st_mode & 0o111             # executable


def test_launcher_fails_loudly_without_java(tmp_path):
    import fetch_fhir_server as F

    text = F.write_launcher(tmp_path / "r", tmp_path, 8080, "1g").read_text(encoding="utf-8")
    assert "openjdk-17-jre-headless" in text

# --- supply chain: the pinned manifest is the trust anchor -----------------------------
# Digest verification only proves a blob matches the manifest that named it, and that
# manifest comes from the same registry as the blobs. A hostile mirror could serve a
# poisoned manifest plus matching poisoned layers and every digest check would pass.
# Comparing against a manifest pinned from Docker Hub is what closes that.

def _manifest(config="sha256:cfg", layers=("sha256:a", "sha256:b")):
    return {"config": {"digest": config},
            "layers": [{"digest": d, "size": 10} for d in layers]}


def _pin(tmp_path, manifest):
    path = tmp_path / "pin.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path


def test_matching_manifest_is_accepted(tmp_path):
    import fetch_fhir_server as F

    F.verify_against_pin(_manifest(), _pin(tmp_path, _manifest()))


def test_layer_substitution_is_refused(tmp_path):
    """The attack this exists for: a mirror swapping a layer for its own."""
    import fetch_fhir_server as F

    poisoned = _manifest(layers=("sha256:a", "sha256:EVIL"))
    with pytest.raises(RuntimeError, match="DOES NOT MATCH THE PINNED COPY"):
        F.verify_against_pin(poisoned, _pin(tmp_path, _manifest()))


def test_config_substitution_is_refused(tmp_path):
    import fetch_fhir_server as F

    with pytest.raises(RuntimeError, match="config digest differs"):
        F.verify_against_pin(_manifest(config="sha256:EVIL"), _pin(tmp_path, _manifest()))


def test_extra_layer_is_refused(tmp_path):
    import fetch_fhir_server as F

    with pytest.raises(RuntimeError, match="DOES NOT MATCH"):
        F.verify_against_pin(_manifest(layers=("sha256:a", "sha256:b", "sha256:extra")),
                             _pin(tmp_path, _manifest()))


def test_layer_order_alone_is_not_a_mismatch(tmp_path):
    import fetch_fhir_server as F

    F.verify_against_pin(_manifest(layers=("sha256:b", "sha256:a")),
                         _pin(tmp_path, _manifest()))


def test_absent_pin_warns_rather_than_silently_trusting(tmp_path, capsys):
    import fetch_fhir_server as F

    F.verify_against_pin(_manifest(), tmp_path / "does-not-exist.json")
    assert "cannot verify" in capsys.readouterr().out


def test_the_committed_pin_matches_the_image_we_run():
    """Pinned from Docker Hub; any drift here means the upstream image moved."""
    import fetch_fhir_server as F

    pinned = json.loads(F.PINNED_MANIFEST.read_text(encoding="utf-8"))
    config, layers = F.manifest_identity(pinned)
    assert config == ("sha256:a232b7b22b86facd826c042c6c3aba9a"
                      "80769db7fa27e23decd20e40c26c9b98")
    assert len(layers) == 38


def test_check_rejects_a_registry_whose_manifest_fails_the_pin(monkeypatch):
    """A mirror serving different content is reported unsafe, not merely 'working'."""
    import fetch_fhir_server as F

    monkeypatch.setattr(F, "fetch_manifest",
                        lambda reg, *a: {"config": {"digest": "sha256:x"},
                                         "layers": [{"digest": "sha256:y", "size": 1}]})
    monkeypatch.setattr(F, "fetch_blob", lambda *a, **k: pytest.fail("must not download"))
    monkeypatch.setattr(F, "verify_against_pin",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("MISMATCH\nline2")))
    assert F.check(["https://evil"], "r/i", "latest", 5) == []


def test_launcher_binds_loopback_by_default(tmp_path):
    """The packaged config sets no address, so Spring Boot would bind 0.0.0.0 -- an
    unauthenticated FHIR API on every interface. Everything here is same-host."""
    import fetch_fhir_server as F

    text = F.write_launcher(tmp_path / "r", tmp_path, 8080, "2g").read_text(encoding="utf-8")
    assert 'SERVER_ADDRESS="${FHIR_BIND:-127.0.0.1}"' in text


def test_launcher_bind_is_overridable(tmp_path):
    import fetch_fhir_server as F

    text = F.write_launcher(tmp_path / "r", tmp_path, 8080, "2g").read_text(encoding="utf-8")
    assert "FHIR_BIND" in text          # documented escape hatch, not hardcoded
