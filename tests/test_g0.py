"""Gate G0 checks, against faked servers.

The determinism measurement is the part worth testing carefully: it exists to put a
number on logprob drift, and a check that silently reports "fine" when it never ran would
be worse than no check at all.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from g0_environment import (  # noqa: E402
    _total,
    check_determinism,
    check_fhir,
    check_model_server,
)


class FakeResponse:
    def __init__(self, payload, status_code=200):
        self._payload = payload
        self.status_code = status_code
        self.headers = {"content-type": "application/json"}
        self.text = ""

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


def completion(text="GET /fhir/Patient?_id=S1", logprobs=None):
    choice = {"message": {"content": text}, "finish_reason": "stop"}
    if logprobs is not None:
        choice["logprobs"] = {"content": [{"token": "t", "logprob": lp} for lp in logprobs]}
    return {"choices": [choice], "usage": {}}


class FakeModelSession:
    """Serves a scripted sequence of completions."""

    def __init__(self, completions, models=("m",)):
        self._completions = list(completions)
        self._models = list(models)
        self.headers = {}
        self.n_posts = 0

    def get(self, url, timeout=None):
        return FakeResponse({"data": [{"id": m} for m in self._models]})

    def post(self, url, json=None, timeout=None):
        payload = self._completions[min(self.n_posts, len(self._completions) - 1)]
        self.n_posts += 1
        return FakeResponse(payload)


def patch_session(monkeypatch, module_name, session):
    import requests

    monkeypatch.setattr(requests, "Session", lambda: session)
    return session


# --- FHIR ---------------------------------------------------------------------------

def test_fhir_unreachable_is_reported(monkeypatch):
    from uqma.envs.medagentbench import fhir as fhir_mod

    monkeypatch.setattr(fhir_mod.FhirClient, "verify", lambda self: False)
    result = check_fhir("http://nope/fhir", timeout=1)
    assert not result["ok"] and not result["reachable"]


def test_fhir_up_but_empty_fails(monkeypatch):
    """The silent failure this check exists for: live server, no data."""
    from uqma.envs.medagentbench import fhir as fhir_mod

    monkeypatch.setattr(fhir_mod.FhirClient, "verify", lambda self: True)
    monkeypatch.setattr(
        fhir_mod.FhirClient, "get",
        lambda self, url: fhir_mod.GetResult(ok=True, data={"total": 0}),
    )
    result = check_fhir("http://x/fhir", timeout=1)
    assert result["reachable"] and not result["ok"]
    assert "empty" in result["note"]


def test_fhir_populated_passes(monkeypatch):
    from uqma.envs.medagentbench import fhir as fhir_mod

    monkeypatch.setattr(fhir_mod.FhirClient, "verify", lambda self: True)
    monkeypatch.setattr(
        fhir_mod.FhirClient, "get",
        lambda self, url: fhir_mod.GetResult(ok=True, data={"total": 1234}),
    )
    result = check_fhir("http://x/fhir", timeout=1)
    assert result["ok"]
    assert result["counts"]["Patient"] == 1234


# --- model server -------------------------------------------------------------------

def test_model_server_without_logprobs_fails(monkeypatch):
    patch_session(monkeypatch, "g0", FakeModelSession([completion(logprobs=None)]))
    result = check_model_server("http://x/v1", timeout=1)
    assert not result["ok"]
    assert "logprobs" in result["error"]


def test_model_server_with_logprobs_passes(monkeypatch):
    patch_session(monkeypatch, "g0", FakeModelSession([completion(logprobs=[-0.1, -0.2])]))
    result = check_model_server("http://x/v1", timeout=1)
    assert result["ok"] and result["n_logprobs"] == 2 and result["model"] == "m"


def test_model_server_with_no_models_fails(monkeypatch):
    patch_session(monkeypatch, "g0", FakeModelSession([completion()], models=()))
    assert not check_model_server("http://x/v1", timeout=1)["ok"]


# --- determinism --------------------------------------------------------------------

def test_determinism_clean_run(monkeypatch):
    same = completion("A", logprobs=[-0.10, -0.20])
    patch_session(monkeypatch, "g0", FakeModelSession([same] * 4))
    result = check_determinism("http://x/v1", "m", trials=4, timeout=1)
    assert result["ok"]
    assert result["n_distinct_texts"] == 1
    assert result["max_logprob_delta"] == 0.0


def test_determinism_detects_text_divergence(monkeypatch):
    patch_session(monkeypatch, "g0", FakeModelSession(
        [completion("A", [-0.1]), completion("B", [-0.1]), completion("A", [-0.1])]
    ))
    result = check_determinism("http://x/v1", "m", trials=3, timeout=1)
    assert not result["ok"]
    assert result["n_distinct_texts"] == 2


def test_determinism_quantifies_logprob_drift(monkeypatch):
    patch_session(monkeypatch, "g0", FakeModelSession(
        [completion("A", [-0.10, -0.20]),
         completion("A", [-0.10, -0.20005]),
         completion("A", [-0.10, -0.19990])]
    ))
    result = check_determinism("http://x/v1", "m", trials=3, timeout=1)
    assert result["ok"]  # text identical
    assert result["max_logprob_delta"] > 0  # but logprobs drifted
    assert result["max_logprob_delta"] < 1e-3


def test_determinism_handles_varying_token_counts(monkeypatch):
    patch_session(monkeypatch, "g0", FakeModelSession(
        [completion("A", [-0.1, -0.2]), completion("A", [-0.1])]
    ))
    result = check_determinism("http://x/v1", "m", trials=2, timeout=1)
    assert not result["consistent_token_count"]
    assert result["max_logprob_delta"] is None


def test_determinism_note_warns_about_batch_composition(monkeypatch):
    patch_session(monkeypatch, "g0", FakeModelSession([completion("A", [-0.1])] * 2))
    result = check_determinism("http://x/v1", "m", trials=2, timeout=1)
    assert "batch composition is not controlled" in result["note"]


# --- FHIR content-type parity ---------------------------------------------------------

def test_fhir_bodies_stay_text_because_hapi_sends_application_fhir_json(monkeypatch):
    """PARITY-CRITICAL regression test.

    HAPI answers with 'application/fhir+json', which does not contain the substring
    'application/json', so the client leaves the body as text -- exactly as upstream's
    send_get_request does. Parsing it would make the observation f-string render a Python
    dict repr instead of JSON and silently change every prompt.
    """
    import requests

    from uqma.envs.medagentbench.fhir import FhirClient

    class FhirResponse:
        status_code = 200
        headers = {"content-type": "application/fhir+json;charset=UTF-8"}
        text = '{"resourceType":"Bundle","total":695}'

        def json(self):
            raise AssertionError("must not parse: upstream feeds the model raw text")

        def raise_for_status(self):
            pass

    class S:
        headers = {}

        def get(self, url, timeout=None):
            return FhirResponse()

    monkeypatch.setattr(requests, "Session", lambda: S())
    result = FhirClient("http://x/fhir").get("http://x/fhir/Patient")
    assert result.ok
    assert isinstance(result.data, str)
    assert result.data == '{"resourceType":"Bundle","total":695}'


def test_g0_parses_the_count_out_of_that_text():
    from uqma.envs.medagentbench.fhir import GetResult

    assert _total(GetResult(ok=True, data='{"resourceType":"Bundle","total":695}')) == 695
    assert _total(GetResult(ok=True, data={"total": 12})) == 12
    assert _total(GetResult(ok=True, data="not json")) is None
    assert _total(GetResult(ok=True, data='{"resourceType":"Bundle"}')) is None
