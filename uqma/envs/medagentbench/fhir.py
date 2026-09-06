"""FHIR client and the tool catalogue.

Two things worth knowing before changing anything here:

* Upstream never sends POSTs. It parses the payload for JSON validity and replies with a
  canned success string; the server is never touched (plan §3 R4). We reproduce that,
  because a landing write would fork us from the published baselines. ``allow_writes``
  exists only so the decision is visible in code rather than implicit — it is off, and
  the plan says to leave it off.
* Upstream builds the GET URL as ``r[3:].strip() + '&_format=json'``, unconditionally.
  A URL with no query string therefore gets a malformed ``&``. That is a real upstream
  quirk and we replicate it, because gate G4 compares per-task outcomes and "we fixed
  their bug" shows up as disagreement.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import requests

# POST-capable resources per funcs_v1.json. Used to validate write payloads and, if
# writes were ever enabled, as the allowlist -- never a passthrough (plan discussion).
WRITABLE_RESOURCES = frozenset({"Observation", "MedicationRequest", "ServiceRequest"})


def load_functions(path: str | Path) -> list[dict]:
    with open(path, encoding="utf-8") as handle:
        funcs = json.load(handle)
    if not isinstance(funcs, list):
        raise ValueError(f"{path}: expected a JSON list of function specs")
    return funcs


@dataclass
class GetResult:
    ok: bool
    data: object = None
    error: str | None = None


class FhirClient:
    """Read-only client against a HAPI FHIR server.

    Parameters
    ----------
    api_base:
        e.g. ``http://localhost:8080/fhir``. No trailing slash.
    timeout:
        Per-request timeout in seconds. A hung server otherwise stalls a whole sweep.
    max_chars:
        Truncation applied to the serialised response before it goes into the model's
        context. Upstream applies no cap, but an unbounded FHIR Bundle can blow the
        context window and, locally, memory. Default is generous; set ``None`` for exact
        upstream behaviour when running gate G4.
    """

    def __init__(
        self,
        api_base: str,
        timeout: float = 30.0,
        max_chars: int | None = 200_000,
        session: requests.Session | None = None,
        allow_writes: bool = False,
    ) -> None:
        self.api_base = api_base.rstrip("/")
        self.timeout = timeout
        self.max_chars = max_chars
        self.allow_writes = allow_writes
        self._session = session or requests.Session()

    def verify(self) -> bool:
        """Upstream's connection check: GET {api_base}/metadata must return 200."""
        try:
            response = self._session.get(f"{self.api_base}/metadata", timeout=self.timeout)
            return response.status_code == 200
        except requests.RequestException:
            return False

    def get(self, url: str) -> GetResult:
        try:
            response = self._session.get(url, timeout=self.timeout)
            response.raise_for_status()
        except requests.RequestException as exc:
            return GetResult(ok=False, error=str(exc))

        # PARITY-CRITICAL. HAPI answers with 'application/fhir+json', which does NOT
        # contain the substring 'application/json', so this branch is not taken and the
        # body stays a *string*. Upstream's send_get_request has the identical check, so
        # upstream also feeds the model raw JSON text. Parsing here would make the
        # observation f-string render a Python dict repr (single quotes, True/False)
        # instead of JSON, silently changing every prompt and breaking gate G4.
        # Do not "fix" this. Callers that need structured data should json.loads it.
        content_type = response.headers.get("content-type", "")
        try:
            data = response.json() if "application/json" in content_type else response.text
        except ValueError:
            data = response.text

        if self.max_chars is not None:
            rendered = data if isinstance(data, str) else json.dumps(data)
            if len(rendered) > self.max_chars:
                return GetResult(
                    ok=True,
                    data=rendered[: self.max_chars] + f"...[truncated at {self.max_chars} chars]",
                )
        return GetResult(ok=True, data=data)


@dataclass
class PostCheck:
    """Outcome of validating a POST payload without sending it."""

    json_valid: bool
    resource_type: str | None = None
    resource_allowed: bool = False
    error: str | None = None
    payload: dict | None = None

    @property
    def schema_valid(self) -> bool:
        """Parses as JSON *and* names a resource type the benchmark can write.

        This is the numerator of the schema-valid rate in gate G2, which separates
        "the model cannot emit well-formed tool calls" from "the model is clinically
        wrong" -- the format-vs-clinical distinction of plan §4.4.
        """
        return self.json_valid and self.resource_allowed


def check_post_payload(body: str) -> PostCheck:
    """Validate a POST body the way upstream does, plus a resource-type check.

    Upstream only does ``json.loads``; the resource-type check is ours and is reported
    separately so that parity with upstream is preserved (a payload upstream accepts is
    still ``json_valid`` here even if the resource type is wrong).
    """
    try:
        payload = json.loads(body)
    except (json.JSONDecodeError, ValueError) as exc:
        return PostCheck(json_valid=False, error=str(exc))

    if not isinstance(payload, dict):
        return PostCheck(
            json_valid=True, error=f"payload is {type(payload).__name__}, expected object"
        )

    resource_type = payload.get("resourceType")
    return PostCheck(
        json_valid=True,
        resource_type=resource_type,
        resource_allowed=resource_type in WRITABLE_RESOURCES,
        payload=payload,
    )
