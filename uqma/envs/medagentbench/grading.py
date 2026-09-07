"""Grading via the upstream reference solutions.

``refsol.py`` is distributed separately (a Box link in the upstream README), almost
certainly to keep the answer key out of training crawls. It is gitignored here and must
never be committed -- see plan §6.1, which also flags that our derived turn-level labels
inherit the same constraint and need a gated release.

Because ``sol`` is ``null`` for 9 of the 10 v1 templates, the task file alone cannot grade
anything: the graders recompute expected answers by querying FHIR, and read the POST
payload out of the conversation history -- which is the only place it exists, since POSTs
are never sent to the server. So without ``refsol.py`` there is no success rate at all,
and callers must say so rather than silently reporting zero.

Two shims are needed to run upstream graders against our trajectories:

* ``refsol.py`` opens with ``from .utils import *`` and calls ``send_get_request``. Loaded
  standalone that relative import fails, so we install a synthetic parent package whose
  ``utils`` provides that function via our own ``FhirClient``.
* ``extract_posts`` walks ``results.history`` expecting **objects** with ``.role`` and
  ``.content``, and expects the assistant role to be spelled ``'agent'`` (AgentBench's
  convention). Our loop stores dicts with ``'assistant'``, as the chat API requires, so
  ``GradingInput.from_trajectory`` translates.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import types
from dataclasses import dataclass, field
from pathlib import Path

# Upstream spells the assistant role this way; graders match on it literally.
AGENT_ROLE = "agent"


@dataclass(frozen=True)
class Message:
    """History entry with attribute access, as upstream graders expect."""

    role: str
    content: str


@dataclass
class GradingInput:
    """Shim matching what upstream graders expect of a ``TaskOutput``.

    Upstream calls ``grader(case_data, results, fhir_api_base)`` where ``results`` is a
    ``TaskOutput``. Graders read ``.result`` (the FINISH payload) and, for action tasks,
    ``.history``.
    """

    result: str | None
    history: list[Message] = field(default_factory=list)
    status: str = ""
    index: int = 0

    @classmethod
    def from_trajectory(cls, trajectory) -> GradingInput:
        """Build grading input from one of our ``Trajectory`` records."""
        return cls(
            result=trajectory.result,
            history=to_agent_history(trajectory.history),
            status=trajectory.status,
        )


def to_agent_history(history: list[dict]) -> list[Message]:
    """Convert our dict transcript to upstream's attribute-and-``agent`` form."""
    return [
        Message(
            role=AGENT_ROLE if entry.get("role") == "assistant" else entry.get("role", ""),
            content=entry.get("content") or "",
        )
        for entry in history
    ]


def _make_utils_module(fhir_api_base: str) -> types.ModuleType:
    """Provide the ``utils`` names ``refsol`` star-imports.

    Implemented over our own client rather than vendoring upstream's file, so there is a
    single HTTP path and one place where timeouts and truncation are configured.
    """
    from .fhir import FhirClient

    module = types.ModuleType("uqma_refsol_pkg.utils")
    client = FhirClient(fhir_api_base, max_chars=None)

    def send_get_request(url, params=None, headers=None):
        """Upstream's contract: {"status_code", "data"} on success, {"error"} otherwise.

        ``data`` MUST be the raw response text, not a parsed object. Upstream's version
        only parses when the content type contains 'application/json', and HAPI sends
        'application/fhir+json' -- so in practice upstream always returns text, and the
        graders call ``json.loads`` on it themselves:

            get_res = json.loads(send_get_request(url)['data'])

        Handing back a dict makes that raise TypeError, which the graders swallow in a
        bare ``except`` and turn into False. Every FHIR-querying grader (task2, 4, 6, 7,
        9, 10) then fails regardless of the agent's answer, and the whole run reads 0%.
        """
        result = client.get(url if not params else f"{url}?{_encode(params)}")
        if not result.ok:
            return {"error": result.error}
        data = result.data
        if not isinstance(data, str):
            data = json.dumps(data)
        return {"status_code": 200, "data": data}

    def verify_fhir_server(api_base):
        return FhirClient(api_base).verify()

    module.send_get_request = send_get_request
    module.verify_fhir_server = verify_fhir_server
    module.json = json
    return module


def _encode(params: dict) -> str:
    from urllib.parse import urlencode

    return urlencode(params)


class RefsolGrader:
    """Loads ``refsol.py`` by path and dispatches on the task category.

    Mirrors upstream ``eval()``: the grader function is named after the task category
    (``task7`` for ``task7_3``), and any exception counts as incorrect.
    """

    def __init__(self, refsol_path: str | Path, fhir_api_base: str) -> None:
        self.refsol_path = Path(refsol_path)
        self.fhir_api_base = fhir_api_base.rstrip("/") + "/"  # graders build f'{base}Observation'
        self._module = self._load(self.refsol_path, self.fhir_api_base)

    @staticmethod
    def _load(path: Path, fhir_api_base: str):
        if not path.exists():
            raise FileNotFoundError(
                f"refsol.py not found at {path}. Download it from the Box link in the "
                "MedAgentBench README and place it there. It is gitignored on purpose; "
                "do not commit it."
            )

        # Synthetic parent package so `from .utils import *` resolves.
        pkg = types.ModuleType("uqma_refsol_pkg")
        pkg.__path__ = [str(path.parent)]
        sys.modules["uqma_refsol_pkg"] = pkg
        sys.modules["uqma_refsol_pkg.utils"] = _make_utils_module(fhir_api_base)

        spec = importlib.util.spec_from_file_location("uqma_refsol_pkg.refsol", path)
        if spec is None or spec.loader is None:
            raise ImportError(f"could not load a module from {path}")
        module = importlib.util.module_from_spec(spec)
        module.__package__ = "uqma_refsol_pkg"
        sys.modules["uqma_refsol_pkg.refsol"] = module
        spec.loader.exec_module(module)
        return module

    def available_categories(self) -> list[str]:
        return sorted(
            name
            for name in dir(self._module)
            if name.startswith("task") and callable(getattr(self._module, name))
        )

    def grade(self, case_data: dict, output: GradingInput) -> bool:
        """True if the task passes. Exceptions are swallowed into False, as upstream does."""
        category = case_data["id"].split("_")[0]
        grader = getattr(self._module, category, None)
        if grader is None:
            raise AttributeError(f"refsol has no grader named {category!r}")
        try:
            return grader(case_data, output, self.fhir_api_base) is True
        except Exception:
            return False


def grade_task(grader: RefsolGrader | None, case_data: dict, output: GradingInput) -> bool | None:
    """Apply the upstream success rule.

    Returns None when no grader is loaded, so callers can distinguish "not graded" from
    "graded and wrong". Conflating those is how a missing answer key turns into a
    reported 0% success rate.
    """
    if grader is None:
        return None
    if output.result is None:
        return False  # upstream: only COMPLETED samples are graded; the rest are incorrect
    return grader.grade(case_data, output)
