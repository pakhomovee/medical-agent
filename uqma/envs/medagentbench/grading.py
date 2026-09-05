"""Grading via the upstream reference solutions.

``refsol.py`` is distributed separately (a Box link in the upstream README), almost
certainly to keep the answer key out of training crawls. It is gitignored here and must
never be committed -- see plan §6.1, which also flags that our derived turn-level labels
inherit the same constraint and need a gated release.

Because ``sol`` is ``null`` for 9 of the 10 v1 templates, the task file alone cannot
grade anything: the graders recompute expected answers by querying FHIR and by reading
the POST payload out of the conversation history. So without ``refsol.py`` there is no
success rate at all, and gate G2 must say so rather than silently reporting zero.
"""

from __future__ import annotations

import importlib.util
import sys
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class GradingInput:
    """Shim matching what upstream graders expect of a ``TaskOutput``.

    Upstream calls ``grader(case_data, results, fhir_api_base)`` where ``results`` is a
    ``TaskOutput``. Graders read ``.result`` (the FINISH payload) and, for action tasks,
    ``.history`` -- which is where the POST payload lives, since POSTs are never sent to
    the server and so leave no trace in FHIR state.
    """

    result: str | None
    history: list[dict] = field(default_factory=list)
    status: str = ""
    index: int = 0


class RefsolGrader:
    """Loads ``refsol.py`` by path and dispatches on the task category.

    Mirrors upstream ``eval()``: the grader function is named after the task category
    (``task7`` for ``task7_3``), and any exception counts as incorrect.
    """

    def __init__(self, refsol_path: str | Path, fhir_api_base: str) -> None:
        self.refsol_path = Path(refsol_path)
        self.fhir_api_base = fhir_api_base
        self._module = self._load(self.refsol_path)

    @staticmethod
    def _load(path: Path):
        if not path.exists():
            raise FileNotFoundError(
                f"refsol.py not found at {path}. Download it from the Box link in the "
                "MedAgentBench README and place it there. It is gitignored on purpose; "
                "do not commit it."
            )
        spec = importlib.util.spec_from_file_location("uqma_refsol", path)
        if spec is None or spec.loader is None:
            raise ImportError(f"could not load a module from {path}")
        module = importlib.util.module_from_spec(spec)
        # Registered so that any relative machinery inside refsol resolves.
        sys.modules["uqma_refsol"] = module
        spec.loader.exec_module(module)
        return module

    def available_categories(self) -> list[str]:
        return sorted(
            name
            for name in dir(self._module)
            if name.startswith("task") and callable(getattr(self._module, name))
        )

    def grade(self, case_data: dict, output: GradingInput) -> bool:
        """True if the task passes. Exceptions are swallowed into False, as upstream does.

        Upstream additionally skips grading entirely when ``result`` is None (the agent
        never called FINISH) and counts those as incorrect; ``grade_task`` below applies
        that rule so the arithmetic matches.
        """
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
