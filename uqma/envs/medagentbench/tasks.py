"""Task loading and template analysis for MedAgentBench.

Gate G1 asks whether the 300 (public: 100) tasks are parameterised templates over a
patient pool, because that determines the statistical power available to the whole
thesis (plan §3 R2). This module answers it two independent ways:

  1. the ``id`` field, which is authoritative: ids are ``task{category}_{instance}``
  2. masking identifiers out of the instruction text and grouping the residue

Agreement between the two is the evidence; disagreement means the ids do not mean
what we think they mean.
"""

from __future__ import annotations

import json
import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

# MedAgentBench MRNs are an S followed by digits. Ordered most-specific first so that
# e.g. a datetime is masked before its component integers are.
_MASKS: list[tuple[str, re.Pattern[str]]] = [
    ("<MRN>", re.compile(r"\bS\d{6,}\b")),
    ("<DATETIME>", re.compile(r"\d{4}-\d{2}-\d{2}T[\d:+\-.]+")),
    ("<DATE>", re.compile(r"\b\d{4}-\d{2}-\d{2}\b")),
    ("<QUOTED>", re.compile(r"[\"“”][^\"“”]{1,80}[\"“”]")),
    ("<NUM>", re.compile(r"\b\d+(?:\.\d+)?\b")),
]

# Capitalised bigrams that are plausibly person names. Only applied to the residue after
# the masks above, and deliberately conservative: over-masking would merge templates that
# are genuinely distinct, which is the error that would make G1 look better than it is.
_NAME = re.compile(r"\b[A-Z][a-z]{1,20} [A-Z][a-z]{1,20}\b")

# POST-capable FHIR resources in funcs_v1.json. A task whose instruction implies writing
# one of these is an "action" task in the sense of the 150/150 split.
_ACTION_CUES = re.compile(
    r"\b(order|orders|record|records|prescrib\w*|refer(?:ral)?|place\b|document)\b", re.I
)
_CONDITIONAL_CUES = re.compile(r"\bif\b", re.I)


def normalise(text: str) -> str:
    """NFKC-normalise and collapse whitespace.

    The corpus mixes ASCII and curly apostrophes ("What's" vs "What's"), which would
    otherwise split one template into two.
    """
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("’", "'").replace("‘", "'")
    return re.sub(r"\s+", " ", text).strip()


def mask(text: str) -> str:
    """Replace instance-specific values so that sibling instances collapse to one string."""
    out = normalise(text)
    for token, pattern in _MASKS:
        out = pattern.sub(token, out)
    return _NAME.sub("<NAME>", out)


@dataclass(frozen=True)
class Task:
    id: str
    instruction: str
    context: str
    sol: object
    eval_mrn: str | None

    @property
    def category(self) -> str:
        """``task7_3`` -> ``task7``. This is also the refsol grader function name."""
        return self.id.split("_")[0]

    @property
    def instance(self) -> str:
        parts = self.id.split("_", 1)
        return parts[1] if len(parts) > 1 else ""

    @property
    def has_solution(self) -> bool:
        """Whether a reference answer ships with the task.

        Almost always False: graders in ``refsol.py`` recompute the expected answer by
        querying the FHIR server, so the task file alone is not sufficient to grade.
        """
        return self.sol is not None

    def prompt_fields(self) -> dict[str, str]:
        return {"context": self.context, "question": self.instruction}


@dataclass
class TemplateGroup:
    key: str
    masked_instruction: str
    task_ids: list[str] = field(default_factory=list)
    is_action: bool = False
    is_conditional: bool = False

    @property
    def n_instances(self) -> int:
        return len(self.task_ids)


def load_tasks(path: str | Path) -> list[Task]:
    with open(path, encoding="utf-8") as handle:
        raw = json.load(handle)
    if not isinstance(raw, list):
        raise ValueError(f"{path}: expected a JSON list of tasks, got {type(raw).__name__}")
    return [
        Task(
            id=item["id"],
            instruction=normalise(item.get("instruction", "")),
            context=normalise(item.get("context", "") or ""),
            sol=item.get("sol"),
            eval_mrn=item.get("eval_MRN"),
        )
        for item in raw
    ]


def classify(tasks: list[Task]) -> tuple[bool, bool]:
    """Return (is_action, is_conditional) for a group of sibling tasks.

    Heuristic, and labelled as such in the report: authoritative classification needs
    ``refsol.py``, which checks whether a POST is expected. Siblings vote so that one
    oddly-worded instance cannot flip a whole template.
    """
    action_votes = sum(bool(_ACTION_CUES.search(t.instruction)) for t in tasks)
    cond_votes = sum(bool(_CONDITIONAL_CUES.search(t.instruction)) for t in tasks)
    n = max(len(tasks), 1)
    return action_votes * 2 > n, cond_votes * 2 > n


def group_by_id(tasks: list[Task]) -> dict[str, list[Task]]:
    groups: dict[str, list[Task]] = defaultdict(list)
    for task in tasks:
        groups[task.category].append(task)
    return dict(groups)


def group_by_mask(tasks: list[Task]) -> dict[str, list[Task]]:
    groups: dict[str, list[Task]] = defaultdict(list)
    for task in tasks:
        groups[mask(task.instruction)].append(task)
    return dict(groups)


def templates(tasks: list[Task]) -> list[TemplateGroup]:
    """Template groups keyed on the authoritative ``id`` prefix."""
    out = []
    for category, members in sorted(
        group_by_id(tasks).items(), key=lambda kv: _category_sort_key(kv[0])
    ):
        is_action, is_conditional = classify(members)
        out.append(
            TemplateGroup(
                key=category,
                masked_instruction=mask(members[0].instruction),
                task_ids=[t.id for t in members],
                is_action=is_action,
                is_conditional=is_conditional,
            )
        )
    return out


def _category_sort_key(category: str) -> tuple[int, str]:
    match = re.search(r"(\d+)$", category)
    return (int(match.group(1)) if match else 1 << 30, category)


def agreement(tasks: list[Task]) -> dict[str, object]:
    """Cross-check id-based grouping against mask-based grouping.

    A mask group that spans several id categories means the masking is too aggressive;
    an id category split across mask groups means instances are not really siblings.
    Both are reported, because either would undermine the clustering assumption in §5.
    """
    by_id = group_by_id(tasks)
    by_mask = group_by_mask(tasks)

    split_categories = {}
    for category, members in by_id.items():
        distinct = {mask(t.instruction) for t in members}
        if len(distinct) > 1:
            split_categories[category] = len(distinct)

    merged_masks = {}
    for masked, members in by_mask.items():
        cats = {t.category for t in members}
        if len(cats) > 1:
            merged_masks[masked[:80]] = sorted(cats)

    return {
        "n_id_groups": len(by_id),
        "n_mask_groups": len(by_mask),
        "categories_split_across_masks": split_categories,
        "masks_spanning_categories": merged_masks,
        "consistent": not split_categories and not merged_masks,
    }


def instances_per_template(tasks: list[Task]) -> Counter[int]:
    return Counter(len(members) for members in group_by_id(tasks).values())


def distinct_patients(tasks: list[Task]) -> set[str]:
    return {t.eval_mrn for t in tasks if t.eval_mrn}
