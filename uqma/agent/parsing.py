"""Parse a model turn into an action, reproducing upstream semantics exactly.

Upstream does, in order:

    r = content.strip().replace('```tool_code', '').replace('```', '').strip()
    GET      -> url = r[3:].strip() + '&_format=json'
    POST     -> payload = json.loads('\\n'.join(r.split('\\n')[1:]))
    FINISH(  -> result = r[len('FINISH('):-1]
    else     -> AGENT_INVALID_ACTION (terminates the episode)

Two upstream behaviours are deliberately preserved because gate G4 compares per-task
outcomes and a "fix" would register as a parity failure:

* ``'&_format=json'`` is appended unconditionally, so a URL with no query string gets a
  stray ``&``.
* A POST whose body is not JSON does *not* terminate the episode; it injects an error
  message and the loop continues. Only an unrecognised prefix terminates.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum


class ActionKind(str, Enum):
    GET = "get"
    POST = "post"
    FINISH = "finish"
    INVALID = "invalid"


@dataclass
class Action:
    kind: ActionKind
    raw: str
    url: str | None = None
    post_body: str | None = None
    finish_payload: str | None = None

    @property
    def is_terminal(self) -> bool:
        return self.kind in (ActionKind.FINISH, ActionKind.INVALID)


def clean(content: str) -> str:
    """Upstream's normalisation, including the Gemini-2.0-Flash fence stripping."""
    return content.strip().replace("```tool_code", "").replace("```", "").strip()


def parse(content: str) -> Action:
    text = clean(content)

    if text.startswith("GET"):
        return Action(kind=ActionKind.GET, raw=text, url=text[3:].strip() + "&_format=json")

    if text.startswith("POST"):
        # Everything after the first line is the payload; the first line carries the URL.
        body = "\n".join(text.split("\n")[1:])
        return Action(kind=ActionKind.POST, raw=text, post_body=body)

    if text.startswith("FINISH("):
        return Action(
            kind=ActionKind.FINISH, raw=text, finish_payload=text[len("FINISH(") : -1]
        )

    return Action(kind=ActionKind.INVALID, raw=text)


def parse_finish_list(payload: str | None) -> list | None:
    """Best-effort decode of the FINISH argument into a list.

    Upstream stores the raw trimmed string and leaves interpretation to each grader, so
    this is only for our own reporting -- never feed the decoded value to a grader.
    """
    if payload is None:
        return None
    try:
        value = json.loads(payload)
    except (json.JSONDecodeError, ValueError):
        return None
    return value if isinstance(value, list) else [value]


def canonical_action(action: Action) -> str:
    """A canonical string for an action, so two samples can be compared exactly.

    Structured tool calls are what make semantic entropy cheap in this environment: two
    generations mean the same thing iff they are the same call with the same arguments,
    so no natural-language-inference model is needed to decide equivalence (proposal
    §6.2). That only holds if incidental differences are normalised away first --
    query-parameter order, JSON key order, whitespace.

    Returns a string rather than a hash so that logs stay inspectable.
    """
    if action.kind is ActionKind.GET:
        return f"get {_canonical_url(action.url or '')}"
    if action.kind is ActionKind.POST:
        try:
            payload = json.loads(action.post_body or "")
        except (json.JSONDecodeError, ValueError):
            return "post <unparseable>"
        return "post " + json.dumps(payload, sort_keys=True, separators=(",", ":"))
    if action.kind is ActionKind.FINISH:
        parsed = parse_finish_list(action.finish_payload)
        if parsed is None:
            return f"finish {(action.finish_payload or '').strip()}"
        return "finish " + json.dumps(parsed, sort_keys=True, separators=(",", ":"))
    return "invalid"


def _canonical_url(url: str) -> str:
    base, _, query = url.partition("?")
    if not query:
        return base.rstrip("&")
    params = sorted(p for p in query.split("&") if p and p != "_format=json")
    return base + ("?" + "&".join(params) if params else "")
