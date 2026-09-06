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
import re
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


# Reasoning models (Qwen3, DeepSeek-R1 and friends) wrap chain-of-thought in these.
# Upstream never met one, so it has no handling -- see strip_reasoning below.
_THINK = re.compile(r"<(think|thinking)>.*?</\1>", re.DOTALL | re.IGNORECASE)
_UNCLOSED_THINK = re.compile(r"^\s*<(think|thinking)>.*", re.DOTALL | re.IGNORECASE)


def strip_reasoning(content: str) -> str:
    """Remove <think> blocks before dispatching on the action prefix.

    Qwen3 emits reasoning by default, so a response begins ``<think>...`` and the
    prefix dispatch below would classify every turn as INVALID -- action success would
    read 0% for a pure format reason, which is exactly the format-versus-clinical
    confound of plan §4.4.

    This is a SCAFFOLD CHANGE, not a parity fix: upstream has no such handling because
    it predates reasoning models. It is therefore off by default and must be switched on
    explicitly as a named tier (plan §8 open decision 2), so that any effect on success
    rate is attributable rather than silent.

    An unclosed block (the model hit max_tokens mid-thought) leaves nothing to dispatch
    on, which correctly parses as INVALID -- a truncated turn is a failed turn.
    """
    without_blocks = _THINK.sub("", content).strip()
    if without_blocks:
        return without_blocks
    return "" if _UNCLOSED_THINK.match(content) else content.strip()


def clean(content: str, strip_think: bool = False) -> str:
    """Upstream's normalisation, including the Gemini-2.0-Flash fence stripping."""
    text = strip_reasoning(content) if strip_think else content
    return text.strip().replace("```tool_code", "").replace("```", "").strip()


def parse(content: str, strip_think: bool = False) -> Action:
    text = clean(content, strip_think=strip_think)

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
