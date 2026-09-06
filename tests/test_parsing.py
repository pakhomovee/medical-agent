"""Parser parity with upstream.

These pin behaviours that look like bugs and are deliberately preserved, because gate G4
compares per-task outcomes against the official harness and a silent "fix" registers as
a parity failure.
"""

from __future__ import annotations

import json

from uqma.agent.parsing import ActionKind, clean, parse, parse_finish_list


def test_get_appends_format_json():
    action = parse("GET http://localhost:8080/fhir/Patient?name=Smith")
    assert action.kind is ActionKind.GET
    assert action.url == "http://localhost:8080/fhir/Patient?name=Smith&_format=json"


def test_get_appends_stray_ampersand_when_no_query_string():
    # Upstream quirk: '&_format=json' is appended unconditionally. Preserved on purpose.
    action = parse("GET http://localhost:8080/fhir/Patient")
    assert action.url == "http://localhost:8080/fhir/Patient&_format=json"


def test_post_body_is_everything_after_the_first_line():
    payload = {"resourceType": "Observation", "status": "final"}
    action = parse(f"POST http://x/fhir/Observation\n{json.dumps(payload)}")
    assert action.kind is ActionKind.POST
    assert json.loads(action.post_body) == payload


def test_post_body_survives_multiline_json():
    body = '{\n  "resourceType": "ServiceRequest",\n  "status": "active"\n}'
    action = parse(f"POST http://x/fhir/ServiceRequest\n{body}")
    assert json.loads(action.post_body)["resourceType"] == "ServiceRequest"


def test_finish_trims_to_list_payload():
    action = parse('FINISH([1.5, "2023-01-01"])')
    assert action.kind is ActionKind.FINISH
    assert action.finish_payload == '[1.5, "2023-01-01"]'
    assert parse_finish_list(action.finish_payload) == [1.5, "2023-01-01"]


def test_unrecognised_prefix_is_invalid():
    action = parse("I will now look up the patient.")
    assert action.kind is ActionKind.INVALID
    assert action.is_terminal


def test_code_fences_are_stripped():
    assert clean("```tool_code\nGET http://x?a=b\n```") == "GET http://x?a=b"
    assert parse("```tool_code\nGET http://x?a=b\n```").kind is ActionKind.GET


def test_leading_whitespace_does_not_break_dispatch():
    assert parse("   \n GET http://x?a=b  ").kind is ActionKind.GET


def test_finish_is_terminal_and_get_is_not():
    assert parse("FINISH([])").is_terminal
    assert not parse("GET http://x?a=b").is_terminal
    assert not parse("POST http://x\n{}").is_terminal


def test_parse_finish_list_returns_none_for_unparseable():
    assert parse_finish_list("not json at all") is None
    assert parse_finish_list(None) is None


# --- reasoning models -----------------------------------------------------------------
# Qwen3 emits <think> blocks by default, so a response begins '<think>' and the prefix
# dispatch classifies every turn INVALID -- action SR would read 0% for a pure format
# reason. Stripping is a SCAFFOLD CHANGE (plan §8), hence opt-in and off by default.

THINKING = '<think>\nThe user wants a magnesium level. I should query Observation.\n</think>\nGET http://x/fhir/Observation?code=MG'


def test_reasoning_block_makes_the_turn_invalid_by_default():
    assert parse(THINKING).kind is ActionKind.INVALID


def test_stripping_recovers_the_action():
    action = parse(THINKING, strip_think=True)
    assert action.kind is ActionKind.GET
    assert action.url == "http://x/fhir/Observation?code=MG&_format=json"


def test_stripping_handles_a_post_with_its_payload_intact():
    text = '<think>dose is 10</think>\nPOST http://x/fhir/MedicationRequest\n{"resourceType": "MedicationRequest"}'
    action = parse(text, strip_think=True)
    assert action.kind is ActionKind.POST
    assert json.loads(action.post_body)["resourceType"] == "MedicationRequest"


def test_thinking_tag_variant_is_handled():
    assert parse("<thinking>hm</thinking>\nFINISH([1])", strip_think=True).kind is ActionKind.FINISH


def test_multiple_blocks_are_all_removed():
    text = "<think>a</think>\n<think>b</think>\nFINISH([2])"
    assert parse(text, strip_think=True).finish_payload == "[2]"


def test_unclosed_block_stays_invalid():
    """Truncated mid-thought (hit max_tokens) is a failed turn, not a recoverable one."""
    assert parse("<think>still reasoning and ran out of tok", strip_think=True).kind is ActionKind.INVALID


def test_stripping_is_a_noop_on_a_plain_action():
    assert parse("GET http://x?a=1", strip_think=True).url == parse("GET http://x?a=1").url


def test_stripping_does_not_touch_think_inside_a_payload():
    # A literal '<think>' in free text must survive: only well-formed blocks are removed.
    payload = {"resourceType": "ServiceRequest", "note": {"text": "patient did not think clearly"}}
    text = f'POST http://x/fhir/ServiceRequest\n{json.dumps(payload)}'
    action = parse(text, strip_think=True)
    assert json.loads(action.post_body)["note"]["text"] == "patient did not think clearly"
