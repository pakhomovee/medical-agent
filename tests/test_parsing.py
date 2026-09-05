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
