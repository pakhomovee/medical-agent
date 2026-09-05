"""Schema round-tripping and run-artifact handling.

Round-trip fidelity is cheap to test and expensive to lack: a log-format bug silently
corrupts a sweep that cost hundreds of GPU-hours, and it would not surface until stage 4
(plan §6.4).
"""

from __future__ import annotations

import json

import pytest

from uqma.trajectory.schema import (
    SCHEMA_VERSION,
    HiddenStateRef,
    PrefixSource,
    Sample,
    Trajectory,
    Turn,
)
from uqma.trajectory.store import (
    Manifest,
    RunWriter,
    check_compatible,
    config_hash,
    iter_trajectories,
    read_manifest,
)


def _trajectory() -> Trajectory:
    return Trajectory(
        task_id="task5_3",
        category="task5",
        status="completed",
        result="[1.2]",
        correct=True,
        seed=7,
        run_id="r1",
        wall_s=1.25,
        history=[{"role": "user", "content": "prompt"}],
        turns=[
            Turn(
                index=0,
                generation="GET http://x/fhir/Observation?code=MG",
                action_kind="get",
                action_raw="GET http://x/fhir/Observation?code=MG",
                observation="Here is the response...",
                url="http://x/fhir/Observation?code=MG&_format=json",
                token_logprobs=[-0.1, -0.9],
                n_generated_tokens=2,
                samples=[
                    Sample(
                        index=0,
                        text="GET http://x/fhir/Observation?code=MG",
                        action_kind="get",
                        action_raw="GET http://x/fhir/Observation?code=MG",
                        token_logprobs=[-0.2],
                        canonical_action="get http://x/fhir/Observation?code=MG",
                    )
                ],
                hidden_states=HiddenStateRef(path="h.safetensors", layer_indices=[8, 16]),
            ),
            Turn(
                index=1,
                generation="POST http://x/fhir/Observation\n{}",
                action_kind="post",
                action_raw="POST http://x/fhir/Observation\n{}",
                post_check={"json_valid": True, "schema_valid": True},
                correct=False,
                error_class="clinical",
            ),
        ],
    )


def test_round_trip_is_exact():
    original = _trajectory()
    assert Trajectory.from_dict(original.to_dict()) == original


def test_round_trip_survives_json_encoding():
    original = _trajectory()
    revived = Trajectory.from_dict(json.loads(json.dumps(original.to_dict())))
    assert revived == original


def test_nested_types_are_reconstructed_not_left_as_dicts():
    revived = Trajectory.from_dict(_trajectory().to_dict())
    assert isinstance(revived.turns[0], Turn)
    assert isinstance(revived.turns[0].samples[0], Sample)
    assert isinstance(revived.turns[0].hidden_states, HiddenStateRef)


def test_unknown_fields_are_rejected_loudly():
    payload = _trajectory().to_dict()
    payload["invented_field"] = 1
    with pytest.raises(ValueError, match="unknown field"):
        Trajectory.from_dict(payload)


def test_commitment_turns_are_post_turns():
    trajectory = _trajectory()
    assert [t.index for t in trajectory.commitments] == [1]
    assert trajectory.n_post_attempts == 1
    assert trajectory.n_schema_valid_posts == 1


def test_defaults_are_stable():
    turn = Turn(index=0, generation="", action_kind="get", action_raw="")
    assert turn.prefix_source == PrefixSource.ROLLOUT.value
    assert turn.samples == []
    assert turn.gated is False
    assert turn.correct is None


def test_optional_none_fields_are_omitted_but_restored():
    payload = _trajectory().to_dict()
    assert "hidden_states" not in payload["turns"][1]  # omitted to keep logs small
    assert Trajectory.from_dict(payload).turns[1].hidden_states is None


def test_writer_creates_manifest_and_streams(tmp_path):
    manifest = Manifest.create({"model": "qwen3-8b", "seed": 0}, notes="smoke")
    with RunWriter(tmp_path, manifest) as writer:
        writer.write(_trajectory())
        writer.write(_trajectory())
        run_dir = writer.dir

    stored = read_manifest(run_dir)
    assert stored["schema_version"] == SCHEMA_VERSION
    assert stored["config"]["model"] == "qwen3-8b"
    assert stored["notes"] == "smoke"
    assert "python" in stored["libraries"]

    revived = list(iter_trajectories(run_dir))
    assert len(revived) == 2
    assert revived[0] == _trajectory()


def test_run_id_embeds_model_and_config_hash(tmp_path):
    manifest = Manifest.create({"model": "Qwen/Qwen3-8B", "k": 10})
    assert manifest.config_hash in manifest.run_id
    assert "Qwen-Qwen3-8B" in manifest.run_id


def test_config_hash_is_order_independent():
    assert config_hash({"a": 1, "b": 2}) == config_hash({"b": 2, "a": 1})


def test_config_hash_changes_with_values():
    assert config_hash({"a": 1}) != config_hash({"a": 2})


def test_runs_are_immutable_by_default(tmp_path):
    manifest = Manifest.create({"model": "m"})
    with RunWriter(tmp_path, manifest) as writer:
        writer.write(_trajectory())
        run_dir = writer.dir

    with pytest.raises(FileExistsError, match="immutable"):
        RunWriter(run_dir, manifest)

    RunWriter(run_dir, manifest, overwrite=True)  # deliberate override is allowed


def test_partial_run_is_readable_after_a_crash(tmp_path):
    """A sweep that dies halfway must leave the completed half loadable."""
    manifest = Manifest.create({"model": "m"})
    writer = RunWriter(tmp_path, manifest)
    writer.write(_trajectory())
    # deliberately not closed, simulating a hard crash
    assert len(list(iter_trajectories(writer.dir))) == 1


def test_schema_mismatch_is_refused(tmp_path):
    manifest = Manifest.create({"model": "m"})
    manifest.schema_version = "0.0.1"
    with RunWriter(tmp_path, manifest) as writer:
        writer.write(_trajectory())
        run_dir = writer.dir
    with pytest.raises(ValueError, match="migration"):
        check_compatible(run_dir)


def test_corrupt_line_names_the_file_and_line(tmp_path):
    manifest = Manifest.create({"model": "m"})
    with RunWriter(tmp_path, manifest) as writer:
        writer.write(_trajectory())
        run_dir = writer.dir
    path = run_dir / "trajectories.jsonl"
    path.write_text(path.read_text(encoding="utf-8") + '{"task_id": "x", "bogus": 1}\n',
                    encoding="utf-8")
    with pytest.raises(ValueError, match=r"trajectories\.jsonl:2"):
        list(iter_trajectories(run_dir))


def test_prompt_for_turn_reconstructs_what_the_model_saw():
    """The per-turn prompt is a slice of history, not a stored copy (O(n^2) otherwise)."""
    from uqma.agent.loop import run_episode
    from uqma.envs.medagentbench.fhir import FhirClient, GetResult
    from uqma.envs.medagentbench.tasks import Task
    from uqma.inference.stub import ScriptedBackend

    class FakeFhir(FhirClient):
        def __init__(self):
            super().__init__("http://fake/fhir")

        def get(self, url):
            return GetResult(ok=True, data={"total": 1})

    seen: list[list[dict]] = []

    class Recording(ScriptedBackend):
        def generate(self, messages, capture_logprobs=True, **kwargs):
            seen.append([dict(m) for m in messages])
            return super().generate(messages, capture_logprobs, **kwargs)

    task = Task(id="task4_1", instruction="x", context="", sol=None, eval_mrn="S1")
    trajectory = run_episode(
        task,
        Recording(["GET http://fake/fhir/o?a=1", "GET http://fake/fhir/o?a=2", "FINISH([])"]),
        FakeFhir(),
        [{"name": "f"}],
    )

    assert trajectory.n_turns == 3
    for i in range(trajectory.n_turns):
        assert trajectory.prompt_for_turn(i) == seen[i]


def test_prompt_for_turn_rejects_out_of_range():
    with pytest.raises(IndexError):
        _trajectory().prompt_for_turn(5)
