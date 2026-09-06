"""GPU detection and serving recommendations.

The two traps this guards against, both silent otherwise:

* bfloat16 needs compute capability 8.0, and a Colab T4 is 7.5. vLLM may fall back rather
  than fail, and a silent dtype change is a change to the measurement instrument
  (proposal §6.6).
* An 8B model in 16-bit is ~16 GB and does not fit a 16 GB card once the KV cache is
  accounted for.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from gpu_profile import probe, recommend, render, serve_command  # noqa: E402


def gpu(name, memory_gb, cap):
    major = int(str(cap).split(".")[0])
    return {"name": name, "memory_gb": memory_gb, "compute_capability": str(cap),
            "supports_bf16": major >= 8}


T4 = gpu("Tesla T4", 15.0, "7.5")
L4 = gpu("NVIDIA L4", 23.0, "8.9")
A100 = gpu("NVIDIA A100-SXM4-40GB", 40.0, "8.0")
RTX5090 = gpu("NVIDIA GeForce RTX 5090", 31.4, "12.0")


# --- dtype ----------------------------------------------------------------------------

def test_t4_gets_float16_not_bfloat16():
    assert recommend(T4)["dtype"] == "float16"


def test_ampere_and_later_get_bfloat16():
    for card in (L4, A100, RTX5090):
        assert recommend(card)["dtype"] == "bfloat16", card["name"]


def test_float16_fallback_is_flagged_as_affecting_comparability():
    """A dtype change must be recorded, not absorbed -- logits are the instrument."""
    notes = " ".join(recommend(T4)["notes"])
    assert "float16" in notes
    assert "§6.6" in notes and "not directly comparable" in notes


def test_bf16_capable_card_raises_no_dtype_note():
    assert not any("float16" in note for note in recommend(A100)["notes"])


# --- model sizing ---------------------------------------------------------------------

def test_8b_does_not_fit_a_t4():
    assert recommend(T4)["model"] != "Qwen/Qwen3-8B"


def test_8b_fits_an_a100():
    assert recommend(A100)["model"] in ("Qwen/Qwen3-8B", "Qwen/Qwen3-14B")


def test_larger_card_never_recommends_a_smaller_model():
    sizes = {"Qwen/Qwen3-1.7B": 1, "Qwen/Qwen3-4B": 2, "Qwen/Qwen3-8B": 3,
             "Qwen/Qwen3-14B": 4, "Qwen/Qwen3-32B": 5}
    picks = [recommend(card)["model"] for card in (T4, L4, A100)]
    ranks = [sizes[p] for p in picks if p]
    assert ranks == sorted(ranks)


def test_downgrade_explains_that_g2_may_fail():
    notes = " ".join(recommend(T4)["notes"])
    assert "G2" in notes and "action success rate" in notes


def test_tiny_card_recommends_nothing_and_says_so():
    result = recommend(gpu("GTX 1050", 4.0, "6.1"))
    assert result["model"] is None
    assert any("No candidate fits" in note for note in result["notes"])


def test_budget_leaves_room_for_kv_cache():
    # 8B in 16-bit is ~16.4 GB of weights; a 23 GB L4 must not be told that is fine
    # without accounting for activations and cache.
    assert recommend(L4)["budget_gb"] < 23.0 * 0.9


# --- serve command --------------------------------------------------------------------

def test_serve_command_carries_the_flags_the_sweeps_need():
    command = serve_command("Qwen/Qwen3-8B", "bfloat16", 40.0)
    assert "--enable-prefix-caching" in command       # k samples share a long prompt
    assert "--no-enable-chunked-prefill" in command   # hidden-state extraction needs it
    assert "--dtype bfloat16" in command
    assert "--served-model-name Qwen3-8B" in command  # keeps run ids readable


def test_small_card_gets_a_shorter_context():
    assert "--max-model-len 4096" in serve_command("Qwen/Qwen3-4B", "float16", 15.0)
    assert "--max-model-len 8192" in serve_command("Qwen/Qwen3-8B", "bfloat16", 40.0)


# --- probing --------------------------------------------------------------------------

def test_probe_without_nvidia_smi_is_not_an_error(monkeypatch):
    import gpu_profile

    monkeypatch.setattr(gpu_profile.shutil, "which", lambda _: None)
    result = probe()
    assert result["available"] is False
    assert "nvidia-smi" in result["reason"]


def test_probe_parses_nvidia_smi_output(monkeypatch):
    import subprocess as sp

    import gpu_profile

    class Done:
        stdout = "Tesla T4, 15360, 7.5\n"

    monkeypatch.setattr(gpu_profile.shutil, "which", lambda _: "/usr/bin/nvidia-smi")
    monkeypatch.setattr(sp, "run", lambda *a, **k: Done())
    result = probe()
    assert result["gpus"][0]["name"] == "Tesla T4"
    assert result["gpus"][0]["supports_bf16"] is False
    assert result["gpus"][0]["memory_gb"] == pytest.approx(15.0, abs=0.1)


def test_probe_handles_multiple_gpus(monkeypatch):
    import subprocess as sp

    import gpu_profile

    class Done:
        stdout = "NVIDIA L4, 23034, 8.9\nNVIDIA L4, 23034, 8.9\n"

    monkeypatch.setattr(gpu_profile.shutil, "which", lambda _: "/usr/bin/nvidia-smi")
    monkeypatch.setattr(sp, "run", lambda *a, **k: Done())
    assert probe()["n_gpus"] == 2


def test_render_without_a_gpu_points_at_the_colab_setting():
    assert "Change runtime type" in render({"available": False, "reason": "x"})


def test_render_surfaces_the_bf16_warning():
    output = render({"available": True, "n_gpus": 1, "gpus": [T4]})
    assert "bfloat16: NO" in output
    assert "float16" in output
