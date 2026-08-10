"""Real tests for the local Indus LLM baseline checkpoint.

Verifies that the checkpoint loads, has the expected shape, and produces
real (non-empty) outputs.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from ik_indus_llm.config import IndusConfig
from ik_indus_llm.model import Indus
from ik_indus_llm.runtime import IndusLLMRuntime


CHECKPOINT = Path(
    "packages/ik_indus_llm/ik_indus_llm/artifacts/checkpoints/pretrain/indus_tiny_v0.3.0.pt"
)


@pytest.mark.skipif(not CHECKPOINT.exists(), reason="checkpoint not present")
def test_checkpoint_loads():
    payload = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)
    assert "cfg" in payload
    assert "model" in payload


@pytest.mark.skipif(not CHECKPOINT.exists(), reason="checkpoint not present")
def test_baseline_param_count():
    payload = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)
    cfg = IndusConfig(**payload["cfg"])
    model = Indus(cfg)
    model.load_state_dict(payload["model"], strict=True)
    assert model.num_params() == 1_120_392


@pytest.mark.skipif(not CHECKPOINT.exists(), reason="checkpoint not present")
def test_runtime_generate():
    rt = IndusLLMRuntime(CHECKPOINT, device="cpu")
    out = rt.generate("Hello", max_new_tokens=10, temperature=0.5)
    assert isinstance(out, str)
    assert len(out) > 0


@pytest.mark.skipif(not CHECKPOINT.exists(), reason="checkpoint not present")
def test_tokenizer_vocab_matches_config():
    rt = IndusLLMRuntime(CHECKPOINT, device="cpu")
    assert rt.tokenizer.vocab_size == rt.model.cfg.vocab_size
