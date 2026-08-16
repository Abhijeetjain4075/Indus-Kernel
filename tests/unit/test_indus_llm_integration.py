from pathlib import Path

import torch

from ik_indus_llm.config import IndusConfig
from ik_indus_llm.model import Indus


def test_indus_llm_baseline_checkpoint_shape():
    ckpt = Path("packages/ik_indus_llm/ik_indus_llm/artifacts/checkpoints/pretrain/indus_tiny_v0.3.0.pt")
    payload = torch.load(ckpt, map_location="cpu", weights_only=True)
    cfg = IndusConfig(**payload["cfg"])
    model = Indus(cfg)
    model.load_state_dict(payload["model"], strict=True)
    assert model.num_params() == 1_120_392
