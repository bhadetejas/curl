"""Loading saved pointer-policy checkpoints."""
from __future__ import annotations

from pathlib import Path

import torch

from curl.models.pointer import PointerPolicy


def load_policy(path: Path, device: torch.device | None = None) -> PointerPolicy:
    """Load a policy in eval mode. The default device is the CPU."""
    device = device or torch.device("cpu")
    policy = PointerPolicy().to(device)
    policy.load_state_dict(torch.load(path, map_location=device))
    policy.eval()
    return policy
