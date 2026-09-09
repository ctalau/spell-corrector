"""Process-wide RNG seeding."""

from __future__ import annotations

import os
import random

import numpy as np
import torch

DEFAULT_SEED = 1337


def seed_everything(seed: int = DEFAULT_SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
