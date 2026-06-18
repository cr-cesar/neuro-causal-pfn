"""Deterministic seeding for random, numpy and torch."""
import os
import random

import numpy as np


def set_seed(seed: int = 0) -> None:
    """Sets the seed in the relevant libraries. torch is imported lazily so the
    rest of the package can be used without torch installed."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass
