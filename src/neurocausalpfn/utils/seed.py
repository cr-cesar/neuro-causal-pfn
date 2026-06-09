"""Fijacion determinista de semillas en random, numpy y torch."""
import os
import random

import numpy as np


def set_seed(seed: int = 0) -> None:
    """Fija la semilla en las librerias relevantes. torch se importa de forma
    perezosa para que el resto del paquete pueda usarse sin torch instalado."""
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
