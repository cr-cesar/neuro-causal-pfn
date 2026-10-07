"""Batch assembly for the transformer.

Converts the numpy batch produced by the cohort into torch tensors. Each context
patient carries its covariates, its treatment and its outcome; each query patient
carries only its covariates and the treatment of interest. The concrete
construction of the tokens (the concatenation and the linear projection) is done
by the model; here only the tensors are prepared.

In full mode, the linear projections of this skeleton would be replaced by the
column-then-row tabular encoding described in the plan.
"""
from typing import Dict

import numpy as np
import torch

# keys of the cohort batch that are per-process metadata, not model inputs
METADATA_KEYS = ("processes",)


def is_tensor_field(key: str, value) -> bool:
    """True for the numeric arrays of a batch; False for the metadata the real
    cohorts attach (``processes``: one dict per item describing its process)."""
    if key in METADATA_KEYS:
        return False
    if isinstance(value, (list, tuple)):
        return len(value) > 0 and not isinstance(value[0], (dict, str))
    return not isinstance(value, (dict, str))


def to_tensors(batch_np: Dict[str, "object"], device: str = "cpu",
               dtype: torch.dtype = torch.float32) -> Dict[str, torch.Tensor]:
    return {k: torch.as_tensor(np.asarray(v), dtype=dtype, device=device)
            for k, v in batch_np.items() if is_tensor_field(k, v)}
