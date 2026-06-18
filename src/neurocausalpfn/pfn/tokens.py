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

import torch


def to_tensors(batch_np: Dict[str, "object"], device: str = "cpu",
               dtype: torch.dtype = torch.float32) -> Dict[str, torch.Tensor]:
    return {k: torch.as_tensor(v, dtype=dtype, device=device) for k, v in batch_np.items()}
