"""The trained transformer as an in-context estimator for the Phase 1 replica.

``PFNEstimator`` loads a checkpoint written by ``train_pfn.run_pfn`` and
exposes the ``fn(Xtr, W, Y, Xte) -> (p1, p0)`` signature of
``giles_replica.register_estimator``: the whole train slice of a deficit and
fold is the context (latents, treatment, outcome) and the test slice is the
query set, evaluated once under treatment and once under control. No
parameter is fitted at scoring time, so the transformer is scored with
exactly the same folds, simulated trials and PEHE scale as the logistic
regression and extra trees of the replica.

Covariates are passed as the checkpoint saw them during prior training. With
``standardize=True`` both context and queries are standardised with the
context statistics (useful when the latents of the scored cohort are not the
pool the prior was built on).
"""
from typing import Dict, Optional, Tuple

import numpy as np


class PFNEstimator:
    def __init__(self, ckpt_path: str, device: Optional[str] = None,
                 standardize: bool = False, max_queries: int = 512):
        import torch

        from ..train.train_pfn import build_model

        ckpt = torch.load(ckpt_path, map_location="cpu")
        self.cfg: Dict = ckpt["cfg"]
        state = ckpt["state_dict"]
        # d_x is recovered from the context embedding of the saved weights
        w = state.get("ctx_emb.weight")
        if w is None:                                   # TabICL variant
            w = next(v for k, v in state.items() if k.endswith("ctx_emb.weight"))
        self.d_x = int(w.shape[1]) - 2
        self.model = build_model(self.cfg, self.d_x)
        self.model.load_state_dict(state)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device).eval()
        self.standardize = bool(standardize)
        self.max_queries = int(max_queries)
        self.name = "pfn"

    def __call__(self, Xtr: np.ndarray, W: np.ndarray, Y: np.ndarray,
                 Xte: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        import torch

        from .inference import predict_cate

        Xtr = np.asarray(Xtr, dtype=np.float32); Xte = np.asarray(Xte, dtype=np.float32)
        if Xtr.shape[1] != self.d_x:
            raise ValueError(f"checkpoint expects d_x={self.d_x}, got {Xtr.shape[1]}")
        if self.standardize:
            mu, sd = Xtr.mean(0, keepdims=True), Xtr.std(0, keepdims=True) + 1e-6
            Xtr, Xte = (Xtr - mu) / sd, (Xte - mu) / sd
        Xc = torch.as_tensor(Xtr, device=self.device)[None]
        Tc = torch.as_tensor(np.asarray(W, dtype=np.float32), device=self.device)[None]
        Yc = torch.as_tensor(np.asarray(Y, dtype=np.float32), device=self.device)[None]
        p1, p0 = [], []
        for start in range(0, len(Xte), self.max_queries):      # queries never attend to each other
            Xq = torch.as_tensor(Xte[start:start + self.max_queries], device=self.device)[None]
            out = predict_cate(self.model, Xc, Tc, Yc, Xq)
            p1.append(out["mu1"][0].float().cpu().numpy()); p0.append(out["mu0"][0].float().cpu().numpy())
        return np.clip(np.concatenate(p1), 0, 1), np.clip(np.concatenate(p0), 0, 1)
