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

The context is capped at the length the checkpoint was trained with
(``cfg["pfn"]["context_max"]``): a longer train slice is subsampled at random
(fixed seed, treatment proportion preserved), so every checkpoint is scored in
the context regime it saw, and the cost of a forward pass on CPU, which grows
faster than the context length, stays bounded (a 3,252-row slice costs about
six times a 1,024-row one). ``context_cap`` or the environment variable
``PFN_CONTEXT_CAP`` override it; 0 disables the cap (the extrapolation
reading of the E12 ``ctx`` variant).
"""
import os
from typing import Dict, Optional, Tuple

import numpy as np


def cap_context(Xtr: np.ndarray, W: np.ndarray, Y: np.ndarray, cap: int):
    """Subsample a context longer than ``cap`` at random, keeping the treated
    share; deterministic for a given slice (seeded by its size and label
    counts), so two estimators capped at the same length see the same rows.
    ``cap`` <= 0 returns the slice unchanged."""
    n = len(Xtr)
    if cap <= 0 or n <= cap:
        return Xtr, W, Y
    W = np.asarray(W); Y = np.asarray(Y)
    rng = np.random.default_rng([n, int(W.sum()), int(Y.sum()), int(cap)])
    treated = np.flatnonzero(W == 1); control = np.flatnonzero(W != 1)
    k1 = int(round(cap * len(treated) / n))
    k1 = min(max(k1, 1 if len(treated) else 0), len(treated), cap)
    k0 = min(cap - k1, len(control))
    keep = np.sort(np.concatenate([rng.choice(treated, k1, replace=False),
                                   rng.choice(control, k0, replace=False)]))
    return Xtr[keep], W[keep], Y[keep]


class PFNEstimator:
    def __init__(self, ckpt_path: str, device: Optional[str] = None,
                 standardize: Optional[bool] = None, max_queries: int = 512,
                 context_cap: Optional[int] = None):
        import torch

        from ..train.train_pfn import build_model

        ckpt = torch.load(ckpt_path, map_location="cpu")
        self.cfg: Dict = ckpt["cfg"]
        state = ckpt["state_dict"]
        # d_x is recovered from the saved weights: the context embedding of the
        # linear model (d_x + 2 inputs) or the column table of the TabICL one
        if "ctx_emb.weight" in state:
            self.d_x = int(state["ctx_emb.weight"].shape[1]) - 2
        else:
            self.d_x = int(state["col_embed"].shape[0]) - 2
        self.model = build_model(self.cfg, self.d_x)
        self.model.load_state_dict(state)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device).eval()
        # a prior trained with coordinate-free augmentation standardised its
        # pool, so the context is standardised the same way unless overridden
        if standardize is None:
            standardize = bool(self.cfg.get("prior", {}).get("augment", False))
        self.standardize = bool(standardize)
        self.max_queries = int(max_queries)
        if context_cap is None:
            env = os.environ.get("PFN_CONTEXT_CAP", "").strip()
            context_cap = int(env) if env else int(self.cfg.get("pfn", {}).get("context_max", 0) or 0)
        self.context_cap = int(context_cap)          # 0: the whole train slice is the context
        self.n_capped = 0                            # calls whose context was subsampled
        self.name = "pfn"

    def _cap_context(self, Xtr: np.ndarray, W: np.ndarray, Y: np.ndarray):
        Xtr2, W2, Y2 = cap_context(Xtr, W, Y, self.context_cap)
        self.n_capped += len(Xtr2) < len(Xtr)
        return Xtr2, W2, Y2

    def __call__(self, Xtr: np.ndarray, W: np.ndarray, Y: np.ndarray,
                 Xte: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        import torch

        from .inference import predict_cate

        Xtr = np.asarray(Xtr, dtype=np.float32); Xte = np.asarray(Xte, dtype=np.float32)
        if Xtr.shape[1] != self.d_x:
            raise ValueError(f"checkpoint expects d_x={self.d_x}, got {Xtr.shape[1]}")
        Xtr, W, Y = self._cap_context(Xtr, W, Y)
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


class CausalPFNEstimator:
    """The off-the-shelf CausalPFN of Balazadeh et al. (2025) with fixed weights:
    the Tier-4 evaluator of the design (encoder selection under a fixed
    estimator). Same ``fn(Xtr, W, Y, Xte) -> (p1, p0)`` signature, so it is
    scored inside the replica exactly like the sklearn classifiers and our own
    transformer. Needs ``pip install causalpfn faiss-cpu`` and the pretrained
    weights in the cache (download once on a node with internet:
    ``python -c "from causalpfn import CATEEstimator; CATEEstimator('cpu').load_model()"``).
    Nothing is fine-tuned; ``fit`` only stores the context and trains the
    package's own stratification model for long contexts."""

    def __init__(self, device: Optional[str] = None, model_path: str = "vdblm/causalpfn",
                 cache_dir: Optional[str] = None, max_context_length: int = 4096,
                 num_neighbours: int = 1024, context_cap: Optional[int] = None):
        import torch
        from causalpfn import CATEEstimator

        # H3 of the Phase-2 audit: CausalPFN retrieves up to 2,048 nearest rows
        # per arm while our transformer reads 1,024 random rows. With a cap
        # (``context_cap`` or CAUSALPFN_CONTEXT_CAP) its context is subsampled
        # by the same rule as PFNEstimator's, so both see the same rows.
        if context_cap is None:
            env = os.environ.get("CAUSALPFN_CONTEXT_CAP", "").strip()
            context_cap = int(env) if env else 0
        self.context_cap = int(context_cap)
        self.n_capped = 0

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.est = CATEEstimator(device=self.device, model_path=model_path, cache_dir=cache_dir,
                                 max_context_length=max_context_length, max_query_length=max_context_length,
                                 num_neighbours=num_neighbours, calibrate=False, verbose=False)
        self._k = int(num_neighbours)
        self.name = "causalpfn"
        # F08 of the audit: CausalPFN compresses any representation wider than
        # its max_feature_size with a TruncatedSVD (uncentred) fitted on each
        # context, while narrower ones enter raw. Every call records it and the
        # first call of each input width prints one line to the scoring log.
        self.last_info: dict = {}
        self._reported_widths: set = set()
        # audit of 15 Oct: the repository pins one version and the neighbour
        # search changed between releases, so every scoring states which ran
        try:
            from importlib.metadata import version

            self.version = version("causalpfn")
        except Exception:                       # not installed as a distribution
            self.version = "unknown"

    def __call__(self, Xtr: np.ndarray, W: np.ndarray, Y: np.ndarray,
                 Xte: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        Xtr = np.asarray(Xtr, dtype=np.float32); Xte = np.asarray(Xte, dtype=np.float32)
        W = np.asarray(W, dtype=np.float32); Y = np.asarray(Y, dtype=np.float32)
        n_in = len(Xtr)
        Xtr, W, Y = cap_context(Xtr, W, Y, self.context_cap)
        self.n_capped += len(Xtr) < n_in
        # the neighbour search needs k <= smallest treatment group of the context
        self.est.num_neighbours = max(1, min(self._k, int((W == 1).sum()), int((W == 0).sum())))
        self.est.fit(Xtr, W, Y)
        mfs = self.est.max_feature_size
        self.last_info = {"d_in": int(Xtr.shape[1]), "max_feature_size": mfs,
                          "truncated_svd": bool(mfs is not None and Xtr.shape[1] > mfs),
                          "version": self.version, "context_cap": self.context_cap,
                          "n_context": int(len(Xtr))}
        if self.last_info["d_in"] not in self._reported_widths:
            self._reported_widths.add(self.last_info["d_in"])
            print(f"causalpfn {self.version}: d_in={self.last_info['d_in']} max_feature_size={mfs} "
                  f"truncated_svd={self.last_info['truncated_svd']} context_cap={self.context_cap}", flush=True)
        n = len(Xte)
        Xq = np.concatenate([Xte, Xte], axis=0)
        if self.est.max_feature_size is not None and Xq.shape[1] > self.est.max_feature_size:
            Xq = self.est.x_dim_transformer.transform(Xq)
        tq = np.concatenate([np.zeros(n, dtype=np.float32), np.ones(n, dtype=np.float32)])
        mu = np.asarray(self.est._predict_cepo(self.est.X_train, self.est.t_train, self.est.y_train,
                                               Xq, tq, temperature=self.est.temperature), dtype=float)
        return np.clip(mu[n:], 0, 1), np.clip(mu[:n], 0, 1)
