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


def weak_effects(Xtr: np.ndarray, W: np.ndarray, Y: np.ndarray, Xq: np.ndarray):
    """Estimated effects of the context and the query rows from the same weak
    S-learner CausalPFN 0.1.4 fits before retrieval (GradientBoostingRegressor,
    100 trees, depth 6, min_samples_leaf n/100, seed 111, on [X, 1-T, T])."""
    from sklearn.ensemble import GradientBoostingRegressor

    W = np.asarray(W, dtype=float)
    feats = lambda X, t: np.concatenate([X, (1 - t)[:, None], t[:, None]], axis=1)
    m = GradientBoostingRegressor(n_estimators=100, max_depth=6,
                                  min_samples_leaf=max(1, int(len(Xtr) / 100)), random_state=111)
    m.fit(feats(Xtr, W), np.asarray(Y, dtype=float))
    eff = lambda X: m.predict(feats(X, np.ones(len(X)))) - m.predict(feats(X, np.zeros(len(X))))
    return eff(Xtr), eff(Xq)


def effect_strata(e_ctx: np.ndarray, W: np.ndarray, e_qry: np.ndarray, k: int, cap: int):
    """Query strata and their contexts chosen by estimated effect, as
    CausalPFN does: for each query the k context rows of each arm nearest in
    estimated effect; queries are taken in order of estimated effect and a
    stratum grows while the union of its neighbours stays within ``cap`` rows
    per arm. Yields (query_idx, context_idx)."""
    W = np.asarray(W)
    arms = {w: np.flatnonzero(W == w) for w in (1, 0)}
    order = np.argsort(e_qry)
    nbrs = {}
    for w, idx in arms.items():
        kk = min(k, len(idx))
        d = np.abs(e_ctx[idx][None, :] - e_qry[:, None])
        part = np.argpartition(d, kk - 1, axis=1)[:, :kk] if kk < len(idx) else np.tile(np.arange(len(idx)), (len(e_qry), 1))
        nbrs[w] = idx[part]
    start = 0
    while start < len(order):
        end, union = start, {w: set() for w in arms}
        while end < len(order):
            q = order[end]
            trial = {w: union[w] | set(nbrs[w][q].tolist()) for w in arms}
            if end > start and any(len(trial[w]) > cap for w in arms):
                break
            union, end = trial, end + 1
        ctx = np.sort(np.concatenate([np.fromiter(union[w], int) for w in arms]))
        yield order[start:end], ctx
        start = end


class PFNEstimator:
    def __init__(self, ckpt_path: str, device: Optional[str] = None,
                 standardize: Optional[bool] = None, max_queries: int = 512,
                 context_cap: Optional[int] = None):
        import torch

        from ..train.train_pfn import build_model

        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
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
        if os.environ.get("PFN_CONTEXT_SELECT", "").strip() == "effect":
            # audit of 19 Oct, point 4: the context chosen like CausalPFN's,
            # by estimated effect, k rows per arm and query (PFN_CONTEXT_K,
            # default 1024), at most PFN_CONTEXT_ARM_CAP per arm (default 2048)
            return self._call_effect_strata(Xtr, W, Y, Xte)
        Xtr, W, Y = self._cap_context(Xtr, W, Y)
        rep = int(os.environ.get("PFN_CONTEXT_REPEAT", "1") or 1)
        if rep > 1:
            # audit of 19 Oct, point 1: sensitivity to duplicated context rows
            # (training contexts drawn with replacement repeat lesions)
            Xtr, W, Y = np.repeat(Xtr, rep, axis=0), np.repeat(np.asarray(W), rep), np.repeat(np.asarray(Y), rep)
        if self.standardize:
            mu, sd = Xtr.mean(0, keepdims=True), Xtr.std(0, keepdims=True) + 1e-6
            Xtr, Xte = (Xtr - mu) / sd, (Xte - mu) / sd
        # rotation ensembling at prediction time (27 Oct, diagnostic 1b): the
        # model was trained under random latent rotations and predicts from
        # one; PFN_ROTATIONS=K averages K random orthogonal rotations (plus the
        # identity), PFN_ARM_FLIP=1 also averages with the arms swapped
        n_rot = int(os.environ.get("PFN_ROTATIONS", "0") or 0)
        flip = os.environ.get("PFN_ARM_FLIP", "").strip() in ("1", "true", "yes")
        if n_rot > 0 or flip:
            rng = np.random.default_rng(int(os.environ.get("PFN_ROTATIONS_SEED", "0") or 0))
            mats = [np.eye(Xtr.shape[1], dtype=np.float32)]
            for _ in range(n_rot):
                q, r = np.linalg.qr(rng.normal(size=(Xtr.shape[1], Xtr.shape[1])))
                mats.append((q * np.sign(np.diag(r))[None, :]).astype(np.float32))
            W = np.asarray(W, dtype=np.float32)
            acc1, acc0, n = 0.0, 0.0, 0
            for R in mats:
                for swap in ((False, True) if flip else (False,)):
                    Wv = 1.0 - W if swap else W
                    a, b = self._predict(Xtr @ R, Wv, Y, Xte @ R)
                    if swap:
                        a, b = b, a
                    acc1, acc0, n = acc1 + a, acc0 + b, n + 1
            return np.clip(acc1 / n, 0, 1), np.clip(acc0 / n, 0, 1)
        return self._predict(Xtr, W, Y, Xte)

    def _predict(self, Xtr, W, Y, Xte):
        """Potential outcomes of the queries from one (already standardised) context."""
        import torch

        from .inference import predict_cate

        Xc = torch.as_tensor(np.ascontiguousarray(Xtr, dtype=np.float32), device=self.device)[None]
        Tc = torch.as_tensor(np.asarray(W, dtype=np.float32), device=self.device)[None]
        Yc = torch.as_tensor(np.asarray(Y, dtype=np.float32), device=self.device)[None]
        p1, p0 = [], []
        for start in range(0, len(Xte), self.max_queries):      # queries never attend to each other
            Xq = torch.as_tensor(np.ascontiguousarray(Xte[start:start + self.max_queries], dtype=np.float32),
                                 device=self.device)[None]
            out = predict_cate(self.model, Xc, Tc, Yc, Xq)
            p1.append(out["mu1"][0].float().cpu().numpy()); p0.append(out["mu0"][0].float().cpu().numpy())
        return np.clip(np.concatenate(p1), 0, 1), np.clip(np.concatenate(p0), 0, 1)

    def _forward(self, Xtr, W, Y, Xte):
        """One context for a set of queries (standardisation as configured)."""
        import torch

        from .inference import predict_cate

        if self.standardize:
            mu, sd = Xtr.mean(0, keepdims=True), Xtr.std(0, keepdims=True) + 1e-6
            Xtr, Xte = (Xtr - mu) / sd, (Xte - mu) / sd
        Xc = torch.as_tensor(np.asarray(Xtr, np.float32), device=self.device)[None]
        Tc = torch.as_tensor(np.asarray(W, dtype=np.float32), device=self.device)[None]
        Yc = torch.as_tensor(np.asarray(Y, dtype=np.float32), device=self.device)[None]
        Xq = torch.as_tensor(np.asarray(Xte, np.float32), device=self.device)[None]
        out = predict_cate(self.model, Xc, Tc, Yc, Xq)
        return out["mu1"][0].float().cpu().numpy(), out["mu0"][0].float().cpu().numpy()

    def _call_effect_strata(self, Xtr, W, Y, Xte):
        k = int(os.environ.get("PFN_CONTEXT_K", "1024") or 1024)
        cap = int(os.environ.get("PFN_CONTEXT_ARM_CAP", "2048") or 2048)
        W = np.asarray(W); Y = np.asarray(Y)
        e_ctx, e_qry = weak_effects(Xtr, W, Y, Xte)
        p1, p0 = np.empty(len(Xte)), np.empty(len(Xte))
        self.n_strata, rows = 0, 0
        for q_idx, c_idx in effect_strata(e_ctx, W, e_qry, k, cap):
            for start in range(0, len(q_idx), self.max_queries):
                qs = q_idx[start:start + self.max_queries]
                a, b = self._forward(Xtr[c_idx], W[c_idx], Y[c_idx], Xte[qs])
                p1[qs], p0[qs] = a, b
            self.n_strata += 1
            rows += len(c_idx) * len(q_idx)
        # with arms no larger than cap a single stratum holds every query and
        # its context is close to the whole slice: the log says so once
        frac = rows / max(1, len(Xtr) * len(Xte))
        self.effect_context_log = getattr(self, "effect_context_log", [])
        self.effect_context_log.append((self.n_strata, frac))
        if len(self.effect_context_log) == 1:
            print(f"pfn effect-context: k={k} cap={cap}: first slice {len(Xtr)} rows, {self.n_strata} strata, "
                  f"context {100 * frac:.0f}% of the slice per query", flush=True)
        return np.clip(p1, 0, 1), np.clip(p0, 0, 1)


def leaf_safe_weak_learner(train):
    """Wrap CausalPFN's ``_train_weak_learner`` so that its stratifier gets
    ``min_samples_leaf >= 1``. The package sets ``int(n / 100)``, which is 0
    (and rejected by sklearn) when the context has fewer than 100 rows, as in
    the slices of a small external cohort. The stratifier only picks
    neighbours when the context exceeds ``max_context_length``, so below 100
    rows it never shapes a prediction; from 100 rows up nothing changes."""
    import causalpfn.causal_estimator as ce

    def _train(X, t, y):
        original = ce.GradientBoostingRegressor

        def _regressor(**kw):
            leaf = kw.get("min_samples_leaf")
            if isinstance(leaf, int) and leaf < 1:
                kw["min_samples_leaf"] = 1
            return original(**kw)

        ce.GradientBoostingRegressor = _regressor
        try:
            return train(X, t, y)
        finally:
            ce.GradientBoostingRegressor = original

    return _train


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
        self.est._train_weak_learner = leaf_safe_weak_learner(self.est._train_weak_learner)
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
