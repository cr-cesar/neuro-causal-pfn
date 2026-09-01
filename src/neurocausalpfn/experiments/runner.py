"""The Table-9 orchestrator.

Turns each experiment (a row of Table 9) into concrete training runs, evaluates
them through the tiered harness with the stop/go gates, aggregates across the
>= 3 random seeds required by section 14, and propagates the "winner" of a
selection experiment to the ones that depend on it (E3's backbone becomes the
default; E4's dimensionality becomes the default; E6's channel choice; E2's
Dice weight).

It composes the existing training entry points; it does not duplicate them:

    entry            training call                       arms
    -------------    --------------------------------    --------------------
    vae              train_vae.run_vae (per modality)    A (E1,E2,E3,E5,E8), B
    e3_sweep         two VAEs per (d_les,d_dis) point    A (E4)
    early_fusion     train_vae.run_vae (2-channel)       A (E7a)
    dmvae            train_dmvae.run_dmvae               A (E7b)
    contrastive      train_contrastive.run_contrastive  C (E9a,E9b)
    mae              train_mae.run_mae                  D (E9c)
    dscm             train_dscm.run_dscm                E (E10a,E10b,E10c)
    curriculum       curriculum.run_curriculum_ablation CausalPFN (E12)
    audit            Arm-A winner + stratified Tier 4    All (E11a)

Command line::

    python -m neurocausalpfn.experiments.runner --experiment E3 --mode prototype
    python -m neurocausalpfn.experiments.runner --arm A --mode prototype --seeds 3
    python -m neurocausalpfn.experiments.runner --all --mode prototype
"""
from __future__ import annotations

import copy
import itertools
import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from ..utils.logging_utils import get_logger
from ..data.paths import set_tier
from . import artifacts as art
from .logging_backend import ExperimentLogger
from .registry import (ARM_A_ORDER, Experiment, arm_experiments,
                       dependency_order, get_experiment)
from .tiers import run_tiers

log = get_logger()


# --------------------------------------------------------------------------- #
# Mode-aware dimensionality (the registry uses full-mode dims; prototype shrinks)
# --------------------------------------------------------------------------- #
def scale_dim(mode: str, d: int) -> int:
    if mode == "full":
        return int(d)
    return max(2, int(round(d * 16.0 / 100.0)))   # {25,50,75,100} -> {4,8,12,16}


@dataclass
class RunSpec:
    label: str
    kind: str
    meta: Dict = field(default_factory=dict)


@dataclass
class RunResult:
    eid: str
    label: str
    seed: int
    summary: Dict
    report: object = None            # tiers.TierReport
    meta: Dict = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# Builders: Experiment (+ resolved context) -> list of RunSpec
# --------------------------------------------------------------------------- #
def _grid_points(grid: Dict) -> List[Dict]:
    if not grid:
        return [{}]
    keys = list(grid)
    return [dict(zip(keys, vals)) for vals in itertools.product(*(grid[k] for k in keys))]


def build_runs(exp: Experiment, mode: str, context: Dict) -> List[RunSpec]:
    p = exp.params
    ctx_backbone = context.get("backbone", "resnet")
    ctx_w_dice = context.get("w_dice", 1.0)
    ctx_dims = context.get("dims", (50, 50))

    if exp.entry == "vae":
        runs = []
        for pt in _grid_points(p.get("grid", {})):
            meta = {
                "representation": p.get("representation", "lesion"),
                "fusion_mode": pt.get("fusion_mode", p.get("fusion_mode", "both")),
                "backbone": pt.get("backbone", p.get("backbone", ctx_backbone)),
                "w_dice": pt.get("w_dice", p.get("w_dice", ctx_w_dice)),
                "use_daft": p.get("use_daft", False),
                # the certified re-judging adopted the ARD prior (E5) for the
                # downstream chain; experiments that pin their own use_ard in
                # params keep it, everything else inherits the context decision
                "use_ard": p.get("use_ard", context.get("use_ard", False)),
                "use_pns": p.get("use_pns", False),
                "lambda_pns": pt.get("lambda_pns", p.get("lambda_pns", 0.1)),
                "d_lesion": scale_dim(mode, p.get("zdim", ctx_dims[0])),
                "d_disco": scale_dim(mode, p.get("zdim", ctx_dims[1])),
            }
            label = _label(exp.eid, pt, meta)
            runs.append(RunSpec(label=label, kind="fusion_vae", meta=meta))
        return runs

    if exp.entry == "e3_sweep":
        # The E4 grid, mirrored from train.sweep_dims.e3_grid but inlined so the
        # builder stays free of the (torch-heavy) training import.
        symmetric = [(25, 25), (50, 50), (75, 75), (100, 100)]
        asym = [(75, 25), (60, 40), (40, 60), (25, 75)]
        grid = symmetric + (asym if p.get("asymmetric", True) else [])
        runs = []
        for d_les, d_dis in grid:
            meta = {"representation": "lesion", "fusion_mode": "both",
                    "backbone": ctx_backbone, "w_dice": ctx_w_dice,
                    "d_lesion": scale_dim(mode, d_les), "d_disco": scale_dim(mode, d_dis),
                    "nominal_dims": (d_les, d_dis)}
            runs.append(RunSpec(label=f"{exp.eid}[{d_les}+{d_dis}]", kind="fusion_vae", meta=meta))
        return runs

    if exp.entry == "early_fusion":
        meta = {"representation": "early_fusion", "backbone": ctx_backbone,
                "w_dice": ctx_w_dice, "d_lesion": scale_dim(mode, ctx_dims[0]),
                "d_disco": 0}
        return [RunSpec(label=exp.eid, kind="vae_single", meta=meta)]

    if exp.entry == "dmvae":
        meta = {"shared_dim": scale_dim(mode, p.get("shared_dim", 50)),
                "private_dim": scale_dim(mode, p.get("private_dim", 25)),
                "backbone": ctx_backbone}
        return [RunSpec(label=exp.eid, kind="dmvae", meta=meta)]

    if exp.entry == "contrastive":
        meta = {"no_recon": p.get("no_recon", False), "backbone": ctx_backbone}
        return [RunSpec(label=exp.eid, kind="contrastive", meta=meta)]

    if exp.entry == "mae":
        meta = {"mask_ratio": p.get("mask_ratio", 0.75),
                "lesion_weight": p.get("lesion_weight", 5.0)}
        return [RunSpec(label=exp.eid, kind="mae", meta=meta)]

    if exp.entry == "dscm":
        meta = {"multi_env": p.get("multi_env", False), "use_ard": p.get("use_ard", False)}
        return [RunSpec(label=exp.eid, kind="dscm", meta=meta)]

    if exp.entry == "curriculum":
        return [RunSpec(label=exp.eid, kind="curriculum",
                        meta={"variants": p.get("variants", ["reference"])})]

    if exp.entry == "reference":
        return [RunSpec(label=_label(exp.eid, pt, {}), kind="reference",
                        meta={"method": pt["method"]})
                for pt in _grid_points(p.get("grid", {"method": ["nmf50", "nmf21", "volume"]}))]

    if exp.entry == "interaction_2x2":
        # the two best backbones x the two best Dice weights, read from the
        # rankings the selection experiments stored in the winner context
        backbones = _top2_from_ranking(context, "E3", "backbone", [ctx_backbone, "cnn"])
        w_dices = [float(v) for v in _top2_from_ranking(context, "E2", "w_dice",
                                                        [ctx_w_dice, 0.5])]
        runs = []
        for bb in backbones:
            for wd in w_dices:
                meta = {"representation": "lesion", "fusion_mode": "both",
                        "backbone": bb, "w_dice": wd, "use_daft": False,
                        "use_ard": False, "use_pns": False, "lambda_pns": 0.1,
                        "d_lesion": scale_dim(mode, ctx_dims[0]),
                        "d_disco": scale_dim(mode, ctx_dims[1])}
                runs.append(RunSpec(label=f"{exp.eid}[backbone={bb},w_dice={wd}]",
                                    kind="fusion_vae", meta=meta))
        return runs

    if exp.entry == "audit":
        meta = {"representation": "lesion", "fusion_mode": "both",
                "backbone": ctx_backbone, "w_dice": ctx_w_dice,
                "d_lesion": scale_dim(mode, ctx_dims[0]),
                "d_disco": scale_dim(mode, ctx_dims[1]), "audit": True}
        return [RunSpec(label=exp.eid, kind="fusion_vae", meta=meta)]

    raise ValueError(f"unknown entry {exp.entry!r} for {exp.eid}")


def _label(eid: str, pt: Dict, meta: Dict) -> str:
    if pt:
        return eid + "[" + ",".join(f"{k}={v}" for k, v in pt.items()) + "]"
    return eid


def _label_value(label: str, key: str) -> Optional[str]:
    """The value of ``key`` inside a label like E3[backbone=cnn,w_dice=0.5]."""
    if key + "=" not in label:
        return None
    v = label.split(key + "=", 1)[1]
    return v.split(",", 1)[0].rstrip("]")


def _top2_from_ranking(context: Dict, eid: str, key: str, fallback) -> List[str]:
    """The two best values of ``key`` from the eid's stored ranking, padded
    with fallbacks when fewer than two are available."""
    vals: List[str] = []
    for lbl in context.get("ranking", {}).get(eid, []):
        v = _label_value(lbl, key)
        if v is not None and v not in vals:
            vals.append(v)
        if len(vals) == 2:
            return vals
    for f in fallback:
        if str(f) not in vals:
            vals.append(str(f))
        if len(vals) == 2:
            break
    return vals[:2]


# --------------------------------------------------------------------------- #
# Executors: RunSpec + seed -> artifacts (and optional Tier-4 override)
# --------------------------------------------------------------------------- #
def _apply_overrides(cfg: Dict, mode: str, overrides: Optional[Dict]):
    if mode == "prototype" and overrides:
        cfg.setdefault("data", {})
        for k in ("resolution", "n_synth", "val_frac"):
            if k in overrides:
                cfg["data"][k] = overrides[k]
        if "epochs" in overrides and "vae" in cfg:
            cfg["vae"]["epochs"] = overrides["epochs"]
        if "epochs" in overrides and "train" in cfg:
            cfg["train"]["epochs"] = overrides["epochs"]
        if "channels" in overrides:
            if "vae" in cfg:
                cfg["vae"]["channels"] = overrides["channels"]
            if "model" in cfg:
                cfg["model"]["channels"] = overrides["channels"]
        if "batch_size" in overrides:
            if "vae" in cfg:
                cfg["vae"]["batch_size"] = overrides["batch_size"]
            if "train" in cfg:
                cfg["train"]["batch_size"] = overrides["batch_size"]
    return cfg


# Within one runner invocation, identical modality trainings are computed once.
# The key is the full effective config, so only true repeats hit the cache —
# e.g. the E2 w_dice sweep, where the disconnectome VAE (MSE loss, w_dice
# forced to 1.0 below) is identical across the three lambda values and would
# otherwise be retrained per variant (~10 GPU-hours per E2 run wasted).
_MODALITY_CACHE: Dict[str, "art.VaeArtifacts"] = {}


def _train_modality(mode, representation, zdim, meta, seed, out_dir, overrides):
    """Train one VAE (lesion or disconnectome) and return its VaeArtifacts."""
    import json

    from ..train.train_vae import (_build_dataset, full_config, prototype_config,
                                   run_vae)
    cfg = prototype_config() if mode == "prototype" else full_config()
    cfg["seed"] = seed
    cfg["out_dir"] = out_dir
    cfg["export"] = False
    cfg["representation"] = representation
    cfg["vae"]["zdim"] = int(zdim)
    cfg["vae"]["backbone"] = meta.get("backbone", "resnet")
    cfg["vae"]["w_dice"] = float(meta.get("w_dice", 1.0))
    cfg["vae"]["use_daft"] = bool(meta.get("use_daft", False))
    cfg["vae"]["use_ard"] = bool(meta.get("use_ard", False))
    cfg["vae"]["use_pns"] = bool(meta.get("use_pns", False))
    cfg["vae"]["lambda_pns"] = float(meta.get("lambda_pns", 0.1))
    if representation == "disconnectome":
        cfg["vae"]["w_dice"] = 1.0  # ignored by MSE loss
    _apply_overrides(cfg, mode, overrides)

    key = json.dumps({"mode": mode, "rep": representation, "seed": seed,
                      "vae": cfg["vae"], "data": cfg["data"],
                      "clinical": cfg.get("clinical_csv"),
                      "outcome": cfg.get("outcome_csv")},
                     sort_keys=True, default=str)
    cached = _MODALITY_CACHE.get(key)
    if cached is not None:
        log.info("  reusing cached %s VAE (identical config, seed %d) instead of retraining",
                 representation, seed)
        return cached

    model, _ = run_vae(cfg)
    in_shape = tuple(cfg["data"]["resolution"])
    dataset, _, _ = _build_dataset(cfg, representation, in_shape, cfg["vae"]["use_daft"])
    a = art.vae_artifacts(model, dataset, device=cfg.get("device", "cpu"),
                          batch_size=cfg["vae"]["batch_size"],
                          representation=representation,
                          use_daft=cfg["vae"]["use_daft"])
    _MODALITY_CACHE[key] = a
    return a


def _exec_fusion_vae(spec: RunSpec, mode: str, seed: int, out_dir: str,
                     overrides: Optional[Dict]):
    fusion = spec.meta.get("fusion_mode", "both")
    parts = []
    if fusion in ("both", "lesion"):
        parts.append(_train_modality(mode, "lesion", spec.meta["d_lesion"], spec.meta,
                                      seed, os.path.join(out_dir, "lesion"), overrides))
    if fusion in ("both", "disconnectome"):
        parts.append(_train_modality(mode, "disconnectome", spec.meta.get("d_disco", spec.meta["d_lesion"]),
                                      spec.meta, seed, os.path.join(out_dir, "disco"), overrides))
    # concatenate the modality codes into the fused representation
    lead = parts[0]
    Z = np.concatenate([p.Z for p in parts], axis=1)
    logvar = np.concatenate([p.logvar for p in parts], axis=1)
    dice = next((p.dice for p in parts if p.dice is not None), None)
    prior_var = None
    if any(p.prior_var is not None for p in parts):
        prior_var = np.concatenate([
            p.prior_var if p.prior_var is not None else np.ones(p.Z.shape[1]) for p in parts])
    return art.VaeArtifacts(Z=Z, logvar=logvar, dice=dice, bce=lead.bce,
                            clinical=lead.clinical, volume=lead.volume,
                            prior_var=prior_var, has_posterior=True,
                            meta={"fusion_mode": fusion, "zdim": int(Z.shape[1])})


def _exec_vae_single(spec: RunSpec, mode: str, seed: int, out_dir: str, overrides):
    a = _train_modality(mode, "early_fusion", spec.meta["d_lesion"], spec.meta,
                        seed, out_dir, overrides)
    return a


def _exec_reference(spec: RunSpec, mode, seed, out_dir, overrides):
    """E0: no-VAE reference representations from Table 2 (Giles) -- NMF factor
    scores over the binary lesion masks (sklearn, sparse input) or the lesion
    volume as a single feature. CPU-only; the same tiers judge the result."""
    import torch
    from scipy import sparse
    from sklearn.decomposition import NMF

    from ..data.nifti_dataset import LesionMaskDataset
    from ..train.train_vae import full_config, prototype_config

    cfg = prototype_config() if mode == "prototype" else full_config()
    cfg["seed"] = seed
    _apply_overrides(cfg, mode, overrides)
    in_shape = tuple(cfg["data"]["resolution"])
    ds = LesionMaskDataset(root=cfg["data"]["root"], in_shape=in_shape,
                           n_synth=cfg["data"]["n_synth"], seed=seed, binarize=True)
    n = len(ds)
    method = spec.meta["method"]

    vols = np.zeros(n, dtype=np.float64)
    rows_idx, cols_idx = [], []
    with torch.no_grad():
        for i in range(n):
            item = ds[i]
            x = item[0] if isinstance(item, (tuple, list)) else item
            nz = torch.nonzero(x.flatten() > 0.5).flatten().numpy()
            vols[i] = float(len(nz))
            if method != "volume":
                rows_idx.append(np.full(len(nz), i, dtype=np.int64))
                cols_idx.append(nz.astype(np.int64))

    if method == "volume":
        Z = (vols[:, None] / max(float(vols.max()), 1.0)).astype(np.float64)
    else:
        X = sparse.csr_matrix(
            (np.ones(int(vols.sum()), dtype=np.float32),
             (np.concatenate(rows_idx), np.concatenate(cols_idx))),
            shape=(n, int(np.prod(in_shape))))
        k_nominal = 50 if method == "nmf50" else 21
        k = max(2, min(scale_dim(mode, k_nominal), n - 1))
        nmf = NMF(n_components=k, init="nndsvd", max_iter=200,
                  random_state=seed, tol=1e-3)
        Z = nmf.fit_transform(X)
    log.info("  E0 %s: Z %s from %d %s masks", method, Z.shape, n,
             "synthetic" if ds.synthetic else "real")
    return art.latent_artifacts(Z, volume=vols, has_posterior=False,
                                meta={"method": method, "zdim": int(Z.shape[1])})


def _exec_exported(kind: str, run_fn, cfg: Dict, out_dir: str, npz_name: str,
                   has_posterior: bool = False):
    cfg["export"] = True
    run_fn(cfg)
    path = os.path.join(out_dir, npz_name)
    with np.load(path) as _npz:            # close the handle (Windows file-lock safe)
        Z = _npz["Z"].copy()
    return art.latent_artifacts(Z, has_posterior=has_posterior)


def _exec_dmvae(spec, mode, seed, out_dir, overrides):
    from ..train.train_dmvae import full_config, prototype_config, run_dmvae
    cfg = prototype_config() if mode == "prototype" else full_config()
    cfg["seed"] = seed; cfg["out_dir"] = out_dir
    cfg["model"]["shared_dim"] = spec.meta["shared_dim"]
    cfg["model"]["private_dim"] = spec.meta["private_dim"]
    _apply_overrides(cfg, mode, overrides)
    return _exec_exported("dmvae", run_dmvae, cfg, out_dir, "latents_dmvae.npz",
                          has_posterior=True)


def _exec_contrastive(spec, mode, seed, out_dir, overrides):
    from ..train.train_contrastive import (full_config, prototype_config,
                                           run_contrastive)
    cfg = prototype_config() if mode == "prototype" else full_config()
    cfg["seed"] = seed; cfg["out_dir"] = out_dir
    if "model" in cfg and spec.meta.get("no_recon"):
        cfg["model"]["recon"] = False
    _apply_overrides(cfg, mode, overrides)
    return _exec_exported("contrastive", run_contrastive, cfg, out_dir,
                          "latents_contrastive.npz", has_posterior=False)


def _exec_mae(spec, mode, seed, out_dir, overrides):
    from ..train.train_mae import full_config, prototype_config, run_mae
    cfg = prototype_config() if mode == "prototype" else full_config()
    cfg["seed"] = seed; cfg["out_dir"] = out_dir
    if "mae" in cfg:
        cfg["mae"]["mask_ratio"] = spec.meta.get("mask_ratio", 0.75)
        cfg["mae"]["lesion_weight"] = spec.meta.get("lesion_weight", 5.0)
    _apply_overrides(cfg, mode, overrides)
    return _exec_exported("mae", run_mae, cfg, out_dir, "latents_mae.npz",
                          has_posterior=False)


def _exec_dscm(spec, mode, seed, out_dir, overrides):
    from ..train.train_dscm import full_config, prototype_config, run_dscm
    cfg = prototype_config() if mode == "prototype" else full_config()
    cfg["seed"] = seed; cfg["out_dir"] = out_dir
    if spec.meta.get("multi_env") and "dscm" in cfg:
        cfg["dscm"]["multi_env"] = True
    if spec.meta.get("use_ard") and "dscm" in cfg:
        cfg["dscm"]["use_ard"] = True
    _apply_overrides(cfg, mode, overrides)
    return _exec_exported("dscm", run_dscm, cfg, out_dir, "latents_dscm.npz",
                          has_posterior=False)


# --------------------------------------------------------------------------- #
# One experiment across its variants and seeds
# --------------------------------------------------------------------------- #
def _strata_from(a: art.VaeArtifacts) -> Dict[str, np.ndarray]:
    n = len(a.Z)
    strata = {"volume_quartile": art.volume_quartiles(a.volume)}
    if a.clinical is not None and a.clinical.shape[1] >= 2:
        age = a.clinical[:, 0]
        strata["age_band"] = np.digitize(age, np.quantile(age, [0.5])) if age.std() > 0 else np.zeros(n, int)
        sex = a.clinical[:, 1]
        strata["sex"] = (sex > np.median(sex)).astype(int)
    # anterior/posterior territory proxy: split the cohort on the sign of the
    # first latent direction (a stand-in until a real territory label is joined)
    strata["territory"] = (a.Z[:, 0] > np.median(a.Z[:, 0])).astype(int)
    return strata


def run_experiment(eid: str, mode: str = "prototype", seeds: int = 3,
                   context: Optional[Dict] = None, base_seed: int = 0,
                   out_root: str = "outputs/experiments",
                   overrides: Optional[Dict] = None,
                   logger: Optional[ExperimentLogger] = None,
                   only: Optional[str] = None) -> Dict:
    """Run every variant of an experiment across ``seeds`` seeds, evaluate the
    tiers, aggregate and select a winner. Returns a result dict and, as a side
    effect, updates ``context`` with the winner for downstream experiments.

    ``only`` (comma-separated substrings) restricts the variants, so a big
    grid can be sharded across parallel cluster jobs (one backbone per GPU).
    A sharded invocation sees only part of the grid, so it does NOT select or
    propagate a winner; run ``--finalize`` once all shards are done."""
    exp = get_experiment(eid)
    context = context if context is not None else {}
    log.info("=== %s (arm %s): %s ===", exp.eid, exp.arm, exp.title)

    # E12 is CausalPFN-only: handled by the curriculum ablation directly.
    if exp.entry == "curriculum":
        return _run_curriculum(exp, mode, context, out_root, logger)

    specs = build_runs(exp, mode, context)
    if only:
        pats = [p.strip().lower() for p in only.split(",") if p.strip()]
        specs = [sp for sp in specs if any(p in sp.label.lower() for p in pats)]
        if not specs:
            raise ValueError(f"--only {only!r} matched no {eid} variant")
        log.info("shard: running %d of the %s variants (%s)", len(specs), eid,
                 ", ".join(sp.label for sp in specs))
    per_label: Dict[str, List[RunResult]] = {}

    for spec in specs:
        for s in range(seeds):
            seed = base_seed + s
            out_dir = os.path.join(out_root, exp.eid, spec.label.replace("/", "_"), f"seed{seed}")
            os.makedirs(out_dir, exist_ok=True)
            a, t4_override = _dispatch(spec, mode, seed, out_dir, overrides)
            strata = _strata_from(a)
            report = run_tiers(exp, a, seed=seed, strata=strata, t4_result=t4_override)
            summary = report.summary()
            summary.update({"label": spec.label, "seed": seed, **{f"meta.{k}": v
                            for k, v in spec.meta.items() if isinstance(v, (int, float, str))}})
            per_label.setdefault(spec.label, []).append(
                RunResult(exp.eid, spec.label, seed, summary, report, spec.meta))
            if logger:
                logger.log_metrics(summary, tag=f"{exp.eid}/{spec.label}")
            log.info("  %-28s seed %d  %s", spec.label, seed, _fmt_summary(summary))

    agg = _aggregate(exp, per_label)
    if only:
        # partial grid: leave winner selection to the --finalize pass
        log.info("shard complete; run --experiment %s --finalize when all shards are done", eid)
        return {"eid": exp.eid, "arm": exp.arm, "aggregate": agg, "winner": None}
    winner = _select_winner(exp, agg)
    _propagate(exp, winner, agg, context)
    apply_manual_pins(context)
    result = {"eid": exp.eid, "arm": exp.arm, "aggregate": agg, "winner": winner}
    if logger:
        logger.log_metrics({"eid": exp.eid, "winner": winner.get("label") if winner else None,
                            **(winner or {})}, tag=f"{exp.eid}/winner")
    return result


def finalize_experiment(eid: str, context: Dict,
                        out_root: str = "outputs/experiments") -> Dict:
    """Aggregate an experiment from ``runs.jsonl`` (no training), select the
    winner and propagate it into ``context``. Used after parallel shards, or to
    re-derive the winner after pruning stale rows from the history."""
    import types

    exp = get_experiment(eid)
    path = os.path.join(out_root, "runs.jsonl")
    rows: Dict = {}
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                label = str(r.get("label", ""))
                if "seed" not in r:
                    continue
                if label == eid or label.startswith(eid + "["):
                    rows[(label, int(r["seed"]))] = r      # last write wins
    if not rows:
        raise ValueError(f"no {eid} rows found in {path}")

    per_label: Dict[str, List[RunResult]] = {}
    for (label, seed), summary in rows.items():
        shim = types.SimpleNamespace(passed=bool(summary.get("passed")),
                                     deprioritized=bool(summary.get("deprioritized", False)))
        meta = {k[len("meta."):]: v for k, v in summary.items() if k.startswith("meta.")}
        per_label.setdefault(label, []).append(
            RunResult(eid, label, seed, summary, shim, meta))

    agg = _aggregate(exp, per_label)
    winner = _select_winner(exp, agg)
    _propagate(exp, winner, agg, context)
    apply_manual_pins(context)
    log.info("%s finalized over %d rows: winner %s", eid, len(rows),
             winner.get("label") if winner else None)
    return {"eid": exp.eid, "arm": exp.arm, "aggregate": agg, "winner": winner}


def apply_manual_pins(context: Dict) -> None:
    """Re-assert curated decisions stored under context["manual"].

    A pin is ``{key: {"value": ..., "reason": "..."}}`` (or a bare value).
    Plain keys (backbone, w_dice, dims, fusion_mode) overwrite the context
    entry; a ``winners.Ex`` key pins that experiment's winner label. Pins are
    applied after every propagate/finalize and on load, so an automated
    winner selection can never silently override a recorded manual decision —
    which happened twice with the E3 backbone incumbent."""
    for key, pin in (context.get("manual") or {}).items():
        val = pin.get("value") if isinstance(pin, dict) else pin
        if key.startswith("winners."):
            context.setdefault("winners", {})[key.split(".", 1)[1]] = val
        elif key == "dims":
            context[key] = tuple(val) if isinstance(val, (list, tuple)) else val
        else:
            context[key] = val


def load_context(out_root: str) -> Dict:
    """The cross-job winner context (out_root/context.json). Winners selected
    in one cluster job are inherited by the next through this file."""
    path = os.path.join(out_root, "context.json")
    if os.path.exists(path):
        with open(path) as f:
            ctx = json.load(f)
        if isinstance(ctx.get("dims"), list):
            ctx["dims"] = tuple(ctx["dims"])
        apply_manual_pins(ctx)
        return ctx
    return {}


def save_context(out_root: str, context: Dict) -> None:
    """Write the context, preserving manual pins against the read-modify-write
    race: a long cluster job loads the context at start and saves at exit, so
    without this merge it would clobber any pin recorded on disk meanwhile."""
    path = os.path.join(out_root, "context.json")
    try:
        with open(path) as f:
            on_disk = json.load(f)
        merged = {**(on_disk.get("manual") or {}), **(context.get("manual") or {})}
        if merged:
            context["manual"] = merged
    except (FileNotFoundError, json.JSONDecodeError):
        pass
    apply_manual_pins(context)
    with open(path, "w") as f:
        json.dump(context, f, indent=2, default=list)


def _dispatch(spec: RunSpec, mode, seed, out_dir, overrides):
    if spec.kind == "fusion_vae":
        return _exec_fusion_vae(spec, mode, seed, out_dir, overrides), None
    if spec.kind == "vae_single":
        return _exec_vae_single(spec, mode, seed, out_dir, overrides), None
    if spec.kind == "dmvae":
        return _exec_dmvae(spec, mode, seed, out_dir, overrides), None
    if spec.kind == "contrastive":
        return _exec_contrastive(spec, mode, seed, out_dir, overrides), None
    if spec.kind == "mae":
        return _exec_mae(spec, mode, seed, out_dir, overrides), None
    if spec.kind == "dscm":
        return _exec_dscm(spec, mode, seed, out_dir, overrides), None
    if spec.kind == "reference":
        return _exec_reference(spec, mode, seed, out_dir, overrides), None
    raise ValueError(f"unknown run kind {spec.kind!r}")


def _run_curriculum(exp, mode, context, out_root, logger):
    from ..train.curriculum import (full_config, prototype_config,
                                    run_curriculum_ablation)
    cfg = prototype_config() if mode == "prototype" else full_config()
    cfg["out_dir"] = os.path.join(out_root, exp.eid)
    variants = tuple(exp.params.get("variants", ["reference"]))
    res = run_curriculum_ablation(cfg, variants=variants)
    flat = {v: {k: r[k] for k in ("root_pehe", "prescriptive_accuracy",
                                  "steps", "steps_to_half")} for v, r in res.items()}
    if logger:
        for v, m in flat.items():
            logger.log_metrics({"variant": v, **m}, tag=f"{exp.eid}/{v}")
    winner = min(flat.items(), key=lambda kv: kv[1]["root_pehe"])
    return {"eid": exp.eid, "arm": exp.arm, "aggregate": flat,
            "winner": {"label": winner[0], **winner[1]}}


# --------------------------------------------------------------------------- #
# Aggregation, selection, winner propagation
# --------------------------------------------------------------------------- #
def _numeric_keys(rows: List[RunResult]) -> List[str]:
    keys = set()
    for r in rows:
        for k, v in r.summary.items():
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                keys.add(k)
    return sorted(keys)


def _aggregate(exp: Experiment, per_label: Dict[str, List[RunResult]]) -> Dict:
    agg = {}
    for label, rows in per_label.items():
        entry = {"label": label, "n_seeds": len(rows)}
        for k in _numeric_keys(rows):
            vals = [r.summary[k] for r in rows if isinstance(r.summary.get(k), (int, float))]
            if vals:
                entry[f"{k}.mean"] = float(np.mean(vals))
                entry[f"{k}.std"] = float(np.std(vals))
        # gate pass-rate across seeds
        entry["passed_frac"] = float(np.mean([1.0 if r.report.passed else 0.0 for r in rows]))
        entry["deprioritized_any"] = any(r.report.deprioritized for r in rows)
        agg[label] = entry
    return agg


def _select_winner(exp: Experiment, agg: Dict) -> Optional[Dict]:
    if not agg:
        return None
    select_by = exp.params.get("select_by", "T4")
    metric = {"T4": "T4.root_pehe.mean", "T2": "T2.r2_nihss.mean",
              "T1": "T1.dice.mean", "T3": "T3.active_dims.mean"}.get(select_by, "T4.root_pehe.mean")
    lower_is_better = select_by in ("T4",)
    candidates = [e for e in agg.values() if metric in e]
    if not candidates:
        return list(agg.values())[0]
    winner = (min if lower_is_better else max)(candidates, key=lambda e: e[metric])
    return winner


def _rank_labels(exp: Experiment, agg: Dict) -> List[str]:
    """Variant labels ordered best-first by the experiment's selection metric
    (the same rule as _select_winner); consumed by E11b's 2x2 grid."""
    select_by = exp.params.get("select_by", "T4")
    metric = {"T4": "T4.root_pehe.mean", "T2": "T2.r2_nihss.mean",
              "T1": "T1.dice.mean", "T3": "T3.active_dims.mean"}.get(select_by, "T4.root_pehe.mean")
    lower = select_by in ("T4",)
    cands = [e for e in agg.values() if metric in e]
    return [e["label"] for e in sorted(cands, key=lambda e: e[metric], reverse=not lower)]


def _propagate(exp: Experiment, winner: Optional[Dict], agg: Dict, context: Dict):
    if agg:
        context.setdefault("ranking", {})[exp.eid] = _rank_labels(exp, agg)
    if not winner:
        return
    label = winner["label"]
    if exp.eid == "E2" and "w_dice=" in label:
        context["w_dice"] = float(label.split("w_dice=")[1].rstrip("]"))
    if exp.eid == "E3" and "backbone=" in label:
        context["backbone"] = label.split("backbone=")[1].rstrip("]")
    if exp.eid == "E6" and "fusion_mode=" in label:
        context["fusion_mode"] = label.split("fusion_mode=")[1].rstrip("]")
    if exp.eid == "E4" and "[" in label:
        pair = label.split("[")[1].rstrip("]")
        if "+" in pair:
            d_les, d_dis = pair.split("+")
            context["dims"] = (int(d_les), int(d_dis))
    context.setdefault("winners", {})[exp.eid] = label


def _fmt_summary(s: Dict) -> str:
    keys = ["T1.dice", "T2.r2_nihss", "T3.active_dims", "T4.root_pehe"]
    bits = [f"{k.split('.')[-1]}={s[k]:.3f}" for k in keys
            if isinstance(s.get(k), (int, float))]
    bits.append("PASS" if s.get("passed") else f"stop@{s.get('stopped_at')}")
    return "  ".join(bits)


# --------------------------------------------------------------------------- #
# Arm / full-programme drivers
# --------------------------------------------------------------------------- #
def run_arm(arm: str, mode: str = "prototype", seeds: int = 3,
            out_root: str = "outputs/experiments", overrides: Optional[Dict] = None,
            context: Optional[Dict] = None, logger: Optional[ExperimentLogger] = None) -> Dict:
    context = context if context is not None else {}
    order = ARM_A_ORDER if arm == "A" else tuple(e.eid for e in arm_experiments(arm))
    results = {}
    for eid in order:
        results[eid] = run_experiment(eid, mode=mode, seeds=seeds, context=context,
                                      out_root=out_root, overrides=overrides, logger=logger)
    return {"arm": arm, "context": context, "results": results}


def run_all(mode: str = "prototype", seeds: int = 3,
            out_root: str = "outputs/experiments", overrides: Optional[Dict] = None,
            logger: Optional[ExperimentLogger] = None) -> Dict:
    context: Dict = {}
    results = {}
    for arm in ("A", "B", "C", "D", "E"):
        results[arm] = run_arm(arm, mode=mode, seeds=seeds, out_root=out_root,
                               overrides=overrides, context=context, logger=logger)
    # cross-arm audit and the CausalPFN curriculum, both after the arms
    for eid in ("E11a", "E11b", "E12"):
        results[eid] = run_experiment(eid, mode=mode, seeds=seeds, context=context,
                                      out_root=out_root, overrides=overrides, logger=logger)
    return {"context": context, "results": results}


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main(argv=None):
    import argparse

    ap = argparse.ArgumentParser(description="Table-9 experiment orchestrator")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--experiment", help="a single experiment id, e.g. E3")
    g.add_argument("--arm", help="an arm letter A|B|C|D|E")
    g.add_argument("--all", action="store_true", help="the whole programme")
    ap.add_argument("--mode", default="prototype", choices=["prototype", "full"])
    ap.add_argument("--data-tier", default=None, choices=["trial", "full"],
                    help="cohort tier: 'trial' (data/Trial data) or 'full' (data/Full data)")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--out-root", default="outputs/experiments")
    ap.add_argument("--backend", default="auto",
                    help="logging backend: auto|wandb|mlflow|local")
    ap.add_argument("--report", action="store_true",
                    help="write the leaderboard report at the end")
    ap.add_argument("--only", default=None,
                    help="run only the variants whose label contains one of these "
                         "comma-separated substrings (shard a grid across jobs); "
                         "winner selection is deferred to --finalize")
    ap.add_argument("--finalize", action="store_true",
                    help="no training: aggregate --experiment from runs.jsonl, "
                         "select the winner and update context.json")
    args = ap.parse_args(argv)

    if args.data_tier is not None:
        set_tier(args.data_tier)
    if (args.only or args.finalize) and not args.experiment:
        ap.error("--only/--finalize require --experiment")

    os.makedirs(args.out_root, exist_ok=True)
    # winners chosen by earlier cluster jobs are inherited through this file
    context = load_context(args.out_root)
    if context:
        log.info("inherited context: %s", context)

    if args.finalize:
        out = finalize_experiment(args.experiment, context, out_root=args.out_root)
        save_context(args.out_root, context)
        if args.report:
            from .report import build_report
            log.info("report written to %s", build_report(args.out_root))
        return out

    logger = ExperimentLogger(args.out_root, run_name=args.experiment or args.arm or "all",
                              backend=args.backend)
    try:
        if args.experiment:
            out = run_experiment(args.experiment, mode=args.mode, seeds=args.seeds,
                                 out_root=args.out_root, logger=logger,
                                 context=context, only=args.only)
        elif args.arm:
            out = run_arm(args.arm, mode=args.mode, seeds=args.seeds,
                          out_root=args.out_root, logger=logger, context=context)
        else:
            out = run_all(mode=args.mode, seeds=args.seeds, out_root=args.out_root, logger=logger)
        if not args.only:
            save_context(args.out_root, context)
    finally:
        logger.finish()

    if args.report:
        from .report import build_report
        paths = build_report(args.out_root)
        log.info("report written to %s", paths)
    return out


if __name__ == "__main__":
    main()
