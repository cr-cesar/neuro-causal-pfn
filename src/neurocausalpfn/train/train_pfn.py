"""Stage 2 training (the causal transformer).

Trains on batches sampled from the Neuro-Prior, minimizing the histogram loss
over the true expected conditional potential outcome. The context length
follows a curriculum from shorter to longer.

The prior is selected by configuration (cfg["prior"]["kind"]):
- "synthetic": the lightweight generator (Gaussian covariates from scratch).
- "intersynth": the real anatomical mechanism, which crosses the lesions with
  the functional parcellation. In this mode, d_x is derived from the prior (the
  encoder latent if provided, or the observed covariates otherwise).
"""
import copy
import os
from typing import Dict, Optional

import numpy as np
import torch

from ..eval.metrics import prescriptive_accuracy, root_pehe
from ..prior.cohort import NeuroPrior
from ..data.paths import ATLAS_DIR
from ..utils.logging_utils import get_logger
from ..utils.seed import set_seed
from ..pfn.inference import predict_cate
from ..pfn.model import NeuroCausalPFN
from ..pfn.tokens import to_tensors
from ..utils.runtime import (autocast_ctx, log_runtime, make_grad_scaler,
                             optim_step, resolve_device, use_amp)

log = get_logger()


def build_model(cfg: Dict, d_x: int):
    """Builds the Stage 2 transformer according to the requested architecture:
    'linear' (row-wise projection) or 'tabicl' (column-wise attention and then
    row-wise). ``head_lo`` / ``head_hi`` set the range of the outcome bins
    (default [0, 1], every checkpoint before 28 Oct): with the gaussian target
    of width sigma, a range ending at 1 cannot put its mean at 1, which the
    ideal cell's outcomes need; [-0.05, 1.05] can."""
    p = cfg["pfn"]
    rng_kw = dict(lo=float(p.get("head_lo", 0.0)), hi=float(p.get("head_hi", 1.0)))
    if p.get("arch", "linear") == "tabicl":
        if int(p.get("tabicl_version", 0)) < 2:
            # checkpoints written before the TabICL re-implementation
            from ..pfn.tabicl_v0 import NeuroCausalPFNTabICLv0

            return NeuroCausalPFNTabICLv0(
                d_x=d_x, d_model=p["d_model"], n_row_layers=p["n_layers"],
                n_col_layers=p.get("n_col_layers", 2), n_heads=p["n_heads"],
                n_bins=p["n_bins"], sigma=p["sigma"], **rng_kw)
        from ..pfn.tabicl_model import NeuroCausalPFNTabICL

        return NeuroCausalPFNTabICL(
            d_x=d_x, embed_dim=p.get("embed_dim", 128), n_col_layers=p.get("n_col_layers", 3),
            n_row_layers=p.get("n_row_layers", 3), n_icl_layers=p["n_layers"], n_heads=p["n_heads"],
            n_cls=p.get("n_cls", 4), n_inds=p.get("n_inds", 128), n_bins=p["n_bins"], sigma=p["sigma"],
            **rng_kw)
    return NeuroCausalPFN(
        d_x=d_x, d_model=p["d_model"], n_layers=p["n_layers"], n_heads=p["n_heads"],
        n_bins=p["n_bins"], sigma=p["sigma"], **rng_kw)


def pfn_loss(model, batch: Dict, p: Dict):
    """Training loss of one batch. Default: the histogram loss of each query
    at its own drawn arm (the target is the true expected outcome mu_q).
    ``p["both_arms"]``: every query is scored at both arms, against mu0 and
    mu1, in one pass (queries attend only to the context, so stacking the two
    copies is exact). ``p["cate_weight"]`` > 0 (with both_arms) adds that
    weight times the squared error of the predicted effect, the difference of
    the two predicted means, against mu1 - mu0: the quantity PEHE scores."""
    if not p.get("both_arms"):
        logits = model(batch["Xc"], batch["Tc"], batch["Yc"], batch["Xq"], batch["Tq"])
        return model.head.loss(logits, batch["mu_q"])
    Xq = batch["Xq"]; nq = Xq.shape[1]
    t0, t1 = torch.zeros_like(batch["Tq"]), torch.ones_like(batch["Tq"])
    logits = model(batch["Xc"], batch["Tc"], batch["Yc"], torch.cat([Xq, Xq], dim=1), torch.cat([t0, t1], dim=1))
    loss = model.head.loss(logits, torch.cat([batch["mu0"], batch["mu1"]], dim=1))
    lam = float(p.get("cate_weight", 0.0) or 0.0)
    if lam > 0:
        m = model.head.expect(logits.float())
        cate = m[:, nq:] - m[:, :nq]
        loss = loss + lam * ((cate - (batch["mu1"] - batch["mu0"])) ** 2).mean()
    return loss


def prototype_config() -> Dict:
    return {
        "seed": 0,
        "out_dir": "outputs/pfn_prototype",
        "prior": {"kind": "synthetic"},
        "pfn": {"d_x": 16, "d_model": 128, "n_layers": 2, "n_heads": 4,
                "n_bins": 256, "sigma": 0.02, "arch": "linear", "n_col_layers": 2,
                "context_min": 64, "context_max": 256, "n_query": 16,
                "batch_size": 8, "iters": 2000,
                "lr": 3e-4, "weight_decay": 0.01, "grad_clip": 1.0},
        "device": "cpu",
        "log_every": 100,
    }


def full_config() -> Dict:
    return {
        "seed": 0,
        "out_dir": "outputs/pfn_full",
        # change kind to "intersynth" to use the real anatomical substrate;
        # point atlas_dir to data/atlases and, ideally, pass the encoder latents
        # as z_pool from an orchestration script.
        "prior": {"kind": "synthetic",
                  "atlas_dir": ATLAS_DIR, "atlas_shape": [96, 112, 96],
                  "modality": "receptor",
                  "pool_size": 4000, "unobserved_strength": 0.0},
        "pfn": {"d_x": 104, "d_model": 512, "n_layers": 12, "n_heads": 8,
                "n_bins": 1024, "sigma": 0.02, "arch": "tabicl", "tabicl_version": 2, "embed_dim": 128,
                "n_col_layers": 3, "n_row_layers": 3, "n_cls": 4, "n_inds": 128,
                "context_min": 1000, "context_max": 20000, "n_query": 64,
                "batch_size": 8, "iters": 162000,
                "lr": 3e-4, "weight_decay": 0.01, "grad_clip": 1.0},
        "device": "auto",
        "amp": True,
        "num_workers": 4,
        "log_every": 200,
    }


def reduced_config() -> Dict:
    """First real training: fits in 20-40 A100-hours. TabICL encoder, 6 row
    layers of width 256, context curriculum 256 -> 4,119 (the UCLH cohort),
    Neuro-Prior v1 on a precomputed anatomy cache (set prior.cache and
    prior.atlas_dir). d_x comes from the cache."""
    return {
        "seed": 0,
        "out_dir": "outputs/pfn_reduced",
        "pfn": {"d_x": 50, "d_model": 256, "n_layers": 6, "n_heads": 8,
                "n_bins": 256, "sigma": 0.02, "arch": "tabicl", "tabicl_version": 2, "embed_dim": 96,
                "n_col_layers": 2, "n_row_layers": 2, "n_cls": 4, "n_inds": 64,
                "n_query": 64, "batch_size": 4, "lr": 2e-4, "weight_decay": 0.01,
                "grad_clip": 1.0, "iters": 20000, "context_min": 256, "context_max": 4119},
        "prior": {"kind": "neuro_prior", "family": "theory", "cache": None, "atlas_dir": str(ATLAS_DIR),
                  "augment": True},
        "device": "auto",
        "amp": True,
        "log_every": 100,
    }


def _context_length(cfg: Dict, it: int) -> int:
    """Context length at iteration ``it``. ``cfg["pfn"]["curriculum"]``:
    ``"context"`` (default) grows the context linearly from context_min to
    context_max over the first half of training; ``"none"`` keeps it at
    context_max from the start (the E12 ablation without curriculum)."""
    p = cfg["pfn"]
    if p.get("curriculum", "context") == "none":
        return int(p["context_max"])
    frac = min(1.0, (it + 1) / max(1, int(0.5 * p["iters"])))
    return int(p["context_min"] + frac * (p["context_max"] - p["context_min"]))


def _stage_index(cfg: Dict, it: int) -> int:
    """Index of the hyper-prior stage active at iteration ``it``; -1 without
    stages. ``cfg["prior"]["stages"]`` is a list of {"until": fraction of
    iters, "hyper": {ranges}} in order (the design's staged confounding:
    mild gamma and moderate beta first, the full ranges afterwards)."""
    stages = cfg.get("prior", {}).get("stages") or []
    if not stages:
        return -1
    frac = (it + 1) / max(1, int(cfg["pfn"]["iters"]))
    for k, st in enumerate(stages):
        if frac <= float(st.get("until", 1.0)):
            return k
    return len(stages) - 1


def _as_ranges(ranges: Dict) -> Dict:
    return {k: tuple(v) if isinstance(v, list) else v for k, v in ranges.items()}


def parse_hyper_opts(text: str) -> Dict:
    """Hyper-prior overrides given as a string, e.g. the PRIOR_HYPER variable of
    the Myriad jobs (qsub -v cannot carry commas): space-separated key=value
    items, a range as lo:hi, a flag as a number ("two_arms=1 effect_scale=0.2:1
    binary_outcome=0.5"). Returns a dict for cfg["prior"]["hyper"]."""
    out: Dict = {}
    for item in (text or "").split():
        key, _, val = item.partition("=")
        if not key or not val:
            raise ValueError(f"bad hyper option {item!r}: expected key=value")
        if ":" in val:
            out[key] = [float(x) for x in val.split(":")]
        else:
            try:
                out[key] = float(val)
            except ValueError:          # a channel name, e.g. d_source=network
                out[key] = val
    return out


def _lr_schedule(p: Dict):
    """Learning rate per iteration. ``p["lr_schedule"]`` is "constant" (the
    default, the pilots' behaviour) or "cosine": linear warmup over
    ``p["warmup"]`` iterations (default 2 % of the run) from zero to ``lr``,
    then cosine decay to ``lr * p["min_lr_frac"]`` (default 0.1) at the last
    iteration. The TabICL blocks start as the identity (zero-initialised
    residual branches), which is what the warmup is for."""
    import math

    lr, iters = float(p["lr"]), int(p["iters"])
    kind = str(p.get("lr_schedule", "constant"))
    if kind == "constant":
        return lambda it: lr
    if kind != "cosine":
        raise ValueError(f"unknown lr_schedule {kind!r}: constant | cosine")
    warmup = int(p.get("warmup", max(1, round(0.02 * iters))))
    floor = float(p.get("min_lr_frac", 0.1)) * lr

    def lr_at(it: int) -> float:
        if it < warmup:
            return lr * (it + 1) / warmup
        frac = (it - warmup) / max(1, iters - warmup)
        return floor + 0.5 * (lr - floor) * (1.0 + math.cos(math.pi * min(1.0, frac)))

    return lr_at


def _hyper_for(cfg: Dict, ranges: Dict):
    """Hyper-prior of the configured family with ``ranges`` overriding the
    defaults; for the mixture these are the theory ranges (the ones the
    curriculum stages narrow)."""
    from ..prior.generic_mix import split_generic_hyper

    pr = cfg.get("prior", {})
    if pr.get("family", "theory") in ("theory", "mixture"):
        from ..prior.intersynth_theory import TheoryHyperPrior as _Hyper
    else:
        from ..prior.neuro_prior import HyperPrior as _Hyper
    ranges, _generic = split_generic_hyper(ranges)   # generic_* options belong to the wrapper, not the family
    return _Hyper(**_as_ranges(ranges))


E12_VARIANTS = ("nocurr", "ctx", "ctx+stages")


# the first-half ranges of ctx+stages, per family: mild confounding first.
# theory: gamma (confounding) and beta (effect scale), as in the design;
# giles: the allocation strength of the virtual trial (its confounding).
_STAGE1_THEORY = {"gamma": [0.0, 0.3], "beta": [0.2, 0.4]}
_STAGE1_GILES = {"bias": [0.0, 0.3]}


def with_family(cfg: Dict, family: str, weights=None, context_pool: Optional[str] = None) -> Dict:
    """Set the Neuro-Prior family of a reduced / E12 configuration
    (theory | giles | mixture), for the mixture the family weights, and
    optionally the context pool (susceptible | all)."""
    from ..prior.cohort import CONTEXT_POOLS, FAMILIES

    if family not in FAMILIES:
        raise ValueError(f"unknown prior family {family!r}; choose from {FAMILIES}")
    cfg["prior"]["family"] = family
    if family == "mixture":
        cfg["prior"]["family_weights"] = list(weights) if weights is not None else [0.5, 0.5]
    if context_pool is not None:
        if context_pool not in CONTEXT_POOLS:
            raise ValueError(f"unknown context_pool {context_pool!r}; choose from {CONTEXT_POOLS}")
        cfg["prior"]["context_pool"] = context_pool
    if cfg["prior"].get("context_pool", "susceptible") == "susceptible":
        # the evaluation's contexts are the susceptible patients of one deficit
        # (tens to a few hundred); a 4,119-row context of a slice would be
        # dozens of copies of each lesion, at quadratic cost in attention
        cfg["pfn"]["context_max"] = min(int(cfg["pfn"]["context_max"]), 1024)
    return cfg


def e12_config(variant: str, family: str = "theory", weights=None, context_pool: Optional[str] = None) -> Dict:
    """The curriculum ablation of the design (E12) on the reduced
    configuration. ``nocurr``: fixed context, full hyper-prior ranges from
    the start. ``ctx``: the context-length curriculum of the pilot.
    ``ctx+stages``: context curriculum plus the staged confounding of the
    design for the first half (theory: gamma in [0, 0.3] and beta in
    [0.2, 0.4]; giles: allocation strength in [0, 0.3]; the mixture narrows
    both), full ranges afterwards."""
    if variant not in E12_VARIANTS:
        raise ValueError(f"unknown E12 variant {variant!r}; choose from {E12_VARIANTS}")
    cfg = with_family(reduced_config(), family, weights, context_pool)
    cfg["pfn"]["curriculum"] = "none" if variant == "nocurr" else "context"
    if variant == "ctx+stages":
        first = {"theory": {"hyper": _STAGE1_THEORY}, "giles": {"hyper": _STAGE1_GILES},
                 "mixture": {"hyper": _STAGE1_THEORY, "hyper_giles": _STAGE1_GILES}}[family]
        cfg["prior"]["stages"] = [{"until": 0.5, **copy.deepcopy(first)}, {"until": 1.0, "hyper": {}}]
    cfg["variant"] = variant
    return cfg


def _build_prior(cfg: Dict, seed_offset: int = 0):
    """Returns (prior_object_or_None, d_x, is_intersynth). For the synthetic
    prior the object is None (it is re-instantiated per iteration); for InterSynth
    it is built once and reused."""
    p = cfg["pfn"]
    pr = cfg.get("prior", {"kind": "synthetic"})
    if pr.get("kind") in ("intersynth", "neuro_prior"):
        from ..prior.atlas import FunctionalAtlas
        from ..prior.cohort import NeuroPriorCohort, NeuroPriorInterSynth, build_synthetic_lesion_pool

        shape = tuple(pr.get("atlas_shape", [48, 56, 48]))
        seed = cfg["seed"] + seed_offset
        modality = pr.get("modality", "receptor")
        atlas = FunctionalAtlas.from_dir(pr.get("atlas_dir"), shape=shape, seed=seed, modality=modality)
        shape = atlas.shape   # the lesion set must live on the atlas grid
        pool = None if pr.get("cache") else build_synthetic_lesion_pool(int(pr.get("pool_size", 128)), shape=shape, seed=seed)
        if pr.get("kind") == "neuro_prior":
            # Neuro-Prior: a hyper-prior over processes on the anatomy cache.
            # family "theory" (default) is the design's InterSynth (D, S,
            # alpha, beta, gamma, four mechanisms; prior/intersynth_theory.py);
            # family "giles" the v1 virtual-trial generators (prior/neuro_prior.py);
            # family "mixture" draws each process from one of the two with
            # pr["family_weights"] = [theory, giles] (default [0.5, 0.5]).
            # Ranges can be narrowed per curriculum stage through
            # pr["hyper"] = {"gamma": [0, 0.3], "beta": [0.2, 0.4], ...} (the
            # theory ranges for theory and mixture); pr["hyper_giles"] sets the
            # virtual-trial ranges of the mixture.
            from ..prior.neuro_prior import HyperPrior

            family = pr.get("family", "theory")
            hyper = _hyper_for(cfg, pr.get("hyper", {}))
            # context_pool "susceptible" (default): a process draws its context
            # from the lesions it can affect, as the virtual trial evaluates
            # each deficit on its susceptible patients; "all" is the pilot's
            # sampling over the whole pool
            kw = dict(seed=seed, n_context=p["context_max"], n_query=p["n_query"], hyper=hyper,
                      augment=bool(pr.get("augment", True)), family=family,
                      context_pool=pr.get("context_pool", "susceptible"),
                      context_replace=bool(pr.get("context_replace", True)),
                      standardize=pr.get("standardize", "pool"))
            if family == "mixture":
                kw["hyper_giles"] = HyperPrior(**_as_ranges(pr.get("hyper_giles", {})))
                kw["family_weights"] = tuple(pr.get("family_weights", (0.5, 0.5)))
            # generic_weight > 0: a share of the processes get a simple effect
            # in latent space (sparse linear, or three groups with a null
            # share), the function class the anatomy family lacks (23 Oct)
            from ..prior.generic_mix import split_generic_hyper, wrap_generic
            from ..prior.multi_cache import build_multi_cache

            _, generic = split_generic_hyper(pr.get("hyper", {}))
            caches = pr.get("cache")
            if caches:
                # real anatomy and latents, precomputed by scripts/build_prior_cache.py;
                # a list is a multi-encoder pool (prior/multi_cache.py): one cohort
                # per cache, drawn per batch, the first cache the anchor
                paths = [caches] if isinstance(caches, str) else list(caches)
                cohorts = []
                for i, path in enumerate(paths):
                    s = seed + 7919 * i
                    cohort = NeuroPriorCohort.from_cache(atlas, path, **dict(kw, seed=s))
                    cohorts.append(wrap_generic(cohort, generic, seed=s))
                prior = build_multi_cache(cohorts, pr, seed)
            else:
                prior = wrap_generic(NeuroPriorCohort(atlas, pool, **kw), generic, seed=seed)
            if hasattr(prior, "describe") and (generic.get("generic_weight") or isinstance(caches, (list, tuple))):
                log.info("prior: %s", prior.describe())
            return prior, prior.d_x, True
        prior = NeuroPriorInterSynth(atlas, pool, seed=seed,
                                     n_context=p["context_max"], n_query=p["n_query"],
                                     unobserved_strength=float(pr.get("unobserved_strength", 0.0)))
        return prior, prior.d_x, True
    return None, p["d_x"], False


def run_pfn(cfg: Dict):
    set_seed(cfg["seed"])
    device = resolve_device(cfg)
    amp = use_amp(cfg, device)
    scaler = make_grad_scaler(amp)
    p = cfg["pfn"]

    prior_obj, d_x, is_intersynth = _build_prior(cfg)
    model = build_model(cfg, d_x).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=p["lr"], weight_decay=p["weight_decay"])
    lr_at = _lr_schedule(p)
    log_runtime("PFN", device, amp)
    n_params = sum(t.numel() for t in model.parameters())
    log.info("PFN: %.2fM parameters, d_x=%d, arch=%s, prior=%s",
             n_params / 1e6, d_x, p.get("arch", "linear"), "intersynth" if is_intersynth else "synthetic")

    history = []
    model.train()
    stage = -1
    for it in range(p["iters"]):
        n_ctx = _context_length(cfg, it)
        k = _stage_index(cfg, it)
        if k != stage and is_intersynth and hasattr(prior_obj, "hyper"):
            # staged hyper-prior (E12 ctx+stages): swap the ranges in place
            st = cfg["prior"]["stages"][k]
            # the stage narrows ranges on top of the configured hyper-prior, so
            # its other fields (the theory extensions) persist across stages
            prior_obj.hyper = _hyper_for(cfg, {**cfg["prior"].get("hyper", {}), **st.get("hyper", {})})
            if getattr(prior_obj, "family", None) == "mixture":
                from ..prior.neuro_prior import HyperPrior

                # the stage narrows the configured virtual-trial ranges too (audit of 27 Oct)
                prior_obj.hyper_giles = HyperPrior(**_as_ranges({**cfg["prior"].get("hyper_giles", {}),
                                                                 **st.get("hyper_giles", {})}))
            log.info("hyper-prior stage %d from iter %d: %s%s", k, it + 1, prior_obj.hyper,
                     f" + {prior_obj.hyper_giles}" if getattr(prior_obj, "family", None) == "mixture" else "")
            stage = k
        if is_intersynth:
            batch_np = prior_obj.sample_batch(p["batch_size"], n_context=n_ctx)
        else:
            prior = NeuroPrior(d_x=d_x, n_context=n_ctx, n_query=p["n_query"], seed=cfg["seed"] + it)
            batch_np = prior.sample_batch(p["batch_size"])
        batch = to_tensors(batch_np, device=device)
        with autocast_ctx(device, amp):
            loss = pfn_loss(model, batch, p)
        lr_now = lr_at(it)
        for group in opt.param_groups:
            group["lr"] = lr_now
        optim_step(loss, opt, scaler, params=model.parameters(), grad_clip=p["grad_clip"])

        history.append({"iter": it, "loss": float(loss.detach()), "n_ctx": n_ctx, "lr": lr_now})
        if (it + 1) % cfg["log_every"] == 0 or it == 0:
            log.info("iter %d/%d  n_ctx=%d  loss=%.4f", it + 1, p["iters"], n_ctx, float(loss.detach()))

    os.makedirs(cfg["out_dir"], exist_ok=True)
    ckpt = os.path.join(cfg["out_dir"], "pfn.pt")
    # 28 Oct review: the checkpoint now says which code and launch built it
    from .launch_cfg import provenance

    prov = provenance()
    torch.save({"state_dict": model.state_dict(), "cfg": cfg, "provenance": prov}, ckpt)
    import json

    with open(os.path.join(cfg["out_dir"], "provenance.json"), "w") as f:
        json.dump(prov, f, indent=1, default=str)
    log.info("checkpoint saved to %s (code %s%s)", ckpt, prov.get("git_commit"),
             " dirty" if prov.get("git_dirty") else "")
    return model, history


@torch.no_grad()
def quick_eval(model, cfg: Dict, n_eval: int = 8) -> Dict[str, float]:
    """Quick evaluation on held-out processes (with the same kind of prior)."""
    p = cfg["pfn"]
    device = resolve_device(cfg)
    prior_obj, d_x, is_intersynth = _build_prior(cfg, seed_offset=10_000)
    if is_intersynth:
        batch_np = prior_obj.sample_batch(n_eval, n_context=p["context_max"])
    else:
        prior = NeuroPrior(d_x=d_x, n_context=p["context_max"], n_query=p["n_query"],
                           seed=10_000 + cfg["seed"])
        batch_np = prior.sample_batch(n_eval)
    batch = to_tensors(batch_np, device=device)
    out = predict_cate(model, batch["Xc"], batch["Tc"], batch["Yc"], batch["Xq"])
    cate_true = batch["mu1"] - batch["mu0"]
    return {"root_pehe": root_pehe(out["cate"], cate_true),
            "prescriptive_accuracy": prescriptive_accuracy(out["cate"], cate_true)}


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="prototype", choices=["prototype", "full"])
    ap.add_argument("--prior", default=None, choices=["synthetic", "intersynth"],
                    help="overrides cfg['prior']['kind']")
    ap.add_argument("--arch", default=None, choices=["linear", "tabicl"],
                    help="overrides cfg['pfn']['arch']")
    args = ap.parse_args()
    cfg = prototype_config() if args.mode == "prototype" else full_config()
    if args.prior is not None:
        cfg.setdefault("prior", {})["kind"] = args.prior
    if args.arch is not None:
        cfg["pfn"]["arch"] = args.arch
    trained, _ = run_pfn(cfg)
    log.info("quick evaluation: %s", quick_eval(trained, cfg))
