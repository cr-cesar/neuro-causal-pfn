"""E12: CausalPFN curriculum ablation.

Compares the reference three-stage curriculum against two reduced schedules, by
convergence and final root-PEHE. The stages vary the in-context size, the
confounding strength and the treatment-effect size: a cheap synthetic warm-up
(stage 1), then the full cohort under full confounding (stage 2), then an
augmented context (stage 3). The variants are the reference (all three stages),
two-stage (skip stage 1) and one-stage (the full setting throughout). This trains
the PFN, it reuses the model, prior and CATE evaluation already in the codebase.
"""
import os
from typing import Dict, List

import numpy as np
import torch

from ..eval.metrics import prescriptive_accuracy, root_pehe
from ..pfn.inference import predict_cate
from ..pfn.tokens import to_tensors
from ..prior.cohort import NeuroPrior
from ..utils.logging_utils import get_logger
from ..utils.seed import set_seed
from .train_pfn import build_model

log = get_logger()


def prototype_config() -> Dict:
    return {
        "seed": 0,
        "out_dir": "outputs/e12_prototype",
        "pfn": {"d_x": 16, "d_model": 128, "n_layers": 2, "n_heads": 4,
                "n_bins": 256, "sigma": 0.02, "arch": "linear", "n_col_layers": 2,
                "n_query": 16, "batch_size": 8, "lr": 3e-4, "weight_decay": 0.01,
                "grad_clip": 1.0, "context_max": 192},
        "curriculum": {
            "stage1": {"steps": 60, "n_context": 64, "confound": [0.0, 0.3], "effect": [0.2, 0.4]},
            "stage2": {"steps": 30, "n_context": 128, "confound": [0.0, 1.0], "effect": [0.2, 1.0]},
            "stage3": {"steps": 20, "n_context": 192, "confound": [0.0, 1.0], "effect": [0.2, 1.0]},
        },
        "device": "cpu",
        "log_every": 50,
    }


def full_config() -> Dict:
    return {
        "seed": 0,
        "out_dir": "outputs/e12_full",
        "pfn": {"d_x": 104, "d_model": 512, "n_layers": 12, "n_heads": 8,
                "n_bins": 1024, "sigma": 0.02, "arch": "tabicl", "n_col_layers": 3,
                "n_query": 64, "batch_size": 8, "lr": 3e-4, "weight_decay": 0.01,
                "grad_clip": 1.0, "context_max": 20000},
        "curriculum": {
            "stage1": {"steps": 160000, "n_context": 1024, "confound": [0.0, 0.3], "effect": [0.2, 0.4]},
            "stage2": {"steps": 1500, "n_context": 4119, "confound": [0.0, 1.0], "effect": [0.2, 1.0]},
            "stage3": {"steps": 550, "n_context": 20000, "confound": [0.0, 1.0], "effect": [0.2, 1.0]},
        },
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "log_every": 200,
    }


def stages_for_variant(cfg: Dict, variant: str) -> List[Dict]:
    c = cfg["curriculum"]
    s1, s2, s3 = c["stage1"], c["stage2"], c["stage3"]
    if variant == "reference":
        return [s1, s2, s3]
    if variant == "two_stage":
        return [s2, s3]
    if variant == "one_stage":
        total = s1["steps"] + s2["steps"] + s3["steps"]
        return [{"steps": total, "n_context": s2["n_context"],
                 "confound": s2["confound"], "effect": s2["effect"]}]
    raise ValueError(f"unknown variant: {variant}")


def run_curriculum(cfg: Dict, variant: str):
    set_seed(cfg["seed"])
    device = cfg.get("device", "cpu")
    p = cfg["pfn"]
    model = build_model(cfg, p["d_x"]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=p["lr"], weight_decay=p["weight_decay"])
    stages = stages_for_variant(cfg, variant)
    model.train()
    history, gstep = [], 0
    for si, st in enumerate(stages):
        crange = tuple(st["confound"])
        erange = tuple(st["effect"])
        for _ in range(st["steps"]):
            prior = NeuroPrior(d_x=p["d_x"], n_context=st["n_context"], n_query=p["n_query"],
                               seed=cfg["seed"] + gstep, confound_range=crange, effect_range=erange)
            batch = to_tensors(prior.sample_batch(p["batch_size"]), device=device)
            logits = model(batch["Xc"], batch["Tc"], batch["Yc"], batch["Xq"], batch["Tq"])
            loss = model.head.loss(logits, batch["mu_q"])
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), p["grad_clip"])
            opt.step()
            history.append({"step": gstep, "stage": si, "loss": float(loss.detach()),
                            "n_ctx": st["n_context"]})
            if (gstep + 1) % cfg["log_every"] == 0 or gstep == 0:
                log.info("[%s] step %d  stage %d  n_ctx=%d  loss=%.4f",
                         variant, gstep + 1, si, st["n_context"], float(loss.detach()))
            gstep += 1
    return model, history


def evaluate_pehe(model, cfg: Dict, n_eval: int = 16, seed_offset: int = 10_000) -> Dict[str, float]:
    p = cfg["pfn"]
    device = cfg.get("device", "cpu")
    prior = NeuroPrior(d_x=p["d_x"], n_context=p["context_max"], n_query=p["n_query"],
                       seed=seed_offset + cfg["seed"])
    batch = to_tensors(prior.sample_batch(n_eval), device=device)
    out = predict_cate(model, batch["Xc"], batch["Tc"], batch["Yc"], batch["Xq"])
    cate_true = batch["mu1"] - batch["mu0"]
    return {"root_pehe": float(root_pehe(out["cate"], cate_true)),
            "prescriptive_accuracy": float(prescriptive_accuracy(out["cate"], cate_true))}


def _steps_to_threshold(history, frac=0.5):
    """A simple convergence measure: steps to reach within frac of the way from
    the first to the best loss."""
    losses = [h["loss"] for h in history]
    first, best = losses[0], min(losses)
    if first <= best:
        return len(losses)
    target = first - frac * (first - best)
    for i, v in enumerate(losses):
        if v <= target:
            return i + 1
    return len(losses)


def run_curriculum_ablation(cfg: Dict, variants=("reference", "two_stage", "one_stage")) -> Dict:
    results = {}
    for v in variants:
        model, hist = run_curriculum(cfg, v)
        ev = evaluate_pehe(model, cfg)
        results[v] = {"steps": len(hist), "final_loss": hist[-1]["loss"],
                      "steps_to_half": _steps_to_threshold(hist),
                      "root_pehe": ev["root_pehe"],
                      "prescriptive_accuracy": ev["prescriptive_accuracy"],
                      "history": hist}
        log.info("E12 %-10s steps=%d  final_loss=%.4f  steps_to_half=%d  root_pehe=%.4f",
                 v, len(hist), hist[-1]["loss"], results[v]["steps_to_half"], ev["root_pehe"])
    if cfg.get("out_dir"):
        os.makedirs(cfg["out_dir"], exist_ok=True)
    return results


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="prototype", choices=["prototype", "full"])
    ap.add_argument("--variants", default="reference,two_stage,one_stage",
                    help="comma-separated subset of variants to compare")
    args = ap.parse_args()
    cfg = prototype_config() if args.mode == "prototype" else full_config()
    res = run_curriculum_ablation(cfg, variants=tuple(args.variants.split(",")))
    print("\nE12 curriculum ablation (lower root-PEHE and fewer steps are better):")
    for v, r in res.items():
        print(f"  {v:10s}  steps={r['steps']:6d}  root_pehe={r['root_pehe']:.4f}  "
              f"prescriptive_accuracy={r['prescriptive_accuracy']:.3f}")
