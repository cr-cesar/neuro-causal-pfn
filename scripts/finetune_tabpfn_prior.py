"""Fine-tune TabPFN on processes of the Neuro-Prior (route A, 26 Oct).

The model that wins the benchmark, TabPFN, was pre-trained on a far larger
synthetic corpus than ours and reads binary outcomes natively; our
transformer, trained from scratch on one family of mechanisms, learned that
family and nothing wider. This script continues TabPFN's training on
synthetic neuro-trials staged on real lesion anatomy: every step draws
processes from the configured Neuro-Prior (anatomy family, generic effect
processes, or their mixture), turns each into one tabular dataset (features =
the latents plus the treatment column for the S-learner, or one arm at a time
for the T-learner; target = the binary outcome), and takes one gradient step
of TabPFN's own fine-tuning path (``fit_from_preprocessed`` + ``forward`` with
raw logits, cross-entropy on the query rows). The result is a TabPFN
checkpoint that the panel loads through ``TABPFN_MODEL_PATH`` and scores
exactly like the untuned one, so the gain of the prior is read on one scale.

    python scripts/finetune_tabpfn_prior.py --cache outputs/prior_cache/fold_pools/<pool>/fold0.npz \\
        --family mixture --family-weights 0.5 0.5 \\
        --giles-hyper "random_partitions=1 threshold=0.03:0.08 p_te=0.5:1" \\
        --prior-hyper "two_arms=1 effect_scale=0.2:1 d_source=network s_source=genetics_1 \\
                       s_b_source=genetics_2 effect_headroom=1 generic_weight=0.5" \\
        --steps 2000 --n-context 2000 --out outputs/tabpfn_ft/bcb_thr03

Needs the panel environment (tabpfn >= 9.2, torch >= 2.5) and a GPU. The
weights' licence (Prior Labs, non-commercial) covers research use; the
checkpoint written here contains fine-tuned weights of that model and stays
under the same terms. Memory: n_context 2,000 with 2 estimators fits a 32 GB
card; use --n-estimators 1 or a shorter context on 16 GB.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from functools import partial
from typing import Dict, List, Tuple

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "src"))


def prior_from_args(cache: str, hyper_text: str, seed: int, n_context: int, n_query: int,
                    context_pool: str = "susceptible", atlas_dir: str = "data/atlases",
                    family: str = "theory", family_weights=None, giles_hyper_text: str = ""):
    """The Neuro-Prior of the training jobs (same config keys as
    scripts/train_pfn_myriad.qsub.sh: FAMILY, FAMILY_WEIGHTS, PRIOR_HYPER,
    GILES_HYPER), built once."""
    from neurocausalpfn.train.train_pfn import _build_prior, parse_hyper_opts

    prior_cfg = {"kind": "neuro_prior", "family": family, "cache": cache, "atlas_dir": atlas_dir,
                 "hyper": parse_hyper_opts(hyper_text), "context_pool": context_pool,
                 "context_replace": True, "standardize": "pool", "augment": True}
    if family == "mixture":
        w = [float(x) for x in (family_weights or [0.5, 0.5])]
        if len(w) != 2 or min(w) < 0 or sum(w) <= 0:
            sys.exit("--family-weights takes two non-negative numbers: theory giles")
        prior_cfg["family_weights"] = [w[0] / sum(w), w[1] / sum(w)]
    if giles_hyper_text:
        prior_cfg["hyper_giles"] = parse_hyper_opts(giles_hyper_text)
    cfg = {"seed": int(seed),
           "pfn": {"context_max": int(n_context), "n_query": int(n_query), "d_x": 0},
           "prior": prior_cfg}
    prior, d_x, ok = _build_prior(cfg)
    if not ok:
        sys.exit("the configuration did not build a Neuro-Prior")
    return prior, d_x, cfg


def datasets_from_batch(b: Dict, rng: np.random.Generator, learner: str = "s") -> List[Tuple[np.ndarray, np.ndarray]]:
    """One tabular dataset per process of a prior batch: rows = context and
    query patients, target = binary outcome (Bernoulli of the process's
    probability for the arm the patient is in). S-learner: the treatment is a
    feature column; T-learner: one dataset per arm, latents only."""
    out = []
    B = len(b["Xc"])
    for i in range(B):
        Xc, Tc, Yc, Xq, Tq, muq = (np.asarray(b[k][i]) for k in ("Xc", "Tc", "Yc", "Xq", "Tq", "mu_q"))
        yc = Yc if set(np.unique(Yc)) <= {0.0, 1.0} else (rng.uniform(size=len(Yc)) < Yc).astype(float)
        yq = (rng.uniform(size=len(muq)) < muq).astype(float)
        X = np.vstack([Xc, Xq]).astype(np.float32)
        T = np.concatenate([Tc, Tq]).astype(np.float32)
        y = np.concatenate([yc, yq]).astype(int)
        if learner == "s":
            out.append((np.column_stack([X, T]), y))
        else:
            for w in (0, 1):
                m = T == w
                if m.sum() >= 8 and len(np.unique(y[m])) == 2:
                    out.append((X[m], y[m]))
    return [(X, y) for X, y in out if len(np.unique(y)) == 2]


def ce_loss_from_raw_logits(logits_QBEL, y_query, device):
    """TabPFN's fine-tuning loss: raw logits (queries, batch=1, estimators,
    classes) against the query targets, the estimator axis treated as batch."""
    import torch

    Q, B, E, L = logits_QBEL.shape
    logits = logits_QBEL.permute(1, 2, 3, 0).reshape(B * E, L, Q)
    targets = torch.as_tensor(np.asarray(y_query), device=device).long().reshape(1, Q).repeat(B * E, 1)
    return torch.nn.functional.cross_entropy(logits, targets)


def finetune(args) -> str:
    import torch
    from torch.utils.data import DataLoader

    from tabpfn import TabPFNClassifier
    from tabpfn.architectures.interface import PerformanceOptions
    from tabpfn.finetuning._torch_compat import GradScaler, autocast, sdpa_kernel_context
    from tabpfn.finetuning.data_util import get_preprocessed_dataset_chunks
    from tabpfn.finetuning.finetuned_base import _parameters_unused_by_task, _ratio_ctx_query_split
    from tabpfn.finetuning.train_util import get_and_init_optimizer, get_cosine_schedule_with_warmup
    from tabpfn.model_loading import save_tabpfn_model
    from tabpfn.utils import meta_dataset_collator
    from torch.optim.lr_scheduler import LambdaLR

    os.makedirs(args.out, exist_ok=True)
    device = args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
    rng = np.random.default_rng(args.seed)
    prior, d_x, cfg = prior_from_args(args.cache, args.prior_hyper, args.seed, args.n_context, args.n_query,
                                      context_pool=args.context_pool, atlas_dir=args.atlas_dir,
                                      family=args.family, family_weights=args.family_weights,
                                      giles_hyper_text=args.giles_hyper)
    print(f"prior: family {args.family}" + (f" {cfg['prior'].get('family_weights')}" if args.family == "mixture" else "")
          + f", giles hyper {cfg['prior'].get('hyper_giles', {})}, d_x={d_x}, context {args.n_context}+{args.n_query}, "
          f"{getattr(prior, 'describe', lambda: '')()}", flush=True)

    kw = dict(device=device, n_estimators=args.n_estimators, random_state=args.seed,
              ignore_pretraining_limits=True, fit_mode="batched")
    if args.model_path:
        kw["model_path"] = args.model_path
    est = TabPFNClassifier(**kw)
    est._initialize_model_variables()
    model = est.model_.to(device)
    for p_ in _parameters_unused_by_task(model, "multiclass"):
        p_.requires_grad = False
    n_train = sum(p_.numel() for p_ in model.parameters() if p_.requires_grad)
    print(f"tabpfn model: {type(model).__name__}, {n_train / 1e6:.1f}M trainable parameters, device {device}", flush=True)
    optimizer = get_and_init_optimizer(model.parameters(), args.lr, args.weight_decay, None, device)
    total = args.steps
    scheduler = LambdaLR(optimizer, lr_lambda=get_cosine_schedule_with_warmup(
        total_steps=total, warmup_steps=max(1, int(0.1 * total)), warmup_only=False))
    perf = PerformanceOptions(force_recompute_layer=True, use_chunkwise_inference=False)
    use_amp = device.startswith("cuda")
    scaler = GradScaler() if use_amp else None
    split_fn = partial(_ratio_ctx_query_split, query_ratio=args.query_ratio, min_query_size=2,
                       random_state=args.seed)

    history = []
    step, t0 = 0, time.time()
    model.train()
    while step < total:
        b = prior.sample_batch(args.processes_per_draw, n_context=args.n_context)
        sets = datasets_from_batch(b, rng, learner=args.learner)
        if not sets:
            continue
        ds = get_preprocessed_dataset_chunks(
            calling_instance=est, X_raw=[X for X, _ in sets], y_raw=[y for _, y in sets], split_fn=split_fn,
            max_data_size=None, model_type="classifier", equal_split_size=False,
            data_shuffle_seed=args.seed + step, preprocessing_random_state=args.seed)
        loader = DataLoader(ds, batch_size=1, collate_fn=meta_dataset_collator, shuffle=True,
                            generator=torch.Generator().manual_seed(args.seed + step))
        for batch in loader:
            if step >= total:
                break
            # a context must show both classes, else the model has nothing to read
            if any(len(torch.unique(yc)) < 2 for yc in batch.y_context):
                continue
            optimizer.zero_grad()
            est.fit_from_preprocessed(batch.X_context, batch.y_context, batch.cat_indices, batch.configs,
                                      performance_options=perf)
            with autocast(enabled=use_amp), sdpa_kernel_context():
                logits = est.forward(batch.X_query, return_raw_logits=True)
                loss = ce_loss_from_raw_logits(logits, batch.y_query, device)
            if scaler is not None:
                with sdpa_kernel_context():
                    scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                scaler.step(optimizer); scaler.update()
            else:
                with sdpa_kernel_context():
                    loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()
            scheduler.step()
            step += 1
            history.append({"step": step, "loss": float(loss.item()), "lr": float(scheduler.get_last_lr()[0])})
            if step % args.log_every == 0 or step == total:
                recent = np.mean([h["loss"] for h in history[-args.log_every:]])
                print(f"step {step}/{total}  loss {recent:.4f}  lr {history[-1]['lr']:.2e}  "
                      f"({(time.time() - t0) / 60:.1f} min)", flush=True)
            if args.save_every and step % args.save_every == 0 and step < total:
                _save(est, args.out, step, history, cfg, args, save_tabpfn_model)
    path = _save(est, args.out, step, history, cfg, args, save_tabpfn_model)
    print(f"wrote {path} ({step} steps, {(time.time() - t0) / 60:.1f} min)", flush=True)
    return path


def _save(est, out, step, history, cfg, args, save_tabpfn_model) -> str:
    path = os.path.join(out, "tabpfn_ft.pth")
    save_tabpfn_model(model=est, save_path=path,
                      additional_fields={"neuro_prior_finetune": {"steps": int(step), "learner": args.learner,
                                                                  "n_context": int(args.n_context),
                                                                  "prior_hyper": args.prior_hyper,
                                                                  "family": args.family,
                                                                  "family_weights": args.family_weights,
                                                                  "giles_hyper": args.giles_hyper,
                                                                  "cache": os.path.basename(args.cache)}})
    with open(os.path.join(out, "history.json"), "w") as f:
        json.dump({"history": history, "config": {k: v for k, v in vars(args).items()},
                   "prior": cfg["prior"]}, f, indent=1)
    return path


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache", required=True, help="anatomy cache of the pool (scripts/build_prior_cache.py)")
    ap.add_argument("--prior-hyper", default="two_arms=1 effect_scale=0.2:1 d_source=network s_source=genetics_1 "
                                            "s_b_source=genetics_2 effect_headroom=1 generic_weight=0.5")
    ap.add_argument("--family", default="theory", choices=["theory", "giles", "mixture"],
                    help="Neuro-Prior family, as FAMILY of the training jobs")
    ap.add_argument("--family-weights", nargs=2, type=float, default=None, metavar=("THEORY", "GILES"),
                    help="mixture weights, as FAMILY_WEIGHTS")
    ap.add_argument("--giles-hyper", default="",
                    help="virtual-trial family ranges, as GILES_HYPER, e.g. 'random_partitions=1 threshold=0.03:0.08'")
    ap.add_argument("--atlas-dir", default="data/atlases")
    ap.add_argument("--context-pool", default="susceptible", choices=["susceptible", "all"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--model-path", default=None, help="start from this TabPFN checkpoint (default: the package's)")
    ap.add_argument("--learner", default="s", choices=["s", "t"],
                    help="s: treatment as a feature column (one dataset per process); t: one dataset per arm")
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--processes-per-draw", type=int, default=4)
    ap.add_argument("--n-context", type=int, default=2000)
    ap.add_argument("--n-query", type=int, default=256)
    ap.add_argument("--query-ratio", type=float, default=0.2)
    ap.add_argument("--n-estimators", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--log-every", type=int, default=25)
    ap.add_argument("--save-every", type=int, default=500)
    args = ap.parse_args(argv)
    finetune(args)


if __name__ == "__main__":
    main()
