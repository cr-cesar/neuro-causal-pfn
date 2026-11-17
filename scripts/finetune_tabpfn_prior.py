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

``--prior-from-env`` (16 Nov, job D) takes the prior from the launch
environment instead of the options above: the cfg of
``launch_cfg.cfg_from_env``, the one scripts/train_pfn_myriad.qsub.sh trains
and scripts/prior_gate.py checks, handed to the same builder, so the processes
come from the prior object a PFN launch with that environment trains on
(scripts/audit_20261028/g4c_launch.sh with RUN_TRAINER=tabpfn runs it after its
gate; scripts/audit_20261028/tabpfn_final_launch.sh is the final prior's
launch). That mode also writes ``provenance.json`` beside the checkpoint:
commit, launch variables, the pool's md5, the tabpfn version and the starting
checkpoint, the configuration, and ``finished`` once the last step is saved.
In that mode the script refuses what it cannot account for:

- the prior options of the other mode (--cache, --family, --family-weights,
  --prior-hyper, --giles-hyper, --n-query, --context-pool, --atlas-dir): the
  prior is the environment's, so giving one is an error, not ignored;
- a pool without its digest (--pool-md5, checked here against the file) and an
  atlas without the digest the launch computed (--atlas-md5, recorded);
- starting weights it cannot name: without --model-path the package must be
  set to --model-version (default v3.5) and its file for that version must be
  on disk; the file is hashed and handed to TabPFN by path;
- an output folder that holds a finished model of another configuration.
  The same configuration, finished, is left as it is.

The global torch generator is seeded with the seed, and checkpoints written
before the last step are named ``tabpfn_ft.partial.pth``.

The script reads the pool cache and the atlas and nothing else: no fold, no
image, no outcome of a real patient; it scores nothing.

Needs the panel environment (tabpfn >= 9.2, torch >= 2.5) and a GPU. The
weights' licence (Prior Labs, non-commercial) covers research use; the
checkpoint written here contains fine-tuned weights of that model and stays
under the same terms. Memory: n_context 2,000 with 2 estimators fits a 32 GB
card; use --n-estimators 1 or a shorter context on 16 GB.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from functools import partial
from typing import Dict, List, Optional, Tuple

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


def prior_from_env(env=None):
    """The Neuro-Prior of a PFN launch, from its environment (CACHE, SEED,
    FAMILY, FAMILY_WEIGHTS, PRIOR_HYPER, GILES_HYPER, CONTEXT_POOL, ...): the
    cfg scripts/train_pfn_myriad.qsub.sh trains on, built by the function it
    uses and handed to the builder ``run_pfn`` uses. The queries per process
    (``pfn.n_query``, which also sets the smallest slice a process may have)
    and the seed are therefore the launch's, not this script's options."""
    from neurocausalpfn.train.launch_cfg import cfg_from_env
    from neurocausalpfn.train.train_pfn import _build_prior

    cfg = cfg_from_env(env)
    if not cfg["prior"].get("cache"):
        sys.exit("--prior-from-env needs CACHE (the pool) in the environment")
    prior, d_x, ok = _build_prior(cfg)
    if not ok:
        sys.exit("the launch environment did not build a Neuro-Prior")
    return prior, d_x, cfg


def file_md5(path: str, chunk: int = 1 << 20) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def pool_record(cfg: Dict, expected_md5: Optional[str] = None) -> List[Dict]:
    """Path, size and md5 of every pool cache of the cfg; with
    ``expected_md5`` (a full digest or its first characters, as
    final1m_launch.sh's POOL_MD5) a single pool must match it."""
    caches = cfg["prior"]["cache"]
    paths = [caches] if isinstance(caches, str) else list(caches)
    rec = [{"path": p, "bytes": os.path.getsize(p), "md5": file_md5(p)} for p in paths]
    if expected_md5:
        if len(rec) != 1 or not rec[0]["md5"].startswith(expected_md5):
            sys.exit(f"pool md5 {[r['md5'][:12] for r in rec]}, expected {expected_md5}")
    return rec


def _version_key(v) -> str:
    """'v3.5', '3.5', 'V3_5' and the package's enum all read as '3.5'."""
    return str(getattr(v, "value", v)).strip().lower().lstrip("v").replace("_", ".")


def base_checkpoint_record(model_path: Optional[str], expect_version: Optional[str] = None) -> Dict:
    """Which TabPFN weights the run starts from: the package version, the
    model version the package is set to (``TABPFN_MODEL_VERSION`` or its
    default), the version asked for, the package's own file for that version
    as its resolver names it, and the file the run starts from (the
    --model-path file, else that package file) with its size and md5. The
    settings and the resolver differ between tabpfn releases, so a failure to
    read them is recorded here; ``check_start`` decides whether the run may go
    on."""
    from importlib import metadata

    rec: Dict = {"model_path_arg": model_path, "model_version_expected": expect_version}
    try:
        rec["tabpfn"] = metadata.version("tabpfn")
    except metadata.PackageNotFoundError:
        rec["tabpfn"] = None
    try:
        from tabpfn.settings import settings

        v = settings.tabpfn.model_version
        rec["model_version"] = getattr(v, "value", str(v))
    except Exception as e:                      # another release of the package
        rec["settings_error"] = repr(e)
    default = None
    version = expect_version or rec.get("model_version")
    if version:
        try:
            from tabpfn.model_loading import resolve_model_path

            default = str(resolve_model_path(None, "classifier", "v" + _version_key(version))[0][0])
            rec["package_file_for_version"] = os.path.basename(default)
        except Exception as e:                  # another release of the package
            rec["resolve_error"] = repr(e)
    path = model_path or default
    rec["path"] = path
    if path and os.path.isfile(path):
        rec.update(file=os.path.basename(path), bytes=os.path.getsize(path), md5=file_md5(path))
    else:
        rec.update(file=os.path.basename(path) if path else None, md5=None)
    if not model_path:
        rec["version_of_file"] = "the package's file for the version asked for"
    elif rec["md5"] and default and os.path.isfile(default) and file_md5(default) == rec["md5"]:
        rec["version_of_file"] = "identical to the package's file for the version asked for"
    else:
        rec["version_of_file"] = "not checked: a file given by path"
    return rec


def check_start(rec: Dict, expect_md5: Optional[str] = None) -> None:
    """Stop unless the starting weights are named: a file on disk, hashed
    (and equal to ``expect_md5`` when given). Taken from the package, the
    package must also be set to the version asked for. A file given by path
    carries no version this script can read; ``version_of_file`` says so."""
    asked = rec.get("model_version_expected")
    if rec.get("md5") is None:
        sys.exit(f"the starting TabPFN checkpoint is not a file on disk (looked for {rec.get('path')!r}"
                 + (f"; the package's resolver failed: {rec['resolve_error']}" if rec.get("resolve_error") else "")
                 + "): pass --model-path")
    if expect_md5 and not rec["md5"].startswith(expect_md5):
        sys.exit(f"starting checkpoint md5 {rec['md5'][:12]}, expected {expect_md5}")
    if not rec.get("model_path_arg") and asked:
        if "model_version" not in rec:
            sys.exit(f"cannot read the model version the tabpfn package is set to ({rec.get('settings_error')}): "
                     "pass --model-path")
        if _version_key(rec["model_version"]) != _version_key(asked):
            sys.exit(f"the tabpfn package is set to model version {rec['model_version']}, the run asks for {asked}: "
                     f"set TABPFN_MODEL_VERSION={asked} or pass --model-path")


# what a finished checkpoint and a new request must share to be the same run;
# save_every, the paths of the pool and of the starting file are left out
# (their digests are compared)
def run_identity(prov: Dict) -> Dict:
    ident = {"git_commit": prov.get("git_commit"),
             "pool_md5": [p.get("md5") for p in prov.get("pool", [])],
             "atlas_md5": prov.get("atlas_md5_reported_by_launch"),
             "tabpfn_package": prov.get("tabpfn", {}).get("tabpfn"),
             "model_version_expected": prov.get("tabpfn", {}).get("model_version_expected"),
             "start_md5": prov.get("tabpfn", {}).get("md5"),
             "seed": prov.get("cfg", {}).get("seed"), "pfn": prov.get("cfg", {}).get("pfn"),
             "prior": {k: v for k, v in prov.get("cfg", {}).get("prior", {}).items() if k != "cache"}}
    ident.update({f"finetune.{k}": v for k, v in prov.get("finetune", {}).items()
                  if k not in ("save_every", "model_path")})
    return json.loads(json.dumps(ident, default=str))


def finished_checkpoint(out: str, requested: Dict) -> Optional[str]:
    """The finished checkpoint in ``out`` when it is the run now asked for
    (``requested``: the provenance this run would write), None when the folder
    holds no finished model. A finished model of another configuration stops
    the run: it is neither reused nor overwritten."""
    path, prov_path = os.path.join(out, "tabpfn_ft.pth"), os.path.join(out, "provenance.json")
    if not (os.path.isfile(path) and os.path.isfile(prov_path)):
        return None
    with open(prov_path) as f:
        saved = json.load(f)
    if not saved.get("finished"):
        return None
    old, new = run_identity(saved), run_identity(requested)
    diff = [f"{k}: {old.get(k)!r} -> {new.get(k)!r}" for k in sorted(set(old) | set(new)) if old.get(k) != new.get(k)]
    if diff:
        sys.exit(f"{out} holds a finished model of another configuration (saved -> asked): " + "; ".join(diff)
                 + ". Use another output folder, or remove this one.")
    return path


def provenance_record(args, cfg: Dict, pools: List[Dict], base: Dict, step: int, finished: bool) -> Dict:
    """What a checkpoint of the launch-environment mode records beside it: the
    code and launch of ``launch_cfg.provenance`` (commit, dirty flag, launch
    variables, library versions), the pool and atlas digests, the starting
    weights, this script's options and the cfg of the prior."""
    from neurocausalpfn.train.launch_cfg import provenance

    prov = provenance()
    prov.update({"trainer": "scripts/finetune_tabpfn_prior.py --prior-from-env",
                 "pool": pools, "pool_md5_expected": args.pool_md5,
                 "atlas_md5_reported_by_launch": args.atlas_md5,
                 "tabpfn": base, "torch_seed": int(args.seed),
                 "finetune": {k: getattr(args, k) for k in (
                     "steps", "lr", "weight_decay", "grad_clip", "n_context", "n_query", "n_estimators",
                     "processes_per_draw", "query_ratio", "learner", "seed", "save_every", "model_path")},
                 "cfg": {"seed": cfg["seed"], "prior": cfg["prior"],
                         "pfn": {k: cfg["pfn"].get(k) for k in ("n_query", "context_max")}},
                 "steps_done": int(step), "finished": bool(finished)})
    return prov


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
    from torch.optim.lr_scheduler import LambdaLR
    try:                                                    # tabpfn 9.2: the collator lives with the data utilities
        from tabpfn.finetuning.data_util import meta_dataset_collator
    except ImportError:                                     # earlier releases exported it from tabpfn.utils
        from tabpfn.utils import meta_dataset_collator

    os.makedirs(args.out, exist_ok=True)
    device = args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
    args.provenance = None
    start_path = args.model_path
    if args.prior_from_env:
        prior, d_x, cfg = prior_from_env()
        # the seed and the queries per process are the launch's (SEED, pfn.n_query)
        if args.seed is not None and args.seed != int(cfg["seed"]):
            sys.exit(f"--seed {args.seed} differs from the launch's SEED {cfg['seed']}")
        args.seed, args.n_query = int(cfg["seed"]), int(cfg["pfn"]["n_query"])
        args.family = cfg["prior"].get("family", "theory")
        pools = pool_record(cfg, args.pool_md5)
        base = base_checkpoint_record(args.model_path, args.model_version)
        check_start(base, args.model_md5)
        start_path = base["path"]               # the file that was hashed, by path: never the package's silent default
        args.provenance = lambda step, finished: provenance_record(args, cfg, pools, base, step, finished)
        done = finished_checkpoint(args.out, args.provenance(args.steps, True))
        if done:
            print(f"already finished with this configuration, left as it is: {done}", flush=True)
            return done
        print(f"pool md5 {[p['md5'][:12] for p in pools]}, tabpfn {base.get('tabpfn')}, model version "
              f"{base.get('model_version', 'unread')} (asked {args.model_version}), starting weights "
              f"{base['file']} md5 {base['md5'][:12]} ({base['version_of_file']})", flush=True)
        # the global torch generator, per seed (the prior, the outcome draws,
        # TabPFN's random_state, the splits and the loader were seeded already)
        torch.manual_seed(args.seed)
    else:
        args.seed = 0 if args.seed is None else args.seed
        prior, d_x, cfg = prior_from_args(args.cache, args.prior_hyper, args.seed, args.n_context, args.n_query,
                                          context_pool=args.context_pool, atlas_dir=args.atlas_dir,
                                          family=args.family, family_weights=args.family_weights,
                                          giles_hyper_text=args.giles_hyper)
    rng = np.random.default_rng(args.seed)
    print(f"prior: family {args.family}" + (f" {cfg['prior'].get('family_weights')}" if args.family == "mixture" else "")
          + f", giles hyper {cfg['prior'].get('hyper_giles', {})}, d_x={d_x}, context {args.n_context}+{args.n_query}, "
          f"{getattr(prior, 'describe', lambda: '')()}", flush=True)

    kw = dict(device=device, n_estimators=args.n_estimators, random_state=args.seed,
              ignore_pretraining_limits=True, fit_mode="batched")
    if start_path:
        kw["model_path"] = start_path
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
    path = _save(est, args.out, step, history, cfg, args, save_tabpfn_model, finished=True)
    print(f"wrote {path} ({step} steps, {(time.time() - t0) / 60:.1f} min)", flush=True)
    return path


def checkpoint_fields(step: int, cfg: Dict, args) -> Dict:
    """The ``neuro_prior_finetune`` record inside the checkpoint. With the
    prior given by options: those options, as before. With --prior-from-env:
    the cfg's prior, the seed and the commit."""
    rec = {"steps": int(step), "learner": args.learner, "n_context": int(args.n_context)}
    if getattr(args, "prior_from_env", False):
        from neurocausalpfn.train.launch_cfg import _git

        rec.update(prior=cfg["prior"], seed=int(cfg["seed"]), n_query=int(cfg["pfn"]["n_query"]),
                   git_commit=_git("rev-parse", "HEAD"))
    else:
        rec.update(prior_hyper=args.prior_hyper, family=args.family, family_weights=args.family_weights,
                   giles_hyper=args.giles_hyper, cache=os.path.basename(args.cache))
    return rec


def _save(est, out, step, history, cfg, args, save_tabpfn_model, finished: bool = False) -> str:
    env_mode = getattr(args, "provenance", None) is not None
    path, partial = os.path.join(out, "tabpfn_ft.pth"), os.path.join(out, "tabpfn_ft.partial.pth")
    # with --prior-from-env only the last step is written as tabpfn_ft.pth, so
    # that name is never an unfinished model; the other mode keeps one name
    target = partial if env_mode and not finished else path
    save_tabpfn_model(model=est, save_path=target,
                      additional_fields={"neuro_prior_finetune": checkpoint_fields(step, cfg, args)})
    # the options of the other mode stay out of this mode's record, so the
    # file without --prior-from-env has the keys it had before that option
    skip = ("provenance",) if env_mode else ("provenance",) + ENV_MODE_OPTIONS
    with open(os.path.join(out, "history.json"), "w") as f:
        json.dump({"history": history, "config": {k: v for k, v in vars(args).items() if k not in skip},
                   "prior": cfg["prior"]}, f, indent=1)
    if env_mode:
        # written after the checkpoint, so "finished": true means the last step is on disk
        with open(os.path.join(out, "provenance.json"), "w") as f:
            json.dump(args.provenance(step, finished), f, indent=1, default=str)
        if finished and os.path.isfile(partial):
            os.remove(partial)
    return target


# options that belong to one way of giving the prior
PRIOR_OPTIONS = ("cache", "family", "family_weights", "prior_hyper", "giles_hyper", "n_query", "context_pool",
                 "atlas_dir")
ENV_MODE_OPTIONS = ("prior_from_env", "pool_md5", "atlas_md5", "model_version", "model_md5")


def explicit_options(ap, argv) -> set:
    """The destinations of the options written on the command line, whatever
    their values (an option given its default value counts)."""
    probe = argparse.ArgumentParser(add_help=False, allow_abbrev=ap.allow_abbrev)
    for a in ap._actions:
        if not a.option_strings or isinstance(a, argparse._HelpAction):
            continue
        kw = {"dest": a.dest, "default": argparse.SUPPRESS}
        if isinstance(a, argparse._StoreTrueAction):
            kw["action"] = "store_true"
        else:
            kw["nargs"] = a.nargs
        probe.add_argument(*a.option_strings, **kw)
    return set(vars(probe.parse_known_args(argv)[0]))


def check_args(ap, args, argv=None) -> None:
    """--cache or --prior-from-env, not both. With --prior-from-env every
    option that describes the prior is refused, because the prior is the
    environment's: none is read, so none may be given. ``argv`` (the command
    line) tells which options were written; without it an option counts as
    given when its value is not the parser's default."""
    if argv is not None:
        given = explicit_options(ap, argv)
    else:
        given = {k for k, v in vars(args).items() if v != ap.get_default(k)}
    opt = lambda dest: "--" + dest.replace("_", "-")
    if args.prior_from_env:
        clash = [opt(k) for k in PRIOR_OPTIONS if k in given]
        if clash:
            ap.error("--prior-from-env takes the prior from the launch environment and reads none of "
                     f"{' '.join(clash)}: drop them (FAMILY, FAMILY_WEIGHTS, PRIOR_HYPER, GILES_HYPER, "
                     "N_QUERY, CONTEXT_POOL and CACHE of the launch set them)")
        if not args.pool_md5 or not args.atlas_md5:
            ap.error("--prior-from-env needs --pool-md5 (checked against the pool) and --atlas-md5 (the digest "
                     "the launch computed, recorded)")
    else:
        if not args.cache:
            ap.error("--cache is required (or --prior-from-env)")
        clash = [opt(k) for k in ENV_MODE_OPTIONS[1:] if k in given]
        if clash:
            ap.error(f"{' '.join(clash)} need --prior-from-env")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache", default=None, help="anatomy cache of the pool (scripts/build_prior_cache.py)")
    ap.add_argument("--prior-from-env", action="store_true",
                    help="take the prior from the launch environment (launch_cfg.cfg_from_env: CACHE, SEED, FAMILY, "
                         "FAMILY_WEIGHTS, PRIOR_HYPER, GILES_HYPER, ...) instead of --cache and the prior options "
                         "below, and write provenance.json beside the checkpoint")
    ap.add_argument("--pool-md5", default=None,
                    help="with --prior-from-env, required: the pool's md5 (or its first characters) must be this")
    ap.add_argument("--atlas-md5", default=None,
                    help="with --prior-from-env, required: the atlas digest the launch computed, recorded in "
                         "provenance.json (this script does not recompute it)")
    ap.add_argument("--model-version", default="v3.5",
                    help="with --prior-from-env: the TabPFN model version the run starts from. Without --model-path "
                         "the package must be set to it (TABPFN_MODEL_VERSION) and its file for it is used by path")
    ap.add_argument("--model-md5", default=None,
                    help="with --prior-from-env: the starting checkpoint's md5 (or its first characters) must be this")
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
    ap.add_argument("--processes-per-draw", type=int, default=4,
                    help="simulated trials one draw of the prior holds. Each trial is one update (the loader's "
                         "batch is 1), so this is not a number of trials per update")
    ap.add_argument("--n-context", type=int, default=2000)
    ap.add_argument("--n-query", type=int, default=256)
    ap.add_argument("--query-ratio", type=float, default=0.2)
    ap.add_argument("--n-estimators", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=None, help="default 0; with --prior-from-env the launch's SEED")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--log-every", type=int, default=25)
    ap.add_argument("--save-every", type=int, default=500)
    return ap


def main(argv=None):
    ap = build_parser()
    argv = sys.argv[1:] if argv is None else list(argv)
    args = ap.parse_args(argv)
    check_args(ap, args, argv)
    finetune(args)


if __name__ == "__main__":
    main()
