"""The training configuration of a launch, built from its environment, and the
provenance a checkpoint carries.

scripts/train_pfn_myriad.qsub.sh, the pre-launch checks and the prior gate
build the configuration with ``cfg_from_env``, so the three cannot drift
apart (28 Oct review: each had its own copy). ``provenance`` is what the
checkpoint records about the code and the launch: the commit and whether the
tree was dirty, every launch variable that was set, and the library versions.
"""
import os
import platform
import subprocess
from typing import Dict, Mapping, Optional

# the variables scripts/train_pfn_myriad.qsub.sh reads (and the defaults it
# gives them); anything else in the environment does not reach the cfg
LAUNCH_VARS = ("CACHE", "CACHE_ANCHOR_WEIGHT", "OUT", "ITERS", "SEED", "ARCH", "VARIANT", "FAMILY",
               "FAMILY_WEIGHTS", "CONTEXT_POOL", "PRIOR_HYPER", "GILES_HYPER", "LR_SCHEDULE", "BATCH",
               "CONTEXT_MIN", "CONTEXT_MAX", "AUGMENT", "CONTEXT_REPLACE", "STANDARDIZE", "D_MODEL",
               "N_LAYERS", "BOTH_ARMS", "CATE_WEIGHT", "HEAD_LO", "HEAD_HI", "N_QUERY", "CURRICULUM")
# scoring-time switches of pfn/estimator.py: not part of the checkpoint, so
# the scoring logs record them (estimator.scoring_settings)
SCORING_VARS = ("PFN_CONTEXT_CAP", "PFN_CONTEXT_SELECT", "PFN_CONTEXT_K", "PFN_CONTEXT_ARM_CAP",
                "PFN_CONTEXT_REPEAT", "PFN_ROTATIONS", "PFN_ROTATIONS_SEED", "PFN_ARM_FLIP")


def cfg_from_env(env: Optional[Mapping[str, str]] = None) -> Dict:
    """The cfg scripts/train_pfn_myriad.qsub.sh trains, from its variables."""
    from .train_pfn import e12_config, parse_hyper_opts, reduced_config, with_family

    env = os.environ if env is None else env
    get = lambda k, d="": (env.get(k) or d)
    fam = get("FAMILY") or "theory"
    w = [float(x) for x in get("FAMILY_WEIGHTS").split()] or None
    cp = get("CONTEXT_POOL", "susceptible")
    variant = get("VARIANT")
    cfg = e12_config(variant, fam, w, cp) if variant else with_family(reduced_config(), fam, w, cp)
    cfg["out_dir"] = get("OUT", "outputs/pfn_reduced"); cfg["seed"] = int(get("SEED", "0"))
    cfg["pfn"]["iters"] = int(get("ITERS", "20000")); cfg["pfn"]["arch"] = get("ARCH", "tabicl")
    cfg["pfn"]["lr_schedule"] = get("LR_SCHEDULE", "constant")
    if get("BATCH"): cfg["pfn"]["batch_size"] = int(get("BATCH"))
    if get("CONTEXT_MIN"): cfg["pfn"]["context_min"] = int(get("CONTEXT_MIN"))
    if get("CONTEXT_MAX"): cfg["pfn"]["context_max"] = int(get("CONTEXT_MAX"))
    if get("N_QUERY"): cfg["pfn"]["n_query"] = int(get("N_QUERY"))
    if get("CURRICULUM"): cfg["pfn"]["curriculum"] = get("CURRICULUM")    # context | none | loguniform
    if get("AUGMENT"): cfg["prior"]["augment"] = bool(int(get("AUGMENT")))
    if get("CONTEXT_REPLACE"): cfg["prior"]["context_replace"] = bool(int(get("CONTEXT_REPLACE")))
    if get("STANDARDIZE"): cfg["prior"]["standardize"] = get("STANDARDIZE")
    if get("D_MODEL"): cfg["pfn"]["d_model"] = int(get("D_MODEL"))
    if get("N_LAYERS"): cfg["pfn"]["n_layers"] = int(get("N_LAYERS"))
    # 28 Oct: both-arm targets, the effect term and the outcome-bin range
    if get("BOTH_ARMS"): cfg["pfn"]["both_arms"] = bool(int(get("BOTH_ARMS")))
    if get("CATE_WEIGHT"): cfg["pfn"]["cate_weight"] = float(get("CATE_WEIGHT"))
    if get("HEAD_LO"): cfg["pfn"]["head_lo"] = float(get("HEAD_LO"))
    if get("HEAD_HI"): cfg["pfn"]["head_hi"] = float(get("HEAD_HI"))
    caches = get("CACHE").split()
    cfg["prior"]["cache"] = (caches[0] if len(caches) == 1 else caches) if caches else ""
    if get("CACHE_ANCHOR_WEIGHT"): cfg["prior"]["cache_anchor_weight"] = float(get("CACHE_ANCHOR_WEIGHT"))
    cfg["prior"]["atlas_dir"] = "data/atlases"
    cfg["prior"]["hyper"] = {**cfg["prior"].get("hyper", {}), **parse_hyper_opts(get("PRIOR_HYPER"))}
    if get("GILES_HYPER"):
        cfg["prior"]["hyper_giles"] = {**cfg["prior"].get("hyper_giles", {}), **parse_hyper_opts(get("GILES_HYPER"))}
    return cfg


def _git(*args) -> Optional[str]:
    """Output of a git command in this file's checkout; None when git fails
    (an empty output, e.g. the status of a clean tree, is "", not None)."""
    try:
        r = subprocess.run(["git", *args], capture_output=True, text=True, timeout=10,
                           cwd=os.path.dirname(os.path.abspath(__file__)))
    except (OSError, subprocess.SubprocessError):
        return None
    return r.stdout.strip() if r.returncode == 0 else None


def provenance(env: Optional[Mapping[str, str]] = None) -> Dict:
    """Code, launch and runtime of this process (no data, no paths beyond the
    launch variables themselves)."""
    import numpy as np
    import torch

    env = os.environ if env is None else env
    status = _git("status", "--porcelain", "--untracked-files=no")
    return {"git_commit": _git("rev-parse", "HEAD") or None,
            "git_dirty": None if status is None else bool(status),
            "launch_env": {k: env[k] for k in LAUNCH_VARS if env.get(k)},
            "host": platform.node(), "job_id": env.get("JOB_ID"),
            "python": platform.python_version(), "torch": torch.__version__, "numpy": np.__version__,
            "cuda": torch.version.cuda}
