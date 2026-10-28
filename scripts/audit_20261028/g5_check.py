"""G5 (28 Oct): check, before training, that the configuration the training
script will build from the environment has randomised treatment allocation in
every family, as the scored trial does (BIAS = 0). Aggregates only.

The configuration is built exactly as scripts/train_pfn_myriad.qsub.sh builds
it, from the same environment variables. With --sample N, the prior is also
instantiated (cache and atlas must exist) and N processes per family are drawn:
every virtual-trial process must have bias 0 of the "axis" type (P(T=1) = 0.5
for every lesion) and every theory process gamma 0.

    python scripts/audit_20261028/g5_check.py [--sample 200]
"""
import argparse
import os
import sys

import numpy as np

from neurocausalpfn.train.launch_cfg import cfg_from_env
from neurocausalpfn.train.train_pfn import _as_ranges, _build_prior

# the training script's own builder (28 Oct review: this file had a copy)
build_cfg_from_env = cfg_from_env


def check_config(cfg):
    from neurocausalpfn.prior.intersynth_theory import TheoryHyperPrior
    from neurocausalpfn.prior.neuro_prior import HyperPrior
    from neurocausalpfn.prior.generic_mix import split_generic_hyper

    hg = HyperPrior(**_as_ranges(cfg["prior"].get("hyper_giles", {})))
    ranges, _ = split_generic_hyper(cfg["prior"].get("hyper", {}))
    ht = TheoryHyperPrior(**_as_ranges(ranges))
    w = np.asarray(hg.bias_weights, dtype=float)[: len(hg.bias_types)]
    types = {t: float(p) for t, p in zip(hg.bias_types, w / w.sum())}
    errors = []
    if tuple(hg.bias) != (0.0, 0.0):
        errors.append(f"giles bias range {hg.bias}, expected (0, 0)")
    if any(p > 0 for t, p in types.items() if t != "axis"):
        errors.append(f"giles bias types {types}: only 'axis' gives P(T=1) = 0.5 at bias 0")
    if tuple(ht.gamma) != (0.0, 0.0):
        errors.append(f"theory gamma range {ht.gamma}, expected (0, 0)")
    # 28 Oct review, point 8: the two values G4c and G5 share were printed, not required
    if float(hg.random_partitions) != 1.0:
        errors.append(f"giles random_partitions {hg.random_partitions}, expected 1 (G4c's)")
    if float(cfg["prior"]["hyper"].get("generic_weight", 0.0) or 0.0) != 0.5:
        errors.append(f"generic_weight {cfg['prior']['hyper'].get('generic_weight')}, expected 0.5 (G4c's)")
    print(f"giles: bias {tuple(hg.bias)}, types {types}, p_te {tuple(hg.p_te)}, p_re {tuple(hg.p_re)}, "
          f"label_noise {tuple(hg.label_noise)}, random_partitions {hg.random_partitions}")
    print(f"theory: gamma {tuple(ht.gamma)}; family {cfg['prior'].get('family')}, "
          f"weights {cfg['prior'].get('family_weights')}, generic_weight {cfg['prior']['hyper'].get('generic_weight')}")
    return errors


def check_sample(cfg, n):
    prior, _, _ = _build_prior(cfg)
    base = getattr(prior, "base", prior)
    errors, seen = [], {"giles": 0, "theory": 0}
    for fam in ("giles", "theory"):
        for _ in range(n):
            dgp = base.sample_process(fam)
            seen[fam] += 1
            if fam == "giles":
                if dgp.bias != 0.0 or dgp.bias_type != "axis":
                    errors.append(f"giles process with bias {dgp.bias} type {dgp.bias_type}")
                    break
                p = dgp.propensity(base.centroids[:50], base.volumes[:50], [None] * 50, [None] * 50)
                if not np.allclose(p, 0.5):
                    errors.append("giles process with P(T=1) != 0.5")
                    break
            elif dgp.gamma != 0.0:
                errors.append(f"theory process with gamma {dgp.gamma}")
                break
    print(f"sampled processes: {seen}")
    return errors


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=0, help="also draw N processes per family from the prior")
    a = ap.parse_args()
    cfg = build_cfg_from_env()
    errors = check_config(cfg)
    # G5 differs from G4c (C1 on this code) only in the allocation: no other
    # override of the training script may be set
    left = [k for k in ("CACHE_ANCHOR_WEIGHT", "ITERS", "VARIANT", "BATCH", "CONTEXT_MIN", "AUGMENT",
                        "CONTEXT_REPLACE", "STANDARDIZE", "D_MODEL", "N_LAYERS",
                        "BOTH_ARMS", "CATE_WEIGHT", "HEAD_LO", "HEAD_HI", "N_QUERY") if os.environ.get(k)]
    if left:
        errors.append(f"environment overrides set: {left}")
    expected = {"FAMILY": "mixture", "FAMILY_WEIGHTS": "0.5 0.5", "SEED": "0", "ARCH": "linear",
                "LR_SCHEDULE": "constant", "CONTEXT_POOL": "susceptible", "CONTEXT_MAX": "3000"}
    wrong = {k: os.environ.get(k) for k, v in expected.items() if os.environ.get(k) != v}
    if wrong:
        errors.append(f"launch settings differ from G4c: {wrong}")
    print(f"context: min {cfg['pfn']['context_min']} max {cfg['pfn']['context_max']}, "
          f"replace {cfg['prior'].get('context_replace', True)}, iters {cfg['pfn']['iters']}, "
          f"batch {cfg['pfn']['batch_size']}, lr {cfg['pfn']['lr']}")
    if not errors and a.sample:
        errors += check_sample(cfg, a.sample)
    if errors:
        print("G5 CHECK FAILED: " + "; ".join(errors))
        sys.exit(1)
    print("G5 check ok: randomised allocation in every family")


if __name__ == "__main__":
    main()
