import json, sys
import numpy as np, torch
from neurocausalpfn.train.train_pfn import _build_prior

def cfg_of(ckpt):
    c = torch.load(ckpt, map_location="cpu", weights_only=False)["cfg"]
    return json.loads(json.dumps(c))

def measure(cfg, family, weights=None, n_proc=400, n_ctx=1024, seed=0):
    c = json.loads(json.dumps(cfg)); c["seed"] = seed
    c["prior"]["family"] = family
    if weights is not None:
        c["prior"]["family_weights"] = list(weights)
    prior, _, _ = _build_prior(c)
    med, q95, small, big, nz = [], [], [], [], []
    for _ in range(n_proc):
        b = prior.sample_batch(1, n_context=n_ctx)
        tau = np.abs(np.asarray(b["mu1"][0]) - np.asarray(b["mu0"][0])).ravel()
        med.append(np.median(tau)); q95.append(np.quantile(tau, 0.95))
        small.append((tau < 0.05).mean()); big.append((tau > 0.5).mean()); nz.append((tau > 1e-9).mean())
    print(f"{family:8s} w={weights} shape_mu1={np.shape(b['mu1'])} | median|tau| {np.mean(med):.3f} "
          f"(median over processes {np.median(med):.3f}) | q95 {np.mean(q95):.3f} | "
          f"frac<0.05 {np.mean(small):.3f} | frac>0.5 {np.mean(big):.3f} | frac nonzero {np.mean(nz):.3f}")

r7 = cfg_of(sys.argv[1])   # checkpoint de r7 uclhkch-s0 fold0
c1 = cfg_of(sys.argv[2])   # checkpoint de C1 fold0
measure(r7, "theory")
measure(c1, "mixture", weights=(1, 0))   # parte theory de C1
measure(c1, "mixture", weights=(0, 1))   # parte giles de C1
