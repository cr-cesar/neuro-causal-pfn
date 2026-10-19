"""Can the single-axis model train at a longer context on one GPU? (Phase 2 audit, question 4)

Builds the reduced configuration's transformer and runs a few training steps
on random batches at each context length and batch size asked, reporting peak
GPU memory and seconds per iteration, or OOM. Random covariates are enough:
the memory and time of a step depend on the context length, the batch and the
model, not on where the batch came from. A few minutes of GPU.

    python scripts/context_memory_probe.py --contexts 1024 2048 3000 4119 --batch-sizes 4 2 \\
        --iters 20 --out outputs/context_probe.csv
"""
import argparse
import csv
import os
import time

import numpy as np
import torch

from neurocausalpfn.train.train_pfn import build_model, reduced_config


def one_cell(cfg, d_x, n_ctx, batch, iters, device, n_query=64):
    p = dict(cfg["pfn"]); p["context_max"] = n_ctx
    model = build_model({**cfg, "pfn": p}, d_x).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=p["lr"])
    rng = np.random.default_rng(0)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    t = []
    try:
        for it in range(iters):
            Xc = torch.as_tensor(rng.normal(size=(batch, n_ctx, d_x)), dtype=torch.float32, device=device)
            Tc = torch.as_tensor(rng.integers(0, 2, (batch, n_ctx)), dtype=torch.float32, device=device)
            Yc = torch.as_tensor(rng.random((batch, n_ctx)), dtype=torch.float32, device=device)
            Xq = torch.as_tensor(rng.normal(size=(batch, n_query, d_x)), dtype=torch.float32, device=device)
            Tq = torch.as_tensor(rng.integers(0, 2, (batch, n_query)), dtype=torch.float32, device=device)
            mu = torch.as_tensor(rng.random((batch, n_query)), dtype=torch.float32, device=device)
            t0 = time.time()
            logits = model(Xc, Tc, Yc, Xq, Tq)
            loss = model.head.loss(logits, mu)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            if it >= 2:                       # skip warm-up iterations
                t.append(time.time() - t0)
        peak = torch.cuda.max_memory_allocated(device) / 2 ** 30 if device.type == "cuda" else float("nan")
        return {"context": n_ctx, "batch": batch, "status": "ok", "peak_gb": round(peak, 2),
                "s_per_iter": round(float(np.mean(t)), 3) if t else float("nan")}
    except RuntimeError as e:                 # CUDA out of memory
        if "out of memory" not in str(e).lower():
            raise
        return {"context": n_ctx, "batch": batch, "status": "OOM", "peak_gb": float("nan"),
                "s_per_iter": float("nan")}
    finally:
        del model, opt
        if device.type == "cuda":
            torch.cuda.empty_cache()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--contexts", type=int, nargs="+", default=[1024, 2048, 3000, 4119])
    ap.add_argument("--batch-sizes", type=int, nargs="+", default=[4, 2, 1])
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--d-x", type=int, default=50)
    ap.add_argument("--arch", default="linear", choices=["linear", "tabicl"])
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    device = torch.device(("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device)
    cfg = reduced_config(); cfg["pfn"]["arch"] = args.arch
    name = torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu"
    total = torch.cuda.get_device_properties(device).total_memory / 2 ** 30 if device.type == "cuda" else float("nan")
    print(f"device {name} ({total:.0f} GB), arch {args.arch}, d_x {args.d_x}, {args.iters} iterations per cell")
    rows = []
    for n_ctx in args.contexts:
        for batch in args.batch_sizes:
            r = one_cell(cfg, args.d_x, n_ctx, batch, args.iters, device)
            r.update({"device": name, "arch": args.arch})
            rows.append(r)
            print(f"context {n_ctx:5d} batch {batch}: {r['status']:3s} peak {r['peak_gb']} GB, {r['s_per_iter']} s/iter", flush=True)
            if r["status"] == "ok":
                break                         # the largest batch that fits is enough per context
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader(); w.writerows(rows)
        print("wrote", args.out)
    return rows


if __name__ == "__main__":
    main()
