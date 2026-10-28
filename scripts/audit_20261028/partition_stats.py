"""T2: virtual partitions of C1's giles family vs the 32 real portions (aggregates only, CPU).

Per giles process of the prior of a checkpoint (random partitions, generic part
off so every draw is a virtual-trial process), and per real portion (16
networks x receptor/genetics halves) on the same cache pool:

- null_frac: share of susceptible lesions (above threshold on any side) that
  hit both sides, i.e. CATE 0 in the virtual trial
- size_ratio_lesions: smaller / larger number of lesions hitting only one side
- size_ratio_voxels: smaller / larger side volume in the atlas
- dice_voxels: Dice of the two sides' voxel masks (spatial overlap)
- jaccard_lesions: lesions hitting both / lesions hitting either

A process with several causal networks reports the mean over its networks for
the side statistics; null_frac uses the DGP's own susceptibility rule.

    python partition_stats.py <pfn.pt> --processes 400 --out <dir>
"""
import argparse, copy, json, os
import numpy as np
import torch

from neurocausalpfn.train.train_pfn import _build_prior
from neurocausalpfn.prior.atlas import FunctionalAtlas


def side_mask(atoms, masks):
    m = None
    for ch, k, h in atoms:
        a, b = masks["genetics" if ch == "genetics" else "receptor"][k]
        x = (a | b) if h == -1 else (a if h == 0 else b)
        m = x.copy() if m is None else (m | x)
    return m


def side_stats(oa, ob, thr, ma, mb):
    ha, hb = oa > thr, ob > thr
    na, nb, both = int((ha & ~hb).sum()), int((hb & ~ha).sum()), int((ha & hb).sum())
    anyh = int((ha | hb).sum())
    va, vb = int(ma.sum()), int(mb.sum())
    return {"null_frac": both / anyh if anyh else np.nan,
            "size_ratio_lesions": min(na, nb) / max(na, nb) if max(na, nb) else np.nan,
            "size_ratio_voxels": min(va, vb) / max(va, vb) if max(va, vb) else np.nan,
            "dice_voxels": 2 * int((ma & mb).sum()) / (va + vb) if va + vb else np.nan,
            "jaccard_lesions": both / anyh if anyh else np.nan, "n_susceptible": anyh}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt"); ap.add_argument("--processes", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--real-threshold", type=float, default=0.05)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)

    cfg = json.loads(json.dumps(torch.load(a.ckpt, map_location="cpu", weights_only=False)["cfg"]))
    cfg["seed"] = a.seed; cfg["prior"]["family_weights"] = [0.0, 1.0]
    cfg["prior"]["hyper"]["generic_weight"] = 0.0
    prior, _, _ = _build_prior(cfg); base = getattr(prior, "base", prior)

    cache = np.load(cfg["prior"]["cache"], allow_pickle=True)
    nets = [int(n) for n in cache["networks"]]
    shape = tuple(cfg["prior"].get("atlas_shape", [48, 56, 48]))
    masks = {}
    for mod in ("receptor", "genetics"):
        at = FunctionalAtlas.from_dir(cfg["prior"]["atlas_dir"], shape=shape, seed=0, modality=mod)
        masks[mod] = [tuple(np.asarray(s, bool) for s in at.subnetworks[n]) for n in nets]

    rows = []
    for p in range(a.processes):
        fam, dgp, sl = base._draw_process()
        ov = base._giles_ov(dgp); desc = getattr(dgp, "virtual_desc", None)
        if fam != "giles" or desc is None:
            continue
        per = []
        for j, n in enumerate(dgp.networks):
            i = dgp.atlas.networks.index(n)
            per.append(side_stats(ov[:, i, 0], ov[:, i, 1], dgp.threshold,
                                  side_mask(desc[j]["A"], masks), side_mask(desc[j]["B"], masks)))
        sus = [dgp.susceptibility(ov[r]) for r in range(ov.shape[0])]
        s_vals = [s for _, s in sus if s is not None]
        r = {k: float(np.nanmean([q[k] for q in per])) for k in per[0]}
        r["null_frac"] = float(np.mean([s == dgp.both_hit_value for s in s_vals])) if s_vals else np.nan
        r.update(kind="virtual", n_networks=len(dgp.networks), threshold=dgp.threshold,
                 n_susceptible=len(s_vals), atoms_per_side=float(np.mean([len(d["A"]) + len(d["B"]) for d in desc])) / 2)
        rows.append(r)

    for mod in ("receptor", "genetics"):
        ov = cache[f"overlaps_{mod}"]
        for k, n in enumerate(nets):
            ma, mb = masks[mod][k]
            r = side_stats(ov[:, k, 0], ov[:, k, 1], a.real_threshold, ma, mb)
            r.update(kind=f"real_{mod}", n_networks=1, threshold=a.real_threshold, atoms_per_side=1.0)
            rows.append(r)

    import pandas as pd
    d = pd.DataFrame(rows)
    d.to_csv(os.path.join(a.out, "partition_stats_rows.csv"), index=False)
    cols = ["null_frac", "size_ratio_lesions", "size_ratio_voxels", "dice_voxels", "jaccard_lesions", "n_susceptible"]
    d["group"] = np.where(d.kind == "virtual", "virtual", "real_32")
    summ = d.groupby("group")[cols].quantile([.1, .5, .9]).unstack().round(3)
    summ.columns = [f"{c}_q{int(q*100)}" for c, q in summ.columns]
    summ.insert(0, "n", d.groupby("group").size())
    summ.to_csv(os.path.join(a.out, "partition_stats_summary.csv"))
    pd.set_option("display.width", 250)
    for c in cols:
        print(c, summ[[f"{c}_q10", f"{c}_q50", f"{c}_q90"]].to_string().replace("\n", " | "))
    print("n", summ["n"].to_dict(), "| virtual n_networks", d[d.kind == "virtual"].n_networks.value_counts().to_dict())


if __name__ == "__main__":
    main()
