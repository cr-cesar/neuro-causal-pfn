"""E3: latent dimensionality sweep over the separate-encoder design.

The lesion and the disconnectome are encoded by separate VAEs whose latents are
concatenated (the E6 separate-fusion design). E3 sweeps the per-modality latent
size, both symmetric ({25, 50, 75, 100} on each side) and asymmetric (the same
total of 100 split unevenly, e.g. 75+25), to give the first data-driven answer to
the dimensionality question rather than inheriting the arbitrary 50 (Giles) or
100 (Rondina). Each grid point trains the two VAEs at their assigned sizes and
records the validation reconstruction; this orchestrates train_vae, it does not
duplicate it.
"""
import copy
import os
from typing import Dict, List, Tuple

from ..utils.logging_utils import get_logger
from .train_vae import full_config as vae_full_config
from .train_vae import prototype_config as vae_prototype_config
from .train_vae import run_vae

log = get_logger()


def e3_grid(asymmetric: bool = True) -> List[Tuple[int, int]]:
    """The (d_lesion, d_disconnectome) pairs of the E3 sweep."""
    symmetric = [(25, 25), (50, 50), (75, 75), (100, 100)]
    asym = [(75, 25), (60, 40), (40, 60), (25, 75)]
    return symmetric + (asym if asymmetric else [])


def prototype_config() -> Dict:
    cfg = vae_prototype_config()
    cfg["out_dir"] = "outputs/e3_prototype"
    return cfg


def full_config() -> Dict:
    cfg = vae_full_config()
    cfg["out_dir"] = "outputs/e3_full"
    return cfg


def run_dim_sweep(base_cfg: Dict, grid: List[Tuple[int, int]] = None) -> List[Dict]:
    grid = grid or e3_grid()
    root_out = base_cfg.get("out_dir", "outputs/e3")
    results = []
    for d_les, d_dis in grid:
        rec = {"d_lesion": d_les, "d_disco": d_dis, "total_dim": d_les + d_dis}
        for rep, d, key in (("lesion", d_les, "val_lesion"), ("disconnectome", d_dis, "val_disco")):
            cfg = copy.deepcopy(base_cfg)
            cfg["representation"] = rep
            cfg["vae"]["zdim"] = d
            cfg["export"] = False
            cfg["out_dir"] = os.path.join(root_out, f"{rep}_d{d}")
            if rep == "disconnectome" and cfg["data"].get("root") == "data/lesions":
                cfg["data"]["root"] = "data/disconnectomes"
            _, hist = run_vae(cfg)
            rec[key] = hist[-1]["val_total"]
        results.append(rec)
        log.info("E3 point: d_lesion=%d d_disco=%d total=%d val_lesion=%.4f val_disco=%.4f",
                 d_les, d_dis, rec["total_dim"], rec["val_lesion"], rec["val_disco"])
    return results


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="prototype", choices=["prototype", "full"])
    ap.add_argument("--no-asymmetric", action="store_true", help="symmetric grid only")
    args = ap.parse_args()
    cfg = prototype_config() if args.mode == "prototype" else full_config()
    run_dim_sweep(cfg, grid=e3_grid(asymmetric=not args.no_asymmetric))
