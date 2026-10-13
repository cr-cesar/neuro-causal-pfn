"""Image channel of a lesion pool: detection, provenance and the summary of a
prior cache.

The Neuro-Prior chain reads one pool of images three times: the encoder turns
them into the covariates X (scripts/export_latents.py), the anatomy cache
turns them into overlap fractions that define which lesions a process can
affect (scripts/build_prior_cache.py), and the virtual trial labels the scored
images with the same overlap rule (giles_replica.label_images). A cohort has
two channels on disk, binary lesion masks and continuous disconnectomes, with
the same basenames, and nothing in the arrays tells them apart once the
overlaps are computed: a mask binarised at 0.5 is itself, a disconnectome
binarised at 0.5 covers a much larger support, and the per-network responsive
fractions differ by about an order of magnitude. The helpers here detect the
channel from the voxel values, name it in the npz files the chain writes, and
compare the two sides so that a cache built from one channel is not paired
with latents of the other without saying so.

Detector rule: a volume is binary when every voxel is exactly 0 or 1 (an
empty volume counts as binary). A set of files is ``"lesion"`` when all the
checked volumes (the first ``N_CHECK`` in the given order, or all when there
are fewer) are binary, and ``"disconnectome"`` otherwise.
"""
import os
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

LESION, DISCONNECTOME = "lesion", "disconnectome"
CHANNELS = (LESION, DISCONNECTOME)
N_CHECK = 50                    # files inspected by detect_channel
RESPONSIVE_THRESHOLD = 0.05     # the virtual trial's hit threshold, for the summaries

# representation of an encoder checkpoint -> channel of the images it encodes
# (early_fusion takes a lesion mask and a disconnectome and is left undecided)
REPRESENTATION_CHANNEL = {LESION: LESION, DISCONNECTOME: DISCONNECTOME}


def is_binary_volume(vol: np.ndarray) -> bool:
    """True when every voxel is exactly 0 or 1."""
    vol = np.asarray(vol)
    return bool(np.all((vol == 0) | (vol == 1)))


def detect_channel(files: Sequence[str], n_check: int = N_CHECK, loader=None) -> str:
    """Channel of a set of NIfTI files from their voxel values: ``"lesion"``
    when the first ``n_check`` volumes (all when fewer) are binary,
    ``"disconnectome"`` otherwise. ``loader`` maps a path to an array
    (default: nibabel get_fdata)."""
    if loader is None:
        import nibabel as nib

        def loader(path):
            return np.asarray(nib.load(path).get_fdata())
    checked = list(files)[: max(int(n_check), 1)]
    if not checked:
        raise ValueError("detect_channel needs at least one file")
    return LESION if all(is_binary_volume(loader(f)) for f in checked) else DISCONNECTOME


def channel_from_basename(path: str) -> Optional[str]:
    """Channel implied by the naming convention of export_latents.default_name:
    ``*_disco.npz`` -> disconnectome, ``*_lesion.npz`` -> lesion, else None."""
    stem = os.path.basename(path)
    stem = stem[:-4] if stem.endswith(".npz") else stem
    if stem.endswith("_disco") or stem == "disco":
        return DISCONNECTOME
    if stem.endswith("_lesion") or stem == "lesion":
        return LESION
    return None


def _scalar(value) -> Optional[str]:
    if value is None:
        return None
    s = str(np.asarray(value).item()) if np.asarray(value).ndim == 0 else str(value)
    return s or None


def latents_channel(meta: Dict, path: str = "") -> Tuple[Optional[str], str]:
    """(channel, how) of a latents npz: the channel of the images it was
    exported from, and the evidence it rests on. ``meta`` holds the npz keys
    (``images_channel``, ``representation`` as export_latents writes them; both
    may be missing in older files). Precedence: the recorded image channel,
    then the encoder's representation (lesion or disconnectome; early_fusion is
    undecided), then the basename convention; (None, reason) when nothing
    decides."""
    ch = _scalar(meta.get("images_channel"))
    if ch in CHANNELS:
        return ch, "images_channel recorded by export_latents"
    rep = _scalar(meta.get("representation"))
    if rep in REPRESENTATION_CHANNEL:
        return REPRESENTATION_CHANNEL[rep], f"encoder representation {rep!r}"
    by_name = channel_from_basename(path) if path else None
    if by_name is not None:
        return by_name, f"basename of {os.path.basename(path)}"
    if rep:
        return None, f"encoder representation {rep!r} does not fix the image channel"
    return None, "no channel metadata and no _disco/_lesion suffix"


def responsive_fractions(overlaps: np.ndarray, threshold: float = RESPONSIVE_THRESHOLD) -> np.ndarray:
    """Per network, the fraction of lesions whose overlap with either
    subnetwork exceeds ``threshold`` (strict, as the prior's hit rule).
    ``overlaps`` is [m, K, 2]; returns [K]."""
    ov = np.asarray(overlaps)
    return (ov > threshold).any(axis=2).mean(axis=0)


def summarize_overlaps(overlaps: np.ndarray, threshold: float = RESPONSIVE_THRESHOLD) -> Dict[str, float]:
    """Hit rate and min / median / max of the per-network responsive
    fractions at ``threshold``; the numbers printed when a cache is built
    and logged when it is loaded."""
    ov = np.asarray(overlaps)
    frac = responsive_fractions(ov, threshold)
    return {"any_hit": float((ov.max(axis=(1, 2)) > threshold).mean()),
            "min": float(frac.min()), "median": float(np.median(frac)), "max": float(frac.max())}


def format_summary(summary: Dict[str, float], threshold: float = RESPONSIVE_THRESHOLD) -> str:
    pct = 100 * threshold
    return (f"{100 * summary['any_hit']:.0f} % hit at least one subnetwork above {pct:g} %; "
            f"per-network responsive fraction min/median/max "
            f"{summary['min']:.3f}/{summary['median']:.3f}/{summary['max']:.3f}")


def cache_channel(cache: Dict) -> Optional[str]:
    """The channel recorded in a prior cache (a dict of its npz keys); None
    for caches written before it was recorded."""
    ch = _scalar(cache["channel"]) if "channel" in cache else None
    return ch if ch in CHANNELS else None


def describe_cache(cache: Dict, threshold: float = RESPONSIVE_THRESHOLD) -> str:
    """One line for the training log: channel, images dir and the responsive
    fractions of a cache (a dict of its npz keys)."""
    ch = cache_channel(cache) or "unrecorded"
    src = _scalar(cache["images_dir"]) if "images_dir" in cache else None
    head = f"prior cache: channel {ch}" + (f" ({src})" if src else "")
    return head + "; " + format_summary(summarize_overlaps(cache["overlaps"], threshold), threshold)
