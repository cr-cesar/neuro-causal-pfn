"""Single source of truth for the on-disk data layout.

Two cohort tiers share one folder structure, and the atlases are common to
both:

    data/
      Trial data/
        lesions/          binary DWI lesion masks (MNI 2mm)
        disconnectomes/   continuous disconnection maps in [0, 1], same id
      Full data/
        lesions/
        disconnectomes/
      atlases/            functional parcellation + Giles subdivisions

``trial`` is the pilot subset used to validate the pipeline end to end and
``full`` is the complete cohort for the real Phase-1 runs. The tier is chosen
with ``--data-tier`` on the entry points or the ``NEUROCAUSAL_DATA_TIER``
environment variable (default: ``full``). Folder-name matching is
case-insensitive ("Trial data" and "trial data" both resolve), and when no
tiered folder exists the resolver falls back to the legacy flat layout
(``data/lesions``, ``data/disconnectomes``) so older checkouts keep working.
"""
import os
from typing import Optional

DATA_DIR = "data"
ATLAS_DIR = os.path.join(DATA_DIR, "atlases")
TIER_DIRS = {"trial": "Trial data", "full": "Full data"}
TIER_ENV_VAR = "NEUROCAUSAL_DATA_TIER"


def current_tier() -> str:
    """The active data tier: NEUROCAUSAL_DATA_TIER, defaulting to 'full'."""
    tier = os.environ.get(TIER_ENV_VAR, "full").strip().lower()
    if tier not in TIER_DIRS:
        raise ValueError(f"{TIER_ENV_VAR}={tier!r}: expected one of {sorted(TIER_DIRS)}")
    return tier


def set_tier(tier: str) -> str:
    """Set the active tier for this process (used by the --data-tier flags)."""
    tier = str(tier).strip().lower()
    if tier not in TIER_DIRS:
        raise ValueError(f"data tier {tier!r}: expected one of {sorted(TIER_DIRS)}")
    os.environ[TIER_ENV_VAR] = tier
    return tier


def tier_dir(tier: Optional[str] = None) -> str:
    """The tier folder, matched case-insensitively against what is on disk.

    Returns the canonical path (e.g. ``data/Full data``) when nothing exists
    yet, so callers can use it in messages and mkdir it.
    """
    if tier is None:
        tier = current_tier()
    else:
        tier = str(tier).strip().lower()
        if tier not in TIER_DIRS:
            raise ValueError(f"data tier {tier!r}: expected one of {sorted(TIER_DIRS)}")
    canonical = os.path.join(DATA_DIR, TIER_DIRS[tier])
    if os.path.isdir(DATA_DIR):
        want = TIER_DIRS[tier].lower()
        for name in sorted(os.listdir(DATA_DIR)):
            cand = os.path.join(DATA_DIR, name)
            if name.lower() == want and os.path.isdir(cand):
                return cand
    return canonical


def _modality_root(kind: str, tier: Optional[str]) -> str:
    tiered = os.path.join(tier_dir(tier), kind)
    if os.path.isdir(tiered):
        return tiered
    # Legacy flat layout: only when the tiered folder does not exist at all,
    # so a half-populated tier is reported as missing rather than silently
    # swapped for the wrong cohort.
    legacy = os.path.join(DATA_DIR, kind)
    if os.path.isdir(legacy):
        return legacy
    return tiered


def lesion_root(tier: Optional[str] = None) -> str:
    """Folder with the binary lesion masks for the active (or given) tier."""
    return _modality_root("lesions", tier)


def disconnectome_root(tier: Optional[str] = None) -> str:
    """Folder with the continuous disconnectome maps for the active (or given) tier."""
    return _modality_root("disconnectomes", tier)
