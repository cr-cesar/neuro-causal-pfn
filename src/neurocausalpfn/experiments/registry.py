"""The declarative catalogue of Table 9 (Planned experiments, baselines, and
evaluation criteria).

Each :class:`Experiment` is one row of Table 9. The table is the authoritative
Phase-1 roadmap: every row changes one design element relative to the E1
baseline and records the arm it belongs to, the variant tested and the
evaluation tiers used to judge it (section 14, "How to read this table").

The catalogue is pure data so it can be imported and asserted on without any
heavy dependency. The mapping from an entry to a concrete training run lives in
:mod:`neurocausalpfn.experiments.runner`; the evaluation gates live in
:mod:`neurocausalpfn.experiments.tiers`.

Reconciliation with the implemented code (section 22 / Table 15):

- The working reconstruction baseline is binary cross-entropy plus soft Dice on
  the lesion channel and mean squared error on the disconnectome channel, i.e.
  the E2 Dice variant is the adopted baseline (note 1).
- NIHSS and time-to-scan are absent from the real Giles-derived cohort, so E8
  DAFT conditioning and Tier-2 NIHSS probing apply only where those variables
  exist (note 2). Age and sex (plus two missing-data indicators) are recoverable.
- Every VAE arm additionally optimises the KL term; the table lists only the
  reconstruction and auxiliary terms.

Experiment ids follow the Theory document's Table 7 exactly (E3 = backbone,
E4 = dimensionality, E5 = ARD, E8 = DAFT, E7a/b = fusion, E9 = Arms C/D,
E10 = Arm E, E11a/b = audits). HISTORICAL NOTE: rows written to runs.jsonl
before this alignment used an older scheme (E7 was the backbone, E3 the
dimensionality, E5a the DAFT conditioning, E8x the Arm-E experiments, E10x
Arms C/D, E11 the audit); migrate an old history in place with
``python scripts/migrate_experiment_ids.py <out_root>``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


# --------------------------------------------------------------------------- #
# Tiers and their stop/go gates (section 13: Intermediate benchmarking).
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Gate:
    """A tier decision rule.

    kind is one of:
      - "gate"        : a hard stop/go. Failing excludes the config from all
                        subsequent (more expensive) tiers.
      - "deprioritize": a soft gate. Failing deprioritises the config before the
                        costly Tier-4 evaluation but does not delete it.
      - "info"        : informational only, no threshold.
    """
    tier: str
    metric: str
    op: str            # ">=" or "<"
    threshold: Optional[float]
    kind: str
    rationale: str = ""

    def passes(self, value: Optional[float]) -> Optional[bool]:
        """None when the gate is informational or the value is missing."""
        if self.kind == "info" or self.threshold is None or value is None:
            return None
        if self.op == ">=":
            return bool(value >= self.threshold)
        if self.op == "<":
            return bool(value < self.threshold)
        if self.op == "<=":
            return bool(value <= self.threshold)
        if self.op == ">":
            return bool(value > self.threshold)
        raise ValueError(f"unknown op {self.op!r}")


# The thresholds are exactly those of section 13.
TIER_GATES: Dict[str, Gate] = {
    "T1": Gate(tier="T1", metric="dice", op=">=", threshold=0.70, kind="gate",
               rationale="Reconstruction gate (Pombo baseline). Configs failing "
               "here are excluded from all subsequent tiers."),
    "T2": Gate(tier="T2", metric="r2_nihss", op=">=", threshold=0.05,
               kind="deprioritize",
               rationale="Linear probe above chance for NIHSS prediction. Configs "
               "below 0.05 are deprioritised before Tier 4."),
    "T3": Gate(tier="T3", metric="active_dims", op="none", threshold=None,
               kind="info",
               rationale="Informational tier: latent structure and "
               "disentanglement, no stop/go threshold."),
    "T4": Gate(tier="T4", metric="root_pehe", op="<", threshold=0.349, kind="gate",
               rationale="Causal gate: beat the Giles VAE-50 (disconnectome) baseline "
               "root-PEHE (0.349)."),
}

# Reference baselines available without any VAE training (section 13, "Baseline").
GILES_BASELINES = {
    "vae50_root_pehe": 0.349,   # the Tier-4 stop/go reference (Giles VAE-50, disconnectome)
    "pombo_dice": 0.70,         # the Tier-1 stop/go reference
    "nihss_r2_chance": 0.05,    # the Tier-2 above-chance reference
}


# --------------------------------------------------------------------------- #
# Experiments (rows of Table 9).
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Experiment:
    eid: str                      # E1, E2, ... (the Table-9 experiment id)
    arm: str                      # A | B | C | D | E | All | CausalPFN
    title: str
    what_changes: str             # the single design element changed vs baseline
    baseline: str                 # the explicit baseline config it departs from
    variant: str                  # the variant / grid tested
    tiers: Tuple[str, ...]        # required evaluation tiers, in cost order
    entry: str                    # runner dispatch key (see runner.ENTRYPOINTS)
    depends_on: Tuple[str, ...] = ()
    params: Dict = field(default_factory=dict)   # flags and grids for the builder
    extra_metrics: Tuple[str, ...] = ()          # e.g. "ood_gap", "mcc"
    notes: str = ""

    @property
    def is_gated(self) -> bool:
        return any(TIER_GATES[t].kind == "gate" for t in self.tiers)


# The order below follows the arm grouping of Table 9. The execution order is
# given by dependency_order(); the Dependencies note of section 14 is encoded in
# each entry's depends_on.
REGISTRY: List[Experiment] = [
    # ----------------------------- Arm A ---------------------------------- #
    Experiment(
        eid="E0", arm="A", title="Reference baselines (no VAE training)",
        what_changes="(references, Table 2)",
        baseline="No learned encoder at all.",
        variant="NMF-50, NMF-21 (non-negative matrix factorisation of the "
                "binary lesion masks, sklearn) and lesion volume (single "
                "feature). Same downstream Tier-4 evaluator as every arm.",
        tiers=("T4",), entry="reference",
        params={"grid": {"method": ["nmf50", "nmf21", "volume"]}},
        notes="The no-VAE reference points of Table 2 (Giles): representation "
              "quality context for the leaderboard. Never selected or "
              "propagated; CPU-only.",
    ),
    Experiment(
        eid="E1", arm="A", title="Generative baseline (reference point)",
        what_changes="(reference point)",
        baseline="Dual 3D-CVAE, lesion 50 + disconnectome 50 (=100-dim); "
                 "recon = BCE; objective L = BCE + beta*KL; N(0,I) prior; "
                 "Giles ResNet-type blocks (~5M); Bernoulli decoder.",
        variant="none",
        tiers=("T1", "T2", "T4"), entry="vae",
        params={"representation": "lesion", "backbone": "resnet",
                "w_dice": 0.0, "fusion_mode": "both"},
        notes="Reproduces the Giles baseline; objective is BCE plus the KL term "
              "(note 3). Rows E2 onward each introduce a single change.",
    ),
    Experiment(
        eid="E2", arm="A", title="Reconstruction loss",
        what_changes="recon -> BCE + Soft Dice (scan lambda_Dice)",
        baseline="E1: recon = BCE (objective BCE + beta*KL)",
        variant="recon -> BCE + Soft Dice; lambda_Dice in {0.1, 0.5, 1.0}. "
                "Adopted working baseline: BCE + Soft Dice.",
        tiers=("T1", "T4"), entry="vae", depends_on=("E1",),
        # zdim pinned to the pre-registered 50+50 baseline: E2 precedes E4 in
        # the chain, so a re-run must not inherit E4's winning dims from the
        # context (it would silently change the axis under comparison)
        params={"representation": "lesion", "backbone": "resnet", "zdim": 50,
                "grid": {"w_dice": [0.1, 0.5, 1.0]}},
        notes="Per section 22 note 1 the E2 Dice variant is adopted as the "
              "working baseline for all subsequent arms.",
    ),
    Experiment(
        eid="E3", arm="A", title="Encoder backbone",
        what_changes="encoder backbone",
        baseline="E1 + E2 loss; backbone = Giles ResNet-type blocks (~5M)",
        variant="(a) Pombo vanilla CNN (~2M), (b) 3D ResNet-18 (11M), "
                "(c) 3D ResNet-50 (25M), (d) Wide CNN (~25M)",
        tiers=("T1", "T2", "T4"), entry="vae", depends_on=("E2",),
        # zdim pinned like E2: E3 precedes E4, so a re-run must stay at the
        # pre-registered 50+50 rather than inherit E4's winner from the context
        params={"representation": "lesion", "zdim": 50,
                "grid": {"backbone": ["cnn", "resnet18", "resnet50", "wide"]},
                "select_by": "T4"},
        notes="Winner backbone becomes the default for all subsequent arms.",
    ),
    Experiment(
        eid="E4", arm="A", title="Latent dimensionality",
        what_changes="latent dimensionality (symmetric + asymmetric)",
        baseline="E3-winner backbone + E2 loss, at 50+50; N(0,I) prior",
        variant="symmetric {25+25, 50+50, 75+75, 100+100} and asymmetric "
                "{75+25, 60+40, 40+60, 25+75}",
        tiers=("T2", "T3", "T4"), entry="e3_sweep", depends_on=("E3",),
        params={"asymmetric": True, "select_by": "T4"},
        notes="First data-driven answer to the dimensionality question; winner "
              "total-dim propagates to later arms.",
    ),
    Experiment(
        eid="E5", arm="A", title="Prior (fixed dim -> data-driven ARD)",
        what_changes="prior: N(0,I) at fixed 100+100 -> ARD prior",
        baseline="E4 reference, fixed 100+100, N(0,I) prior",
        variant="ARD prior on each VAE; active-dim count compared against the "
                "E4 dimensionality scan",
        tiers=("T3",), entry="vae", depends_on=("E4",),
        params={"representation": "lesion", "use_ard": True, "zdim": 100},
        notes="ARD effective-dim count (active dims, KL > 0.01), per modality.",
    ),
    Experiment(
        eid="E6", arm="A", title="Input channels",
        what_changes="input channels",
        baseline="Lesion + disconnectome (100-dim)",
        variant="Lesion only (50-dim); disconnectome only (50-dim)",
        tiers=("T4",), entry="vae", depends_on=("E3",),
        params={"grid": {"fusion_mode": ["both", "lesion", "disconnectome"]},
                "select_by": "T4"},
        notes="root-PEHE for lesion-only vs disc-only vs both.",
    ),
    Experiment(
        eid="E7a", arm="A", title="Channel fusion",
        what_changes="channel fusion: separate encoders -> early fusion",
        baseline="Separate encoders (50+50)",
        variant="Early fusion (2-channel input -> single 100-dim VAE)",
        tiers=("T1", "T2", "T4"), entry="early_fusion", depends_on=("E6",),
        params={"representation": "early_fusion"},
    ),
    Experiment(
        eid="E7b", arm="A", title="Latent decomposition",
        what_changes="latent decomposition (DMVAE)",
        baseline="Early fusion",
        variant="DMVAE: shared 50 + private 25+25",
        tiers=("T4",), entry="dmvae", depends_on=("E7a",),
        params={"shared_dim": 50, "private_dim": 25},
        notes="Watches for KL-collapse of the private spaces.",
    ),

    Experiment(
        eid="E8", arm="A", title="Clinical conditioning (none -> DAFT)",
        what_changes="clinical conditioning: none -> DAFT",
        baseline="No clinical conditioning (E3 winner)",
        variant="DAFT at all levels; clinical = [age, sex, NIHSS*, "
                "time_to_scan*] (* absent in the real cohort)",
        tiers=("T2", "T3", "T4"), entry="vae", depends_on=("E3",),
        params={"representation": "lesion", "use_daft": True},
        notes="Per section 22 note 2 NIHSS/time-to-scan probing applies only to "
              "cohorts or synthetic settings where those variables exist.",
    ),
    # ----------------------------- Arm B ---------------------------------- #
    Experiment(
        eid="E5b", arm="B", title="+ PNS auxiliary loss",
        what_changes="objective -> recon + beta*KL + lambda_PNS*PNS_loss",
        baseline="No auxiliary loss (Arm A architecture; objective recon + "
                 "beta*KL)",
        variant="lambda_PNS in {0.01, 0.1, 0.5}",
        tiers=("T2", "T3"), entry="vae", depends_on=("E3",),
        params={"representation": "lesion", "use_pns": True,
                "grid": {"lambda_pns": [0.01, 0.1, 0.5]}, "select_by": "T3"},
        notes="PNS lower bound (V-aligned) and IOSS are the objective whose "
              "value is being tested here (Arm B).",
    ),
    Experiment(
        eid="E5c", arm="B", title="DAFT + PNS together",
        what_changes="DAFT conditioning + PNS auxiliary loss",
        baseline="Neither",
        variant="DAFT conditioning + PNS auxiliary loss",
        tiers=("T2", "T3", "T4"), entry="vae", depends_on=("E8", "E5b"),
        params={"representation": "lesion", "use_daft": True, "use_pns": True},
    ),

    # ----------------------------- Arm C ---------------------------------- #
    Experiment(
        eid="E9a", arm="C", title="Objective (reconstruction -> SupCon)",
        what_changes="objective: reconstruction -> SupCon (no KL term)",
        baseline="E3-winner encoder (Arm A)",
        variant="SupCon + binary augmentations, hierarchical fusion. "
                "Arm C is not a VAE - no KL term.",
        tiers=("T2", "T3", "T4"), entry="contrastive", depends_on=("E3",),
        params={"no_recon": True}, extra_metrics=("ood_gap",),
        notes="OOD: root-PEHE gap Env A->B (Environment stress test, 9.6).",
    ),
    Experiment(
        eid="E9b", arm="C", title="Reconstruction + contrastive combined",
        what_changes="L = L_recon + lambda*L_SupCon (section 8.4)",
        baseline="E3 winner",
        variant="L = L_recon + lambda*L_SupCon",
        tiers=("T1", "T2", "T3", "T4"), entry="contrastive", depends_on=("E3",),
        params={"no_recon": False}, extra_metrics=("ood_gap",),
        notes="Audit D7: confirm whether the VAE branch keeps beta*KL.",
    ),

    # ----------------------------- Arm D ---------------------------------- #
    Experiment(
        eid="E9c", arm="D", title="Objective (masked modelling)",
        what_changes="objective: masked modelling (Hi-End-MAE, no KL term)",
        baseline="E3-winner encoder (Arm A)",
        variant="Hi-End-MAE, 75% vascular-block masking, lesion-weighted BCE. "
                "Arm D is a masked autoencoder - no KL term.",
        tiers=("T1", "T2", "T4"), entry="mae", depends_on=("E3",),
        params={"mask_ratio": 0.75, "lesion_weight": 5.0},
        extra_metrics=("ood_gap",),
    ),

    # ----------------------------- Arm E ---------------------------------- #
    Experiment(
        eid="E10a", arm="E", title="Prior (factorized -> conditional p(z|pa_x))",
        what_changes="prior: standard VAE -> conditional HVAE p(z|pa_x)",
        baseline="Standard VAE -> CausalPFN (objective recon + beta*KL)",
        variant="Conditional HVAE p(z|pa_x); KL taken against the conditional "
                "prior",
        tiers=("T3", "T4"), entry="dscm", depends_on=("E4", "E3"),
        params={},
        notes="Identifiability / IOSS; root-PEHE especially under strong "
              "confounding.",
    ),
    Experiment(
        eid="E10b", arm="E", title="+ InterSynth environment index",
        what_changes="+ InterSynth env index (multi-environment iVAE)",
        baseline="E10a on pa_x only",
        variant="+ InterSynth env index (multi-environment iVAE)",
        tiers=("T4",), entry="dscm", depends_on=("E10a",),
        params={"multi_env": True}, extra_metrics=("mcc",),
        notes="MCC (latent identifiability); root-PEHE.",
    ),
    Experiment(
        eid="E10c", arm="E", title="ARD on top of the conditional prior",
        what_changes="conditional HVAE + ARD prior",
        baseline="E10a (conditional HVAE)",
        variant="Conditional HVAE + ARD prior",
        tiers=("T3", "T4"), entry="dscm", depends_on=("E10a",),
        params={"use_ard": True},
        notes="Active-dim count within the causal prior.",
    ),

    # --------------------------- Cross-arm audit -------------------------- #
    Experiment(
        eid="E11a", arm="All", title="Equity audit across best configs",
        what_changes="(audit across best configs)",
        baseline="All arms, best config",
        variant="root-PEHE stratified: territory (A/P), volume quartiles, age, "
                "sex",
        tiers=("T4",), entry="audit", depends_on=(),
        params={"strata": ["territory", "volume_quartile", "age_band", "sex"]},
        notes="max/min root-PEHE ratio across subgroups; balanced prescriptive "
              "accuracy. Runs after all arms have a best config.",
    ),

    # ----------------------------- CausalPFN ------------------------------ #
    Experiment(
        eid="E11b", arm="All", title="Backbone x loss interaction audit",
        what_changes="the separability assumption of the greedy selection",
        baseline="The greedy winners of E2 (loss) and E3 (backbone).",
        variant="2x2 grid: the two best backbones x the two best Dice weights "
                "(from the E3/E2 rankings in the winner context), at the "
                "winning dimensionality; three seeds; bootstrap-paired "
                "root-PEHE. Adopt the joint winner if it differs from the "
                "greedy one (section 3.2.2).",
        tiers=("T1", "T4"), entry="interaction_2x2", depends_on=("E2", "E3"),
        params={"select_by": "T4"},
        notes="Close calls warrant extra seeds (section 3.2.3).",
    ),
    Experiment(
        eid="E12", arm="CausalPFN", title="Training curriculum",
        what_changes="training curriculum (1-stage / 2-stage vs 3-stage)",
        baseline="3-stage curriculum",
        variant="(a) 1-stage (N=4,119 throughout), (b) 2-stage (skip Stage 1), "
                "reference 3-stage",
        tiers=("T4",), entry="curriculum", depends_on=(),
        params={"variants": ["reference", "two_stage", "one_stage"]},
        notes="Convergence speed and final root-PEHE. CausalPFN-only, "
              "independent of the encoder arms.",
    ),
]


# The Arm-A execution order from the Dependencies note of section 14:
# E1 -> E2 (loss) -> E3 (backbone; winner defaults) -> E4 (dimensionality) ->
# E5 (ARD). E5-E6 parallel after E3. E9 after E6.
# Selection order of section 3.2.1: loss -> backbone -> dimensionality -> ARD
# -> channels -> fusion -> conditioning. E0 provides the no-VAE references.
ARM_A_ORDER: Tuple[str, ...] = (
    "E0", "E1", "E2", "E3", "E4", "E5", "E6", "E7a", "E7b", "E8",
)


_BY_ID = {e.eid: e for e in REGISTRY}


def get_experiment(eid: str) -> Experiment:
    if eid not in _BY_ID:
        raise KeyError(f"unknown experiment {eid!r}; known: {sorted(_BY_ID)}")
    return _BY_ID[eid]


def arm_experiments(arm: str) -> List[Experiment]:
    """All experiments of an arm, in dependency order."""
    ids = [e.eid for e in REGISTRY if e.arm == arm]
    return [get_experiment(e) for e in dependency_order(ids)]


def dependency_order(ids: Optional[List[str]] = None) -> List[str]:
    """A topological order of the requested experiment ids that respects
    depends_on. Dependencies outside the requested set are ignored (they are
    assumed already available from a prior run, e.g. the E3 winner)."""
    ids = list(_BY_ID) if ids is None else list(ids)
    wanted = set(ids)
    ordered: List[str] = []
    seen: set = set()

    def visit(eid: str, stack: Tuple[str, ...] = ()):  # noqa: ANN001
        if eid in seen or eid not in wanted:
            return
        if eid in stack:
            raise ValueError(f"dependency cycle at {eid}: {stack}")
        for dep in get_experiment(eid).depends_on:
            visit(dep, stack + (eid,))
        seen.add(eid)
        ordered.append(eid)

    # Preserve the registry order among otherwise-independent nodes.
    for e in REGISTRY:
        if e.eid in wanted:
            visit(e.eid)
    return ordered
