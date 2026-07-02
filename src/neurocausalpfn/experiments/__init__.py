"""Experiment orchestration for Table 9 (Planned experiments, baselines, and
evaluation criteria).

This package turns the authoritative Phase-1 roadmap of the architecture
specification (Table 9, section 14) into runnable, gated and reported
experiments. It does not re-implement any training or metric; it composes the
pieces already in the codebase:

- ``registry``  : the declarative catalogue of E1-E12, one entry per Table-9 row
                  (arm, what changes, baseline, variant, evaluation tiers,
                  dependencies).
- ``tiers``     : the tiered evaluation harness T1-T4 with the stop/go gates
                  (T1 Dice >= 0.70, T2 R2 >= 0.05, T3 informational,
                  T4 root-PEHE < 0.349) described in section 13.
- ``artifacts`` : extraction of tier inputs (Dice, the latent code Z = mu,
                  logvar, clinical/outcome vectors) from a trained Stage-1 VAE.
- ``estimators``: the Tier-4 causal evaluators (semi-synthetic potential
                  outcomes on Z plus a T-learner, or the prior-fitted network).
- ``logging_backend`` : Weights & Biases / MLflow with a local JSON+CSV fallback.
- ``runner``    : the orchestrator (dependency order, winner propagation across
                  >= 3 seeds) and its command-line interface.
- ``report``    : the consolidated leaderboard and the bootstrap-paired test on
                  root-PEHE.
"""
from .registry import (ARM_A_ORDER, REGISTRY, TIER_GATES, Experiment, Gate,
                       arm_experiments, dependency_order, get_experiment)

__all__ = [
    "REGISTRY",
    "TIER_GATES",
    "ARM_A_ORDER",
    "Experiment",
    "Gate",
    "get_experiment",
    "arm_experiments",
    "dependency_order",
]
