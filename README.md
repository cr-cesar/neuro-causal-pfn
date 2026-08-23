# Neuro-Causal-PFN

Causal foundation model based on neuroimaging for estimating individualized
treatment effects in ischemic stroke, derived from lesion anatomy and using
in-context learning. The work is organised in two phases.

Phase 1 (encoder construction and selection): the VAE encoder arms are trained on
real lesion masks and evaluated through Tiers 1 to 4, and the winning
representation is frozen. Tier 4 uses the off-the-shelf CausalPFN (Balazadeh et
al. 2025) as a fixed evaluator, with no transformer training in Phase 1. Two 3D
convolutional VAEs compress a lesion mask and a disconnectome map into a compact
latent code.

Phase 2 (foundation model training and clinical validation): the Neuro-Causal-PFN
transformer is trained from scratch with the prior-fitted network methodology on
the Neuro-Prior synthetic generator, using the frozen encoder's latent codes as
covariates. It returns each patient's individualized treatment effect (CATE)
without retraining, and is then validated on real trial data. Phase 2 is the
primary training of the foundation model and the central contribution of the
project.

Throughout, *curriculum round 1/2/3* denotes only the transformer's training
curriculum (context length 1,024, then 4,119, then 20,000), never a phase.

A single codebase runs in two modes from the same source. Prototype mode runs on
CPU with reduced data and synthetic masks, without needing the real data or the
cluster. Full mode scales to the V100 nodes. Only the configuration values
change.

## Structure

    configs/                  configuration profiles (Hydra) for prototype and full
    src/neurocausalpfn/
      data/                   NIfTI loading, transforms, clinical covariates
      vae/                    3D VAE, losses (BCE + Dice + KL and continuous MSE), modality fusion, export
      prior/                  InterSynth generator, confounding, R1/R2 verifier, cohort
      pfn/                    tokens, attention mask, CEPO-PPD head, transformer (linear and TabICL-style), inference
      train/                  training of the encoder (two modalities) and the transformer, real wiring
      eval/                   root-PEHE, prescriptive accuracy, coverage
    tests/                    unit tests and end-to-end smoke test
    scripts/                  prototype run and cluster template

## Installation

Prototype mode (CPU):

    conda env create -f env/environment.prototype.yml
    conda activate neuro-causal-pfn-proto
    pip install -e .

For the real NIfTI data and the causal baselines, add the extras:

    pip install -e ".[imaging,baselines,cluster]"

### Apple Silicon (M1/M2/M3) notes

On recent macOS ARM machines, PyTorch, FAISS and OpenMP can clash and cause
`Segmentation fault: 11` during the prototype run, or FAISS may fail to import
with an OpenMP symbol error.

The following setup has been tested to work on Apple Silicon:

```bash
# create and activate the prototype environment
conda env create -f env/environment.prototype.yml
conda activate neuro-causal-pfn-proto

# install FAISS from conda (recommended on macOS ARM)
conda install -c conda-forge faiss-cpu

# install this repo in editable mode
pip install -e .

# recommended runtime flags to avoid OpenMP / MPS issues
export OMP_NUM_THREADS=1
export PYTORCH_MPS_DISABLE=1
export KMP_DUPLICATE_LIB_OK=TRUE
```

Then the prototype smoke test can be run as usual:

```bash
bash scripts/run_prototype.sh
```

This should train the encoder and the transformer in prototype mode and write
checkpoints to `outputs/vae_prototype/vae_lesion.pt` and
`outputs/pfn_prototype/pfn.pt`.

## Phase 1 across platforms (Windows / macOS / Myriad)

The intended workflow is: develop on Windows, sanity-check on a Mac, train at
scale on UCL Myriad. The same source runs everywhere; only the entry script
changes.

**Windows (development).** Use the PowerShell runner for the CPU smoke test:

    powershell -ExecutionPolicy Bypass -File scripts\run_prototype.ps1

and the tests with `set PYTHONPATH=src` (cmd) or `$env:PYTHONPATH="src"`
(PowerShell) before `pytest -q`. Keep `num_workers: 0` in prototype configs on
Windows: DataLoader workers use `spawn` there and add no value at prototype
sizes.

**macOS (sanity check).** Follow the Apple Silicon notes below, then
`bash scripts/run_prototype.sh`. CPU is the default device on Apple Silicon
(3D convolutions on MPS are still unreliable); see PORTABILITY.md.

**UCL Myriad (full training).** Myriad schedules with SGE (`qsub`), not SLURM,
so use `scripts/run_arm_a_myriad.qsub.sh` rather than the `.sbatch` templates
(those target SLURM clusters). One-off setup on a login node:

    cd ~/Scratch && git clone <this repo> && cd neuro-causal-pfn
    module load python/miniconda3
    conda env create -f env/environment.cluster.yml
    conda activate neuro-causal-pfn
    pip install -e ".[imaging,baselines,cluster]"

then place the real data under `data/` (see the Data section) and submit:

    qsub scripts/run_arm_a_myriad.qsub.sh

The job runs the Arm A programme (E1 → E2 → E4 → E5/E5 → E6/E3 → E8) through
the orchestrator and writes the leaderboard to
`outputs/experiments/leaderboard.{csv,md}`. The repo must live under
`~/Scratch`, because compute nodes cannot write to `$HOME`.

**Login-node etiquette (Arbiter2).** Myriad login nodes are shared and policed:
each user must stay under 6 cores / 30 GB there, and violations bring
escalating CPU/memory penalties on the account. Anything heavier than editing,
`git`, or a quick one-liner belongs in a job. In particular, run the test suite
as a compute-node job (`qsub scripts/run_tests_myriad.qsub.sh`), never as plain
`pytest` on the login node — as a safety net, `tests/conftest.py` detects
`login*` hostnames and caps PyTorch at 2 threads.

## Quick run (smoke test)

    bash scripts/run_prototype.sh

This trains the encoder and the transformer in prototype mode on CPU in seconds,
with synthetic data. Each component can also be called separately:

    python -m neurocausalpfn.train.train_vae --mode prototype
    python -m neurocausalpfn.train.train_pfn --mode prototype

## Tests

    pip install pytest
    PYTHONPATH=src pytest -q

The two most important tests are the attention-mask test (that the weight of one
query on another is exactly zero) and the identifiability-verifier test (that it
accepts an ignorable process and rejects one with an unobserved confounder),
because the latter operationalizes the convergence requirement of the
prior-fitted network.

## Data

Folder layout (everything under `data/`, which is in `.gitignore`). Two cohort
tiers share one structure and the atlases are common to both:

    data/
      Trial data/
        lesions/          pilot lesion masks (binary, MNI 2mm)       -> encoder input
        disconnectomes/   pilot disconnection maps (0..1)            -> second modality, paired by id
      Full data/
        lesions/          full cohort lesion masks (Giles)
        disconnectomes/   full cohort disconnection maps
      atlases/            functional parcellation and subdivisions   -> only if real InterSynth is used
      representation/     representation_{hash}.npz (Z + clinical)   -> encoder-to-transformer bridge

The tier is selected with `--data-tier trial|full` on every entry point (the
trainers, the experiments runner, `run_stage2_real`) or with the
`NEUROCAUSAL_DATA_TIER` environment variable; the default is `full`. The
resolver (`neurocausalpfn/data/paths.py`, the single source of truth for these
locations) matches the folder name case-insensitively and falls back to the
legacy flat layout (`data/lesions`, `data/disconnectomes`) when no tiered
folder exists.

The lesion dataset (`LesionMaskDataset`) looks for NIfTI masks in the directory
given in `configs/data/lesion.yaml` (`root: data/lesions`). If none exist, it
synthesizes lesion-like masks so the prototype can run. The Giles masks are
already in MNI at 91x109x91; the code pads them to 96x112x96 and binarizes them,
so no further preprocessing is needed for the VAE.

Age and sex do not come in a table but in the filename, with the pattern
`lesion{id}_{age}_{sex}.nii.gz` and the literal `NA` when missing. The parser in
`data/clinical.py` extracts them and builds a covariate vector with missing-data
indicators; `LesionMaskDataset.clinical_matrix()` returns that matrix aligned
with the order of the masks.

## The two modalities: lesion and disconnectome

Each patient can enter through two complementary images, each with its own VAE in
the encoder:

- Lesion: a binary mask. Reconstruction with BCE plus soft Dice (`vae_loss`),
  because the foreground is a tiny fraction of the volume.
- Disconnectome: a continuous disconnection-probability map in [0, 1], already
  computed by the lab (BCBtoolkit style) in MNI at 2mm. Reconstruction with MSE
  on the predicted probability (`vae_loss_mse`), without binarizing.

They are trained with the same entry point, changing the modality:

    python -m neurocausalpfn.train.train_vae --mode full --representation lesion
    python -m neurocausalpfn.train.train_vae --mode full --representation disconnectome

The disconnectome shares the name pattern `lesion{id}_{age}_{sex}.nii.gz`, so
`PairedLesionDisconnectomeDataset` pairs lesion and disconnectome by patient id.
The fusion of the two latents (`vae/fusion.py`) offers three variants ready to
compare, chosen by `fusion_mode`: `lesion` (only the lesion latent),
`disconnectome` (only the disconnectome latent) and `both` (the concatenation of
the two, which doubles the covariate dimension).

## The transformer prior: synthetic or InterSynth

The transformer is trained on a process prior, chosen by configuration in
`cfg["prior"]["kind"]`:

- `synthetic`: the lightweight generator (`prior/intersynth.py`), which samples
  Gaussian covariates from scratch. It is the default and the one used by the
  prototype and the smoke test.
- `intersynth`: the real anatomical mechanism (`prior/intersynth_atlas.py` plus
  `prior/atlas.py`), which intersects each lesion with the functional
  parcellation to fabricate the ground truth: deficit from an overlap of at least
  5% with a subnetwork, treatment susceptibility according to the dominant
  subnetwork (transcriptomic or receptomic), outcome from a combination of
  treatment effect and spontaneous recovery, and assignment with observed
  confounding (centroid distance) or optionally unobserved. The covariate seen by
  the transformer is the encoder latent if `z_pool` is passed, or the observed
  covariates otherwise. To enable it: `--prior intersynth`, with `atlas_dir`
  pointing to `data/atlases`. The loader reads the real Giles structure:
  `functional_parcellation_2mm.nii.gz` (networks labeled 1..K) and
  `2mm_parcellations/{modality}/` with one file per network whose two subnetworks
  are labels 1 and 2. The modality is `receptor` (Hansen receptome) or `genetics`
  (Allen transcriptome), selectable by configuration.

## Transformer wiring on real data (run_stage2_real)

`train/run_stage2_real.py` joins the two components on real data: it loads the frozen
encoders, computes each patient's latent (lesion and, depending on the variant,
disconnectome), fuses them, and builds the anatomical Neuro-Prior by passing
those latents as `z_pool` and the lesions on their native grid for the overlaps
with the atlas. It then trains the transformer and saves the checkpoint.
Inference on real data (`infer_cate_real`) takes an observed cohort as context
(latents, treatment and outcome) and returns the individualized effect of each
new patient with a credible interval.

The full cluster job is in `scripts/run_full_cluster.sbatch`: it trains the two
VAEs (with `--resume` to resume if the job is interrupted) and then runs
`run_stage2_real`.

## Implementation notes

- The encoder is frozen after training; its output is exported once and
  versioned by a hash of the weights, so that every transformer result is
  traceable back to an exact representation.
- The transformer objective is the histogram loss over the true expected
  conditional potential outcome, with the context length on a curriculum from
  shorter to longer.
- There are two encoders for the transformer, selectable by `cfg["pfn"]["arch"]`:
  `linear` (one projection per row, useful as a baseline and for the prototype)
  and `tabicl` (TabICL style), which first applies column-wise attention across
  the samples, so that each cell becomes aware of its whole variable, and then
  row-wise attention across patients. Both the column and row passes share the context-only mask, so
  no query prediction depends on another. The attention is still dense; for the
  large contexts of full mode it would be replaced by a more efficient attention.

## Table 9 experiments (the orchestrator)

`src/neurocausalpfn/experiments/` turns Table 9 (Planned experiments, baselines,
and evaluation criteria) into runnable, gated and reported experiments. It does
not re-implement training or metrics; it composes the entry points above.

- `registry.py` is the declarative catalogue: one entry per Table-9 row (E1-E12)
  with its arm, what it changes, the baseline it departs from, the variant or
  grid tested, the evaluation tiers and the dependencies.
- `tiers.py` is the tiered harness with the stop/go gates of section 13: T1
  reconstruction (Dice >= 0.70, a hard gate), T2 clinical probing (R2 >= 0.05, a
  soft gate that deprioritises), T3 latent quality (informational), T4 causal
  (root-PEHE < 0.349, the Giles VAE-50 (disconnectome) reference; a hard gate). Failing a hard
  gate short-circuits the more expensive tiers.
- `estimators.py` provides the Tier-4 evaluators: a representation-aware
  semi-synthetic potential-outcomes problem built on the arm's own latents with a
  cross-fitted T-learner (default, cheap), the prior-fitted network on the
  Neuro-Prior (sanity, used by E12), and the real CausalPFN over the frozen
  latents via `run_stage2_real` (production, full mode).
- `runner.py` is the orchestrator: it runs each experiment across >= 3 seeds,
  evaluates the tiers, aggregates, and propagates the winner of a selection
  experiment to those that depend on it (E3's backbone, E4's dimensionality,
  E6's channels, E2's Dice weight).
- `logging_backend.py` logs to Weights & Biases or MLflow with an always-on
  local JSON+CSV fallback; `report.py` writes the leaderboard and the
  bootstrap-paired test on root-PEHE.

Run in prototype mode (CPU, synthetic masks, seconds to minutes):

    bash scripts/run_experiments.sh A 3        # Arm A, 3 seeds, dependency order
    bash scripts/run_experiments.sh E3 1       # a single experiment
    python -m neurocausalpfn.experiments.runner --experiment E3 --mode prototype

Run Arm A in full mode on the cluster:

    sbatch scripts/run_arm_a.sbatch

The leaderboard lands in `outputs/experiments/leaderboard.{csv,md}`; every run
also appends to `outputs/experiments/runs.jsonl`. Select the logging backend with
`--backend wandb|mlflow|local` or the `NEUROCAUSAL_LOGGER` environment
variable.