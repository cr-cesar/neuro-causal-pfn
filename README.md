# Neuro-Causal-PFN

Causal foundation model based on neuroimaging for estimating individualized
treatment effects in ischemic stroke, derived from lesion anatomy and using
in-context learning. The work is organised in two phases. Phase 1 (build and
select) constructs and selects the representation and the causal model and
evaluates them in silico on the InterSynth prior; it comprises two components,
trained in sequence and then composed:

- The encoder: two 3D convolutional variational autoencoders compress a lesion
  mask and a disconnectome map into a compact code.
- The transformer: trained from scratch with the prior-fitted network
  methodology on a synthetic cohort with known counterfactual outcomes (the
  Neuro-Prior), which returns for each patient the distribution of the expected
  conditional potential outcome under treatment and under control. The
  difference is the individualized treatment effect.

Phase 2 (clinical validation) validates the selected model on an external
real-world trial. Throughout, *Stage 1/2/3* denotes only the transformer's
training curriculum (context length 1,024, then 4,119, then 20,000), never a
component or a phase.

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

Folder layout (everything under `data/`, which is in `.gitignore`):

    data/
      lesions/          lesion masks (lesions.zip from Giles)        -> encoder input
      atlases/          functional parcellation and subdivisions     -> only if real InterSynth is used
      disconnectomes/   continuous disconnection maps (0..1)         -> second modality, paired by id
      representation/   representation_{hash}.npz (Z + clinical)     -> encoder-to-transformer bridge

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
  experiment to those that depend on it (E7's backbone, E3's dimensionality,
  E6's channels, E2's Dice weight).
- `logging_backend.py` logs to Weights & Biases or MLflow with an always-on
  local JSON+CSV fallback; `report.py` writes the leaderboard and the
  bootstrap-paired test on root-PEHE.

Run in prototype mode (CPU, synthetic masks, seconds to minutes):

    bash scripts/run_experiments.sh A 3        # Arm A, 3 seeds, dependency order
    bash scripts/run_experiments.sh E7 1       # a single experiment
    python -m neurocausalpfn.experiments.runner --experiment E7 --mode prototype

Run Arm A in full mode on the cluster:

    sbatch scripts/run_arm_a.sbatch

The leaderboard lands in `outputs/experiments/leaderboard.{csv,md}`; every run
also appends to `outputs/experiments/runs.jsonl`. Select the logging backend with
`--backend wandb|mlflow|local` or the `NEUROCAUSAL_LOGGER` envi