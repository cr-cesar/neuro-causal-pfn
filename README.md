# Neuro-Causal-PFN

Causal foundation model based on neuroimaging for estimating individualized
treatment effects in ischemic stroke, derived from lesion anatomy and using
in-context learning. The project has two stages that are trained in sequence and
then composed:

- Stage 1: two 3D convolutional variational autoencoders compress a lesion mask
  and a disconnectome map into a compact code.
- Stage 2: a transformer trained from scratch with the prior-fitted network
  methodology on a synthetic cohort with known counterfactual outcomes (the
  Neuro-Prior), which returns for each patient the distribution of the expected
  conditional potential outcome under treatment and under control. The
  difference is the individualized treatment effect.

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
      train/                  training of Stage 1 (two modalities) and Stage 2, real wiring
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

This should train Stage 1 and Stage 2 in prototype mode and write checkpoints to
`outputs/vae_prototype/vae_lesion.pt` and `outputs/pfn_prototype/pfn.pt`.

## Quick run (smoke test)

    bash scripts/run_prototype.sh

This trains Stage 1 and Stage 2 in prototype mode on CPU in seconds, with
synthetic data. Each stage can also be called separately:

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
      lesions/          lesion masks (lesions.zip from Giles)        -> Stage 1 input
      atlases/          functional parcellation and subdivisions     -> only if real InterSynth is used
      disconnectomes/   continuous disconnection maps (0..1)         -> second modality, paired by id
      representation/   representation_{hash}.npz (Z + clinical)     -> Stage 1 to Stage 2 bridge

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
Stage 1:

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

## The Stage 2 prior: synthetic or InterSynth

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

## Stage 2 wiring on real data (run_stage2_real)

`train/run_stage2_real.py` joins the two stages on real data: it loads the frozen
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

- The VAE encoder is frozen after Stage 1; its output is exported once and
  versioned by a hash of the weights, so that every Stage 2 result is traceable
  back to an exact representation.
- The transformer objective is the histogram loss over the true expected
  conditional potential outcome, with the context length on a curriculum from
  shorter to longer.
- There are two encoders for the transformer, selectable by `cfg["pfn"]["arch"]`:
  `linear` (one projection per row, useful as a baseline and for the prototype)
  and `tabicl` (TabICL style), which first applies column-wise attention across
  the samples, so that each cell becomes aware of its whole variable, and then
  row-wise attention across patients. Both stages share the context-only mask, so
  no query prediction depends on another. The attention is still dense; for the
  large contexts of full mode it would be replaced by a more efficient attention.

## Open points

Still to be confirmed: the identity of the validation trial, the scale of the
precision target, the size of the transformer (to be justified with the backbone
ablation) and the license of the reference VAE. The provenance of the
disconnectome is already resolved: the lab has the continuous maps, paired by id
with the lesions. The detail is in the implementation plan document.
