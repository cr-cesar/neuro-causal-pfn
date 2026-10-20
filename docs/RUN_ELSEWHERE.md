# Training Neuro-Causal-PFNs on a machine outside Myriad

What runs here: the pre-training of the PFN on the anatomy prior, one job per
checkpoint, one GPU each, 0.5 to 2 hours. What stays on the cluster: the
images, the encoders, the scoring on the virtual trials (it needs the UCLH
latents and the image listing). From here only checkpoints go back.

## Needs

- Linux, one NVIDIA GPU with 16 GB or more, driver for CUDA 11.8 or newer,
  8 CPU cores, 32 GB RAM, 20 GB of disk (the Python environment takes most
  of it).
- Python 3.10 or 3.11 as `python3` (or set `PYTHON=/path/to/python3.11`).
- Access to GitHub for the repository and to the PyTorch index for the wheels.
- The bundle `neuro-causal-pfn-bundle.tgz` (about 200 MB): anatomy caches
  and per-fold pools with hashed file names, the KCH latents of the fold
  encoders, the atlases, a jobs list. It holds no images, no names and no
  outcomes.

## Once

```bash
git clone git@github.com:cr-cesar/neuro-causal-pfn.git
cd neuro-causal-pfn
git checkout claude/phase-1-vae-encoder-f02ugc      # or the branch you were told
bash scripts/setup_env_local.sh                      # 5 to 10 minutes; ends with "torch ... available True"
tar -xzf /path/to/neuro-causal-pfn-bundle.tgz
cp -r bundle/outputs bundle/data bundle/jobs .       # caches under outputs/, atlases under data/
```

`bash scripts/setup_env_local.sh check` repeats the import and GPU check.

## Each time

```bash
bash scripts/run_jobs.sh jobs/<list>.txt
```

The runner takes the lines of the list one by one (each line is one training
with its variables), writes `logs/<n>_<name>.log`, skips a job whose
`OUT/pfn.pt` already exists and goes on after a failure. It can be stopped
with Ctrl-C and rerun; finished jobs are not repeated. Each job prints its
configuration, the GPU it found and the loss every few hundred iterations.

When the list is done, send back the output folders it names
(`outputs/pfn_reduced_*/`, a few MB each: `pfn.pt`, `history.json`, the
printed config) and the `logs/` folder. Nothing else is needed on the cluster
to score them.

## If something fails

- `no Python environment`: run the setup script first.
- `torch ... available False`: the driver is older than CUDA 11.8 or the
  GPU is not visible; `nvidia-smi` shows which.
- `no prior cache`: the bundle was not unpacked into the repository root.
- A job that dies with an out-of-memory error: add `BATCH=2` to that line.
