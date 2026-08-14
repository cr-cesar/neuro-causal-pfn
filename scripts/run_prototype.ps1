# Prototype smoke test on CPU, Windows equivalent of run_prototype.sh.
# From the repo root:
#   powershell -ExecutionPolicy Bypass -File scripts\run_prototype.ps1
$ErrorActionPreference = "Stop"

$env:PYTHONPATH = "$PWD\src;$env:PYTHONPATH"
# Same portability flags as scripts/portable_env.sh: harmless where not needed.
if (-not $env:KMP_DUPLICATE_LIB_OK) { $env:KMP_DUPLICATE_LIB_OK = "TRUE" }
if (-not $env:OMP_NUM_THREADS) { $env:OMP_NUM_THREADS = "1" }

Write-Host ">> Stage 1: lesion autoencoder (prototype)"
python -m neurocausalpfn.train.train_vae --mode prototype
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host ">> Stage 2: causal transformer (prototype)"
python -m neurocausalpfn.train.train_pfn --mode prototype
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
