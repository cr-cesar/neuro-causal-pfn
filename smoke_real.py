import sys; sys.path.insert(0, "src")
import numpy as np
from neurocausalpfn.data.nifti_dataset import LesionMaskDataset
from neurocausalpfn.train.train_vae import prototype_config, run_vae

# 1) check that it detects and reads your real lesions, and that it parses age and sex
ds = LesionMaskDataset(root="data/lesions", in_shape=(96, 112, 96))
print("detected lesions:", len(ds), "| synthetic?", ds.synthetic)
print("shape of a volume:", tuple(ds[0].shape))
print("clinical covariates [N,4]:\n", np.round(ds.clinical_matrix(), 3))

# 2) train the VAE for a few epochs, just to see that it runs and the loss is finite
cfg = prototype_config()
cfg["data"]["root"] = "data/lesions"
cfg["data"]["n_synth"] = 0
cfg["data"]["resolution"] = [96, 112, 96]
cfg["vae"]["epochs"] = 25
cfg["vae"]["lr"] = 1e-3
cfg["vae"]["warmup_frac"] = 0.4
cfg["out_dir"] = "outputs/vae_smoke_real"
run_vae(cfg)