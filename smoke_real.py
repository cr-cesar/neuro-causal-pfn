import sys; sys.path.insert(0, "src")
import numpy as np
from neurocausalpfn.data.nifti_dataset import LesionMaskDataset
from neurocausalpfn.train.train_vae import prototype_config, run_vae

# 1) comprobar que detecta y lee tus lesiones reales, y que parsea edad y sexo
ds = LesionMaskDataset(root="data/lesions", in_shape=(96, 112, 96))
print("lesiones detectadas:", len(ds), "| sintetico?", ds.synthetic)
print("forma de un volumen:", tuple(ds[0].shape))
print("covariables clinicas [N,4]:\n", np.round(ds.clinical_matrix(), 3))

# 2) entrenar el VAE unas pocas epocas, solo para ver que corre y la perdida es finita
cfg = prototype_config()
cfg["data"]["root"] = "data/lesions"
cfg["data"]["n_synth"] = 0
cfg["data"]["resolution"] = [96, 112, 96]
cfg["vae"]["epochs"] = 25
cfg["vae"]["lr"] = 1e-3
cfg["vae"]["warmup_frac"] = 0.4
cfg["out_dir"] = "outputs/vae_smoke_real"
run_vae(cfg)