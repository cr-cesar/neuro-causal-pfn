"""Full-mode configs say device='auto'; every sink must resolve it, never pass
it raw to torch. Regression test for the E1 full-mode crash in vae_artifacts."""
from neurocausalpfn.data.nifti_dataset import LesionMaskDataset
from neurocausalpfn.experiments.artifacts import vae_artifacts
from neurocausalpfn.utils.runtime import resolve_device
from neurocausalpfn.vae.conv3d_vae import ConvVAE3D


def test_resolve_device_accepts_auto():
    assert resolve_device({"device": "auto"}) in ("cuda", "cpu")


def test_vae_artifacts_accepts_auto_device():
    shape = (24, 28, 24)
    model = ConvVAE3D(in_channels=1, zdim=8, in_shape=shape, channels=(8, 16, 32, 64))
    dataset = LesionMaskDataset(root=None, in_shape=shape, n_synth=4, seed=0, binarize=True)
    out = vae_artifacts(model, dataset, device="auto", batch_size=2)
    assert out.Z.shape == (4, 8)
