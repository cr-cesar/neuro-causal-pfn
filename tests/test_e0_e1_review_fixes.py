"""Regression tests for the E0/E1 code review fixes:

1. the equity "sex" stratum reads the sex column of the clinical vector
   (column 2), not the age-missing indicator (column 1), and unknown sex is
   excluded from the per-group ratio;
2. the per-epoch loss parts are epoch means weighted by batch size, not the
   values of the last (remainder) batch;
3. the soft Dice is computed in float32 so a float16 input cannot overflow.
"""
import numpy as np
import pytest

from neurocausalpfn.eval.equity import stratified_pehe
from neurocausalpfn.experiments import artifacts as art
from neurocausalpfn.experiments.runner import _strata_from


def test_sex_stratum_reads_sex_column_and_excludes_missing():
    n = 12
    rng = np.random.default_rng(0)
    clinical = np.zeros((n, 4), dtype=np.float64)   # [age_norm, age_missing, sex_val, sex_missing]
    clinical[:, 0] = rng.normal(size=n)
    clinical[:3, 1] = 1.0                            # three images with a missing age
    clinical[:, 2] = np.where(np.arange(n) % 2 == 0, 0.5, -0.5)   # alternating M / F
    clinical[-1, 2], clinical[-1, 3] = 0.0, 1.0      # last image: sex unknown
    a = art.VaeArtifacts(Z=rng.normal(size=(n, 5)), logvar=None, clinical=clinical,
                         volume=rng.uniform(size=n), has_posterior=False)
    strata = _strata_from(a)
    sex = strata["sex"]
    expected = np.where(clinical[:, 2] > 0, 1, 0)
    expected[-1] = -1
    assert np.array_equal(sex, expected)
    # the old bug: the stratum tracked the age-missing indicator instead
    assert not np.array_equal(sex, (clinical[:, 1] > np.median(clinical[:, 1])).astype(int))


def test_stratified_pehe_skips_negative_labels():
    pred = np.array([0.0, 0.0, 0.0, 5.0])
    true = np.zeros(4)
    groups = np.array([0, 0, 1, -1])                 # the outlier is unlabelled
    out = stratified_pehe(pred, true, groups)
    assert set(out) == {"all", "0", "1", "max_min_ratio", "passes"}
    assert out["0"] == 0.0 and out["1"] == 0.0
    assert out["all"] > 0.0                          # still counted in the overall value


def test_epoch_returns_batch_weighted_means():
    torch = pytest.importorskip("torch")
    from torch.utils.data import DataLoader, TensorDataset

    from neurocausalpfn.train.train_vae import _epoch

    class Tiny(torch.nn.Module):
        def forward(self, x):
            mu = torch.zeros(x.shape[0], 2)
            return x, mu, torch.zeros_like(mu), mu

    def loss_fn(logits, x, mu, logvar, beta=1.0, prior_var=None):
        v = float(x.mean())                          # per-batch value = mean of the batch
        return torch.tensor(v, requires_grad=True), {"total": v, "rec": v, "kl": 0.0, "beta": beta}

    # 4 volumes with values 1, 2, 3, 10 in batches of 3 + 1: the last batch is
    # the outlier, which is what the old last-batch behaviour reported
    x = torch.tensor([1.0, 2.0, 3.0, 10.0]).view(4, 1)
    loader = DataLoader(TensorDataset(x), batch_size=3, shuffle=False)
    parts = _epoch(Tiny(), loader, loss_fn, beta=0.5, device="cpu", opt=None)
    assert parts["total"] == pytest.approx(4.0)      # (1+2+3+10)/4, not 10
    assert parts["beta"] == pytest.approx(0.5)
    assert parts["kl"] == 0.0


def test_soft_dice_is_finite_for_half_precision_input():
    torch = pytest.importorskip("torch")
    from neurocausalpfn.vae.losses import soft_dice_loss

    # 200k voxels at sigmoid(0) = 0.5 sum to 1e5 > 65504 (float16 max)
    logits = torch.zeros(1, 200_000, dtype=torch.float16)
    target = torch.zeros(1, 200_000)
    val = soft_dice_loss(logits, target)
    assert torch.isfinite(val)
    ref = soft_dice_loss(logits.float(), target)
    assert float(val) == pytest.approx(float(ref))
