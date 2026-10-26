"""Route A helpers: prior batches become per-process tabular datasets with
binary outcomes, and the loss reshapes TabPFN's raw logits as its own
fine-tuning path does. The loop itself needs tabpfn >= 9.2 and a GPU."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))
import finetune_tabpfn_prior as ft  # noqa: E402


def _batch(B=3, n=40, q=8, d=5, seed=0):
    rng = np.random.default_rng(seed)
    Tc, Tq = rng.integers(0, 2, size=(B, n)), rng.integers(0, 2, size=(B, q))
    Yc = rng.uniform(0.1, 0.9, size=(B, n)); muq = rng.uniform(0.1, 0.9, size=(B, q))
    return {"Xc": rng.normal(size=(B, n, d)), "Tc": Tc, "Yc": Yc, "Xq": rng.normal(size=(B, q, d)),
            "Tq": Tq, "mu_q": muq}


def test_s_learner_datasets_have_the_treatment_column_and_binary_targets():
    b = _batch()
    sets = ft.datasets_from_batch(b, np.random.default_rng(1), learner="s")
    assert len(sets) == 3
    X, y = sets[0]
    assert X.shape == (48, 6) and X.dtype == np.float32 and set(np.unique(y)) == {0, 1}
    np.testing.assert_array_equal(X[:40, 5], b["Tc"][0]); np.testing.assert_array_equal(X[40:, 5], b["Tq"][0])
    # context outcomes already binary stay as they are
    b["Yc"] = (b["Yc"] > 0.5).astype(float)
    X2, y2 = ft.datasets_from_batch(b, np.random.default_rng(2), learner="s")[0]
    np.testing.assert_array_equal(y2[:40], b["Yc"][0].astype(int))


def test_t_learner_datasets_are_one_per_arm_without_the_treatment():
    b = _batch()
    sets = ft.datasets_from_batch(b, np.random.default_rng(1), learner="t")
    assert 3 <= len(sets) <= 6 and all(X.shape[1] == 5 for X, _ in sets)
    assert all(set(np.unique(y)) == {0, 1} for _, y in sets)


def test_ce_loss_reshapes_raw_logits_like_tabpfn():
    torch = pytest.importorskip("torch")
    Q, E, L = 6, 2, 2
    logits = torch.zeros(Q, 1, E, L); logits[:, 0, :, 1] = 3.0        # confident class 1
    y = np.ones(Q, dtype=int)
    loss = ft.ce_loss_from_raw_logits(logits, y, "cpu")
    assert float(loss) < 0.1
    y0 = np.zeros(Q, dtype=int)
    assert float(ft.ce_loss_from_raw_logits(logits, y0, "cpu")) > 2.5
