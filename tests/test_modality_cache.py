"""The runner's modality cache: identical VAE trainings run once per process.

The concrete case it protects: in the E2 w_dice sweep the disconnectome VAE
(MSE loss; w_dice forced to 1.0) is identical across the lambda variants and
must be reused, while any change that reaches the effective config (zdim,
backbone, seed) must miss the cache and retrain.
"""
from neurocausalpfn.experiments import runner

_FAST = {"epochs": 1, "n_synth": 8, "resolution": [24, 28, 24],
         "batch_size": 4, "channels": [8, 16, 32, 64]}


def _train(meta, seed, out, tmp_path):
    return runner._train_modality("prototype", "disconnectome", 8, meta, seed,
                                  str(tmp_path / out), dict(_FAST))


def test_disconnectome_reused_across_w_dice_variants(tmp_path):
    runner._MODALITY_CACHE.clear()
    a1 = _train({"backbone": "cnn", "w_dice": 0.1}, 0, "a", tmp_path)
    a2 = _train({"backbone": "cnn", "w_dice": 0.5}, 0, "b", tmp_path)
    assert a2 is a1          # w_dice never reaches the disconnectome objective
    assert len(runner._MODALITY_CACHE) == 1


def test_config_changes_miss_the_cache(tmp_path):
    runner._MODALITY_CACHE.clear()
    a1 = _train({"backbone": "cnn"}, 0, "a", tmp_path)
    a2 = _train({"backbone": "cnn"}, 1, "b", tmp_path)          # other seed
    assert a2 is not a1
    a3 = runner._train_modality("prototype", "disconnectome", 16,   # other zdim
                                {"backbone": "cnn"}, 0, str(tmp_path / "c"), dict(_FAST))
    assert a3 is not a1
    assert len(runner._MODALITY_CACHE) == 3
