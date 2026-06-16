import sys; sys.path.insert(0, "src")
from neurocausalpfn.train.train_pfn import prototype_config, run_pfn, quick_eval

cfg = prototype_config()
cfg["prior"] = {"kind": "intersynth", "atlas_dir": "data/atlases",
                "modality": "receptor", "pool_size": 60}
cfg["pfn"]["iters"] = 60
cfg["pfn"]["batch_size"] = 4
cfg["pfn"]["context_max"] = 128
cfg["pfn"]["n_query"] = 8
cfg["out_dir"] = "outputs/pfn_intersynth_realatlas"

model, hist = run_pfn(cfg)
print("ultimas perdidas:", [round(h["loss"], 3) for h in hist[-5:]])
print("root_PEHE:", round(quick_eval(model, cfg, n_eval=12)["root_pehe"], 4))