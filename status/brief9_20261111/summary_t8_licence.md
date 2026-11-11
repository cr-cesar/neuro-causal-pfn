Task 8, step 1 — TabPFN-3.5 licence (no training)
Decisions for César: whether a TabPFN-3.5 fine-tuned on R1's prior is admissible as a rival. The weights' licence allows non-commercial research fine-tuning, but §2(d) forbids using TabPFN-3.5 outputs to train, fine-tune or distill a model competitive with it, and §3(d) forbids distributing or hosting derivatives. Fine-tuning TabPFN itself for an internal comparator looks allowed; publishing the fine-tuned weights does not.
Ran: siena, ~/env_panel (the venv of the dissertation's "TabPFN-3.5" scorings: logs read "tabpfn 9.2.0: model=v3.5"), 7 Nov. No training, no model download.
Facts:
- pip show tabpfn: version 9.2.0, License-Expression Apache-2.0, author Prior Labs. The package's dist-info LICENSE is the Apache License 2.0 (202 lines). Verified (dist-info).
- Checkpoint in use: tabpfn-v3.5-20260909.safetensors (876,027,932 bytes, ~/.cache/tabpfn), from Hugging Face repo Prior-Labs/tabpfn_3_5 (model_loading.py, get_v3_5). Download requires a one-time licence acceptance (browser_auth.py). Verified (files).
- The weights carry a separate licence, "tabpfn-3-5-license-v1.0" (model card on huggingface.co/Prior-Labs/tabpfn_3_5): permissive for research and internal evaluation; "the model, its derivatives, and its outputs cannot be used for any commercial or production purpose". Reported (model card, read through a web fetch).
- Clauses (LICENSE file of the HF repo, read through a web fetch tool that summarises; quote to be checked against the full text before it is cited). Reported:
  - §1(a) Derivative: "(including but not limited to any customized, fine-tuned, retrained, or otherwise adapted version thereof)".
  - §1(c) Non-Commercial Purpose: "use for testing, evaluation, or research not tied to commercial gain, production deployment, or revenue generation"; "This includes internal benchmarking, academic research, and experimentation on private or public datasets"; not non-commercial: "(c) to train, fine-tune, or distill other models for commercial use".
  - §2(a) Grant: "solely for your Non-Commercial Purposes".
  - §2(d) Outputs: "You may not use Outputs to train, fine-tune, or distill a model that is competitive with the TABPFN-3.5 Model."
  - §3(c): a modified model must carry "a statement that you have modified the applicable TABPFN-3.5 Model".
  - §3(d) No Hosted Service: "You may not Distribute, host, or make available the TABPFN-3.5 Model or any Derivative".
Reading:
- Verified: the Apache-2.0 licence covers the tabpfn Python package only, not the v3.5 weights.
- Reported: research fine-tuning of the weights is within the grant (§1(a), §1(c), §2(a)); distribution of fine-tuned weights is not (§3(d)).
- Conjecture: §2(d) concerns training another model on TabPFN's outputs; a Neuro-CausalPFN trained on its own prior does not use TabPFN outputs, so it is not affected, but any distillation from TabPFN predictions would be. César (or Prior Labs) should confirm before step 2.
Deviations from this runbook: the HF licence text was read through a summarising fetch, not printed verbatim; read the full text once before quoting it in the paper.
