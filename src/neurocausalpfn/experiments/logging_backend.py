"""Experiment logging with a graceful fallback.

Section 14 asks for logging via Weights & Biases or MLflow. Neither is available
offline or without credentials, and the cluster jobs must not fail for want of a
tracking server, so the logger tries, in order:

    1. Weights & Biases  (if installed and NEUROCAUSAL_LOGGER allows it)
    2. MLflow            (if installed and allowed)
    3. a local JSON + CSV writer (always available)

The local writer is not a second-class citizen: every run always writes its
metrics to ``<out_dir>/runs.jsonl`` and appends a flat row to
``<out_dir>/metrics.csv`` regardless of the remote backend, so results are never
lost. Select a backend explicitly with the ``backend`` argument or the
``NEUROCAUSAL_LOGGER`` environment variable ("wandb", "mlflow", "local", "auto").
"""
from __future__ import annotations

import csv
import json
import os
import time
from typing import Dict, List, Optional


def _flatten(d: Dict, prefix: str = "") -> Dict:
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(_flatten(v, key + "."))
        elif isinstance(v, (list, tuple)):
            continue  # keep the flat CSV rectangular; lists live in the JSONL
        else:
            out[key] = v
    return out


class ExperimentLogger:
    def __init__(self, out_dir: str, project: str = "neuro-causal-pfn",
                 run_name: Optional[str] = None, backend: str = "auto",
                 config: Optional[Dict] = None):
        self.out_dir = out_dir
        self.project = project
        self.run_name = run_name or f"run-{int(time.time())}"
        os.makedirs(out_dir, exist_ok=True)
        self.jsonl_path = os.path.join(out_dir, "runs.jsonl")
        self.csv_path = os.path.join(out_dir, "metrics.csv")
        self._csv_cols: List[str] = []
        self.config = config or {}
        self.backend = self._select_backend(backend)
        self._remote = None
        self._init_remote()

    # ---------------------------- backend choice --------------------------- #
    @staticmethod
    def _select_backend(requested: str) -> str:
        env = os.environ.get("NEUROCAUSAL_LOGGER")
        pick = (env or requested or "auto").lower()
        if pick in ("wandb", "mlflow", "local"):
            return pick
        # auto: prefer W&B, then MLflow, then local
        if os.environ.get("WANDB_API_KEY") and _module_available("wandb"):
            return "wandb"
        if _module_available("mlflow") and os.environ.get("MLFLOW_TRACKING_URI"):
            return "mlflow"
        return "local"

    def _init_remote(self) -> None:
        try:
            if self.backend == "wandb":
                import wandb
                self._remote = wandb.init(project=self.project, name=self.run_name,
                                          config=self.config, dir=self.out_dir,
                                          reinit=True)
            elif self.backend == "mlflow":
                import mlflow
                mlflow.set_experiment(self.project)
                self._remote = mlflow.start_run(run_name=self.run_name)
                if self.config:
                    mlflow.log_params(_flatten(self.config))
        except Exception:
            # never let logging break a run; fall back to local
            self.backend = "local"
            self._remote = None

    # ------------------------------- logging ------------------------------- #
    def log_config(self, config: Dict) -> None:
        self.config.update(config)
        self._write_jsonl({"kind": "config", "run": self.run_name, "config": config})

    def log_metrics(self, metrics: Dict, step: Optional[int] = None,
                    tag: str = "metrics") -> None:
        record = {"kind": tag, "run": self.run_name, "step": step, **metrics}
        self._write_jsonl(record)
        self._write_csv(_flatten({"run": self.run_name, "tag": tag, **metrics}))
        try:
            if self.backend == "wandb" and self._remote is not None:
                self._remote.log(_flatten(metrics), step=step)
            elif self.backend == "mlflow" and self._remote is not None:
                import mlflow
                for k, v in _flatten(metrics).items():
                    if isinstance(v, (int, float)):
                        mlflow.log_metric(k.replace(".", "_"), float(v), step=step or 0)
        except Exception:
            pass

    def log_tier(self, eid: str, tier_report) -> None:
        """Convenience: log a TierReport summary."""
        self.log_metrics(tier_report.summary(), tag=f"tiers/{eid}")

    def finish(self) -> None:
        try:
            if self.backend == "wandb" and self._remote is not None:
                self._remote.finish()
            elif self.backend == "mlflow" and self._remote is not None:
                import mlflow
                mlflow.end_run()
        except Exception:
            pass

    # ------------------------------ local IO ------------------------------- #
    def _write_jsonl(self, record: Dict) -> None:
        with open(self.jsonl_path, "a") as f:
            f.write(json.dumps(record, default=_json_default) + "\n")

    def _write_csv(self, row: Dict) -> None:
        new_cols = [c for c in row if c not in self._csv_cols]
        if new_cols:
            self._csv_cols.extend(new_cols)
            self._rewrite_csv_header(row)
        else:
            with open(self.csv_path, "a", newline="") as f:
                csv.DictWriter(f, fieldnames=self._csv_cols).writerow(
                    {c: row.get(c, "") for c in self._csv_cols})

    def _rewrite_csv_header(self, row: Dict) -> None:
        rows = []
        if os.path.exists(self.csv_path):
            with open(self.csv_path, newline="") as f:
                rows = list(csv.DictReader(f))
        rows.append(row)
        with open(self.csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=self._csv_cols)
            w.writeheader()
            for r in rows:
                w.writerow({c: r.get(c, "") for c in self._csv_cols})

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.finish()
        return False


def _module_available(name: str) -> bool:
    import importlib.util
    return importlib.util.find_spec(name) is not None


def _json_default(o):
    import numpy as np
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)
