#!/usr/bin/env python3
"""Migrate an experiment history to the Theory-document numbering.

The code's experiment ids were renamed to match the Theory document's Table 7
(old E7 -> E3 backbone, old E3 -> E4 dimensionality, old E4 -> E5 ARD, old
E5a -> E8 DAFT, old E9a/b -> E7a/b fusion, old E10a/c/b -> E9a/b/c Arms C/D,
old E8a/b/c -> E10a/b/c Arm E, old E11 -> E11a). Histories written before the
rename keep the old labels; this script rewrites them in place (runs.jsonl and
context.json under the given out-root) and leaves a .pre_rename.bak copy.

Run it ONCE, after every job launched with the old code has finished:

    python scripts/migrate_experiment_ids.py outputs/experiments
"""
import json
import os
import shutil
import sys

# old id -> new id. Order does not matter: labels are matched whole.
MAP = {"E7": "E3", "E3": "E4", "E4": "E5", "E5a": "E8",
       "E9a": "E7a", "E9b": "E7b",
       "E10a": "E9a", "E10c": "E9b", "E10b": "E9c",
       "E8a": "E10a", "E8b": "E10b", "E8c": "E10c",
       "E11": "E11a"}


def migrate_label(label: str) -> str:
    """E7[backbone=cnn] -> E3[backbone=cnn]; whole-id match only."""
    for old, new in MAP.items():
        if label == old:
            return new
        if label.startswith(old + "["):
            return new + label[len(old):]
    return label


def main(out_root: str) -> None:
    changed_total = 0

    runs = os.path.join(out_root, "runs.jsonl")
    if os.path.exists(runs):
        shutil.copy(runs, runs + ".pre_rename.bak")
        lines, changed = [], 0
        with open(runs) as f:
            for line in f:
                if not line.strip():
                    continue
                r = json.loads(line)
                for key in ("label", "eid"):
                    if key in r:
                        new = migrate_label(str(r[key]))
                        if new != r[key]:
                            r[key] = new
                            changed += 1
                lines.append(json.dumps(r))
        with open(runs, "w") as f:
            f.write("\n".join(lines) + "\n")
        print(f"runs.jsonl: {changed} campos renombrados (backup .pre_rename.bak)")
        changed_total += changed

    ctx_path = os.path.join(out_root, "context.json")
    if os.path.exists(ctx_path):
        shutil.copy(ctx_path, ctx_path + ".pre_rename.bak")
        with open(ctx_path) as f:
            ctx = json.load(f)
        for section in ("winners", "ranking"):
            if section in ctx:
                ctx[section] = {migrate_label(k): ([migrate_label(v) for v in vals]
                                                   if isinstance(vals, list)
                                                   else migrate_label(str(vals)))
                                for k, vals in ctx[section].items()}
        with open(ctx_path, "w") as f:
            json.dump(ctx, f, indent=2)
        print("context.json migrado")

    if changed_total:
        print("Regenera el leaderboard con:")
        print(f"  python -c \"from neurocausalpfn.experiments.report import build_report; "
              f"build_report('{out_root}')\"")
    else:
        print("nada que migrar")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "outputs/experiments")
