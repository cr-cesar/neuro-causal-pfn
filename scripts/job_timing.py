"""Where the time of finished SGE jobs went: queue wait against run time.

Reads the job ids from SGE log files (<name>.o<jobid>[.<task>]), asks qacct
for each one and prints, per job, the wait in the queue, the run time, the
node and the training time the per-fold trainer printed itself
("... (0.42 h)"). The summary says whether the bottleneck is the scheduler
(long waits, few jobs at a time) or the compute (long runs), which decides
what speeds a re-run up. Read-only: qacct and the log files; seconds on a
login node.

    python scripts/job_timing.py ~/Scratch/neuro-causal-pfn-perfold/e11br*.o*
    python scripts/job_timing.py --last 60 ~/Scratch/neuro-causal-pfn-perfold/e3r*.o*
"""
import argparse
import datetime as dt
import os
import re
import statistics
import subprocess
import sys

_FMT = "%a %b %d %H:%M:%S %Y"


def parse_records(text: str) -> list:
    """Every record of a qacct output, as dicts."""
    rec, recs = {}, []
    for line in text.splitlines():
        if line.startswith("=" * 10):
            if rec:
                recs.append(rec)
            rec = {}
            continue
        parts = line.split(None, 1)
        if len(parts) == 2:
            rec[parts[0]] = parts[1].strip()
    if rec:
        recs.append(rec)
    return recs


def pick(recs: list) -> dict:
    """The successful record of a job (a rescheduled job has several)."""
    done = [r for r in recs if r.get("failed", "0").split()[0] == "0"]
    return (done or recs or [{}])[-1]


def parse_qacct(text: str) -> dict:
    """The last record of a qacct -j output (a rescheduled job has several)."""
    rec, recs = {}, []
    for line in text.splitlines():
        if line.startswith("=" * 10):
            if rec:
                recs.append(rec)
            rec = {}
            continue
        parts = line.split(None, 1)
        if len(parts) == 2:
            rec[parts[0]] = parts[1].strip()
    if rec:
        recs.append(rec)
    done = [r for r in recs if r.get("failed", "0").split()[0] == "0"]
    return (done or recs or [{}])[-1]


def _t(s: str):
    try:
        return dt.datetime.strptime(s.strip(), _FMT)
    except (ValueError, AttributeError):
        return None


def job_row(log_path: str, records: dict = None) -> dict:
    """``records``: jobnumber -> qacct records from one batched query; without
    it, one qacct call for this job (slow on a large accounting file)."""
    m = re.search(r"\.o(\d+)(?:\.(\d+))?$", log_path)
    if not m:
        return {}
    jid = m.group(1)
    if records is not None:
        r = pick(records.get(jid, []))
    else:
        out = subprocess.run(["qacct", "-j", jid], capture_output=True, text=True).stdout
        r = parse_qacct(out)
    q, s, e = _t(r.get("qsub_time", "")), _t(r.get("start_time", "")), _t(r.get("end_time", ""))
    train_h = None
    try:
        with open(log_path, errors="replace") as fh:
            for line in fh:
                mm = re.search(r"\((\d+\.\d+) h\)\s*$", line)
                if mm:
                    train_h = float(mm.group(1))
    except OSError:
        pass
    return {"job": jid, "node": r.get("hostname", "?").split(".")[0],
            "wait_h": (s - q).total_seconds() / 3600 if q and s else None,
            "run_h": (e - s).total_seconds() / 3600 if s and e else None,
            "train_h": train_h, "start": s, "end": e,
            "maxvmem": r.get("maxvmem", "?"), "failed": r.get("failed", "?")}


def summarise(rows):
    def stats(key):
        v = [r[key] for r in rows if r.get(key) is not None]
        if not v:
            return "n/a"
        return f"median {statistics.median(v):.2f} h, min {min(v):.2f}, max {max(v):.2f} (n={len(v)})"

    print(f"queue wait : {stats('wait_h')}")
    print(f"run time   : {stats('run_h')}")
    print(f"train+encode printed by the trainer: {stats('train_h')}")
    spans = [(r["start"], r["end"]) for r in rows if r.get("start") and r.get("end")]
    if spans:
        events = sorted([(a, 1) for a, _ in spans] + [(b, -1) for _, b in spans])
        cur = peak = 0
        for _, d in events:
            cur += d
            peak = max(peak, cur)
        first, last = min(a for a, _ in spans), max(b for _, b in spans)
        busy = sum((b - a).total_seconds() for a, b in spans)
        wall = (last - first).total_seconds()
        print(f"jobs running at once: peak {peak}, mean {busy / wall:.1f} over {wall / 3600:.1f} h of wall clock")
    nodes = {}
    for r in rows:
        nodes[r["node"]] = nodes.get(r["node"], 0) + 1
    print("nodes:", ", ".join(f"{k} x{v}" for k, v in sorted(nodes.items(), key=lambda kv: -kv[1])[:8]))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("logs", nargs="+")
    ap.add_argument("--last", type=int, default=0, help="only the N most recent log files")
    ap.add_argument("--rows", action="store_true", help="print one line per job")
    ap.add_argument("--days", type=int, default=7, help="accounting window passed to qacct -d")
    ap.add_argument("--name", default=None, help="job-name pattern for qacct (default: common prefix of the logs + *)")
    ap.add_argument("--timeout", type=int, default=300)
    args = ap.parse_args(argv)
    logs = sorted(set(args.logs), key=lambda p: os.path.getmtime(p) if os.path.exists(p) else 0)
    if args.last:
        logs = logs[-args.last:]
    # one qacct scan for every job, by name pattern and limited in days: one
    # call per job re-reads the whole accounting file each time
    names = [re.sub(r"\.o\d+(?:\.\d+)?$", "", os.path.basename(p)) for p in logs]
    prefix = os.path.commonprefix(names) or names[0]
    pattern = args.name or (prefix + "*")
    print(f"qacct -d {args.days} -j '{pattern}' (one scan) ...", flush=True)
    try:
        out = subprocess.run(["qacct", "-d", str(args.days), "-j", pattern], capture_output=True,
                             text=True, timeout=args.timeout).stdout
    except subprocess.TimeoutExpired:
        sys.exit(f"qacct took more than {args.timeout} s; lower --days")
    records = {}
    for rec in parse_records(out):
        records.setdefault(rec.get("jobnumber", ""), []).append(rec)
    rows = [r for r in (job_row(p, records) for p in logs) if r]
    if not rows:
        sys.exit("no SGE job id found in the given file names (expected <name>.o<jobid>)")
    if args.rows:
        for r in rows:
            f = lambda x: "  -  " if x is None else f"{x:5.2f}"
            print(f"{r['job']:>8} {r['node']:>16} wait {f(r['wait_h'])} h  run {f(r['run_h'])} h  "
                  f"train {f(r['train_h'])} h  mem {r['maxvmem']}")
    summarise(rows)


if __name__ == "__main__":
    main()
