"""Keeps the test suite polite on shared login nodes.

Cluster login nodes are shared and policed (on UCL Myriad, Arbiter2 enforces
under 6 cores / 30 GB per user and applies escalating penalties). PyTorch
otherwise grabs every visible core for its intra-op thread pool, so a plain
`pytest` run trips the monitor. When the hostname looks like a login node
(login13, login02.cluster, ...) — or NEUROCAUSAL_LOGIN_SAFE is set — cap the
numeric libraries at 2 threads. Compute-node jobs and laptops are unaffected.

The real place to run the suite on a cluster is a compute node:
    qsub scripts/run_tests_myriad.qsub.sh
"""
import os
import re
import socket


def _on_login_node() -> bool:
    if os.environ.get("NEUROCAUSAL_LOGIN_SAFE"):
        return True
    host = socket.gethostname() or ""
    return re.match(r"login\d+", host) is not None


if _on_login_node():
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS",
                "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(var, "2")
    try:
        import torch

        torch.set_num_threads(2)
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass    # already initialised; intra-op cap above still applies
    except ImportError:
        pass
