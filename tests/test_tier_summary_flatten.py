"""TierReport.summary must carry dict-valued metrics as dotted scalar keys.
E11a's equity breakdown was computed by Tier 4 and then silently dropped by
the scalar-only filter, so the audit's verdict never reached runs.jsonl."""
from neurocausalpfn.experiments.tiers import TierReport, TierResult, evaluate_t4


def _report_with_equity():
    t4 = {"root_pehe": 0.11, "ate_bias": 0.004, "prescriptive_accuracy": 0.87,
          "n_eval": 2060, "estimator": "ridge",
          "equity": {"territory": {"all": 0.11, "0": 0.10, "1": 0.13,
                                   "max_min_ratio": 1.3, "passes": True},
                     "volume_quartile": {"all": 0.11, "0": 0.09, "3": 0.21,
                                         "max_min_ratio": 2.33, "passes": False}}}
    result = evaluate_t4(t4_result=t4)
    return TierReport(eid="E11a", results=[result])


def test_summary_flattens_equity_to_scalars():
    s = _report_with_equity().summary()
    assert s["T4.equity.territory.max_min_ratio"] == 1.3
    assert s["T4.equity.territory.passes"] is True
    assert s["T4.equity.volume_quartile.max_min_ratio"] == 2.33
    assert s["T4.equity.volume_quartile.passes"] is False
    assert s["T4.equity.volume_quartile.3"] == 0.21


def test_summary_keeps_scalars_and_drops_strings():
    s = _report_with_equity().summary()
    assert s["T4.root_pehe"] == 0.11
    assert s["T4.n_eval"] == 2060
    assert "T4.estimator" not in s                     # strings stay out
    assert all(not isinstance(v, dict) for v in s.values())


def test_summary_without_equity_unchanged():
    result = evaluate_t4(t4_result={"root_pehe": 0.2, "ate_bias": 0.0,
                                    "prescriptive_accuracy": 0.8, "n_eval": 10,
                                    "estimator": "ridge"})
    s = TierReport(eid="E3", results=[result]).summary()
    assert s["T4.root_pehe"] == 0.2 and s["passed"] is True
