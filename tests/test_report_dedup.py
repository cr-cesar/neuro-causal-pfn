"""The leaderboard deduplicates re-runs (last write wins, like finalize)."""
from neurocausalpfn.experiments.report import _leaderboard


def test_reruns_replace_stale_rows_instead_of_stacking():
    rows = [
        {"kind": "E2/x", "label": "E2[w_dice=0.5]", "seed": 0,
         "T4.root_pehe": 0.20, "passed": True},
        {"kind": "E2/x", "label": "E2[w_dice=0.5]", "seed": 1,
         "T4.root_pehe": 0.20, "passed": True},
        # the clean re-run of the same seeds must replace, not stack
        {"kind": "E2/x", "label": "E2[w_dice=0.5]", "seed": 0,
         "T4.root_pehe": 0.10, "passed": True},
        {"kind": "E2/x", "label": "E2[w_dice=0.5]", "seed": 1,
         "T4.root_pehe": 0.10, "passed": True},
    ]
    board = _leaderboard(rows)
    assert len(board) == 1
    assert board[0]["n_seeds"] == 2
    assert abs(board[0]["T4.root_pehe.mean"] - 0.10) < 1e-9
