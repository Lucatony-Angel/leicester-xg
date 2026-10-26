import numpy as np
import pandas as pd
import pytest

from simulate import expected_points, p_at_least, p_at_most, simulate_goals


def test_certain_shots():
    assert (simulate_goals([0.0, 0.0]) == 0).all()
    assert (simulate_goals([1.0, 1.0, 1.0]) == 3).all()


def test_mean_matches_total_xg():
    xg = np.full(500, 0.1)
    assert simulate_goals(xg).mean() == pytest.approx(50, rel=0.02)


def test_p_values():
    sims = np.arange(100)            # 0..99
    assert p_at_least(90, sims) == pytest.approx(0.10)
    assert p_at_most(9, sims) == pytest.approx(0.10)


def test_expected_points():
    matches = pd.DataFrame({"match_id": [1, 2], "home_team": ["A", "A"], "away_team": ["B", "C"],
                            "home_score": [1, 0], "away_score": [0, 0]})
    shots = pd.DataFrame({"match_id": [1, 1, 2], "team": ["A", "A", "C"], "xg": [1.0, 0.0, 0.0]})
    table = expected_points(shots, matches, n_sims=1000)
    # Match 1: A scores for certain, B has no shots -> A 3 xPts, B 0.
    # Match 2: nobody can score -> a certain draw, 1 xPt each.
    assert table.loc["A", "xpts"] == pytest.approx(4)
    assert table.loc["B", "xpts"] == pytest.approx(0)
    assert table.loc["C", "xpts"] == pytest.approx(1)
    assert table.loc["A", "points"] == 4


def test_expected_points_ignore_row_order():
    matches = pd.DataFrame({"match_id": [1, 2, 3], "home_team": ["A", "B", "C"], "away_team": ["B", "C", "A"],
                            "home_score": [1, 0, 2], "away_score": [1, 2, 0]})
    shots = pd.DataFrame({"match_id": [1, 1, 1, 1, 2, 2, 3, 3, 3],
                          "team": ["A", "A", "A", "B", "B", "C", "C", "A", "A"],
                          "xg": [0.3, 0.05, 0.6, 0.4, 0.2, 0.5, 0.35, 0.25, 0.1]})
    forward = expected_points(shots, matches, n_sims=2000)
    shuffled = expected_points(shots.iloc[::-1], matches.iloc[[2, 0, 1]], n_sims=2000)
    pd.testing.assert_frame_equal(forward.sort_index(), shuffled.sort_index())


def test_goal_simulation_ignores_shot_order():
    xg = np.array([0.05, 0.6, 0.3, 0.12, 0.45])
    assert (simulate_goals(xg) == simulate_goals(xg[::-1])).all()


def test_expected_points_ties_sorted_the_same_on_every_platform():
    # A and B finish level on points; their order must not depend on input order or sort algorithm.
    matches = pd.DataFrame({"match_id": [1, 2], "home_team": ["B", "A"], "away_team": ["C", "C"],
                            "home_score": [1, 1], "away_score": [0, 0]})
    shots = pd.DataFrame({"match_id": [1, 2], "team": ["B", "A"], "xg": [0.3, 0.3]})
    forward = expected_points(shots, matches, n_sims=500)
    backward = expected_points(shots.iloc[::-1], matches.iloc[::-1], n_sims=500)
    assert list(forward.index) == list(backward.index) == ["A", "B", "C"]
