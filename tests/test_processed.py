"""Checks on the committed data/processed/ files the app and analysis use.

These run anywhere (including CI) because the files are in the repo.
"""

import re

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import log_loss, roc_auc_score

from conftest import PROCESSED
from team_stats import GAME_STATES, ON_TARGET
from train import BOOLEAN, CATEGORICAL, NUMERIC

BUCKETS = {-2: "-2 or worse", -1: "-1", 0: "0", 1: "+1", 2: "+2 or better"}


# --- shots_with_xg.parquet ---

def test_shots_schema(shots):
    required = {"shot_id", "match_id", "competition", "team", "opponent", "player", "x", "y", "is_penalty",
                "game_state", "goal_diff", "formation", "opponent_formation", "statsbomb_xg", "is_goal", "is_test",
                "xg", *NUMERIC, *BOOLEAN, *CATEGORICAL}
    assert required <= set(shots.columns)


def test_shot_ids_unique(shots):
    assert shots.shot_id.is_unique


def test_xg_is_a_probability(shots):
    assert shots.xg.notna().all()
    assert shots.xg.between(0, 1).all()


def test_penalties_get_conversion_rate(shots):
    pens = shots[shots.is_penalty]
    assert pens.xg.nunique() == 1
    assert pens.xg.iloc[0] == pytest.approx(pens.is_goal.mean())


def test_penalties_never_in_test_set(shots):
    assert not shots[shots.is_penalty].is_test.any()


def test_test_set_grouped_by_match(np_shots):
    # No match may have shots in both train and test.
    assert (np_shots.groupby("match_id").is_test.nunique() == 1).all()


def test_test_set_is_about_20_percent_of_matches(np_shots):
    share = np_shots.groupby("match_id").is_test.first().mean()
    assert 0.18 < share < 0.22


def test_test_set_spans_every_league(np_shots):
    assert np_shots[np_shots.is_test].competition.nunique() == np_shots.competition.nunique()


def test_total_xg_matches_total_goals(np_shots):
    assert np_shots.xg.sum() == pytest.approx(np_shots.is_goal.sum(), rel=0.01)


def test_calibrated_by_decile(np_shots):
    bins = pd.qcut(np_shots.xg, 10, labels=False, duplicates="drop")
    by_bin = np_shots.groupby(bins).agg(pred=("xg", "mean"), actual=("is_goal", "mean"), n=("xg", "size"))
    assert (abs(by_bin.pred - by_bin.actual) < 0.03).all(), by_bin
    expected_calibration_error = (abs(by_bin.pred - by_bin.actual) * by_bin.n).sum() / by_bin.n.sum()
    assert expected_calibration_error < 0.01


def test_xg_beats_naive_and_ranks_well(np_shots):
    naive = np.full(len(np_shots), np_shots.is_goal.mean())
    assert log_loss(np_shots.is_goal, np_shots.xg) < 0.9 * log_loss(np_shots.is_goal, naive)
    assert roc_auc_score(np_shots.is_goal, np_shots.xg) > 0.78


def test_xg_agrees_with_statsbomb(np_shots):
    assert np.corrcoef(np_shots.xg, np_shots.statsbomb_xg)[0, 1] > 0.85


def test_game_state_matches_goal_diff(shots):
    expected = shots.goal_diff.clip(-2, 2).map(BUCKETS)
    assert (expected == shots.game_state).all()


def test_formation_strings(shots):
    pattern = re.compile(r"^\d(-\d)+$")
    assert shots.formation.map(lambda f: bool(pattern.match(f))).all()
    assert shots.formation.map(lambda f: sum(map(int, f.split("-")))).eq(10).all()   # 10 outfield players


def test_shots_belong_to_their_match(shots, matches):
    m = shots.merge(matches, on="match_id", suffixes=("", "_m"))
    assert len(m) == len(shots)
    in_match = (m.team == m.home_team) | (m.team == m.away_team)
    opp_in_match = (m.opponent == m.home_team) | (m.opponent == m.away_team)
    assert in_match.all() and opp_in_match.all() and (m.team != m.opponent).all()


def test_goals_never_exceed_final_score(shots, matches):
    # Own goals aren't shots, so shot goals can be below the final score but never above it.
    goals = shots.groupby(["match_id", "team"]).is_goal.sum()
    m = matches.set_index("match_id")
    for (match_id, team), g in goals.items():
        row = m.loc[match_id]
        assert g <= (row.home_score if team == row.home_team else row.away_score)


# --- matches.parquet ---

def test_matches(matches):
    assert len(matches) == 1551
    assert matches.match_id.is_unique
    assert (matches[["home_score", "away_score"]] >= 0).all().all()
    assert (matches.home_team != matches.away_team).all()


def test_full_seasons(matches):
    games = pd.concat([matches[["competition", "home_team"]].set_axis(["c", "t"], axis=1),
                       matches[["competition", "away_team"]].set_axis(["c", "t"], axis=1)])
    per_team = games.groupby(["c", "t"]).size()
    for league in ["England - Premier League", "Spain - La Liga", "Italy - Serie A"]:
        assert (per_team[league] == 38).all()


# --- formation_minutes.parquet ---

def test_formation_minutes(formation_minutes, matches):
    totals = formation_minutes.groupby(["match_id", "team"]).minutes.sum()
    assert len(totals) == 2 * len(matches)
    assert totals.between(88, 110).all()
    assert (formation_minutes.minutes > 0).all()
    assert set(formation_minutes.game_state) <= set(GAME_STATES)


# --- funnel.parquet (shot funnel stage probabilities) ---

@pytest.fixture(scope="module")
def funnel(np_shots):
    f = pd.read_parquet(PROCESSED / "funnel.parquet")
    return np_shots.merge(f, on="shot_id")


def test_funnel_covers_every_non_penalty_shot(np_shots, funnel):
    assert len(funnel) == len(np_shots)


def test_funnel_probabilities(funnel):
    for col in ["p_not_blocked", "p_on_target", "p_goal_on_target"]:
        assert funnel[col].between(0, 1).all() and funnel[col].notna().all()


@pytest.mark.parametrize("stage, applies, passes", [
    ("p_not_blocked", lambda s: s.outcome.notna(), lambda s: s.outcome != "Blocked"),
    ("p_on_target", lambda s: s.outcome != "Blocked", lambda s: s.outcome.isin(ON_TARGET)),
    ("p_goal_on_target", lambda s: s.outcome.isin(ON_TARGET), lambda s: s.is_goal == 1),
])
def test_funnel_stages_calibrated(funnel, stage, applies, passes):
    s = funnel[applies(funnel)]
    y = passes(s).astype(int)
    assert s[stage].mean() == pytest.approx(y.mean(), abs=0.005)
    bins = pd.DataFrame({"p": s[stage], "y": y}).groupby(pd.qcut(s[stage], 10, labels=False, duplicates="drop"))
    by_bin = bins.agg(p=("p", "mean"), y=("y", "mean"))
    assert (abs(by_bin.p - by_bin.y) < 0.04).all(), by_bin


def test_outcomes_known(shots):
    known = {"Goal", "Saved", "Saved to Post", "Blocked", "Off T", "Wayward", "Post", "Saved Off Target"}
    assert set(shots.outcome) <= known
    assert ((shots.outcome == "Goal") == (shots.is_goal == 1)).all()
