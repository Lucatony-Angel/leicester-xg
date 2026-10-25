"""Unit tests for the team and formation math, on tiny hand-built leagues."""

import numpy as np
import pandas as pd
import pytest

from team_stats import FLIP_STATE, GAME_STATES, formation_profile, ordinal, rank, share_vs_league, team_breakdown

LEAGUE = "Test League"


def make_shot(team, opponent, xg, goal, match_id=1, penalty=False, pattern="Regular Play", assist="None",
              formation="4-4-2", opponent_formation="4-3-3", state="0"):
    return {"shot_id": f"{team}-{match_id}-{xg}-{goal}-{penalty}", "competition": LEAGUE, "match_id": match_id,
            "team": team, "opponent": opponent, "xg": xg, "is_goal": goal, "is_penalty": penalty,
            "play_pattern": pattern, "assist_type": assist, "formation": formation,
            "opponent_formation": opponent_formation, "game_state": state}


@pytest.fixture
def league():
    matches = pd.DataFrame({"competition": LEAGUE, "match_id": [1, 2], "home_team": ["A", "B"],
                            "away_team": ["B", "A"]})
    shots = pd.DataFrame([
        make_shot("A", "B", 0.5, 1, 1, pattern="From Counter"),
        make_shot("A", "B", 0.3, 1, 1),
        make_shot("A", "B", 0.2, 0, 2),
        make_shot("A", "B", 0.76, 1, 2, penalty=True),     # excluded from breakdowns
        make_shot("B", "A", 0.4, 0, 1),
        make_shot("B", "A", 0.1, 1, 2),
    ])
    return shots, matches


def test_team_breakdown(league):
    shots, matches = league
    t = team_breakdown(shots, matches, LEAGUE)
    a = t.loc["A"]
    assert a.matches == 2
    assert a.shots == 3 and a.shots_per_match == 1.5
    assert a.xg == pytest.approx(1.0) and a.goals == 2 and a.finishing == pytest.approx(1.0)
    assert a.xg_per_shot == pytest.approx(1 / 3)
    assert a.shots_against == 2 and a.xga == pytest.approx(0.5) and a.goals_against == 1
    assert a.goals_prevented == pytest.approx(-0.5)
    assert a.penalties == 1 and a.penalty_goals == 1
    assert a.xg_diff == pytest.approx(0.5)
    assert t.loc["B"].penalties == 0


def test_breakdown_is_zero_sum(league):
    t = team_breakdown(*league, LEAGUE)
    assert t.xg_diff.sum() == pytest.approx(0)
    assert t.goals.sum() == t.goals_against.sum()


def test_share_vs_league(league):
    shots, _ = league
    out = share_vs_league(shots, "play_pattern", team="A", competition=LEAGUE)
    assert out.team_share.sum() == pytest.approx(1)
    assert out.loc["From Counter", "team_share"] == pytest.approx(0.5)
    assert out.loc["From Counter", "rest_of_league_share"] == 0


@pytest.mark.parametrize("n, expected", [(1, "1st"), (2, "2nd"), (3, "3rd"), (4, "4th"), (11, "11th"),
                                         (12, "12th"), (13, "13th"), (21, "21st"), (22, "22nd"), (111, "111th")])
def test_ordinal(n, expected):
    assert ordinal(n) == expected


def test_rank_ties_and_direction():
    t = pd.DataFrame({"v": [3, 1, 3]}, index=["A", "B", "C"])
    assert rank(t, "v", "A") == 1 and rank(t, "v", "C") == 1 and rank(t, "v", "B") == 3
    assert rank(t, "v", "B", ascending=True) == 1


def test_flip_state_is_symmetric():
    assert all(FLIP_STATE[FLIP_STATE[s]] == s for s in GAME_STATES)
    assert FLIP_STATE["0"] == "0" and FLIP_STATE["+1"] == "-1"


@pytest.fixture
def formations():
    # One match, mirrored like real data. X plays 4-4-2: 90 minutes level (1 xG), then 90 minutes
    # winning by 1 (3 xG). Y plays 4-3-3: 90 minutes level (2 xG), then 90 minutes losing by 1.
    # League-wide, half of all minutes are level and a quarter each are +1 and -1.
    minutes = pd.DataFrame({
        "competition": LEAGUE, "match_id": 1,
        "team": ["X", "X", "Y", "Y"], "formation": ["4-4-2", "4-4-2", "4-3-3", "4-3-3"],
        "game_state": ["0", "+1", "0", "-1"], "minutes": [90.0, 90.0, 90.0, 90.0],
    })
    shots = pd.DataFrame([
        make_shot("X", "Y", 1.0, 1, formation="4-4-2", opponent_formation="4-3-3", state="0"),
        make_shot("X", "Y", 3.0, 2, formation="4-4-2", opponent_formation="4-3-3", state="+1"),
        make_shot("Y", "X", 2.0, 1, formation="4-3-3", opponent_formation="4-4-2", state="0"),
    ])
    return shots, minutes


def test_formation_raw_rates(formations):
    p = formation_profile(*formations, adjust=False)
    assert p.loc["4-4-2", "xg_per90"] == pytest.approx(2.0)       # 4 xG in 180 minutes
    assert p.loc["4-3-3", "xg_per90"] == pytest.approx(1.0)       # 2 xG in 180 minutes


def test_formation_game_state_adjustment(formations):
    # 4-4-2 never played at -1, so the league weights (0: 0.5, +1: 0.25) are renormalized to 2/3 and 1/3:
    # level counts twice as much as winning, instead of 50/50 as in its own minutes.
    p = formation_profile(*formations)
    assert p.loc["4-4-2", "xg_per90"] == pytest.approx(2 / 3 * 1.0 + 1 / 3 * 3.0)


def test_formation_single_game_state(formations):
    p = formation_profile(*formations, game_state="0")
    assert p.loc["4-4-2", "xg_per90"] == pytest.approx(1.0)
    assert p.loc["4-4-2", "minutes"] == 90


def test_shots_against_use_defending_formation_and_flipped_state(formations):
    # Y's shot counts against 4-4-2. X's +1 shot counts against 4-3-3 at Y's -1.
    p = formation_profile(*formations, adjust=False)
    assert p.loc["4-4-2", "xga_per90"] == pytest.approx(2.0 / 180 * 90)
    assert p.loc["4-3-3", "xga_per90"] == pytest.approx(4.0 / 180 * 90)
    losing = formation_profile(*formations, game_state="-1")
    assert losing.loc["4-3-3", "xga_per90"] == pytest.approx(3.0)
    assert np.isnan(losing.loc["4-3-3", "xg_per_shot"])          # no shots of its own: NaN, not a crash


def test_team_filters(formations):
    shots, minutes = formations
    only_x = formation_profile(shots, minutes, team="X")
    without_x = formation_profile(shots, minutes, exclude_team="X")
    assert list(only_x.index) == ["4-4-2"] and list(without_x.index) == ["4-3-3"]
