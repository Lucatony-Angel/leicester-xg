"""Model QA: no leakage, sensible behavior on hand-built shots, results ordering, split logic."""

import math
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from train import BASELINE, BOOLEAN, CATEGORICAL, FEATURES, NUMERIC, split_test_matches

ROOT = Path(__file__).resolve().parents[1]

# Recorded at or after the shot, or used only for analysis: never model inputs.
NOT_FEATURES = {"is_goal", "outcome", "statsbomb_xg", "xg", "end_x", "end_y", "shot_end_location",
                "game_state", "goal_diff", "formation", "opponent_formation", "team", "opponent", "player",
                "player_id", "match_id", "competition", "is_test"}


@pytest.fixture(scope="module")
def model():
    return joblib.load(ROOT / "models" / "xgboost_final.joblib")


def shot(x=108.0, y=40.0, **overrides) -> pd.DataFrame:
    """A plain open-play shot at (x, y): right foot, no assist, no pressure, nobody in the way."""
    row = {
        "distance": math.dist((x, y), (120, 40)),
        "angle": abs(math.atan2(44 - y, 120 - x) - math.atan2(36 - y, 120 - x)),
        "defenders_in_cone": 0, "nearest_defender_dist": 5.0, "gk_dist_from_line": 1.0, "gk_off_center": 0.0,
        "first_time": 0, "under_pressure": 0, "open_goal": 0,
        "body_part": "Right Foot", "technique": "Normal", "shot_type": "Open Play",
        "play_pattern": "Regular Play", "assist_type": "Other Pass",
    }
    row.update(overrides)
    return pd.DataFrame([row])[FEATURES]


def predict(model, **kwargs) -> float:
    return float(model.predict_proba(shot(**kwargs))[0, 1])


# --- leakage guards ---

def test_no_leaky_features():
    assert not set(FEATURES) & NOT_FEATURES


def test_feature_lists_consistent():
    assert FEATURES == NUMERIC + BOOLEAN + CATEGORICAL
    assert len(FEATURES) == len(set(FEATURES)) == 14
    assert set(BASELINE) <= set(NUMERIC)


def test_saved_model_uses_exactly_the_features(model):
    assert list(model.feature_names_in_) == FEATURES


# --- behavior on hand-built shots ---

def test_predictions_are_probabilities(model):
    for x in [100, 108, 115]:
        assert 0 < predict(model, x=x) < 1


def test_closer_is_better(model):
    xgs = [predict(model, x=120 - d) for d in [6, 12, 18, 24, 30]]
    assert all(a > b for a, b in zip(xgs, xgs[1:], strict=False)), xgs


def test_central_beats_wide(model):
    assert predict(model, x=108, y=40) > predict(model, x=108, y=22)


def test_header_worse_than_foot(model):
    assert predict(model, body_part="Head") < predict(model)


def test_defenders_in_the_way_hurt(model):
    assert predict(model, defenders_in_cone=3, nearest_defender_dist=1.0) < predict(model)


def test_open_goal_helps(model):
    assert predict(model, open_goal=1, gk_dist_from_line=10.0) > predict(model)


def test_through_ball_helps(model):
    assert predict(model, assist_type="Through Ball") > predict(model)


def test_penalty_spot_shot_in_plausible_range(model):
    assert 0.1 < predict(model) < 0.6


def test_long_shot_is_unlikely(model):
    assert predict(model, x=85, defenders_in_cone=2) < 0.05


def test_unseen_category_does_not_crash(model):
    assert 0 < predict(model, technique="Something New") < 1


# --- results ---

def test_results_ordering():
    table = pd.read_csv(ROOT / "figures" / "model_results.md", sep="|", skiprows=[1], skipinitialspace=True)
    table.columns = [c.strip() for c in table.columns]
    table = table.dropna(axis=1, how="all").set_index("Model")
    table.index = table.index.str.strip()
    loss = table["Log loss"].astype(float)
    assert loss["XGBoost"] < loss["Logistic regression"] < loss["Baseline (distance + angle)"]
    assert loss["XGBoost"] < 1.02 * loss["StatsBomb xG"]          # within 2% of StatsBomb


# --- splitting ---

def test_split_is_grouped_and_deterministic():
    df = pd.DataFrame({"match_id": np.repeat(np.arange(100), 10)})
    first, second = split_test_matches(df), split_test_matches(df)
    assert first.equals(second)
    assert (df.assign(t=first).groupby("match_id").t.nunique() == 1).all()
    assert first.groupby(df.match_id).first().mean() == pytest.approx(0.2)
