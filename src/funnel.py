"""The shot funnel: how an average team's shots get blocked, miss, or get saved.

Every shot faces three stages: avoid a block, hit the target, beat the keeper. For each
stage, an XGBoost model on the same pre-shot features (with the main model's tuned
settings) gives the league-average chance of getting through. Comparing a defending
team's actual results at each stage with these chances splits its goals prevented
(xG against minus goals conceded) into blocks, missed target and goalkeeping.

Why not post-shot xG from shot end locations: in the free data, saved shots end where the
keeper stopped them and goals end at the goal line, so end-location features partly encode
the outcome (a test model could tell goals from saves almost perfectly). The funnel only
uses pre-shot information as inputs. Run after train.py.
"""

import json

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from train import BOOLEAN, FEATURES, MODELS_DIR, N_FOLDS, PROCESSED_DIR, make_models

ON_TARGET = ["Goal", "Saved", "Saved to Post"]
STAGES = {
    # name: (which shots the stage applies to, what counts as getting through)
    "p_not_blocked": (lambda s: s.index == s.index, lambda s: s.outcome != "Blocked"),
    "p_on_target": (lambda s: s.outcome != "Blocked", lambda s: s.outcome.isin(ON_TARGET)),
    "p_goal_on_target": (lambda s: s.outcome.isin(ON_TARGET), lambda s: s.is_goal == 1),
}


def stage_model():
    # Better calibrated per bin than logistic regression here (worst decile off by 3 points, not 6).
    model = make_models()["xgboost"]
    return model.set_params(**json.loads((MODELS_DIR / "xgboost_params.json").read_text()))


def predict_stages(shots: pd.DataFrame) -> pd.DataFrame:
    """Out-of-fold stage probabilities for every shot, grouped by match."""
    shots = shots.assign(**{c: shots[c].astype(int) for c in BOOLEAN})
    out = pd.DataFrame(index=shots.index, columns=list(STAGES), dtype=float)
    for train_idx, test_idx in GroupKFold(N_FOLDS).split(shots, groups=shots.match_id):
        fit_on, predict_for = shots.iloc[train_idx], shots.iloc[test_idx]
        for name, (applies, passes) in STAGES.items():
            subset = fit_on[applies(fit_on)]
            model = stage_model().fit(subset[FEATURES], passes(subset).astype(int))
            out.loc[predict_for.index, name] = model.predict_proba(predict_for[FEATURES])[:, 1]
    return out


def main() -> None:
    shots = pd.read_parquet(PROCESSED_DIR / "shots_with_xg.parquet")
    np_shots = shots[~shots.is_penalty].reset_index(drop=True)
    stages = predict_stages(np_shots)
    funnel = pd.concat([np_shots[["shot_id"]], stages], axis=1)
    assert np.isfinite(funnel[list(STAGES)].to_numpy()).all()
    funnel.to_parquet(PROCESSED_DIR / "funnel.parquet", index=False)

    for name, (applies, passes) in STAGES.items():
        mask = applies(np_shots)
        print(f"{name}: predicted {stages.loc[mask, name].mean():.3f}, actual {passes(np_shots[mask]).mean():.3f}")


if __name__ == "__main__":
    main()
