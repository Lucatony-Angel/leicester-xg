"""Train xG models, evaluate on held-out matches, and save xG for every shot.

Models: baseline logistic (distance + angle), full logistic, XGBoost.
Splits are always grouped by match so no match is in both train and test.
"""

import json
from pathlib import Path

import duckdb
import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, GroupKFold, GroupShuffleSplit, cross_val_predict
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[1]
DB_PATH = ROOT / "data" / "leicester_xg.duckdb"
MODELS_DIR = ROOT / "models"
PROCESSED_DIR = ROOT / "data" / "processed"

SEED = 42
N_FOLDS = 5

NUMERIC = ["distance", "angle", "defenders_in_cone", "nearest_defender_dist", "gk_dist_from_line", "gk_off_center"]
BOOLEAN = ["first_time", "under_pressure", "open_goal"]
CATEGORICAL = ["body_part", "technique", "shot_type", "play_pattern", "assist_type"]
FEATURES = NUMERIC + BOOLEAN + CATEGORICAL
BASELINE = ["distance", "angle"]
TARGET = "is_goal"


def load_shots() -> pd.DataFrame:
    con = duckdb.connect(str(DB_PATH), read_only=True)
    # Fixed order: CV folds and simulations depend on row order, so rebuilds must match.
    shots = con.execute("SELECT * FROM shot_table ORDER BY match_id, period, minute, second, shot_id").df()
    con.close()
    shots[BOOLEAN] = shots[BOOLEAN].astype(int)
    return shots


def split_test_matches(shots: pd.DataFrame) -> pd.Series:
    """True for shots in the held-out 20% of matches."""
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
    _, test_idx = next(splitter.split(shots, groups=shots.match_id))
    is_test = pd.Series(False, index=shots.index)
    is_test.iloc[test_idx] = True
    return is_test


def preprocessor(scale: bool) -> ColumnTransformer:
    numeric_steps = [SimpleImputer(strategy="median")] + ([StandardScaler()] if scale else [])
    return ColumnTransformer([
        ("num", make_pipeline(*numeric_steps), NUMERIC),
        ("bool", "passthrough", BOOLEAN),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL),
    ], verbose_feature_names_out=False)


def make_models() -> dict[str, Pipeline]:
    baseline = Pipeline([
        ("prep", ColumnTransformer([("num", StandardScaler(), BASELINE)])),
        ("model", LogisticRegression(max_iter=1000)),
    ])
    logistic = Pipeline([
        ("prep", preprocessor(scale=True)),
        ("model", LogisticRegression(max_iter=2000)),
    ])
    # XGBoost handles missing values itself, but imputing keeps one preprocessor shape.
    xgb = Pipeline([
        ("prep", preprocessor(scale=False)),
        ("model", XGBClassifier(objective="binary:logistic", eval_metric="logloss",
                                random_state=SEED, n_jobs=-1)),
    ])
    return {"baseline": baseline, "logistic": logistic, "xgboost": xgb}


def tune_xgboost(model: Pipeline, X, y, groups) -> tuple[Pipeline, dict]:
    grid = {
        "model__max_depth": [3, 4, 5],
        "model__learning_rate": [0.03, 0.05, 0.1],
        "model__n_estimators": [100, 200, 400],
        "model__min_child_weight": [1, 10],
    }
    search = GridSearchCV(model, grid, scoring="neg_log_loss", cv=GroupKFold(N_FOLDS), n_jobs=1)
    search.fit(X, y, groups=groups)
    print(f"best xgboost params: {search.best_params_} (cv log loss {-search.best_score_:.4f})")
    return search.best_estimator_, search.best_params_


def main() -> None:
    shots = load_shots()
    np_shots = shots[~shots.is_penalty].reset_index(drop=True)
    np_shots["is_test"] = split_test_matches(np_shots)
    train, test = np_shots[~np_shots.is_test], np_shots[np_shots.is_test]
    print(f"train: {len(train):,} shots / {train.match_id.nunique()} matches, "
          f"test: {len(test):,} shots / {test.match_id.nunique()} matches")

    models = make_models()
    models["xgboost"], best_params = tune_xgboost(models["xgboost"], train[FEATURES], train[TARGET], train.match_id)

    MODELS_DIR.mkdir(exist_ok=True)
    test_preds = test[["shot_id", "match_id", "competition", TARGET, "statsbomb_xg"]].copy()
    cv_preds = train[["shot_id", TARGET]].copy()
    for name, model in models.items():
        # Out-of-fold predictions on the training matches, for cross-validated metrics.
        cv_preds[name] = cross_val_predict(model, train[FEATURES], train[TARGET], groups=train.match_id,
                                           cv=GroupKFold(N_FOLDS), method="predict_proba")[:, 1]
        model.fit(train[FEATURES], train[TARGET])
        test_preds[name] = model.predict_proba(test[FEATURES])[:, 1]
        joblib.dump(model, MODELS_DIR / f"{name}.joblib")
        print(f"trained {name}")

    test_preds.to_parquet(PROCESSED_DIR / "test_predictions.parquet", index=False)
    cv_preds.to_parquet(PROCESSED_DIR / "cv_predictions.parquet", index=False)
    (MODELS_DIR / "xgboost_params.json").write_text(json.dumps(best_params, indent=2))

    # xG for every shot: out-of-fold over all matches, so no shot's xG comes from a
    # model that saw that shot's match (otherwise finishing skill leaks into xG).
    final = models["xgboost"]
    np_shots["xg"] = cross_val_predict(final, np_shots[FEATURES], np_shots[TARGET], groups=np_shots.match_id,
                                       cv=GroupKFold(N_FOLDS), method="predict_proba")[:, 1]
    final.fit(np_shots[FEATURES], np_shots[TARGET])
    joblib.dump(final, MODELS_DIR / "xgboost_final.joblib")

    # Penalties get the observed penalty conversion rate as their xG.
    pens = shots[shots.is_penalty].copy()
    pens["xg"] = pens[TARGET].mean()
    pens["is_test"] = False
    print(f"penalty xg: {pens.xg.iloc[0]:.3f}")

    out = pd.concat([np_shots, pens], ignore_index=True)
    out[BOOLEAN] = out[BOOLEAN].astype(bool)
    out.to_parquet(PROCESSED_DIR / "shots_with_xg.parquet", index=False)
    print(f"saved {len(out):,} shots to shots_with_xg.parquet "
          f"(non-penalty: {out[~out.is_penalty].xg.sum():.1f} xG vs {out[~out.is_penalty][TARGET].sum()} goals)")


if __name__ == "__main__":
    main()
