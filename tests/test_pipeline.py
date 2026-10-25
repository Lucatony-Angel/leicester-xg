"""Run the pipeline scripts into a temp folder and check they reproduce the committed outputs."""

import json
import shutil

import duckdb
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

import analysis
import champions
import evaluate
import formations
import freeze_frame
import funnel
import train
from conftest import FIGURES, PROCESSED


@pytest.fixture
def out_dir(tmp_path, monkeypatch):
    # The scripts write to FIGURES_DIR; each module holds its own reference to it.
    for module in (analysis, champions, formations, evaluate):
        monkeypatch.setattr(module, "FIGURES_DIR", tmp_path)
    return tmp_path


def assert_same(out_dir, name):
    assert (out_dir / name).read_text() == (FIGURES / name).read_text(), f"{name} is out of date: rerun the script"


def test_analysis_reproduces_results(out_dir):
    analysis.main()
    assert_same(out_dir, "leicester_results.md")
    for png in ["team_scatter.png", "goals_vs_xg.png", "simulation.png", "expected_points.png",
                "leicester_shot_map.png"]:
        assert (out_dir / png).stat().st_size > 10_000


def test_champions_reproduces_results(out_dir, monkeypatch):
    monkeypatch.setattr(champions, "PROCESSED_DIR", out_dir)
    champions.main()
    assert_same(out_dir, "champions_results.md")
    pd.testing.assert_frame_equal(pd.read_parquet(out_dir / "champions.parquet"),
                                  pd.read_parquet(PROCESSED / "champions.parquet"))
    assert (out_dir / "champions.png").stat().st_size > 10_000


def test_formations_reproduces_results(out_dir):
    formations.main()
    assert_same(out_dir, "formations_results.md")
    assert (out_dir / "formations.png").stat().st_size > 10_000
    assert (out_dir / "formation_442_teams.png").stat().st_size > 10_000


def test_evaluate_reproduces_results(out_dir):
    if not (PROCESSED / "test_predictions.parquet").exists():
        pytest.skip("prediction files are created by src/train.py and not committed")
    evaluate.main()
    assert_same(out_dir, "model_results.md")
    assert_same(out_dir, "example_shots.md")
    for png in ["calibration.png", "shap_summary.png", "shap_vardy_goal.png"]:
        assert (out_dir / png).stat().st_size > 10_000


def test_funnel_reproduces_committed_file(tmp_path, monkeypatch):
    shutil.copy(PROCESSED / "shots_with_xg.parquet", tmp_path)
    monkeypatch.setattr(funnel, "PROCESSED_DIR", tmp_path)
    funnel.main()
    pd.testing.assert_frame_equal(pd.read_parquet(tmp_path / "funnel.parquet"),
                                  pd.read_parquet(PROCESSED / "funnel.parquet"))


def test_models_train_on_a_sample(shots):
    sample = shots[~shots.is_penalty].sample(3000, random_state=0)
    sample = sample.assign(**{c: sample[c].astype(int) for c in train.BOOLEAN})
    for name, model in train.make_models().items():
        model.fit(sample[train.FEATURES], sample[train.TARGET])
        p = model.predict_proba(sample[train.FEATURES])[:, 1]
        assert p.min() > 0 and p.max() < 1, name
        assert roc_auc_score(sample[train.TARGET], p) > 0.7, name


def test_saved_params_are_from_the_grid():
    params = json.loads((train.MODELS_DIR / "xgboost_params.json").read_text())
    assert params["model__max_depth"] in [3, 4, 5]
    assert params["model__learning_rate"] in [0.03, 0.05, 0.1]
    assert params["model__n_estimators"] in [100, 200, 400]


def test_build_freeze_frame_table():
    frame = [{"location": [114, 40], "teammate": False, "position": {"name": "Center Back"}},
             {"location": [118, 41], "teammate": False, "position": {"name": "Goalkeeper"}}]
    con = duckdb.connect()
    con.register("shots_df", pd.DataFrame({"shot_id": ["a", "b"], "x": [108.0, 100.0], "y": [40.0, 40.0],
                                           "freeze_frame": [json.dumps(frame), None]}))
    con.execute("CREATE TABLE shots AS SELECT * FROM shots_df")
    freeze_frame.build_freeze_frame_table(con)
    out = con.execute("SELECT * FROM freeze_frame_features ORDER BY shot_id").df().set_index("shot_id")
    assert out.loc["a", "defenders_in_cone"] == 1 and out.loc["a", "gk_dist_from_line"] == pytest.approx(2)
    assert out.loc["b", "defenders_in_cone"] == 0 and pd.isna(out.loc["b", "gk_dist_from_line"])


@pytest.mark.slow
def test_rebuild_and_retrain_reproduce_committed_data(tmp_path, monkeypatch):
    """Full rebuild from the raw download: about 4 minutes. Run with `pytest -m slow`."""
    import build_db

    if not build_db.RAW_DIR.joinpath("matches.parquet").exists():
        pytest.skip("raw StatsBomb download not present (run src/download.py)")
    (tmp_path / "data" / "processed").mkdir(parents=True)
    monkeypatch.setattr(build_db, "ROOT", tmp_path)
    monkeypatch.setattr(build_db, "DB_PATH", tmp_path / "rebuilt.duckdb")
    build_db.main()

    processed = tmp_path / "data" / "processed"
    for name, key in [("matches.parquet", ["match_id"]),
                      ("formation_minutes.parquet", ["match_id", "team", "formation", "game_state"])]:
        new = pd.read_parquet(processed / name).sort_values(key).reset_index(drop=True)
        old = pd.read_parquet(PROCESSED / name).sort_values(key).reset_index(drop=True)
        pd.testing.assert_frame_equal(new, old, check_dtype=False)

    monkeypatch.setattr(train, "DB_PATH", tmp_path / "rebuilt.duckdb")
    monkeypatch.setattr(train, "MODELS_DIR", tmp_path / "models")
    monkeypatch.setattr(train, "PROCESSED_DIR", processed)
    train.main()
    new = pd.read_parquet(processed / "shots_with_xg.parquet").set_index("shot_id").sort_index()
    old = pd.read_parquet(PROCESSED / "shots_with_xg.parquet").set_index("shot_id").sort_index()
    pd.testing.assert_series_equal(new.xg, old.xg)
    assert (tmp_path / "models" / "xgboost_params.json").read_text() == \
        (train.MODELS_DIR / "xgboost_params.json").read_text()
