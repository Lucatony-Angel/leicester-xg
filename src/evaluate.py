"""Metrics table, calibration plot, StatsBomb comparison and SHAP charts.

Run after train.py. Writes figures/ and figures/model_results.md.
"""

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

from train import BOOLEAN, FEATURES, MODELS_DIR, PROCESSED_DIR, ROOT, TARGET

FIGURES_DIR = ROOT / "figures"
MODELS = {"baseline": "Baseline (distance + angle)", "logistic": "Logistic regression",
          "xgboost": "XGBoost", "statsbomb_xg": "StatsBomb xG"}
# Fixed categorical order from the dataviz reference palette.
COLORS = {"xgboost": "#2a78d6", "logistic": "#eb6834", "statsbomb_xg": "#1baf7a"}
INK, MUTED = "#2b2b2b", "#8a8a8a"


def metrics(y, p) -> dict:
    return {"log_loss": log_loss(y, p), "brier": brier_score_loss(y, p), "roc_auc": roc_auc_score(y, p),
            "total_xg": p.sum(), "goals": y.sum()}


def results_table(test: pd.DataFrame, cv: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col, label in MODELS.items():
        row = {"model": label, **metrics(test[TARGET], test[col])}
        row["cv_log_loss"] = log_loss(cv[TARGET], cv[col]) if col in cv else np.nan
        rows.append(row)
    return pd.DataFrame(rows).set_index("model")


def style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(MUTED)
    ax.tick_params(colors=INK)
    ax.grid(alpha=0.25)


def calibration_bins(y: pd.Series, p: pd.Series, n_bins: int = 10) -> pd.DataFrame:
    """Mean prediction, actual goal rate and a 95% interval for bins with equal shot counts."""
    bins = pd.qcut(p, n_bins, labels=False, duplicates="drop")
    out = pd.DataFrame({"y": y, "p": p}).groupby(bins).agg(pred=("p", "mean"), actual=("y", "mean"), n=("y", "size"))
    out["ci"] = 1.96 * np.sqrt(out.actual * (1 - out.actual) / out.n)
    return out


def calibration_plot(test: pd.DataFrame, table: pd.DataFrame) -> None:
    # One panel per model: three near-identical lines on one chart are hard to tell apart.
    panels = ["xgboost", "logistic", "statsbomb_xg"]
    bins = {col: calibration_bins(test[TARGET], test[col]) for col in panels}
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.8), sharex=True, sharey=True)
    for ax, col in zip(axes, panels, strict=True):
        ax.plot([0, 0.5], [0, 0.5], color=MUTED, linestyle="--", linewidth=1, label="Perfect calibration")
        b = bins[col]
        ax.errorbar(b.pred, b.actual, yerr=b.ci, color=COLORS[col], marker="o", markersize=5, linewidth=2,
                    elinewidth=1, capsize=3, label="Actual goal rate (95% interval)")
        style(ax)
        ax.set_aspect("equal")
        ax.set_xlim(0, 0.5)
        ax.set_ylim(0, 0.5)
        ax.set_xlabel("Predicted xG", color=INK)
        row = table.loc[MODELS[col]]
        ax.set_title(f"{MODELS[col]}\n{row.total_xg:.0f} xG vs {row.goals:.0f} goals", loc="left",
                     fontsize=10, color=INK)
    axes[0].set_ylabel("Actual goal rate", color=INK)
    axes[0].legend(frameon=False, loc="upper left", fontsize=8.5)
    within = {col: int(((b.actual - b.pred).abs() <= b.ci).sum()) for col, b in bins.items()}
    fig.suptitle(f"Predicted xG tracks real goal rates on held-out matches: {within['xgboost']} of "
                 f"{len(bins['xgboost'])} XGBoost bins sit within their 95% interval of the diagonal",
                 x=0.05, ha="left", fontsize=12, color=INK)
    fig.text(0.05, -0.06, "Held-out test set: 7,554 shots from 311 matches, in 10 bins with equal shot counts. "
             "Bars show the 95% interval for each bin's goal rate.", fontsize=8.5, color=MUTED)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(FIGURES_DIR / "calibration.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def xgb_explainer(model_file: str):
    pipeline = joblib.load(MODELS_DIR / model_file)
    prep, booster = pipeline.named_steps["prep"], pipeline.named_steps["model"]
    return prep, shap.TreeExplainer(booster)


def model_inputs(prep, shots: pd.DataFrame) -> pd.DataFrame:
    shots = shots.assign(**{c: shots[c].astype(int) for c in BOOLEAN})
    return pd.DataFrame(prep.transform(shots[FEATURES]), columns=prep.get_feature_names_out()).astype(float)


def shap_summary(test_shots: pd.DataFrame) -> None:
    prep, explainer = xgb_explainer("xgboost.joblib")
    X = model_inputs(prep, test_shots)
    values = explainer(X)
    plt.figure()
    shap.plots.beeswarm(values, max_display=15, show=False)
    plt.title("Distance, angle and defenders in the cone drive the model most", loc="left", fontsize=11)
    plt.savefig(FIGURES_DIR / "shap_summary.png", dpi=150, bbox_inches="tight")
    plt.close()


def example_shots(shots: pd.DataFrame) -> pd.DataFrame:
    """Vardy's least likely goal, Mahrez's least likely goal, and Leicester's biggest miss."""
    lei = shots[(shots.team == "Leicester City") & ~shots.is_penalty]
    vardy_goals = lei[lei.player.str.contains("Vardy") & (lei[TARGET] == 1)]
    mahrez_goals = lei[lei.player.str.contains("Mahrez") & (lei[TARGET] == 1)]
    misses = lei[lei[TARGET] == 0]
    return pd.concat([
        vardy_goals.nsmallest(1, "xg").assign(label="Vardy's least likely goal"),
        mahrez_goals.nsmallest(1, "xg").assign(label="Mahrez's least likely goal"),
        misses.nlargest(1, "xg").assign(label="Leicester's biggest miss"),
    ])


def shap_examples(shots: pd.DataFrame) -> None:
    prep, explainer = xgb_explainer("xgboost_final.joblib")
    examples = example_shots(shots)
    X = model_inputs(prep, examples)
    values = explainer(X)
    for i, (_, shot) in enumerate(examples.iterrows()):
        plt.figure()
        shap.plots.waterfall(values[i], max_display=10, show=False)
        plt.title(f"{shot.label}: {shot.player.split()[-1]} vs {shot.opponent}, {shot.match_date} "
                  f"({shot.xg:.2f} xG)", loc="left", fontsize=10)
        slug = shot.label.split("'")[0].lower() + ("_miss" if "miss" in shot.label else "_goal")
        plt.savefig(FIGURES_DIR / f"shap_{slug}.png", dpi=150, bbox_inches="tight")
        plt.close()
    table = examples[["label", "player", "opponent", "match_date", "minute", "body_part", "distance",
                      "nearest_defender_dist", "xg", "statsbomb_xg"]]
    (FIGURES_DIR / "example_shots.md").write_text(
        "# Example shots explained with SHAP\n\nGenerated by `src/evaluate.py`.\n\n"
        + table.round(3).to_markdown(index=False) + "\n")
    print(table.to_string(index=False))


def main() -> None:
    FIGURES_DIR.mkdir(exist_ok=True)
    test = pd.read_parquet(PROCESSED_DIR / "test_predictions.parquet")
    cv = pd.read_parquet(PROCESSED_DIR / "cv_predictions.parquet")
    shots = pd.read_parquet(PROCESSED_DIR / "shots_with_xg.parquet")

    table = results_table(test, cv)
    display = table.rename(columns={"log_loss": "Log loss", "brier": "Brier", "roc_auc": "ROC AUC",
                                    "total_xg": "Total xG", "goals": "Goals", "cv_log_loss": "CV log loss"})
    display = display.rename_axis("Model").round({"Total xG": 1}).round(4).astype(object).fillna("—")
    (FIGURES_DIR / "model_results.md").write_text(display.to_markdown())
    print(table.round(4).to_string())
    xgb, sb = table.loc["XGBoost"], table.loc["StatsBomb xG"]
    print(f"\nXGBoost log loss is {100 * (xgb.log_loss / sb.log_loss - 1):.1f}% above StatsBomb's; "
          f"AUC {xgb.roc_auc:.3f} vs {sb.roc_auc:.3f}")

    calibration_plot(test, table)
    shap_summary(shots[shots.is_test])
    shap_examples(shots)
    print(f"saved figures to {FIGURES_DIR}")


if __name__ == "__main__":
    main()
