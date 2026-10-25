import streamlit as st

from common import FIGURES_DIR, credit

st.title("About the model")
st.write(
    "An XGBoost classifier predicts the chance that each non-penalty shot becomes a goal, using only what the "
    "shot looked like: where it was taken from, how, the assist, and where defenders and the goalkeeper stood. "
    "It was trained on 38,318 shots from the 2015/16 season across 5 leagues and checked on 311 matches it "
    "never saw."
)

st.subheader("Within 1% of StatsBomb's own model on held-out matches")
st.markdown((FIGURES_DIR / "model_results.md").read_text())
st.caption("Lower log loss and Brier are better; higher ROC AUC is better. CV = 5-fold cross-validation "
           "grouped by match. Splits are always by match, so no match is in both training and test data.")

left, right = st.columns(2)
with left:
    st.subheader("Calibrated: 0.2 xG goes in about 20% of the time")
    st.image(str(FIGURES_DIR / "calibration.png"), width="stretch")
with right:
    st.subheader("Distance, angle and defenders in the way matter most")
    st.image(str(FIGURES_DIR / "shap_summary.png"), width="stretch")

st.subheader("Why the model rated three Leicester shots the way it did")
cols = st.columns(3)
for col, name in zip(cols, ["shap_vardy_goal.png", "shap_mahrez_goal.png", "shap_leicester_miss.png"]):
    col.image(str(FIGURES_DIR / name), width="stretch")

st.subheader("Features")
st.markdown("""
| Group | Features |
|---|---|
| Geometry | Distance to goal, angle between the posts |
| Shot context | Body part, technique, free kick vs open play, play pattern, first time, under pressure, open goal |
| Assist | Through ball, cutback, cross, corner, other pass, or none |
| Freeze frame | Defenders between the ball and goal, nearest defender, keeper distance from line and center |
""")
st.write("Not used as features: anything recorded after the shot (end location, outcome) and StatsBomb's own "
         "xG, which is only the benchmark. Game state and formation are used for analysis, not prediction.")
credit()
