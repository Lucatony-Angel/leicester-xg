import altair as alt
import streamlit as st

from common import LEICESTER, credit, get_champions, highlight, ordinal, short_name

summary = get_champions()
lei = summary.loc[LEICESTER]
others = summary.drop(LEICESTER)

st.title("Leicester vs the other champions")
st.write(f"Barcelona, Juventus and PSG out-shot their leagues by {others.shots_vs_league.min():.0%} to "
         f"{others.shots_vs_league.max():.0%}, beat their xG by {others.finishing.min():.0f} to "
         f"{others.finishing.max():.0f} goals, and topped their leagues on expected points. "
         f"Leicester took {lei.shots_vs_league:+.0%} shots, finished at an average rate "
         f"({lei.finishing:+.1f}) and ranked {ordinal(lei.xpts_rank)} on expected points.")

data = summary.reset_index().assign(name=lambda d: d.team.map(short_name))
order = data.name.tolist()
PANELS = [
    ("shots_vs_league", "Shots per match vs league average", "%"),
    ("quality_vs_league", "xG per shot vs league average", "%"),
    ("finishing", "Goals minus xG", "goals"),
    ("shots_against_vs_league", "Shots conceded vs league average", "%"),
    ("quality_against_vs_league", "xG per shot conceded vs league average", "%"),
    ("goals_prevented", "xG against minus goals conceded", "goals"),
]
for row in [PANELS[:3], PANELS[3:]]:
    cols = st.columns(3)
    for col, (field, title, kind) in zip(cols, row):
        fmt = "+.0%" if kind == "%" else "+.1f"
        chart = alt.Chart(data).mark_bar(cornerRadiusEnd=3).encode(
            y=alt.Y("name:N", sort=order, title=None),
            x=alt.X(f"{field}:Q", title=None, axis=alt.Axis(format="%" if kind == "%" else ".0f")),
            color=highlight("team", LEICESTER),
            tooltip=[alt.Tooltip("team:N", title="Team"), alt.Tooltip(f"{field}:Q", title=title, format=fmt)],
        )
        col.altair_chart(chart.properties(title=title, height=180), width="stretch")

st.subheader("Full comparison")
table = summary[["league", "matches", "shots_per_match", "xg_per_shot", "goals", "xg", "goals_against", "xga",
                 "points", "xpts", "xpts_rank", "counter_xg_ratio"]].rename(columns={
    "league": "League", "matches": "Matches", "shots_per_match": "Shots/match", "xg_per_shot": "xG/shot",
    "goals": "Goals", "xg": "xG", "goals_against": "Conceded", "xga": "xG against", "points": "Points",
    "xpts": "Expected points", "xpts_rank": "Rank on xPts", "counter_xg_ratio": "Counter xG vs league"})
st.dataframe(table.style.format({"Shots/match": "{:.1f}", "xG/shot": "{:.3f}", "Goals": "{:.0f}", "xG": "{:.1f}",
                                 "Conceded": "{:.0f}", "xG against": "{:.1f}", "Points": "{:.0f}",
                                 "Expected points": "{:.1f}", "Counter xG vs league": "{:.1f}x"}),
             width="stretch")
st.caption("Non-penalty shots. Each team is compared with the average team in its own league, because leagues "
           "differ in scoring. Bayern isn't shown: the free data covers only Leverkusen's Bundesliga matches. "
           "PSG: 37 of 38 matches.")
credit()
