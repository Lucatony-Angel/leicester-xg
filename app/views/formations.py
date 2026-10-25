import altair as alt
import pandas as pd
import streamlit as st

from common import (GAME_STATES, GRAY, INK, LEAGUES, LEICESTER, LEICESTER_BLUE, LIGHT_GRAY, credit,
                    get_formation_profile, get_team_formation_profiles)

ADJUSTED = "All scorelines (adjusted)"
STATE_LABELS = {"-2 or worse": "Losing by 2+", "-1": "Losing by 1", "0": "Level", "+1": "Winning by 1",
                "+2 or better": "Winning by 2+"}

st.title("Formations")
st.write("Shots and xG per 90 minutes actually played in each formation, tracked through Tactical Shifts. "
         "Formation choice is tied to team quality, so these are patterns, not causes.")

c1, c2, c3 = st.columns(3)
league_label = c1.selectbox("League", ["All 5 leagues"] + [k for k in LEAGUES if "Bundesliga" not in k])
competition = LEAGUES.get(league_label)
state_label = c2.selectbox("Game state", [ADJUSTED] + [STATE_LABELS[s] for s in GAME_STATES])
game_state = next((s for s, label in STATE_LABELS.items() if label == state_label), None)

min_minutes = 5000 if competition is None else 1500
if game_state:
    min_minutes //= 3
profile = get_formation_profile(competition, game_state)
profile = profile[profile.minutes >= min_minutes]
if profile.empty:
    st.info("Not enough minutes in any formation for this filter.")
    st.stop()
formation = c3.selectbox("Highlight formation", profile.index.tolist(),
                         index=profile.index.tolist().index("4-4-2") if "4-4-2" in profile.index else 0)

note = ("weighted to the same mix of scorelines for every formation" if game_state is None
        else f"only minutes spent {state_label.lower()}")
row = profile.loc[formation]
best = profile.xg_per90.idxmax()
st.subheader(f"{formation}: {row.xg_per90:.2f} xG created and {row.xga_per90:.2f} conceded per 90 "
             f"({best} creates the most: {profile.loc[best, 'xg_per90']:.2f})")

data = profile.reset_index(names="formation")
order = data.sort_values("xg_per90", ascending=False).formation.tolist()
color = alt.condition(alt.datum.formation == formation, alt.value(INK), alt.value(LIGHT_GRAY))
cols = st.columns(4)
for col, (field, title, fmt) in zip(cols, [("xg_per90", "xG created per 90", ".2f"),
                                           ("xga_per90", "xG conceded per 90", ".2f"),
                                           ("shots_per90", "Shots per 90", ".1f"),
                                           ("xg_per_shot", "xG per shot", ".3f")]):
    chart = alt.Chart(data).mark_bar(cornerRadiusEnd=3).encode(
        y=alt.Y("formation:N", sort=order, title=None),
        x=alt.X(f"{field}:Q", title=None),
        color=color,
        tooltip=[alt.Tooltip("formation:N", title="Formation"), alt.Tooltip(f"{field}:Q", title=title, format=fmt),
                 alt.Tooltip("minutes:Q", title="Minutes", format=",.0f")],
    )
    col.altair_chart(chart.properties(title=title, height=260), width="stretch")
st.caption(f"Rates are {note}. Formations with at least {min_minutes:,} minutes. Non-penalty shots.")

st.subheader(f"Where {formation}'s xG comes from, vs the average of all formations")
mix_cols = {"counter_share": "Counterattacks", "set_piece_share": "Possessions from set pieces",
            "cross_share": "Crosses", "through_ball_share": "Through balls"}
weights = profile.minutes / profile.minutes.sum()
mix = pd.DataFrame({
    "category": list(mix_cols.values()) * 2,
    "who": [formation] * 4 + ["All formations"] * 4,
    "share": [row[c] for c in mix_cols] + [(profile[c] * weights).sum() for c in mix_cols],
})
bars = alt.Chart(mix).mark_bar(cornerRadiusEnd=3).encode(
    y=alt.Y("category:N", sort=list(mix_cols.values()), title=None, axis=alt.Axis(labelOverlap=False)),
    x=alt.X("share:Q", title="Share of xG", axis=alt.Axis(format="%")),
    yOffset=alt.YOffset("who:N", sort=[formation, "All formations"]),
    color=alt.Color("who:N", scale=alt.Scale(domain=[formation, "All formations"], range=[INK, LIGHT_GRAY]),
                    legend=alt.Legend(title=None, orient="bottom")),
    tooltip=[alt.Tooltip("category:N", title=""), alt.Tooltip("who:N", title=""),
             alt.Tooltip("share:Q", title="Share of xG", format=".1%")],
)
st.altair_chart(bars.properties(height=240), width="stretch")
st.caption("Set-piece possessions include everything after a corner, free kick or throw-in until the other team "
           "wins the ball, as StatsBomb tags them.")

if game_state is None:
    teams = get_team_formation_profiles(formation).reset_index()
    if competition:
        teams = teams[teams.competition == competition]
    if not teams.empty:
        lei_note = " Leicester in blue." if LEICESTER in teams.team.values else ""
        st.subheader(f"Teams that played {formation} for 900+ minutes")
        dots = alt.Chart(teams).mark_circle(opacity=1, stroke="white", strokeWidth=1).encode(
            x=alt.X("shots_per90:Q", title=f"Shots per 90 in {formation}", scale=alt.Scale(zero=False)),
            y=alt.Y("xg_per_shot:Q", title=f"xG per shot in {formation}", scale=alt.Scale(zero=False)),
            size=alt.Size("minutes:Q", legend=None, scale=alt.Scale(range=[40, 300])),
            color=alt.condition(alt.datum.team == LEICESTER, alt.value(LEICESTER_BLUE), alt.value(GRAY)),
            tooltip=[alt.Tooltip("team:N", title="Team"), alt.Tooltip("minutes:Q", title="Minutes", format=",.0f"),
                     alt.Tooltip("xg_per90:Q", title="xG per 90", format=".2f"),
                     alt.Tooltip("xga_per90:Q", title="xG conceded per 90", format=".2f"),
                     alt.Tooltip("counter_share:Q", title="Counter share", format=".1%")],
        )
        st.altair_chart(dots.properties(height=380), width="stretch")
        st.caption(f"Game-state adjusted. Dot size = minutes in {formation}. Hover for details.{lei_note}")

credit()
