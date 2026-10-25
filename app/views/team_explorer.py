import altair as alt
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from mplsoccer import VerticalPitch

from common import (
    GRAY,
    INK,
    LEAGUES,
    LEICESTER,
    LEICESTER_BLUE,
    LIGHT_GRAY,
    credit,
    get_breakdown,
    get_data,
    ordinal,
    rank,
    share_vs_league,
)

shots, _ = get_data()

st.title("Team explorer")
c1, c2, c3 = st.columns([1, 1, 1])
league_label = c1.selectbox("League", list(LEAGUES))
competition = LEAGUES[league_label]
t = get_breakdown(competition)
teams = sorted(t.index)
team = c2.selectbox("Team", teams, index=teams.index(LEICESTER) if LEICESTER in teams else 0)
side = c3.radio("Shots", ["For", "Against"], horizontal=True)

if competition == "Germany - 1. Bundesliga":
    st.info("The free StatsBomb data only covers Bayer Leverkusen's Bundesliga matches. "
            "Other teams here have 1 or 2 matches.")

col = "team" if side == "For" else "opponent"
s = shots[(shots.competition == competition) & (shots[col] == team) & ~shots.is_penalty]
goals, xg = int(s.is_goal.sum()), s.xg.sum()
row = t.loc[team]
mark = LEICESTER_BLUE if team == LEICESTER else INK
n = len(t)

if side == "For":
    headline = (f"{team} scored {goals} non-penalty goals from {xg:.1f} xG "
                f"({goals - xg:+.1f}, {ordinal(rank(t, 'finishing', team))} of {n} for finishing)")
else:
    headline = (f"{team} conceded {goals} non-penalty goals from {xg:.1f} xG against "
                f"({xg - goals:+.1f} prevented, {ordinal(rank(t, 'goals_prevented', team))} of {n})")
st.subheader(headline)

m1, m2, m3, m4 = st.columns(4)
per_match = "shots_per_match" if side == "For" else "shots_against_per_match"
quality = "xg_per_shot" if side == "For" else "xga_per_shot"
m1.metric("Matches", f"{row.matches:.0f}", border=True)
m2.metric("Shots per match", f"{row[per_match]:.1f}",
          f"league avg {t[per_match].mean():.1f}", delta_color="off", delta_arrow="off", border=True)
m3.metric("xG per shot", f"{row[quality]:.3f}", f"league avg {t[quality].mean():.3f}", delta_color="off",
          delta_arrow="off", border=True)
m4.metric("Goals vs xG", f"{goals} vs {xg:.1f}", f"{goals - xg:+.1f}", border=True,
          delta_color="normal" if side == "For" else "inverse")

left, right = st.columns([1.1, 1])
with left:
    pitch = VerticalPitch(half=True, pitch_type="statsbomb", line_color="#c8c8c8", pad_bottom=-8)
    fig, ax = pitch.draw(figsize=(7, 6))
    misses, scored = s[s.is_goal == 0], s[s.is_goal == 1]
    pitch.scatter(misses.x, misses.y, s=misses.xg * 600 + 12, ax=ax, facecolors="none", edgecolors=GRAY,
                  linewidth=1, label="Miss")
    pitch.scatter(scored.x, scored.y, s=scored.xg * 600 + 12, ax=ax, color=mark, edgecolors="white",
                  linewidth=1, label="Goal")
    ax.legend(loc="lower left", frameon=False, fontsize=9)
    ax.set_title(f"{len(s)} shots {'by' if side == 'For' else 'against'} {team}. Dot size = xG",
                 fontsize=11, color=INK)
    st.pyplot(fig, width="stretch")
    plt.close(fig)

with right:
    # Cumulative goals vs xG over the season: shows when the gap opened up.
    by_match = (s.groupby("match_date").agg(goals=("is_goal", "sum"), xg=("xg", "sum")).sort_index().cumsum()
                .reset_index())
    by_match["match_date"] = pd.to_datetime(by_match.match_date)
    long = by_match.melt("match_date", var_name="series", value_name="total")
    long["series"] = long.series.map({"goals": "Goals", "xg": "xG"})
    line = alt.Chart(long).mark_line(strokeWidth=2.5).encode(
        x=alt.X("match_date:T", title=None),
        y=alt.Y("total:Q", title="Cumulative total"),
        color=alt.Color("series:N", scale=alt.Scale(domain=["Goals", "xG"], range=[mark, GRAY]),
                        legend=alt.Legend(title=None, orient="top-left")),
        strokeDash=alt.StrokeDash("series:N", scale=alt.Scale(domain=["Goals", "xG"], range=[[1, 0], [5, 3]]),
                                  legend=None),
        tooltip=[alt.Tooltip("match_date:T", title="Date"), alt.Tooltip("series:N", title=""),
                 alt.Tooltip("total:Q", title="Total", format=".1f")],
    )
    st.altair_chart(line.properties(title="Goals vs xG through the season", height=300), width="stretch")

    shares = share_vs_league(shots, "play_pattern", team, competition, side=col).reset_index(names="pattern")
    long = shares.melt("pattern", ["team_share", "rest_of_league_share"], var_name="who", value_name="share")
    long["who"] = long.who.map({"team_share": team, "rest_of_league_share": "Rest of league"})
    top = shares.sort_values("ratio", ascending=False).iloc[0]
    bars = alt.Chart(long).mark_bar(cornerRadiusEnd=3).encode(
        y=alt.Y("pattern:N", sort=shares.pattern.tolist(), title=None, axis=alt.Axis(labelOverlap=False)),
        x=alt.X("share:Q", title="Share of non-penalty xG", axis=alt.Axis(format="%")),
        yOffset=alt.YOffset("who:N", sort=[team, "Rest of league"]),
        color=alt.Color("who:N", scale=alt.Scale(domain=[team, "Rest of league"], range=[mark, LIGHT_GRAY]),
                        legend=alt.Legend(title=None, orient="bottom")),
        tooltip=[alt.Tooltip("pattern:N", title="Play pattern"), alt.Tooltip("who:N", title=""),
                 alt.Tooltip("share:Q", title="Share of xG", format=".1%")],
    )
    st.altair_chart(bars.properties(
        title=f"Biggest tilt: {top.pattern} ({top.ratio:.1f}x the league share)", height=380),
        width="stretch")

st.caption("Non-penalty shots. Penalties are analyzed separately.")
credit()
