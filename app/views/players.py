import altair as alt
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from mplsoccer import VerticalPitch

from common import (GRAY, INK, LEICESTER, LEICESTER_BLUE, LIGHT_GRAY, credit, get_data, get_players,
                    get_simulation, p_at_least, simulation_chart)

MIN_SHOTS = 30

shots, _ = get_data()
players = get_players()

st.title("Player lookup")
show_small = st.checkbox(f"Include players with fewer than {MIN_SHOTS} shots", value=False)
options = players if show_small else players[players.shots >= MIN_SHOTS]
labels = options.label.tolist()
default = next((i for i, label in enumerate(labels) if label.startswith("Jamie Vardy")), 0)
choice = st.selectbox(f"Player ({len(options):,} players, sorted by xG)", labels, index=default)
player = options[options.label == choice].iloc[0]
player_id = options.index[options.label == choice][0]

s = shots[(shots.player_id == player_id) & ~shots.is_penalty]
goals, xg = int(s.is_goal.sum()), s.xg.sum()
sims = get_simulation(tuple(s.xg))
low, high = int(sims.quantile(0.05)), int(sims.quantile(0.95))
p = p_at_least(goals, sims.values)
competition = s.competition.iloc[0]
is_leicester = LEICESTER in player.teams
mark = LEICESTER_BLUE if is_leicester else INK

if player.shots < MIN_SHOTS:
    st.warning(f"Small sample: {player.shots} shots. Goals minus xG this small is mostly noise.")

verdict = ("well above" if p < 0.05 else "well below" if p > 0.95 else "within")
st.subheader(f"{player.player} scored {goals} non-penalty goals from {xg:.1f} xG, {verdict} the range "
             f"an average finisher would hit ({low} to {high})")

m1, m2, m3, m4 = st.columns(4)
m1.metric("Shots", f"{len(s)}", player.teams, delta_color="off", delta_arrow="off", border=True)
m2.metric("Goals minus xG", f"{goals - xg:+.1f}", f"{goals} goals from {xg:.1f} xG", delta_color="off",
          delta_arrow="off", border=True)
m3.metric("xG per shot", f"{xg / len(s):.3f}", border=True)
m4.metric("Chance at average finishing", f"{p:.1%}", f"of scoring {goals}+", delta_color="off",
          delta_arrow="off", border=True)

left, right = st.columns([1.1, 1])
with left:
    pitch = VerticalPitch(half=True, pitch_type="statsbomb", line_color="#c8c8c8", pad_bottom=-8)
    fig, ax = pitch.draw(figsize=(7, 6))
    misses, scored = s[s.is_goal == 0], s[s.is_goal == 1]
    pitch.scatter(misses.x, misses.y, s=misses.xg * 600 + 15, ax=ax, facecolors="none", edgecolors=GRAY,
                  linewidth=1, label="Miss")
    pitch.scatter(scored.x, scored.y, s=scored.xg * 600 + 15, ax=ax, color=mark, edgecolors="white",
                  linewidth=1, label="Goal")
    ax.legend(loc="lower left", frameon=False, fontsize=9)
    ax.set_title(f"{player.player}: {len(s)} non-penalty shots. Dot size = xG", fontsize=11, color=INK)
    st.pyplot(fig, width="stretch")
    plt.close(fig)

with right:
    st.markdown(f"**10,000 simulated seasons** from the same shots: {goals} actual vs {xg:.1f} xG (p = {p:.3f})")
    st.altair_chart(simulation_chart(sims, goals), width="stretch")

    # Conversion by body part against the league average for the same body part.
    league = shots[(shots.competition == competition) & ~shots.is_penalty]
    rows = []
    for part, g in s.groupby("body_part"):
        lg = league[league.body_part == part]
        rows += [{"body_part": f"{part} ({len(g)})", "who": player.player, "conversion": g.is_goal.mean()},
                 {"body_part": f"{part} ({len(g)})", "who": "League average", "conversion": lg.is_goal.mean()}]
    conv = pd.DataFrame(rows)
    bars = alt.Chart(conv).mark_bar(cornerRadiusEnd=3).encode(
        y=alt.Y("body_part:N", title=None, axis=alt.Axis(labelOverlap=False)),
        x=alt.X("conversion:Q", title="Shots converted", axis=alt.Axis(format="%")),
        yOffset=alt.YOffset("who:N", sort=[player.player, "League average"]),
        color=alt.Color("who:N", scale=alt.Scale(domain=[player.player, "League average"], range=[mark, LIGHT_GRAY]),
                        legend=alt.Legend(title=None, orient="bottom")),
        tooltip=[alt.Tooltip("body_part:N", title="Body part (shots)"), alt.Tooltip("who:N", title=""),
                 alt.Tooltip("conversion:Q", title="Converted", format=".1%")],
    )
    st.altair_chart(bars.properties(title="Conversion by body part vs league average",
                                    height=alt.Step(16, **{"for": "offset"})), width="stretch")

st.caption(f"Non-penalty shots. Players with fewer than {MIN_SHOTS} shots are hidden unless you include them. "
           "League average uses the player's first league in the data.")
credit()
