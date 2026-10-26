"""Leicester xG app. Run with: streamlit run app/streamlit_app.py"""

from pathlib import Path

import streamlit as st

st.set_page_config(page_title="Leicester xG", page_icon="⚽", layout="wide")

pages = [
    st.Page("views/story.py", title="The Leicester story", default=True),
    st.Page("views/team_explorer.py", title="Team explorer"),
    st.Page("views/players.py", title="Player lookup"),
    st.Page("views/champions.py", title="Champions"),
    st.Page("views/formations.py", title="Formations"),
    st.Page("views/model.py", title="About the model"),
]
page = st.navigation(pages, position="sidebar")

LOGO = Path(__file__).resolve().parents[1] / "figures" / "statsbomb_logo.png"

with st.sidebar:
    st.caption("An expected goals model on the 2015/16 season. "
               "Data provided by [StatsBomb](https://github.com/statsbomb/open-data).")
    if LOGO.exists():    # StatsBomb's terms ask for their logo; add the file from their media pack
        st.image(str(LOGO), width=140)

page.run()
