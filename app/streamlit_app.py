"""Leicester xG app. Run with: streamlit run app/streamlit_app.py"""

import streamlit as st

st.set_page_config(page_title="Leicester xG", page_icon="⚽", layout="wide")

pages = [
    st.Page("pages/story.py", title="The Leicester story", default=True),
    st.Page("pages/team_explorer.py", title="Team explorer"),
    st.Page("pages/model.py", title="About the model"),
]
page = st.navigation(pages, position="sidebar")

with st.sidebar:
    st.caption("An expected goals model on the 2015/16 season. "
               "Data provided by [StatsBomb](https://github.com/statsbomb/open-data).")

page.run()
