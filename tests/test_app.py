"""Smoke tests: every app page runs without errors on the real processed data."""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py")


@pytest.mark.parametrize("page", ["pages/story.py", "pages/team_explorer.py", "pages/model.py"])
def test_page_runs(page):
    at = AppTest.from_file(APP, default_timeout=120)
    at.run()
    at.switch_page(page).run()
    assert not at.exception, at.exception


def test_team_explorer_every_team():
    at = AppTest.from_file(APP, default_timeout=120)
    at.run()
    at.switch_page("pages/team_explorer.py").run()
    for league in at.selectbox[0].options:
        at.selectbox[0].select(league).run()
        for team in at.selectbox[1].options:
            for side in ["For", "Against"]:
                at.selectbox[1].select(team)
                at.radio[0].set_value(side).run()
                assert not at.exception, f"{league} / {team} / {side}: {at.exception}"
