"""Smoke tests: every app page runs without errors on the real processed data."""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
APP = str(ROOT / "app" / "streamlit_app.py")


@pytest.mark.parametrize("page", ["views/story.py", "views/team_explorer.py", "views/players.py",
                                  "views/champions.py", "views/formations.py", "views/model.py"])
def test_page_runs(page):
    at = AppTest.from_file(APP, default_timeout=120)
    at.run()
    at.switch_page(page).run()
    assert not at.exception, at.exception


def test_team_explorer_every_team():
    at = AppTest.from_file(APP, default_timeout=120)
    at.run()
    at.switch_page("views/team_explorer.py").run()
    for league in at.selectbox[0].options:
        at.selectbox[0].select(league).run()
        for team in at.selectbox[1].options:
            for side in ["For", "Against"]:
                at.selectbox[1].select(team)
                at.radio[0].set_value(side).run()
                assert not at.exception, f"{league} / {team} / {side}: {at.exception}"


def test_formations_every_filter():
    at = AppTest.from_file(APP, default_timeout=120)
    at.run()
    at.switch_page("views/formations.py").run()
    for league in at.selectbox[0].options:
        at.selectbox[0].select(league).run()
        for state in at.selectbox[1].options:
            at.selectbox[1].select(state).run()
            assert not at.exception, f"{league} / {state}: {at.exception}"


def test_players_including_small_samples():
    at = AppTest.from_file(APP, default_timeout=120)
    at.run()
    at.switch_page("views/players.py").run()
    assert at.subheader[0].value.startswith("Jamie Vardy scored 19")
    for label in at.selectbox[0].options[:25]:
        at.selectbox[0].select(label).run()
        assert not at.exception, f"{label}: {at.exception}"
    at.checkbox[0].check().run()
    small = at.selectbox[0].options[-1]
    at.selectbox[0].select(small).run()
    assert not at.exception, at.exception
    assert any("Small sample" in w.value for w in at.warning)


def test_app_requirements_pinned_like_the_project():
    # The deployed app installs app/requirements.txt; its pins must match the tested environment.
    def pins(path):
        lines = (ROOT / path).read_text().splitlines()
        return {line.strip() for line in lines if line.strip() and not line.startswith(("#", "-r"))}
    assert pins("app/requirements.txt") <= pins("requirements.txt")
