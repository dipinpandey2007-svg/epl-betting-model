import pandas as pd
import pytest

from eplmodel.analysis.promoted_teams import (
    LONG_ABSENCE,
    RECENT_YOYO,
    TEAM_HISTORY_FILE,
    categorize_team,
    load_team_history,
)


def test_team_history_table():
    history = load_team_history()
    assert history["Norwich"] == 1
    assert history["Watford"] == 8  # corrected from 15 in the exploratory script
    assert history["Brighton"] is None


def test_every_team_history_row_is_sourced():
    table = pd.read_csv(TEAM_HISTORY_FILE, comment="#", dtype=str)
    for col in ("source_pl_spells", "source_return_season"):
        assert table[col].str.startswith("https://").all(), col
    assert table["accessed"].str.fullmatch(r"\d{4}-\d{2}-\d{2}").all()
    assert table["team"].is_unique


def test_categorisation_threshold():
    history = {"Yoyo": 2, "Long": 3, "New": None}
    assert categorize_team("Yoyo", history, yoyo_threshold=2) == RECENT_YOYO
    assert categorize_team("Long", history, yoyo_threshold=2) == LONG_ABSENCE
    assert categorize_team("New", history, yoyo_threshold=2) == LONG_ABSENCE


def test_missing_team_is_an_error_not_a_silent_default():
    with pytest.raises(KeyError):
        categorize_team("Unknown FC", {"Norwich": 1})
