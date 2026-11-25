"""Parsing the data library's quirks: stacked tables, sentinels, percent units."""

from __future__ import annotations

import numpy as np
import pytest

from riskcheck.frenchdata import load_panel, parse_sections, read_csv_text, section_to_panel

from .conftest import FIXTURE_CSV, write_zip


def test_sections_are_split_and_titled():
    sections = parse_sections(FIXTURE_CSV)
    assert len(sections) == 2
    assert "Value Weighted" in sections[0].title
    assert "Equal Weighted" in sections[1].title
    assert sections[0].columns == ("Alpha", "Beta", "Gamma")
    assert sections[0].dates.tolist() == [19690701, 19690702, 19690703, 19690707, 19690708]


def test_value_weighted_block_is_selected_by_title_not_position(french_fixture_zip):
    """The equal-weighted block must not be picked up by accident."""
    panel = load_panel(french_fixture_zip, title_contains="Average Value Weighted Returns")
    # The equal-weighted fixture block is all 9.99%; the value-weighted is not.
    assert not np.allclose(np.nan_to_num(panel.returns), 0.0999)

    equal = load_panel(french_fixture_zip, title_contains="Average Equal Weighted Returns")
    assert np.allclose(equal.returns, 0.0999)


def test_both_sentinels_become_missing():
    panel = section_to_panel(parse_sections(FIXTURE_CSV)[0])
    expected = np.zeros((5, 3), dtype=bool)
    expected[0, 2] = True  # -99.99
    expected[2, 0] = True  # -99.99
    expected[3, 2] = True  # -999
    np.testing.assert_array_equal(panel.missing, expected)
    assert np.isnan(panel.returns[panel.missing]).all()
    assert not np.isnan(panel.returns[~panel.missing]).any()


def test_percent_is_converted_to_decimal():
    panel = section_to_panel(parse_sections(FIXTURE_CSV)[0])
    # 0.56% -> 0.0056, and 3.57% -> 0.0357.
    assert panel.returns[0, 0] == pytest.approx(0.0056)
    assert panel.returns[3, 0] == pytest.approx(0.0357)
    # Nothing survives at percent scale.
    assert np.nanmax(np.abs(panel.returns)) < 0.1


def test_dates_are_parsed_from_yyyymmdd():
    panel = section_to_panel(parse_sections(FIXTURE_CSV)[0])
    assert panel.dates[0] == np.datetime64("1969-07-01")
    assert panel.dates[-1] == np.datetime64("1969-07-08")
    assert panel.dates.dtype == np.dtype("datetime64[D]")


def test_minus_one_hundred_percent_is_allowed():
    """A portfolio can lose all its value; that is data, not a sentinel."""
    text = FIXTURE_CSV.replace(
        "19690708,   0.10,   0.20,   0.30", "19690708, -100.00,   0.20,   0.30"
    )
    panel = section_to_panel(parse_sections(text)[0])
    assert panel.returns[4, 0] == pytest.approx(-1.0)
    assert not panel.missing[4, 0]


def test_unknown_sentinel_is_rejected_rather_than_silently_used():
    """A -9999 marker would otherwise become a -99x daily return."""
    text = FIXTURE_CSV.replace(
        "19690708,   0.10,   0.20,   0.30", "19690708, -9999.0,   0.20,   0.30"
    )
    with pytest.raises(ValueError, match="below -100"):
        section_to_panel(parse_sections(text)[0])


def test_reads_from_zip_and_plain_csv_alike(tmp_path):
    zip_path = write_zip(tmp_path / "a_CSV.zip", "a_Daily.csv", FIXTURE_CSV)
    csv_path = tmp_path / "a_Daily.csv"
    csv_path.write_text(FIXTURE_CSV)
    assert read_csv_text(zip_path) == read_csv_text(csv_path)


def test_missing_title_is_an_error_not_a_silent_fallback(french_fixture_zip):
    with pytest.raises(ValueError, match="no section titled"):
        load_panel(french_fixture_zip, title_contains="Average Cap Weighted Returns")
