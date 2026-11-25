"""Building panels: alignment to the factors, the drop policy, and its record.

Everything here runs against fixture zips written into a temporary directory
with the same filenames the data library uses, so the whole build path is
exercised -- zip reading, section selection, sentinel handling, factor
alignment and the drop log -- without a network call.
"""

from __future__ import annotations

import numpy as np
import pytest

from riskcheck import config
from riskcheck.build import build_panel, load_factors
from riskcheck.fetch import SOURCES_BY_KEY

from .conftest import write_zip

#: Long enough that the drop policy can distinguish "a series with a chronic
#: gap" from "a single missing day". The tolerance is 1% of the span, so a
#: seven-day fixture would make one gap look like a 14%-missing series and
#: both would be dropped the same way.
N_FIXTURE_DAYS = 150
_BUSINESS_DAYS = np.busday_offset(
    np.datetime64("1969-07-01"), np.arange(N_FIXTURE_DAYS), roll="forward"
)
DATES = [str(d).replace("-", "") for d in _BUSINESS_DAYS]

#: Index of the single day on which Gamma is missing.
GAMMA_GAP = 4


def _assets_csv(rows: list[str], names: str = "Alpha,Beta,Gamma") -> str:
    body = "\n".join(rows)
    return (
        "Toy industry file.\n"
        "\n"
        "Missing data are indicated by -99.99 or -999.\n"
        "\n"
        "\n"
        "  Average Value Weighted Returns -- Daily\n"
        f",{names}\n"
        f"{body}\n"
        "\n"
        "  Average Equal Weighted Returns -- Daily\n"
        f",{names}\n"
        + "\n".join(f"{d},   9.99,   9.99,   9.99" for d in DATES)
        + "\n\nCopyright 2026 Eugene F. Fama and Kenneth R. French\n"
    )


def _ff5_csv(dates: list[str]) -> str:
    rows = "\n".join(
        f"{d},    0.50,    0.10,   -0.20,    0.05,    0.15,    0.02" for d in dates
    )
    return (
        "Toy factor file.\n"
        "\n"
        ",Mkt-RF,SMB,HML,RMW,CMA,RF\n"
        f"{rows}\n"
        "\nCopyright 2026 Eugene F. Fama and Kenneth R. French\n"
    )


def _mom_csv(dates: list[str]) -> str:
    rows = "\n".join(f"{d},   0.25" for d in dates)
    return (
        "Toy momentum file.\n"
        "\n"
        "Missing data are indicated by -99.99 or -999.\n"
        "\n"
        ",Mom\n"
        f"{rows}\n"
        "\nCopyright 2026 Eugene F. Fama and Kenneth R. French\n"
    )


@pytest.fixture
def raw_dir(tmp_path):
    """A fixture data/raw with one clean series, one sparse, one single gap."""
    rows = []
    for i, date in enumerate(DATES):
        alpha = "   0.56" if i % 2 else "  -0.07"
        # Beta is missing for the first 60 days: a chronic gap.
        beta = " -99.99" if i < 60 else "   0.31"
        # Gamma is missing on exactly one day, using the other sentinel.
        gamma = "   -999" if i == GAMMA_GAP else "   1.00"
        rows.append(f"{date},{alpha},{beta},{gamma}")
    directory = tmp_path / "raw"
    write_zip(
        directory / SOURCES_BY_KEY["industry49"].zip_name,
        SOURCES_BY_KEY["industry49"].csv_name,
        _assets_csv(rows),
    )
    write_zip(
        directory / SOURCES_BY_KEY["ff5"].zip_name,
        SOURCES_BY_KEY["ff5"].csv_name,
        _ff5_csv(DATES),
    )
    write_zip(
        directory / SOURCES_BY_KEY["momentum"].zip_name,
        SOURCES_BY_KEY["momentum"].csv_name,
        _mom_csv(DATES),
    )
    return directory


def test_factors_are_assembled_in_the_configured_order(raw_dir):
    dates, factors, rf = load_factors(raw_dir)
    assert factors.shape == (len(DATES), len(config.FACTOR_NAMES))
    assert dates[0] == np.datetime64("1969-07-01")
    # Mkt-RF 0.50% and Mom 0.25%, both in decimal.
    assert factors[0, 0] == pytest.approx(0.005)
    assert factors[0, -1] == pytest.approx(0.0025)
    assert rf[0] == pytest.approx(0.0002)


def test_sparse_series_is_dropped_and_recorded(raw_dir):
    panel = build_panel("industry49", raw_dir)
    assert "Beta" not in panel.names
    dropped = {d["name"]: d for d in panel.drops.dropped_series}
    assert "Beta" in dropped
    assert dropped["Beta"]["n_missing_days"] == 60
    assert dropped["Beta"]["missing_fraction"] == pytest.approx(60 / N_FIXTURE_DAYS)
    assert "above the" in dropped["Beta"]["reason"]


def test_remaining_single_gap_drops_the_date_not_the_series(raw_dir):
    """Gamma is missing one day; the day goes, the series stays."""
    panel = build_panel("industry49", raw_dir)
    assert "Gamma" in panel.names
    gap_date = _BUSINESS_DAYS[GAMMA_GAP]
    assert panel.drops.n_dropped_dates == 1
    assert panel.drops.dropped_dates == [str(gap_date)]
    assert gap_date not in panel.dates


def test_built_panel_has_no_missing_values(raw_dir):
    panel = build_panel("industry49", raw_dir)
    assert not np.isnan(panel.excess_returns).any()
    assert not np.isnan(panel.factors).any()
    # One date dropped for Gamma's gap, one series (Beta) dropped entirely.
    assert panel.excess_returns.shape == (N_FIXTURE_DAYS - 1, 2)


def test_excess_returns_subtract_the_daily_risk_free_rate(raw_dir):
    panel = build_panel("industry49", raw_dir)
    np.testing.assert_allclose(
        panel.excess_returns, panel.total_returns - panel.rf[:, None], rtol=1e-12
    )
    # Alpha's first return is -0.07% less the 0.02% risk-free rate.
    assert panel.excess_returns[0, 0] == pytest.approx(-0.0007 - 0.0002)


def test_dates_outside_factor_coverage_are_dropped_and_counted(tmp_path, raw_dir):
    """Asset dates with no matching factor row cannot be used by the factor model."""
    # Rewrite the factor files without the last two dates.
    write_zip(
        raw_dir / SOURCES_BY_KEY["ff5"].zip_name,
        SOURCES_BY_KEY["ff5"].csv_name,
        _ff5_csv(DATES[:-2]),
    )
    write_zip(
        raw_dir / SOURCES_BY_KEY["momentum"].zip_name,
        SOURCES_BY_KEY["momentum"].csv_name,
        _mom_csv(DATES[:-2]),
    )
    panel = build_panel("industry49", raw_dir)
    assert panel.drops.dates_outside_factor_coverage == 2
    assert panel.dates[-1] == _BUSINESS_DAYS[-3]


def test_panel_round_trips_through_disk(raw_dir, tmp_path):
    from riskcheck.build import BuiltPanel

    panel = build_panel("industry49", raw_dir)
    path = tmp_path / "panel.npz"
    panel.save(path)
    reloaded = BuiltPanel.load(path)

    np.testing.assert_array_equal(reloaded.excess_returns, panel.excess_returns)
    np.testing.assert_array_equal(reloaded.dates, panel.dates)
    assert reloaded.names == panel.names
    assert reloaded.factor_names == panel.factor_names
    assert reloaded.drops.dropped_series == panel.drops.dropped_series
    assert reloaded.label == panel.label


def test_start_date_is_respected(raw_dir):
    start = _BUSINESS_DAYS[100]
    panel = build_panel("industry49", raw_dir, start=start)
    assert panel.dates[0] >= start
    assert panel.drops.n_dates_in_span == N_FIXTURE_DAYS - 100
