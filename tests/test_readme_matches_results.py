"""Every number in the README must come from a run saved under results/.

This is the project's central claim about its own honesty, so it is a test
rather than a promise. The README's tables are parsed out of the Markdown and
checked cell by cell against ``results/metrics.csv``, ``results/stress.csv``
and ``results/log_score_paired.csv``.

If the evaluation is re-run and the numbers move, this fails until the README
is updated -- which is the intended behaviour.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "README.md"
RESULTS = ROOT / "results"

#: README display names in the order the tables use them, mapped to the
#: `estimator` labels the runner writes.
ESTIMATORS = [
    "Sample",
    "EWMA",
    "Ledoit-Wolf",
    "PCA factor",
    "FF5+Mom factor",
    "EWMA + VRA",
]

pytestmark = pytest.mark.skipif(
    not (RESULTS / "metrics.csv").exists(),
    reason="results/ not present; run `riskcheck evaluate` first",
)


def _tables(text: str) -> list[tuple[list[str], list[list[str]]]]:
    """Pull every Markdown table out of the document."""
    out: list[tuple[list[str], list[list[str]]]] = []
    current: list[list[str]] = []
    for line in text.splitlines():
        if line.startswith("|"):
            current.append([cell.strip() for cell in line.strip("|").split("|")])
        else:
            if len(current) > 2:
                out.append((current[0], current[2:]))
            current = []
    if len(current) > 2:
        out.append((current[0], current[2:]))
    return out


def _number(cell: str) -> float:
    """Read a README cell, tolerating bold, en-dash minus and percent signs."""
    cleaned = cell.replace("**", "").replace("−", "-").replace("%", "").strip()
    return float(cleaned)


@pytest.fixture(scope="module")
def readme_tables():
    return _tables(README.read_text())


@pytest.fixture(scope="module")
def metrics():
    return pd.read_csv(RESULTS / "metrics.csv")


@pytest.fixture(scope="module")
def stress():
    return pd.read_csv(RESULTS / "stress.csv")


@pytest.fixture(scope="module")
def paired():
    return pd.read_csv(RESULTS / "log_score_paired.csv")


def _find(tables, first_header: str, n_columns: int):
    for headers, rows in tables:
        if headers[0] == first_header and len(headers) == n_columns:
            return headers, rows
    raise AssertionError(f"no README table starting {first_header!r} with {n_columns} columns")


def test_headline_table_matches_metrics(readme_tables, metrics):
    _, rows = _find(readme_tables, "Estimator", 7)
    frame = metrics[(metrics.asset_set == "industry49") & (metrics.horizon == "1w")]
    frame = frame.set_index("estimator")
    assert len(rows) == len(ESTIMATORS)

    for row, name in zip(rows, ESTIMATORS, strict=True):
        actual = frame.loc[name]
        _, long_only, long_short, min_var, in_band, mv_vol, score = row
        assert _number(long_only) == pytest.approx(actual.long_only_bias_mean, abs=5e-4), name
        assert _number(long_short) == pytest.approx(actual.long_short_bias_mean, abs=5e-4), name
        assert _number(min_var) == pytest.approx(actual.mv_bias_mean, abs=5e-4), name
        assert _number(in_band) == pytest.approx(actual.mv_share_in_band * 100, abs=0.51), name
        assert _number(mv_vol) == pytest.approx(
            actual.mv_realized_vol_annual * 100, abs=5e-3
        ), name
        assert _number(score) == pytest.approx(actual.log_score, abs=5e-4), name


def test_minimum_variance_breakdown_matches_metrics(readme_tables, metrics):
    _, rows = _find(readme_tables, "Estimator", 5)
    cells = [
        ("industry49", "1w"),
        ("industry49", "1m"),
        ("size_bm_25", "1w"),
        ("size_bm_25", "1m"),
    ]
    for row, name in zip(rows, ESTIMATORS, strict=True):
        for value, (asset_set, horizon) in zip(row[1:], cells, strict=True):
            actual = metrics[
                (metrics.asset_set == asset_set)
                & (metrics.horizon == horizon)
                & (metrics.estimator == name)
            ].iloc[0]
            assert _number(value) == pytest.approx(actual.mv_bias_mean, abs=5e-4), (
                name,
                asset_set,
                horizon,
            )


def _stress_table(readme_tables):
    """Two five-column tables start with "Estimator"; pick the stress one."""
    for headers, rows in readme_tables:
        if headers[0] == "Estimator" and "GFC" in headers[1]:
            return headers, rows
    raise AssertionError("no stress table in the README")


def test_stress_table_matches_stress_results(readme_tables, stress):
    _, rows = _stress_table(readme_tables)

    windows = [
        "GFC 2008-2009",
        "COVID crash (Mar 2020)",
        "COVID crash (wide)",
        "2022 drawdown",
    ]
    frame = stress[(stress.asset_set == "industry49") & (stress.horizon == "1d")]
    for row, name in zip(rows, ESTIMATORS, strict=True):
        for value, window in zip(row[1:], windows, strict=True):
            actual = frame[(frame.estimator == name) & (frame.window == window)].iloc[0]
            assert _number(value) == pytest.approx(actual.mv_bias, abs=5e-4), (name, window)


def test_stress_observation_counts_in_headers_are_right(readme_tables, stress):
    headers, _ = _stress_table(readme_tables)
    frame = stress[(stress.asset_set == "industry49") & (stress.horizon == "1d")]
    counts = {
        "GFC": "GFC 2008-2009",
        "March 2020": "COVID crash (Mar 2020)",
        "Feb–Apr 2020": "COVID crash (wide)",
        "2022": "2022 drawdown",
    }
    for header in headers[1:]:
        claimed = int(header.split("n=")[1].rstrip(")"))
        window = next(v for k, v in counts.items() if header.startswith(k))
        assert claimed == int(frame[frame.window == window].iloc[0].n_obs), header


def test_paired_log_score_table_matches_results(readme_tables, paired):
    _, rows = _find(readme_tables, "Estimator", 6)
    frame = paired[(paired.asset_set == "industry49") & (paired.horizon == "1w")]
    frame = frame.set_index("estimator")
    # This table is ordered by score, not by family.
    order = ["Ledoit-Wolf", "EWMA + VRA", "EWMA", "FF5+Mom factor", "Sample", "PCA factor"]
    for row, name in zip(rows, order, strict=True):
        actual = frame.loc[name]
        assert _number(row[1]) == pytest.approx(actual.log_score, abs=5e-4), name
        if row[3] != "—":
            assert _number(row[3]) == pytest.approx(actual.difference, abs=5e-5), name
        verdict = row[5].replace("**", "").strip()
        if verdict in ("yes", "no"):
            assert (verdict == "yes") == bool(actual.worse_than_baseline), name


def test_regime_adjustment_table_matches_metrics(readme_tables, metrics):
    rows = None
    for headers, candidate in readme_tables:
        if headers[0] == "" and headers[1].startswith("49 ind."):
            rows = candidate
            break
    assert rows is not None, "no regime-adjustment table in the README"

    cells = [
        ("industry49", "1d"),
        ("industry49", "1w"),
        ("industry49", "1m"),
        ("size_bm_25", "1d"),
        ("size_bm_25", "1w"),
        ("size_bm_25", "1m"),
    ]
    for row, name in zip(rows, ["EWMA", "EWMA + VRA"], strict=True):
        for value, (asset_set, horizon) in zip(row[1:], cells, strict=True):
            actual = metrics[
                (metrics.asset_set == asset_set)
                & (metrics.horizon == horizon)
                & (metrics.estimator == name)
            ].iloc[0]
            assert _number(value) == pytest.approx(actual.mv_bias_mean, abs=5e-4), (
                name,
                asset_set,
                horizon,
            )


# ---------------------------------------------------------------------------
# The README's qualitative claims, as assertions
# ---------------------------------------------------------------------------


def test_every_minimum_variance_cell_understates_risk(metrics):
    """README: 'Not one of the 24 cells is below 1, and not one is inside its band.'"""
    frame = metrics[metrics.horizon.isin(("1w", "1m"))]
    assert len(frame) == 24
    assert (frame.mv_bias_mean > 1.0).all()
    assert (frame.mv_bias_mean - 1.0 > frame.mv_bias_band).all()


def test_regime_adjustment_cannot_change_the_portfolio(metrics):
    """README: minimum-variance weights are scale-invariant, so realised vol is identical."""
    for asset_set in ("industry49", "size_bm_25"):
        for horizon in ("1d", "1w", "1m"):
            subset = metrics[
                (metrics.asset_set == asset_set) & (metrics.horizon == horizon)
            ].set_index("estimator")
            assert subset.loc["EWMA"].mv_realized_vol_annual == pytest.approx(
                subset.loc["EWMA + VRA"].mv_realized_vol_annual, abs=1e-12
            ), (asset_set, horizon)


def test_regime_adjustment_improves_minimum_variance_calibration_everywhere(metrics):
    """README: 'improves minimum-variance calibration in all six cells'."""
    for asset_set in ("industry49", "size_bm_25"):
        for horizon in ("1d", "1w", "1m"):
            subset = metrics[
                (metrics.asset_set == asset_set) & (metrics.horizon == horizon)
            ].set_index("estimator")
            assert subset.loc["EWMA + VRA"].mv_bias_mean < subset.loc["EWMA"].mv_bias_mean


def test_size_book_to_market_takes_more_leverage_than_industries(metrics):
    """README: 'mean gross leverage 5.21 against 3.26, averaged across the six estimators'."""
    frame = metrics[metrics.horizon == "1w"].groupby("asset_set").mv_gross_leverage.mean()
    assert frame["size_bm_25"] == pytest.approx(5.21, abs=0.005)
    assert frame["industry49"] == pytest.approx(3.26, abs=0.005)
    assert frame["size_bm_25"] > frame["industry49"]


def test_longer_horizon_is_worse_for_every_model(metrics):
    """README: 'Every model degrades from one week to one month.'"""
    for asset_set in ("industry49", "size_bm_25"):
        weekly = metrics[
            (metrics.asset_set == asset_set) & (metrics.horizon == "1w")
        ].set_index("estimator")
        monthly = metrics[
            (metrics.asset_set == asset_set) & (metrics.horizon == "1m")
        ].set_index("estimator")
        for name in ESTIMATORS:
            assert monthly.loc[name].mv_bias_mean > weekly.loc[name].mv_bias_mean, name


def test_factor_models_give_the_highest_realised_minimum_variance_volatility(metrics):
    """README: the factor models 'lose on both counts'."""
    frame = metrics[
        (metrics.asset_set == "industry49") & (metrics.horizon == "1w")
    ].set_index("estimator")
    worst = frame.mv_realized_vol_annual.nlargest(2).index.tolist()
    assert set(worst) == {"PCA factor", "FF5+Mom factor"}


def test_frozen_hyperparameters_in_the_readme_match_the_saved_choice():
    """The estimator table must name what `riskcheck validate` actually froze."""
    import json

    blob = json.loads((RESULTS / "hyperparameters.json").read_text())
    labels = {s["family"]: s["params"] for s in blob["final_specs"]}
    assert labels["sample"]["window"] == 756
    assert labels["ewma"]["halflife"] == 252.0
    assert labels["ledoit_wolf"]["window"] == 252
    assert labels["pca"] == {"window": 252, "n_factors": 5}
    assert labels["fundamental"] == {"window": 252, "factor_halflife": 63.0}
    assert labels["vra"]["halflife"] == 504.0
    assert blob["vra_base"]["family"] == "ewma"

    readme = README.read_text()
    for claim in (
        "3-year window",
        "half-life 252 days",
        "1-year window",
        "k = 5, 1-year window",
        "1-year betas, factor half-life 63 days",
        "on EWMA(252d), half-life 504 days",
    ):
        assert claim in readme, claim
