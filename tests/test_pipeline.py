"""End-to-end: the runner, the saved artefacts, the report and the figures.

Runs on a synthetic panel small enough to be quick but shaped exactly like the
real one, so the wiring between modules is exercised rather than mocked.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from riskcheck import config, plots, report, runner
from riskcheck.build import BuiltPanel, DropRecord
from riskcheck.estimators import Spec

#: Starts in 2004 so that, after the 756-day burn-in, the scored span covers
#: the 2008-2009 stress window. Everything here is after TEST_START, which is
#: what we want: this fixture tests the machinery, not the sample split.
N_DAYS = 2600
PANEL_START = np.datetime64("2004-01-02")


@pytest.fixture(scope="module")
def toy_panel():
    from .conftest import FactorWorld

    world = FactorWorld(n_assets=5, n_factors=3)
    returns, _ = world.simulate(N_DAYS, seed=31)
    # The fundamental model needs six factor columns; pad the three real ones.
    rng = np.random.default_rng(2)
    factors = np.column_stack(
        [
            rng.multivariate_normal(np.zeros(3), world.factor_cov, size=N_DAYS),
            rng.normal(0, 0.004, size=(N_DAYS, 3)),
        ]
    )
    dates = np.busday_offset(PANEL_START, np.arange(N_DAYS), roll="forward")
    return BuiltPanel(
        asset_set="industry49",  # reuse a configured key so report helpers apply
        label="toy portfolios",
        dates=dates,
        names=tuple(f"A{i}" for i in range(world.n_assets)),
        excess_returns=returns,
        total_returns=returns,
        factors=factors,
        factor_names=config.FACTOR_NAMES,
        rf=np.zeros(N_DAYS),
        drops=DropRecord(
            span_start=str(dates[0]),
            span_end=str(dates[-1]),
            n_dates_in_span=N_DAYS,
            dropped_series=[],
            dropped_dates=[],
            n_dropped_dates=0,
            dates_outside_factor_coverage=0,
        ),
    )


@pytest.fixture(scope="module")
def toy_run(toy_panel):
    specs = [
        Spec("sample", {"window": 252}),
        Spec("ledoit_wolf", {"window": 252}),
        Spec("vra", {"halflife": 126.0}, base=Spec("sample", {"window": 252})),
    ]
    return runner.run_test_evaluation(
        {"industry49": toy_panel},
        specs,
        asset_sets=("industry49",),
        verbose=False,
    )


def test_run_produces_one_result_per_spec_and_horizon(toy_run):
    assert len(toy_run.results) == 3 * len(config.HORIZONS)
    table = toy_run.table
    assert set(table.horizon) == set(config.HORIZONS)
    assert table.n_obs.min() > 0


def test_metrics_are_finite_and_in_plausible_ranges(toy_run):
    table = toy_run.table
    for column in (
        "long_only_bias_mean",
        "long_short_bias_mean",
        "mv_bias_mean",
        "mv_realized_vol_annual",
        "log_score",
    ):
        assert np.isfinite(table[column]).all(), column
    # Bias statistics are positive and not absurd on well-behaved synthetic data.
    assert table.long_only_bias_mean.between(0.5, 2.0).all()
    assert (table.mv_bias_mean > 0).all()
    # Volatilities are annualised fractions, not percentages.
    assert table.mv_realized_vol_annual.between(0.0, 2.0).all()


def test_log_score_interval_brackets_the_point_estimate(toy_run):
    table = toy_run.table
    assert (table.log_score_lo <= table.log_score).all()
    assert (table.log_score <= table.log_score_hi).all()


def test_minimum_variance_bias_matches_the_vol_ratio(toy_run):
    """mv bias should be close to realised vol over forecast vol.

    Not exact -- one is a root-mean-square about zero and the other a sample
    standard deviation about the mean, and the forecast vol is a mean of
    per-date volatilities rather than a root-mean-square -- but a large gap
    would mean the two are not measuring the same thing.
    """
    table = toy_run.table
    ratio = table.mv_realized_vol_annual / table.mv_forecast_vol_annual
    np.testing.assert_allclose(table.mv_bias_mean, ratio, rtol=0.15)


def test_stress_windows_are_reported_with_their_observation_counts(toy_run):
    stress = toy_run.stress_table
    assert not stress.empty
    # The fixture spans 2004-2014, so the GFC window is the one that lands.
    assert "GFC 2008-2009" in set(stress.window)
    assert {"estimator", "window", "n_obs", "mv_bias"} <= set(stress.columns)
    assert (stress.n_obs > 0).all()


def test_rolling_curves_cover_every_estimator_and_portfolio_set(toy_run):
    rolling = toy_run.rolling
    assert set(rolling.estimator) == {"Sample", "Ledoit-Wolf", "Sample + VRA"}
    for column in ("min_variance_bias", "long_only_bias", "long_short_bias"):
        assert column in rolling.columns
        assert rolling[column].notna().any()


def test_write_results_saves_every_artefact(toy_run, tmp_path):
    runner.write_results(toy_run, tmp_path)
    expected = (
        "metrics.csv",
        "stress.csv",
        "rolling_bias.csv",
        "metrics.json",
        "run_metadata.json",
    )
    for name in expected:
        assert (tmp_path / name).exists(), name

    metadata = json.loads((tmp_path / "run_metadata.json").read_text())
    # Provenance a reader needs to trust the numbers.
    for key in ("run_utc", "jax", "float64", "portfolio_seed", "burn_in_days", "specs"):
        assert key in metadata, key

    reloaded = runner.load_metrics(tmp_path)
    assert len(reloaded) == len(toy_run.results)


def test_headline_table_renders_every_estimator(toy_run):
    table, caption = report.headline_table(toy_run.table, "industry49", "1w")
    assert table.startswith("| Estimator |")
    for name in ("Sample", "Ledoit-Wolf", "Sample + VRA"):
        assert name in table
    assert caption["n_obs"] > 0
    assert caption["band"] == pytest.approx(np.sqrt(2 / caption["n_obs"]))


def test_report_tables_are_well_formed_markdown(toy_run):
    for text in (
        report.headline_table(toy_run.table, "industry49", "1w")[0],
        report.stress_table(toy_run.stress_table, "industry49", "1d"),
    ):
        lines = text.splitlines()
        width = lines[0].count("|")
        assert all(line.count("|") == width for line in lines), text[:200]


def test_figures_render_in_both_themes(toy_run, tmp_path):
    written = plots.rolling_bias_figure(toy_run.rolling, tmp_path / "fig")
    assert [p.name for p in written] == ["fig_light.png", "fig_dark.png"]
    for path in written:
        assert path.stat().st_size > 10_000  # a real plot, not a blank canvas


def test_estimator_order_is_stable_and_puts_adjusted_models_last():
    frame = pd.DataFrame({"estimator": ["EWMA + VRA", "Ledoit-Wolf", "Sample", "EWMA"]})
    assert report._estimator_order(frame) == ["Sample", "EWMA", "Ledoit-Wolf", "EWMA + VRA"]


def test_spec_round_trips_through_json():
    spec = Spec("vra", {"halflife": 504.0}, base=Spec("ewma", {"halflife": 252.0}))
    restored = Spec.from_json(json.loads(json.dumps(spec.to_json())))
    assert restored == spec
    assert restored.key == spec.key
    assert restored.label == spec.label
