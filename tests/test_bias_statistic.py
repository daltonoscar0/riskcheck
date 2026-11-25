"""The bias statistic must say 1 when the forecast is right, and 2 when the
forecast volatility is half what it should be.

These are the calibration checks for the metric itself. If they fail, every
number in the README is meaningless, so they are run against the real scoring
path (``build_observations`` -> ``bias_summary_chunked`` /
``minimum_variance_summary``) rather than a reimplementation.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from riskcheck import evaluate, portfolios
from riskcheck.evaluate import band_halfwidth

N_DAYS = 60_000
STRIDE = 5


@pytest.fixture(scope="module")
def truth():
    """Returns drawn iid from a known covariance, plus that covariance at every date."""
    from .conftest import FactorWorld

    world = FactorWorld()
    returns, _ = world.simulate(N_DAYS, seed=21)
    dates = np.arange(
        np.datetime64("1970-01-01"), np.datetime64("1970-01-01") + N_DAYS, dtype="datetime64[D]"
    )
    t_idx = np.arange(STRIDE - 1, N_DAYS - 1, STRIDE, dtype=np.int32)
    exact = jnp.broadcast_to(
        jnp.asarray(world.covariance), (len(t_idx), world.n_assets, world.n_assets)
    )
    return world, returns, dates, t_idx, exact


@pytest.mark.parametrize("horizon", ["1d", "1w", "1m"])
def test_true_covariance_gives_a_bias_statistic_of_one(truth, horizon):
    world, returns, dates, t_idx, exact = truth
    obs = evaluate.build_observations(returns, dates, t_idx, horizon)
    weights = {"long_only": portfolios.random_long_only(world.n_assets, 200, seed=4)}
    variance = portfolios.quadratic_forms(weights["long_only"], exact)

    summary = evaluate.bias_summary_chunked(obs, weights["long_only"], variance)
    tolerance = 4.0 * band_halfwidth(len(obs))
    assert summary.mean == pytest.approx(1.0, abs=tolerance)


@pytest.mark.parametrize("horizon", ["1d", "1w", "1m"])
def test_halving_forecast_volatility_gives_a_bias_statistic_of_two(truth, horizon):
    """Variance scaled by 1/4 means volatility halved, so z doubles."""
    world, returns, dates, t_idx, exact = truth
    obs = evaluate.build_observations(returns, dates, t_idx, horizon)
    weights = portfolios.random_long_only(world.n_assets, 200, seed=4)
    variance = portfolios.quadratic_forms(weights, exact / 4.0)

    summary = evaluate.bias_summary_chunked(obs, weights, variance)
    tolerance = 8.0 * band_halfwidth(len(obs))
    assert summary.mean == pytest.approx(2.0, abs=tolerance)


def test_doubling_forecast_volatility_gives_a_half(truth):
    world, returns, dates, t_idx, exact = truth
    obs = evaluate.build_observations(returns, dates, t_idx, "1w")
    weights = portfolios.random_long_only(world.n_assets, 200, seed=4)
    variance = portfolios.quadratic_forms(weights, exact * 4.0)
    summary = evaluate.bias_summary_chunked(obs, weights, variance)
    assert summary.mean == pytest.approx(0.5, abs=4.0 * band_halfwidth(len(obs)))


def test_long_short_portfolios_are_also_calibrated_by_the_truth(truth):
    world, returns, dates, t_idx, exact = truth
    obs = evaluate.build_observations(returns, dates, t_idx, "1w")
    weights = portfolios.random_long_short(world.n_assets, 200, seed=4)
    variance = portfolios.quadratic_forms(weights, exact)
    summary = evaluate.bias_summary_chunked(obs, weights, variance)
    assert summary.mean == pytest.approx(1.0, abs=4.0 * band_halfwidth(len(obs)))


def test_minimum_variance_is_calibrated_when_the_covariance_is_right(truth):
    """The hard portfolio is only hard because the covariance is estimated.

    Given the true matrix, the minimum-variance portfolio is as well calibrated
    as any other -- which is what makes its test-period bias statistic evidence
    about estimation error rather than about the metric.
    """
    _, returns, dates, t_idx, exact = truth
    obs = evaluate.build_observations(returns, dates, t_idx, "1w")
    weights = portfolios.minimum_variance_weights(exact)
    variance = portfolios.minimum_variance_forecast_variance(exact)
    summary = evaluate.minimum_variance_summary(obs, weights, variance)
    assert summary.bias.mean == pytest.approx(1.0, abs=4.0 * band_halfwidth(len(obs)))


def test_band_halfwidth_matches_the_published_formula():
    for n in (52, 336, 1345):
        assert band_halfwidth(n) == pytest.approx(np.sqrt(2.0 / n))


def test_observations_are_non_overlapping_and_tile_the_sample(truth):
    """Every day is used at most once, which is what the band assumes."""
    _, returns, dates, t_idx, _ = truth
    for horizon, days in (("1d", 1), ("1w", 5), ("1m", 20)):
        obs = evaluate.build_observations(returns, dates, t_idx, horizon)
        assert obs.days == days
        # End dates must be strictly increasing and spaced by the horizon.
        gaps = np.diff(obs.end_date).astype(int)
        assert np.all(gaps > 0)
        if horizon != "1d":
            # Exactly `days` trading days apart in index terms.
            spacing = np.diff(obs.forecast_index)
            assert np.all(spacing == spacing[0])


def test_horizon_returns_are_the_sum_of_the_days_they_cover(truth):
    _, returns, dates, t_idx, _ = truth
    weekly = evaluate.build_observations(returns, dates, t_idx, "1w")
    start = int(t_idx[0]) + 1
    np.testing.assert_allclose(weekly.returns[0], returns[start : start + 5].sum(axis=0))
    monthly = evaluate.build_observations(returns, dates, t_idx, "1m")
    np.testing.assert_allclose(monthly.returns[0], returns[start : start + 20].sum(axis=0))


def test_log_score_is_minimised_by_the_true_covariance(truth):
    """A proper scoring rule must prefer the truth to any rescaling of it."""
    _, returns, dates, t_idx, exact = truth
    obs = evaluate.build_observations(returns, dates, t_idx, "1w")
    at_truth = evaluate.log_score_per_observation(obs, exact).mean()
    for factor in (0.25, 0.5, 2.0, 4.0):
        assert evaluate.log_score_per_observation(obs, exact * factor).mean() > at_truth


def test_block_bootstrap_interval_brackets_the_mean():
    rng = np.random.default_rng(0)
    series = rng.normal(3.0, 1.0, size=2000)
    low, high = evaluate.block_bootstrap_ci(series, block=52, draws=500, seed=1)
    assert low < series.mean() < high
    assert high - low > 0
