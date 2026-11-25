"""No forecast may depend on a return that had not happened yet.

The test replaces every return after a cut date with different random numbers
and recomputes. Any forecast made at or before the cut must come back
**bit-for-bit identical**: not merely close, because the arithmetic performed
for those dates is the same sequence of operations on the same inputs, so
anything other than exact equality means a later value reached it.

This is run for every estimator, including the volatility regime adjustment,
whose multiplier is an average over past forecast errors and is the easiest
place to leak a future return by accident.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from riskcheck import estimators as E
from riskcheck import evaluate, portfolios

N_DAYS = 3000
WINDOW = 252
STRIDE = 5

SPECS = [
    E.Spec("sample", {"window": WINDOW}),
    E.Spec("ewma", {"halflife": 42.0}),
    E.Spec("ewma", {"halflife": 252.0}),
    E.Spec("ledoit_wolf", {"window": WINDOW}),
    E.Spec("pca", {"window": WINDOW, "n_factors": 3}),
    E.Spec("fundamental", {"window": WINDOW, "factor_halflife": 63.0}),
    E.Spec("vra", {"halflife": 126.0}, base=E.Spec("ewma", {"halflife": 252.0})),
    E.Spec("vra", {"halflife": 63.0}, base=E.Spec("sample", {"window": WINDOW})),
]


@pytest.fixture(scope="module")
def panels():
    from .conftest import FactorWorld

    world = FactorWorld()
    returns, factors = world.simulate(N_DAYS, seed=17)
    t_idx = np.arange(WINDOW - 1, N_DAYS - 1, STRIDE, dtype=np.int32)
    return world, returns, factors, t_idx


def _perturb_after(array: np.ndarray, row: int, seed: int) -> np.ndarray:
    """Replace every row strictly after `row` with unrelated draws."""
    rng = np.random.default_rng(seed)
    out = array.copy()
    tail = out[row + 1 :]
    out[row + 1 :] = rng.normal(0.0, 5.0 * tail.std(), size=tail.shape)
    return out


@pytest.mark.parametrize("spec", SPECS, ids=lambda s: s.key)
def test_forecasts_are_unchanged_by_later_returns(panels, spec):
    _, returns, factors, t_idx = panels
    cut_position = len(t_idx) // 3
    cut_row = int(t_idx[cut_position])

    original = np.asarray(E.forecast(spec, jnp.asarray(returns), jnp.asarray(factors), t_idx))
    disturbed = np.asarray(
        E.forecast(
            spec,
            jnp.asarray(_perturb_after(returns, cut_row, seed=1)),
            jnp.asarray(_perturb_after(factors, cut_row, seed=2)),
            t_idx,
        )
    )

    np.testing.assert_array_equal(
        original[: cut_position + 1],
        disturbed[: cut_position + 1],
        err_msg=f"{spec.key} forecasts up to the cut changed when the future changed",
    )
    # Sanity: the perturbation must actually have done something downstream,
    # otherwise the test above would pass trivially.
    assert not np.allclose(original[-1], disturbed[-1])


@pytest.mark.parametrize("spec", SPECS, ids=lambda s: s.key)
def test_every_forecast_depends_only_on_its_own_past(panels, spec):
    """Sweep the cut across the sample rather than testing one point."""
    _, returns, factors, t_idx = panels
    original = np.asarray(E.forecast(spec, jnp.asarray(returns), jnp.asarray(factors), t_idx))

    for cut_position in (0, 5, len(t_idx) // 2, len(t_idx) - 2):
        cut_row = int(t_idx[cut_position])
        disturbed = np.asarray(
            E.forecast(
                spec,
                jnp.asarray(_perturb_after(returns, cut_row, seed=3 + cut_position)),
                jnp.asarray(_perturb_after(factors, cut_row, seed=99 + cut_position)),
                t_idx,
            )
        )
        np.testing.assert_array_equal(
            original[: cut_position + 1],
            disturbed[: cut_position + 1],
            err_msg=f"{spec.key} leaked future data at position {cut_position}",
        )


def test_minimum_variance_weights_inherit_causality(panels):
    """Weights are a function of the forecast, so they must be causal too."""
    _, returns, factors, t_idx = panels
    spec = E.Spec("ledoit_wolf", {"window": WINDOW})
    cut_position = len(t_idx) // 2
    cut_row = int(t_idx[cut_position])

    before = portfolios.minimum_variance_weights(
        E.forecast(spec, jnp.asarray(returns), jnp.asarray(factors), t_idx)
    )
    after = portfolios.minimum_variance_weights(
        E.forecast(
            spec,
            jnp.asarray(_perturb_after(returns, cut_row, seed=7)),
            jnp.asarray(factors),
            t_idx,
        )
    )
    np.testing.assert_array_equal(
        np.asarray(before)[: cut_position + 1], np.asarray(after)[: cut_position + 1]
    )


def test_observations_never_score_a_forecast_with_its_own_data(panels):
    """Each observation's realised window must start after its forecast date."""
    _, returns, _, t_idx = panels
    dates = np.arange(
        np.datetime64("1980-01-01"), np.datetime64("1980-01-01") + N_DAYS, dtype="datetime64[D]"
    )
    for horizon in ("1d", "1w", "1m"):
        obs = evaluate.build_observations(returns, dates, t_idx, horizon)
        forecast_rows = t_idx[obs.forecast_index]
        end_rows = np.searchsorted(dates, obs.end_date)
        # The realised period ends after the forecast date, and starts after it
        # too (end - horizon + 1 > forecast row).
        assert np.all(end_rows > forecast_rows), horizon
        assert np.all(end_rows - obs.days + 1 > forecast_rows), horizon
