"""Portfolio construction: constraints, and the closed form on a toy case."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from riskcheck import portfolios


def test_long_only_weights_are_non_negative_and_sum_to_one():
    weights = portfolios.random_long_only(12, 500, seed=1)
    assert weights.shape == (500, 12)
    assert np.all(weights >= 0)
    np.testing.assert_allclose(weights.sum(axis=1), 1.0, rtol=1e-12)


def test_long_short_weights_are_dollar_neutral_with_unit_gross():
    weights = portfolios.random_long_short(12, 500, seed=1)
    assert weights.shape == (500, 12)
    np.testing.assert_allclose(weights.sum(axis=1), 0.0, atol=1e-12)
    np.testing.assert_allclose(np.abs(weights).sum(axis=1), 1.0, rtol=1e-12)
    assert np.any(weights < 0) and np.any(weights > 0)


def test_the_two_random_sets_are_different_draws():
    long_only = portfolios.random_long_only(8, 50, seed=5)
    long_short = portfolios.random_long_short(8, 50, seed=5)
    assert not np.allclose(long_only, long_short)


def test_weights_are_reproducible_from_the_seed():
    first = portfolios.random_long_only(8, 50, seed=5)
    second = portfolios.random_long_only(8, 50, seed=5)
    np.testing.assert_array_equal(first, second)


def test_minimum_variance_matches_the_two_asset_closed_form():
    """For two assets the fully invested minimum-variance weight is

        w1 = (s2^2 - rho s1 s2) / (s1^2 + s2^2 - 2 rho s1 s2).
    """
    s1, s2, rho = 0.10, 0.20, 0.30
    covariance = np.array(
        [[s1**2, rho * s1 * s2], [rho * s1 * s2, s2**2]], dtype=np.float64
    )
    expected_w1 = (s2**2 - rho * s1 * s2) / (s1**2 + s2**2 - 2 * rho * s1 * s2)

    weights = np.asarray(portfolios.minimum_variance_weights(jnp.asarray(covariance)[None]))[0]
    assert weights[0] == pytest.approx(expected_w1)
    assert weights[1] == pytest.approx(1.0 - expected_w1)
    assert weights.sum() == pytest.approx(1.0)


def test_minimum_variance_satisfies_its_constraint_and_is_optimal():
    """Fully invested, and no feasible perturbation lowers the variance."""
    rng = np.random.default_rng(3)
    root = rng.normal(size=(6, 6))
    covariance = root @ root.T + 0.5 * np.eye(6)

    weights = np.asarray(portfolios.minimum_variance_weights(jnp.asarray(covariance)[None]))[0]
    assert weights.sum() == pytest.approx(1.0)

    variance = weights @ covariance @ weights
    # Any perturbation that keeps the weights summing to one must not help.
    for _ in range(200):
        direction = rng.normal(size=6)
        direction -= direction.mean()  # keeps the sum at one
        perturbed = weights + 1e-3 * direction
        assert perturbed @ covariance @ perturbed >= variance - 1e-15


def test_minimum_variance_forecast_variance_equals_the_quadratic_form():
    """`1 / (1' S^-1 1)` is what `w' S w` collapses to at the optimum."""
    rng = np.random.default_rng(4)
    root = rng.normal(size=(5, 7, 7))
    covariance = jnp.asarray(root @ np.swapaxes(root, 1, 2) + 0.4 * np.eye(7))

    weights = portfolios.minimum_variance_weights(covariance)
    shortcut = np.asarray(portfolios.minimum_variance_forecast_variance(covariance))
    w = np.asarray(weights)
    explicit = np.einsum("di,dij,dj->d", w, np.asarray(covariance), w)
    np.testing.assert_allclose(shortcut, explicit, rtol=1e-10)


def test_minimum_variance_beats_equal_weight_in_sample():
    rng = np.random.default_rng(6)
    root = rng.normal(size=(8, 8))
    covariance = root @ root.T + 0.3 * np.eye(8)
    weights = np.asarray(portfolios.minimum_variance_weights(jnp.asarray(covariance)[None]))[0]
    equal = np.full(8, 1 / 8)
    assert weights @ covariance @ weights < equal @ covariance @ equal


def test_quadratic_forms_match_a_direct_computation():
    rng = np.random.default_rng(8)
    weights = rng.normal(size=(37, 9))
    root = rng.normal(size=(11, 9, 9))
    covariance = jnp.asarray(root @ np.swapaxes(root, 1, 2))

    computed = np.asarray(portfolios.quadratic_forms(weights, covariance))
    expected = np.einsum("pi,dij,pj->dp", weights, np.asarray(covariance), weights)
    assert computed.shape == (11, 37)
    np.testing.assert_allclose(computed, expected, rtol=1e-10)


def test_quadratic_forms_are_chunked_without_changing_the_answer():
    """The chunk size is a memory knob and must not affect the result."""
    rng = np.random.default_rng(9)
    weights = rng.normal(size=(64, 5))
    root = rng.normal(size=(20, 5, 5))
    covariance = jnp.asarray(root @ np.swapaxes(root, 1, 2))

    big = np.asarray(portfolios.quadratic_forms(weights, covariance, max_elements=10**9))
    small = np.asarray(portfolios.quadratic_forms(weights, covariance, max_elements=1))
    np.testing.assert_allclose(big, small, rtol=1e-12)
