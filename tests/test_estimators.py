"""Each estimator must recover a covariance it is capable of representing.

The synthetic world is a three-factor model plus diagonal noise, so its true
covariance is exactly low-rank-plus-diagonal. Every estimator here is able to
represent that structure, which makes a convergence failure a bug rather than
misspecification.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from riskcheck import estimators as E

from .conftest import relative_error

#: Long enough that sampling error is small relative to the tolerances below.
N_DAYS = 40_000


@pytest.fixture(scope="module")
def simulated(request):
    from .conftest import FactorWorld

    world = FactorWorld()
    returns, factors = world.simulate(N_DAYS, seed=3)
    return world, jnp.asarray(returns), jnp.asarray(factors)


def _last_two_dates(n_days: int) -> np.ndarray:
    """A two-date grid ending at the last row, with the usual 5-day stride."""
    return np.array([n_days - 6, n_days - 1], dtype=np.int32)


def test_halflife_to_lambda_halves_the_weight():
    for halflife in (11.0, 21.0, 252.0):
        assert E.halflife_to_lambda(halflife) ** halflife == pytest.approx(0.5)


def test_sample_covariance_converges(simulated):
    world, returns, _ = simulated
    window = 20_000
    estimate = E.sample_covariance(returns, np.array([N_DAYS - 1], dtype=np.int32), window)
    assert relative_error(np.asarray(estimate[0]), world.covariance) < 0.05


def test_ewma_covariance_converges(simulated):
    world, returns, _ = simulated
    estimate = E.ewma_covariance(returns, _last_two_dates(N_DAYS), halflife=4000.0)
    assert relative_error(np.asarray(estimate[-1]), world.covariance) < 0.08


def test_ledoit_wolf_converges(simulated):
    """With enough data the shrinkage intensity goes to zero and it matches."""
    world, returns, _ = simulated
    estimate = E.ledoit_wolf_covariance(
        returns, np.array([N_DAYS - 1], dtype=np.int32), window=20_000
    )
    assert relative_error(np.asarray(estimate[0]), world.covariance) < 0.05


def test_ledoit_wolf_beats_the_sample_matrix_when_data_is_scarce(simulated):
    """Shrinkage earns its keep only at small T/N, so test it there.

    At a 20-day window for 6 assets the sample matrix is badly conditioned and
    shrinkage should win clearly. Averaged over many windows, because on any
    single draw either can come out ahead.
    """
    world, returns, _ = simulated
    window = 20
    dates = np.arange(window - 1, window - 1 + 300 * 7, 7, dtype=np.int32)

    sample = np.asarray(E.sample_covariance(returns, dates, window))
    shrunk = np.asarray(E.ledoit_wolf_covariance(returns, dates, window))

    sample_error = np.mean([relative_error(m, world.covariance) for m in sample])
    shrunk_error = np.mean([relative_error(m, world.covariance) for m in shrunk])
    assert shrunk_error < sample_error

    # And it should be better conditioned, which is the practical benefit.
    assert np.median(np.linalg.cond(shrunk)) < np.median(np.linalg.cond(sample))


def test_ledoit_wolf_lies_between_the_sample_matrix_and_its_target(simulated):
    """A valid shrinkage intensity is in [0, 1], so the result is a blend."""
    _, returns, _ = simulated
    window = 30
    t = np.array([window - 1], dtype=np.int32)
    sample = np.asarray(E.sample_covariance(returns, t, window))[0]
    shrunk = np.asarray(E.ledoit_wolf_covariance(returns, t, window))[0]

    scale = np.sqrt(np.diag(sample))
    correlation = sample / np.outer(scale, scale)
    n = sample.shape[0]
    mean_correlation = (correlation.sum() - np.trace(correlation)) / (n * (n - 1))
    target = mean_correlation * np.outer(scale, scale)
    np.fill_diagonal(target, np.diag(sample))

    off = ~np.eye(n, dtype=bool)
    implied = (shrunk - sample)[off] / (target - sample)[off]
    assert np.all(implied > -1e-9) and np.all(implied < 1 + 1e-9)
    # One intensity for the whole matrix, so every element implies the same one.
    assert implied.std() < 1e-9


def test_ledoit_wolf_preserves_variances(simulated):
    """The constant-correlation target shares the sample's diagonal."""
    _, returns, _ = simulated
    window = 500
    t = np.array([window - 1], dtype=np.int32)
    sample = np.asarray(E.sample_covariance(returns, t, window))[0]
    shrunk = np.asarray(E.ledoit_wolf_covariance(returns, t, window))[0]
    np.testing.assert_allclose(np.diag(shrunk), np.diag(sample), rtol=1e-10)


def test_pca_converges_when_specific_risk_is_small():
    """PCA recovers a factor structure only to the extent the diagonal is small.

    A principal-component model splits the matrix into a low-rank part plus a
    *diagonal*; a factor model splits it into a low-rank part plus
    idiosyncratic noise. Those agree only when the specific variances are small
    or all equal, so this uses a world where the common factors dominate. With
    a large uneven diagonal PCA keeps a bias no sample size removes -- a
    property of the estimator, not a defect in the code.
    """
    from .conftest import FactorWorld

    world = FactorWorld(specific_scale=0.25)
    returns, _ = world.simulate(N_DAYS, seed=5)
    estimate = E.pca_covariance(
        jnp.asarray(returns),
        np.array([N_DAYS - 1], dtype=np.int32),
        window=20_000,
        n_factors=world.n_factors,
    )
    assert relative_error(np.asarray(estimate[0]), world.covariance) < 0.05


def test_pca_with_the_true_factor_count_beats_too_few(simulated):
    world, returns, _ = simulated
    t = np.array([N_DAYS - 1], dtype=np.int32)
    errors = {
        k: relative_error(
            np.asarray(E.pca_covariance(returns, t, 20_000, k))[0], world.covariance
        )
        for k in (1, 2, world.n_factors)
    }
    assert errors[world.n_factors] < errors[2] < errors[1]


def test_pca_at_full_rank_reproduces_the_sample_matrix(simulated):
    """With k = N the common part is the whole matrix; only the floor remains."""
    world, returns, _ = simulated
    window, t = 1000, np.array([999], dtype=np.int32)
    sample = np.asarray(E.sample_covariance(returns, t, window))[0]
    full = np.asarray(E.pca_covariance(returns, t, window, n_factors=world.n_assets))[0]
    # The specific-variance floor adds a tiny ridge; nothing else should differ.
    np.testing.assert_allclose(full, sample + np.diag(np.diag(full - sample)), rtol=1e-9)
    assert relative_error(full, sample) < 1e-3


def test_pca_reproduces_the_sample_diagonal_exactly(simulated):
    """Specific variance is defined as the residual, so variances must match."""
    _, returns, _ = simulated
    window, t = 1000, np.array([999], dtype=np.int32)
    sample = np.asarray(E.sample_covariance(returns, t, window))[0]
    pca = np.asarray(E.pca_covariance(returns, t, window, n_factors=2))[0]
    np.testing.assert_allclose(np.diag(pca), np.diag(sample), rtol=1e-8)


def test_pca_multi_k_matches_the_single_k_path(simulated):
    _, returns, _ = simulated
    t = np.arange(1000, 1200, 5, dtype=np.int32)
    decomposition = E.pca_eigendecomposition(returns, t, window=500)
    for k in (1, 3, 5):
        shared = np.asarray(E.pca_assemble(decomposition, k))
        direct = np.asarray(E.pca_covariance(returns, t, 500, k))
        np.testing.assert_allclose(shared, direct, rtol=1e-12, atol=0)


def test_fundamental_model_converges_with_the_true_factors(simulated):
    world, returns, factors = simulated
    estimate = E.fundamental_covariance(
        returns,
        factors,
        _last_two_dates(N_DAYS),
        window=20_000,
        factor_halflife=4000.0,
    )
    assert relative_error(np.asarray(estimate[-1]), world.covariance) < 0.08


def test_every_estimator_returns_a_positive_definite_symmetric_matrix(simulated):
    world, returns, factors = simulated
    t = np.arange(800, 1000, 5, dtype=np.int32)
    specs = [
        E.Spec("sample", {"window": 252}),
        E.Spec("ewma", {"halflife": 42.0}),
        E.Spec("ledoit_wolf", {"window": 252}),
        E.Spec("pca", {"window": 252, "n_factors": 3}),
        E.Spec("fundamental", {"window": 252, "factor_halflife": 42.0}),
        E.Spec("vra", {"halflife": 63.0}, base=E.Spec("ewma", {"halflife": 42.0})),
    ]
    for spec in specs:
        covariances = np.asarray(E.forecast(spec, returns, factors, t))
        assert covariances.shape == (len(t), world.n_assets, world.n_assets), spec.key
        np.testing.assert_allclose(
            covariances, np.swapaxes(covariances, 1, 2), rtol=1e-10, err_msg=spec.key
        )
        assert np.linalg.eigvalsh(covariances).min() > 0, spec.key


def test_ewma_normalisation_is_exact_not_asymptotic():
    """At a long half-life the truncated weight sum matters; it must be divided out.

    A constant-magnitude return series has a known second moment whatever the
    weights are, so an estimator that assumed the weights summed to
    ``1/(1-lambda)`` would come back low by the truncation error.
    """
    n_days = 400
    returns = jnp.full((n_days, 1), 0.01)
    t = np.array([n_days - 6, n_days - 1], dtype=np.int32)
    estimate = E.ewma_covariance(returns, t, halflife=1000.0)
    np.testing.assert_allclose(np.asarray(estimate)[:, 0, 0], 1e-4, rtol=1e-12)


def test_volatility_regime_multiplier_is_one_at_the_first_date(simulated):
    """Nothing has been observed yet, so there is nothing to adjust."""
    _, returns, _ = simulated
    t = np.arange(800, 1000, 5, dtype=np.int32)
    base = E.sample_covariance(returns, t, 252)
    multiplier = E.volatility_regime_multiplier(returns, base, t, halflife=63.0)
    assert float(multiplier[0]) == pytest.approx(1.0)
    assert np.all(np.asarray(multiplier) > 0)


def test_volatility_regime_multiplier_scales_up_when_risk_is_understated(simulated):
    """Feed it a forecast that is four times too small and it should push up."""
    _, returns, _ = simulated
    t = np.arange(800, 1000, 5, dtype=np.int32)
    base = E.sample_covariance(returns, t, 252)
    multiplier = np.asarray(E.volatility_regime_multiplier(returns, base / 4.0, t, halflife=63.0))
    # Variance understated fourfold => the multiplier should approach 4.
    assert multiplier[-1] == pytest.approx(4.0, rel=0.35)


def test_window_longer_than_the_burn_in_is_refused():
    from riskcheck import config

    spec = E.Spec("sample", {"window": config.BURN_IN + 1})
    with pytest.raises(ValueError, match="burn-in"):
        E.forecast(
            spec, jnp.zeros((2000, 3)), jnp.zeros((2000, 6)), np.array([1500], dtype=np.int32)
        )
