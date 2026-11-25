"""The portfolios the forecasts are judged on.

Three sets, in increasing order of how hard they are on a covariance model:

* **Random long-only.** Weights from a flat Dirichlet, so every portfolio is a
  diversified long book. These are easy: they load mostly on the market
  direction, which every estimator gets roughly right.
* **Random long-short.** Dollar-neutral with unit gross exposure. Harder,
  because they live in the directions the market factor does not explain.
* **Minimum variance.** Built from each model's *own* forecast, so every model
  is asked about the portfolio it is most confident in. This is the one that
  separates the estimators, and it does so for a structural reason: the
  optimiser puts weight exactly where the model thinks risk is low, which is
  where estimation error has pushed the forecast down.

The random weights are drawn once from a fixed seed and then held constant
across every estimator, date and horizon, so differences between models are
never differences between portfolio draws.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from . import config


def random_long_only(n_assets: int, n_portfolios: int, seed: int) -> np.ndarray:
    """Flat-Dirichlet long-only weights, each row non-negative and summing to 1."""
    rng = np.random.default_rng(seed)
    weights = rng.dirichlet(np.ones(n_assets), size=n_portfolios)
    return weights.astype(np.float64)


def random_long_short(n_assets: int, n_portfolios: int, seed: int) -> np.ndarray:
    """Dollar-neutral weights with unit gross exposure.

    Gaussian draws are demeaned cross-sectionally so each row sums to zero,
    then scaled so the absolute weights sum to one. Fixing gross exposure
    rather than variance keeps the set independent of any covariance estimate.
    """
    rng = np.random.default_rng(seed + 1)
    raw = rng.standard_normal((n_portfolios, n_assets))
    raw -= raw.mean(axis=1, keepdims=True)
    gross = np.abs(raw).sum(axis=1, keepdims=True)
    return (raw / gross).astype(np.float64)


def standard_portfolio_sets(n_assets: int) -> dict[str, np.ndarray]:
    """The two fixed random sets, drawn from the configured seed."""
    return {
        "long_only": random_long_only(
            n_assets, config.N_RANDOM_PORTFOLIOS, config.PORTFOLIO_SEED
        ),
        "long_short": random_long_short(
            n_assets, config.N_RANDOM_PORTFOLIOS, config.PORTFOLIO_SEED
        ),
    }


@jax.jit
def minimum_variance_weights(covariances: jnp.ndarray) -> jnp.ndarray:
    """Fully invested minimum-variance weights, ``w = S^-1 1 / (1' S^-1 1)``.

    No sign constraint, which is the point: the unconstrained solution is free
    to take the large offsetting positions that expose a mis-estimated
    covariance matrix, and a long-only constraint would hide most of the effect.
    Solved rather than inverted, for numerical reasons.
    """
    ones = jnp.ones(covariances.shape[-1], dtype=covariances.dtype)
    raw = jnp.linalg.solve(covariances, ones)  # (D, N)
    return raw / jnp.sum(raw, axis=-1, keepdims=True)


@jax.jit
def minimum_variance_forecast_variance(covariances: jnp.ndarray) -> jnp.ndarray:
    """The model's own forecast variance for its minimum-variance portfolio.

    Equal to ``1 / (1' S^-1 1)``, which is what ``w' S w`` collapses to at the
    optimum.
    """
    ones = jnp.ones(covariances.shape[-1], dtype=covariances.dtype)
    raw = jnp.linalg.solve(covariances, ones)
    return 1.0 / jnp.sum(raw, axis=-1)


def _quadratic_form_chunk(weights: jnp.ndarray, covariances: jnp.ndarray) -> jnp.ndarray:
    """``w' S w`` for every weight row against every matrix in the chunk."""

    def one_date(cov):
        return jnp.sum((weights @ cov) * weights, axis=1)

    return jax.vmap(one_date)(covariances)


_quadratic_form_chunk_jit = jax.jit(_quadratic_form_chunk)


def quadratic_forms(
    weights: np.ndarray, covariances: jnp.ndarray, max_elements: int = 8_000_000
) -> jnp.ndarray:
    """Forecast variances for a fixed weight set at every date, shape ``(D, P)``.

    Written as a loop over date chunks because the natural one-shot einsum
    needs a ``(dates, portfolios, assets)`` intermediate -- two gigabytes for
    4,000 portfolios over 1,300 dates -- while chunking holds it to tens of
    megabytes at no cost in speed.
    """
    weights_j = jnp.asarray(weights)
    n_portfolios, n_assets = weights_j.shape
    chunk = max(1, max_elements // max(1, n_portfolios * n_assets))
    pieces = [
        _quadratic_form_chunk_jit(weights_j, covariances[lo : lo + chunk])
        for lo in range(0, covariances.shape[0], chunk)
    ]
    return jnp.concatenate(pieces, axis=0)
