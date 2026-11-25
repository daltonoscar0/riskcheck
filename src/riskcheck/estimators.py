"""Covariance forecasters, all in JAX and all causal by construction.

Every estimator maps a rebalance date index ``t`` to a forecast of the
covariance of *daily* returns, using only rows ``0..t`` of the return matrix.
The causality is structural rather than checked after the fact:

* window estimators read ``R[t-W+1 : t+1]`` through a ``dynamic_slice`` whose
  start index is ``t-W+1`` and whose length is fixed;
* the exponentially weighted ones are a forward ``lax.scan`` that has only
  consumed days up to ``t`` when it emits the forecast for ``t``.

There is no code path in which a later return reaches an earlier forecast, and
``tests/test_no_lookahead.py`` perturbs the tail of the panel to confirm it.

Two conventions are shared by all six estimators, so that the comparison is
like-for-like:

**Second moments, not centred covariances.** All of them estimate
``E[r r']`` rather than ``E[(r-mu)(r-mu)']``. For daily equity returns the mean
is around 0.03% against a 1.3% standard deviation, so ``mu^2`` is about 0.05%
of the variance -- far smaller than the differences between these estimators --
and leaving it out avoids spending estimation error on a parameter the
evaluation does not need. It also keeps the forecast consistent with the bias
statistic, which standardises raw returns rather than demeaned ones.

**Exact exponential normalisation.** The EWMA estimators divide by the realised
sum of weights rather than assuming it has converged to ``1/(1-lambda)``. At a
126-day half-life the truncation error at the first forecast date is 1.6%, which
would show up directly as a bias-statistic miss.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import partial
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from . import config

# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def halflife_to_lambda(halflife_days: float) -> float:
    """RiskMetrics decay from a half-life, so ``lambda ** halflife == 0.5``."""
    return float(2.0 ** (-1.0 / halflife_days))


def rebalance_indices(n_dates: int, burn_in: int = config.BURN_IN, stride: int = config.REBALANCE_STRIDE) -> np.ndarray:
    """Row indices at which forecasts are made.

    The first is ``burn_in - 1`` so that the longest window an estimator may
    ask for is exactly available, and they then step by ``stride`` trading
    days. The last one keeps at least one day of future return to score.
    """
    if n_dates <= burn_in:
        raise ValueError(f"need more than {burn_in} dates, got {n_dates}")
    return np.arange(burn_in - 1, n_dates - 1, stride, dtype=np.int32)


def _window(returns: jnp.ndarray, t: jnp.ndarray, window: int) -> jnp.ndarray:
    """The ``window`` rows ending at and including row ``t``.

    ``dynamic_slice`` wants every start index to share one dtype, which the
    literal column index would otherwise break under ``jax_enable_x64``.
    """
    start = jnp.asarray(t) - (window - 1)
    column = jnp.zeros((), dtype=start.dtype)
    return jax.lax.dynamic_slice(returns, (start, column), (window, returns.shape[1]))


def _map_over_dates(
    per_date, arrays: tuple[jnp.ndarray, ...], t_idx: np.ndarray, chunk_size: int = 256
) -> jnp.ndarray:
    """vmap ``per_date(*arrays, t)`` over rebalance dates, in chunks.

    Chunking is what keeps this inside memory: materialising every window at
    once for 2,700 dates and a three-year window would be gigabytes, while a
    chunk of 256 is a few tens of megabytes. The work itself is a batched
    matmul, so the chunks cost nothing in speed.

    The panels are passed as arguments rather than captured in a closure, so
    jit treats them as inputs instead of folding a 14,000-row constant into
    every compiled kernel.
    """
    mapped = jax.jit(jax.vmap(per_date, in_axes=(None,) * len(arrays) + (0,)))
    pieces = [
        mapped(*arrays, jnp.asarray(t_idx[lo : lo + chunk_size]))
        for lo in range(0, len(t_idx), chunk_size)
    ]
    return jnp.concatenate(pieces, axis=0)


def _symmetrise(matrix: jnp.ndarray) -> jnp.ndarray:
    return 0.5 * (matrix + jnp.swapaxes(matrix, -1, -2))


def _floor_specific(specific: jnp.ndarray, reference: jnp.ndarray) -> jnp.ndarray:
    """Keep specific variances strictly positive so the matrix stays invertible."""
    floor = config.SPECIFIC_VARIANCE_FLOOR * jnp.mean(reference)
    return jnp.maximum(specific, floor)


# ---------------------------------------------------------------------------
# Exponentially weighted second moment, evaluated on a regular date grid
# ---------------------------------------------------------------------------


def ewma_second_moment(
    data: jnp.ndarray,
    lam: float,
    t_first: int,
    stride: int,
    n_dates: int,
    first_day: int = 0,
) -> jnp.ndarray:
    """Exponentially weighted ``E[x x']`` at ``t_first + i*stride`` for ``i < n_dates``.

    Each output includes days ``first_day .. t_i`` inclusive, weighting day
    ``d`` by ``lam ** (t_i - d)`` and dividing by the sum of those weights. Any
    output whose window is empty comes back as zeros.

    The first date is handled by one direct weighted matmul over the burn-in,
    and the rest by a scan over ``stride``-day blocks, so the cost is linear in
    the number of days and nothing of size ``T x K x K`` is ever materialised.
    """
    n_cols = data.shape[1]

    # Phase A: everything up to and including the first evaluation date.
    length = t_first - first_day + 1
    if length > 0:
        block = jax.lax.dynamic_slice(data, (first_day, 0), (length, n_cols))
        weights = lam ** jnp.arange(length - 1, -1, -1, dtype=data.dtype)
        moment = (block * weights[:, None]).T @ block
        weight_sum = jnp.sum(weights)
    else:
        moment = jnp.zeros((n_cols, n_cols), dtype=data.dtype)
        weight_sum = jnp.asarray(0.0, dtype=data.dtype)

    if n_dates == 1:
        total = jnp.where(weight_sum > 0, weight_sum, 1.0)
        return (moment / total)[None]

    # Phase B: blocks of `stride` days, one emission per block.
    n_blocks = n_dates - 1
    blocks = jax.lax.dynamic_slice(
        data, (t_first + 1, 0), (n_blocks * stride, n_cols)
    ).reshape(n_blocks, stride, n_cols)

    def step(carry, block):
        moment_c, weight_c = carry
        # `stride` is a Python int, so this unrolls into a handful of updates
        # and the scan carries one K x K matrix rather than one per day.
        for k in range(stride):
            row = block[k]
            moment_c = lam * moment_c + jnp.outer(row, row)
            weight_c = lam * weight_c + 1.0
        return (moment_c, weight_c), (moment_c, weight_c)

    _, (moments, weight_sums) = jax.lax.scan(step, (moment, weight_sum), blocks)

    all_moments = jnp.concatenate([moment[None], moments], axis=0)
    all_weights = jnp.concatenate([weight_sum[None], weight_sums], axis=0)
    safe = jnp.where(all_weights > 0, all_weights, 1.0)
    return all_moments / safe[:, None, None]


# ---------------------------------------------------------------------------
# The estimators
# ---------------------------------------------------------------------------


def sample_covariance(returns: jnp.ndarray, t_idx: np.ndarray, window: int) -> jnp.ndarray:
    """Equally weighted second moment over the trailing ``window`` days."""

    def per_date(data, t):
        block = _window(data, t, window)
        return block.T @ block / window

    return _map_over_dates(per_date, (returns,), t_idx)


def ewma_covariance(returns: jnp.ndarray, t_idx: np.ndarray, halflife: float) -> jnp.ndarray:
    """RiskMetrics-style exponentially weighted second moment."""
    stride = int(t_idx[1] - t_idx[0]) if len(t_idx) > 1 else config.REBALANCE_STRIDE
    return ewma_second_moment(
        returns,
        halflife_to_lambda(halflife),
        t_first=int(t_idx[0]),
        stride=stride,
        n_dates=len(t_idx),
    )


def ledoit_wolf_covariance(
    returns: jnp.ndarray, t_idx: np.ndarray, window: int
) -> jnp.ndarray:
    """Ledoit-Wolf shrinkage of the sample second moment to constant correlation.

    This is the estimator of Ledoit and Wolf (2004), "Honey, I Shrunk the
    Sample Covariance Matrix", with the shrinkage target built from the average
    sample correlation and the intensity estimated from the data rather than
    tuned. The window length is the only free parameter.
    """

    def per_date(data, t):
        block = _window(data, t, window)
        w = float(window)
        sample = block.T @ block / w
        variances = jnp.diag(sample)
        scale = jnp.sqrt(variances)
        outer_scale = jnp.outer(scale, scale)

        correlation = sample / outer_scale
        n = sample.shape[0]
        off_diagonal_count = n * (n - 1)
        mean_correlation = (jnp.sum(correlation) - jnp.sum(jnp.diag(correlation))) / off_diagonal_count

        target = mean_correlation * outer_scale
        target = target - jnp.diag(jnp.diag(target)) + jnp.diag(variances)

        # pi: asymptotic variance of the sample second moments.
        squared = block**2
        pi_matrix = squared.T @ squared / w - sample**2
        pi_hat = jnp.sum(pi_matrix)

        # rho: covariance between the sample moments and the target's moments.
        # theta_ii,ij = E[x_i^3 x_j] - s_ii s_ij, and symmetrically for jj.
        cubed = block**3
        third = cubed.T @ block / w
        theta_ii = third - variances[:, None] * sample
        theta_jj = third.T - variances[None, :] * sample

        ratio = scale[None, :] / scale[:, None]  # sqrt(s_jj / s_ii)
        cross = 0.5 * mean_correlation * (ratio * theta_ii + (1.0 / ratio) * theta_jj)
        off_diagonal = ~jnp.eye(n, dtype=bool)
        rho_hat = jnp.sum(jnp.diag(pi_matrix)) + jnp.sum(jnp.where(off_diagonal, cross, 0.0))

        gamma_hat = jnp.sum((target - sample) ** 2)

        intensity = (pi_hat - rho_hat) / jnp.maximum(gamma_hat, 1e-300) / w
        intensity = jnp.clip(intensity, 0.0, 1.0)
        return _symmetrise(intensity * target + (1.0 - intensity) * sample)

    return _map_over_dates(per_date, (returns,), t_idx)


def pca_covariance(
    returns: jnp.ndarray, t_idx: np.ndarray, window: int, n_factors: int
) -> jnp.ndarray:
    """Statistical factor model: top ``k`` principal components plus diagonal noise.

    The diagonal is reproduced exactly -- specific variance is defined as what
    the common factors leave over -- so the model differs from the sample
    estimate only in the off-diagonal structure.
    """

    def per_date(data, t):
        block = _window(data, t, window)
        sample = block.T @ block / window
        # eigh returns ascending eigenvalues, so the last k are the largest.
        eigenvalues, eigenvectors = jnp.linalg.eigh(sample)
        top_values = jnp.maximum(eigenvalues[-n_factors:], 0.0)
        loadings = eigenvectors[:, -n_factors:] * jnp.sqrt(top_values)[None, :]
        common = loadings @ loadings.T
        specific = _floor_specific(jnp.diag(sample) - jnp.diag(common), jnp.diag(sample))
        return _symmetrise(common + jnp.diag(specific))

    return _map_over_dates(per_date, (returns,), t_idx)


def pca_covariance_multi_k(
    returns: jnp.ndarray, t_idx: np.ndarray, window: int, factor_counts: tuple[int, ...]
) -> dict[int, jnp.ndarray]:
    """``pca_covariance`` for several ``k`` at once, sharing the eigendecomposition.

    The batched ``eigh`` is by far the most expensive thing riskcheck does, and
    it does not depend on ``k``. Doing it once per window instead of once per
    (window, k) pair is what keeps the validation sweep to seconds.
    """

    def per_date(data, t):
        block = _window(data, t, window)
        sample = block.T @ block / window
        eigenvalues, eigenvectors = jnp.linalg.eigh(sample)
        return jnp.maximum(eigenvalues, 0.0), eigenvectors, jnp.diag(sample)

    mapped = jax.jit(jax.vmap(per_date, in_axes=(None, 0)))
    chunk = 256
    values, vectors, variances = (
        jnp.concatenate(parts, axis=0)
        for parts in zip(
            *(
                mapped(returns, jnp.asarray(t_idx[lo : lo + chunk]))
                for lo in range(0, len(t_idx), chunk)
            ),
            strict=True,
        )
    )

    @partial(jax.jit, static_argnames=("k",))
    def assemble(values, vectors, variances, k: int):
        loadings = vectors[:, :, -k:] * jnp.sqrt(values[:, None, -k:])
        common = loadings @ jnp.swapaxes(loadings, 1, 2)
        common_variance = jnp.diagonal(common, axis1=1, axis2=2)
        floor = config.SPECIFIC_VARIANCE_FLOOR * jnp.mean(variances, axis=1, keepdims=True)
        specific = jnp.maximum(variances - common_variance, floor)
        return _symmetrise(common + jax.vmap(jnp.diag)(specific))

    return {k: assemble(values, vectors, variances, k) for k in factor_counts}


def fundamental_covariance(
    returns: jnp.ndarray,
    factors: jnp.ndarray,
    t_idx: np.ndarray,
    window: int,
    factor_halflife: float,
) -> jnp.ndarray:
    """Fama-French 5 + momentum factor model.

    Betas come from a time-series regression over the trailing window, which is
    slow-moving by design; the factor covariance is exponentially weighted, so
    the model's overall volatility can react quickly. That split is the usual
    commercial arrangement, and the two parameters -- regression window and
    factor-covariance half-life -- are chosen on the validation span.
    """
    stride = int(t_idx[1] - t_idx[0]) if len(t_idx) > 1 else config.REBALANCE_STRIDE
    factor_cov = ewma_second_moment(
        factors,
        halflife_to_lambda(factor_halflife),
        t_first=int(t_idx[0]),
        stride=stride,
        n_dates=len(t_idx),
    )

    n_factors = factors.shape[1]

    def per_date(asset_data, factor_data, t, factor_cov_t):
        asset_block = _window(asset_data, t, window)
        factor_block = _window(factor_data, t, window)
        gram = factor_block.T @ factor_block
        # A whisker of ridge keeps the solve well behaved; the factors are not
        # collinear over a year of daily data, but this costs nothing.
        gram = gram + 1e-12 * jnp.trace(gram) * jnp.eye(n_factors, dtype=gram.dtype)
        betas = jnp.linalg.solve(gram, factor_block.T @ asset_block)  # (K, N)
        residuals = asset_block - factor_block @ betas
        specific = jnp.mean(residuals**2, axis=0)
        total_variance = jnp.mean(asset_block**2, axis=0)
        specific = _floor_specific(specific, total_variance)
        common = betas.T @ factor_cov_t @ betas
        return _symmetrise(common + jnp.diag(specific))

    mapped = jax.jit(jax.vmap(per_date, in_axes=(None, None, 0, 0)))
    chunk = 256
    pieces = [
        mapped(
            returns,
            factors,
            jnp.asarray(t_idx[lo : lo + chunk]),
            factor_cov[lo : lo + chunk],
        )
        for lo in range(0, len(t_idx), chunk)
    ]
    return jnp.concatenate(pieces, axis=0)


# ---------------------------------------------------------------------------
# Volatility regime adjustment
# ---------------------------------------------------------------------------


@jax.jit
def _daily_cross_sectional_bias(
    returns_block: jnp.ndarray, forecast_variances: jnp.ndarray
) -> jnp.ndarray:
    """Cross-sectional bias statistic for each day owned by a forecast.

    ``returns_block`` is ``(n_blocks, stride, N)`` and ``forecast_variances`` is
    ``(n_blocks, N)``, the diagonal of the forecast in force over that block.
    """
    standardised = returns_block**2 / forecast_variances[:, None, :]
    return jnp.sqrt(jnp.mean(standardised, axis=2))  # (n_blocks, stride)


def volatility_regime_multiplier(
    returns: jnp.ndarray,
    base_covariances: jnp.ndarray,
    t_idx: np.ndarray,
    halflife: float,
) -> jnp.ndarray:
    """Barra-style volatility regime adjustment factor, one per rebalance date.

    For each past day the cross-sectional bias statistic
    ``B_d = sqrt(mean_i r_di^2 / sigma_di^2)`` says how far that day's returns
    ran from what the model forecast for it. Exponentially averaging ``B^2``
    over days strictly before the forecast date gives a multiplier that scales
    the whole covariance matrix up in a volatility regime the model is lagging
    and down when it is over-forecasting.

    Only days already owned by an earlier forecast enter the average, so the
    first rebalance date gets a multiplier of exactly 1.
    """
    stride = int(t_idx[1] - t_idx[0])
    n_dates = len(t_idx)
    first = int(t_idx[0])
    n_blocks = n_dates - 1

    forecast_variances = jnp.diagonal(base_covariances, axis1=1, axis2=2)[:-1]  # (n-1, N)
    day_block = jax.lax.dynamic_slice(
        returns, (first + 1, 0), (n_blocks * stride, returns.shape[1])
    ).reshape(n_blocks, stride, returns.shape[1])
    bias = _daily_cross_sectional_bias(day_block, forecast_variances)  # (n-1, stride)

    # Lay the per-day statistic back onto the day axis so the EWMA machinery
    # (which indexes by absolute day) can consume it, then average B^2 over
    # days `first+1 .. t_i`.
    series = jnp.zeros((returns.shape[0], 1), dtype=returns.dtype)
    series = series.at[first + 1 : first + 1 + n_blocks * stride, 0].set(bias.reshape(-1))

    mean_square = ewma_second_moment(
        series,
        halflife_to_lambda(halflife),
        t_first=first,
        stride=stride,
        n_dates=n_dates,
        first_day=first + 1,
    )[:, 0, 0]
    # Empty window at the first date comes back as zero; that means "no
    # adjustment", not "zero volatility".
    mean_square = jnp.where(mean_square > 0, mean_square, 1.0)
    return mean_square


# ---------------------------------------------------------------------------
# Specs and dispatch
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Spec:
    """A fully specified estimator: family plus frozen hyperparameters."""

    family: str
    params: dict[str, Any] = field(default_factory=dict)
    base: Spec | None = None  # only for the regime adjustment

    @property
    def key(self) -> str:
        bits = "_".join(f"{k}{v}" for k, v in sorted(self.params.items()))
        if self.base is not None:
            return f"{self.family}[{self.base.key}]" + (f"_{bits}" if bits else "")
        return f"{self.family}_{bits}" if bits else self.family

    @property
    def label(self) -> str:
        p = self.params
        match self.family:
            case "sample":
                return f"Sample ({p['window'] // 252}y window)"
            case "ewma":
                return f"EWMA (half-life {p['halflife']:g}d)"
            case "ledoit_wolf":
                return f"Ledoit-Wolf ({p['window'] // 252}y window)"
            case "pca":
                return f"PCA factor (k={p['n_factors']}, {p['window'] // 252}y)"
            case "fundamental":
                return f"FF5+Mom factor ({p['window'] // 252}y, factor HL {p['factor_halflife']:g}d)"
            case "vra":
                assert self.base is not None
                return f"{self.base.label} + regime adj. (HL {p['halflife']:g}d)"
        raise ValueError(f"unknown family {self.family}")

    @property
    def short_label(self) -> str:
        match self.family:
            case "sample":
                return "Sample"
            case "ewma":
                return "EWMA"
            case "ledoit_wolf":
                return "Ledoit-Wolf"
            case "pca":
                return "PCA factor"
            case "fundamental":
                return "FF5+Mom factor"
            case "vra":
                assert self.base is not None
                return f"{self.base.short_label} + VRA"
        raise ValueError(f"unknown family {self.family}")

    def to_json(self) -> dict:
        out: dict[str, Any] = {"family": self.family, "params": dict(self.params)}
        if self.base is not None:
            out["base"] = self.base.to_json()
        return out

    @classmethod
    def from_json(cls, blob: dict) -> Spec:
        base = cls.from_json(blob["base"]) if blob.get("base") else None
        return cls(family=blob["family"], params=dict(blob["params"]), base=base)

    @property
    def required_window(self) -> int:
        """Longest history the estimator reads, used to sanity-check the burn-in."""
        if self.family == "vra":
            assert self.base is not None
            return self.base.required_window
        return int(self.params.get("window", 0))


def forecast(
    spec: Spec, returns: jnp.ndarray, factors: jnp.ndarray, t_idx: np.ndarray
) -> jnp.ndarray:
    """Covariance forecasts at every rebalance date, shape ``(n_dates, N, N)``."""
    if spec.required_window > config.BURN_IN:
        raise ValueError(
            f"{spec.key} needs {spec.required_window} days but the burn-in is "
            f"{config.BURN_IN}; the first forecast would read past the start of the panel"
        )
    p = spec.params
    match spec.family:
        case "sample":
            return sample_covariance(returns, t_idx, int(p["window"]))
        case "ewma":
            return ewma_covariance(returns, t_idx, float(p["halflife"]))
        case "ledoit_wolf":
            return ledoit_wolf_covariance(returns, t_idx, int(p["window"]))
        case "pca":
            return pca_covariance(returns, t_idx, int(p["window"]), int(p["n_factors"]))
        case "fundamental":
            return fundamental_covariance(
                returns, factors, t_idx, int(p["window"]), float(p["factor_halflife"])
            )
        case "vra":
            assert spec.base is not None
            base = forecast(spec.base, returns, factors, t_idx)
            multiplier = volatility_regime_multiplier(
                returns, base, t_idx, float(p["halflife"])
            )
            return base * multiplier[:, None, None]
    raise ValueError(f"unknown family {spec.family}")
