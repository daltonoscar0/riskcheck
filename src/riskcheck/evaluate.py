"""Walk-forward scoring of covariance forecasts.

The headline metric is the **bias statistic**. Standardise each realised
portfolio return by the volatility the model forecast for it,

    z_t = w' r_{t+1..t+h} / sqrt(h * w' S_t w),

and take the root mean square of ``z`` over an evaluation window. A model whose
volatility forecasts are right gives 1; above 1 means it under-forecast risk,
below 1 means it over-forecast. For ``T`` independent observations the sampling
standard error is about ``1/sqrt(2T)``, giving the conventional band
``1 +- sqrt(2/T)`` used here.

That band is only honest if the observations do not overlap, which is why the
evaluation grid is built the way it is. Forecasts are refit every five trading
days; the one-week horizon scores each refit over the following five days, and
the one-month horizon scores every fourth refit over the following twenty, so
consecutive observations tile the calendar instead of sliding along it. Scoring
a twenty-day horizon weekly would reuse sixteen of every twenty days and make
the band roughly twice as tight as it should be.

A daily horizon is also reported. It is not one of the two horizons the project
set out to compare, but it is what makes the stress windows readable: March 2020
contains four non-overlapping weekly observations and twenty-two daily ones, and
a bias statistic on four observations says almost nothing. The daily numbers use
the forecast from the most recent refit, so they are up to four days stale --
exactly as a weekly-refit production model would be.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from . import config, estimators, portfolios
from .estimators import Spec

# ---------------------------------------------------------------------------
# Non-overlapping observations on the rebalance grid
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Observations:
    """Realised horizon returns, each tied to the forecast that must explain it."""

    horizon: str
    days: int
    returns: np.ndarray  # (n_obs, N) summed over the horizon
    forecast_index: np.ndarray  # (n_obs,) row of the covariance stack
    end_date: np.ndarray  # (n_obs,) datetime64[D], last day of the realised period

    def __len__(self) -> int:
        return self.returns.shape[0]

    def select(self, mask: np.ndarray) -> Observations:
        return Observations(
            horizon=self.horizon,
            days=self.days,
            returns=self.returns[mask],
            forecast_index=self.forecast_index[mask],
            end_date=self.end_date[mask],
        )

    def between(self, start: np.datetime64 | None, end: np.datetime64 | None) -> Observations:
        mask = np.ones(len(self), dtype=bool)
        if start is not None:
            mask &= self.end_date >= start
        if end is not None:
            mask &= self.end_date <= end
        return self.select(mask)


def build_observations(
    returns: np.ndarray, dates: np.ndarray, t_idx: np.ndarray, horizon: str
) -> Observations:
    """Tile the post-burn-in sample into non-overlapping horizon observations."""
    spec = config.HORIZONS[horizon]
    days = spec["days"]
    group = spec["rebalance_stride"]
    stride = int(t_idx[1] - t_idx[0])
    n_assets = returns.shape[1]
    first = int(t_idx[0])
    n_blocks = len(t_idx) - 1

    # One block per forecast: the `stride` days that forecast is responsible for.
    blocks = returns[first + 1 : first + 1 + n_blocks * stride].reshape(
        n_blocks, stride, n_assets
    )

    if days == 1:
        obs_returns = blocks.reshape(-1, n_assets)
        forecast_index = np.repeat(np.arange(n_blocks), stride)
        end_row = first + 1 + np.arange(n_blocks * stride)
    else:
        if days != group * stride:
            raise ValueError(
                f"horizon {horizon!r} wants {days} days but {group} rebalance "
                f"periods of {stride} days is {group * stride}; the grid would not tile"
            )
        n_groups = n_blocks // group
        obs_returns = blocks[: n_groups * group].reshape(n_groups, days, n_assets).sum(axis=1)
        forecast_index = np.arange(n_groups) * group
        end_row = t_idx[forecast_index] + days

    return Observations(
        horizon=horizon,
        days=days,
        returns=obs_returns,
        forecast_index=forecast_index.astype(np.int64),
        end_date=dates[end_row],
    )


# ---------------------------------------------------------------------------
# Bias statistic
# ---------------------------------------------------------------------------


def band_halfwidth(n_obs: int) -> float:
    """Half-width of the conventional ``1 +- sqrt(2/T)`` band."""
    return float(np.sqrt(2.0 / max(n_obs, 1)))


@dataclass
class BiasSummary:
    n_obs: int
    n_portfolios: int
    mean: float
    median: float
    p05: float
    p95: float
    band: float
    share_in_band: float
    rolling_window: int
    rolling_dates: list[str] = field(default_factory=list)
    rolling_mean: list[float] = field(default_factory=list)

    def to_row(self) -> dict:
        return {
            "bias_mean": self.mean,
            "bias_median": self.median,
            "bias_p05": self.p05,
            "bias_p95": self.p95,
            "bias_band": self.band,
            "share_in_band": self.share_in_band,
        }


def _rolling_mean(values: np.ndarray, window: int) -> np.ndarray:
    """Rolling mean down axis 0, via cumulative sums."""
    cumulative = np.cumsum(values, axis=0)
    head = cumulative[window - 1 :]
    tail = np.concatenate([np.zeros((1, *values.shape[1:])), cumulative[:-window]], axis=0)
    return (head - tail) / window


def bias_summary(
    realized: np.ndarray,
    forecast_variance: np.ndarray,
    end_date: np.ndarray,
    rolling_window: int = config.ROLLING_BIAS_WINDOW,
) -> BiasSummary:
    """Bias statistic for a set of portfolios, overall and rolling.

    ``realized`` and ``forecast_variance`` are both ``(n_obs, n_portfolios)``.
    The overall statistic is per portfolio; the rolling statistic is computed
    per portfolio and then averaged across them, and ``share_in_band`` is the
    fraction of all (window, portfolio) pairs that land inside the band.
    """
    squared = realized**2 / forecast_variance
    n_obs, n_portfolios = squared.shape
    overall = np.sqrt(squared.mean(axis=0))

    window = min(rolling_window, n_obs)
    half = band_halfwidth(window)
    if n_obs >= window:
        rolling = np.sqrt(_rolling_mean(squared, window))
        share = float(np.mean(np.abs(rolling - 1.0) <= half))
        rolling_dates = [str(d) for d in end_date[window - 1 :]]
        rolling_mean = [float(v) for v in rolling.mean(axis=1)]
    else:
        share = float("nan")
        rolling_dates, rolling_mean = [], []

    return BiasSummary(
        n_obs=n_obs,
        n_portfolios=n_portfolios,
        mean=float(overall.mean()),
        median=float(np.median(overall)),
        p05=float(np.quantile(overall, 0.05)),
        p95=float(np.quantile(overall, 0.95)),
        band=band_halfwidth(n_obs),
        share_in_band=share,
        rolling_window=window,
        rolling_dates=rolling_dates,
        rolling_mean=rolling_mean,
    )


def bias_summary_chunked(
    obs: Observations,
    weights: np.ndarray,
    forecast_variance_by_date: jnp.ndarray,
    rolling_window: int = config.ROLLING_BIAS_WINDOW,
    chunk: int = 500,
) -> BiasSummary:
    """``bias_summary`` for a large fixed weight set, chunked over portfolios.

    With a few thousand portfolios and several thousand daily observations the
    standardised matrix alone is hundreds of megabytes, so portfolios are
    processed in chunks and the summaries are recombined.
    """
    variance_by_date = np.asarray(forecast_variance_by_date)
    n_portfolios = weights.shape[0]
    scale = float(obs.days)

    overall_parts: list[np.ndarray] = []
    rolling_sum: np.ndarray | None = None
    in_band_count = 0
    in_band_total = 0
    window = min(rolling_window, len(obs))
    half = band_halfwidth(window)

    for lo in range(0, n_portfolios, chunk):
        sl = slice(lo, lo + chunk)
        realized = obs.returns @ weights[sl].T
        forecast_variance = scale * variance_by_date[obs.forecast_index][:, sl]
        squared = realized**2 / forecast_variance
        overall_parts.append(np.sqrt(squared.mean(axis=0)))
        if len(obs) >= window:
            rolling = np.sqrt(_rolling_mean(squared, window))
            contribution = rolling.sum(axis=1)
            rolling_sum = contribution if rolling_sum is None else rolling_sum + contribution
            in_band_count += int(np.sum(np.abs(rolling - 1.0) <= half))
            in_band_total += rolling.size

    overall = np.concatenate(overall_parts)
    if rolling_sum is not None:
        rolling_mean = [float(v) for v in rolling_sum / n_portfolios]
        rolling_dates = [str(d) for d in obs.end_date[window - 1 :]]
        share = in_band_count / in_band_total
    else:
        rolling_mean, rolling_dates, share = [], [], float("nan")

    return BiasSummary(
        n_obs=len(obs),
        n_portfolios=n_portfolios,
        mean=float(overall.mean()),
        median=float(np.median(overall)),
        p05=float(np.quantile(overall, 0.05)),
        p95=float(np.quantile(overall, 0.95)),
        band=band_halfwidth(len(obs)),
        share_in_band=share,
        rolling_window=window,
        rolling_dates=rolling_dates,
        rolling_mean=rolling_mean,
    )


# ---------------------------------------------------------------------------
# Gaussian log score (a proper scoring rule) and its block bootstrap
# ---------------------------------------------------------------------------


@jax.jit
def _log_score_pieces(covariances: jnp.ndarray, scale: float) -> tuple[jnp.ndarray, jnp.ndarray]:
    chol = jnp.linalg.cholesky(scale * covariances)
    log_determinant = 2.0 * jnp.sum(jnp.log(jnp.diagonal(chol, axis1=1, axis2=2)), axis=1)
    return chol, log_determinant


@jax.jit
def _mahalanobis(chol: jnp.ndarray, residual: jnp.ndarray) -> jnp.ndarray:
    solved = jax.scipy.linalg.solve_triangular(chol, residual[:, :, None], lower=True)
    return jnp.sum(solved[:, :, 0] ** 2, axis=1)


def log_score_per_observation(
    obs: Observations, covariances: jnp.ndarray, chunk: int = 1024
) -> np.ndarray:
    """Gaussian negative log-likelihood per observation, per asset.

    ``log det(h S) + y' (h S)^-1 y``, divided by the number of assets so the
    two asset sets are on a comparable scale. This is a strictly proper scoring
    rule for a covariance forecast: it is minimised in expectation only by the
    true conditional covariance, so unlike the bias statistic it cannot be
    gamed by inflating the whole matrix.
    """
    chol, log_determinant = _log_score_pieces(covariances, float(obs.days))
    n_assets = obs.returns.shape[1]
    scores = np.empty(len(obs), dtype=np.float64)
    returns = jnp.asarray(obs.returns)
    index = jnp.asarray(obs.forecast_index)
    for lo in range(0, len(obs), chunk):
        sl = slice(lo, lo + chunk)
        rows = index[sl]
        quadratic = _mahalanobis(chol[rows], returns[sl])
        scores[sl] = np.asarray(log_determinant[rows] + quadratic)
    return scores / n_assets


def block_bootstrap_ci(
    series: np.ndarray,
    block: int = config.BOOTSTRAP_BLOCK,
    draws: int = config.BOOTSTRAP_DRAWS,
    seed: int = config.BOOTSTRAP_SEED,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """Percentile interval for the mean of an autocorrelated series.

    Volatility is strongly persistent, so resampling individual observations
    would treat a run of crisis weeks as independent draws and give an interval
    several times too narrow. This resamples contiguous blocks of about a year.
    """
    n = series.shape[0]
    if n == 0:
        return float("nan"), float("nan")
    block = min(block, n)
    n_blocks = int(np.ceil(n / block))
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, n - block + 1, size=(draws, n_blocks))
    offsets = np.arange(block)
    index = (starts[:, :, None] + offsets[None, None, :]).reshape(draws, -1)[:, :n]
    means = series[index].mean(axis=1)
    return float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


def paired_log_score_difference(
    series: np.ndarray,
    baseline: np.ndarray,
    block: int = config.BOOTSTRAP_BLOCK,
    draws: int = config.BOOTSTRAP_DRAWS,
    seed: int = config.BOOTSTRAP_SEED,
    alpha: float = 0.05,
) -> dict:
    """Block-bootstrap the *difference* in log score against a baseline model.

    Comparing two models by whether their separate confidence intervals
    overlap is the wrong test, and badly so here: every estimator is scored on
    the same realised returns, so most of the sampling variation in the level
    is a common shock that cancels in the difference. On this data the interval
    around each model's own log score is about 0.45 wide while the interval
    around the difference between two models is an order of magnitude tighter.

    Positive means worse than the baseline, since the score is a loss.
    """
    difference = series - baseline
    low, high = block_bootstrap_ci(difference, block=block, draws=draws, seed=seed, alpha=alpha)
    return {
        "difference": float(difference.mean()),
        "difference_lo": low,
        "difference_hi": high,
        "worse_than_baseline": bool(low > 0.0),
    }


# ---------------------------------------------------------------------------
# Minimum-variance diagnostics
# ---------------------------------------------------------------------------


@dataclass
class MinVarianceSummary:
    realized_vol_annual: float
    forecast_vol_annual: float
    gross_leverage: float
    bias: BiasSummary

    def to_row(self) -> dict:
        row = {
            "mv_realized_vol_annual": self.realized_vol_annual,
            "mv_forecast_vol_annual": self.forecast_vol_annual,
            "mv_gross_leverage": self.gross_leverage,
        }
        row.update({f"mv_{k}": v for k, v in self.bias.to_row().items()})
        return row


def minimum_variance_summary(
    obs: Observations,
    weights: jnp.ndarray,
    forecast_variance: jnp.ndarray,
    rolling_window: int = config.ROLLING_BIAS_WINDOW,
) -> MinVarianceSummary:
    """Score each model on the minimum-variance portfolio it chose for itself."""
    weights_np = np.asarray(weights)
    variance_np = np.asarray(forecast_variance)
    scale = float(obs.days)

    chosen = weights_np[obs.forecast_index]
    realized = np.sum(obs.returns * chosen, axis=1)
    forecast_var = scale * variance_np[obs.forecast_index]

    summary = bias_summary(
        realized[:, None], forecast_var[:, None], obs.end_date, rolling_window
    )
    annualise = np.sqrt(config.TRADING_DAYS_PER_YEAR / obs.days)
    # A narrow stress window can hold a single observation (March 2020 at the
    # monthly horizon), where a sample standard deviation is undefined.
    realized_vol = realized.std(ddof=1) if realized.size > 1 else np.nan
    return MinVarianceSummary(
        realized_vol_annual=float(realized_vol * annualise),
        forecast_vol_annual=float(np.sqrt(forecast_var).mean() * annualise),
        gross_leverage=float(np.abs(weights_np).sum(axis=1).mean()),
        bias=summary,
    )


# ---------------------------------------------------------------------------
# Putting one estimator through the whole evaluation
# ---------------------------------------------------------------------------


@dataclass
class SpecResult:
    spec_key: str
    spec_label: str
    short_label: str
    asset_set: str
    horizon: str
    n_obs: int
    n_assets: int
    bias: dict[str, BiasSummary]
    min_variance: MinVarianceSummary
    log_score_mean: float
    log_score_ci: tuple[float, float]
    stress: dict[str, dict] = field(default_factory=dict)
    #: Per-observation scores, kept so the runner can bootstrap *paired*
    #: differences between estimators. Never written to JSON.
    log_score_series: np.ndarray | None = field(default=None, repr=False)

    def to_row(self) -> dict:
        row: dict[str, Any] = {
            "estimator": self.short_label,
            "estimator_detail": self.spec_label,
            "spec_key": self.spec_key,
            "asset_set": self.asset_set,
            "horizon": self.horizon,
            "n_obs": self.n_obs,
            "n_assets": self.n_assets,
            "log_score": self.log_score_mean,
            "log_score_lo": self.log_score_ci[0],
            "log_score_hi": self.log_score_ci[1],
        }
        for name, summary in self.bias.items():
            row.update({f"{name}_{k}": v for k, v in summary.to_row().items()})
        row.update(self.min_variance.to_row())
        return row


@dataclass
class ForecastBundle:
    """A spec's forecasts plus everything derived from them that horizons share."""

    spec: Spec
    covariances: jnp.ndarray
    forecast_variance: dict[str, jnp.ndarray]  # portfolio set -> (D, P)
    mv_weights: jnp.ndarray  # (D, N)
    mv_variance: jnp.ndarray  # (D,)


def build_forecasts(
    spec: Spec,
    returns: jnp.ndarray,
    factors: jnp.ndarray,
    t_idx: np.ndarray,
    weight_sets: dict[str, np.ndarray],
) -> ForecastBundle:
    """Run one estimator over the whole grid and precompute per-date quantities.

    The quadratic forms ``w' S_t w`` depend on the date and not on the horizon,
    so they are computed once here and reused by every horizon.
    """
    covariances = estimators.forecast(spec, returns, factors, t_idx)
    return ForecastBundle(
        spec=spec,
        covariances=covariances,
        forecast_variance={
            name: portfolios.quadratic_forms(weights, covariances)
            for name, weights in weight_sets.items()
        },
        mv_weights=portfolios.minimum_variance_weights(covariances),
        mv_variance=portfolios.minimum_variance_forecast_variance(covariances),
    )


def evaluate_bundle(
    bundle: ForecastBundle,
    obs: Observations,
    weight_sets: dict[str, np.ndarray],
    asset_set: str,
    stress_windows: tuple[tuple[str, str, str], ...] = config.STRESS_WINDOWS,
) -> SpecResult:
    """Score one estimator at one horizon over the observations supplied."""
    bias = {
        name: bias_summary_chunked(obs, weights, bundle.forecast_variance[name])
        for name, weights in weight_sets.items()
    }
    min_variance = minimum_variance_summary(obs, bundle.mv_weights, bundle.mv_variance)
    scores = log_score_per_observation(obs, bundle.covariances)
    low, high = block_bootstrap_ci(scores)

    stress: dict[str, dict] = {}
    for name, start, end in stress_windows:
        window = obs.between(np.datetime64(start), np.datetime64(end))
        if len(window) == 0:
            continue
        window_mv = minimum_variance_summary(
            window, bundle.mv_weights, bundle.mv_variance, rolling_window=len(window)
        )
        entry = {
            "n_obs": len(window),
            "band": band_halfwidth(len(window)),
            "mv_bias": window_mv.bias.mean,
            "mv_realized_vol_annual": window_mv.realized_vol_annual,
            "mv_forecast_vol_annual": window_mv.forecast_vol_annual,
            "log_score": float(np.mean(log_score_per_observation(window, bundle.covariances))),
        }
        for pset, weights in weight_sets.items():
            entry[f"{pset}_bias"] = bias_summary_chunked(
                window, weights, bundle.forecast_variance[pset], rolling_window=len(window)
            ).mean
        stress[name] = entry

    return SpecResult(
        spec_key=bundle.spec.key,
        spec_label=bundle.spec.label,
        short_label=bundle.spec.short_label,
        asset_set=asset_set,
        horizon=obs.horizon,
        n_obs=len(obs),
        n_assets=obs.returns.shape[1],
        bias=bias,
        min_variance=min_variance,
        log_score_mean=float(scores.mean()),
        log_score_ci=(low, high),
        stress=stress,
        log_score_series=scores,
    )


def result_to_json(result: SpecResult) -> dict:
    blob = asdict(result)
    blob.pop("log_score_series", None)
    # The rolling curves are written to their own file; keep the JSON readable.
    for summary in blob["bias"].values():
        summary.pop("rolling_dates", None)
        summary.pop("rolling_mean", None)
    blob["min_variance"]["bias"].pop("rolling_dates", None)
    blob["min_variance"]["bias"].pop("rolling_mean", None)
    return blob
