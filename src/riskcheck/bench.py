"""Timing the walk-forward, honestly.

The thing being timed is the whole evaluation for one asset set: forecasting
every estimator at every rebalance date, the quadratic forms for all the random
portfolios, the minimum-variance solves, and the log scores. Compilation is
excluded by a warm-up pass, because it is a fixed cost that does not scale with
the problem and would otherwise dominate the small cases.

The interesting question is not "is the GPU faster" but "at what size does it
start to be". At N=49 and a few thousand portfolios this is a sequence of small
matrix operations that a CPU handles at memory speed, and a GPU will mostly be
waiting on kernel launches. Scaling N up via the 100-portfolio file and pushing
the portfolio count up is what moves the work into the regime a GPU wants.
"""

from __future__ import annotations

import platform
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from . import __version__, config, estimators, evaluate, portfolios, using_x64
from .build import BuiltPanel
from .estimators import Spec


def _timed_evaluation(
    panel: BuiltPanel,
    specs: list[Spec],
    weight_sets: dict[str, np.ndarray],
    observations: dict[str, evaluate.Observations],
) -> None:
    """One full pass, identical in shape to what `riskcheck evaluate` does."""
    returns = jnp.asarray(panel.excess_returns)
    factors = jnp.asarray(panel.factors)
    t_idx = estimators.rebalance_indices(panel.dates.shape[0])
    for spec in specs:
        bundle = evaluate.build_forecasts(spec, returns, factors, t_idx, weight_sets)
        for obs in observations.values():
            # Scoring pulls every forecast back into numpy, so there is no
            # deferred JAX work left outstanding when this returns.
            evaluate.evaluate_bundle(bundle, obs, weight_sets, panel.asset_set, stress_windows=())


def run_benchmark(
    panel: BuiltPanel,
    specs: list[Spec],
    n_portfolios: int = config.N_RANDOM_PORTFOLIOS,
    repeats: int = 1,
    horizons: tuple[str, ...] = tuple(config.HORIZONS),
) -> dict:
    """Warm up once, then time ``repeats`` passes and report the best and mean."""
    t_idx = estimators.rebalance_indices(panel.dates.shape[0])
    observations = {
        h: evaluate.build_observations(
            panel.excess_returns, panel.dates, t_idx, h
        ).between(config.TEST_START, None)
        for h in horizons
    }
    weight_sets = {
        "long_only": portfolios.random_long_only(
            panel.n_assets, n_portfolios, config.PORTFOLIO_SEED
        ),
        "long_short": portfolios.random_long_short(
            panel.n_assets, n_portfolios, config.PORTFOLIO_SEED
        ),
    }

    warm_up_start = time.perf_counter()
    _timed_evaluation(panel, specs, weight_sets, observations)
    compile_seconds = time.perf_counter() - warm_up_start

    timings = []
    for _ in range(repeats):
        start = time.perf_counter()
        _timed_evaluation(panel, specs, weight_sets, observations)
        timings.append(time.perf_counter() - start)

    return {
        "riskcheck_version": __version__,
        "asset_set": panel.asset_set,
        "n_assets": panel.n_assets,
        "n_dates": int(panel.dates.shape[0]),
        "n_rebalance_dates": int(len(t_idx)),
        "n_test_observations": {h: len(o) for h, o in observations.items()},
        "n_estimators": len(specs),
        "n_portfolios_per_set": n_portfolios,
        "n_portfolio_sets": len(weight_sets),
        "backend": jax.default_backend(),
        "devices": [str(d) for d in jax.devices()],
        "float64": using_x64(),
        "platform": platform.platform(),
        "processor": platform.processor() or platform.machine(),
        "python": platform.python_version(),
        "jax": jax.__version__,
        "first_pass_with_compile_seconds": round(compile_seconds, 3),
        "repeats": repeats,
        "best_seconds": round(min(timings), 3),
        "mean_seconds": round(float(np.mean(timings)), 3),
    }


def load_records(path: Path) -> list[dict]:
    import json

    return json.loads(path.read_text()) if path.exists() else []
