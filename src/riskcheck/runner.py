"""The test-period run: one pass, with hyperparameters already frozen.

This module is deliberately thin. All the decisions were made in ``config`` and
``validate``; what happens here is a single sweep over the frozen estimator
list, writing every number it computes to ``results/`` so that nothing quoted
elsewhere has to be taken on trust.
"""

from __future__ import annotations

import json
import platform
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import jax
import pandas as pd

from . import __version__, config, evaluate, portfolios, using_x64
from .build import BuiltPanel
from .estimators import Spec
from .validate import panel_context


@dataclass
class TestRun:
    results: list[evaluate.SpecResult]
    rolling: pd.DataFrame
    metadata: dict

    @property
    def table(self) -> pd.DataFrame:
        return pd.DataFrame([r.to_row() for r in self.results])

    @property
    def paired_log_score_table(self) -> pd.DataFrame:
        """Log-score differences against the best model, with paired intervals.

        Every estimator is scored on the same realised returns, so the level of
        the log score moves together across models and its own interval says
        little about which model is better. Differencing against a common
        baseline removes that shared variation.
        """
        rows = []
        groups: dict[tuple[str, str], list[evaluate.SpecResult]] = {}
        for result in self.results:
            if result.log_score_series is not None:
                groups.setdefault((result.asset_set, result.horizon), []).append(result)

        for (asset_set, horizon), members in groups.items():
            best = min(members, key=lambda r: r.log_score_mean)
            for result in members:
                comparison = evaluate.paired_log_score_difference(
                    result.log_score_series, best.log_score_series
                )
                rows.append(
                    {
                        "asset_set": asset_set,
                        "horizon": horizon,
                        "estimator": result.short_label,
                        "baseline": best.short_label,
                        "log_score": result.log_score_mean,
                        "log_score_lo": result.log_score_ci[0],
                        "log_score_hi": result.log_score_ci[1],
                        **comparison,
                    }
                )
        return pd.DataFrame(rows)

    @property
    def stress_table(self) -> pd.DataFrame:
        rows = []
        for result in self.results:
            for window, entry in result.stress.items():
                rows.append(
                    {
                        "estimator": result.short_label,
                        "asset_set": result.asset_set,
                        "horizon": result.horizon,
                        "window": window,
                        **entry,
                    }
                )
        return pd.DataFrame(rows)


def run_test_evaluation(
    panels: dict[str, BuiltPanel],
    specs: list[Spec],
    horizons: tuple[str, ...] = tuple(config.HORIZONS),
    asset_sets: tuple[str, ...] = config.MAIN_ASSET_SETS,
    verbose: bool = True,
) -> TestRun:
    """Score every frozen estimator on the test span."""
    results: list[evaluate.SpecResult] = []
    rolling_rows: list[dict] = []
    timings: dict[str, float] = {}
    # The realised window of every scored observation, recorded so the split
    # can be audited from results/ alone: against the validation span in
    # hyperparameters.json, these dates are what show the two do not overlap.
    scored_windows: dict[str, dict[str, dict]] = {}
    started = time.perf_counter()

    for asset_set in asset_sets:
        panel = panels[asset_set]
        contexts = {h: panel_context(panel, h, "test") for h in horizons}
        scored_windows[asset_set] = {
            h: {
                "first_observation_ends": str(ctx.observations.end_date[0]),
                "last_observation_ends": str(ctx.observations.end_date[-1]),
                "n_observations": len(ctx.observations),
            }
            for h, ctx in contexts.items()
        }
        weight_sets = portfolios.standard_portfolio_sets(panel.n_assets)
        if verbose:
            first = contexts[horizons[0]].observations
            print(
                f"  {asset_set}: N={panel.n_assets}, "
                f"test observations {first.end_date[0]}..{first.end_date[-1]}"
            )

        for spec in specs:
            spec_started = time.perf_counter()
            context = contexts[horizons[0]]
            bundle = evaluate.build_forecasts(
                spec, context.returns, context.factors, context.t_idx, weight_sets
            )
            for horizon in horizons:
                obs = contexts[horizon].observations
                result = evaluate.evaluate_bundle(bundle, obs, weight_sets, asset_set)
                results.append(result)
                summary = result.min_variance.bias
                for date, value in zip(summary.rolling_dates, summary.rolling_mean, strict=True):
                    rolling_rows.append(
                        {
                            "estimator": result.short_label,
                            "asset_set": asset_set,
                            "horizon": horizon,
                            "date": date,
                            "min_variance_bias": value,
                        }
                    )
                for name, bias in result.bias.items():
                    for date, value in zip(bias.rolling_dates, bias.rolling_mean, strict=True):
                        rolling_rows.append(
                            {
                                "estimator": result.short_label,
                                "asset_set": asset_set,
                                "horizon": horizon,
                                "date": date,
                                f"{name}_bias": value,
                            }
                        )
            elapsed = time.perf_counter() - spec_started
            timings[f"{asset_set}/{spec.key}"] = elapsed
            if verbose:
                print(f"    {spec.short_label:24s} {elapsed:6.2f}s")

    rolling = pd.DataFrame(rolling_rows)
    if not rolling.empty:
        value_columns = [c for c in rolling.columns if c.endswith("_bias")]
        rolling = (
            rolling.groupby(["estimator", "asset_set", "horizon", "date"], as_index=False)[
                value_columns
            ]
            .first()
            .sort_values(["asset_set", "horizon", "estimator", "date"])
        )

    metadata = {
        "riskcheck_version": __version__,
        "run_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "wall_seconds": round(time.perf_counter() - started, 2),
        "per_spec_seconds": {k: round(v, 3) for k, v in timings.items()},
        "platform": platform.platform(),
        "processor": platform.processor() or platform.machine(),
        "python": platform.python_version(),
        "jax": jax.__version__,
        "jax_backend": jax.default_backend(),
        "jax_devices": [str(d) for d in jax.devices()],
        "float64": using_x64(),
        "test_span": {
            "start": str(config.TEST_START),
            # Hyperparameters were chosen on observations ending at or before
            # this date; nothing here may end on or before it.
            "selection_ended": str(config.VALIDATION_END),
            "horizons": {
                h: {
                    "days": config.HORIZONS[h]["days"],
                    "rebalance_stride": config.HORIZONS[h]["rebalance_stride"],
                }
                for h in horizons
            },
            "scored_windows": scored_windows,
        },
        "rebalance_stride_days": config.REBALANCE_STRIDE,
        "burn_in_days": config.BURN_IN,
        "n_random_portfolios": config.N_RANDOM_PORTFOLIOS,
        "portfolio_seed": config.PORTFOLIO_SEED,
        "rolling_bias_window": config.ROLLING_BIAS_WINDOW,
        "bootstrap": {
            "draws": config.BOOTSTRAP_DRAWS,
            "block": config.BOOTSTRAP_BLOCK,
            "seed": config.BOOTSTRAP_SEED,
        },
        "panels": {
            key: {
                "label": panels[key].label,
                "n_assets": panels[key].n_assets,
                "n_dates": int(panels[key].dates.shape[0]),
                "first_date": str(panels[key].dates[0]),
                "last_date": str(panels[key].dates[-1]),
                "dropped_series": panels[key].drops.dropped_series,
                "n_dropped_dates": panels[key].drops.n_dropped_dates,
                "dropped_dates": panels[key].drops.dropped_dates,
            }
            for key in asset_sets
        },
        "specs": [s.to_json() | {"label": s.label} for s in specs],
    }

    return TestRun(results=results, rolling=rolling, metadata=metadata)


def write_results(run: TestRun, results_dir: Path, raw_dir: Path | None = None) -> None:
    """Write every table, curve and provenance record the README draws on."""
    results_dir.mkdir(parents=True, exist_ok=True)

    run.table.to_csv(results_dir / "metrics.csv", index=False)
    run.stress_table.to_csv(results_dir / "stress.csv", index=False)
    run.rolling.to_csv(results_dir / "rolling_bias.csv", index=False)
    run.paired_log_score_table.to_csv(results_dir / "log_score_paired.csv", index=False)

    detail = [evaluate.result_to_json(r) for r in run.results]
    (results_dir / "metrics.json").write_text(json.dumps(detail, indent=2) + "\n")

    metadata = dict(run.metadata)
    if raw_dir is not None:
        manifest_path = raw_dir / "manifest.json"
        if manifest_path.exists():
            metadata["data_manifest"] = json.loads(manifest_path.read_text())
    (results_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")


def load_metrics(results_dir: Path) -> pd.DataFrame:
    path = results_dir / "metrics.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; run `riskcheck evaluate` first")
    return pd.read_csv(path)
