"""Markdown tables built from the saved test-period results.

Nothing here computes a statistic. Every number is read back out of
``results/metrics.csv``, which is what makes the README reproducible: change
the evaluation and the tables change with it, and a number that is not in the
CSV cannot appear in the write-up.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from . import config

#: Report order, matching the order estimators are introduced in the README.
FAMILY_ORDER = ("Sample", "EWMA", "Ledoit-Wolf", "PCA factor", "FF5+Mom factor")

HEADLINE_ASSET_SET = "industry49"
HEADLINE_HORIZON = "1w"


def _markdown_table(headers: list[str], rows: list[list[str]], align: str = "r") -> str:
    """Render a markdown table without pulling in a formatting dependency."""
    right = align == "r"
    separators = [":--" if i == 0 or not right else "--:" for i in range(len(headers))]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(separators) + " |"]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join(lines)


def _estimator_order(frame: pd.DataFrame) -> list[str]:
    """Base families in report order, then anything with a regime adjustment."""
    present = list(dict.fromkeys(frame["estimator"]))
    base = [name for name in FAMILY_ORDER if name in present]
    rest = [name for name in present if name not in base]
    return base + rest


def _fmt(value: float, digits: int = 3) -> str:
    if pd.isna(value):
        return "-"
    return f"{value:.{digits}f}"


def _pct(value: float, digits: int = 2) -> str:
    if pd.isna(value):
        return "-"
    return f"{100 * value:.{digits}f}%"


def headline_table(
    metrics: pd.DataFrame,
    asset_set: str = HEADLINE_ASSET_SET,
    horizon: str = HEADLINE_HORIZON,
) -> tuple[str, dict]:
    """The first-screen table, plus the facts needed to caption it."""
    frame = metrics[(metrics.asset_set == asset_set) & (metrics.horizon == horizon)]
    if frame.empty:
        raise ValueError(f"no rows for asset_set={asset_set!r}, horizon={horizon!r}")

    headers = [
        "Estimator",
        "Random long-only",
        "Random long-short",
        "Min-variance",
        "Min-var in band",
        "Min-var vol (ann.)",
        "Log score",
    ]
    rows = []
    for name in _estimator_order(frame):
        row = frame[frame.estimator == name].iloc[0]
        rows.append(
            [
                name,
                _fmt(row.long_only_bias_mean),
                _fmt(row.long_short_bias_mean),
                f"**{_fmt(row.mv_bias_mean)}**",
                _pct(row.mv_share_in_band, 0),
                _pct(row.mv_realized_vol_annual),
                _fmt(row.log_score, 3),
            ]
        )

    reference = frame.iloc[0]
    caption = {
        "asset_set": asset_set,
        "n_assets": int(reference.n_assets),
        "horizon": horizon,
        "n_obs": int(reference.n_obs),
        "band": float(reference.mv_bias_band),
        "rolling_window": config.ROLLING_BIAS_WINDOW,
        "rolling_band": float((2.0 / config.ROLLING_BIAS_WINDOW) ** 0.5),
    }
    return _markdown_table(headers, rows), caption


def breakdown_table(metrics: pd.DataFrame, horizons: tuple[str, ...] | None = None) -> str:
    """Minimum-variance results split by asset set and horizon."""
    horizons = horizons or config.HEADLINE_HORIZONS
    frame = metrics[metrics.horizon.isin(horizons)]
    headers = [
        "Estimator",
        "Asset set",
        "Horizon",
        "Obs",
        "Min-var bias",
        "Band",
        "In band",
        "Min-var vol (ann.)",
        "Forecast vol (ann.)",
        "Gross leverage",
        "Log score",
    ]
    rows = []
    for asset_set in config.MAIN_ASSET_SETS:
        for horizon in horizons:
            block = frame[(frame.asset_set == asset_set) & (frame.horizon == horizon)]
            for name in _estimator_order(block):
                row = block[block.estimator == name].iloc[0]
                rows.append(
                    [
                        name,
                        f"{int(row.n_assets)} assets",
                        horizon,
                        str(int(row.n_obs)),
                        _fmt(row.mv_bias_mean),
                        f"±{_fmt(row.mv_bias_band)}",
                        _pct(row.mv_share_in_band, 0),
                        _pct(row.mv_realized_vol_annual),
                        _pct(row.mv_forecast_vol_annual),
                        _fmt(row.mv_gross_leverage, 2),
                        _fmt(row.log_score, 3),
                    ]
                )
    return _markdown_table(headers, rows)


def random_portfolio_table(metrics: pd.DataFrame) -> str:
    """Random-portfolio bias statistics, where the models mostly look fine."""
    headers = [
        "Estimator",
        "Asset set",
        "Horizon",
        "Long-only bias",
        "In band",
        "Long-short bias",
        "In band",
    ]
    rows = []
    for asset_set in config.MAIN_ASSET_SETS:
        for horizon in config.HEADLINE_HORIZONS:
            block = metrics[(metrics.asset_set == asset_set) & (metrics.horizon == horizon)]
            for name in _estimator_order(block):
                row = block[block.estimator == name].iloc[0]
                rows.append(
                    [
                        name,
                        f"{int(row.n_assets)} assets",
                        horizon,
                        _fmt(row.long_only_bias_mean),
                        _pct(row.long_only_share_in_band, 0),
                        _fmt(row.long_short_bias_mean),
                        _pct(row.long_short_share_in_band, 0),
                    ]
                )
    return _markdown_table(headers, rows)


def stress_table(
    stress: pd.DataFrame, asset_set: str = HEADLINE_ASSET_SET, horizon: str = "1d"
) -> str:
    """Minimum-variance bias statistic inside each stress window.

    Defaults to the daily horizon: March 2020 holds four non-overlapping weekly
    observations but twenty-two daily ones, and a bias statistic on four
    observations is not worth printing.
    """
    frame = stress[(stress.asset_set == asset_set) & (stress.horizon == horizon)]
    if frame.empty:
        return "_No stress-window observations at this horizon._"

    windows = [name for name, _, _ in config.STRESS_WINDOWS if name in set(frame.window)]
    headers = ["Estimator"] + [
        f"{name}<br>(n={int(frame[frame.window == name].iloc[0].n_obs)})" for name in windows
    ]
    rows = []
    for name in _estimator_order(frame):
        block = frame[frame.estimator == name]
        cells = [name]
        for window in windows:
            match = block[block.window == window]
            cells.append(_fmt(match.iloc[0].mv_bias) if not match.empty else "-")
        rows.append(cells)
    return _markdown_table(headers, rows)


def hyperparameter_table(blob: dict) -> str:
    """What was chosen on the validation span, and what it beat."""
    headers = ["Family", "Chosen on validation", "Grid searched", "Validation log score"]
    rows = []
    for family, grid in blob["grids"].items():
        best = min(grid, key=lambda r: r["criterion"])
        rows.append(
            [
                family,
                best["label"],
                f"{len(grid)} configurations",
                _fmt(best["criterion"], 4),
            ]
        )
    best_vra = min(blob["vra_grid"], key=lambda r: r["criterion"])
    rows.append(
        [
            "vra",
            best_vra["label"],
            f"{len(blob['vra_grid'])} configurations",
            _fmt(best_vra["criterion"], 4),
        ]
    )
    return _markdown_table(headers, rows)


def render_report(results_dir: Path) -> str:
    """The headline report, as printed by ``riskcheck report``."""
    metrics = pd.read_csv(results_dir / "metrics.csv")
    stress = pd.read_csv(results_dir / "stress.csv")
    metadata = json.loads((results_dir / "run_metadata.json").read_text())
    hyperparameters = json.loads((results_dir / "hyperparameters.json").read_text())

    table, caption = headline_table(metrics)
    panel = metadata["panels"][caption["asset_set"]]

    parts = [
        "## Headline: out-of-sample calibration, 2000 onwards",
        "",
        f"{panel['label']} ({caption['n_assets']} assets), "
        f"{caption['horizon']} horizon, {caption['n_obs']} non-overlapping observations. "
        f"A well-calibrated model gives a bias statistic of 1.00; the band for the full "
        f"test period is ±{caption['band']:.3f}. "
        f"\"In band\" is the share of rolling {caption['rolling_window']}-observation "
        f"windows within ±{caption['rolling_band']:.3f} of 1.",
        "",
        table,
        "",
        "## Minimum variance, by asset set and horizon",
        "",
        breakdown_table(metrics),
        "",
        "## Random portfolios",
        "",
        random_portfolio_table(metrics),
        "",
        "## Stress windows (minimum-variance bias statistic, daily horizon)",
        "",
        stress_table(stress),
        "",
        "## Hyperparameters, chosen on 1972-1999 and frozen",
        "",
        hyperparameter_table(hyperparameters),
        "",
        f"_Run {metadata['run_utc']} on {metadata['processor']}, "
        f"JAX {metadata['jax']} ({metadata['jax_backend']}), "
        f"float64={metadata['float64']}; "
        f"evaluation wall time {metadata['wall_seconds']}s._",
    ]
    return "\n".join(parts)
