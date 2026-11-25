"""The rolling bias-statistic figure.

Two stacked panels sharing a time axis, because the single most important thing
to see is a contrast rather than a level: the same six models are close to
calibrated on random long-only portfolios (lower panel, hugging 1) and badly
miscalibrated on the minimum-variance portfolios they build from their own
forecasts (upper panel). A one-panel chart of either alone invites the wrong
conclusion.

Light and dark versions are rendered from the same data so the README can serve
whichever matches the reader's theme.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from . import config  # noqa: E402
from .report import _estimator_order  # noqa: E402

#: Categorical slots in fixed order, validated for colour-vision deficiency on
#: the adjacent pairlist in both modes (worst adjacent CVD dE 9.1 light / 8.4
#: dark). Assigned by estimator, never cycled.
SERIES_LIGHT = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300")
SERIES_DARK = ("#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300")


@dataclass(frozen=True)
class Theme:
    name: str
    surface: str
    text_primary: str
    text_secondary: str
    muted: str
    gridline: str
    baseline: str
    stress: str
    band: str
    series: tuple[str, ...]


LIGHT = Theme(
    name="light",
    surface="#fcfcfb",
    text_primary="#0b0b0b",
    text_secondary="#52514e",
    muted="#898781",
    gridline="#e1e0d9",
    baseline="#c3c2b7",
    stress="#0b0b0b",
    band="#c3c2b7",
    series=SERIES_LIGHT,
)

DARK = Theme(
    name="dark",
    surface="#1a1a19",
    text_primary="#ffffff",
    text_secondary="#c3c2b7",
    muted="#898781",
    gridline="#2c2c2a",
    baseline="#383835",
    stress="#ffffff",
    band="#383835",
    series=SERIES_DARK,
)

#: Only the windows worth shading on a multi-decade axis; the wide COVID window
#: would just double-shade March 2020.
SHADED_WINDOWS = ("GFC 2008-2009", "COVID crash (Mar 2020)", "2022 drawdown")


def _style(theme: Theme) -> dict:
    return {
        "figure.facecolor": theme.surface,
        "axes.facecolor": theme.surface,
        "savefig.facecolor": theme.surface,
        "text.color": theme.text_primary,
        "axes.labelcolor": theme.text_secondary,
        "axes.edgecolor": theme.baseline,
        "xtick.color": theme.muted,
        "ytick.color": theme.muted,
        "xtick.labelcolor": theme.text_secondary,
        "ytick.labelcolor": theme.text_secondary,
        "grid.color": theme.gridline,
        "grid.linewidth": 0.8,
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 9,
        "axes.titlesize": 10,
        "legend.fontsize": 8.5,
        "figure.dpi": 160,
    }


def rolling_bias_figure(
    rolling: pd.DataFrame,
    path_stem: Path,
    asset_set: str = "industry49",
    horizon: str = "1w",
    asset_label: str = "49 industry portfolios",
) -> list[Path]:
    """Render the figure in both themes; returns the paths written."""
    frame = rolling[(rolling.asset_set == asset_set) & (rolling.horizon == horizon)].copy()
    if frame.empty:
        raise ValueError(f"no rolling rows for {asset_set!r} at horizon {horizon!r}")
    frame["date"] = pd.to_datetime(frame["date"])
    order = _estimator_order(frame)

    window = config.ROLLING_BIAS_WINDOW
    tolerance = float(np.sqrt(2.0 / window))

    written = []
    for theme in (LIGHT, DARK):
        colours = {name: theme.series[i % len(theme.series)] for i, name in enumerate(order)}
        with plt.rc_context(_style(theme)):
            figure, axes = plt.subplots(
                2, 1, figsize=(9.2, 6.4), sharex=True, height_ratios=(1.45, 1.0)
            )
            panels = (
                (axes[0], "min_variance_bias", "Minimum-variance portfolios"),
                (axes[1], "long_only_bias", "Random long-only portfolios"),
            )

            for axis, column, title in panels:
                # Tolerance band first, so the data sits on top of it.
                axis.axhspan(
                    1 - tolerance, 1 + tolerance, color=theme.band, alpha=0.35, linewidth=0
                )
                axis.axhline(1.0, color=theme.baseline, linewidth=1.0, zorder=2)

                for name, start, end in config.STRESS_WINDOWS:
                    if name not in SHADED_WINDOWS:
                        continue
                    axis.axvspan(
                        np.datetime64(start),
                        np.datetime64(end),
                        color=theme.stress,
                        alpha=0.055,
                        linewidth=0,
                        zorder=1,
                    )

                for name in order:
                    series = frame[frame.estimator == name].sort_values("date")
                    axis.plot(
                        series["date"],
                        series[column],
                        color=colours[name],
                        linewidth=1.6,
                        label=name,
                        solid_capstyle="round",
                        zorder=3,
                    )

                axis.set_title(title, loc="left", color=theme.text_primary, pad=6)
                axis.grid(axis="y", zorder=0)
                axis.set_axisbelow(True)
                for side in ("top", "right"):
                    axis.spines[side].set_visible(False)
                axis.spines["left"].set_color(theme.baseline)
                axis.spines["bottom"].set_color(theme.baseline)

            axes[0].set_ylabel("Bias statistic")
            axes[1].set_ylabel("Bias statistic")

            # Autoscale, then add headroom on the upper panel for the window
            # labels. Clipping either panel to a tidier range would flatten
            # real excursions -- exactly the ones the chart exists to show.
            for axis in (axes[0], axes[1]):
                axis.margins(y=0.06)
            lower, upper = axes[0].get_ylim()
            axes[0].set_ylim(lower, upper + 0.16 * (upper - lower))

            # Label the shaded windows along the top, alternating the vertical
            # offset: March 2020 is a one-month sliver sitting close enough to
            # 2022 that same-height labels overlap.
            top = axes[0].get_ylim()[1]
            for position, (name, start, end) in enumerate(
                [w for w in config.STRESS_WINDOWS if w[0] in SHADED_WINDOWS]
            ):
                midpoint = np.datetime64(start) + (np.datetime64(end) - np.datetime64(start)) / 2
                axes[0].annotate(
                    name.replace(" (Mar 2020)", " 2020"),
                    xy=(midpoint, top),
                    xytext=(0, -8 - 11 * (position % 2)),
                    textcoords="offset points",
                    ha="center",
                    va="top",
                    fontsize=7.5,
                    color=theme.muted,
                )

            axes[1].xaxis.set_major_locator(mdates.YearLocator(4))
            axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
            axes[1].set_xlabel(None)

            handles, labels = axes[0].get_legend_handles_labels()
            legend = figure.legend(
                handles,
                labels,
                loc="lower center",
                ncol=min(len(order), 3),
                frameon=False,
                bbox_to_anchor=(0.5, -0.015),
            )
            for text in legend.get_texts():
                text.set_color(theme.text_secondary)

            figure.suptitle(
                f"Rolling bias statistic, {window}-observation windows "
                f"({asset_label}, {horizon} horizon)",
                x=0.015,
                ha="left",
                fontsize=11.5,
                color=theme.text_primary,
            )
            figure.text(
                0.015,
                0.925,
                "1.00 means forecast risk matched realised risk; the band is "
                f"±√(2/{window}) = ±{tolerance:.2f}. Shaded columns are stress windows.",
                ha="left",
                fontsize=8.5,
                color=theme.text_secondary,
            )
            figure.tight_layout(rect=(0, 0.045, 1, 0.9))

            out = path_stem.with_name(f"{path_stem.name}_{theme.name}.png")
            out.parent.mkdir(parents=True, exist_ok=True)
            figure.savefig(out, bbox_inches="tight")
            plt.close(figure)
            written.append(out)
    return written
