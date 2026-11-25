"""Fixed choices that define the experiment.

Everything here is a decision made once, before any test-period number was
looked at. The sample split in particular is not negotiable after the fact:
hyperparameters are picked on the validation span and the test span is scored
once.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Sample span and split
# ---------------------------------------------------------------------------

#: First date of the daily panel. Chosen because the 49-industry file has a
#: complete Hlth (healthcare) series only from 1969-07-01; starting earlier
#: would mean dropping an industry rather than a handful of early dates.
PANEL_START = np.datetime64("1969-07-01")

#: Last date of the validation span. Hyperparameters are chosen here.
VALIDATION_END = np.datetime64("1999-12-31")

#: First date of the test span. Scored once, with hyperparameters frozen.
TEST_START = np.datetime64("2000-01-01")

# ---------------------------------------------------------------------------
# Walk-forward geometry
# ---------------------------------------------------------------------------

#: Covariance forecasts are refit every 5 trading days ("weekly"). Using a
#: trading-day stride rather than calendar weeks keeps the evaluation grid
#: regular, which is what lets the whole walk-forward be one vmap.
REBALANCE_STRIDE = 5

#: Longest estimation window any estimator may ask for. The first forecast is
#: made only once this many trading days of history exist, so every estimator
#: sees the same set of forecast dates and the comparison is like-for-like.
BURN_IN = 756  # three years of trading days

#: Forecast horizons in trading days, with the evaluation stride (in units of
#: rebalance periods) that makes consecutive observations non-overlapping.
#: Non-overlap matters: the 1 +- sqrt(2/T) band on the bias statistic assumes
#: independent observations, and overlapping windows would make it far too
#: narrow.
HORIZONS: dict[str, dict[str, int]] = {
    # One day. Each day is scored by the forecast made at the most recent
    # rebalance (so up to four days stale, as a weekly-refit model would be).
    "1d": {"days": 1, "rebalance_stride": 1},
    # One week.
    "1w": {"days": 5, "rebalance_stride": 1},
    # One month, taken as four rebalance periods so the windows tile exactly.
    "1m": {"days": 20, "rebalance_stride": 4},
}

HEADLINE_HORIZONS = ("1w", "1m")

# ---------------------------------------------------------------------------
# Asset sets
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AssetSet:
    key: str
    source_key: str
    label: str
    #: Section title inside the French file. Both portfolio files stack
    #: value-weighted then equal-weighted returns; we want value-weighted.
    section_title: str = "Average Value Weighted Returns"
    #: Drop a series if it is missing on more than this fraction of the span.
    max_missing_fraction: float = 0.01


ASSET_SETS: dict[str, AssetSet] = {
    "industry49": AssetSet(
        key="industry49",
        source_key="industry49",
        label="49 industry portfolios",
    ),
    "size_bm_25": AssetSet(
        key="size_bm_25",
        source_key="size_bm_25",
        label="25 size / book-to-market portfolios",
    ),
    # Used only by `riskcheck bench`, to push N up. Five of the hundred corner
    # portfolios are empty on a large share of days (BIG HiBM is missing on 23%
    # of test-period days), which is why the headline evaluation uses the 25.
    "size_bm_100": AssetSet(
        key="size_bm_100",
        source_key="size_bm_100",
        label="100 size / book-to-market portfolios",
        max_missing_fraction=0.005,
    ),
}

#: Asset sets used by `validate` and `evaluate`.
MAIN_ASSET_SETS = ("industry49", "size_bm_25")

# ---------------------------------------------------------------------------
# Factors
# ---------------------------------------------------------------------------

FACTOR_NAMES: tuple[str, ...] = ("Mkt-RF", "SMB", "HML", "RMW", "CMA", "Mom")

# ---------------------------------------------------------------------------
# Portfolio sets used for the bias statistic
# ---------------------------------------------------------------------------

#: Random weights are drawn once from this seed and then held fixed, so the
#: same portfolios are scored for every estimator and every date.
PORTFOLIO_SEED = 20260101
N_RANDOM_PORTFOLIOS = 2000

# ---------------------------------------------------------------------------
# Evaluation detail
# ---------------------------------------------------------------------------

#: Rolling window, in observations, for the time series of the bias statistic.
ROLLING_BIAS_WINDOW = 52

#: Stress windows, reported separately. March 2020 is deliberately included
#: even though it is only a handful of weekly observations; the observation
#: count is reported alongside so the noise is visible.
STRESS_WINDOWS: tuple[tuple[str, str, str], ...] = (
    ("GFC 2008-2009", "2008-01-01", "2009-12-31"),
    ("COVID crash (Mar 2020)", "2020-03-01", "2020-03-31"),
    ("COVID crash (wide)", "2020-02-15", "2020-04-30"),
    ("2022 drawdown", "2022-01-01", "2022-12-31"),
)

#: Block bootstrap for the log-score intervals. Returns are strongly
#: autocorrelated in volatility, so resampling single observations would
#: understate the sampling error badly.
BOOTSTRAP_DRAWS = 2000
BOOTSTRAP_BLOCK = 52  # observations, i.e. about a year at the weekly horizon
BOOTSTRAP_SEED = 7

#: Floor applied to specific variances in the factor models, as a fraction of
#: the cross-sectional mean specific variance, to keep matrices invertible.
SPECIFIC_VARIANCE_FLOOR = 1e-4

TRADING_DAYS_PER_YEAR = 252


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Paths:
    root: Path = field(default_factory=lambda: Path.cwd())

    @property
    def raw(self) -> Path:
        return self.root / "data" / "raw"

    @property
    def processed(self) -> Path:
        return self.root / "data" / "processed"

    @property
    def results(self) -> Path:
        return self.root / "results"

    @property
    def hyperparameters(self) -> Path:
        return self.results / "hyperparameters.json"
