"""Turn the raw French files into aligned, gap-free daily panels.

The output of this module is the only thing the estimators ever see: a matrix
of excess returns with no NaNs, a matching factor matrix, and a written record
of every series and date that was dropped to get there.

The drop policy is deliberately dumb and stated up front, because anything
cleverer would be a decision made with knowledge of the whole sample:

1. Restrict to the configured span, intersected with the dates on which the
   factors also exist.
2. Drop an asset series if it is missing on more than ``max_missing_fraction``
   of that span.
3. Drop any remaining date on which any surviving series is still missing.

Step 2 does look at the full span, so it is a universe choice made with
hindsight about *data availability* (never about returns). That is why the
headline size/book-to-market set is the 25, where nothing is ever missing and
step 2 does nothing at all.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from . import config
from .fetch import SOURCES_BY_KEY
from .frenchdata import load_panel


@dataclass
class DropRecord:
    """Everything thrown away, and why."""

    span_start: str
    span_end: str
    n_dates_in_span: int
    dropped_series: list[dict]
    dropped_dates: list[str]
    n_dropped_dates: int
    dates_outside_factor_coverage: int


@dataclass
class BuiltPanel:
    asset_set: str
    label: str
    dates: np.ndarray  # datetime64[D] (T,)
    names: tuple[str, ...]
    excess_returns: np.ndarray  # (T, N) decimal, in excess of the daily RF
    total_returns: np.ndarray  # (T, N) decimal, as published
    factors: np.ndarray  # (T, K) decimal
    factor_names: tuple[str, ...]
    rf: np.ndarray  # (T,) decimal daily risk-free rate
    drops: DropRecord

    @property
    def n_assets(self) -> int:
        return len(self.names)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            dates=self.dates,
            names=np.asarray(self.names),
            excess_returns=self.excess_returns,
            total_returns=self.total_returns,
            factors=self.factors,
            factor_names=np.asarray(self.factor_names),
            rf=self.rf,
        )
        meta = {
            "asset_set": self.asset_set,
            "label": self.label,
            "n_dates": int(self.dates.shape[0]),
            "n_assets": self.n_assets,
            "first_date": str(self.dates[0]),
            "last_date": str(self.dates[-1]),
            "names": list(self.names),
            "factor_names": list(self.factor_names),
            "drops": asdict(self.drops),
        }
        path.with_suffix(".json").write_text(json.dumps(meta, indent=2) + "\n")

    @classmethod
    def load(cls, path: Path) -> BuiltPanel:
        with np.load(path, allow_pickle=False) as z:
            meta = json.loads(path.with_suffix(".json").read_text())
            return cls(
                asset_set=meta["asset_set"],
                label=meta["label"],
                dates=z["dates"],
                names=tuple(str(n) for n in z["names"]),
                excess_returns=z["excess_returns"],
                total_returns=z["total_returns"],
                factors=z["factors"],
                factor_names=tuple(str(n) for n in z["factor_names"]),
                rf=z["rf"],
                drops=DropRecord(**meta["drops"]),
            )


def load_factors(raw_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (dates, factors (T,6), rf (T,)) for FF5 + momentum.

    The factor returns are already excess (Mkt-RF) or long-short (the rest), so
    nothing is subtracted from them.
    """
    ff5_src = SOURCES_BY_KEY["ff5"]
    mom_src = SOURCES_BY_KEY["momentum"]
    ff5 = load_panel(raw_dir / ff5_src.zip_name, ff5_src.csv_name)
    mom = load_panel(raw_dir / mom_src.zip_name, mom_src.csv_name)

    for panel, name in ((ff5, "FF5"), (mom, "momentum")):
        if panel.missing.any():
            raise ValueError(
                f"{name} factor file has {int(panel.missing.sum())} sentinel values; "
                "riskcheck assumes the factor series are complete"
            )

    dates, i_ff5, i_mom = np.intersect1d(ff5.dates, mom.dates, return_indices=True)
    ff5_cols = {name: k for k, name in enumerate(ff5.names)}
    mom_col = mom.names.index("Mom")

    columns = []
    for name in config.FACTOR_NAMES:
        if name == "Mom":
            columns.append(mom.returns[i_mom, mom_col])
        else:
            columns.append(ff5.returns[i_ff5, ff5_cols[name]])
    factors = np.column_stack(columns)
    rf = ff5.returns[i_ff5, ff5_cols["RF"]]
    return dates, factors, rf


def build_panel(
    asset_set_key: str,
    raw_dir: Path,
    start: np.datetime64 | None = None,
    end: np.datetime64 | None = None,
) -> BuiltPanel:
    """Build one aligned, gap-free panel."""
    spec = config.ASSET_SETS[asset_set_key]
    source = SOURCES_BY_KEY[spec.source_key]
    assets = load_panel(
        raw_dir / source.zip_name,
        source.csv_name,
        title_contains=spec.section_title,
    )

    factor_dates, factors, rf = load_factors(raw_dir)

    start = config.PANEL_START if start is None else start
    # Step 1: configured span, intersected with factor coverage.
    in_window = assets.dates >= start
    if end is not None:
        in_window &= assets.dates <= end
    n_in_window = int(in_window.sum())

    shared, i_assets, i_factors = np.intersect1d(
        assets.dates[in_window], factor_dates, return_indices=True
    )
    n_outside_factors = n_in_window - shared.shape[0]

    window_returns = assets.returns[in_window][i_assets]
    window_missing = assets.missing[in_window][i_assets]
    window_factors = factors[i_factors]
    window_rf = rf[i_factors]

    # Step 2: drop series that are missing too often over the span.
    missing_fraction = window_missing.mean(axis=0)
    keep = missing_fraction <= spec.max_missing_fraction
    dropped_series = [
        {
            "name": assets.names[j],
            "missing_fraction": round(float(missing_fraction[j]), 6),
            "n_missing_days": int(window_missing[:, j].sum()),
            "reason": (
                f"missing on {missing_fraction[j]:.2%} of the span, above the "
                f"{spec.max_missing_fraction:.2%} tolerance"
            ),
        }
        for j in np.flatnonzero(~keep)
    ]

    kept_names = tuple(n for n, k in zip(assets.names, keep, strict=True) if k)
    kept_returns = window_returns[:, keep]
    kept_missing = window_missing[:, keep]

    # Step 3: drop dates where a surviving series is still missing.
    bad_date = kept_missing.any(axis=1)
    dropped_dates = [str(d) for d in shared[bad_date]]

    final_dates = shared[~bad_date]
    final_returns = kept_returns[~bad_date]
    final_factors = window_factors[~bad_date]
    final_rf = window_rf[~bad_date]

    if np.isnan(final_returns).any() or np.isnan(final_factors).any():
        raise ValueError("NaNs survived the build; the drop policy did not do its job")

    drops = DropRecord(
        span_start=str(final_dates[0]),
        span_end=str(final_dates[-1]),
        n_dates_in_span=int(shared.shape[0]),
        dropped_series=dropped_series,
        # Keep the list short enough to read; the count is always exact.
        dropped_dates=dropped_dates[:50],
        n_dropped_dates=len(dropped_dates),
        dates_outside_factor_coverage=int(n_outside_factors),
    )

    return BuiltPanel(
        asset_set=asset_set_key,
        label=spec.label,
        dates=final_dates,
        names=kept_names,
        excess_returns=final_returns - final_rf[:, None],
        total_returns=final_returns,
        factors=final_factors,
        factor_names=config.FACTOR_NAMES,
        rf=final_rf,
        drops=drops,
    )


def build_all(
    raw_dir: Path, processed_dir: Path, asset_sets: tuple[str, ...] | None = None
) -> dict[str, BuiltPanel]:
    keys = asset_sets if asset_sets is not None else tuple(config.ASSET_SETS)
    out: dict[str, BuiltPanel] = {}
    for key in keys:
        panel = build_panel(key, raw_dir)
        panel.save(processed_dir / f"{key}.npz")
        out[key] = panel
    return out
