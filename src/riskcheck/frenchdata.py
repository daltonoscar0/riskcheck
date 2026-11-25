"""Parsing for the Kenneth French Data Library CSV files.

Two quirks of these files have to be handled explicitly, and both are easy to
get silently wrong:

* **Missing values are sentinels, not blanks.** French writes ``-99.99`` or
  ``-999`` where a portfolio had no constituents. Read naively those become a
  -99.99% daily return, which would dominate any covariance estimate.
* **Returns are in percent.** ``0.56`` means 0.56%, i.e. 0.0056.

A single physical file also holds several stacked tables -- the industry file
carries value-weighted returns, then equal-weighted returns, each introduced by
a title line and a header line beginning with a comma. We want the first
(value-weighted) block, so sections are parsed out rather than assumed.
"""

from __future__ import annotations

import io
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

#: Sentinels French uses for "this portfolio had no constituents on this day".
MISSING_SENTINELS: tuple[float, ...] = (-99.99, -999.0)

#: A long portfolio can lose all of its value but no more, so a return of
#: exactly -100% is legitimate (it happens twice in the 100-portfolio file, in
#: 1927 and 1932, both long before the span riskcheck uses). Anything strictly
#: below -100% means a sentinel slipped through the parser.
IMPOSSIBLE_PERCENT = -100.0


@dataclass(frozen=True)
class Section:
    """One stacked table inside a French CSV."""

    title: str
    columns: tuple[str, ...]
    dates: np.ndarray  # int64 YYYYMMDD
    percent: np.ndarray  # float64 (T, N), still in percent, sentinels intact


@dataclass(frozen=True)
class Panel:
    """A parsed returns table: decimal returns plus an explicit missing mask."""

    title: str
    names: tuple[str, ...]
    dates: np.ndarray  # datetime64[D], shape (T,)
    returns: np.ndarray  # float64 (T, N), decimal, NaN where missing
    missing: np.ndarray  # bool (T, N)

    def __post_init__(self) -> None:
        if self.returns.shape != self.missing.shape:
            raise ValueError("returns and missing must have the same shape")
        if self.returns.shape[0] != self.dates.shape[0]:
            raise ValueError("returns and dates disagree on T")
        if self.returns.shape[1] != len(self.names):
            raise ValueError("returns and names disagree on N")


def _is_data_row(line: str) -> bool:
    """True for a row like ``19260701,   0.56,  -0.07,...``."""
    return len(line) > 8 and line[:8].isdigit() and line[8] == ","


def parse_sections(text: str) -> list[Section]:
    """Split a French CSV into its stacked tables, in file order.

    A section is a header line starting with ``,`` followed by consecutive
    date-prefixed data rows. The title is the nearest preceding non-blank line.
    """
    lines = text.splitlines()
    sections: list[Section] = []
    last_text_line = ""
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line:
            i += 1
            continue
        if not line.startswith(","):
            last_text_line = line
            i += 1
            continue

        columns = tuple(c.strip() for c in line.split(",")[1:])
        ncol = len(columns)
        dates: list[int] = []
        rows: list[list[float]] = []
        i += 1
        while i < len(lines):
            row = lines[i].strip()
            if not _is_data_row(row):
                break
            parts = row.split(",")
            # Some tables (e.g. "Number of Firms") repeat the same width; a
            # width change means we have run into a differently shaped block.
            if len(parts) - 1 != ncol:
                break
            dates.append(int(parts[0]))
            rows.append([float(p) for p in parts[1:]])
            i += 1

        if rows:
            sections.append(
                Section(
                    title=last_text_line,
                    columns=columns,
                    dates=np.asarray(dates, dtype=np.int64),
                    percent=np.asarray(rows, dtype=np.float64),
                )
            )
        last_text_line = ""
    return sections


def read_csv_text(path: Path, csv_name: str | None = None) -> str:
    """Read the CSV text from either a zip from the library or a plain CSV."""
    path = Path(path)
    if path.suffix.lower() != ".zip":
        return path.read_text(errors="replace")
    with zipfile.ZipFile(path) as zf:
        names = [n for n in zf.namelist() if n.lower().endswith(".csv")]
        if not names:
            raise ValueError(f"no CSV inside {path}")
        if csv_name is not None:
            wanted = [n for n in names if Path(n).name.lower() == csv_name.lower()]
            name = wanted[0] if wanted else names[0]
        else:
            name = names[0]
        with zf.open(name) as fh:
            return io.TextIOWrapper(fh, encoding="utf-8", errors="replace").read()


def section_to_panel(section: Section) -> Panel:
    """Convert percent-with-sentinels into decimal-with-NaN plus a mask."""
    percent = section.percent
    missing = np.zeros(percent.shape, dtype=bool)
    for sentinel in MISSING_SENTINELS:
        missing |= np.isclose(percent, sentinel, rtol=0.0, atol=1e-6)

    clean = np.where(missing, np.nan, percent)
    stray = clean < IMPOSSIBLE_PERCENT - 1e-6
    if np.any(stray):
        worst = float(np.nanmin(clean))
        raise ValueError(
            f"{int(stray.sum())} values below {IMPOSSIBLE_PERCENT}% (worst {worst}) "
            "survived sentinel removal; the missing-value markers in this file are "
            "not the ones riskcheck knows about"
        )

    dates = np.asarray(
        [
            np.datetime64(f"{d // 10000:04d}-{d // 100 % 100:02d}-{d % 100:02d}")
            for d in section.dates
        ],
        dtype="datetime64[D]",
    )
    return Panel(
        title=section.title,
        names=section.columns,
        dates=dates,
        returns=clean / 100.0,
        missing=missing,
    )


def load_panel(
    path: Path,
    csv_name: str | None = None,
    section_index: int = 0,
    title_contains: str | None = None,
) -> Panel:
    """Load one table from a French file as decimal returns.

    ``section_index`` picks a stacked table by position; ``title_contains``
    picks it by title substring (case-insensitive) and takes precedence, which
    is how we make "value weighted, not equal weighted" explicit at the call
    site rather than relying on file ordering.
    """
    sections = parse_sections(read_csv_text(path, csv_name))
    if not sections:
        raise ValueError(f"no parseable sections in {path}")
    if title_contains is not None:
        needle = title_contains.lower()
        matches = [s for s in sections if needle in s.title.lower()]
        if not matches:
            titles = [s.title for s in sections]
            raise ValueError(f"no section titled like {title_contains!r} in {path}; saw {titles}")
        chosen = matches[0]
    else:
        chosen = sections[section_index]
    return section_to_panel(chosen)
