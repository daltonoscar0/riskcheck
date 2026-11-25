"""Download the Kenneth French Data Library files used by riskcheck.

Every download is recorded in ``data/raw/manifest.json`` with the URL, the UTC
download timestamp, the byte size and a SHA-256 checksum, so a later run can
tell whether it is looking at the same vintage of the data. French updates
these files monthly and rewrites history when CRSP revises, so the checksum is
the only honest way to say "this result came from that data".
"""

from __future__ import annotations

import hashlib
import json
import urllib.request
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

BASE_URL = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp"

# Terms of use live at
# https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html
# and ask only that the source be credited; see README.md.


@dataclass(frozen=True)
class Source:
    """One zipped CSV from the data library."""

    key: str
    zip_name: str
    csv_name: str
    description: str

    @property
    def url(self) -> str:
        return f"{BASE_URL}/{self.zip_name}"


SOURCES: tuple[Source, ...] = (
    Source(
        key="industry49",
        zip_name="49_Industry_Portfolios_daily_CSV.zip",
        csv_name="49_Industry_Portfolios_Daily.csv",
        description="49 industry portfolios, daily, value- and equal-weighted",
    ),
    Source(
        key="size_bm_25",
        zip_name="25_Portfolios_5x5_Daily_CSV.zip",
        csv_name="25_Portfolios_5x5_Daily.csv",
        description="25 portfolios formed on size and book-to-market, daily",
    ),
    Source(
        key="size_bm_100",
        zip_name="100_Portfolios_10x10_Daily_CSV.zip",
        csv_name="100_Portfolios_10x10_Daily.csv",
        description="100 portfolios formed on size and book-to-market, daily",
    ),
    Source(
        key="ff5",
        zip_name="F-F_Research_Data_5_Factors_2x3_daily_CSV.zip",
        csv_name="F-F_Research_Data_5_Factors_2x3_daily.csv",
        description="Fama-French 5 research factors plus the risk-free rate, daily",
    ),
    Source(
        key="momentum",
        zip_name="F-F_Momentum_Factor_daily_CSV.zip",
        csv_name="F-F_Momentum_Factor_daily.csv",
        description="Momentum factor, daily",
    ),
)

SOURCES_BY_KEY = {s.key: s for s in SOURCES}


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while block := fh.read(chunk):
            h.update(block)
    return h.hexdigest()


def download(source: Source, raw_dir: Path, timeout: float = 300.0) -> Path:
    """Download one source zip into ``raw_dir`` and return its path."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    dest = raw_dir / source.zip_name
    # A plain urlopen gets a 403 from the Dartmouth server without a UA.
    request = urllib.request.Request(source.url, headers={"User-Agent": "riskcheck/0.1"})
    with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
        payload = response.read()
    dest.write_bytes(payload)
    return dest


def fetch_all(raw_dir: Path, keys: list[str] | None = None, timeout: float = 300.0) -> dict:
    """Download the requested sources and write/refresh the manifest."""
    wanted = SOURCES if keys is None else tuple(SOURCES_BY_KEY[k] for k in keys)
    manifest_path = raw_dir / "manifest.json"
    manifest: dict = {}
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())

    entries: dict = manifest.get("files", {})
    for source in wanted:
        path = download(source, raw_dir, timeout=timeout)
        entries[source.key] = {
            "url": source.url,
            "zip_name": source.zip_name,
            "csv_name": source.csv_name,
            "description": source.description,
            "downloaded_utc": datetime.now(UTC).isoformat(timespec="seconds"),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }

    manifest = {
        "source": "Kenneth R. French Data Library, Tuck School of Business at Dartmouth",
        "library_url": (
            "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html"
        ),
        "files": entries,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest
