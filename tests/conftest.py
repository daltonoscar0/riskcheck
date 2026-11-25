"""Shared fixtures, and a hard guarantee that the suite never touches the network.

The network block is an autouse fixture rather than a convention. "The tests
run offline" is a property worth enforcing: without it, a stray download would
pass locally on a laptop with a connection and fail in CI, or -- worse -- quietly
make the suite depend on a Dartmouth web server being up.
"""

from __future__ import annotations

import socket
import zipfile
from pathlib import Path

import numpy as np
import pytest


class NetworkUseInTests(RuntimeError):
    pass


@pytest.fixture(autouse=True)
def no_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make any attempt to open a socket fail loudly."""

    def blocked(*args, **kwargs):
        raise NetworkUseInTests(
            "the test suite must run offline; use a fixture instead of downloading"
        )

    monkeypatch.setattr(socket, "socket", blocked)
    monkeypatch.setattr(socket, "create_connection", blocked)


# ---------------------------------------------------------------------------
# Synthetic returns with a covariance we know exactly
# ---------------------------------------------------------------------------


class FactorWorld:
    """A simulated market whose true covariance is known in closed form.

    Returns are generated as ``r = f B + e`` with ``f`` a K-factor series and
    ``e`` diagonal noise, so the true covariance is exactly
    ``B' Sigma_f B + D``. That structure is deliberate: it is low-rank plus
    diagonal, so the principal-component and fundamental factor models are
    capable of recovering it exactly, and a convergence test on them is a test
    of the implementation rather than of model misspecification.
    """

    def __init__(
        self,
        n_assets: int = 6,
        n_factors: int = 3,
        seed: int = 11,
        specific_scale: float = 1.0,
    ):
        rng = np.random.default_rng(seed)
        self.n_assets = n_assets
        self.n_factors = n_factors

        self.loadings = rng.normal(0.0, 0.9, size=(n_factors, n_assets))  # (K, N)
        factor_vol = np.array([0.011, 0.006, 0.004])[:n_factors]
        correlation = np.eye(n_factors) + 0.15 * (
            np.ones((n_factors, n_factors)) - np.eye(n_factors)
        )
        self.factor_cov = correlation * np.outer(factor_vol, factor_vol)
        # `specific_scale` shrinks the idiosyncratic share. It matters for the
        # principal-component model: PCA splits a matrix into a low-rank part
        # plus a *diagonal*, which is not the same decomposition as
        # factors-plus-idiosyncratic-noise unless the specific variances are
        # small or equal. With a large, uneven specific component, PCA has an
        # asymptotic bias that no amount of data removes.
        self.specific = (specific_scale * rng.uniform(0.004, 0.011, size=n_assets)) ** 2

        self.covariance = (
            self.loadings.T @ self.factor_cov @ self.loadings + np.diag(self.specific)
        )

    def simulate(self, n_days: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(returns (T, N), factors (T, K))`` drawn from the true model."""
        rng = np.random.default_rng(seed)
        factors = rng.multivariate_normal(np.zeros(self.n_factors), self.factor_cov, size=n_days)
        noise = rng.normal(0.0, np.sqrt(self.specific), size=(n_days, self.n_assets))
        return factors @ self.loadings + noise, factors


@pytest.fixture(scope="session")
def world() -> FactorWorld:
    return FactorWorld()


def relative_error(estimate: np.ndarray, truth: np.ndarray) -> float:
    """Frobenius-norm relative error, the summary used by the convergence tests."""
    return float(np.linalg.norm(estimate - truth) / np.linalg.norm(truth))


# ---------------------------------------------------------------------------
# A miniature French file, with all the quirks the real ones have
# ---------------------------------------------------------------------------

FIXTURE_CSV = """This file was created using the 202609 CRSP database.
It contains value- and equal-weighted returns for 3 toy portfolios.

Missing data are indicated by -99.99 or -999.


  Average Value Weighted Returns -- Daily
,Alpha,Beta,Gamma
19690701,   0.56,  -0.07, -99.99
19690702,   0.29,   0.06,   1.50
19690703, -99.99,   0.18,  -0.33
19690707,   3.57,  -0.15,  -999
19690708,   0.10,   0.20,   0.30

  Average Equal Weighted Returns -- Daily
,Alpha,Beta,Gamma
19690701,   9.99,   9.99,   9.99
19690702,   9.99,   9.99,   9.99
19690703,   9.99,   9.99,   9.99
19690707,   9.99,   9.99,   9.99
19690708,   9.99,   9.99,   9.99

Copyright 2026 Eugene F. Fama and Kenneth R. French
"""


def write_zip(path: Path, csv_name: str, text: str) -> Path:
    """Write a one-CSV zip shaped like the ones the data library serves."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(csv_name, text)
    return path


@pytest.fixture
def french_fixture_zip(tmp_path: Path) -> Path:
    return write_zip(tmp_path / "toy_CSV.zip", "toy_Daily.csv", FIXTURE_CSV)
