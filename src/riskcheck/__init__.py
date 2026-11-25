"""riskcheck: walk-forward validation of equity covariance forecasts.

Double precision is switched on before anything builds an array. Minimum-
variance weights require inverting a covariance matrix whose condition number
runs into the thousands, and in float32 the resulting weights -- and so the
realised volatility that is the headline result -- move by enough to matter.
Set ``RISKCHECK_X64=0`` to override, which is useful on a GPU where float64
throughput is a thirty-second of float32.
"""

from __future__ import annotations

import os

import jax

jax.config.update("jax_enable_x64", os.environ.get("RISKCHECK_X64", "1") == "1")

__version__ = "0.1.0"


def using_x64() -> bool:
    return bool(jax.config.read("jax_enable_x64"))
