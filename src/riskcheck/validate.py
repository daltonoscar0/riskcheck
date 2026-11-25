"""Hyperparameter selection, on the validation span only.

Every tunable number in the project is chosen here and then frozen into
``results/hyperparameters.json`` before ``riskcheck evaluate`` is allowed to
look at the test span. The selection criterion is the **Gaussian log score**
(negative log-likelihood of the realised returns under the forecast
covariance), averaged over the two asset sets at the one-week horizon.

Three reasons for that criterion rather than the bias statistic:

* It is a strictly proper scoring rule, so it is minimised in expectation only
  by the true conditional covariance. The bias statistic is not -- a model can
  sit exactly on 1 while getting the correlation structure completely wrong,
  because the bias statistic only looks at one portfolio's scale at a time.
* It does not depend on an arbitrary choice of test portfolios.
* It keeps the headline test-period results genuinely out of sample as
  *calibration* checks. The minimum-variance bias statistic is the number the
  project is really about, and tuning on it would make its test-period value a
  partly in-sample quantity.

The cost is that nothing is tuned to make the minimum-variance portfolios look
good, which is the honest arrangement: the log score is what a modeller would
reasonably optimise, and the minimum-variance failure is what happens anyway.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from . import config, estimators, evaluate
from .build import BuiltPanel
from .estimators import Spec

#: The criterion horizon. One week is the project's primary horizon and has
#: enough validation observations (about 1,390) for a stable ranking.
CRITERION_HORIZON = "1w"

WINDOWS = (252, 756)

# The half-life grids run well past the RiskMetrics default of about 11 days
# (lambda = 0.94). They have to: a 49-asset covariance matrix estimated with an
# 11-day half-life has roughly 16 effective observations and is close to
# singular, so the log score falls off a cliff at the short end. Extending the
# grid upward until the winner is interior is what stops the frozen choice from
# being an artefact of where the grid happened to stop.
EWMA_HALFLIVES = (11.0, 21.0, 42.0, 63.0, 126.0, 252.0, 504.0)
PCA_FACTORS = (1, 2, 3, 5, 8, 10)
FACTOR_COV_HALFLIVES = (21.0, 42.0, 63.0, 126.0, 252.0)
VRA_HALFLIVES = (21.0, 42.0, 63.0, 126.0, 252.0, 504.0, 1008.0)

#: Families searched for a base model, in report order.
BASE_FAMILIES = ("sample", "ewma", "ledoit_wolf", "pca", "fundamental")


def default_grids() -> dict[str, list[Spec]]:
    """The candidate set for each estimator family."""
    return {
        "sample": [Spec("sample", {"window": w}) for w in WINDOWS],
        "ewma": [Spec("ewma", {"halflife": h}) for h in EWMA_HALFLIVES],
        "ledoit_wolf": [Spec("ledoit_wolf", {"window": w}) for w in WINDOWS],
        "pca": [
            Spec("pca", {"window": w, "n_factors": k}) for w in WINDOWS for k in PCA_FACTORS
        ],
        "fundamental": [
            Spec("fundamental", {"window": w, "factor_halflife": h})
            for w in WINDOWS
            for h in FACTOR_COV_HALFLIVES
        ],
    }


@dataclass
class GridRow:
    spec: Spec
    scores: dict[str, float]
    criterion: float

    def to_json(self) -> dict:
        return {
            "spec": self.spec.to_json(),
            "label": self.spec.label,
            "log_score_by_asset_set": self.scores,
            "criterion": self.criterion,
        }


@dataclass
class PanelContext:
    """Everything a sweep needs for one asset set, computed once."""

    key: str
    panel: BuiltPanel
    returns: jnp.ndarray
    factors: jnp.ndarray
    t_idx: np.ndarray
    observations: evaluate.Observations


def panel_context(panel: BuiltPanel, horizon: str, span: str) -> PanelContext:
    """Build the context for one asset set, restricted to one span.

    ``span`` is ``"validation"`` or ``"test"``. Covariances are still computed
    over the whole grid -- the exponentially weighted estimators need their
    history -- but only observations realised inside the span are scored.
    """
    t_idx = estimators.rebalance_indices(panel.dates.shape[0])
    obs = evaluate.build_observations(panel.excess_returns, panel.dates, t_idx, horizon)
    if span == "validation":
        obs = obs.between(None, config.VALIDATION_END)
    elif span == "test":
        obs = obs.between(config.TEST_START, None)
    else:
        raise ValueError(f"span must be 'validation' or 'test', got {span!r}")
    return PanelContext(
        key=panel.asset_set,
        panel=panel,
        returns=jnp.asarray(panel.excess_returns),
        factors=jnp.asarray(panel.factors),
        t_idx=t_idx,
        observations=obs,
    )


def score_specs(specs: list[Spec], contexts: list[PanelContext]) -> list[GridRow]:
    """Mean log score of each spec on each asset set, plus the averaged criterion."""
    per_spec: dict[str, dict[str, float]] = {spec.key: {} for spec in specs}
    for context in contexts:
        for spec, covariances in estimators.forecasts_for_specs(
            specs, context.returns, context.factors, context.t_idx
        ):
            scores = evaluate.log_score_per_observation(context.observations, covariances)
            per_spec[spec.key][context.key] = float(np.mean(scores))

    rows = []
    for spec in specs:
        scores = per_spec[spec.key]
        rows.append(
            GridRow(
                spec=spec,
                scores=scores,
                criterion=float(np.mean(list(scores.values()))),
            )
        )
    return rows


def run_validation(panels: dict[str, BuiltPanel], verbose: bool = True) -> dict:
    """Sweep every grid on the validation span and freeze the winners."""
    contexts = [
        panel_context(panels[key], CRITERION_HORIZON, "validation")
        for key in config.MAIN_ASSET_SETS
    ]
    grids = default_grids()

    grid_results: dict[str, list[GridRow]] = {}
    chosen: dict[str, Spec] = {}
    for family in BASE_FAMILIES:
        rows = score_specs(grids[family], contexts)
        rows.sort(key=lambda r: r.criterion)
        grid_results[family] = rows
        chosen[family] = rows[0].spec
        if verbose:
            print(f"  {family}: chose {rows[0].spec.label}  (log score {rows[0].criterion:.4f})")

    # The regime adjustment is applied to whichever base model wins outright.
    best_base = min(
        (rows[0] for rows in grid_results.values()), key=lambda r: r.criterion
    ).spec
    vra_specs = [Spec("vra", {"halflife": h}, base=best_base) for h in VRA_HALFLIVES]
    vra_rows = score_specs(vra_specs, contexts)
    vra_rows.sort(key=lambda r: r.criterion)
    chosen_vra = vra_rows[0].spec
    if verbose:
        print(f"  regime adjustment on {best_base.short_label}: chose {chosen_vra.label}")
        print(f"    (log score {vra_rows[0].criterion:.4f})")

    final_specs = [chosen[family] for family in BASE_FAMILIES] + [chosen_vra]

    return {
        "criterion": (
            "mean Gaussian log score per asset (lower is better), "
            f"{CRITERION_HORIZON} horizon, averaged over asset sets"
        ),
        "criterion_horizon": CRITERION_HORIZON,
        "validation_span": {
            "first_observation": str(contexts[0].observations.end_date[0]),
            "last_observation": str(contexts[0].observations.end_date[-1]),
            "n_observations": {c.key: len(c.observations) for c in contexts},
        },
        "test_span_start": str(config.TEST_START),
        "grids": {
            family: [row.to_json() for row in rows] for family, rows in grid_results.items()
        },
        "vra_base": best_base.to_json(),
        "vra_grid": [row.to_json() for row in vra_rows],
        "chosen": {family: spec.to_json() for family, spec in chosen.items()},
        "chosen_vra": chosen_vra.to_json(),
        "final_specs": [spec.to_json() for spec in final_specs],
    }


def save_hyperparameters(blob: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(blob, indent=2) + "\n")


def load_final_specs(path: Path) -> list[Spec]:
    """The frozen estimator list, as written by ``riskcheck validate``."""
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found; run `riskcheck validate` before `riskcheck evaluate` "
            "so the hyperparameters are chosen on the validation span"
        )
    blob = json.loads(path.read_text())
    return [Spec.from_json(s) for s in blob["final_specs"]]
