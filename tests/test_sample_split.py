"""The half-life is chosen on returns that were not scored.

The causality tests in ``test_no_lookahead`` show that no single forecast reads
its own future. That is necessary but not sufficient for the claim the README
makes: hyperparameters could still be picked using returns that the test span
then scores, which would make the frozen half-life a partly in-sample choice
without any individual forecast cheating.

What makes the two spans disjoint is that ``Observations.between`` filters on
``end_date`` -- the last day of the realised return window, the date the return
became known -- rather than on the date the forecast was made. These tests pin
that, because filtering on the forecast date instead would look correct, pass
every causality test, and let a one-month window opened in December 1999 close
inside the test span.
"""

from __future__ import annotations

import numpy as np
import pytest

from riskcheck import config, estimators, evaluate
from riskcheck.build import BuiltPanel
from riskcheck.validate import panel_context

#: Long enough to straddle the split with room for the burn-in on either side:
#: the panel has to reach back far enough before 1999 that validation
#: observations exist at all, and far enough past 2000 to have a test span.
N_DAYS = 9000
PANEL_START = np.datetime64("1980-01-02")


@pytest.fixture(scope="module")
def straddling_panel():
    """A synthetic panel spanning the real split date."""
    from .conftest import FactorWorld

    world = FactorWorld(n_assets=6, n_factors=3)
    returns, _ = world.simulate(N_DAYS, seed=5)
    rng = np.random.default_rng(11)
    factors = np.column_stack(
        [
            rng.multivariate_normal(np.zeros(3), world.factor_cov, size=N_DAYS),
            rng.normal(0, 0.004, size=(N_DAYS, 3)),
        ]
    )
    dates = np.busday_offset(PANEL_START, np.arange(N_DAYS), roll="forward")
    return BuiltPanel(
        asset_set="industry49",
        label="straddling toy panel",
        dates=dates,
        names=tuple(f"A{i}" for i in range(6)),
        excess_returns=returns,
        total_returns=returns,
        factors=factors,
        factor_names=config.FACTOR_NAMES,
        rf=np.zeros(N_DAYS),
        drops=None,
    )


@pytest.mark.parametrize("horizon", list(config.HORIZONS))
def test_selection_and_evaluation_never_share_a_realised_return(
    straddling_panel, horizon
):
    """No return is both used to pick a hyperparameter and scored."""
    selection = panel_context(straddling_panel, horizon, "validation").observations
    evaluation = panel_context(straddling_panel, horizon, "test").observations

    assert len(selection) > 0 and len(evaluation) > 0, "the split must bite"
    assert selection.end_date.max() <= config.VALIDATION_END
    assert evaluation.end_date.min() >= config.TEST_START
    assert selection.end_date.max() < evaluation.end_date.min()
    assert not set(map(str, selection.end_date)) & set(map(str, evaluation.end_date))


@pytest.mark.parametrize("horizon", list(config.HORIZONS))
def test_no_selection_window_reaches_into_the_test_span(straddling_panel, horizon):
    """The *whole* realised window, not just its last day, stays before the split.

    The one-month horizon is where this matters: its window is twenty trading
    days, so a selection observation ending 1999-12-14 opened in November, and
    one ending any later could have opened in a month the test span also sees.
    """
    obs = panel_context(straddling_panel, horizon, "validation").observations
    dates = straddling_panel.dates
    end_rows = np.searchsorted(dates, obs.end_date)
    start_rows = end_rows - obs.days + 1

    assert np.all(dates[start_rows] <= config.VALIDATION_END)
    assert np.all(dates[end_rows] <= config.VALIDATION_END)


def test_filtering_on_the_forecast_date_would_have_leaked(straddling_panel):
    """The distinction this rests on is a real one, not a theoretical one.

    Had the split filtered on the date the forecast was made, the one-month
    horizon would have admitted selection observations whose returns realised
    after 1999-12-31. This asserts that such observations exist, so the test
    above is not passing because the two rules happen to agree on this panel.
    """
    panel = straddling_panel
    t_idx = estimators.rebalance_indices(panel.dates.shape[0])
    obs = evaluate.build_observations(panel.excess_returns, panel.dates, t_idx, "1m")

    forecast_dates = panel.dates[t_idx[obs.forecast_index]]
    would_be_selected = forecast_dates <= config.VALIDATION_END
    leaked = obs.end_date[would_be_selected] > config.VALIDATION_END

    assert leaked.any(), (
        "no window straddles the split on this panel, so the end_date rule "
        "is untested here -- lengthen the panel or move the split"
    )


def test_the_chosen_ewma_halflife_is_interior_to_its_grid():
    """A winner at the edge of the grid is a grid artefact, not a choice.

    Read from the frozen file when there is one: this is a claim about the
    published number, not about the code.
    """
    import json

    path = config.Paths().hyperparameters
    if not path.exists():
        pytest.skip("no frozen hyperparameters; run `riskcheck validate`")

    from riskcheck.validate import EWMA_HALFLIVES

    chosen = json.loads(path.read_text())["chosen"]["ewma"]["params"]["halflife"]
    assert chosen in EWMA_HALFLIVES
    assert min(EWMA_HALFLIVES) < chosen < max(EWMA_HALFLIVES), (
        f"half-life {chosen} sits at the edge of {EWMA_HALFLIVES}; "
        "extend the grid until the winner is interior"
    )
