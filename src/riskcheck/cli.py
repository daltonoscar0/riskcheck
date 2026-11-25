"""The ``riskcheck`` command line.

The commands are meant to be run in order, and each refuses to guess at the
output of the one before it:

    riskcheck fetch      # download the French files, record checksums
    riskcheck build      # parse and align them into gap-free panels
    riskcheck validate   # choose hyperparameters on 1972-1999, freeze them
    riskcheck evaluate   # score the test span once, write results/
    riskcheck report     # print the headline tables as Markdown
    riskcheck bench      # time the walk-forward
"""

from __future__ import annotations

import json
from pathlib import Path

import typer

from . import config, plots, report, runner, validate
from .build import BuiltPanel, build_all
from .config import Paths
from .fetch import SOURCES, fetch_all

app = typer.Typer(
    add_completion=False,
    help="Walk-forward validation of equity covariance forecasts.",
    no_args_is_help=True,
)


def _paths(root: Path | None) -> Paths:
    return Paths(root=root or Path.cwd())


def _load_panels(paths: Paths, keys: tuple[str, ...]) -> dict[str, BuiltPanel]:
    panels = {}
    for key in keys:
        path = paths.processed / f"{key}.npz"
        if not path.exists():
            raise typer.BadParameter(
                f"{path} not found; run `riskcheck build` first (and `riskcheck fetch` "
                "before that if data/raw is empty)"
            )
        panels[key] = BuiltPanel.load(path)
    return panels


@app.command()
def fetch(
    root: Path = typer.Option(None, help="Project root; defaults to the current directory."),
    only: list[str] = typer.Option(None, "--only", help="Fetch just these source keys."),
) -> None:
    """Download the Kenneth French daily files into data/raw and checksum them."""
    paths = _paths(root)
    keys = list(only) if only else None
    typer.echo(f"Downloading {len(keys) if keys else len(SOURCES)} file(s) to {paths.raw}")
    manifest = fetch_all(paths.raw, keys=keys)
    for key, entry in sorted(manifest["files"].items()):
        typer.echo(
            f"  {key:14s} {entry['bytes'] / 1e6:7.2f} MB  "
            f"sha256 {entry['sha256'][:12]}...  {entry['downloaded_utc']}"
        )
    typer.echo(f"Manifest written to {paths.raw / 'manifest.json'}")


@app.command()
def build(
    root: Path = typer.Option(None, help="Project root; defaults to the current directory."),
) -> None:
    """Parse the raw files into aligned, gap-free daily panels."""
    paths = _paths(root)
    panels = build_all(paths.raw, paths.processed)
    for key, panel in panels.items():
        typer.echo(
            f"  {key:14s} N={panel.n_assets:3d} T={panel.dates.shape[0]:6d}  "
            f"{panel.dates[0]}..{panel.dates[-1]}"
        )
        for dropped in panel.drops.dropped_series:
            typer.echo(f"      dropped series {dropped['name']}: {dropped['reason']}")
        if panel.drops.n_dropped_dates:
            typer.echo(
                f"      dropped {panel.drops.n_dropped_dates} date(s) with a remaining gap: "
                f"{', '.join(panel.drops.dropped_dates[:5])}"
            )
    typer.echo(f"Panels written to {paths.processed}")


def validate_(
    root: Path = typer.Option(None, help="Project root; defaults to the current directory."),
) -> None:
    """Choose hyperparameters on the validation span and freeze them."""
    paths = _paths(root)
    panels = _load_panels(paths, config.MAIN_ASSET_SETS)
    typer.echo(
        f"Validating on observations up to {config.VALIDATION_END} "
        f"(test span starts {config.TEST_START})"
    )
    blob = validate.run_validation(panels)
    validate.save_hyperparameters(blob, paths.hyperparameters)
    typer.echo(f"Frozen to {paths.hyperparameters}")


# `validate` is a method name on typer's internals, so register it explicitly.
app.command("validate")(validate_)


@app.command()
def evaluate(
    root: Path = typer.Option(None, help="Project root; defaults to the current directory."),
    skip_plots: bool = typer.Option(False, help="Write tables but not figures."),
) -> None:
    """Score the frozen estimators on the test span and write results/."""
    paths = _paths(root)
    panels = _load_panels(paths, config.MAIN_ASSET_SETS)
    specs = validate.load_final_specs(paths.hyperparameters)
    typer.echo(f"Evaluating {len(specs)} estimators from {config.TEST_START}:")
    for spec in specs:
        typer.echo(f"    {spec.label}")

    run = runner.run_test_evaluation(panels, specs)
    runner.write_results(run, paths.results, raw_dir=paths.raw)
    typer.echo(f"Wrote tables to {paths.results} ({run.metadata['wall_seconds']}s)")

    if not skip_plots:
        written = plots.rolling_bias_figure(
            run.rolling,
            paths.results / "rolling_bias",
            asset_label=panels[report.HEADLINE_ASSET_SET].label,
        )
        for path in written:
            typer.echo(f"  figure {path}")


def report_(
    root: Path = typer.Option(None, help="Project root; defaults to the current directory."),
    out: Path = typer.Option(None, help="Also write the Markdown to this file."),
) -> None:
    """Print the headline tables as Markdown."""
    paths = _paths(root)
    text = report.render_report(paths.results)
    typer.echo(text)
    if out is not None:
        out.write_text(text + "\n")


app.command("report")(report_)


@app.command()
def bench(
    root: Path = typer.Option(None, help="Project root; defaults to the current directory."),
    asset_set: str = typer.Option(
        "industry49", help="Asset set to time; size_bm_100 is the large-N case."
    ),
    n_portfolios: int = typer.Option(
        config.N_RANDOM_PORTFOLIOS, help="Random portfolios per set."
    ),
    repeats: int = typer.Option(1, help="Timed repeats after the warm-up."),
    out: Path = typer.Option(None, help="Append the timing record to this JSON file."),
) -> None:
    """Time the full walk-forward evaluation on this machine."""
    from . import bench as bench_module

    paths = _paths(root)
    panels = _load_panels(paths, (asset_set,))
    specs = validate.load_final_specs(paths.hyperparameters)
    record = bench_module.run_benchmark(
        panels[asset_set], specs, n_portfolios=n_portfolios, repeats=repeats
    )
    typer.echo(json.dumps(record, indent=2))
    if out is not None:
        existing = json.loads(out.read_text()) if out.exists() else []
        existing.append(record)
        out.write_text(json.dumps(existing, indent=2) + "\n")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
