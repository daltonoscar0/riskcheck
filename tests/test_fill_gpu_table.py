"""The GPU table arrives by copy-paste, so the paste is what gets tested.

Oscar runs the Colab notebook and pipes its table into
``scripts/fill_gpu_table.py``. Nothing about that path is type-checked by the
language, so the failure modes are all human: the wrong cell copied, a run
where the GPU never attached, a header renamed on one side only. Each of those
has a test here, because a bad paste ends up in the README, where
``test_readme_matches_results`` then holds it to be a measured fact.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "fill_gpu_table.py"
README = ROOT / "README.md"


@pytest.fixture(scope="module")
def fill():
    spec = importlib.util.spec_from_file_location("fill_gpu_table", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


#: What `print(table.to_markdown(index=False))` emits: pandas pads the cells
#: and uses a dashes-and-colons alignment row.
PASTED = """\
| Precision   |   Assets |   Portfolios per set |   T4 best (s) |   First pass incl. compile (s) |
|:------------|---------:|---------------------:|--------------:|-------------------------------:|
| float32     |       49 |                 2000 |          11.4 |                           31.2 |
| float64     |       49 |                 2000 |          58.9 |                           79.1 |
| float32     |       95 |                20000 |          41.7 |                           66.0 |
| float64     |       95 |                20000 |         612.3 |                          640.8 |
"""

SKELETON = """\
# A README

Some prose.

<!-- GPU-TABLE-START -->
> Not yet run on a GPU.
<!-- GPU-TABLE-END -->

More prose.
"""


def _readme(tmp_path: Path, text: str = SKELETON) -> Path:
    path = tmp_path / "README.md"
    path.write_text(text)
    return path


def run(fill, stdin: str, readme: Path, monkeypatch, *, extra=()) -> int:
    monkeypatch.setattr("sys.stdin", __import__("io").StringIO(stdin))
    return fill.main(["--readme", str(readme), *extra])


def test_a_pasted_notebook_table_lands_between_the_markers(
    fill, tmp_path, monkeypatch
):
    readme = _readme(tmp_path)
    assert run(fill, PASTED, readme, monkeypatch) == 0

    text = readme.read_text()
    assert "Not yet run on a GPU" not in text
    assert text.startswith("# A README")
    assert text.rstrip().endswith("More prose.")
    assert "| float64     |     95 |              20000 |     612.3 |" in text.replace(
        "  ", "  "
    ) or "612.3" in text
    # The markers survive, so a second run can replace the block again.
    assert fill.START in text and fill.END in text


def test_filing_twice_replaces_rather_than_appends(fill, tmp_path, monkeypatch):
    readme = _readme(tmp_path)
    run(fill, PASTED, readme, monkeypatch)
    once = readme.read_text()
    run(fill, PASTED, readme, monkeypatch)
    assert readme.read_text() == once
    assert once.count(fill.START) == 1


def test_the_rows_survive_the_round_trip(fill, tmp_path, monkeypatch):
    readme = _readme(tmp_path)
    run(fill, PASTED, readme, monkeypatch)
    text = readme.read_text()
    block = text.split(fill.START)[1].split(fill.END)[0]

    header, rows = fill.parse_table(block)
    assert header == fill.EXPECTED_HEADER
    assert len(rows) == 4
    assert [r[0] for r in rows].count("float64") == 2
    # Values, not just shape.
    by_key = {(r[0], r[1], r[2]): r[3] for r in rows}
    assert by_key[("float64", "95", "20000")] == "612.3"
    assert by_key[("float32", "49", "2000")] == "11.4"


def test_the_wrong_cell_is_rejected_rather_than_filed(fill, tmp_path, monkeypatch):
    """The notebook prints other tables; pasting one must not corrupt the README."""
    readme = _readme(tmp_path)
    wrong = (
        "| backend | float64 | n_assets | best_seconds |\n"
        "|---|---|---|---|\n"
        "| gpu | True | 49 | 58.9 |\n"
    )
    assert run(fill, wrong, readme, monkeypatch) == 2
    assert readme.read_text() == SKELETON, "a rejected paste must change nothing"


def test_text_with_no_table_is_rejected(fill, tmp_path, monkeypatch):
    readme = _readme(tmp_path)
    assert run(fill, "No GPU attached: set Runtime -> ...\n", readme, monkeypatch) == 2
    assert readme.read_text() == SKELETON


def test_nvidia_smi_noise_around_the_table_is_tolerated(fill, tmp_path, monkeypatch):
    """Realistically the paste includes whatever else the cell printed."""
    readme = _readme(tmp_path)
    noisy = f"gpu x64=True  N= 49 P=  2000  best   58.89s\n\n{PASTED}\nCopy the table above\n"
    assert run(fill, noisy, readme, monkeypatch) == 0
    assert "612.3" in readme.read_text()


def test_a_non_numeric_timing_is_rejected(fill, tmp_path, monkeypatch):
    readme = _readme(tmp_path)
    broken = PASTED.replace("58.9", "n/a")
    assert run(fill, broken, readme, monkeypatch) == 2
    assert readme.read_text() == SKELETON


def test_an_unknown_precision_is_rejected(fill, tmp_path, monkeypatch):
    readme = _readme(tmp_path)
    broken = PASTED.replace("float32", "bfloat16")
    assert run(fill, broken, readme, monkeypatch) == 2


def test_float64_beating_float32_is_warned_about_not_silently_filed(fill, capsys):
    """On a T4 double precision runs at 1/32 throughput, so this is suspect."""
    header, rows = fill.parse_table(PASTED.replace("58.9", "1.9"))
    warnings = fill.check_rows(rows)
    assert any("float64" in w and "faster than" in w for w in warnings)


def test_a_first_pass_faster_than_the_best_pass_is_warned_about(fill):
    """The first pass includes compilation, so it cannot be the quicker one."""
    header, rows = fill.parse_table(PASTED.replace("79.1", "9.1"))
    warnings = fill.check_rows(rows)
    assert any("first pass" in w for w in warnings)


def test_a_clean_table_warns_about_nothing(fill):
    header, rows = fill.parse_table(PASTED)
    assert fill.check_rows(rows) == []


def test_a_readme_without_the_markers_is_refused(fill, tmp_path, monkeypatch):
    readme = _readme(tmp_path, "# A README\n\nNo markers here.\n")
    assert run(fill, PASTED, readme, monkeypatch) == 2


def test_dry_run_prints_the_block_and_writes_nothing(
    fill, tmp_path, monkeypatch, capsys
):
    readme = _readme(tmp_path)
    assert run(fill, PASTED, readme, monkeypatch, extra=["--dry-run"]) == 0
    assert readme.read_text() == SKELETON
    assert "T4 best (s)" in capsys.readouterr().out


# --- the real README --------------------------------------------------------


def test_the_project_readme_has_somewhere_to_put_the_table(fill):
    """Otherwise the script is correct and the paste still has nowhere to go."""
    text = README.read_text()
    assert fill.START in text and fill.END in text
    assert text.index(fill.START) < text.index(fill.END)


def test_the_colab_badge_points_at_the_notebook_on_main(fill):
    text = README.read_text()
    url = (
        "https://colab.research.google.com/github/daltonoscar0/riskcheck/"
        "blob/main/notebooks/colab_gpu_benchmark.ipynb"
    )
    assert url in text, "the Open in Colab badge must open this repo's notebook from main"
    assert (ROOT / "notebooks" / "colab_gpu_benchmark.ipynb").exists()


def test_the_notebook_prints_the_header_the_script_demands(fill):
    """The two sides of the copy-paste must agree, and nothing else enforces it."""
    import json

    nb = json.loads((ROOT / "notebooks" / "colab_gpu_benchmark.ipynb").read_text())
    sources = [
        "".join(cell["source"]) for cell in nb["cells"] if cell["cell_type"] == "code"
    ]
    printing = [s for s in sources if "to_markdown" in s]
    assert len(printing) == 1, "expected exactly one cell to print the table"
    for column in fill.EXPECTED_HEADER:
        assert f"'{column}'" in printing[0], f"notebook never emits a {column!r} column"
