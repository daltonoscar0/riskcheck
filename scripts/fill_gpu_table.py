#!/usr/bin/env python3
"""File the Colab GPU timings into the README.

    python scripts/fill_gpu_table.py < table.md
    riskcheck-colab-output | python scripts/fill_gpu_table.py

Reads the Markdown table the notebook's last cell prints on stdin and replaces
whatever sits between the ``GPU-TABLE-START`` and ``GPU-TABLE-END`` markers in
README.md. Everything else in the file is left alone, so this is safe to run
again when a second set of timings arrives.

It validates rather than trusts. A table pasted from the wrong cell, or from a
run where the GPU was never attached, is the likely mistake here -- so the
header has to be the one the notebook prints, the rows have to parse as
numbers, and a float64 run claiming to be faster than its float32 twin is
reported as suspect rather than published. None of that is pedantry: these
numbers go straight into the README, where a test then holds them to be true.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "README.md"

START = "<!-- GPU-TABLE-START -->"
END = "<!-- GPU-TABLE-END -->"

#: The notebook's columns, in order. Pasting a table with any other header is
#: the error this is here to catch.
EXPECTED_HEADER = [
    "Precision",
    "Assets",
    "Portfolios per set",
    "T4 best (s)",
    "First pass incl. compile (s)",
]


class BadTable(ValueError):
    """The pasted text is not the table the notebook prints."""


def split_row(line: str) -> list[str]:
    return [cell.strip() for cell in line.strip().strip("|").split("|")]


def parse_table(text: str) -> tuple[list[str], list[list[str]]]:
    """The first Markdown table in ``text``, as (header, rows).

    Tolerant about what surrounds it, because the realistic input is a whole
    Colab cell's output with `nvidia-smi` noise and progress bars around the
    table.
    """
    lines = [line for line in text.splitlines() if line.strip().startswith("|")]
    if not lines:
        raise BadTable(
            "no Markdown table on stdin. Copy the whole table the notebook's "
            "last cell prints, pipe-characters included."
        )

    header = split_row(lines[0])
    body = [split_row(line) for line in lines[1:]]
    # Drop the alignment row (---, :--, --: and so on).
    body = [row for row in body if not all(set(c) <= set("-: ") for c in row)]
    if not body:
        raise BadTable("the table has a header but no rows")
    return header, body


def check_header(header: list[str]) -> None:
    if header != EXPECTED_HEADER:
        raise BadTable(
            "unexpected columns.\n"
            f"  expected: {' | '.join(EXPECTED_HEADER)}\n"
            f"  got:      {' | '.join(header)}\n"
            "This is the table from the notebook's *last* cell, the one that "
            "calls to_markdown()."
        )


def number(cell: str) -> float:
    return float(cell.replace(",", "").replace("**", "").strip())


def check_rows(rows: list[list[str]]) -> list[str]:
    """Parse every cell, and return human-readable warnings about the values."""
    warnings: list[str] = []
    seen: dict[tuple[str, int, int], float] = {}

    for row in rows:
        if len(row) != len(EXPECTED_HEADER):
            raise BadTable(f"row has {len(row)} cells, expected {len(EXPECTED_HEADER)}: {row}")
        precision = row[0].strip()
        if precision not in {"float64", "float32"}:
            raise BadTable(
                f"precision column must be float64 or float32, got {precision!r}"
            )
        try:
            assets = int(number(row[1]))
            portfolios = int(number(row[2]))
            best = number(row[3])
            first = number(row[4])
        except ValueError as exc:
            raise BadTable(f"could not read a number in row {row}: {exc}") from None

        if best <= 0 or first <= 0:
            raise BadTable(f"non-positive timing in row {row}")
        if first < best:
            warnings.append(
                f"{precision} N={assets} P={portfolios}: first pass ({first}s) is "
                f"faster than the best timed pass ({best}s), which should be "
                f"impossible since the first pass includes compilation"
            )
        seen[(precision, assets, portfolios)] = best

    for (precision, assets, portfolios), best in seen.items():
        if precision != "float64":
            continue
        twin = seen.get(("float32", assets, portfolios))
        if twin is not None and best < twin:
            warnings.append(
                f"N={assets} P={portfolios}: float64 ({best}s) came out faster than "
                f"float32 ({twin}s). On a T4 double precision runs at 1/32 the "
                f"throughput, so this is more likely a mislabelled run than a result"
            )
    return warnings


def render(header: list[str], rows: list[list[str]]) -> str:
    """The block to write between the markers."""
    widths = [
        max(len(header[i]), *(len(row[i]) for row in rows)) for i in range(len(header))
    ]
    align = [":--"] + ["--:"] * (len(header) - 1)

    def line(cells: list[str]) -> str:
        padded = [
            cell.ljust(widths[i]) if i == 0 else cell.rjust(widths[i])
            for i, cell in enumerate(cells)
        ]
        return "| " + " | ".join(padded) + " |"

    out = [
        "Free Colab T4, JAX with CUDA 12. Same workload as the CPU table above; only",
        "`T4 best (s)` compares between them: the first-pass column carries XLA compilation.",
        "",
        line(header),
        "| " + " | ".join(align[i].ljust(widths[i]) for i in range(len(header))) + " |",
    ]
    out += [line(row) for row in rows]
    return "\n".join(out)


def replace_block(readme_text: str, block: str) -> str:
    pattern = re.compile(
        re.escape(START) + r".*?" + re.escape(END), re.DOTALL
    )
    if not pattern.search(readme_text):
        raise BadTable(
            f"markers {START} / {END} not found in the README; "
            "they are what says where the table goes"
        )
    return pattern.sub(f"{START}\n{block}\n{END}", readme_text, count=1)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--readme", type=Path, default=README, help="README to update."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be written and change nothing.",
    )
    args = parser.parse_args(argv)

    text = sys.stdin.read()
    try:
        header, rows = parse_table(text)
        check_header(header)
        warnings = check_rows(rows)
        block = render(header, rows)
        updated = replace_block(args.readme.read_text(), block)
    except BadTable as exc:
        print(f"fill_gpu_table: {exc}", file=sys.stderr)
        return 2

    for warning in warnings:
        print(f"fill_gpu_table: warning: {warning}", file=sys.stderr)

    if args.dry_run:
        print(block)
        return 0

    args.readme.write_text(updated)
    print(f"{args.readme}: filed {len(rows)} GPU timings")
    if warnings:
        print("Check the warnings above before committing.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
