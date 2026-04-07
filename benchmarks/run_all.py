from __future__ import annotations

from pathlib import Path
from typing import Callable

from tabulate import tabulate

from suites import activation, attention, gemm, norm, quantize, softmax

SUITES: list[tuple[str, Callable]] = [
    ("norm", norm.run_suite),
    ("activation", activation.run_suite),
    ("attention", attention.run_suite),
    ("gemm", gemm.run_suite),
    ("quantize", quantize.run_suite),
    ("softmax", softmax.run_suite),
]


def main() -> None:
    rows: list[list[str | int]] = []

    for suite_name, runner in SUITES:
        result, output_path = runner()
        implementations = ", ".join(sorted({entry.name for entry in result.results}))
        rows.append([
            suite_name,
            result.kernel,
            len(result.results),
            implementations,
            str(Path(output_path).relative_to(Path(__file__).resolve().parent.parent)),
        ])

    print("\n=== benchmark summary ===")
    print(
        tabulate(
            rows,
            headers=["Suite", "Kernel", "Measurements", "Implementations", "Export"],
            tablefmt="simple",
        )
    )


if __name__ == "__main__":
    main()
