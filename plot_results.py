#!/usr/bin/env python3
"""Plot only certified solves, using matching matrix/settings for comparisons."""
import argparse
import csv
import math
from pathlib import Path
import statistics
from collections import defaultdict


def load_results(path):
    groups, tasks = defaultdict(list), {}
    required = {"matrix", "n", "nnz", "ordering", "backend", "schedule", "threads",
                "solve_ms", "status", "correct", "rtol", "atol", "max_iterations"}
    with Path(path).open(newline="") as stream:
        reader = csv.DictReader(stream)
        if required - set(reader.fieldnames or []):
            raise ValueError("CSV is missing required benchmark columns")
        for row in reader:
            if None in row or any(value is None for value in row.values()):
                raise ValueError("Malformed CSV row")
            if row["correct"] != "1" or row["status"] != "converged":
                continue
            task = tuple(float(row[name]) for name in ("n", "nnz", "rtol", "atol", "max_iterations"))
            if any(not math.isfinite(value) or value < 0 for value in task):
                raise ValueError("Invalid matrix dimensions or solver settings")
            if tasks.setdefault(row["matrix"], task) != task:
                raise ValueError("Cannot compare different sizes/tolerances under the same matrix name")
            threads, timing = int(row["threads"]), float(row["solve_ms"])
            if threads < 1 or not math.isfinite(timing) or timing < 0:
                raise ValueError("Invalid timing or thread count")
            key = (row["matrix"], row["ordering"], row["backend"], row["schedule"], threads)
            groups[key].append(timing)
    if not groups:
        raise ValueError("No converged, correct rows to plot")
    return dict(groups)


def label(key):
    return f"{key[1]}\n{key[2]} {key[3]} t={key[4]}"


def draw(plt, keys, values, title, ylabel, path, errors=None, reference=None):
    fig, ax = plt.subplots(figsize=(max(7, min(18, len(keys) * .8)), 5))
    ax.bar(range(len(keys)), values, yerr=errors, capsize=3)
    ax.set_xticks(range(len(keys)))
    ax.set_xticklabels([label(key) for key in keys], rotation=40, ha="right")
    if reference is not None:
        ax.axhline(reference, color="gray", linestyle="--")
    ax.set(title=title, ylabel=ylabel)
    ax.grid(axis="y", alpha=.2)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def create_plots(groups, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    matrices = sorted({key[0] for key in groups})
    for index, matrix in enumerate(matrices, 1):
        prefix = "" if len(matrices) == 1 else f"matrix_{index}_"
        keys = sorted(key for key in groups if key[0] == matrix)
        medians = [statistics.median(groups[key]) for key in keys]
        errors = [[median - min(groups[key]) for key, median in zip(keys, medians)],
                  [max(groups[key]) - median for key, median in zip(keys, medians)]]
        draw(plt, keys, medians, f"{matrix}: median solve time (min/max range)", "Milliseconds",
             output / f"{prefix}runtime.png", errors=errors)
        speed_keys, speedups, reorder_keys, reorder_speedups = [], [], [], []
        for key in keys:
            elapsed = statistics.median(groups[key])
            serial = (matrix, key[1], "serial", "none", 1)
            original = (matrix, "original", key[2], key[3], key[4])
            if elapsed > 0 and serial in groups:
                speed_keys.append(key)
                speedups.append(statistics.median(groups[serial]) / elapsed)
            if elapsed > 0 and key[1] == "reordered" and original in groups:
                reorder_keys.append(key)
                reorder_speedups.append(statistics.median(groups[original]) / elapsed)
        if speed_keys:
            draw(plt, speed_keys, speedups, f"{matrix}: speedup versus matching serial",
                 "Serial time / solve time", output / f"{prefix}speedup.png", reference=1)
        if reorder_keys:
            draw(plt, reorder_keys, reorder_speedups, f"{matrix}: reordering effect (solve time only)",
                 "Original time / reordered time", output / f"{prefix}reordering.png", reference=1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv")
    parser.add_argument("--output-dir", help="Default: plots directory beside the input CSV")
    args = parser.parse_args()
    try:
        output = Path(args.output_dir) if args.output_dir else Path(args.csv).parent / "plots"
        create_plots(load_results(args.csv), output)
    except (OSError, ValueError, ImportError) as error:
        parser.exit(1, f"Error: {error}\n")
    print(f"Wrote plots to {output}")


if __name__ == "__main__":
    main()
