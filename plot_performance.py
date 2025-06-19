#!/usr/bin/env python3
# flake8: noqa: E501
"""
plot_performance.py
===================
A small utility that parses the stdout log produced by `src/main.cu` and
creates bar charts for runtime (ms) and achieved throughput (GFLOPS).

The CUDA benchmark prints sections of the form::

    Testing matrix dimensions: 2048x2048 * 2048x2048
    ========================================
    Version: Tensor Core mma
    Time: 0.123456 ms
    Performance: 12345.678901 GFLOPS
    ----------------------------------------
    Version: Tensor Core mma kStage
    Time: 0.112233 ms
    Performance: 13456.789012 GFLOPS
    ----------------------------------------
    Version: cuBLAS Tensor Core
    Time: 0.098765 ms
    Performance: 15000.123456 GFLOPS
    ----------------------------------------

For each matrix dimension block the script will produce two PNG figures:
`time_<dims>.png` and `gflops_<dims>.png` (e.g. `time_2048x2048.png`).

Usage
-----
    python plot_performance.py <log_file> [--out-dir <dir>] [--show]

Options
-------
* ``--out-dir``: Directory where the figures will be saved. Defaults to the
  current working directory.
* ``--show``: If given, the figures are also displayed interactively.

Dependences
-----------
* matplotlib>=3.0
* (optional) pandas for pretty-printing tables (not required by default).

Example
-------
    nvcc -o sgemm main.cu ...
    ./sgemm > benchmark.log
    python plot_performance.py benchmark.log --out-dir figs --show
"""

import argparse
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np

# Regular expressions for parsing the log
RE_DIMS = re.compile(
    r"Testing matrix dimensions:\s+(\d+)x(\d+)\s*\*\s*(\d+)x(\d+)"
)
RE_VERSION = re.compile(r"Version:\s*(.*)")
RE_TIME = re.compile(r"Time:\s*([\d\.]+)\s*ms")
RE_GFLOPS = re.compile(r"Performance:\s*([\d\.]+)\s*GFLOPS")


def parse_log(path: os.PathLike) -> List[Dict]:
    """Parse benchmark log and return list of records.

    Each record is a dict with keys: dims (str), version (str), time_ms (float),
    gflops (float).
    """
    records: List[Dict] = []
    current_dims: str | None = None

    with open(path, "r", encoding="utf-8") as f:
        while True:
            line = f.readline()
            if not line:
                break  # EOF
            line = line.strip()

            # Match dimensions header
            dims_match = RE_DIMS.match(line)
            if dims_match:
                M, K1, K2, N = map(int, dims_match.groups())
                # Store as string "MxN" for convenience
                current_dims = f"{M}x{N}"
                continue

            # Match version line
            ver_match = RE_VERSION.match(line)
            if ver_match:
                version = ver_match.group(1).strip()

                # We expect the next two non-empty lines to be Time and Performance
                time_line = f.readline().strip()
                perf_line = f.readline().strip()

                time_match = RE_TIME.match(time_line)
                perf_match = RE_GFLOPS.match(perf_line)

                # Guard against malformed logs
                if not (time_match and perf_match and current_dims):
                    continue

                time_ms = float(time_match.group(1))
                gflops = float(perf_match.group(1))

                records.append(
                    {
                        "dims": current_dims,
                        "version": version,
                        "time_ms": time_ms,
                        "gflops": gflops,
                    }
                )
    return records


def _determine_version_order(records: List[Dict]) -> List[str]:
    """Return versions in first-occurrence order in the log."""
    order: List[str] = []
    for rec in records:
        v = rec["version"]
        if v not in order:
            order.append(v)
    return order


def _group_records_by_dims(records: List[Dict]) -> Dict[str, Dict[str, Dict[str, float]]]:
    """Return nested mapping dims -> version -> metrics dict."""
    grouped: Dict[str, Dict[str, Dict[str, float]]] = defaultdict(dict)
    for rec in records:
        dims = rec["dims"]
        ver = rec["version"]
        grouped[dims][ver] = {"time_ms": rec["time_ms"], "gflops": rec["gflops"]}
    return grouped


def _get_colors(num: int):
    """Return a list of *num* distinct colors using categorical colormaps."""
    if num <= 10:
        cmap = plt.get_cmap("tab10")
    elif num <= 20:
        cmap = plt.get_cmap("tab20")
    else:  # fall back to a perceptually uniform map sampled evenly
        cmap = plt.get_cmap("turbo", num)
    return [cmap(i) for i in range(num)]


def _plot_group(
    grouped_data: Dict[str, Dict[str, Dict[str, float]]],
    versions: List[str],
    metric: str,
    out_path: Path,
    show: bool,
):
    """Plot metric for selected versions across dims categories."""
    dims_list = sorted(grouped_data.keys(), key=lambda x: int(x.split("x")[0]))
    x = np.arange(len(dims_list))
    num_ver = len(versions)
    if num_ver == 0:
        return
    bar_width = 0.8 / num_ver
    fig, ax = plt.subplots(figsize=(12, 6))

    colors = _get_colors(num_ver)
    for idx, ver in enumerate(versions):
        offsets = x - 0.4 + (idx + 0.5) * bar_width
        values = [
            grouped_data[d].get(ver, {}).get(metric, np.nan) for d in dims_list
        ]
        ax.bar(
            offsets,
            values,
            width=bar_width,
            label=ver,
            color=colors[idx],
            edgecolor="white",
        )

    # Aesthetics
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    
    ax.set_xticks(x)
    ax.set_xticklabels(dims_list)
    ax.set_xlabel("Matrix Dimensions (M=N=K)")
    ylabel = "Time (ms)" if metric == "time_ms" else "GFLOPS"
    ax.set_ylabel(ylabel)
    title_metric = "Runtime" if metric == "time_ms" else "Throughput"
    ax.set_title(f"{title_metric} comparison ({out_path.stem})")
    # Place legend outside if too many entries
    if num_ver > 6:
        ax.legend(
            fontsize="small",
            ncol=max(1, num_ver // 10 + 1),
            bbox_to_anchor=(1.02, 1),
            loc="upper left",
            borderaxespad=0.0,
        )
    else:
        ax.legend(fontsize="small")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show(block=False)
    plt.close(fig)


def plot_performance_grouped(records: List[Dict], out_dir: Path, show: bool = False):
    """Create two figures per metric: first 8 versions, remaining versions."""
    out_dir.mkdir(parents=True, exist_ok=True)
    version_order = _determine_version_order(records)
    group1 = version_order[:9]
    group2 = version_order[9:]

    grouped_data = _group_records_by_dims(records)

    metrics = ["time_ms", "gflops"]
    for metric in metrics:
        # First 8
        if group1:
            out_path = out_dir / f"{metric}_first9.png"
            _plot_group(grouped_data, group1, metric, out_path, show)
        # Remaining
        if group2:
            out_path = out_dir / f"{metric}_remaining.png"
            _plot_group(grouped_data, group2, metric, out_path, show)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Plot SGEMM performance results parsed from benchmark stdout logs."
        ),
    )
    parser.add_argument("log_file", help="Path to the benchmark log file")
    parser.add_argument(
        "--out-dir",
        default="./figs",
        help=(
            "Directory where output PNGs will be stored "
            "(default: ./figs)"
        ),
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the figures interactively after saving",
    )
    args = parser.parse_args()

    log_path = Path(args.log_file)
    if not log_path.is_file():
        raise FileNotFoundError(log_path)

    recs = parse_log(log_path)
    if not recs:
        print("No performance records found in the provided log file.")
        exit(1)

    plot_performance_grouped(recs, Path(args.out_dir), show=args.show) 