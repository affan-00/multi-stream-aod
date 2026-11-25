#!/usr/bin/env python3
"""
Aggregate repeated benchmark_day3.py runs into medians and [min-max] ranges (Table 4a).

Usage:  python summarize_bench.py results/out_bench*.txt
Each file may hold one or several runs (benchmark_day3.py --runs N); every table found is one run.
"""
import re
import statistics as st
import sys
from collections import OrderedDict

KEYS = OrderedDict([
    ("Proposed", "Proposed (event-gated)"),
    ("Continuous MobileNet", "Continuous MobileNet-SSD"),
    ("Continuous YOLOv3", "Continuous YOLOv3-tiny"),
    ("Un", "Ungated MobileNet-SSD + YOLOv3-tiny"),   # "Un-Gated ..." / "Ungated ..."
    ("Continuous YOLOv8n", "Continuous YOLOv8n (416 px)"),
])
COLS = ["1-stream FPS", "4-stream agg FPS", "per-stream FPS", "median ms", "IQR ms", "P95 ms"]


def match(name):
    for k, label in KEYS.items():
        if name.startswith(k):
            return label
    return None


def main(paths):
    data = {label: [[] for _ in COLS] for label in KEYS.values()}
    runs = 0
    for p in paths:
        for line in open(p, encoding="utf-8"):
            if not line.startswith("|"):
                continue
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            label = match(cells[0]) if cells else None
            if label is None or len(cells) < 7:
                continue
            try:
                vals = [float(c) for c in cells[1:7]]
            except ValueError:
                continue
            for i, v in enumerate(vals):
                data[label][i].append(v)
            if label == "Proposed (event-gated)":
                runs += 1
    print(f"runs found: {runs}\n")
    print("| Method | " + " | ".join(COLS) + " | 4-stream speedup of proposed (median) |")
    print("| :--- | " + " | ".join([":---:"] * len(COLS)) + " | :---: |")
    ref = st.median(data["Proposed (event-gated)"][1]) if data["Proposed (event-gated)"][1] else float("nan")
    for label, cols in data.items():
        if not cols[0]:
            continue
        cells = [f"{st.median(c):.2f} [{min(c):.2f}-{max(c):.2f}]" for c in cols]
        sp = "reference" if label.startswith("Proposed") else f"{ref / st.median(cols[1]):.1f}x"
        print(f"| {label} | " + " | ".join(cells) + f" | {sp} |")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    main(sys.argv[1:])
