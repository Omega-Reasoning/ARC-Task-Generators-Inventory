#!/usr/bin/env python3
import argparse, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict
from matplotlib.colors import LogNorm

MIN_SIZE, MAX_SIZE = 5, 30
ODD_TICKS = list(range(5, 31, 2))  # 5,7,9,...,29

# -------- Data iterators --------
def iter_tasks_from_csv(csv_path, generator_name=None):
    df = pd.read_csv(csv_path)
    if generator_name:
        df = df[df.get("generator_name") == generator_name]
    for _, row in df.iterrows():
        try:
            yield json.loads(row["data"])
        except Exception:
            continue

def iter_tasks_from_dir(json_dir):
    for p in Path(json_dir).glob("*.json"):
        try:
            yield json.loads(p.read_text())
        except Exception:
            continue

# -------- Counting --------
def collect_sizes(tasks_iter, rmin=5, rmax=30):
    inp = defaultdict(int)
    out = defaultdict(int)
    for task in tasks_iter:
        # train
        for ex in task.get("train", []):
            A = np.array(ex["input"]);  rA, cA = A.shape
            B = np.array(ex["output"]); rB, cB = B.shape
            if rmin <= rA <= rmax and rmin <= cA <= rmax: inp[(rA, cA)] += 1
            if rmin <= rB <= rmax and rmin <= cB <= rmax: out[(rB, cB)] += 1
        # test
        for ex in task.get("test", []):
            A = np.array(ex["input"]);  rA, cA = A.shape
            if rmin <= rA <= rmax and rmin <= cA <= rmax: inp[(rA, cA)] += 1
            if isinstance(ex.get("output"), list) and len(ex["output"]) > 0:
                B = np.array(ex["output"]); rB, cB = B.shape
                if rmin <= rB <= rmax and rmin <= cB <= rmax: out[(rB, cB)] += 1
    return inp, out

def build_matrix(size_counts, rmin=5, rmax=30):
    rows_vals = list(range(rmin, rmax + 1))
    cols_vals = list(range(rmin, rmax + 1))
    M = np.zeros((len(rows_vals), len(cols_vals)), dtype=int)
    r_index = {r:i for i, r in enumerate(rows_vals)}
    c_index = {c:i for i, c in enumerate(cols_vals)}
    for (r, c), cnt in size_counts.items():
        if rmin <= r <= rmax and rmin <= c <= rmax:
            M[r_index[r], c_index[c]] += cnt
    return M, rows_vals, cols_vals

# -------- Plotting --------
def plot_heatmap(M, rows_vals, cols_vals, title, out_path, vmax):
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 7))
    im = plt.imshow(
        M,
        origin="upper",
        cmap="Reds",
        norm=LogNorm(vmin=1, vmax=max(1, int(vmax)))
    )
    cbar = plt.colorbar(im)
    cbar.set_label("Count (log scale)", fontsize=18)
    cbar.ax.tick_params(labelsize=14)

    xt = [cols_vals.index(x) for x in ODD_TICKS if x in cols_vals]
    yt = [rows_vals.index(y) for y in ODD_TICKS if y in rows_vals]
    plt.xticks(xt, [cols_vals[i] for i in xt], fontsize=14)
    plt.yticks(yt, [rows_vals[i] for i in yt], fontsize=14)

    plt.xlabel("Columns", fontsize=18)
    plt.ylabel("Rows", fontsize=18)
    # Shorter title
    plt.title(title, fontsize=20, fontweight="bold")

    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()
    print(f"[saved] {out_path}")

# -------- Main --------
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Grid-size heatmaps (consistent style)")
    ap.add_argument("--csv", help="Path to CSV with 'data' JSON column")
    ap.add_argument("--json_dir", help="Directory with ARC JSON files")
    ap.add_argument("--generator", help="Filter by generator_name (CSV mode)")
    ap.add_argument("--outdir", default=".")
    args = ap.parse_args()

    assert args.csv or args.json_dir, "Provide --csv or --json_dir"

    tasks = iter_tasks_from_csv(args.csv, args.generator) if args.csv else iter_tasks_from_dir(args.json_dir)
    inp_counts, out_counts = collect_sizes(tasks, MIN_SIZE, MAX_SIZE)

    M_inp, rows_vals, cols_vals = build_matrix(inp_counts, MIN_SIZE, MAX_SIZE)
    M_out, _, _ = build_matrix(out_counts, MIN_SIZE, MAX_SIZE)

    vmax_shared = max(M_inp.max(), M_out.max(), 1)

    outdir = Path(args.outdir)
    plot_heatmap(M_inp, rows_vals, cols_vals,
                 "Input Grid Sizes",
                 outdir / "input_grid_sizes_heatmap.png", vmax_shared)
    plot_heatmap(M_out, rows_vals, cols_vals,
                 "Output Grid Sizes",
                 outdir / "output_grid_sizes_heatmap.png", vmax_shared)
