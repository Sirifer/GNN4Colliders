#!/usr/bin/env python3
"""
Plot the four components of a four-class GNN score stored as a vector branch
in a ROOT TTree.

Example:
    python plot_gnn_score_components.py input.root
    python plot_gnn_score_components.py input.root \
        --tree merged_SR_Sideband \
        --branch ttH_gnn_score \
        --output score_components

The expected class order is assumed to be:
    [0] bkg, [1] ttH, [2] VH, [3] ggF+VBF

Verify this order against the class-label mapping used during training.
"""

import argparse
from pathlib import Path

import numpy as np
import uproot
import matplotlib
matplotlib.use("Agg")  # Suitable for remote clusters without a display.
import matplotlib.pyplot as plt


LABELS = ["bkg", "ttH", "VH", "ggF+VBF"]


def read_score_matrix(tree, branch_name):
    """Read a vector<float> branch and return a numeric array with shape (N, 4)."""
    raw = tree[branch_name].array(library="np")

    # uproot commonly returns std::vector<float> as an object array:
    # one small NumPy array per event.
    if raw.dtype == object or raw.ndim == 1:
        rows = [np.asarray(event, dtype=np.float32).reshape(-1) for event in raw]

        bad = [i for i, row in enumerate(rows) if row.size != 4]
        if bad:
            examples = bad[:10]
            raise ValueError(
                f"Branch '{branch_name}' does not contain exactly 4 values "
                f"for every event. {len(bad)} problematic entries; "
                f"first entry indices: {examples}"
            )

        if not rows:
            return np.empty((0, 4), dtype=np.float32)

        scores = np.vstack(rows)
    else:
        scores = np.asarray(raw, dtype=np.float32)
        if scores.ndim != 2 or scores.shape[1] != 4:
            raise ValueError(
                f"Expected branch shape (N, 4), got {scores.shape}"
            )

    return scores


def main():
    parser = argparse.ArgumentParser(
        description="Plot the four components of a vector-valued GNN score branch."
    )
    parser.add_argument("root_file", help="Input ROOT file")
    parser.add_argument(
        "--tree", default="merged_SR_Sideband",
        help="TTree name (default: merged_SR_Sideband)"
    )
    parser.add_argument(
        "--branch", default="ttH_gnn_score",
        help="Vector score branch (default: ttH_gnn_score)"
    )
    parser.add_argument(
        "--output", default="ttH_gnn_score_components",
        help="Output filename prefix (default: ttH_gnn_score_components)"
    )
    parser.add_argument(
        "--bins", type=int, default=50,
        help="Number of histogram bins (default: 50)"
    )
    parser.add_argument(
        "--weight", default=None,
        help="Optional event-weight branch. If omitted, draw unweighted."
    )
    args = parser.parse_args()

    root_path = Path(args.root_file)
    if not root_path.is_file():
        raise FileNotFoundError(f"ROOT file not found: {root_path}")

    with uproot.open(root_path) as root_file:
        if args.tree not in root_file:
            raise KeyError(
                f"TTree '{args.tree}' not found. Available keys: "
                f"{root_file.keys()}"
            )

        tree = root_file[args.tree]
        if args.branch not in tree:
            raise KeyError(
                f"Branch '{args.branch}' not found in '{args.tree}'. "
                f"Available branches: {tree.keys()}"
            )

        scores = read_score_matrix(tree, args.branch)

        weights = None
        if args.weight:
            if args.weight not in tree:
                raise KeyError(f"Weight branch '{args.weight}' not found.")
            weights = np.asarray(
                tree[args.weight].array(library="np"), dtype=np.float64
            )
            if weights.shape[0] != scores.shape[0]:
                raise ValueError("Weight and score arrays have different lengths.")

    print(f"Input file : {root_path}")
    print(f"TTree      : {args.tree}")
    print(f"Branch     : {args.branch}")
    print(f"Events     : {scores.shape[0]}")
    print(f"Score shape: {scores.shape}")

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    axes = axes.flatten()

    for i, (label, ax) in enumerate(zip(LABELS, axes)):
        values = scores[:, i]
        finite = np.isfinite(values)
        values_finite = values[finite]

        if weights is not None:
            valid = finite & np.isfinite(weights)
            values_to_plot = values[valid]
            weights_to_plot = weights[valid]
        else:
            values_to_plot = values_finite
            weights_to_plot = None

        ax.hist(
            values_to_plot,
            bins=args.bins,
            range=(0.0, 1.0),
            weights=weights_to_plot,
            histtype="step",
            linewidth=1.5,
        )
        ax.set_title(f"{label} score (index {i})")
        ax.set_xlabel("Score")
        ax.set_ylabel("Weighted events" if weights is not None else "Events")
        ax.set_xlim(0.0, 1.0)
        ax.grid(True, alpha=0.25)

        if values_finite.size:
            outside = np.count_nonzero(
                (values_finite < 0.0) | (values_finite > 1.0)
            )
            print(
                f"[{i}] {label:8s}: entries={values_finite.size}, "
                f"mean={np.mean(values_finite):.6f}, "
                f"std={np.std(values_finite):.6f}, "
                f"min={np.min(values_finite):.6f}, "
                f"max={np.max(values_finite):.6f}, "
                f"outside [0,1]={outside}"
            )
        else:
            print(f"[{i}] {label:8s}: no finite entries")

    fig.suptitle(
        f"{args.branch} from {args.tree}\n{root_path.name}",
        fontsize=13
    )
    fig.tight_layout()

    output_prefix = Path(args.output)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    png_path = output_prefix.with_suffix(".png")
    pdf_path = output_prefix.with_suffix(".pdf")

    fig.savefig(png_path, dpi=160, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    print(f"\nSaved:\n  {png_path}\n  {pdf_path}")



if __name__ == "__main__":
    main()
