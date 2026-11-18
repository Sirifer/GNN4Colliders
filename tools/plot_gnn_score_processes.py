#!/usr/bin/env python3
"""
Overlay the four components of the 4-class GNN score for several true MC
processes (Signal/VH, Signal/ttH, Signal/ggF, Signal/VBF, one Bkg_MC sample).

Example:
    python plot_gnn_score_processes.py --era 2025
    python plot_gnn_score_processes.py --era 2025 \
        --bkg Bkg_MC/DYJetsToLL_Stitch --weight weight_lumi \
        --output plots/gnn_score_2025_stitch

The score branch is the vector<float> branch written by inference.py:
    [0] bkg, [1] ttH, [2] VH, [3] ggF+VBF
"""

import argparse
from pathlib import Path

import numpy as np
import uproot
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path("/scratchfs2/cms/huangsuyun/GNN4Colliders")
SCORE_DIR = REPO / "root_p2root_v3_pt_only_scored_ttH_4class_4fold_minloss_fold{fold}"
LABELS = ["bkg", "ttH", "VH", "ggF+VBF"]
DEFAULT_SIGNAL = ["Signal/ttH", "Signal/VH", "Signal/ggF", "Signal/VBF"]
COLORS = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#8c564b", "#e377c2"]


def read_score_matrix(tree, branch_name):
    """Read a vector<float> branch and return an array with shape (N, 4)."""
    raw = tree[branch_name].array(library="np")
    if raw.dtype == object or raw.ndim == 1:
        rows = [np.asarray(event, dtype=np.float32).reshape(-1) for event in raw]
        bad = [i for i, row in enumerate(rows) if row.size != 4]
        if bad:
            raise ValueError(
                f"Branch '{branch_name}' does not contain exactly 4 values "
                f"for every event. {len(bad)} bad entries; first: {bad[:10]}"
            )
        if not rows:
            return np.empty((0, 4), dtype=np.float32)
        return np.vstack(rows)
    scores = np.asarray(raw, dtype=np.float32)
    if scores.ndim != 2 or scores.shape[1] != 4:
        raise ValueError(f"Expected branch shape (N, 4), got {scores.shape}")
    return scores


def load_process(path, tree_name, branch, weight_branch, region):
    with uproot.open(path) as root_file:
        if tree_name not in root_file:
            raise KeyError(f"TTree '{tree_name}' not in {path}")
        tree = root_file[tree_name]
        if branch not in tree:
            raise KeyError(f"Branch '{branch}' not in {tree_name} of {path}")
        scores = read_score_matrix(tree, branch)
        weights = None
        if weight_branch:
            weights = np.asarray(
                tree[weight_branch].array(library="np"), dtype=np.float64
            )
        mask = np.ones(scores.shape[0], dtype=bool)
        if region != "all":
            tag = np.asarray(tree["source_tree_tag"].array(library="np"))
            is_sr = (tag % 2) == 0
            mask = is_sr if region == "sr" else ~is_sr
            scores = scores[mask]
            if weights is not None:
                weights = weights[mask]
    return scores, weights


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--era", default="2025")
    parser.add_argument("--tree", default="merged_SR_Sideband")
    parser.add_argument("--branch", default="ttH_gnn_score")
    parser.add_argument("--signal", nargs="+", default=DEFAULT_SIGNAL,
                        help="Signal process directories to overlay")
    parser.add_argument("--bkg", default="Bkg_MC/TT",
                        help="One background process directory")
    parser.add_argument("--weight", default=None,
                        help="Optional event-weight branch, e.g. weight_lumi")
    parser.add_argument("--region", choices=["all", "sr", "sb"], default="all")
    parser.add_argument("--bins", type=int, default=50)
    parser.add_argument("--normalize", dest="normalize", action="store_true",
                        default=True, help="Normalize each curve to unit area")
    parser.add_argument("--no-normalize", dest="normalize", action="store_false")
    parser.add_argument("--output", default=None,
                        help="Output prefix (default: plots/gnn_score_<era>_processes)")
    parser.add_argument("--component", default="all", choices=["all"] + LABELS,
                        help="Which score component to draw (default: all four)")
    parser.add_argument("--per-process", action="store_true",
                        help="Write one figure per process instead of overlaying them")
    parser.add_argument("--norm-max", action="store_true",
                        help="Scale each curve so its highest bin is 1 (y axis 0-1)")
    parser.add_argument("--ylim", nargs=2, type=float, default=None,
                        metavar=("YMIN", "YMAX"), help="Explicit y-axis range")
    args = parser.parse_args()

    base = Path(str(SCORE_DIR).format(fold=args.fold))
    processes = list(args.signal) + [args.bkg]
    output = Path(args.output) if args.output else \
        REPO / "plots" / f"gnn_score_{args.era}_processes"

    data = {}
    for proc in processes:
        path = base / proc / f"{args.era}.root"
        if not path.is_file():
            raise FileNotFoundError(path)
        scores, weights = load_process(
            path, args.tree, args.branch, args.weight, args.region
        )
        data[proc] = (scores, weights)
        print(f"{proc:36s} events={scores.shape[0]}", flush=True)

    comp_indices = (
        list(range(len(LABELS))) if args.component == "all"
        else [LABELS.index(args.component)]
    )

    def draw(ax, ci, procs, show_legend):
        label = LABELS[ci]
        for k, proc in enumerate(procs):
            scores, weights = data[proc]
            values = scores[:, ci]
            finite = np.isfinite(values)
            vals = values[finite]
            wts = None
            if weights is not None:
                wts = weights[finite]
                finite_w = np.isfinite(wts)
                vals, wts = vals[finite_w], wts[finite_w]
            if vals.size == 0:
                continue
            plot_weights = wts
            density = args.normalize
            if args.norm_max:
                counts, _ = np.histogram(
                    vals, bins=args.bins, range=(0.0, 1.0), weights=wts
                )
                peak = float(np.max(counts)) if counts.size else 0.0
                base = wts if wts is not None else np.ones(vals.shape[0])
                plot_weights = base / peak if peak > 0.0 else base
                density = False
            ax.hist(
                vals,
                bins=args.bins,
                range=(0.0, 1.0),
                weights=plot_weights,
                density=density,
                histtype="step",
                linewidth=1.6,
                color=COLORS[k % len(COLORS)],
                label=proc.split("/")[-1],
            )
            print(
                f"[{label:8s}] {proc:36s} N={vals.size:7d} "
                f"mean={np.mean(vals):.4f} std={np.std(vals):.4f}",
                flush=True,
            )
        ax.set_title(f"{label} score (index {ci})")
        ax.set_xlabel("Score")
        if args.norm_max:
            ylabel = "Max-normalized"
        elif args.normalize:
            ylabel = "Normalized"
        elif args.weight:
            ylabel = "Weighted events"
        else:
            ylabel = "Entries"
        ax.set_ylabel(ylabel)
        ax.set_xlim(0.0, 1.0)
        ax.grid(True, alpha=0.25)
        if args.ylim is not None:
            ax.set_ylim(*args.ylim)
        elif args.norm_max:
            ax.set_ylim(0.0, 1.05)
        if show_legend and len(procs) > 1:
            ax.legend(fontsize=9, loc="best")

    output.parent.mkdir(parents=True, exist_ok=True)
    saved = []
    n_comp = len(comp_indices)

    if args.per_process:
        for proc in processes:
            if n_comp == 1:
                fig, ax = plt.subplots(figsize=(7.5, 5.5))
                axes_list = [ax]
            else:
                fig, axes = plt.subplots(2, 2, figsize=(11, 9))
                axes_list = axes.flatten()
            for ax, ci in zip(axes_list, comp_indices):
                draw(ax, ci, [proc], show_legend=False)
            short = proc.split("/")[-1]
            fig.suptitle(
                f"{short} ({args.era}), fold {args.fold}, region={args.region}"
                + (f", weight={args.weight}" if args.weight else ", unweighted"),
                fontsize=12,
            )
            fig.tight_layout()
            prefix = Path(f"{output}_{short}")
            for suffix in (".png", ".pdf"):
                path = prefix.with_suffix(suffix)
                fig.savefig(path, dpi=160, bbox_inches="tight")
                saved.append(path)
            plt.close(fig)
    else:
        if n_comp == 1:
            fig, ax = plt.subplots(figsize=(8, 5.5))
            axes_list = [ax]
        else:
            fig, axes = plt.subplots(2, 2, figsize=(13, 10))
            axes_list = axes.flatten()
        for j, (ax, ci) in enumerate(zip(axes_list, comp_indices)):
            draw(ax, ci, processes, show_legend=(j == 0))
        fig.suptitle(
            f"4-class GNN score components, {args.era} era, fold {args.fold}, "
            f"region={args.region}"
            + (f", weight={args.weight}" if args.weight else ", unweighted"),
            fontsize=13,
        )
        fig.tight_layout()
        for suffix in (".png", ".pdf"):
            path = output.with_suffix(suffix)
            fig.savefig(path, dpi=160, bbox_inches="tight")
            saved.append(path)
        plt.close(fig)

    print("\nSaved:")
    for path in saved:
        print(f"  {path}")


if __name__ == "__main__":
    main()
