#!/usr/bin/env python3
"""Build 2-fold OOF MC, optimize GNN cuts, and export ttH signal yields.

Install: python -m pip install numpy uproot optuna
Run: python prepare_gnn_categories.py --help

The optimization metric uses OOF MC signal/background in all three score
categories. Data sidebands only enforce a nonempty high category. The exported
S values are ttH MC yields in 120 < m_llg < 130 GeV.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import optuna
import uproot


BASE = Path("/scratchfs2/cms/huangsuyun/GNN4Colliders")
DEFAULT_FOLD0 = BASE / "root_p2root_v2_pt_only_scored_ttH_vs_other_run3_fold0"
DEFAULT_FOLD1 = BASE / "root_p2root_v2_pt_only_scored_ttH_vs_other_run3_fold1"
DEFAULT_OUT = BASE / "tools/results_sideband_fit/gnn_category_result.json"

TREE = "merged_SR_Sideband"
WEIGHT, MASS, SCORE = "weight_lumi", "m_llg", "ttH_gnn_score"
IS_SR, TAG, FOLD = "is_SR", "source_tree_tag", "photon_eta_fold"
MC_BRANCHES = [WEIGHT, MASS, SCORE, IS_SR, TAG, FOLD]
DATA_BRANCHES = [WEIGHT, SCORE, IS_SR, TAG]
SR_TAGS = np.array([0, 2, 4, 6, 8])
SB_TAGS = SR_TAGS + 1
YEARS = ["2022preEE", "2022postEE", "2023preBPix", "2023postBPix", "2024", "2025"]
BACKGROUND_GROUPS = [
    "Signal/VH", "Signal/ggF", "Signal/VBF",
    "Bkg_MC/TT", "Bkg_MC/DYGto2LG_10to100",
]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--fold0", type=Path, default=DEFAULT_FOLD0)
    p.add_argument("--fold1", type=Path, default=DEFAULT_FOLD1)
    p.add_argument("--data-dir", type=Path, default=None,
                   help="Scored Data directory; default: <fold0>/Data")
    p.add_argument("--output", type=Path, default=DEFAULT_OUT)
    p.add_argument("--trials", type=int, default=1000)
    p.add_argument("--seed", type=int, default=12345)
    p.add_argument("--min-b-neff", type=float, default=0.0)
    p.add_argument("--min-high-sr-yield", type=float, default=0.0)
    p.add_argument("--min-high-data-sb-yield", type=float, default=0.0)
    return p.parse_args()


def read_tree(path, branches):
    with uproot.open(path) as root:
        if TREE not in root:
            raise RuntimeError(f"Missing tree {TREE}: {path}")
        tree = root[TREE]
        missing = set(branches) - set(tree.keys())
        if missing:
            raise RuntimeError(f"Missing branches {sorted(missing)}: {path}")
        return tree.arrays(branches, library="np")


def read_oof_group(fold0, fold1, group):
    parts = {key: [] for key in (WEIGHT, SCORE)}
    n_files = 0
    for directory, parity in ((fold0, 0), (fold1, 1)):
        for year in YEARS:
            path = directory / group / f"{year}.root"
            if not path.is_file():
                print(f"Missing MC file: {path}")
                continue
            a = read_tree(path, MC_BRANCHES)
            mask = (
                (a[FOLD] % 2 == parity)
                & (a[IS_SR] == 1)
                & np.isin(a[TAG], SR_TAGS)
                & (a[MASS] > 120.0) & (a[MASS] < 130.0)
                & np.isfinite(a[WEIGHT]) & np.isfinite(a[SCORE])
            )
            for key in parts:
                parts[key].append(np.asarray(a[key][mask], dtype=float))
            n_files += 1
    if not n_files:
        raise RuntimeError(f"No MC files found for {group}")
    out = {key: np.concatenate(value) for key, value in parts.items()}
    print(f"OOF {group}: {len(out[WEIGHT])} selected events from {n_files} files")
    return out


def read_data_sideband(data_dir):
    files = sorted(data_dir.rglob("*.root"))
    if not files:
        raise RuntimeError(f"No Data ROOT files under {data_dir}")
    weights, scores = [], []
    for path in files:
        a = read_tree(path, DATA_BRANCHES)
        # The Data are read once from fold0, never OOF-split or double-counted.
        # Keep the original optimization constraint: all is_SR==0 Data rows.
        # The mass window for the later fit is applied only by script 2.
        mask = ((a[IS_SR] == 0)
                & np.isfinite(a[WEIGHT]) & np.isfinite(a[SCORE]))
        weights.append(np.asarray(a[WEIGHT][mask], dtype=float))
        scores.append(np.asarray(a[SCORE][mask], dtype=float))
    weights, scores = np.concatenate(weights), np.concatenate(scores)
    if len(weights) == 0:
        raise RuntimeError("No valid Data sideband events with is_SR == 0")
    print(f"Data sideband: {len(weights)} unweighted events; weighted sum={weights.sum():.6g}")
    return weights, scores


def yield_info(weights):
    w = np.asarray(weights, dtype=float)
    total = float(w.sum())
    sumw2 = float(np.dot(w, w))
    return {
        "S": total,
        "sigma_S": float(np.sqrt(sumw2)),
        "n": int(len(w)),
        "nneg": int(np.count_nonzero(w < 0)),
        "neff": total * total / sumw2 if sumw2 > 0 else 0.0,
    }


def asimov_z(s, b):
    if s <= 0 or b <= 0:
        return 0.0
    return float(np.sqrt(max(0.0, 2 * ((s + b) * np.log1p(s / b) - s))))


def evaluate(signal, background, data_weights, data_scores, c1, c2, args):
    if not (0.10 <= c1 <= 0.95 and max(0.20, c1 + 0.001) <= c2 <= 0.999):
        return None
    masks = {
        "high": lambda score: score > c2,  # Critical: high uses c2.
        "medium": lambda score: (score > c1) & (score <= c2),
        "low": lambda score: score <= c1,
    }
    result = {}
    for name, select in masks.items():
        sw = signal[WEIGHT][select(signal[SCORE])]
        bw = background[WEIGHT][select(background[SCORE])]
        s, b = yield_info(sw), yield_info(bw)
        data_yield = float(data_weights[select(data_scores)].sum())
        if b["S"] <= 0 or b["neff"] < args.min_b_neff:
            return None
        if name == "high":
            if s["S"] + b["S"] < args.min_high_sr_yield:
                return None
            if data_yield <= args.min_high_data_sb_yield:
                return None
        result[name] = {"S": s, "B": b, "data_sb_weighted": data_yield,
                        "Z_MC": asimov_z(s["S"], b["S"])}
    result["Z_MC_combined"] = float(np.sqrt(sum(result[n]["Z_MC"] ** 2 for n in masks)))
    return result


def main():
    args = parse_args()
    if args.trials < 1:
        raise ValueError("--trials must be positive")
    signal = read_oof_group(args.fold0, args.fold1, "Signal/ttH")
    if not len(signal[WEIGHT]):
        raise RuntimeError("OOF ttH signal selection is empty")
    bkg_parts = [read_oof_group(args.fold0, args.fold1, g) for g in BACKGROUND_GROUPS]
    background = {k: np.concatenate([p[k] for p in bkg_parts]) for k in (WEIGHT, SCORE)}
    if not len(background[WEIGHT]):
        raise RuntimeError("OOF background selection is empty")
    data_dir = args.data_dir or args.fold0 / "Data"
    data_weights, data_scores = read_data_sideband(data_dir)

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    sampler = optuna.samplers.TPESampler(seed=args.seed, n_startup_trials=min(50, args.trials))
    study = optuna.create_study(direction="maximize", sampler=sampler)

    def objective(trial):
        c1 = trial.suggest_float("c1", 0.10, 0.95)
        c2 = trial.suggest_float("c2", max(0.20, c1 + 0.001), 0.999)
        result = evaluate(signal, background, data_weights, data_scores, c1, c2, args)
        trial.set_user_attr("valid", result is not None)
        return result["Z_MC_combined"] if result is not None else -1.0

    study.optimize(objective, n_trials=args.trials, n_jobs=1)
    valid = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE
             and t.user_attrs.get("valid")]
    if not valid:
        raise RuntimeError("No valid cut pair found; inspect MC/Data selections and constraints")
    best = max(valid, key=lambda t: t.value)
    c1, c2 = float(best.params["c1"]), float(best.params["c2"])
    detail = evaluate(signal, background, data_weights, data_scores, c1, c2, args)
    payload = {
        "c1": c1, "c2": c2,
        "high": {k: detail["high"]["S"][k] for k in ("S", "sigma_S", "n", "nneg", "neff")},
        "medium": {k: detail["medium"]["S"][k] for k in ("S", "sigma_S", "n", "nneg", "neff")},
        "metadata": {
            "selection": "is_SR == 1, valid even SR tag, 120 < m_llg < 130",
            "high_cut": "score > c2", "medium_cut": "c1 < score <= c2",
            "oof": "fold0 parity 0; fold1 parity 1",
            "optimization": "maximize quadrature sum of high/medium/low MC Asimov Z",
            "Z_MC_combined": detail["Z_MC_combined"],
            "valid_trials": len(valid), "trials": args.trials,
            "fold0": str(args.fold0), "fold1": str(args.fold1), "data_dir": str(data_dir),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    print(f"Best c1={c1:.6f}, c2={c2:.6f}; MC combined Z={detail['Z_MC_combined']:.6f}")
    for name in ("high", "medium"):
        x = payload[name]
        print(f"{name}: S={x['S']:.6g} ± {x['sigma_S']:.6g} (n={x['n']})")
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
