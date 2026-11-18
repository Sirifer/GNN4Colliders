#!/usr/bin/env python3
"""Four-fold cut-based Asimov significance using the 4-class samples.

Twin of `significance_4class_4fold_cutbased.py`: same scored directories, same
SR/SB definition, same folds, same statistic. **The only difference is the
sideband scale factor:**

    significance_4fold_cutbase.py          ONE common alpha per fold  (this file)
    significance_4class_4fold_cutbased.py  ONE alpha per category

    SR:      is_SR == 1, split by `source_tree_tag`
             0=ttHhad, 2=ttHlep, 4=untag, 6=VH3l, 8=VHpTmiss
    SB:      is_SR == 0 and the matching odd tag (tag + 1); alpha only
    alpha:   N(Data SB) / Y(Bkg MC SB)   (Data counted, MC weighted)
    B_i:     alpha * Y(Bkg MC SR, i)
    Z_i:     Asimov(S_i, B_i)

No mass window is applied: `is_SR` already carries the mass selection, and an
`m_llg` window would select a different, smaller event set.

Folds: fold k -> test parity k (the four test parities are disjoint).
Validation is not used: no cut is tuned here.
"""

import argparse
import csv
from pathlib import Path

import numpy as np
import uproot

REPO = Path("/scratchfs2/cms/huangsuyun/GNN4Colliders")
MC_BASE_NAME = "root_p2root_v3_pt_only_scored_ttH_4class_4fold_fold"
TREE = "merged_SR_Sideband"
WEIGHT_DEFAULT = "weight_lumi_avg"

SIGNAL_DIRS = ["Signal/ttH", "Signal/VH", "Signal/ggF", "Signal/VBF"]
BKG_DIRS = [
   "Bkg_MC/TT", "Bkg_MC/DYGto2LG_10to100",
   "Bkg_MC/DYGto2LG_100to200", "Bkg_MC/DYGto2LG_200to400",
   "Bkg_MC/DYGto2LG_400to600", "Bkg_MC/DYGto2LG_600",
   "Bkg_MC/DYJetsToLL_Stitch",
]
ERAS = [
   "2016preVFP", "2016postVFP", "2017", "2018", "2022preEE",
   "2022postEE", "2023preBPix", "2023postBPix", "2024", "2025", "2026",
]
SR_TAGS = {0: "ttHhad", 2: "ttHlep", 4: "untag", 6: "VH3l", 8: "VHpTmiss"}


def yield_info(weights):
   w = np.asarray(weights, dtype=np.float64)
   if w.size == 0:
       return {"sum": 0.0, "err": 0.0, "n": 0, "nneg": 0, "neff": 0.0}
   if not np.all(np.isfinite(w)):
       raise ValueError("Non-finite MC weights found")
   sw = float(np.sum(w))
   sw2 = float(np.sum(w * w))
   return {"sum": sw, "err": float(np.sqrt(sw2)), "n": int(w.size),
           "nneg": int(np.sum(w < 0.0)),
           "neff": float(sw * sw / sw2) if sw2 > 0.0 else 0.0}


def asimov_z(s, b):
   if not np.isfinite(s) or not np.isfinite(b) or s <= 0.0 or b <= 0.0:
       return 0.0
   value = 2.0 * ((s + b) * np.log1p(s / b) - s)
   return float(np.sqrt(max(value, 0.0)))


def z_uncertainty(s, b, sigma_s, sigma_b, z):
   if z <= 0.0 or s <= 0.0 or b <= 0.0:
       return 0.0
   log_term = np.log1p(s / b)
   return float(np.hypot(log_term / z * sigma_s,
                         (log_term - s / b) / z * sigma_b))


def load_process(base, subdir, weight_branch):
   """Load one process: is_SR, source_tree_tag, fold variable and weight."""
   process_dir = base / subdir
   if not process_dir.is_dir():
       raise FileNotFoundError(f"Missing sample directory: {process_dir}")
   branches = ["is_SR", "source_tree_tag", "photon_eta_fold", weight_branch]
   parts, loaded = [], []
   for era in ERAS:
       path = process_dir / f"{era}.root"
       if not path.exists():
           continue
       with uproot.open(path) as root_file:
           if TREE not in root_file:
               raise KeyError(f"{TREE!r} not found in {path}")
           tree = root_file[TREE]
           missing = [b for b in branches if b not in tree.keys()]
           if missing:
               raise KeyError(f"Missing branches {missing} in {path}")
           arr = tree.arrays(branches, library="np")
       parts.append(tuple(np.asarray(arr[b]) for b in branches))
       loaded.append(era)
   if not parts:
       raise RuntimeError(f"No ROOT files found for {subdir} under {base}")
   return tuple(np.concatenate([p[i] for p in parts]) for i in range(4)), loaded


def merge(parts):
   return tuple(np.concatenate([p[i] for p in parts]) for i in range(4))


def load_fold(fold, weight_branch):
   base = REPO / f"{MC_BASE_NAME}{fold}"
   signal_parts, bkg_parts = [], []
   for subdir in SIGNAL_DIRS:
       signal_parts.append(load_process(base, subdir, weight_branch)[0])
   for subdir in BKG_DIRS:
       bkg_parts.append(load_process(base, subdir, weight_branch)[0])
   return merge(signal_parts), merge(bkg_parts)


def select_test_parity(arrays, parity):
   is_sr, tag, foldvar, weight = arrays
   mask = ((is_sr == 1) & np.isin(tag, list(SR_TAGS))
           & ((foldvar % 4) == parity) & np.isfinite(weight))
   return tuple(a[mask] for a in arrays)



def evaluate(signal, background):
    _, signal_tag, _, signal_weight = signal
    _, bkg_tag, _, bkg_weight = background
    results = {}
    for tag, name in SR_TAGS.items():
        s = yield_info(signal_weight[signal_tag == tag])
        b = yield_info(bkg_weight[bkg_tag == tag])
        z = asimov_z(s["sum"], b["sum"])
        results[name] = {"tag": tag, "S": s, "B": b, "Z": z,
                         "sigma_Z": z_uncertainty(s["sum"], b["sum"],
                                                     s["err"], b["err"], z)}
    return results


def combined_z(results):
    z = float(np.sqrt(sum(r["Z"] ** 2 for r in results.values())))
    if z <= 0.0:
        return 0.0, 0.0
    sigma = float(np.sqrt(sum((r["Z"] / z * r["sigma_Z"]) ** 2
                              for r in results.values())))
    return z, sigma


def print_table(title, results):
    print(f"\n{title}")
    print("-" * 100)
    print(f"{Category:<12}{S:>14}{sigmaS:>14}{B:>14}{sigmaB:>14}{N_S:>10}{N_B:>10}{Z:>12}")
    print("-" * 100)
    for name, r in results.items():
        print(f"{name:<12}{r[S][sum]:>14.6f}{r[S][err]:>14.6f}"
              f"{r[B][sum]:>14.6f}{r[B][err]:>14.6f}"
              f"{r[S][n]:>10d}{r[B][n]:>10d}{r[Z]:>12.6f}")
    z, sigma = combined_z(results)
    print("-" * 100)
    print(f"Combined Z = {z:.6f} +/- {sigma:.6f}")


DATA_BASE_NAME = "root_p2root_v3_pt_only_scored_ttH_4class_4fold_data_fold"


def load_data(fold):
    base = REPO / f"{DATA_BASE_NAME}{fold}" / "Data" / "Data"
    parts = []
    branches = ["is_SR", "source_tree_tag", "photon_eta_fold"]
    for era in ERAS:
        path = base / f"{era}.root"
        if not path.exists():
            continue
        with uproot.open(path) as root_file:
            tree = root_file[TREE]
            arr = tree.arrays(branches, library="np")
        parts.append(tuple(np.asarray(arr[b]) for b in branches))
    if not parts:
        raise RuntimeError(f"No Data ROOT files found under {base}")
    return tuple(np.concatenate([x[i] for x in parts]) for i in range(3))


def select_data_parity(arrays, parity):
    is_sr, tag, foldvar = arrays
    mask = ((is_sr == 1) & np.isin(tag, list(SR_TAGS))
            & ((foldvar % 4) == parity))
    return tuple(a[mask] for a in arrays)


def sr_scale_factor(data, background):
    data_n = len(data[0])
    bkg_y = yield_info(background[3])
    if data_n <= 0 or bkg_y["sum"] <= 0.0:
        raise RuntimeError(f"Invalid SR scale factor inputs: Ndata={data_n}, Ybkg={bkg_y['sum']}")
    alpha = data_n / bkg_y["sum"]
    sigma_alpha = alpha * np.sqrt(1.0 / data_n + (bkg_y["err"] / bkg_y["sum"]) ** 2)
    return {"N_data_sr": data_n, "Y_bkg_sr": bkg_y["sum"],
            "sigma_Y_bkg_sr": bkg_y["err"], "alpha": float(alpha),
            "sigma_alpha": float(sigma_alpha)}


def evaluate_scaled(signal, background, norm):
    _, signal_tag, _, signal_weight = signal
    _, bkg_tag, _, bkg_weight = background
    results = {}
    for tag, name in SR_TAGS.items():
        s = yield_info(signal_weight[signal_tag == tag])
        bmc = yield_info(bkg_weight[bkg_tag == tag])
        b = norm["alpha"] * bmc["sum"]
        sigma_b = float(np.hypot(norm["alpha"] * bmc["err"],
                                 bmc["sum"] * norm["sigma_alpha"]))
        z = asimov_z(s["sum"], b)
        results[name] = {"tag": tag, "S": s, "B_mc": bmc, "B": b,
                         "Z": z, "sigma_Z": z_uncertainty(s["sum"], b,
                                                              s["err"], sigma_b, z),
                         "sigma_B": sigma_b}
    return results


def combined_scaled(results):
    z = float(np.sqrt(sum(r["Z"] ** 2 for r in results.values())))
    if z <= 0.0:
        return 0.0, 0.0
    return z, float(np.sqrt(sum((r["Z"] / z * r["sigma_Z"]) ** 2
                                for r in results.values())))



def select_sideband(arrays, parity):
    is_sr, tag, foldvar, weight = arrays
    sb_tags = [x + 1 for x in SR_TAGS]
    mask = ((is_sr == 0) & np.isin(tag, sb_tags)
            & ((foldvar % 4) == parity) & np.isfinite(weight))
    return tuple(a[mask] for a in arrays)


def sideband_scale_factor(data_sb, bkg_sb):
    n_data = len(data_sb[0])
    bkg = yield_info(bkg_sb[3])
    if n_data <= 0 or bkg["sum"] <= 0.0:
        raise RuntimeError("Invalid sideband scale-factor inputs")
    alpha = n_data / bkg["sum"]
    sigma_alpha = alpha * np.sqrt(1.0 / n_data + (bkg["err"] / bkg["sum"]) ** 2)
    return {"N_data_sb": n_data, "Y_bkg_sb": bkg["sum"],
            "sigma_Y_bkg_sb": bkg["err"], "alpha": float(alpha),
            "sigma_alpha": float(sigma_alpha)}


def select_data_sideband(arrays, parity):
    is_sr, tag, foldvar = arrays
    sb_tags = [x + 1 for x in SR_TAGS]
    mask = ((is_sr == 0) & np.isin(tag, sb_tags)
            & ((foldvar % 4) == parity))
    return tuple(a[mask] for a in arrays)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weight-branch", choices=["weight_lumi_avg", "weight_lumi"], default=WEIGHT_DEFAULT)
    parser.add_argument("--diagnostic-only", action="store_true")
    args = parser.parse_args()
    print("4-fold cut-based significance: all-Higgs signal, SR-only Z with SB scale factor")
    print("alpha = N(Data, SB) / Y(Bkg MC, SB); S/B are evaluated only in SR")
    print("No m_llg mass window is applied.")
    fold_results, csv_rows = [], []
    for fold in range(4):
        print("\nLoading fold %d; test parity = %d" % (fold, fold))
        sig_all, bkg_all = load_fold(fold, args.weight_branch)
        data_all = load_data(fold)
        sig_sr = select_test_parity(sig_all, fold)
        bkg_sr = select_test_parity(bkg_all, fold)
        data_sb = select_data_sideband(data_all, fold)
        bkg_sb = select_sideband(bkg_all, fold)
        norm = sideband_scale_factor(data_sb, bkg_sb)
        results = evaluate_scaled(sig_sr, bkg_sr, norm)
        fold_results.append((results, norm))
        print("  selected SR: signal=%d, background=%d; SB data=%d, SB MC=%d" %
              (len(sig_sr[0]), len(bkg_sr[0]), len(data_sb[0]), len(bkg_sb[0])))
        print("  alpha = %.8f +/- %.8f" % (norm["alpha"], norm["sigma_alpha"]))
        print("%-12s%14s%16s%12s" % ("Category", "S", "B=alpha*BMC", "Z"))
        for name, r in results.items():
            print("%-12s%14.6f%16.6f%12.6f" % (name, r["S"]["sum"], r["B"], r["Z"]))
            csv_rows.append({"fold": fold, "category": name, "source_tree_tag": r["tag"],
                             "N_data_sb": norm["N_data_sb"], "Y_bkg_sb": norm["Y_bkg_sb"],
                             "alpha": norm["alpha"], "sigma_alpha": norm["sigma_alpha"],
                             "S": r["S"]["sum"], "sigma_S": r["S"]["err"],
                             "B_mc_sr": r["B_mc"]["sum"], "B": r["B"],
                             "sigma_B": r["sigma_B"], "Z": r["Z"], "sigma_Z": r["sigma_Z"]})
    fold_z = [combined_scaled(r) for r, _ in fold_results]
    z_final = float(np.sqrt(sum(z * z for z, _ in fold_z)))
    sigma_final = float(np.sqrt(sum((z / z_final * s) ** 2 for z, s in fold_z))) if z_final > 0 else 0.0
    print("\n4-FOLD SUMMARY")
    for fold, (z, s) in enumerate(fold_z):
        print("fold %d: Z = %.6f +/- %.6f" % (fold, z, s))
    print("4-fold combined Z = %.6f +/- %.6f" % (z_final, sigma_final))
    if args.diagnostic_only:
        return
    out_dir = REPO / "tools" / "results_4fold_cutbase"
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "final_z.txt").open("w", encoding="utf-8") as h:
        h.write("4-fold cut-based SR-only significance with SB scale factor\n")
        h.write("alpha = N(Data, SB) / Y(Bkg MC, SB); one COMMON alpha per fold\n")
        h.write("Twin of significance_4class_4fold_cutbased.py (per-category "
                "alpha); only the alpha convention differs.\n")
        h.write("SR: is_SR == 1; SB: is_SR == 0 (matching odd tag); "
                "no mass window\n")
        h.write("Folds: fold k -> test parity k; validation not used\n")
        for fold, (z, s) in enumerate(fold_z):
            h.write("fold %d: Z = %.6f +/- %.6f\n" % (fold, z, s))
        h.write("4-fold combined Z = %.6f +/- %.6f\n" % (z_final, sigma_final))
    with (out_dir / "category_yields.csv").open("w", newline="", encoding="utf-8") as h:
        writer = csv.DictWriter(h, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)
    print("Report: %s" % (out_dir / "final_z.txt"))


if __name__ == "__main__":
    main()
