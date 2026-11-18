#!/usr/bin/env python3
"""
Study DYJetsToLL_Stitch statistics over ALL year samples (2016-2026).

For every year root file in
    /publicfs/cms/user/laipeizhu/Eos/HZgamma/root_p2root_v3_Mll80to100/
        Bkg_MC/DYJetsToLL_Stitch/
it reports, per tree (SR / Sideband / CR_* families) and aggregated:

  * N events (raw) per tree per year
  * N events after |weight| selection where relevant
  * sum(weight_lumi)            -> nominal luminosity-scaled yield
  * sum(weight_central)         -> pre-luminosity yield (sanity)
  * sum of positive / negative / abs weights
  * N_eff = (sum w)^2 / sum(w^2)
  * weight_lumi distribution: min / p1 / median / p99 / max / mean / std
  * "large weight" census:
        count & fractional contribution of events whose |weight_lumi|
        exceeds several relative thresholds (top-0.1%, top-1%, top-5%,
        plus absolute thresholds >1e0, >1e-1, >1e-2).
  * top 20 events by |weight_lumi| with run/lumi/event id.

The nominal analysis weight is `weight_lumi` (matches check_yield.py /
check_run3_yields_10trees.py convention).  `weight_lumi` can be negative
because the DY stitch is NLO (a subset of events carry negative genWeight).

Run:
    python3 study_dyjets_stats.py
Output:
    * stdout summary table
    * dyjets_stats_summary.txt   (human-readable full report)
    * dyjets_weight_hist.png     (weight_lumi distribution per year, log y)
"""

from pathlib import Path
from collections import defaultdict
import numpy as np
import uproot
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA_DIR = Path(
    "/publicfs/cms/user/laipeizhu/Eos/HZgamma/"
    "root_p2root_v3_Mll80to100/Bkg_MC/DYJetsToLL_Stitch"
)

# Nominal weight used by the analysis toolchain.
WEIGHT = "weight_lumi"
WEIGHT_CENTRAL = "weight_central"   # pre-luminosity, sanity check
ID_BRANCHES = ["run", "lumi", "event"]

# Year ordering (Run2 -> Run3 -> 2024/25/26).
YEAR_ORDER = [
    "2016preVFP", "2016postVFP",
    "2017", "2018",
    "2022preEE", "2022postEE",
    "2023preBPix", "2023postBPix",
    "2024", "2025", "2026",
]
RUN2 = ["2016preVFP", "2016postVFP", "2017", "2018"]
RUN3 = ["2022preEE", "2022postEE", "2023preBPix", "2023postBPix"]
RUN_LATE = ["2024", "2025", "2026"]

OUTPUT_TXT = Path(__file__).with_name("dyjets_stats_summary.txt")
OUTPUT_PNG = Path(__file__).with_name("dyjets_weight_hist.png")


def list_years():
    found = []
    for y in YEAR_ORDER:
        p = DATA_DIR / f"{y}.root"
        if p.exists():
            found.append(y)
    return found


def list_trees(year):
    p = DATA_DIR / f"{year}.root"
    f = uproot.open(p)
    return sorted(f.keys(cycle=False))


def safe_sum(arr):
    return float(np.nansum(arr))


def neff(arr):
    s = np.sum(arr)
    s2 = np.sum(arr * arr)
    if s2 <= 0:
        return 0.0
    return float(s * s / s2)


def percentile(arr, q):
    if len(arr) == 0:
        return float("nan")
    return float(np.percentile(arr, q))


def fmt_sci(x):
    if x == 0:
        return "0"
    return f"{x:.4e}"


def study_year(year, tree_names):
    """Return dict: tree -> stats, plus an aggregate over SR + Sideband."""
    p = DATA_DIR / f"{year}.root"
    f = uproot.open(p)
    out = {}
    for tname in tree_names:
        try:
            t = f[tname]
        except KeyError:
            continue
        n = t.num_entries
        if n == 0:
            out[tname] = dict(n=0, empty=True)
            continue
        # Only need weight + central + ids; keep memory bounded.
        want = [WEIGHT, WEIGHT_CENTRAL] + ID_BRANCHES
        have = set(t.keys())
        want = [b for b in want if b in have]
        arr = t.arrays(want, library="np")
        w = np.asarray(arr[WEIGHT], dtype=float) if WEIGHT in arr else np.zeros(n)
        wc = (np.asarray(arr[WEIGHT_CENTRAL], dtype=float)
              if WEIGHT_CENTRAL in arr else np.zeros(n))

        pos = w[w > 0]
        neg = w[w < 0]
        zero = int(np.sum(w == 0))
        aw = np.abs(w)

        # "large weight" census: relative + absolute thresholds.
        n_total = n
        rel_thresholds = {0.999: int(np.sum(aw > np.percentile(aw, 99.9))),
                          0.99:  int(np.sum(aw > np.percentile(aw, 99.0))),
                          0.95:  int(np.sum(aw > np.percentile(aw, 95.0)))}
        abs_thresholds = {1.0:  int(np.sum(aw > 1.0)),
                          0.1:  int(np.sum(aw > 0.1)),
                          0.01: int(np.sum(aw > 0.01))}

        # Contribution of the top-k by |w| to sum(w) (sign-aware).
        order = np.argsort(aw)[::-1]
        top_frac = {}
        for k in (1, 5, 20, 100):
            k = min(k, n)
            idx = order[:k]
            contrib = float(np.sum(w[idx]))
            sw = float(np.sum(w))
            top_frac[k] = dict(
                count=k,
                abs_sum=float(np.sum(aw[idx])),
                signed_sum=contrib,
                frac_of_total=(contrib / sw) if sw != 0 else float("nan"),
            )

        # Top-20 event ids.
        top20_idx = order[: min(20, n)]
        top_rows = []
        for i in top20_idx:
            row = dict(idx=int(i), w=float(w[i]),
                       wc=float(wc[i]) if wc.size else float("nan"))
            for b in ID_BRANCHES:
                if b in arr and i < len(arr[b]):
                    row[b] = int(arr[b][i])
                else:
                    row[b] = None
            top_rows.append(row)

        out[tname] = dict(
            n=n,
            w=w, wc=wc,
            sum_w=safe_sum(w),
            sum_wc=safe_sum(wc),
            sum_pos=safe_sum(pos),
            sum_neg=safe_sum(neg),
            sum_abs=safe_sum(aw),
            n_pos=int(pos.size), n_neg=int(neg.size), n_zero=zero,
            neff=neff(w),
            w_min=float(np.min(w)),
            w_max=float(np.max(w)),
            w_mean=float(np.mean(w)),
            w_std=float(np.std(w)),
            w_p1=percentile(w, 1),
            w_p50=percentile(w, 50),
            w_p99=percentile(w, 99),
            rel_thresholds=rel_thresholds,
            abs_thresholds=abs_thresholds,
            top_frac=top_frac,
            top_rows=top_rows,
            empty=False,
        )
    return out


def aggregate(stats_all_years, group_years, tree_filter=None):
    """Aggregate raw event counts + sums across a set of years.

    tree_filter: optional callable(name)->bool; if None use all trees.
    Returns a dict keyed by tree with N, sum_w, sum_wc, neff(not additive),
    plus a 'TOTAL' entry.
    """
    agg = defaultdict(lambda: dict(n=0, sum_w=0.0, sum_wc=0.0,
                                   n_pos=0, n_neg=0, sum_abs=0.0))
    for y in group_years:
        if y not in stats_all_years:
            continue
        for tname, st in stats_all_years[y].items():
            if tree_filter and not tree_filter(tname):
                continue
            if st.get("empty", False):
                continue
            d = agg[tname]
            d["n"] += st["n"]
            d["sum_w"] += st["sum_w"]
            d["sum_wc"] += st["sum_wc"]
            d["n_pos"] += st["n_pos"]
            d["n_neg"] += st["n_neg"]
            d["sum_abs"] += st["sum_abs"]
    # total
    tot = dict(n=0, sum_w=0.0, sum_wc=0.0, n_pos=0, n_neg=0, sum_abs=0.0)
    for d in agg.values():
        for k in tot:
            tot[k] += d[k]
    agg["__TOTAL__"] = tot
    return agg


def write_report(years, stats, fh):
    P = fh.write

    P("=" * 100 + "\n")
    P("DYJetsToLL_Stitch statistics — ALL years (2016-2026)\n")
    P(f"  Directory: {DATA_DIR}\n")
    P(f"  Nominal weight: {WEIGHT}   (pre-lumi sanity: {WEIGHT_CENTRAL})\n")
    P("  Note: weight_lumi is negative for a subset of events because the\n"
      "        DY stitch is NLO (negative-weight genWeight events).\n")
    P("=" * 100 + "\n\n")

    # 1) Per year raw event counts (per tree).
    P("[1] N events per tree per year\n")
    hdr = f"{'Tree':<32s} " + " ".join(f"{y:>10s}" for y in years) + f" {'TOTAL':>12s}\n"
    P(hdr)
    P("-" * len(hdr) + "\n")
    all_trees = sorted({t for y in years for t in stats[y].keys()})
    for tname in all_trees:
        row = f"{tname:<32s} "
        total = 0
        for y in years:
            st = stats[y].get(tname)
            v = st["n"] if st and not st.get("empty", False) else 0
            total += v
            row += f"{v:>10d} "
        row += f"{total:>12d}\n"
        P(row)
    P("\n")

    # 2) Per year total (all trees) & SR + Sideband split.
    P("[2] Per-year totals (all trees) and SR/Sideband split\n")
    P(f"{'Year':<14s}{'N_all':>12s}{'N_SR':>12s}{'N_Sideband':>12s}"
      f"{'sum_w_lumi':>16s}{'sum_wc_central':>18s}{'Nneg_w':>10s}{'Neff':>12s}\n")
    P("-" * 110 + "\n")
    for y in years:
        st = stats[y]
        n_all = sum(s["n"] for s in st.values() if not s.get("empty", False))
        n_sr = sum(s["n"] for t, s in st.items()
                   if not s.get("empty", False) and t.startswith("SR"))
        n_sb = sum(s["n"] for t, s in st.items()
                   if not s.get("empty", False) and t.startswith("Sideband"))
        # aggregate weight over all trees (concatenate for neff)
        wparts = []
        sw = swc = 0.0
        npos = nneg = 0
        for s in st.values():
            if s.get("empty", False):
                continue
            sw += s["sum_w"]; swc += s["sum_wc"]
            npos += s["n_pos"]; nneg += s["n_neg"]
            if "w" in s:
                wparts.append(s["w"])
        ne = neff(np.concatenate(wparts)) if wparts else 0.0
        P(f"{y:<14s}{n_all:>12d}{n_sr:>12d}{n_sb:>12d}"
          f"{sw:>16.4e}{swc:>18.4e}{nneg:>10d}{ne:>12.2f}\n")
    P("\n")

    # 3) Weight distribution stats per year (all trees pooled).
    P("[3] weight_lumi distribution per year (all trees pooled)\n")
    P(f"{'Year':<14s}{'N':>10s}{'min':>13s}{'p1':>13s}{'p50':>13s}"
      f"{'p99':>13s}{'max':>13s}{'mean':>13s}{'std':>13s}\n")
    P("-" * 115 + "\n")
    for y in years:
        st = stats[y]
        wparts = [s["w"] for s in st.values() if not s.get("empty", False) and "w" in s]
        if not wparts:
            P(f"{y:<14s}{0:>10d}\n")
            continue
        w = np.concatenate(wparts)
        P(f"{y:<14s}{len(w):>10d}{np.min(w):>13.4e}{np.percentile(w,1):>13.4e}"
          f"{np.percentile(w,50):>13.4e}{np.percentile(w,99):>13.4e}"
          f"{np.max(w):>13.4e}{np.mean(w):>13.4e}{np.std(w):>13.4e}\n")
    P("\n")

    # 4) "Large weight" census per year (all trees pooled).
    P("[4] Large-weight census per year (all trees pooled, |weight_lumi|)\n")
    P(f"{'Year':<14s}{'N':>10s}{'|w|>1e0':>10s}{'|w|>1e-1':>10s}{'|w|>1e-2':>10s}"
      f"{'top0.1%':>10s}{'top1%':>8s}{'top5%':>8s}{'max|w|':>13s}\n")
    P("-" * 100 + "\n")
    for y in years:
        st = stats[y]
        wparts = [s["w"] for s in st.values() if not s.get("empty", False) and "w" in s]
        if not wparts:
            continue
        w = np.concatenate(wparts)
        aw = np.abs(w)
        n = len(w)
        P(f"{y:<14s}{n:>10d}{int(np.sum(aw>1.0)):>10d}{int(np.sum(aw>0.1)):>10d}"
          f"{int(np.sum(aw>0.01)):>10d}"
          f"{int(np.sum(aw>np.percentile(aw,99.9))):>10d}"
          f"{int(np.sum(aw>np.percentile(aw,99.0))):>8d}"
          f"{int(np.sum(aw>np.percentile(aw,95.0))):>8d}"
          f"{np.max(aw):>13.4e}\n")
    P("\n")

    # 5) Top-20 events overall (across all years, all trees) by |weight_lumi|.
    P("[5] Top-20 events by |weight_lumi| across ALL years & trees\n")
    candidates = []
    for y in years:
        for tname, s in stats[y].items():
            if s.get("empty", False):
                continue
            for r in s["top_rows"]:
                r["year"] = y
                r["tree"] = tname
                candidates.append(r)
    candidates.sort(key=lambda r: abs(r["w"]), reverse=True)
    P(f"{'Year':<12s}{'Tree':<24s}{'run':>10s}{'lumi':>10s}{'event':>12s}"
      f"{'weight_lumi':>16s}{'weight_central':>16s}\n")
    P("-" * 110 + "\n")
    for r in candidates[:20]:
        P(f"{r.get('year',''):<12s}{r.get('tree',''):<24s}"
          f"{str(r.get('run')):>10s}{str(r.get('lumi')):>10s}"
          f"{str(r.get('event')):>12s}{r['w']:>16.4e}{r.get('wc',float('nan')):>16.4e}\n")
    P("\n")

    # 6) Aggregates by era.
    groups = [("Run2", RUN2), ("Run3", RUN3), ("Run2024-2026", RUN_LATE),
              ("ALL", years)]
    for gname, gy in groups:
        gy = [y for y in gy if y in stats]
        if not gy:
            continue
        agg_all = aggregate(stats, gy)
        agg_sr = aggregate(stats, gy, lambda n: n.startswith("SR"))
        agg_sb = aggregate(stats, gy, lambda n: n.startswith("Sideband"))
        P(f"[6] Aggregate: {gname}  (years: {', '.join(gy)})\n")
        for label, agg in (("ALL", agg_all), ("SR", agg_sr), ("Sideband", agg_sb)):
            d = agg["__TOTAL__"]
            P(f"   {label:<10s} N={d['n']:>10d}  sum_w_lumi={d['sum_w']:+.4e}  "
              f"sum_wc_central={d['sum_wc']:+.4e}  "
              f"Nneg_w={d['n_neg']:>8d}  sum|w|={d['sum_abs']:.4e}\n")
        P("\n")


def plot_weight_hists(years, stats):
    fig, axes = plt.subplots(1, 1, figsize=(9, 6))
    ax = axes
    for y in years:
        wparts = [s["w"] for s in stats[y].values()
                  if not s.get("empty", False) and "w" in s]
        if not wparts:
            continue
        w = np.concatenate(wparts)
        aw = np.abs(w)
        rng = (1e-6, max(aw.max(), 1e-1))
        ax.hist(aw, bins=np.logspace(np.log10(rng[0]), np.log10(rng[1]), 60),
                histtype="step", linewidth=1.4, label=f"{y} (N={len(w)})")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("|weight_lumi|")
    ax.set_ylabel("events / bin")
    ax.set_title("DYJetsToLL_Stitch |weight_lumi| distribution by year")
    ax.legend(fontsize=8, loc="best")
    ax.grid(True, which="both", ls=":", alpha=0.4)
    fig.tight_layout()
    fig.savefig(OUTPUT_PNG, dpi=130)
    plt.close(fig)


def main():
    years = list_years()
    if not years:
        print(f"No year root files found under {DATA_DIR}")
        return
    print(f"Found {len(years)} year files: {years}\n")

    # First year drives the tree list; we keep the union across years.
    tree_names = sorted(set().union(*(set(list_trees(y)) for y in years)))
    print(f"Total distinct trees: {len(tree_names)}\n")

    stats = {}
    for y in years:
        stats[y] = study_year(y, tree_names)
        # quick stdout progress
        n_all = sum(s["n"] for s in stats[y].values()
                    if not s.get("empty", False))
        print(f"  {y:<14s} trees done  N_all={n_all}")

    # Write full report to file.
    with open(OUTPUT_TXT, "w") as fh:
        write_report(years, stats, fh)

    # Stdout summary table.
    print("\n" + "=" * 92)
    print("Per-year summary (all trees pooled). Nominal weight = weight_lumi")
    print("=" * 92)
    print(f"{'Year':<14s}{'N_all':>12s}{'N_SR':>12s}{'N_SB':>12s}"
          f"{'sum_w_lumi':>16s}{'Nneg_w':>10s}{'max|w|':>13s}{'Neff':>12s}")
    print("-" * 92)
    for y in years:
        st = stats[y]
        n_all = sum(s["n"] for s in st.values() if not s.get("empty", False))
        n_sr = sum(s["n"] for t, s in st.items()
                   if not s.get("empty", False) and t.startswith("SR"))
        n_sb = sum(s["n"] for t, s in st.items()
                   if not s.get("empty", False) and t.startswith("Sideband"))
        sw = sum(s["sum_w"] for s in st.values() if not s.get("empty", False))
        nneg = sum(s["n_neg"] for s in st.values() if not s.get("empty", False))
        wparts = [s["w"] for s in st.values()
                  if not s.get("empty", False) and "w" in s]
        awmax = max(float(np.max(np.abs(p))) for p in wparts) if wparts else 0.0
        ne = neff(np.concatenate(wparts)) if wparts else 0.0
        print(f"{y:<14s}{n_all:>12d}{n_sr:>12d}{n_sb:>12d}{sw:>+16.4e}"
              f"{nneg:>10d}{awmax:>13.4e}{ne:>12.2f}")

    # Aggregate row.
    n_all = sum(s["n"] for y in years for s in stats[y].values()
                if not s.get("empty", False))
    sw = sum(s["sum_w"] for y in years for s in stats[y].values()
             if not s.get("empty", False))
    nneg = sum(s["n_neg"] for y in years for s in stats[y].values()
               if not s.get("empty", False))
    wparts = [s["w"] for y in years for s in stats[y].values()
              if not s.get("empty", False) and "w" in s]
    ne = neff(np.concatenate(wparts)) if wparts else 0.0
    print("-" * 92)
    print(f"{'TOTAL':<14s}{n_all:>12d}{'':>12s}{'':>12s}{sw:>+16.4e}"
          f"{nneg:>10d}{'':>13s}{ne:>12.2f}")

    print("\nFull report ->", OUTPUT_TXT)
    print("Weight histogram ->", OUTPUT_PNG)


if __name__ == "__main__":
    main()
