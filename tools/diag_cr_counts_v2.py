"""
Diagnostic: count nominal SB vs CR events after GNN high cut.

Nominal SB: root_p2root_v2_pt_only_scored_ttH_vs_other_run3_fold0/Data/
CR scored:  root_p2root_v2_pt_only_cr_scored_fold0/
            tree=merged_CR, cr_type_tag: 0=lowIDMVA 1=lowPTOverMllg 2=FSRgamma
"""

import json
import pathlib
import numpy as np
import uproot

REPO       = pathlib.Path("/scratchfs2/cms/huangsuyun/GNN4Colliders")
NOM_DIR    = REPO / "root_p2root_v2_pt_only_scored_ttH_vs_other_run3_fold0" / "Data"
CR_DIR     = REPO / "root_p2root_v2_pt_only_cr_scored_fold0"
RESULT_JSON = REPO / "tools" / "results_sideband_fit" / "gnn_category_result.json"
OUT_DIR    = REPO / "tools" / "results_cr_diag"
OUT_DIR.mkdir(exist_ok=True)

RUN3 = ["2022preEE", "2022postEE", "2023preBPix", "2023postBPix", "2024", "2025"]

M_FIT_MIN   = 110.0
M_SIGNAL_LO = 120.0
M_SIGNAL_HI = 130.0
M_FIT_MAX   = 180.0

CR_NAMES = {0: "CR_lowIDMVA", 1: "CR_lowPTOverMllg", 2: "CR_FSRgamma"}

# ---------------------------------------------------------------------------
# Load c2
# ---------------------------------------------------------------------------

with open(RESULT_JSON) as fh:
    C2 = json.load(fh)["c2"]
print("c2 = %.10f\n" % C2)

# ---------------------------------------------------------------------------
# Mass window helpers
# ---------------------------------------------------------------------------

def in_sb(m):
    return ((m >= M_FIT_MIN) & (m < M_SIGNAL_LO)) | ((m >= M_SIGNAL_HI) & (m <= M_FIT_MAX))

def in_sr(m):
    return (m >= M_SIGNAL_LO) & (m < M_SIGNAL_HI)

# ---------------------------------------------------------------------------
# Load nominal scored Data (merged_SR_Sideband, is_SR==0 for SB)
# ---------------------------------------------------------------------------

nom_mass, nom_score, nom_issr = [], [], []

for era in RUN3:
    p = NOM_DIR / (era + ".root")
    if not p.exists():
        print("[WARN] missing nominal: %s" % p)
        continue
    with uproot.open(p) as f:
        t = f["merged_SR_Sideband"]
        arr = t.arrays(["m_llg", "ttH_gnn_score", "is_SR"], library="np")
        nom_mass.append(arr["m_llg"])
        nom_score.append(arr["ttH_gnn_score"])
        nom_issr.append(arr["is_SR"])

nom_mass  = np.concatenate(nom_mass)
nom_score = np.concatenate(nom_score)
nom_issr  = np.concatenate(nom_issr)

finite_nom = np.isfinite(nom_mass) & np.isfinite(nom_score)
nom_sb_high = finite_nom & (nom_score > C2) & (nom_issr == 0) & in_sb(nom_mass)
N_nom_SB = int(nom_sb_high.sum())

print("=== Nominal SB (is_SR==0, score > c2) ===")
print("  N_nom_SB = %d\n" % N_nom_SB)

# ---------------------------------------------------------------------------
# Load CR scored data
# ---------------------------------------------------------------------------

cr_mass, cr_score, cr_type = [], [], []

for era in RUN3:
    p = CR_DIR / (era + ".root")
    if not p.exists():
        print("[WARN] missing CR: %s" % p)
        continue
    with uproot.open(p) as f:
        t = f["merged_CR"]
        arr = t.arrays(["m_llg", "ttH_gnn_score", "cr_type_tag"], library="np")
        cr_mass.append(arr["m_llg"])
        cr_score.append(arr["ttH_gnn_score"])
        cr_type.append(arr["cr_type_tag"])

cr_mass  = np.concatenate(cr_mass)
cr_score = np.concatenate(cr_score)
cr_type  = np.concatenate(cr_type)

finite_cr = np.isfinite(cr_mass) & np.isfinite(cr_score)
high_cr   = finite_cr & (cr_score > C2)

# ---------------------------------------------------------------------------
# Per-CR-type summary
# ---------------------------------------------------------------------------

sep = "=" * 65
results = {}

for tag, name in CR_NAMES.items():
    mask = cr_type == tag
    pre_sb = int((finite_cr & mask & in_sb(cr_mass)).sum())
    pre_sr = int((finite_cr & mask & in_sr(cr_mass)).sum())
    h_sb   = int((high_cr & mask & in_sb(cr_mass)).sum())
    h_sr   = int((high_cr & mask & in_sr(cr_mass)).sum())
    results[name] = dict(pre_sb=pre_sb, pre_sr=pre_sr, h_sb=h_sb, h_sr=h_sr)

    print(sep)
    print("CR type: %s  (cr_type_tag=%d)" % (name, tag))
    print(sep)
    print("  Before GNN cut:  SB=%d  SR=%d" % (pre_sb, pre_sr))
    print("  After  GNN cut:  SB=%d  SR=%d" % (h_sb, h_sr))

    if h_sb > 0 and N_nom_SB > 0:
        alpha  = N_nom_SB / h_sb
        B_pred = alpha * h_sr
        if h_sr > 0:
            sigma = B_pred * np.sqrt(1.0/N_nom_SB + 1.0/h_sb + 1.0/h_sr)
        else:
            sigma = float("nan")
        print("  alpha  = %.4f" % alpha)
        print("  B_pred = %.2f +/- %.2f  (stat only)" % (B_pred, sigma))
    elif h_sb == 0:
        print("  N_CR_SB == 0: cannot compute alpha")
    print()

# ---------------------------------------------------------------------------
# Combined (all 3 CR types)
# ---------------------------------------------------------------------------

h_sb_all = int((high_cr & in_sb(cr_mass)).sum())
h_sr_all = int((high_cr & in_sr(cr_mass)).sum())

print(sep)
print("All CR types combined")
print(sep)
print("  After GNN cut:  SB=%d  SR=%d" % (h_sb_all, h_sr_all))
if h_sb_all > 0 and N_nom_SB > 0:
    alpha  = N_nom_SB / h_sb_all
    B_pred = alpha * h_sr_all
    sigma  = B_pred * np.sqrt(1.0/N_nom_SB + 1.0/h_sb_all + 1.0/h_sr_all) if h_sr_all > 0 else float("nan")
    print("  alpha  = %.4f" % alpha)
    print("  B_pred = %.2f +/- %.2f  (stat only)" % (B_pred, sigma))
print()

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------

lines = ["c2 = %.10f\n\n" % C2]
lines.append("Nominal SB (is_SR==0, score>c2): %d\n\n" % N_nom_SB)
for name, r in results.items():
    lines.append("%s\n" % name)
    lines.append("  pre-GNN SB=%d SR=%d\n" % (r["pre_sb"], r["pre_sr"]))
    lines.append("  high    SB=%d SR=%d\n" % (r["h_sb"],   r["h_sr"]))
    if r["h_sb"] > 0 and N_nom_SB > 0:
        a = N_nom_SB / r["h_sb"]
        b = a * r["h_sr"]
        lines.append("  alpha=%.4f B_pred=%.2f\n" % (a, b))
    lines.append("\n")

out = OUT_DIR / "cr_counts_v2.txt"
out.write_text("".join(lines))
print("Results saved to %s" % out)
