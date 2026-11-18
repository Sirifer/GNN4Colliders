
#!/usr/bin/env python3

from pathlib import Path
import uproot

BASE = Path(
    "/scratchfs2/cms/huangsuyun/GNN4Colliders/"
    "root_p2root_v2/Data"
)

YEARS = {
    "2022preEE.root",
    "2022postEE.root",
    "2023preBPix.root",
    "2023postBPix.root",
    "2024.root",
    "2025.root",
}

for path in sorted(BASE.rglob("*.root")):
    if path.name not in YEARS:
        continue

    # 避免误读 scored 输出文件
    if "scored" in str(path).lower():
        continue

    print(f"\n{'=' * 70}")
    print(f"File: {path}")

    with uproot.open(path) as f:
        trees = [
            key for key in f.keys(cycle=False)
            if str(key).startswith("CR_lowIDMVA__")
        ]

        if not trees:
            print("No CR_lowIDMVA trees")
            continue

        for name in trees:
            tree = f[name]
            branches = set(tree.keys())

            print(
                f"{name:35s} "
                f"entries={tree.num_entries:8d} "
                f"score={'YES' if 'ttH_gnn_score' in branches else 'NO'}"
            )
