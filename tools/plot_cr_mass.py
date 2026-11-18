
#!/usr/bin/env python3

"""
检查单个 Run 3 ROOT 文件中 nominal 与 CR 的 m_llg 分布。

左图：指定 category 的 SR、sideband 和 inverse-ID CR。
右图：inclusive inverse-ID CR 的完整质量分布。

只读取 ROOT 文件，不修改原始数据。
"""

from pathlib import Path

import numpy as np
import uproot
import matplotlib.pyplot as plt


# ===== 用户可以修改的配置 =====

ROOT_FILE = Path(
    "/scratchfs2/cms/huangsuyun/GNN4Colliders/"
    "root_p2root_v2/Data/2022postEE.root"
)

CATEGORY = "untag"       # 例如 untag、ttHhad、ttHlep、VBF
MASS_BRANCH = "m_llg"     # 与当前分析保持一致

MASS_MIN = 100
MASS_MAX = 180
BIN_WIDTH = 2

OUTPUT_DIR = Path(
    "/scratchfs2/cms/huangsuyun/GNN4Colliders/"
    "plots/cr_mass_check"
)


def read_mass(root_file, tree_name, branch):
    """读取指定 TTree 的质量分支。"""
    if tree_name not in root_file:
        print(f"[Missing tree] {tree_name}")
        return None

    tree = root_file[tree_name]

    if branch not in tree.keys():
        print(f"[Missing branch] {tree_name}/{branch}")
        return None

    mass = tree[branch].array(library="np")
    mass = np.asarray(mass, dtype=float)
    mass = mass[np.isfinite(mass)]

    print(
        f"{tree_name:30s} "
        f"entries={tree.num_entries:7d}, "
        f"valid mass={len(mass):7d}"
    )

    return mass


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    bins = np.arange(
        MASS_MIN,
        MASS_MAX + BIN_WIDTH,
        BIN_WIDTH
    )

    fig, axes = plt.subplots(
        1, 2,
        figsize=(14, 5.5),
        sharex=True
    )

    with uproot.open(ROOT_FILE) as f:
        print(f"File: {ROOT_FILE}")
        print(f"Category: {CATEGORY}")
        print(f"Mass branch: {MASS_BRANCH}\n")

        # 左图：同一 category 的 nominal 和 CR
        category_trees = [
            (f"SR__{CATEGORY}", "Nominal SR"),
            (f"Sideband__{CATEGORY}", "Nominal sideband"),
            (f"CR_lowIDMVA__{CATEGORY}", "Inverse-ID CR"),
        ]

        for tree_name, label in category_trees:
            mass = read_mass(f, tree_name, MASS_BRANCH)

            if mass is None:
                continue

            axes[0].hist(
                mass,
                bins=bins,
                histtype="step",
                linewidth=1.8,
                label=f"{label} (N={len(mass)})",
            )

        # 右图：inclusive CR，检查完整质量范围
        inclusive_name = "CR_lowIDMVA__inclusive"
        mass = read_mass(f, inclusive_name, MASS_BRANCH)

        if mass is not None:
            axes[1].hist(
                mass,
                bins=bins,
                histtype="step",
                linewidth=1.8,
                label=f"Inclusive CR (N={len(mass)})",
            )

            # 标出左右 sideband 和 SR 的边界
            for x in (120, 130):
                axes[1].axvline(
                    x,
                    color="gray",
                    linestyle="--",
                    linewidth=1
                )

        for ax in axes:
            ax.set_xlabel(r"$m_{\ell\ell\gamma}$ [GeV]")
            ax.set_ylabel("Number of events")
            ax.set_xlim(MASS_MIN, MASS_MAX)
            ax.grid(alpha=0.2)
            ax.legend(fontsize=9)

        axes[0].set_title(f"Category: {CATEGORY}")
        axes[1].set_title("Inclusive inverse-ID CR")

        fig.suptitle(ROOT_FILE.name)
        fig.tight_layout()

        output = OUTPUT_DIR / (
            f"{ROOT_FILE.stem}_{CATEGORY}_mllg.png"
        )

        fig.savefig(output, dpi=160, bbox_inches="tight")
        plt.close(fig)

        print(f"\nSaved plot: {output}")


if __name__ == "__main__":
    main()
