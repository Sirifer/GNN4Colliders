# 2026-10-22 日志：方法二 CR 推理完成 & 根本性问题发现

## 今日目标

继续方法二（CR 归一化背景估计）：对3种 CR 数据跑 GNN（pt_only）推理，统计通过 GNN high cut 后的事件数，判断方法二可行性。

---

## 已完成的工作

### 1. 理清处理链

pt_only 模型（`ttH_pt_only_data_fold0.yaml`，`in_size=6`）输入用的是 `*_pt_div_mllg` 而不是原始 pt，因此 CR 文件需要先跑 `prepare_pt_only_inputs.py`。完整三步处理链：

```
原始 Data CR tree
  → merge_cr_trees.py       合并3种CR inclusive树，加 cr_type_tag
  → prepare_pt_only_inputs.py  加 *_pt_div_mllg
  → inference (fold0)          加 ttH_gnn_score
```

### 2. 新增文件

| 文件 | 说明 |
|------|------|
| `tools/code/merge_cr_trees.py` | 合并3种CR inclusive树（lowIDMVA/lowPTOverMllg/FSRgamma）→ `merged_CR`，加 `cr_type_tag`（0/1/2） |
| `root_gnn_dgl/configs/hzg/ttH_pt_only_data_fold0_cr.yaml` | CR 专用 inference config，`tree_name: merged_CR` |
| `root_gnn_dgl/jobs/inference/hzg_ttH_pt_only_fold0_inference_cr.sbatch` | CR 推理 sbatch job |
| `tools/diag_cr_counts_v2.py` | 诊断脚本，从 scored CR 文件统计 GNN cut 后各 CR 类型的 SB/SR 事件数 |
| `tools/run_cr_pt_only.sh` | 三步处理的一键脚本（已更新） |

### 3. 推理结果

成功对3种CR的 inclusive 树跑完 fold0 推理，输出在 `root_p2root_v2_pt_only_cr_scored_fold0/`：

| era | entries | ttH_gnn_score |
|-----|---------|---------------|
| 2022preEE | 846 | YES |
| 2022postEE | 2,882 | YES |
| 2023preBPix | 2,497 | YES |
| 2023postBPix | 1,597 | YES |
| 2024 | 17,274 | YES |
| 2025 | 18,701 | YES |
| **合计** | **43,797** | |

注：43,797 = CR_lowIDMVA inclusive（49,356）+ CR_lowPTOverMllg inclusive（32,027）的 Run3 子集，CR_FSRgamma m_llg 在 7-100 GeV 范围，不在分析质量窗口内。

---

## 根本性问题：方法二不可行

### 发现

跑完诊断脚本后：

```
c2 = 0.9940773138

Nominal SB (is_SR==0, score>c2): 12

CR_lowIDMVA:      pre-GNN SB=0  SR=19264   high SB=0  SR=1
CR_lowPTOverMllg: pre-GNN SB=0  SR=12097   high SB=0  SR=0
CR_FSRgamma:      pre-GNN SB=0  SR=0       high SB=0  SR=0
```

**3种CR的 SB 事件数全为 0**，不论是否加 GNN cut。

### 根因

不是我们处理的问题，是**上游数据生产的问题**。直接检查原始文件（包括 backup `*.preCRcat_20260729`）：

```
CR_lowIDMVA__inclusive (2024):  total=19264  SB=0  SR=19264
CR_lowIDMVA__inclusive (2025):  min=120.00  max=130.00
Sideband__inclusive    (2024):  total=198745  SB=154753  SR=2674
```

nominal Sideband 树有完整的 110-180 GeV 事件，但**所有 CR 树在生产时只保存了 120-130 GeV 的事件**，110-120 和 130-180 GeV 的 CR sideband 事件被丢弃了。backup 文件（preCRcat）也是同样的情况。

CR_FSRgamma 的 m_llg 分布在 7-100 GeV，和 Higgs 质量窗口完全不重叠，物理上也无法用于背景估计。

### 结论

**方法二（CR 归一化）在当前数据文件下完全不可行。** alpha = N_nom_SB / N_CR_SB 无法计算，因为 N_CR_SB = 0。

---

## 下一步选项

1. **联系 laipeizhu**：询问是否有包含完整质量范围（110-180 GeV）的 CR 文件，或能否重新生产 ntuples 时把 CR sideband 事件保存进来。

2. **放弃方法二**：继续用方法一（sideband 函数拟合）作为唯一背景估计方法。

3. **探索其他 CR 定义**：看是否有其他控制区的定义方式，使得 m_llg 分布覆盖 110-180 GeV。

---

## 当前文件目录状态

```
root_p2root_v2_cr_merged/          ← merged_CR tree（3种CR inclusive，cr_type_tag标记）
root_p2root_v2_pt_only_cr_data/    ← 加了 *_pt_div_mllg
root_p2root_v2_pt_only_cr_scored_fold0/  ← 含 ttH_gnn_score，但 SB 全为0
```
