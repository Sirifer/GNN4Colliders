# 日报 2026-08-15

## 背景

H→Zγ 分析，Phase 1：GNN ttH vs other 二分类，目标是比较 Z_GNN vs Z_cut。
今天按老师意见统一了 sensitivity 计算条件，改为 3-category GNN，并加入统计误差。

---

## 今天完成的工作

### 1. 统一 sensitivity 计算条件

新脚本：`tools/sensitivity_unified.py`

**三个改动**：

| 改动 | 旧做法 | 新做法 |
|------|--------|--------|
| 质量窗口 | GNN 用 115–135，cut-based 用 120–130 | 两者统一为 **120 < m_llg < 130 GeV** |
| 数据范围 | GNN 只用 test fold (25%) | 两者统一用 **全部样本（Run2+Run3 all folds）** |
| Sideband | 未明确排除 | 用 `is_SR == 1` 严格限定在 **SR 内**，不包含 sideband |

数据来源：`root_p2root_v2_scored/`（已有 `ttH_gnn_score` branch 的 415 个文件）

---

### 2. GNN 从 2-category 改为 3-category

新方法：扫描两个边界 c1 < c2，定义三个区间：

| Category | 定义 |
|----------|------|
| GNN high | score > c2 |
| GNN medium | c1 < score ≤ c2 |
| GNN low | score ≤ c1 |

扫描范围：c1 ∈ [0.10, 0.90]（步长 0.05），c2 ∈ [0.20, 0.95]（步长 0.05）

最优化目标：Z_GNN = √(Z_high² + Z_medium² + Z_low²) 最大

---

### 3. 加入统计误差

公式（基于权重平方和）：

```
σ_S² = Σ w_i²    (ttH 事件)
σ_B² = Σ w_i²    (other 事件)
(σ_Z / Z)² = (σ_S / S)² + 0.25 × (σ_B / B)²
```

对 Z_comb = √(ΣZ_i²)，误差传播：
```
σ_Z_comb = √(Σ (2·Z_i·σ_Z_i)²) / (2·Z_comb)
```

---

## 计算结果

**条件**：120 < m_llg < 130 GeV，is_SR==1，weight_lumi（含符号），Run2+Run3 全样本

### Cut-based 5-category

| Category | S(ttH) | σ_S | B(other) | σ_B | Z | σ_Z |
|----------|--------|-----|----------|-----|---|-----|
| ttHhad | 1.5301 | 0.0056 | 43.69 | 7.30 | 0.2302 | 0.0193 |
| ttHlep | 1.3790 | 0.0052 | 4.691 | 3.83 | 0.6088 | 0.2483 |
| untag | 1.5847 | 0.0057 | 1951.8 | 70.16 | 0.0359 | 0.0007 |
| VH3l | 0.1991 | 0.0019 | 81.99 | 14.94 | 0.0220 | 0.0020 |
| VHpTmiss | 0.0081 | 0.0004 | 216.7 | 22.48 | 0.0005 | 0.0000 |
| **Combined** | | | | | **0.6522** | **0.2319** |

> 注：Z_cut 误差较大主要来自 ttHlep 区域（B 很小，σ_B/B 较大）。

### GNN 3-category（最优边界 c1=0.85, c2=0.95）

| Category | S(ttH) | σ_S | B(other) | σ_B | Z | σ_Z |
|----------|--------|-----|----------|-----|---|-----|
| high (>0.95) | 3.4562 | 0.0083 | 39.58 | 6.99 | 0.5417 | 0.0478 |
| medium (0.85–0.95) | 0.6136 | 0.0035 | 82.14 | 13.54 | 0.0676 | 0.0056 |
| low (<0.85) | 0.6311 | 0.0036 | 2177.1 | 74.07 | 0.0135 | 0.0002 |
| **Combined** | | | | | **0.5460** | **0.0475** |

### 对比总结

| 方法 | Z | σ_Z (stat.) |
|------|---|-------------|
| Cut-based 5-cat | **0.6522** | 0.2319 |
| GNN 3-cat | **0.5460** | 0.0475 |
| Z_GNN / Z_cut | **0.837** | — |

**目前 GNN 3-cat 比 cut-based 低约 16%**，尚未超越。

---

## 解读与下一步讨论

### 为什么 GNN 暂时没赢

1. **ttHlep 是关键区域**：cut-based 的 Z 主要来自 ttHlep（B=4.7，S/B 比最优），而 GNN high category 的 B=39.6 更大，信噪比不如 cut-based 的 ttHlep。
2. **GNN 训练只用 Run3**：config 里 `tree_name: merged_SR_Sideband`，但打分覆盖 Run2+Run3，Run2 样本的 GNN score 可能不如 Run3 准确。
3. **3-category 扫描网格较粗**（步长 0.05），可以更细化扫描高分区。

### 可改进方向

1. 细化 c1、c2 扫描（特别是 c2 在 0.90–1.00 附近）
2. 检查 Run2 vs Run3 的 score 分布是否有差异
3. 统计误差来源：ttHlep 里 B 的权重方差很大，说明少量大权重背景事件主导——可以单独看哪个样本
4. 用 Run2+Run3 全数据重新训练 GNN（现在只训练了 Run3）

---

## 文件路径

| 文件 | 路径 |
|------|------|
| 新计算脚本 | `tools/sensitivity_unified.py` |
| 旧 cut-based 脚本 | `tools/cutbased_sensitivity.py` |
| 旧 GNN significance 结果 | `eval/hzg/ttH_vs_other_run3_v2/significance.txt` |
| scored ROOT 文件 | `root_p2root_v2_scored/` (415 文件) |
  整个流程：
  prep_data ✅ 运行中
      ↓ 跑完提交
  训练 job（~2小时）
      ↓ 看日志确认最佳 epoch
  evaluate_hzg_binary.py（生成 npz）
      ↓
  GNN significance 脚本（3-cat 扫描）vs cut-based Z=0.6522