# 2026-08-25 工作日志

## 今日工作内容

### 背景

在 H→Zγ ttH GNN 分析中，已完成 2-fold 训练：

- fold0: test = event%2==0，train = event%2==1
- fold1: test = event%2==1，train = event%2==0

OOF = fold0 test + fold1 test，覆盖完整 Run3，无需 ×2 或 √2 投影。

---

### 今日主要结果

#### 1. Optuna 寻优（significance_optuna_2fold.py）

背景样本：Signal/VH, Signal/ggF, Signal/VBF, Bkg_MC/TT, Bkg_MC/DYGto2LG_10to100

OOF 总产额：S = 3.2355，B = 952.93

最优 cut：c1 = 0.9087，c2 = 0.9918

| Category | S      | B        | Neff_B | N_bkg | Z      | σZ     |
|----------|--------|----------|--------|-------|--------|--------|
| high     | 0.5645 | 0.0034   | 7.63   | 137   | 2.1679 | 0.0919 |
| medium   | 1.6051 | 22.7266  | 36.09  | 2308  | 0.3328 | 0.0271 |
| low      | 1.0659 | 930.2003 | 2028   | 217554| 0.0349 | 0.0004 |
| **combined** | | | | | **2.1936** | **0.0909** |

**问题**：Z = 2.19 是伪高值（见第三节诊断）。

---

#### 2. 网格扫描（significance_diagnostic.py）

**仅 TT 背景时**：

最优 cut：c1 = 0.880，c2 = 0.980

| | Z | σZ |
|---|---|---|
| Cut-based | 0.6488 | 0.1217 |
| GNN OOF  | 0.7883 | 0.1253 |
| GNN/Cut  | 1.215 (+21.5%) | |

**全背景（+DY+VH+ggF+VBF）时**：

最优 cut：c1 = 0.810，c2 = 0.980

| | Z | σZ |
|---|---|---|
| Cut-based | 0.5884 | 0.1021 |
| GNN OOF  | 0.7728 | 0.1214 |
| GNN/Cut  | 1.313 (+31.3%) | |

cut-based 各 category（全背景）：

| Category | S      | B       | Neff_B | Z      |
|----------|--------|---------|--------|--------|
| ttHhad   | 1.0327 | 25.83   | 38.14  | 0.2018 |
| ttHlep   | 0.9286 | 2.55    | 5.19   | 0.5505 |
| untag    | 1.1295 | 850.23  | 1869.6 | 0.0387 |
| VH3l     | 0.1385 | 20.75   | 53.76  | 0.0304 |
| VHpTmiss | 0.0063 | 53.57   | 104.46 | 0.0009 |

GNN OOF high bin（c2=0.980）：

| Sample        | N   | Yield  | σ      | Neff |
|---------------|-----|--------|--------|------|
| Bkg_MC/TT     | 19  | 2.829  | 1.106  | 6.54 |
| Signal/VH     | 515 | 0.008  | 0.0002 | 323  |
| Signal/ggF    | 5   | 0.003  | 0.003  | 3.37 |
| Signal/VBF    | 3   | 0.0001 | 0.0001 | 2.55 |
| Bkg_MC/DY     | 1   | 0.076  | 0.076  | 1.00 |
| **Total**     | 543 | 2.916  | 1.109  | 6.92 |

---

#### 3. high bin 详细诊断（check_highcat_weights.py，c2 = 0.9918）

| | fold0 | fold1 | ratio |
|---|---|---|---|
| 信号 N    | 1,455  | 49,602 | ×34   |
| 信号 Yield | 0.01365 | 0.5699 | ×41.7 |
| 信号 Neff  | 643    | 20,035 | ×31   |
| 背景 N    | 1      | 139    | ×139  |
| 背景 Yield | 1.71e-5 | 3.39e-3 | ×198 |
| 背景 Neff  | 1.0    | 7.79   | —     |

背景样本构成（OOF，c2=0.9918）：

| Sample     | N   | Yield   | Neff |
|------------|-----|---------|------|
| Signal/VH  | 138 | 0.00218 | 85.2 |
| Signal/ggF | 1   | 0.00119 | 1.00 ← 单个大权重事例 |
| Signal/VBF | 1   | 3.4e-5  | 1.00 |
| Bkg_MC/TT  | 0   | 0       | —    |
| Bkg_MC/DY  | 0   | 0       | —    |

**核心诊断**：

- 信号 score 中位数：fold0 ≈ 0.943，fold1 ≈ 0.980，两个模型 score scale 明显不同
- c2 = 0.9918 几乎卡在 fold0 model score 的上限，fold0 几乎无事例落入 high bin
- high bin 结果几乎全来自 fold1，OOF 不再是均衡的两 fold 平均
- Optuna 的 Z = 2.19 是模型 calibration 不一致引起的伪高值，不可信

---

#### 4. fold-specific cuts 方案（significance_fixedcuts_2fold_two_cuts.py）

对 fold0 和 fold1 各自独立寻优，得到 6 个 categories：

fold0 cuts：c1 = 0.818，c2 = 0.975

| fold0 category | S      | B       | Neff_B | Z      |
|----------------|--------|---------|--------|--------|
| high           | 0.5216 | 2.708   | 6.04   | 0.3075 |
| medium         | 0.5178 | 16.715  | 37.30  | 0.1260 |
| low            | 0.4452 | 424.360 | 1022.7 | 0.0216 |
| fold0 combined |        |         |        | **0.333** |

fold1 cuts：c1 = 0.541，c2 = 0.9917

| fold1 category | S      | B       | Neff_B | Z      |
|----------------|--------|---------|--------|--------|
| high           | 0.5795 | 0.00356 | 8.56   | 2.1878 |
| medium         | 0.9973 | 40.329  | 78.54  | 0.1564 |
| low            | 0.1741 | 468.815 | 933.74 | 0.0080 |
| fold1 combined |        |         |        | **2.193** |

**6-category 总 Z = 2.219**

**问题**：fold1 仍然单独推到 c2=0.9917，问题本质相同——fold1 model 把大量事例打到极高 score，fold0 没有。Z = 2.22 同样不可信。

---

### 今日结论

1. **当前最可信结果**：GNN OOF（c2=0.980，网格扫描）Z = 0.773 ± 0.121，比 cut-based +31%，但 σZ 与增益同量级。

2. **Optuna 及 fold-specific cuts 结果均不可信**：fold0/fold1 两个模型的 score scale 不一致是根本原因，high bin 几乎只有 fold1 的贡献。

3. **high bin 背景统计量极低**：无论哪种 cut，Neff_B ≈ 6–9，TT 只有 19 个 raw events，背景估计不稳定。

4. **需要解决的核心问题**：
   - 诊断 fold0/fold1 score 不对称的原因（训练 log、AUC、loss 曲线）
   - 考虑 score calibration（isotonic regression / Platt scaling）
   - 加载更完整的背景样本
   - 设置合理的 MIN_B_NEFF（建议 ≥ 10）

---

### 明日计划

- [ ] 查看 fold0/fold1 训练 log，比较 AUC 和 loss 收敛曲线
- [ ] 对同一模型，比较 even/odd events 的 score 分布（排除 event split 问题）
- [ ] 考虑加入更多背景样本（WJets, ZJets, diboson 等）
- [ ] 评估 score calibration 方案

---

详细分析总结：`analysis_summary_20260825.txt`
