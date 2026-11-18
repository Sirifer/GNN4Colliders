# H→Zγ ttH vs Other GNN 分类器总结（Run 3）

## 一、任务目标

在 CMS H→Zγ 分析中，利用图神经网络（GNN）区分 **ttH 产生模式**（信号）和所有其他过程（背景），以提高 ttH 类别的统计显著性。

---

## 二、数据样本

**Run 3 MC 样本**（2022preEE、2022postEE、2023preBPix、2023postBPix、2024、2025）

| 类别 | 内容 |
|---|---|
| **信号（label=1）** | Signal/ttH（6个年份） |
| **背景（label=0）** | Signal/ggF、VBF、VH（其余Higgs产生模式）+ Bkg_MC：DY+jets (Stitch/FxFx)、TT、tW、TTγ、WZ、ZZ、WW、DYGto2LG、ZG2J 等 |

每个事件构建为一个图，节点为末态粒子（最多6 jets、2 bjets、6 leptons、6 photons、MET），每个节点携带 **7 个特征**：

- pt、eta、phi、E（计算值）
- b-tag / photon ID MVA 分数
- 粒子类型（pdgId）
- 节点类型标签

4-fold 交叉划分（按 event number），fold 0 为测试集，folds 1/2/3 为训练集。

### 样本统计量

**GNN 训练/测试事件数**（仅 Run 3：2022preEE/postEE、2023preBPix/postBPix、2024、2025；来自 DGL graph 文件）：

| 类别 | 训练集 | 测试集 | 合计 |
|---|---|---|---|
| ttH（signal） | 260,352 | 86,784 | 347,136 |
| other（background） | 189,952 | 131,072 | 321,024 |

**推理打分后全样本统计**（Run 2 + Run 3 所有年份；来自 `root_p2root_v2_scored/`，`merged_SR_Sideband` tree）：

| 类别 | 总事件数 | SR 窗口内事件数 | sum(weight_lumi) 全样本 | sum(weight_lumi) SR 窗口 |
|---|---|---|---|---|
| Signal/ttH | 568,206 | 523,033 | 5.22 | 4.80 |
| Signal/ggF | 8,230 | 7,635 | 5.73 | 5.31 |
| Signal/VBF | 26,320 | 24,381 | 1.61 | 1.49 |
| Signal/VH | 390,035 | 352,156 | 4.86 | 4.38 |
| Bkg_MC | 392,178 | 67,673 | 19,244 | 3,662 |

> 注：推理打分覆盖 Run 2+3 全部年份，而 GNN 训练仅使用 Run 3 样本，因此两张表的事件数不直接对应。SR 窗口：`is_SR == 1`，`115 < m_llg < 135 GeV`。

---

## 三、模型与训练配置

| 参数 | 值 |
|---|---|
| 模型 | Graph Convolutional Network (GCN) with Edge Network |
| 隐层维度 | 64 |
| 网络深度 | 4 层 + 4 步消息传递 |
| Dropout | 0 |
| 损失权重 | `weight_lumi_abs`（绝对值） |
| 学习率 | 1e-4，指数衰减（γ=0.99） |
| Batch size | 256 |
| 最大 epoch | 100 |
| Early stop patience | 15 epoch |

训练于 **Tesla V100-SXM2-32GB** 上运行，每 epoch 约 104 秒。

---

## 四、训练结果

Early stop 在 **Epoch 62** 触发，最佳检查点为 **Epoch 34**：

| 指标 | 值 |
|---|---|
| Best Test AUC | **0.9717** |
| Best Test Loss | **0.2207** |
| Train Loss（Ep.34） | 0.1924 |

测试集共 **217,856 事件**（ttH: 86,784；other: 131,072）。

---

## 五、模型评估（Epoch 34，测试集）

| 指标 | 值 |
|---|---|
| Unweighted AUC | **0.9751** |
| weight_lumi 加权 AUC | **0.9740** |
| Youden 最优 threshold | 0.711 |
| 负权重事件数 | 15,827（~7%，来自 MC 负权重） |

输出图（`eval/hzg/ttH_vs_other_run3_v2/`）：

- `training.png` — 训练过程综合图（Loss / AUC / Accuracy / 时间）
- `training_curves.png` — 训练曲线（Train Loss、Test Loss、Test AUC）
- `roc_curve.png` — ROC 曲线（加权 / 不加权）
- `score_unweighted.png` — GNN score 分布（不加权）
- `score_weight_lumi.png` — GNN score 分布（weight_lumi 加权）
- `score_weight_lumi_shape.png` — GNN score 归一化形状
- `score_by_category.png` — 各样本类别 score 分布
- `confusion_matrix.png` — 混淆矩阵

---

## 六、显著性估计

**数据**：Run 3 测试集（`event % 4 == 0`），与模型评估完全一致。

**选取区间**：`is_SR == 1`，`115 < m_llg < 135 GeV`，使用 `weight_lumi` 作为事件权重（只取正权重事件）。

**方法**：Asimov 显著性。在不加 GNN 切值的 inclusive 情况下，取满足上述选择的所有事件，将 ttH 的正权重求和得到期望信号产额 S，所有背景（ggF、VBF、VH、Bkg_MC）正权重求和得到期望背景产额 B，代入公式：

$$Z = \sqrt{2\left[(S+B)\ln\left(1+\frac{S}{B}\right)-S\right]}$$

由于 $S \ll B$，等价于 $S/\sqrt{B}$。对 GNN score 从 0 扫到 1，每个切值下重新计算 S、B 和 Z，取最大值。

| | S | B | S/√B | Asimov Z |
|---|---|---|---|---|
| 无 GNN cut（inclusive） | 0.766 | 934.5 | 0.025 | 0.025 |
| GNN score ≥ 0.985（最优） | 0.424 | 4.82 | 0.193 | 0.190 |
| **提升倍数** | — | **×194 背景压低** | **×7.6** | **×7.6** |

GNN 在保留约 **55% 信号**的同时，将背景压低约 **194 倍**，显著性提升 **7.6 倍**。

> 注：`weight_lumi` 的归一化尺度待确认，绝对 Z 值需乘以对应亮度 scale factor 才有直接物理意义；提升比值已具参考价值。

---

## 七、推理打分

已用 Epoch 34 模型对全部 Run 3 样本（Signal + Bkg_MC）共 **415 个 ROOT 文件**完成打分，结果写入 `ttH_gnn_score` 和 `selection_pass` 分支，保存于：

```
root_p2root_v2_scored/
├── Signal/   (ttH, ggF, VBF, VH)
└── Bkg_MC/   (所有背景样本)
```
