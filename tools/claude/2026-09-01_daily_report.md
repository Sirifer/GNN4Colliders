# 2026-09-01 工作日志

## 今日工作内容

### 背景

延续 2026-08-31 的工作，在 fold 划分 bug 修复后，今天的主要任务是：
1. 改进/新建分析脚本
2. 修复代码逻辑问题
3. 生成训练曲线和 OOF score 分布图
4. 写完整的分析总结文档

---

### 一、significance_diagnostic.py 重构

**代码位置**：`tools/significance_diagnostic.py`

**改动内容**：

在 `significance_optuna_2fold.py` 的基础上，把 `significance_diagnostic.py` 完整重构，使两者逻辑一致：

1. **BRANCHES 加入 `photon_eta_fold`**，Data branches 加入 `TREE_TAG`
2. **`test_mask` 改为 `photon_eta_fold % 2`**（与 `significance_optuna_2fold.py` 一致）
3. **`yield_info` 统一为 Neff-based**（保留 `err`/`nneg` 字段）
4. **`evaluate_gnn` → `evaluate_3cat`**，加入 data sideband 参数
5. **加入 Optuna 贝叶斯优化**（前）+ **保留 grid scan**（后）
6. **`load_data_sideband` 完整移入**，现在返回 `(weights, scores, tags)`
7. **cut-based 加入 DataSB 列**：每个 SR category 统计对应 sideband tag（SR_tag+1）的 Data 事例数

**cut-based DataSB 逻辑**：
- SR tag=0（ttHhad）→ sideband tag=1
- SR tag=2（ttHlep）→ sideband tag=3
- SR tag=4（untag） → sideband tag=5
- 以此类推，验证 Data 覆盖情况

---

### 二、significance_diagnostic.py bug 修复

**修复了两个实质性问题**：

#### Bug 1：cut-based 只用 fold0 数据
**原代码**：
```python
full_tth = apply_mask(tth0, physics_mask(tth0))
full_bkg = apply_mask(bkg0, physics_mask(bkg0))
```
**问题**：cut-based 的 S/B 只有正确值的一半，Z 偏低约 30%。

**修复**：
```python
full_tth = concatenate([apply_mask(tth0, physics_mask(tth0)),
                        apply_mask(tth1, physics_mask(tth1))])
full_bkg = concatenate([apply_mask(bkg0, physics_mask(bkg0)),
                        apply_mask(bkg1, physics_mask(bkg1))])
```

#### Bug 2：`MIN_B_NEFF = 0.0` 死代码
`B["neff"] < 0.0` 永远 False，删除该无效条件。

**修复后 cut-based baseline**（5 个 SR tree，fold0+fold1 合并）：

| Cat       | S      | B        | NeffB   | DataSB | Z     |
|-----------|--------|----------|---------|--------|-------|
| ttHhad    | 1.0327 | 25.832   | 38.1    | 238    | 0.202 |
| ttHlep    | 0.9286 | 2.552    | 5.2     | 32     | 0.551 |
| untag     | 1.1295 | 850.228  | 1869.6  | 7499   | 0.039 |
| VH3l      | 0.1385 | 20.750   | 53.8    | 412    | 0.030 |
| VHpTmiss  | 0.0063 | 53.570   | 104.5   | 1028   | 0.001 |
| **Combined** | | | | | **0.590** |

---

### 三、plot_training_and_scores.py 改版

**代码位置**：`tools/plot_training_and_scores.py`

**改动**：原来的 2×3 合图拆成 3 个独立文件：
- `fold0_training.png`：1×3（Loss / Test AUC / Train Accuracy）
- `fold1_training.png`：1×3（Loss / Test AUC / Train Accuracy）
- `oof_scores.png`：1×2（raw count + normalised，log scale）

fold0/fold1 各自单色（蓝/橙），每图标注最佳 epoch。

**生成文件**：
- `tools/results/fold0_training.png`
- `tools/results/fold1_training.png`
- `tools/results/oof_scores.png`

---

### 四、training.png 用内置方法生成

使用 `scripts/training_script.py --plot` 重新生成标准格式训练图：
```bash
python scripts/training_script.py --config trainings/hzg/ttH_vs_other_run3_0828_fold0/config.yaml --plot
python scripts/training_script.py --config trainings/hzg/ttH_vs_other_run3_0828_fold1/config.yaml --plot
```
**确认**：`plot_log()` 的 accuracy ylim 已为 `(0, 1.02)`（非旧版的 (0.44, 0.56)），图像正常。

---

### 五、Data tree 分布验证

**确认 Data 按 tree 分了**，且 5 个 SR 类别均有 sideband 事例（跨 6 年份合计）：

| SB tag | 类别     | 总计 |
|--------|----------|------|
| 1      | ttHhad   | 238  |
| 3      | ttHlep   | 32   |
| 5      | untag    | 7499 |
| 7      | VH3l     | 412  |
| 9      | VHpTmiss | 1028 |

注：2022preEE 单文件中 ttHlep SB（tag=3）计数为 0，但其他年份补足，共 32 个。

---

### 六、诊断脚本 diagnose_fold_imbalance.py

**代码位置**：`tools/diagnose_fold_imbalance.py`

针对 ttHhad/ttHlep 背景 fold 不对称问题，逐个样本列出 top-20 大权重事例。

**发现**：
- Bkg_MC/TT ttHhad：83 个 raw events，前 20 个事例贡献 80%+ 的 yield
- Bkg_MC/TT ttHlep：仅 10 个 raw events，fold0 3 个/fold1 7 个，极端分配
- 根因是 MC 样本量不足，不是 fold 划分问题

结果文件：`tools/results/fold_imbalance_diag.txt`

---

### 七、完整分析总结文档

**代码位置**：`tools/analysis_summary_ttH_GNN.txt`

包含 16 个章节，整理所有数字：
- motivation、分类任务定义
- 6 个样本的完整 N/Y 统计（raw + 按 tree 按 fold 分解）
- 图结构设计（7 特征）、模型架构（Edge Network GCN）
- 训练设置（超参数、fold 方案、weight 策略）
- 训练结果（fold0/fold1 每 10 epoch 记录）
- Inference 详情（MC + Data）
- OOF score 分布
- 显著度框架和公式
- Cut-based 基准（Z=0.590）
- GNN 3-category 结果（c1=0.928356，c2=0.994757，Z=1.004，+70%）
- 修复的 4 个 bug
- 文件清单 + 下一步计划

---

### 今日结论

1. **`significance_diagnostic.py` 完成重构**：贝叶斯（前）+ grid scan（后），统一用 fold0+fold1 全量 MC 做 cut-based，DataSB 列新增。

2. **训练曲线图生成**：fold0（AUC=0.9695）和 fold1（AUC=0.9723）均已可视化。

3. **分析总结文档完成**：`tools/analysis_summary_ttH_GNN.txt`，16 章节，所有数字均从代码直接生成。

4. **当前最可信数字**：
   - Cut-based Combined Z = **0.590**
   - GNN Combined Z = **1.004**（c1=0.928，c2=0.995，+70%）
   - 但 high bin Neff=2，估计不稳定，需增加背景 MC

---

### 明日计划

- [ ] 运行 `significance_diagnostic.py`（贝叶斯+scan），得到新的最优 c1/c2
- [ ] 补充更多背景 MC 样本（ZGamma、WGamma 等），改善 Neff
- [ ] 对 fold1 Data 跑 inference
- [ ] 调查 ttHlep 负权重事例问题

---

详细总结：`tools/analysis_summary_ttH_GNN.txt`
