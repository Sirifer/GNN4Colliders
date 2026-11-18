# 2026-10-18 日志：方法二（CR 归一化）背景估计调研

## 今日目标

评估方法二的可行性：用 inverse photon ID CR Data 的质量形状，归一化到 nominal Data sideband，预测 120–130 GeV 的背景产额，与方法一（sideband 函数拟合）交叉检查。

---

## 已确认的事实

### 数据文件结构

| 路径 | 内容 |
|------|------|
| `root_p2root_v2_pt_only_scored_ttH_vs_other_run3_fold0/Data/*.root` | nominal scored Data；tree = `merged_SR_Sideband`；含 `ttH_gnn_score` |
| `/publicfs/cms/user/laipeizhu/Eos/HZgamma/root_p2root_v2/Data/Data/*.root` | 原始数据；包含多个 tree，其中 `CR_lowIDMVA__inclusive` 就是 inverse photon ID CR |
| `tools/code/prepare_pt_only_inputs.py` | 给原始文件加 `*_pt_div_mllg` 派生变量，默认处理 `merged_SR_Sideband` tree，**不处理 CR tree** |

### CR 原始文件情况

- **树名**：`CR_lowIDMVA__inclusive`（以及按分析类别细分的 `CR_lowIDMVA__ttHhad` 等）
- **`is_CR_low_photon_IDMVA`**：CR tree 中全为 1，SR tree 中全为 0，两者严格互斥
- **所有 GNN 输入分支均存在**（pt/eta/phi/btag/photon_id_mva/pdgId 等）
- **没有 `ttH_gnn_score`**：CR 树从未经过 GNN 推理
- **Run3 CR 事件总数**（GNN cut 前）：约 49,356 个

### nominal scored Data 的 `is_CR_low_photon_IDMVA` 全为 0 的原因

`prepare_pt_only_inputs.py` 只处理 `merged_SR_Sideband` tree，CR tree 从未被合并进来。现有 scored 文件里不包含任何 CR 事件。

### GNN 输入中含 `photon_id_mva` 的影响

CR 事件的 `photon_id_mva` 分布与 nominal 不同（CR 定义为 low photon MVA），GNN score 分布会偏低，通过高 c2 cut 后的 CR 事件数可能很少。这是方法二核心的不确定性，需要在推理后验证。

### 当前 c2

从 `results_sideband_fit/gnn_category_result.json` 读取：`c2 = 0.9940773138257518`

---

## 今日诊断结果

在 nominal scored Data（`merged_SR_Sideband` tree，所有 `is_CR_low_photon_IDMVA == 0`）中，用 c2 = 0.9941 统计：

| 区域 | 事件数（GNN high cut 后） |
|------|------|
| nominal SB 左（110–120） | 4 |
| nominal SB 右（130–180） | 8 |
| nominal SB 合计 | 12 |
| nominal SR 120–130（诊断） | 4 |
| CR SB（来自该文件） | 0（因为 CR 事件根本不在这个文件里） |

注：之前统计的 29 个 SB 事件使用的是 `c2 = 0.985`（`results_lypt`），现在 Bayesian 优化结果 c2 = 0.9941，cut 更严，SB 只剩 12 个。

---

## 方法二实现需要做的事

### 第一步：对 CR 数据做 GNN 推理

CR 原始文件已有所有需要的输入特征，**不需要重新训练**，直接用 fold0 模型推理即可。

推理脚本已写好：`tools/run_cr_inference.sh`

关键参数：
```bash
python3 scripts/inference.py \
    --config   configs/hzg/ttH_vs_other_run3_v2_data.yaml \
    --target   <CR_era>.root \
    --tree     CR_lowIDMVA__inclusive \
    --branch_name ttH_gnn_score \
    --write
```

输出目录：`root_p2root_v2_pt_only_cr_lowIDMVA_scored_fold0/Data/`

**但 `prepare_pt_only_inputs.py` 只加了 `*_pt_div_mllg` 变量**，而 `ttH_vs_other_run3_v2` 这个模型的输入实际上**只用 pt/eta/phi/btag/id/pdgId（节点特征）**，不需要 `*_pt_div_mllg`。因此 CR 原始文件可以直接送 inference，不需要先跑 `prepare_pt_only_inputs.py`。

⚠️ **需要先确认**：`ttH_vs_other_run3_v2_data.yaml` 的 fold_var 写的是 `photon_eta`，而 scored 文件里用的是 `photon_eta_fold`。CR 文件里有 `photon_eta` 但没有 `photon_eta_fold`。推理本身不需要 fold 划分，只需要 inference，所以不影响打分，但后续统计时需要用 `photon_eta` 而不是 `photon_eta_fold` 来区分。

### 第二步：打分后做验证统计

打分完成后需验证：

1. CR 通过 GNN high cut（c2 = 0.9941）后 SB 还剩多少事件
2. CR SB 和 nominal SB 的质量形状是否可比较
3. 如果统计量足够，计算 α 和 B_pred

### 第三步：如果 CR 统计量太低

如果 CR 通过 high GNN cut 后事件数接近零，则方法二在 high category 不可行。
此时有两个选项：
- 降低 c2（松弛 GNN cut），用更宽的 high 定义
- 方法二只用于 medium/combined category

---

## 下一步要做的事（按优先级）

1. **运行 CR 推理**：
   ```bash
   cd /scratchfs2/cms/huangsuyun/GNN4Colliders/root_gnn_dgl
   bash ../tools/run_cr_inference.sh
   ```
   预计耗时约几分钟（每个 era 几千到两万个事件）。

2. **统计 CR 通过 GNN cut 的事件数**：
   运行 `tools/diag_cr_counts.py`（已写好，只需指向打分后的 CR 文件）。

3. **验证 CR 质量形状**：
   比较 nominal SB 和 CR SB 的 m_llg 分布形状，判断可迁移性。

4. **如果通过验证**：计算 α、B_pred，与方法一（sideband fit）的 B 对比，计算显著性 Z。

---

## 文件清单（今日新增）

| 文件 | 说明 |
|------|------|
| `tools/run_cr_inference.sh` | 对 CR data 做 GNN 推理的 shell 脚本（待运行） |
| `tools/diag_cr_counts.py` | 诊断脚本，统计 nominal/CR 各区域事件数（已写好） |
| `tools/results_cr_diag/cr_counts.txt` | 诊断结果（当前 CR 全为 0，因为还没推理） |
