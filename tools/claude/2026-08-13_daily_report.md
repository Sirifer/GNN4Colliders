# 日报 2026-08-13

## 背景

H→Zγ 分析，Phase 1：用 GNN 做 ttH vs everything else 的二分类，
替代现有的 cut-based 分类（5个 SR 区域），最终目标是比较 Z_ML vs Z_cut 的 Asimov 显著性。

框架：GNN4Colliders（arXiv:2412.10665），DGL + PyTorch，Edge_Network 架构。

---

## 今天完成的工作

### 1. DGL 图生成（prep_data）——发现并修复 padding 节点 bug

**问题**：训练 job 16344 在第一个 epoch 结束后崩溃，报错：

```
ValueError: Input contains NaN.
```

**排查过程**：
- Test fold 的 .bin 文件里有 **1,745,206 个 Inf 值**，全部集中在 dim 3（CALC_E）
- 原因：ROOT 文件中空槽的 padding 值是 `pt = -999`，`eta = -999`
- `CALC_E = pt × cosh(eta) = -999 × cosh(-999) = -Inf`
- Inf 经过 MLP → NaN scores → `roc_auc_score` 崩溃

**为什么没被过滤掉**：

`dataset.py:212` 原来的过滤条件是：
```python
features = features[features[:,0] != 0]   # pt != 0
```
padding 节点 pt=-999（乘 scale 0.1 后=-99.9），-99.9 ≠ 0，所以没被过滤，留在了图里。

**各 branch 的 padding 情况**（检查 2000 个事件）：

| Branch | padding 值 | 能判断 padding？ |
|--------|-----------|-----------------|
| jet5_pt, bjet1_pt, lep2-5_pt | -999 | ✅ 唯一可靠指标 |
| eta, phi | -999（padding）/ 合法负值（真实粒子）| ❌ 不可靠 |
| pdgId | -999（padding）/ -11,-13（真实 lepton）| ❌ 不可靠 |

结论：**pt < 0 唯一对应 padding**，真实粒子 pt 永远 > 0。
特别地，lep2-5 在所有事件中 100% 是 padding（此分析最多 2 个 lepton）。

**修复**：
```python
# dataset.py:212，改一个字符
features = features[features[:,0] > 0]   # pt > 0
```

**处理方式**：新建 `_v2` 目录，不删旧文件：
- Config `save_dir` → `graphs/hzg/ttH_vs_other_run3_v2/`
- Config `Training_Directory` → `trainings/hzg/ttH_vs_other_run3_v2/`

**重新提交 prep_data**：

```bash
sbatch jobs/prep_data/hzg_ttH_vs_other_run3v2_prep.sbatch
# job 16376，gpu020，耗时约 7.5 小时
```

输出：108 个 .bin 文件（和 v1 相同结构，但不含 padding 节点）。

---

### 2. GNN 训练

**提交**：

```bash
sbatch jobs/training/hzg_ttH_vs_other_run3_train.sbatch
# job 17180，gpu022
```

**训练过程**：

| Epoch | Test_AUC | Test_Loss |
|-------|----------|-----------|
| 0 | 0.9244 | 0.3864 |
| 10 | ~0.96 | - |
| 34 | **0.9717** | **最低** |
| 62 | 0.9699 | 0.2237 |

- Early stopping 在 epoch 62 触发（patience=15，连续 15 个 epoch Test_Loss 不改善）
- **最佳 epoch：34**，Test_AUC = **0.9717**（由框架自动选择）
- 每个 epoch 约 104 秒，总耗时约 1.9 小时

**训练产物**：`trainings/hzg/ttH_vs_other_run3_v2/`
- `model_epoch_0.pt` ~ `model_epoch_62.pt`（共 63 个 checkpoint）
- `training.log`

---

### 3. 整体流程文档

写了 `tools/hzg_ttH_gnn_workflow.md`，记录了从 ROOT 合并到显著性计算的完整流程和命令速查。

---

### 4. Inference 提交（进行中）

**单文件测试**（登录节点）：

```bash
python scripts/inference.py \
    --config configs/hzg/ttH_vs_other_run3.yaml \
    --target .../Signal/ttH/2022preEE.root \
    --destination .../root_p2root_v2_scored/Signal/ttH/2022preEE.root \
    --branch_name ttH_gnn_score \
    --chunks 1 --chunkno 0 --write --tree merged_SR_Sideband
# 成功：9346 events，耗时 75s，自动选 epoch 34
```

**批量 inference 提交**：

```bash
sbatch jobs/inference/hzg_ttH_vs_other_run3_inference.sbatch
# job 17439，排队中（Priority）
```

脚本逻辑：遍历 `root_p2root_v2_merged/` 下所有 415 个 .root 文件，逐个打分，
输出到 `root_p2root_v2_scored/`（镜像目录结构），已存在的文件自动跳过。
每个文件约 75 秒，预计总耗时约 9 小时。

---

## 当前状态

| 步骤 | 状态 |
|------|------|
| ROOT tree 合并 | ✅ 完成（415 文件）|
| DGL 图生成 v1（含 padding bug）| ⚠️ 存档，不使用 |
| DGL 图生成 v2（修复后）| ✅ 完成（108 文件）|
| GNN 训练 | ✅ 完成（best AUC=0.9717，epoch 34）|
| Inference（415 文件打分）| 🔄 job 17439 排队中 |
| 显著性计算 | ⏳ 待做 |

---

## 下一步

### Inference 完成后：计算显著性

写 `tools/compute_significance.py`，逻辑如下：

1. 读取 `root_p2root_v2_scored/` 下所有文件
2. 用 `source_tree_tag` 选出 SR 事件（tag = 0/2/4/6/8）
3. **Z_ML**：设 `ttH_gnn_score` 阈值，统计 S 和 B，计算 Asimov Z
4. **Z_cut**：按原有 5 个 SR（tag 0/2/4/6/8 各自）分别计算 Z，取总 Z = sqrt(ΣZ_i²)
5. 比较 Z_ML / Z_cut

Asimov 显著性公式：
```
Z = sqrt( 2 * [(S+B)*ln(1+S/B) - S] )
```

---

## 重要路径速查

| 内容 | 路径 |
|------|------|
| 训练 config | `root_gnn_dgl/configs/hzg/ttH_vs_other_run3.yaml` |
| 训练产物 | `trainings/hzg/ttH_vs_other_run3_v2/` |
| DGL graphs v2 | `graphs/hzg/ttH_vs_other_run3_v2/` |
| inference sbatch | `root_gnn_dgl/jobs/inference/hzg_ttH_vs_other_run3_inference.sbatch` |
| inference 输出 | `root_p2root_v2_scored/` |
| 日志 | `logs/infer_run3-17439.out`（当前 job）|
| Python 环境 | `/publicfs/cms/user/huangsuyun/conda/envs/pytorch/bin/python` |

---

## 关键路径

```
root_p2root_v2_merged/   合并后 ROOT（merged_SR_Sideband）✅
    ↓  prep_data.py（pt > 0 过滤）
graphs/hzg/ttH_vs_other_run3_v2/   DGL .bin 文件 ✅
    ↓  training_script.py
trainings/hzg/ttH_vs_other_run3_v2/  best epoch=34, AUC=0.9717 ✅
    ↓  inference.py（job 17439）
root_p2root_v2_scored/   带 ttH_gnn_score branch 的 ROOT 🔄
    ↓  compute_significance.py
Z_ML vs Z_cut
```
