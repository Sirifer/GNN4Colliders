# H→Zγ ttH GNN 分析完整流程

**目标**：用 GNN 做 ttH vs everything else 的二分类，比较 Z_ML vs Z_cut 的 Asimov 显著性。

---

## 关键路径

```
root_p2root_v2/          原始 ROOT 文件（10 trees/文件）
    ↓  [步骤 1] merge_trees_all.py
root_p2root_v2_merged/   合并后（1 tree: merged_SR_Sideband）✅ 已完成
    ↓  [步骤 2] prep_data.py
graphs/hzg/ttH_vs_other_run3/   DGL .bin 文件              ✅ 已完成
    ↓  [步骤 3] training_script.py
trainings/hzg/ttH_vs_other_run3/  模型 checkpoints         🔄 训练中（job 16344）
    ↓  [步骤 4] inference.py
root_p2root_v2_scored/   带 GNN score branch 的 ROOT 文件
    ↓  [步骤 5] 显著性计算脚本
Z_ML vs Z_cut 比较结果
```

---

## 步骤 1：ROOT tree 合并 ✅ 已完成

将每个 ROOT 文件中的 10 棵 tree（5区域 × SR+Sideband）合并为一棵 `merged_SR_Sideband`。

```bash
# 单次提交，已完成（SLURM job 15697）
# 输出：415 个文件在 root_p2root_v2_merged/
```

新增字段：
- `source_tree_tag`（0-9，记录来源 tree，见 tag 编码表）
- `weight_lumi_abs`（= abs(weight_lumi)，训练用）

**TREE_TAGS 编码**：

| Tag | Tree |
|-----|------|
| 0 | SR__ttHhad |
| 1 | Sideband__ttHhad |
| 2 | SR__ttHlep |
| 3 | Sideband__ttHlep |
| 4 | SR__untag |
| 5 | Sideband__untag |
| 6 | SR__VH3l |
| 7 | Sideband__VH3l |
| 8 | SR__VHpTmiss |
| 9 | Sideband__VHpTmiss |

---

## 步骤 2：DGL 图生成（prep_data） ✅ 已完成

将 ROOT 文件转换为 DGL graph，每个事件 = 一个全连接图（最多 16 个节点）。

```bash
# 已完成，产出 108 个 .bin 文件：
# - ttH_0.bin ~ ttH_3.bin（4个原始 chunk）
# - other_0.bin ~ other_31.bin（32个原始 chunk）
# - ttH_prebatched_padded_*（8个）
# - other_prebatched_padded_*（64个）
# 输出目录：graphs/hzg/ttH_vs_other_run3/
```

节点定义（16节点，7维特征）：jet0-5, bjet0-1, lep0-5, photon, MET。
pt=0 的节点自动过滤，实际节点数随事件变化。

---

## 步骤 3：GNN 训练 🔄 进行中

**Config**：`root_gnn_dgl/configs/hzg/ttH_vs_other_run3.yaml`

模型参数：Edge_Network，hid_size=64，4层，100 epochs，4-fold（fold 0 test, 1/2/3 train）。

```bash
# 已提交，job 16344，在 gpu016 上运行
# 监控：
tail -f /scratchfs2/cms/huangsuyun/GNN4Colliders/logs/train_run3-16344.out

# 关键日志行（正常应出现）：
# Epoch 0 | LR 0.0001 | Loss ... | Accuracy ... | Test_Loss ... | Test_AUC ... | Time ... s
# Epoch Done.
# Done  （或 Early Termination at Epoch N）
```

训练产物路径：`trainings/hzg/ttH_vs_other_run3/`
- `config.yaml`
- `model_epoch_<n>.pt`（checkpoint）
- `model_epoch_<n>.npz`（per-event scores on test fold）
- `training.log`
- `training.png`

**训练完成后验证**：

```bash
tail -n 30 /scratchfs2/cms/huangsuyun/GNN4Colliders/trainings/hzg/ttH_vs_other_run3/training.log
# 检查：epoch 单调递增，Loss/Test_Loss 有限，Test_AUC 合理（>0.5），有 model_epoch_*.pt 文件
```

---

## 步骤 4：Inference（为所有 ROOT 文件打 GNN score）

训练完成后，对所有合并后的 ROOT 文件做 inference，添加 `ttH_gnn_score` branch。

### 4a. 先在单文件测试

```bash
cd /scratchfs2/cms/huangsuyun/GNN4Colliders/root_gnn_dgl

PYTHON=/publicfs/cms/user/huangsuyun/conda/envs/pytorch/bin/python
CONFIG=configs/hzg/ttH_vs_other_run3.yaml
INPUT=/scratchfs2/cms/huangsuyun/GNN4Colliders/root_p2root_v2_merged/Signal/ttH/2022preEE.root
OUTPUT=/scratchfs2/cms/huangsuyun/GNN4Colliders/root_p2root_v2_scored/Signal/ttH/2022preEE.root

${PYTHON} scripts/inference.py \
    --target ${INPUT} \
    --destination ${OUTPUT} \
    --config ${CONFIG} \
    --branch_name ttH_gnn_score \
    --chunks 1 \
    --chunkno 0 \
    --write
```

验证输出：

```bash
${PYTHON} - <<'PY'
import numpy as np, uproot
path = "/scratchfs2/cms/huangsuyun/GNN4Colliders/root_p2root_v2_scored/Signal/ttH/2022preEE.root"
t = uproot.open(path)["merged_SR_Sideband"]
arr = t["ttH_gnn_score"].array(library="np")
print("entries:", len(arr))
print("NaN:", np.isnan(arr).sum())
print("range:", arr.min(), arr.max(), "mean:", arr.mean())
PY
```

### 4b. 批量 inference（所有文件）

写一个 sbatch，对 `root_p2root_v2_merged/` 下所有文件循环跑 inference，
输出到 `root_p2root_v2_scored/`（保持目录结构一致）。

```bash
# 参考脚本写法（待创建）：
# jobs/inference/hzg_ttH_vs_other_run3_inference.sbatch
```

---

## 步骤 5：显著性计算

对 SR 区域（source_tree_tag 为 0/2/4/6/8 的事件）分别计算：

### 5a. 基于 GNN score 划分 SR（Z_ML）

对 ttH（signal）和 other（background）事件：
- 用 `ttH_gnn_score` 设阈值，划定新的 SR
- 用 `weight_lumi` 加权求和得到 S 和 B
- 计算 Asimov 显著性：

```
Z = sqrt( 2 * [(S+B)*ln(1+S/B) - S] )
```

### 5b. 基于原有 cut-based 区域（Z_cut）

用 `source_tree_tag` 选出原有 5 个 SR（tag = 0, 2, 4, 6, 8），
分别对各 SR 计算 Z，取总显著性 Z_total = sqrt(sum(Z_i^2))。

### 5c. 比较

```
Z_ML / Z_cut > 1  →  GNN 优于 cut-based
```

**脚本待写**：`tools/compute_significance.py`

---

## 常用命令速查

```bash
# 看 job 状态
squeue -u huangsuyun

# 看训练日志
tail -f /scratchfs2/cms/huangsuyun/GNN4Colliders/logs/train_run3-16344.out

# 看 training.log（训练完成后）
tail -n 30 /scratchfs2/cms/huangsuyun/GNN4Colliders/trainings/hzg/ttH_vs_other_run3/training.log

# Python 环境
PYTHON=/publicfs/cms/user/huangsuyun/conda/envs/pytorch/bin/python

# 项目根目录
REPO=/scratchfs2/cms/huangsuyun/GNN4Colliders/root_gnn_dgl
```

---

## 目录结构

```
GNN4Colliders/
├── root_p2root_v2/              原始 ROOT（只读）
├── root_p2root_v2_merged/       合并后 ROOT（merged_SR_Sideband）✅
├── root_p2root_v2_scored/       inference 输出（带 ttH_gnn_score branch）
├── graphs/hzg/ttH_vs_other_run3/  DGL .bin 文件 ✅
├── trainings/hzg/ttH_vs_other_run3/  训练产物（进行中）
├── logs/                        SLURM 日志
├── tools/                       自定义脚本（merge, significance 等）
└── root_gnn_dgl/                框架代码
    ├── configs/hzg/ttH_vs_other_run3.yaml
    ├── scripts/
    │   ├── prep_data.py
    │   ├── training_script.py
    │   └── inference.py
    └── jobs/
        ├── prep_data/hzg_ttH_vs_other_run3_prep.sbatch ✅
        └── training/hzg_ttH_vs_other_run3_train.sbatch ✅
```


--preshuffle   使用前面 shuffle_mode 整理好的数据
--lazy         LazyDataset，按需加载，避免一次全塞内存
--nocompile    不做 torch compile
--restart      从头训练