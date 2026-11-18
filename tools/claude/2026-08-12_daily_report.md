# 日报 2026-08-12

## 背景

H→Zγ 分析，Phase 1：用 GNN 做 ttH vs everything else 的二分类，
替代现有的 cut-based 分类（5个 SR 区域），最终目标是比较 Z_ML vs Z_cut 的 Asimov 显著性。

框架：GNN4Colliders（arXiv:2412.10665），DGL + PyTorch，Edge_Network 架构。

---

## 今天完成的工作

### 1. ROOT 文件 tree 合并

**目标**：将每个 ROOT 文件中的 10 棵 tree（5个区域 × SR+Sideband）合并为一棵
`merged_SR_Sideband`，写到新目录 `root_p2root_v2_merged/`。

**工具**：
- `tools/merge_trees_all.py`：递归处理所有文件，自动添加 `source_tree_tag`（0-9，记录来源 tree）和 `weight_lumi_abs`（= abs(weight_lumi)，处理负权重）
- `tools/merge_trees.py`：单文件版，用于补跑缺失文件

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

**结果**：
- SLURM job 15697 成功完成，输出 **415 个文件**（Signal/ + Bkg_MC/）
- 3 个文件跳过（DYGto2LG_10to100/2025.root 损坏；TTG1Jets_200/2022preEE.root 和 WW/2022preEE.root 无匹配 tree，属正常）
- 输出路径：`/scratchfs2/cms/huangsuyun/GNN4Colliders/root_p2root_v2_merged/`

**遇到的问题**：
- 集群无 cpu 分区，需用 `--partition=gpu --account=cmsgpu`
- compute 节点无法访问 AFS（`.condarc` 权限拒绝），绕过方法：使用绝对路径 `/publicfs/cms/user/huangsuyun/conda/envs/pytorch/bin/python`，不经过 conda activate

---

### 2. 训练 config 制定

**文件**：`root_gnn_dgl/configs/hzg/ttH_vs_other_run3.yaml`

**关键参数**：

| 参数 | 值 |
|------|----|
| 模型 | Edge_Network |
| hid_size | 64 |
| in_size | 7（节点特征维数）|
| out_size | 1（binary）|
| n_layers | 4 |
| n_proc_steps | 4 |
| epochs | 100 |
| batch_size | 256 |
| lr | 1e-4，gamma=0.99 |
| n_folds | 4（fold 0=test，1/2/3=train）|
| fold_var | event（用 event number 分折）|

**节点定义（16节点，7维特征）**：

| 节点 | 槽位 | 特征 dim4（btag/mva）| 特征 dim5（pdgId）|
|------|------|---------------------|-------------------|
| jet0-5 | 6个 | hzg_btag_score | 0 |
| bjet0-1 | 2个 | hzg_btag_score | 0 |
| lep0-5 | 6个 | 0 | lep_pdgId（11/13）|
| photon | 1个 | photon_id_mva | 22 |
| MET | 1个 | 0 | 0 |

pt=0 的节点自动过滤，实际节点数随事件变化，图为全连接。

**样本**：
- ttH（label=1）：6个 Run3 文件（2022preEE/postEE，2023preBPix/postBPix，2024，2025）
- other（label=0）：VH + ggF + VBF + DY + TT + tW + TTγ + Diboson，共 83 个文件

**事件产额**（合并前统计）：
- ttH：568,206 raw，sum(weight_lumi)=5.22
- Background：392,192 raw，sum(weight_lumi)=19,244
- S/B ≈ 1:3700（物理上合理）

---

### 3. DGL 图生成（prep_data）—— chunk 0 测试

**命令**：
```bash
cd /scratchfs2/cms/huangsuyun/GNN4Colliders/root_gnn_dgl
/publicfs/cms/user/huangsuyun/conda/envs/pytorch/bin/python scripts/prep_data.py \
    --config configs/hzg/ttH_vs_other_run3.yaml \
    --dataset ttH \
    --chunk 0
```

**结果**：
- 成功写入 `graphs/hzg/ttH_vs_other_run3/ttH_0.bin`（664MB）
- 87,074 个 graph，每图 16 nodes × 7 features，label=1
- Tracking 4 列：event, weight_lumi_abs, weight_lumi, source_tree_tag

**注意**：末尾出现 `Skipping save for ttH: no graphs were created` 是框架 bug（误报），
`save_chunk()` 已正确写入文件，`save()` 再次被调用时 `graph_chunks` 为空故误报。
文件实际有效，可忽略此警告。

---

### 4. 完整 prep_data job 脚本

**文件**：`root_gnn_dgl/jobs/prep_data/hzg_ttH_vs_other_run3_prep.sbatch`

顺序执行：
1. ttH chunk 0-3（chunk 0 有缓存自动跳过）
2. other chunk 0-31
3. ttH shuffle/prebatch chunk 0-3
4. other shuffle/prebatch chunk 0-31

**预计时间**：6-10 小时（在 24h 时间限制内）

---

## 当前状态

| 步骤 | 状态 |
|------|------|
| ROOT tree 合并 | ✅ 完成（415 文件）|
| 训练 config | ✅ 完成 |
| ttH DGL graph chunk 0 | ✅ 完成（664MB）|
| ttH DGL graph chunk 1-3 | ⏳ 待提交 |
| other DGL graph chunk 0-31 | ⏳ 待提交 |
| shuffle/prebatch | ⏳ 待提交 |
| 训练 | ⏳ 待开始 |

---

## 下一步

### 立即：提交 prep_data job

```bash
cd /scratchfs2/cms/huangsuyun/GNN4Colliders/root_gnn_dgl
sbatch jobs/prep_data/hzg_ttH_vs_other_run3_prep.sbatch
```

查看进度：
```bash
squeue -u huangsuyun
tail -f /scratchfs2/cms/huangsuyun/GNN4Colliders/logs/prep_run3-<jobid>.out
```

### prep_data 完成后：提交训练 job

参考 `root_gnn_dgl/jobs/training/hzg_ttH_vs_other_2022_eval.sbatch`，
写一个新的训练 sbatch，使用 `--preshuffle` 模式（加载 prebatched .bin 文件）。

训练命令大致为：
```bash
python scripts/training_script.py \
    --config configs/hzg/ttH_vs_other_run3.yaml \
    --preshuffle
```

### 训练完成后：评估显著性

- 用训练好的模型对 SR（signal region）做 inference，得到每个事件的 GNN score
- 对 signal 和 background 分别加权求和（用 weight_lumi）
- 计算 Asimov 显著性：Z = sqrt(2[(S+B)ln(1+S/B)-S])
- 与 cut-based 的 5 个 SR 的 Z 比较

---

## 关键路径

```
root_p2root_v2/          原始 ROOT 文件（10 trees/文件）
    ↓  merge_trees_all.py
root_p2root_v2_merged/   合并后（1 tree/文件: merged_SR_Sideband）✅
    ↓  prep_data.py
graphs/hzg/ttH_vs_other_run3/   DGL .bin 文件（部分完成）
    ↓  training_script.py
trainings/hzg/ttH_vs_other_run3/  训练输出、checkpoints
    ↓  inference + 显著性计算
Z_ML vs Z_cut
```

---

## 重要路径速查

| 内容 | 路径 |
|------|------|
| 合并脚本 | `tools/merge_trees_all.py` |
| 训练 config | `root_gnn_dgl/configs/hzg/ttH_vs_other_run3.yaml` |
| prep_data sbatch | `root_gnn_dgl/jobs/prep_data/hzg_ttH_vs_other_run3_prep.sbatch` |
| 合并后数据 | `root_p2root_v2_merged/` |
| DGL graphs | `graphs/hzg/ttH_vs_other_run3/` |
| 日志 | `logs/` |
| Python 环境 | `/publicfs/cms/user/huangsuyun/conda/envs/pytorch/bin/python` |
