# ttH GNN 分析笔记 — pt_only 版本（质量去相关）

---

# 第一部分：分析概述

## 1. 目标

CMS H→Zγ 分析将事件按 `source_tree_tag` 分为 5 个信号区（SR）：

| tag | 类别 |
|---|---|
| 0 | ttHhad（ttH 强子道）|
| 2 | ttHlep（ttH 轻子道）|
| 4 | untag |
| 6 | VH3l |
| 8 | VHpTmiss |

基于截断的传统方法在 ttH 富集区（ttHhad、ttHlep）信号/本底比极低，统计灵敏度有限。

目标：训练一个二分类图神经网络（GNN），将 **ttH 信号**与 **other**（其他 Higgs 产生模式 + MC 本底）区分开，
用 GNN 分数替代或补充传统截断，提高显著性。

### 为什么要质量去相关？

标准做法是将 Energy（E = √(pT² cosh²η + m²)）作为节点特征之一。
但 E 与事件的不变质量 m_llγ 存在关联，可能导致 GNN 分数在不同 m_llγ 处的响应不均匀，
在做 m_llγ 拟合时引入偏差（sculpting）。

本版本（pt_only）的改动：
- **去掉 Energy 特征**，节点特征数从 7 降为 6
- **轻子和光子的 pT 改用 pT / m_llγ**（无量纲化，消除质量标度）
- jet 和 MET 的 pT 保持原始值（× 0.1 缩放）

这样模型主要依赖粒子间的动量比例关系和角度信息，而非绝对能量标度，
有助于使 GNN 分数与 m_llγ 解耦。

---

## 2. 分类任务

| Label | 样本 |
|---|---|
| 1（信号）| Signal/ttH |
| 0（other）| Signal/VH + Signal/ggF + Signal/VBF + Bkg_MC/TT + Bkg_MC/DYGto2LG_10to100 |

VH/ggF/VBF 作为 label=0 的原因：这些过程以"污染"的形式进入 ttH 富集区，
GNN 的目的是识别 ttH 拓扑结构（多 b-jet + 轻子 + 光子），其余一律视为非信号。

---

## 3. 节点特征（6 个特征，最多 16 个节点）

每个事件被表示为全连接图，节点按类型和 pT 排序：

```
节点布局：jet0–jet5（6）| bjet0–bjet1（2）| lep0–lep5（6）| photon（1）| MET（1）
```

| 特征 | 内容 | 缩放 |
|---|---|---|
| 1 | pT：jets/bjets/MET 用原始值，**轻子和光子用 pT/m_llγ** | × 0.1 |
| 2 | η（MET 填 0）| × 1 |
| 3 | φ | × 1 |
| 4 | b-tag 分数（jets/bjets）；photon_id_mva（光子）；0（轻子/MET）| × 1 |
| 5 | pdgId：轻子 ±11/±13，光子 22，其余 0 | × 0.1 |
| 6 | NODE_TYPE（节点类型整数标签）| × 1 |

图为全连接，无 padding（variable-length graph）。

---

## 4. 模型结构

```
模型类：models.GCN.Edge_Network
hid_size     = 64
in_size      = 6      ← 6个节点特征
out_size     = 1
n_layers     = 4
n_proc_steps = 4      ← message passing 迭代次数
dropout      = 0
```

输出为 sigmoid 分数 ∈ [0, 1]，分数 → 1 表示 ttH-like。

---

## 5. 训练方案

### 2-fold OOF（Out-of-Fold）

为避免训练/测试泄漏，采用 2-fold 交叉验证：

- Fold 划分变量：`photon_eta_fold`（= floor(photon_η × scale) 取整）
- fold0 测试集：`photon_eta_fold % 2 == 0`，由 fold0 模型打分
- fold1 测试集：`photon_eta_fold % 2 == 1`，由 fold1 模型打分
- 两折合并 = 全部事件各打一次分，无泄漏

训练权重使用 `weight_lumi_avg`（每个 process 的 weight_lumi 均值），
避免少量大权重事件（TT、DY）主导梯度更新。
显著性评估时用 `weight_lumi`（单事件权重）。

# 第二部分：具体操作流程

## Step 1：原始数据

MC 原始文件：

```
/scratchfs2/cms/huangsuyun/GNN4Colliders/root_p2root_v2_pt_only/
├── Signal/
│   ├── ttH/    2022preEE.root  2022postEE.root  2023preBPix.root  2023postBPix.root  2024.root  2025.root
│   ├── VH/     （同上）
│   ├── ggF/    （同上）
│   └── VBF/    （同上）
├── Bkg_MC/
│   ├── TT/               （同上）
│   └── DYGto2LG_10to100/ （同上）
└── manifest.json
```

Data 原始文件（无 `photon_eta_fold` branch，推断时用 `photon_eta` 代替）：

```
/scratchfs2/cms/huangsuyun/GNN4Colliders/root_p2root_v2_pt_only_data/
└── 2022preEE.root  2022postEE.root  2023preBPix.root  2023postBPix.root  2024.root  2025.root  2026.root
```
每个文件中的 tree 名：`merged_SR_Sideband`

---

## Step 2：配置文件

```
/scratchfs2/cms/huangsuyun/GNN4Colliders/root_gnn_dgl/configs/hzg/
├── ttH_pt_only_fold0.yaml          ← fold0 训练
├── ttH_pt_only_fold1.yaml          ← fold1 训练
├── ttH_pt_only_data_fold0.yaml     ← fold0 模型对 Data 推断
└── ttH_pt_only_data_fold0_cr.yaml  ← fold0 模型对 Control Region 推断
```

训练 config 核心结构（以 fold0 为例）：

```yaml
Training_Name: hzg_ttH_pt_only_fold0
Training_Directory: /scratchfs2/cms/huangsuyun/GNN4Colliders/trainings/hzg/ttH_pt_only_fold0

Model:
  module: models.GCN
  class: Edge_Network
  args:
    hid_size: 64
    in_size: 6          # 6个特征
    out_size: 1
    n_layers: 4
    n_proc_steps: 4
    dropout: 0

Training:
  epochs: 100
  batch_size: 256
  learning_rate: 0.0001
  gamma: 0.99           # 指数LR衰减

Datasets:
  ttH:
    args:
      fold_var: photon_eta_fold   # fold划分变量
      label: 1
      weight_var: weight_lumi_avg
      node_branch_names:          # 16个节点 × 6个特征
        - [jet0_pt, ..., bjet1_pt, lep0_pt_div_mllg, ..., photon_pt_div_mllg, MET_pt]  # feature1: pT
        - [jet0_eta, ..., photon_eta, 0]                                                 # feature2: eta
        - [jet0_phi, ..., photon_phi, MET_phi]                                           # feature3: phi
        - [jet0_hzg_btag_score, ..., photon_id_mva, 0]                                  # feature4: btag/mva
        - [0, ..., lep0_pdgId, ..., 22, 0]                                              # feature5: pdgId
        - NODE_TYPE                                                                       # feature6: 节点类型
      node_feature_scales: [0.1, 1, 1, 1, 0.1, 1]
      raw_dir: /scratchfs2/cms/huangsuyun/GNN4Colliders/root_p2root_v2_pt_only
      save_dir: /scratchfs2/cms/huangsuyun/GNN4Colliders/graphs/hzg/ttH_pt_only
      tracking_info: [weight_lumi, source_tree_tag]
    folding:
      n_folds: 2
      test: [0]    # fold0模型的测试集
      train: [1]
  other:
    args:
      fold_var: photon_eta_fold
      label: 0
      # 包含：Signal/VH, Signal/ggF, Signal/VBF, Bkg_MC/TT, Bkg_MC/DYGto2LG_10to100
    folding:
      n_folds: 2
      test: [0]
      train: [1]
```

Data 推断 config 的主要差异：
- `fold_var: photon_eta`（Data 无 photon_eta_fold）
- 不含 `weight_var`、`tracking_info`、`folding` 字段
- `raw_dir` 指向 `root_p2root_v2_pt_only_data`

---

## Step 3：图缓存

训练时框架自动将 ROOT 文件转换为 DGL 图，缓存路径：

```
/scratchfs2/cms/huangsuyun/GNN4Colliders/graphs/hzg/ttH_pt_only/
```

首次训练时自动生成，之后直接复用，无需重复转换。

---

## Step 4：训练

SLURM 提交脚本：

```
/scratchfs2/cms/huangsuyun/GNN4Colliders/root_gnn_dgl/jobs/training/
├── hzg_ttH_vs_other_run3_v2_fold0.sbatch   ← 提交 fold0
└── hzg_ttH_vs_other_run3_v2_fold1.sbatch   ← 提交 fold1
```

提交命令：

```bash
cd /scratchfs2/cms/huangsuyun/GNN4Colliders/root_gnn_dgl
sbatch jobs/training/hzg_ttH_vs_other_run3_v2_fold0.sbatch
sbatch jobs/training/hzg_ttH_vs_other_run3_v2_fold1.sbatch
```

job 内实际执行：

```bash
srun /publicfs/cms/user/huangsuyun/conda/envs/pytorch/bin/python \
    scripts/training_script.py \
    --config configs/hzg/ttH_pt_only_fold0.yaml \
    --preshuffle --nocompile --lazy --restart
```

SLURM 资源：`--partition=gpu --qos=cmsnormal --gres=gpu:1 --cpus-per-task=2 --mem=32G`

训练日志：

```
/scratchfs2/cms/huangsuyun/GNN4Colliders/logs/train_run3_fold0-<jobid>.out
```

训练输出：

```
/scratchfs2/cms/huangsuyun/GNN4Colliders/trainings/hzg/
├── ttH_pt_only_fold0/
│   ├── config.yaml
│   ├── training.log
│   └── model_epoch_0.pt … model_epoch_58.pt   （最佳模型：epoch 43）
└── ttH_pt_only_fold1/
    ├── config.yaml
    ├── training.log
    └── model_epoch_0.pt … model_epoch_29.pt   （最佳模型：epoch 14）
```

---

## Step 5：推断（打分）

推断脚本：`root_gnn_dgl/scripts/inference.py`

SLURM 提交脚本：

```
/scratchfs2/cms/huangsuyun/GNN4Colliders/root_gnn_dgl/jobs/inference/
├── hzg_ttH_vs_other_run3_v2_fold0_inference.sbatch   ← MC 打分（fold0 模型）
└── hzg_ttH_pt_only_fold0_inference_cr.sbatch          ← Control Region 打分
```

提交命令：

```bash
cd /scratchfs2/cms/huangsuyun/GNN4Colliders/root_gnn_dgl
sbatch jobs/inference/hzg_ttH_vs_other_run3_v2_fold0_inference.sbatch
```

job 对 6 个样本 × 6 年份逐一循环，每个文件执行：

```bash
python scripts/inference.py \
    --config  configs/hzg/ttH_pt_only_fold0.yaml \
    --target  /scratchfs2/cms/huangsuyun/GNN4Colliders/root_p2root_v2_pt_only/<Sample>/<Year>.root \
    --destination /scratchfs2/cms/huangsuyun/GNN4Colliders/root_p2root_v2_pt_only_scored_ttH_vs_other_run3_fold0/<Sample>/<Year>.root \
    --branch_name ttH_gnn_score \
    --chunks 1 --chunkno 0 --write \
    --tree merged_SR_Sideband
```

Data 推断使用 `ttH_pt_only_data_fold0.yaml`，输入来自 `root_p2root_v2_pt_only_data/`，
输出写入 `root_p2root_v2_pt_only_scored_ttH_vs_other_run3_fold0/Data/`。

---

## Step 6：打分结果目录（本目录）

```
root_p2root_v2_pt_only_scored_ttH_vs_other_run3_fold0/
├── Signal/
│   ├── ttH/                    2022preEE.root … 2025.root
│   ├── VH/                     2022preEE.root … 2025.root
│   ├── ggF/                    2022preEE.root … 2025.root
│   └── VBF/                    2022preEE.root … 2025.root
├── Bkg_MC/
│   ├── TT/                     2022preEE.root … 2025.root
│   └── DYGto2LG_10to100/       2022preEE.root … 2025.root
└── Data/                       2022preEE.root … 2025.root
```

每个文件在原有 branch 基础上新增：**`ttH_gnn_score`**（sigmoid 输出，[0,1]，→1 为 ttH-like）

---

## Step 7：显著性评估（待执行）

OOF 打分完成后，用以下脚本做显著性评估：

```
/scratchfs2/cms/huangsuyun/GNN4Colliders/tools/significance_diagnostic.py
```

在 Higgs 质量窗口（120 < m_llγ < 130 GeV，is_SR==1）内，
用 Optuna Bayesian 优化 + grid scan 寻找最优的三分类截断 (c1, c2)：

- high：score > c2
- medium：c1 < score ≤ c2
- low：score ≤ c1

优化目标：Z_combined = √(Z_high² + Z_medium² + Z_low²) 最大化，
其中 Z_i = √(2 × ((S_i+B_i) × ln(1 + S_i/B_i) − S_i))。
