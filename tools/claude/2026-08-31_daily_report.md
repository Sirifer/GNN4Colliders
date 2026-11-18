# 2026-08-31 工作日志

## 今日工作内容

### 背景

在 2026-08-25 分析的基础上，今天的主要工作是：
1. 让 Data 文件跑通 GNN 推断（inference）
2. 修复分析脚本中的两个关键 bug（Data sideband 过滤逻辑、fold 划分逻辑）
3. 改进 `check_fold01_test_distribution.py` 支持正确的 3 类别 Z 计算

---

### 一、Data inference 脚本

**跑了什么代码**：
- 脚本：`jobs/inference/hzg_ttH_vs_other_run3_v2_fold0_inference_data.sbatch`
- 代码位置：`/scratchfs2/cms/huangsuyun/GNN4Colliders/root_gnn_dgl/jobs/inference/`
- 输入：`/scratchfs2/cms/huangsuyun/GNN4Colliders/root_p2root_v2_merged_data/Data/`
- 输出：`/scratchfs2/cms/huangsuyun/GNN4Colliders/root_p2root_v2_scored_ttH_vs_other_run3v2_fold0_0828/Data/`

**遇到的问题及解决**：

问题 1：推断时报错 `KeyError: 'photon_eta_fold'`
- 原因：Data 文件中没有 `photon_eta_fold` branch，但 MC 训练 config 里
  `fold_var: photon_eta_fold` 会被插入 `tracking_info`，inference 脚本
  因此尝试从文件中读取该 branch。
- 解决：新建 Data 专用 config
  `configs/hzg/ttH_vs_other_run3_v2_data.yaml`，改动包括：
  - `fold_var: photon_eta_fold` → `fold_var: photon_eta`（Data 中存在）
  - `tracking_info` 置空
  - 移除 `weight_var` 和 `folding` 块
  - `Training_Directory` 不变，仍加载 fold0 模型权重

**完成情况**：job 70869 COMPLETED，输出文件已在 `Data/2022preEE.root`（目前仅 Run3 2022preEE 一个年份）。

---

### 二、修复 significance_optuna_2fold.py 中的两个 bug

**代码位置**：`/scratchfs2/cms/huangsuyun/GNN4Colliders/tools/significance_optuna_2fold.py`

#### Bug 1：Data sideband 加载失败

**现象**：运行时报错：
```
RuntimeError: No usable Data sideband events were found.
```
**原因**：mask 同时要求 `is_SR == 0`（sideband，source_tree_tag 为奇数）
和 `source_tree_tag in [0,2,4,6,8]`（SR 树，偶数），两个条件互斥。

Data 文件 source_tree_tag 分布（2022preEE 验证）：
- 偶数 tag (0,4,6,8)：来自 SR__ 树 → `is_SR == 1`
- 奇数 tag (1,5,7,9)：来自 Sideband__ 树 → `is_SR == 0`

**解决**：
- `DATA_BRANCHES` 去掉 `source_tree_tag`，加入 `weight_lumi`
- `load_data_sideband` 的 mask 去掉 tag 过滤，只保留 `is_SR == 0`
- `load_data_sideband` 改为同时返回 `(weights, scores)`
- `evaluate_3cat` 里 Data sideband 计数改为 `sum(weight_lumi[mask])`
  （Data 的 weight_lumi 全为 1，数值上等同于 raw count，但逻辑更正确）

#### Bug 2：fold 划分使用了错误的变量

**原因**：`test_mask` 中用 `arr["event"] % 2 == parity` 选 test events，
但实际训练的 fold 划分按 `photon_eta_fold % 2` 定义：
- `photon_eta_fold` 是光子 η 乘以 scale 后取整的整数，奇偶性决定 fold
- `photon_eta_fold` 已写入所有 scored ROOT 文件（不是 0/1，是大整数）

错误划分导致的产额不对称：S Y1/Y0=1.18，B Y1/Y0=1.15（约 15% 偏差）

**解决**：
- `BRANCHES` 加入 `"photon_eta_fold"`
- `test_mask` 改为 `arr["photon_eta_fold"] % 2 == parity`

**修复后 fold 对称性**（before cut）：

| 类别       | Y0     | Y1     | Y1/Y0 |
|------------|--------|--------|-------|
| signal     | 1.613  | 1.622  | 1.006 ✓ |
| background | 492.7  | 460.2  | 0.934 ✓ |

结论：fold0/fold1 本身划分对称，2026-08-25 日志观察到的不对称完全是
错误 fold 划分造成的，不是模型问题。

---

### 三、改进 check_fold01_test_distribution.py

**代码位置**：`/scratchfs2/cms/huangsuyun/GNN4Colliders/tools/check_fold01_test_distribution.py`

同步修复了 fold 划分 bug（同上）和 Data sideband tag 过滤 bug。

新增功能：单个 `CUT` 改为 `C1`/`C2` 两阈值，脚本输出改为
high/medium/low 三类分别显示 S、B、Neff、Z，最后合并 Z_combined。

**跑了什么代码**：
```
python tools/check_fold01_test_distribution.py
```

**结果位置**：直接输出到终端（无文件保存），详见今日详细总结
`tools/analysis_summary_20260831.txt`

**当前结果**（c1=0.10, c2=0.9948）：

| Category | S     | B      | Z     |
|----------|-------|--------|-------|
| high     | 1.094 | 0.944  | 0.974 |
| medium   | 2.105 | 416.8  | 0.103 |
| low      | 0.037 | 535.2  | 0.002 |
| **Combined** |  |        | **0.979** |

high category Data sideband：11 个事件（PASS）

**遗留问题**：
- high bin 背景 Neff 仅 2（fold0 B≈0.003，fold1 B≈0.941），
  背景统计量极低，Z 估计仍不稳定
- significance_optuna_2fold.py fold 已修复，需重新运行 Optuna 得到最优 c1/c2
- Data 只跑了 2022preEE，其余年份待跑

---

### 今日结论

1. **fold 划分 bug（关键）**：之前用 event%2 划分 fold 是错的，改为
   photon_eta_fold%2 后，fold0/fold1 产额对称性恢复正常（Y1/Y0≈1.006）。
   2026-08-25 日志中所有 fold0/fold1 产额数字均已失效。

2. **当前最可信 Z = 0.979**（c1=0.10, c2=0.9948，3类别合并），
   但 high bin 背景 Neff 仅 2，估计不稳定。

3. **背景统计不足**是当前最大瓶颈，需增加 MC 样本。

---

### 明日计划

- [ ] 重新运行 significance_optuna_2fold.py（fold 已修复），得到正确最优 c1/c2
- [ ] 对更多 Run3 年份的 Data 跑 inference
- [ ] 调查 high bin 背景 Neff 过低的问题（考虑加入更多背景样本）

---

详细分析总结：`tools/analysis_summary_20260831.txt`
