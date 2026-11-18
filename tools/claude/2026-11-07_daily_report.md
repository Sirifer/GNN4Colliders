2026-11-07 工作日志
==================
（实际工作时段：2026-11-06 傍晚 → 11-07 凌晨）

任务: v2 样本 + v3 训练的 4-class GNN 的显著性；把 v2 Data 真正分成 4 个 fold

一、做了什么 / 代码在哪 / 结果在哪
---------------------------------
1. v2 版 4 类显著性脚本（前几轮写的，本次继续用）
   /scratchfs2/cms/huangsuyun/GNN4Colliders/tools/significance_4class_v2_bkg_4cat.py
   - 4 类 argmax + 3 个阈值 (c_ttH / c_VH / c_ggFVBF)，optuna TPE 1000 trials
   - SR = even source_tree_tag，SB = odd source_tree_tag，alpha = N(Data SB)/Y(Bkg MC SB)
   - 4 category 平方和合成 sqrt(sum Z_i^2)，alpha 相关性进 sigma_Z；4 fold 再 sqrt(sum Z_k^2)
   - 本次新增 --data-dir（单个 flat 目录、四个 fold 共用一份 Data）

2. v2 Data 的 fold 分支修复脚本（新写）
   /scratchfs2/cms/huangsuyun/GNN4Colliders/tools/fix_v2_data_fold.py
   功能: 读 root_p2root_v2_pt_only_data/<era>.root，把 photon_eta_fold 重算成
         int32(photon_eta*10000)，写到 root_p2root_v2_pt_only_data_foldfix/<era>.root
   自带校验: 条目数不变、其余 361/378 个分支逐分支 array_equal、四个 parity 都在、
             SB 计数不变；每个 era 的旧值/新值/匹配数都写进 foldfix_report.txt
   用法: python tools/fix_v2_data_fold.py --check   # 只报告
         python tools/fix_v2_data_fold.py           # 写

3. Data 打分作业（生成器扩展 + 已提交已跑完）
   /scratchfs2/cms/huangsuyun/GNN4Colliders/tools/make_v2_4class_on_v2_jobs.py
   新增 "DataFix" 家族:
     jobs/inference/hzg_ttH_4class_on_v2_4fold_data_fix_fold{0..3}_inference.sbatch
     输入 root_p2root_v2_pt_only_data_foldfix
     输出 root_p2root_v2_pt_only_scored_ttH_4class_on_v2_4fold_data_fix_fold{k}/Data
     config 复用 ttH_4class_on_v2_4fold_data_fold{k}.yaml（无 weight_var，tracking_info: []）
   作业号 633427 / 633429 / 633430 / 633431，四个全部 "done: 6 new file(s) of 6 expected"
   日志 logs/infer_4class_v2_data_fix_fold{0..3}-*.out

4. 结果
   /scratchfs2/cms/huangsuyun/GNN4Colliders/tools/results_4class_v2_bkg_4cat_foldfix/
     final_z.txt           主结果
     category_yields.csv   每个 (fold, split, category) 的 N_data_sb/Y_bkg_sb/alpha/B/Z
     optuna_fold0..3.csv   四个 fold 的 tuning 记录（四个都自己调了 cut）
   日志 logs/sig_v2_foldfix.log
   跑法:
     python tools/significance_4class_v2_bkg_4cat.py \
       --data-dir <repo>/root_p2root_v2_pt_only_data_foldfix \
       --out-dir tools/results_4class_v2_bkg_4cat_foldfix

二、结果
--------
Data（6 个 era）SB 事件按 parity: 2262 / 2270 / 2347 / 2330，总计 9209
（与修复前总数完全一致，只是重新分桶）

| fold | c_ttH | c_VH | c_ggFVBF | alpha_val | alpha_test | Z_val | Z_test | sigmaZ_test |
|------|-------|------|----------|-----------|------------|-------|--------|-------------|
| 0    | 0.73853 | 0.73675 | 0.33512 | 1.937600 | 1.927516 | 0.246219 | 0.274260 | 0.018675 |
| 1    | 0.69299 | 0.64430 | 0.61707 | 1.981229 | 1.937600 | 0.256847 | 0.236199 | 0.010589 |
| 2    | 0.66269 | 0.80969 | 0.27725 | 1.975069 | 1.981229 | 0.261016 | 0.257179 | 0.012490 |
| 3    | 0.62011 | 0.82669 | 0.50343 | 1.927516 | 1.975069 | 0.256531 | 0.282101 | 0.016240 |

4-fold combined Test Z = 0.526052 +/- 0.015183
  = sqrt(0.27426^2 + 0.23620^2 + 0.25718^2 + 0.28210^2)

per-category test Z（四类）:

| fold | Other  | ttH    | VH     | ggFVBF |
|------|--------|--------|--------|--------|
| 0    | 0.0417 | 0.2213 | 0.1303 | 0.0866 |
| 1    | 0.0548 | 0.1723 | 0.1325 | 0.0744 |
| 2    | 0.0449 | 0.1786 | 0.1606 | 0.0803 |
| 3    | 0.0466 | 0.2104 | 0.1633 | 0.0805 |

即显著性基本来自 ttH 桶（每 fold 0.17-0.22），VH 次之（0.13-0.16），ggFVBF ~0.08，
Other 只贡献 ~0.05。

与之前几组数字对比（同一套 v2 MC、同一套 v3 4-class 模型）:

| 运行 | 4-fold Z | 备注 |
|------|----------|------|
| v3 固定 cut，旧（坏）Data fold | 0.383010 +/- 0.012133 | 已作废 |
| conservative（只 fold 0/3 调 cut） | 0.442951 +/- 0.021092 | 已作废 |
| allfolds（fold 1/2 借 pooled alpha） | 0.448067 +/- 0.021407 | 已作废 |
| **fold-fixed Data，每 fold 自调 cut** | **0.526052 +/- 0.015183** | 采用 |
| v3 参考（v3 样本、7 个 bkg） | 0.682365 +/- 0.012614 | 不同 production |

三、关键发现（本次最重要的一条）
--------------------------------
v2 Data 的 photon_eta_fold 分支是**算错的**，不是"v2 Data 天生只有 2 个 fold"。

证据（逐个事件和 int32(photon_eta*10000) 比）:
  2022preEE.root  n=224   存值只有 {0,1}，parity [102 122 0 0]；重算 [42 48 60 74]；匹配 0/224
  2024.root       n=4346  存值只有 {0,1}，parity [2170 2176 0 0]；重算 [1079 1081 1091 1095]；匹配 0/4346
  2025.root       n=4664  匹配 1/4664
  MC Signal/ttH/2022preEE.root  n=9346  存值 [-25200 -25073 ...]，匹配 9346/9346 ✅

photon_eta 本身是好的，所以 fold 可无损重算。

为什么它把之前所有 v2 数字都压低了:
  坏分支把全部 Data SB（9209）塞进 2 个 parity（每桶约 4600），
  而 Bkg MC SB 仍按 4 个 parity 正常分（每桶约 1170）。
  -> alpha_p = N_data_p / Y_bkg_sb_p 从 ~1.93 翻成 ~3.93（近 2 倍）
  -> SR 里 B_p = alpha_p * Y_bkg_sr_p 也就翻倍
  -> Z 被系统性压低
修复后 alpha 回到 1.93 左右，Z 从 0.443 升到 0.526。

顺带纠正一个之前的判断: "v2 alpha ~3.9 所以和 v3 (~1.70) 不可比" 是坏分支造成的假象。
现在 v2 alpha ≈ 1.93 vs v3 ≈ 1.70，两者真正的差别是 bkg 样本数（2 vs 7）和 preselection，
alpha 数值本身已经不是主要区别。

四、其它踩到/确认的点
--------------------
1. scripts/inference.py 不应用 Datasets[...].folding（folding 在 dataset 的 ignored_keys），
   每个 fold 的输出文件都含全部事件，out-of-fold 靠下游 photon_eta_fold % 4 == k 选。
   所以"分 4 个 fold 打分"只是换 fold 的模型，不换事件集合。
2. inference.py 的 select_dataset_config 只看每个 config 的第一个 dataset，且按 basename
   fnmatch 匹配：MC 的 config 会误配到 Data target，于是 Data 作业继承了
   weight_var: weight_lumi_avg 而崩溃（v2 Data 没有该分支）。Data 必须用单独的
   ttH_4class_on_v2_4fold_data_fold{k}.yaml。
3. MIN_BKG_YIELD=10 只是 optuna/validation 的约束，test 阶段只警告不拒绝。
   v3 自己报的 0.682365 也有 3/16 个 test bin 低于 10；v2 本次同理。
4. sbatch 一次只能吃一个脚本（多给的会被当脚本参数静默忽略），4 个作业要分别提交。
5. 内存文件 memory/hzg-gnn-data-pipeline.md 里"v2 Data 只有 2 个 fold parity"那条已改写为
   "分支坏了 + 修法 + 新结果"，避免以后再把 bug 当成数据属性。

五、待办
--------
1. cut-based 对照：确认 cut-based 那条链用的 fold 定义是否也是（坏）photon_eta_fold；
   若是，需要同样用重算后的 fold 重做，否则两者不是同一个 fold 划分，不能直接比。
2. 若要把 v2 vs v3 的 4 类结论并列，注意两者 bkg 样本不同（2 vs 7），Z 不可直接相减。
3. 可选：用修好的 Data 重跑一遍 MC 侧没有必要（MC 分支本来就对，且本次未动 MC）。
