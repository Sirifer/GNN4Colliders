# 2026-09-07 日志：ttH GNN mass 去关联方案讨论

## 今日状态与授权边界

- 已通过 SSH 登录高能所 lxlogin002，项目根目录：/scratchfs2/cms/huangsuyun/GNN4Colliders。
- 今日读取项目说明、配置、训练和绘图代码、已有分析总结及结果；查看了 mllg_gnn_high.png。
- 除本日志外，未修改项目代码、配置、ROOT 样本、图缓存或训练结果；未提交作业、未运行新训练或新诊断。
- 用户允许考虑重新训练，但要求写代码/实际做事前先询问。当前仅确定讨论方向，没有授权实施某个实验。
- 用户决定今天先记录，明天再继续。不要自动提交训练。

## 目标

为 H→Zγ mass fit 做 ttH topology categorization，降低连续背景中 GNN score 与 m_llg 的依赖，减少分数切选造成的 mass sculpting，同时保留 ttH 区分能力。

## 已确认的项目细节

- 配置：root_gnn_dgl/configs/hzg/ttH_vs_other_run3_v2_fold0.yaml。
- 样本：root_p2root_v2_merged；tree_name = merged_SR_Sideband。
- label 1：ttH；label 0：VH、ggF、VBF、TT、DYGto2LG_10to100。
- 模型：models.GCN.Edge_Network，7 个节点特征，hid_size=64，n_layers=4，n_proc_steps=4。
- 输入包含对象 pt/eta/phi、CALC_E、btag/photon ID、pdgId、NODE_TYPE。
- dataset.py 中 CALC_E = pt*cosh(eta)，为无质量近似能量；删除它仍不能阻止从 pt/eta/phi 推断近似质量。
- 当前训练权重 weight_lumi_avg。样本生成脚本实际按过程计算 mean(weight_lumi)，不是文档开头声称的 mean(abs(weight_lumi))；是否与 ROOT 实值一致仍需后续核验。
- training_script.py 对每个类别分别计算加权平均 BCE，再对类别平均。
- tracking 当前包含 fold、训练权重以及 weight_lumi/source_tree_tag；新增 mass tracking 需要同步处理图缓存。
- fold0 配置：fold1 训练、fold0 测试；fold1 模型反向。OOF 必须按 photon_eta_fold % 2 选取。
- add_photon_eta_fold.py 使用 int(photon_eta*10000)，不能误用 event%2。
- jobs/training/hzg_ttH_vs_other_run3_v2_fold0.sbatch 使用 --preshuffle --nocompile --lazy --restart。
- 已阅读 .codex/skills 下 workflow、training、data-preparation 的 SKILL.md；仅用于理解流程。

## 既有结果与判断

来源：tools/analysis_summary_ttH_GNN.txt（2026-08-31），tools/results，以及 tools/significance_diagnostic.py。

- 已有双 fold MC score，目录 root_p2root_v2_scored_ttH_vs_other_run3v2_fold{0,1}_0828，分支 ttH_gnn_score。
- 旧总结报告 high 阈值 0.994757、medium 下阈值 0.928356；120<m_llg<130 的 high 类别背景 Neff≈2。
- 该 high 类别连续背景仅 5 个 TT、1 个 DYG 原始事件；fold0 背景 yield≈0.003，fold1≈0.941，存在负权重和极端权重影响。
- 总结报告 Z 从约0.590到1.004（约+70%），但背景统计太少，不能当作稳健提升或最终 mass-fit 灵敏度。
- results/mllg_gnn_high.png 和 mllg_tthlep_fullrange.png 日期为9月1日，比总结晚。
- 已查看 high 图：波动和统计误差很大；绘图中的 Background 包含 VH/ggF/VBF，因此125 GeV附近结构不能单凭此图归因于连续背景 sculpting。
- 绘图使用当次 Optuna 的 c2_bay，图中没有写阈值；尚未确认与旧总结阈值一致。
- mass 分支在已有分析脚本中确认为 m_llg；已有 mass 图范围100–180 GeV，这不等于已确认实际完整样本范围。
- 旧总结说 Data 只有 fold0 inference；今日未核验是否已有更新。
- 尚未新读取 ROOT 分布、定量测量 sculpting 或生成 mass 权重表。

## 讨论后的实验思路（未实施、未最终批准）

建议第一轮保持输入和网络一致，区分权重改变与 loss 改变：

1. Baseline：已有模型，当前权重与 BCE。
2. A：mass matching only。训练权重 wi*ry(mi)，保留当前每类归一化 loss。
3. B：DisCo only。原分类 loss + lambda * weighted_dCorr(score,m_llg)^2，在连续背景 TT+DYG 上计算正则。
4. C：mass matching + DisCo，视 A/B 结果再考虑。

A 的关键点：
- mass matching 是让两类形状一致；mass flattening 是让质量分布平坦，两者不同。
- ttH 是窄峰，直接 p_bkg/p_ttH 在峰外可能产生巨大权重；无样本处无法通过重加权创造统计。
- 需要先检查分布重叠、分箱、平滑、权重上限和有效统计量。
- 权重表只用训练 fold 拟合。最终物理评估使用原始 weight_lumi。
- 类别 mass 形状相同不保证 score 与 mass 独立。

B 的关键点：
- m_llg 加入 tracking，仅作为 loss 辅助信息，不作为新增 GNN 输入。
- 需要保留过程标识，从 other 中选出 TT/DYG；VH/ggF/VBF 单独检查。
- DisCo 使用非负权重定义，不能直接把 signed MC 权重当概率权重；明确训练与物理评估权重的区别。
- 扫描 lambda，检查分类性能与去关联程度；TT、DYG 分别检查，避免混合后掩盖问题。
- 目标为 P(score>t | mass, continuum background) 近似常数。
- 参考：DisCo 原论文 https://arxiv.org/abs/2001.05310；作者实现 https://github.com/gkasieczka/DisCo。

输入变量修改作为后续可选实验：
- 单删 CALC_E 不够。
- pt/m_llg、角距离、jet/bjet/lepton multiplicity 等可能减少绝对质量信息，但无量纲不保证统计独立。
- 必须核实 Higgs 候选轻子与 ttH 额外轻子的对应，不能直接假定 lep0/lep1 就是候选。
- 第一轮不同时改输入与 loss，以便归因。

另讨论过不重训的条件背景分位数后处理 u=F_B(score|mass)，但高分背景统计不足会影响标定；用户允许重训，目前不优先此方向。

## 明天继续时

- 先向用户确认希望实施 A、B 或 A/B 对照，并明确改动范围，再写代码或提交任务。
- 可复用已有 baseline；无需重新训练 baseline。
- 实施前核验 mass 支持范围、实际权重、过程标识和缓存结构。
- 若获准补充诊断，固定并记录阈值，比较 TT/DYG 的切选前后 mass shape、mass-bin 通过率、误差和有效统计量；兼看宽松阈值。
- 比较新旧模型应尽量在相近 ttH 效率下进行，不直接沿用相同数值 score 阈值；不只看 AUC/简单计数显著性。
- mass 去关联与背景统计不足是两个问题，reweighting/DisCo 不能补足 MC 统计。
- 使用独立配置、缓存、训练和结果目录，保护现有 baseline。
