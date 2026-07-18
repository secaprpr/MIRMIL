# MIR-MIL 统一架构长期优化交接

更新时间：2026-07-18（Asia/Shanghai）

本文档供切换到长期运行电脑后的 Codex Agent 直接接手。当前任务不是继续微调
标量序数 V3，而是研发一个能够同时保住 PANDA、NSCLC，并继承 BRACS3 增益的
统一 MIR-MIL 主架构。本文档是本轮架构研究的最新依据；服务器上其他历史交接
文件可能包含已经完成的下载或 baseline 任务，不应覆盖本文的研究目标。

## 1. 用户目标与不可妥协的标准

用户希望产出顶会论文级的单模型结果，而不是为一个数据集堆模块或只追求 AUC。
主模型必须：

1. 在 BRACS3 上按 Accuracy、Balanced Accuracy、Macro-AUC、Macro-F1 整体竞争
   SOTA，而不是只看 Macro-AUC。
2. 保持原 MIR-MIL 在 PANDA 六分类上的显著优势。
3. 保持原 MIR-MIL 在 TCGA-NSCLC LUAD/LUSC 上的良好结果。
4. 使用独立 seeds 2024/2025/2026；主结果不能使用 seed 概率集成。
5. 强 validation 候选可以直接跑 test，并立即与最强 baseline 比较四项指标。
6. 架构改进必须有统一理论动机，不能是 residual head 或模块的机械堆叠。

任何新版本如果为了 BRACS 明显牺牲 PANDA 或 NSCLC，都不能作为最终升级版。

## 2. 本地与服务器环境

本地项目：

```text
/Users/seca/Documents/mirmil
```

服务器：

```text
fanhao@175.178.238.223 -p 6000
SSH alias: mirmil-server
项目: /data15/data15_5/fanhao/projects/MIRMIL
环境: mamba activate mirmil
Python: /data15/data15_5/fanhao/miniforge3/envs/mirmil/bin/python
```

用户已明确允许：本地项目是最新代码，可以直接同步到服务器并在服务器实验，
不要求另建隔离目录。仍须保留服务器未跟踪文件，不执行 `git reset --hard`、
`git checkout -- .` 或破坏性清理。

开始工作：

```bash
git pull
ssh mirmil-server
cd /data15/data15_5/fanhao/projects/MIRMIL
git pull
source /data15/data15_5/fanhao/miniforge3/etc/profile.d/conda.sh
mamba activate mirmil
nvidia-smi
```

2026-07-18 交接时 8 张 GPU 均空闲，且没有活跃的本任务训练或测试进程；开始新
实验前仍须重新检查。

## 3. 原 MIR-MIL 为什么必须保留

原 MIR-MIL 将 WSI 视为经验测度，通过向量值 patch 编码、全局组成统计、tail
统计和局部 routes 构成约 520 维 slide state，再进行分类。它在 PANDA 上能够
保留多种 Gleason 形态及其比例、异质性和局部模式，因此表现非常强。

PANDA official QC test 三 seed baseline：

| Feature | Model | Acc | BAcc | Macro-AUC | Macro-F1 |
|---|---|---:|---:|---:|---:|
| UNI | MIR-MIL | 0.790548 | 0.745950 | 0.950390 | 0.749487 |
| R50 | MIR-MIL | 0.647668 | 0.604259 | 0.899205 | 0.601903 |

服务器 baseline 汇总：

```text
/data15/data15_5/fanhao/projects/MIRMIL/artifacts/panda_evaluation/aggregate_results.csv
```

NSCLC baseline 根目录：

```text
/data15/data15_5/fanhao/experiments/MIRMIL_NSCLC
```

NSCLC UNI 整体最强是 RRT-MIL，均值约为 Acc 0.957547、BAcc 0.957799、
Macro-AUC 0.990118、Macro-F1 0.957544；单项最高 AUC 是 MO-MIL 0.990859。
NSCLC R50 中原 MIR-MIL 四项均为当前最强：Acc 0.904874、BAcc 0.904618、
Macro-AUC 0.961717、Macro-F1 0.904776。

结论：不能移除原 MIR-MIL 的向量值神经测度/分布表征，只保留单一 severity。

## 4. BRACS3 的问题与 V3 的有效发现

原 MIR-MIL 在 BRACS3 的主要错误集中于中间 atypical 类。原架构在 bag 压缩后才
引入类别语义，容易把决定相邻边界的小病灶证据压掉。V2 pairwise residual、
class token、ordinal residual、cosine head 等尝试经常出现 validation 提升但
test/PANDA 退化，说明继续增加 residual readout 不是可信方向。

V3 `OrdinalRiskMIL` 用单一 patch severity、normalized entropic bag risk 和有序
累计阈值完整替换原模型。修复 decision temperature 使中间类可达后，GAP1.5 是
BRACS3 最强版本：

| Model (UNI) | Acc | BAcc | Macro-AUC | Macro-F1 |
|---|---:|---:|---:|---:|
| V3 GAP1.5 | 0.693487 ± 0.013272 | 0.676027 ± 0.010957 | 0.848500 ± 0.001930 | 0.677408 ± 0.013166 |
| AC-MIL | 0.681992 ± 0.033181 | 0.666969 ± 0.047102 | 0.852852 ± 0.009653 | 0.662019 ± 0.047646 |

V3 相对 AC-MIL 赢 Acc +1.15 pp、BAcc +0.91 pp、F1 +1.54 pp，AUC 低 0.44 pp。
它证明“显式相邻边界/有序风险”能解决 BRACS 中间类，但不证明标量 severity 是
通用 WSI 表征。

复现文件：

```text
docs/BRACS3_V3_GAP15_BEST_RUN.md
experiments/run_bracs_v3_gap15_best.sh
modules/MIR_MIL/ordinal_risk_mil.py
configs/MIR_MIL_V3.yaml
tests/test_ordinal_risk_mil.py
```

开发实验产物：

```text
artifacts/bracs3_v3_targeted_hpo/uni/phase1/MIR_MIL_V3_HPO_GAP15
```

## 5. V3 跨数据集正式反证

运行脚本：

```text
experiments/run_v3_gap15_panda_nsclc.sh
```

PANDA 必须使用与 baseline 一致的 QC official full split：

```text
PANDA_r50_split_v1_full_qc.csv
PANDA_uni_split_v1_full_qc.csv
```

非 QC test 比 QC 多一个 slide，且缺少特征
`3790f55cad63053e956fb73027179707.pt`。脚本已修正为 QC split，不要恢复旧路径。

V3 GAP1.5 official test，三独立 seed：

| Dataset | Feature | Acc | BAcc | Macro-AUC | Macro-F1 |
|---|---|---:|---:|---:|---:|
| PANDA | UNI | 0.668080 | 0.609215 | 0.901498 | 0.613689 |
| PANDA | R50 | 0.489088 | 0.432524 | 0.815763 | 0.430043 |
| NSCLC | UNI | 0.927673 ± 0.019063 | 0.927528 ± 0.019278 | 0.981333 ± 0.007458 | 0.927605 ± 0.019121 |
| NSCLC | R50 | 0.863208 ± 0.009434 | 0.863901 ± 0.009262 | 0.930971 ± 0.005139 | 0.863130 ± 0.009473 |

完整逐 seed 结果：

```text
artifacts/v3_gap15_generalization/panda/r50/official_test/budget_results.csv
artifacts/v3_gap15_generalization/panda/uni/official_test/budget_results.csv
artifacts/v3_gap15_generalization/nsclc/r50/official_test/budget_results.csv
artifacts/v3_gap15_generalization/nsclc/uni/official_test/budget_results.csv
```

相对原 MIR-MIL/最强 baseline，PANDA 下降约 5–17 pp，NSCLC 下降约 1–4 pp。
这是明确的架构性反证，差距不应通过继续调 V3 超参数来解释。

## 6. 已定位的核心失败原因

V3 的不可接受瓶颈位于聚合之前：

```text
x_i -> scalar severity s_i -> one bag risk R -> class thresholds
```

一个标量只能表达单轴严重度：

- BRACS3 三分类近似满足该假设，因此获益。
- PANDA 六分类虽有顺序，但不同 Gleason pattern 和比例构成非线性、多形态轨迹，
  不能压成一个 patch severity 后再恢复。
- NSCLC LUAD/LUSC 是名义类别；虽然二分类最终有一个 log-odds，但从 patch 到 bag
  的单标量 entropic aggregation 过早丢失了区分形态。

因此失败的不是“序数信息必然无效”，而是“标量 patch 风险是充分统计量”的
假设。下一版必须在 bag state 中保留多维测度信息。

## 7. 下一版统一架构研究要求

注意：仓库已有 `BoundaryRiskMIL` 和 `run_bracs_v4_architecture_ablation.sh`，该 V4
方向已经验证较弱，不要复用 V4 名称造成混淆。建议下一主版本暂称 V5，最终命名
在理论确定后再冻结。

优先研究的统一假设是“多维分布风险 + 可弯曲类别几何”，而不是增加第二预测
分支：

1. patch 映射到向量值 latent responses，而非一个 severity。
2. 对经验测度做保持均值、异质性、tail 的统一聚合；可以从 MIR-MIL measure
   state 出发，但要简化为一个连贯的数学算子。
3. 类别在测度空间中形成可学习的原型轨迹/流形：BRACS 可退化为近似一维有序
   边界，PANDA 可形成弯曲的六级轨迹，NSCLC 两原型可自由分离。
4. 序数强度或轨迹曲率应从数据学习，不允许根据数据集名称手工选择模型模式。
5. 整个 deployed classifier 应是一条统一路径。不要把原 logits 与 V3 logits
   相加，也不要再引入一组独立 query/key/value residual heads。
6. 需要给出可证或可清楚分析的性质，例如 permutation invariance、bag-size
   normalization、标量 ordinal risk 作为退化特例、名义分类作为自由几何特例。

不要直接实现未经验证的大模型。先完成：

- 明确表示与决策方程；
- 参数量和复杂度分析；
- 退化特例说明；
- 单元测试和合成任务，验证多形态 bag 与有序 bag 都可表达；
- 再进入正式数据实验。

## 8. 实验准入与停止规则

为避免再次得到 BRACS-only 模型，建议采用交错验证，不允许只在 BRACS 长时间
搜索后才看 PANDA/NSCLC：

### Stage A：实现与合成验证

- 支持任意 `num_classes >= 2`。
- batch-size-one bag 输入、不同 bag size、forward/backward、CPU/GPU、保存加载。
- 合成 ordinal bag、multimodal six-class bag、nominal binary bag 都能拟合。
- 参数量和计算量合理，不依赖类别名或数据集标识。

### Stage B：单 seed 快速门控

统一使用 seed 2024 和固定 split：

- BRACS3 UNI validation：必须至少显示中间类 decision 指标潜力。
- PANDA UNI validation：Macro-AUC 不应低于原 MIR-MIL seed2024 的 0.951178；参考
  moment-token 为 0.958328。
- NSCLC UNI validation/test sanity：不得出现 V3 约 3 pp 的 accuracy/F1 回退。

若 PANDA AUC 下降超过约 0.5 pp 或 NSCLC Acc/F1 下降超过约 1 pp，先定位架构，
不要用大量 HPO 掩盖。

### Stage C：三 seed validation

- seeds 2024/2025/2026，独立训练。
- 同时检查 Acc、BAcc、Macro-AUC、Macro-F1 和 per-class recall/AUC。
- 不允许依靠单一 exceptional seed。

### Stage D：直接 official test

强 validation 候选应按用户要求灵活地直接测试，并与最强 baseline 对比，而不是
只报告 validation。目标门槛：

- BRACS3：至少保持 V3 GAP1.5 的 Acc/BAcc/F1，并争取超过 AC-MIL AUC 0.852852。
- PANDA：至少不低于原 MIR-MIL；理想目标超过四项原结果。
- NSCLC：R50 至少接近原 MIR-MIL，UNI 至少接近 RRT-MIL 的整体表现。

仅当三个数据集都不弱势时，才称为主架构升级。集成可以作为附加分析，但不能
代替统计值最高的单模型主结果。

## 9. 不要重复的方向

- 当前标量 V3 的 risk temperature、threshold gap、decision temperature 继续 HPO。
- `BoundaryRiskMIL`/旧 V4。
- 原 MIR-MIL logits 后叠加 ordinal residual；旧实验三 seed 不稳定。
- pairwise/class-token/cosine/tail 等独立 residual head 继续排列组合。
- 只根据 BRACS validation AUC 选择模型。
- 用 calibration 或 test 后阈值调整声称架构 SOTA。
- ensemble 当作主结果。

历史诊断与负结果：

```text
docs/MIR_MIL_V3_ARCHITECTURE_DIAGNOSIS.md
reports/bracs_failure_analysis.md
reports/general_mil_improvement_plan.md
reports/panda_bracs_comparison.md
```

## 10. 关键代码入口

```text
modules/MIR_MIL/mir_mil.py                 原 MIR-MIL/MT 主体
modules/MIR_MIL/ordinal_risk_mil.py        V3 标量风险模型
modules/MIR_MIL/boundary_risk_mil.py       已拒绝的旧 V4
utils/model_utils.py                       模型构建与 prediction_mode
utils/loop_utils.py                        训练/指标输出适配
experiments/run_benchmark.py               统一训练入口和 variant 注册
experiments/evaluate_checkpoints.py         official test 入口
configs/MIR_MIL.yaml                       原 MIR-MIL 配置
configs/releases/MIR_MIL_MT_V1.yaml        MT V1 冻结配置
configs/MIR_MIL_V3.yaml                    V3 配置
tests/test_mir_mil.py
tests/test_ordinal_risk_mil.py
tests/test_boundary_risk_mil.py
```

工作树包含本轮 V2/V3/V4 研究改动。不要在未审计的情况下删除历史实现，因为它们
是论文消融和失败分析证据。

## 11. 推荐给新 Codex Agent 的首条指令

```text
请完整阅读 CODEX_MIRMIL_UNIFIED_HANDOFF.md，并审计其中列出的原 MIR-MIL、
OrdinalRiskMIL、实验结果和失败方向。目标是设计并验证一个统一单路径 MIR-MIL
新架构：保留原模型在 PANDA/NSCLC 的多形态测度表达，同时继承 V3 在 BRACS
相邻有序边界上的优势。不要继续调标量 V3，不要堆 residual modules。先给出明确
数学形式、退化特例、最小实现与合成验证，然后按 BRACS/PANDA/NSCLC 交错门控
推进实验；强 validation 后直接 test，并与每个数据集最强 baseline 比较 Acc、
BAcc、Macro-AUC、Macro-F1。服务器可直接使用，实验期间持续记录命令、PID、
日志和三 seed 结果。
```

## 12. 当前状态一句话总结

BRACS3 V3 GAP1.5 是有效的序数边界证据，但标量风险在 PANDA/NSCLC 上被正式
反证；下一阶段必须把“有序决策几何”嵌入保留多维神经测度的统一单路径模型，
而不是继续调参或叠加分支。
