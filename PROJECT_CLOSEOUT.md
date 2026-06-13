# OT-MIL 项目收尾报告

日期：2026-06-13

## 最终状态

OT-MIL 已从概念原型推进到具备完整训练、验证、封存测试、患者级聚合、
bootstrap 统计、数据 provenance 和回归测试的研究实现。当前仓库可以作为
可复现实验档案和后续方法研究的代码基础。

本轮研究到此收尾。原因不是代码无法运行，而是经过多个数据集、任务类型和
理论变体探索后，现有机制没有形成稳定、可泛化的强基线优势。继续在现有测试
集上调参会增加过拟合研究结论的风险，边际研究价值已经较低。

## 核心结论

1. OT 诊断子测度是一个有解释力的建模视角，但不是天然优于 attention pooling。
2. OT-MIL 在 PANDA 上稳定优于仓库内 MO-MIL，在 TCGA-NSCLC 上均值更高。
3. Diagnostic class-mass 在 TCGA-RCC 上显著改善同代码 OT baseline，这是当前
   最可信的机制性正结果。
4. Camelyon16 和 STAD 是明确负结果，说明稀有局灶证据和部分分子亚型任务并不
   适合当前聚合机制。
5. BRCA-PAM50 上 OT-MIL 与 AB-MIL 接近，但没有形成优势。
6. “OT-MIL 更偏好多分类任务”不成立。类别数量本身不是 RCC 正结果的原因。
7. RCC+DLBC 存在器官和项目混杂，只能作为 sanity check，不能作为主要证据。

## 最终结果摘要

以下为三种子正式 macro-AUC。PANDA、Camelyon16 和 NSCLC 行沿用各自原始
正式评估口径；RCC、STAD 和 BRCA 行为 patient-level 聚合结果。不同实验的
特征、预算和统计口径见完整报告。

| 数据集/任务 | OT-MIL | 主要 baseline | 判断 |
| --- | ---: | ---: | --- |
| PANDA，多分类 | 0.9051 | MO-MIL 0.8996 | 稳定正结果 |
| Camelyon16，二分类 | 0.7173 | MO-MIL 0.8116 | 明确负结果 |
| TCGA-NSCLC，二分类 | 0.9235 | MO-MIL 0.9160 | 均值正向，CI 跨零 |
| TCGA-RCC，三分类 diagnostic-mass | 0.9937 | MO-MIL 0.9904 | 最强正结果 |
| TCGA-STAD，四分类 diagnostic-mass | 0.8330 | AB-MIL 0.8943 | 明确负结果 |
| TCGA-BRCA PAM50，四分类 diagnostic-mass | 0.8924 | AB-MIL 0.8953 | 接近但未超过 |

RCC diagnostic-mass 相对同代码 OT baseline：

- macro-AUC `+0.00546`，95% CI `[+0.00078, +0.01136]`
- macro-F1 `+0.01976`，95% CI `[+0.00068, +0.04015]`

这支持“class-conditional transported mass 在特定形态可分任务中有效”，不支持
“OT-MIL 普遍优于其他 MIL”。

## 已完成的工程工作

- 数值稳定的 unbalanced Sinkhorn 与 mass-faithful transport
- learned、class-conditional 和 binary evidence gate
- sufficiency、necessity、minimality、diversity 和 class-mass 目标
- 确定性 patient-level split 与 4096-patch feature cache
- test-column sealing 和 frozen-checkpoint split override
- slide-level 与 patient-level 评估
- paired stratified bootstrap 与任务类型分析
- Hugging Face 大文件断点下载、压缩包校验和 provenance manifest
- AB-MIL、MO-MIL、原始 OT-MIL、class-mass OT-MIL 公平 benchmark runner
- 91 项自动化测试

## 公平性与局限

- 主要正式实验使用 seeds 2024/2025/2026，并由 validation macro-AUC 选模型。
- 新增 STAD 与 BRCA 实验在训练时隐藏 test 列，配置冻结后只评估一次测试集。
- 同一比较中的模型共享 split、patch budget、采样和 patient partition。
- MO-MIL 使用仓库内 dependency-free PyTorch sequence layer，并非官方 Mamba2
  最强实现。
- 仓库没有冻结的环境 lockfile；`pathowm` mamba 环境是实际运行环境。
- 数据特征、checkpoint 和大型日志位于仓库外，未提交到 Git。
- COAD/UCEC 经历过较多开发迭代，应视为 post-development audit，而非独立确认。

## 论文判断

当前结果不足以支撑 AAAI 所需的广泛 SOTA claim。若坚持投稿，较诚实的方向是：

> Diagnostic submeasure learning can improve OT-based MIL on selected
> morphology-separable tasks, while its failures reveal identifiable
> boundaries under sparse lesions and weak morphology-label alignment.

但这一方向仍需要新的外部队列、官方强 baseline 和更清晰的理论预测。基于当前
证据，建议停止继续刷现有数据集，将项目归档。

## 保留与弃用

建议保留：

- `configs/OT_MIL.yaml`：通用实现入口
- `configs/OT_MIL_MULTICLASS.yaml`：RCC 正结果对应配置
- `experiments/run_benchmark.py`：公平多模型比较
- `experiments/evaluate_checkpoints.py`：封存测试评估
- `experiments/aggregate_group_predictions.py`：患者级聚合
- `experiments/prepare_cbioportal_subtype.py`：TCGA subtype 数据准备

不建议继续投入：

- 在 COAD、UCEC、STAD、BRCA 已有测试集上继续调 class-mass 权重
- 用跨器官 TCGA project 分类代替同队列 subtype 分类
- 继续堆叠 binary gate、prototype routing 或 competition 正则
- 以类别数解释任务差异

## 复现入口

```bash
mamba activate pathowm
python -m pytest -q
```

通用四模型比较：

```bash
python experiments/run_benchmark.py \
  --split /path/to/train_val.csv \
  --dataset-name DATASET --num-classes 4 \
  --log-root /path/to/logs \
  --models AB_MIL MO_MIL OT_MIL_ORIGINAL OT_MIL_CLASS_MASS \
  --seeds 2024 2025 2026 \
  --epochs 25 --patience 6 \
  --max-instances 4096 --in-dim 1536
```

完整命令、数据路径、SHA-256、失败记录和所有消融结果见
`docs/EXPERIMENT_REPORT_2026-06-10.md`。

## Git 收尾

项目从 2026-06-09 开始，共经历 77 个提交后进入收尾阶段。关键阶段提交：

- `4979409`：初始 OT-MIL 原型
- `e4c0f90`：稳定 UOT、评估和诊断
- `8159d1a`：可复现 benchmark runner
- `270e7d1`：mass-faithful transport
- `2a05ba5`：class-conditional submeasure
- `159a1b9`：class-mass supervision
- `399cf51`：可复现 TCGA subtype 队列
- `db557b4`：统一多分类 baseline 协议
- `9fc2864`：STAD/BRCA 外部多分类结果

最终原则：保留代码和负结果，停止在当前 benchmark 上继续调参。
