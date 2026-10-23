# benchmarks

屈服应力预测的基线对比、消融与诊断实验。主流程入口仍是根目录的 `run_yield.py`，这里的脚本用来回答"效果从哪来、是否稳健"。

在**项目根目录**以模块方式运行：

```bash
python -m benchmarks.run_yield_lf_baseline_and_ensemble_audit
```

> PINN 基线来自同学提供的原始多保真模型（`agent_workspace/external_baselines/`）。下列实验都不修改原始模型代码，只在外层做数据适配、诊断或替换单个组件。选择统一只用 HF eval，HF test 只做报告。

## 数据准备

| 脚本 | 作用 |
|---|---|
| `prepare_yield_stepwise_splits.py` | 为 12 步 LF/HF 数据生成配对划分和分组留出划分 |
| `prepare_yield_new_csv_classmate_split.py` | 把新版 CSV 适配成原始 PINN 脚本需要的字段结构，并做确定性的 HF/LF 划分，不改标签 |

## 基线对比

| 脚本 | 回答的问题 |
|---|---|
| `run_yield_pinn_baseline_compare.py` | AutoML 结果与专家 PINN 基线相比如何 |
| `run_yield_classmate_pinn_original_compare.py` | 在当前数据上复现同学的原始 PINN 流程，作为最接近原版的对照 |
| `run_yield_lf_only_baseline_compare.py` | 只用低保真数据的简单模型（Ridge）在 HF test 上表现如何 |
| `run_yield_lf_baseline_and_ensemble_audit.py` | LF/HF 数据一致性、统一简单基线、划分敏感性（30 次重采样）；不需要 LLM 和 torch，可直接运行 |
| `evaluate_yield_stepwise_split_baselines.py` | 在分步 LF/HF 划分上评估轻量表格基线 |
| `evaluate_yield_new_csv_split_sklearn_baselines.py` | 在新版 CSV 划分上评估 sklearn 基线 |

## PINN 消融

| 脚本 | 变化的部分 |
|---|---|
| `run_yield_pinn_branch_ablation_experiment.py` | 过程历史分支：原版 / 简化 / 去掉分支 |
| `run_yield_pinn_gate_fusion_ablation.py` | 三分支融合策略（只换融合头） |
| `run_yield_pinn_neural_gate_ablation.py` | 门控网络结构（融合位置保持原版） |
| `run_yield_pinn_bilinear_m1_combo.py` | 组合两个正向发现：双线性门控 + m1 层融合 |
| `run_yield_pinn_bilinear_m1_stability.py` | 降低双线性门控的有效学习率，看稳定性 |

## 诊断与校准

| 脚本 | 回答的问题 |
|---|---|
| `run_yield_pinn_branch_diagnostics.py` | 屏蔽某一组输入分支后预测怎么变（不重训） |
| `run_yield_pinn_calibration_compare.py` | PINN 预测是否有可用偏置 / 仿射校准修正的系统偏差 |
| `run_yield_pinn_lfhf_calibration_compare.py` | 基于 LF 的简单预测器能否胜过保存的 PINN 预测 |
| `run_yield_pinn_residual_compare.py` | 在 PINN 之上加保守的残差修正是否有效 |
| `run_yield_pinn_seed_ensemble_stability.py` | 多随机种子集成的稳定性 |
