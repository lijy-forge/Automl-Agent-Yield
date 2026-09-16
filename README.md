# Yield-Stress AutoML Agent

> **关于本仓库**
>
> 本项目基于开源框架 [AutoML-Agent](https://github.com/DeepAuto-AI/automl-agent)
> (*AutoML-Agent: A Multi-Agent LLM Framework for Full-Pipeline AutoML*, ICML 2025)
> 二次开发，面向高固含浆料屈服应力预测场景重构了编排层、搜索空间与验收护栏。
> 上游框架的版权与许可归原作者所有。
>
> **本仓库为代码展示版**，以下内容未包含在公开仓库中：
>
> - `agent_workspace/` — 运行产物、实验数据与第三方基线代码
> - `docs/` 及 `*.docx` — 课题技术文档与汇报材料
> - `static/`、`prompt_agent/WizardLAMP/*.jsonl` — 上游项目主页资源与 LoRA 训练语料
>
> 因此仓库内代码不可直接端到端运行；如需了解系统设计，建议阅读
> `run_yield.py`（有限状态机编排）与 `operation_agent/yield_guardrails.py`（确定性护栏层）。

This repository is now focused on high-solid-content slurry yield-stress
prediction. The active workflow is managed by `YieldAgentManager`: it profiles
data, searches external rheology/ML sources, builds candidate mechanisms and
models, asks the OperationAgent to generate runnable training code, and verifies
the generated run with deterministic guardrails. If OperationAgent exhausts its
per-round repair attempts, the manager returns to CandidateAgent/ModelAgent with
failure feedback before trying the next managed revision round.

## Main Entry Points

```bash
python run_yield.py
```

Common options:

```bash
python run_yield.py \
  --data-path /path/to/train.csv \
  --test-path /path/to/anchor.csv \
  --run-dir agent_workspace/runs/yield_search_manual \
  --external-search \
  --n-revise 2
```

Live dashboard:

```bash
python run_yield_live.py --port 5052
```

The dashboard serves `templates/yield_dashboard.html` and invokes
`run_yield.py` in the background.

## Active Modules

- `run_yield.py`: owns `YieldAgentManager`, the yield-specific manager that
  orchestrates synthetic data preparation, external search, candidate
  generation, model planning, code generation, verification, and manager-level
  revision.
- `operation_agent/`: generates and executes yield-stress training scripts.
- `operation_agent/yield_guardrails.py`: verifies mandatory metrics,
  predictions, model artifacts, preprocessing artifacts, mechanism reports, and
  synthetic-data audits.
- `knowledge/yield_schema.py`: normalizes supported yield-stress CSV layouts.
- `knowledge/yield_synthetic_data.py`: creates reproducible low-fidelity
  synthetic training data and a real Table 6 anchor CSV.
- `knowledge/yield_retriever.py`: retrieves and ranks yield-stress sources.
- `knowledge/yield_baselines.py`: evaluates fixed reference baselines.
- `live_dashboard.py` and `templates/yield_dashboard.html`: yield-only live UI.

## Required Outputs

Each successful generated run saves artifacts under its run directory:

- `metrics/metrics.json`
- `predictions/yield_predictions.csv`
- `trained_models/`
- `preprocessing/`
- `logs/mechanism_report.json`
- `logs/synthetic_data_report.json` when synthetic data are active

The generated training script must print `Yield pipeline completed successfully`
only after those artifacts are written and verified.
