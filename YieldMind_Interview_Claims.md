# YieldMind 面试与简历数字速查

更新时间：2026-09-22

## 推荐简历四条

- 基于 LangGraph 构建受步数与修订预算约束的 Agent Loop，串联“数据分析→资料检索→方案生成→受控执行→结果校验”，实现失败反馈驱动的方案修订；另在30条离线受控故障注入A/B任务中验证局部恢复与流程重规划，工作流完成率由16.7%提升至83.3%。

- 基于 Function Calling、Pydantic 与 Tool Registry 设计统一工具调用层，通过参数 Schema、ToolPolicy、错误分类和调用 Trace 约束模型行为，支持参数纠错、幂等调用与瞬态异常重试。

- 基于 Qwen3-Embedding、BM25+、Chroma 与 RRF 构建领域知识库，通过结构感知分块、稳定 Chunk ID、版本管理和语料路由，实现混合检索、语料隔离及引用溯源。

- 基于 FastAPI、PostgreSQL 与 Celery/Redis 构建异步任务后端，结合 Docker 沙箱与确定性护栏，实现幂等提交、状态持久化、取消超时及代码安全验收。

`142条分层评测用例`可在面试展开时补充，不建议与第一条的完成率直接并列，以免被误解为142条都参与了故障恢复统计。

## 这些数字分别代表什么

| 数字 | 准确含义 | 证据 | 不能声称 |
| --- | --- | --- | --- |
| 142条 | 37条核心Agent/工具/工作流 + 30条检索 + 32条路由 + 13条Repair Memory + 30条故障恢复；这是多类固定评测之和 | 各固定评测集和JSON报告 | 142条人工标注、真实用户或生产任务 |
| 16.7%→83.3% | 30条离线受控故障注入任务上，恢复预算由0改为局部恢复/流程重规划各1次后的整体完成率 | `recovery_eval_20260921_130839.json` | LLM代码修复率、生产完成率、领域Agent真实修订成功率 |
| 20/20 | 评测集中明确可恢复的故障全部恢复 | 同上 | 任意错误都能修复 |
| 100%路由准确率 | 30条case中的`none/local_repair/replan/fatal`均符合预期 | 同上 | 未知生产分布上的路由准确率 |
| 151 passed | 当前单元/集成工程回归全通过 | `python -m pytest -q` | 151条评测集任务 |
| RMSE 0.2839 / R² 0.9943 | 200条增强开发数据的5折OOF结果 | 数据血缘与模型报告 | 独立业务Holdout或生产泛化结果 |

## 两种 Agent Loop 不要混淆

### 离线 LangGraph 恢复循环

```text
工具节点失败
  -> Manager 读取 failure_kind
  -> local_repair / replan / finish
  -> 重跑目标节点或数据准备链路
```

- `local_repair`：调整有界参数或重跑候选、产物、报告节点。
- `replan`：更新随机种子，放弃失败的生成数据结果，从数据准备重新开始。
- `16.7%→83.3%`测的是这条循环。

### 领域多 Agent 修订循环

```text
CandidateAgent -> ModelAgent -> OperationAgent -> review
       ^                                      |
       +------------ revision_feedback <-----+
```

- OperationAgent真实执行代码并产出指标/预测/模型产物。
- review不通过且仍有`n_revise`预算时，Manager生成修订反馈，回到Candidate/Model/Operation。
- 单次真实LLM端到端验收`run_6f40a7d9d715`已通过，但没有可声称的多样本真实代码修复率。

## Manager 和 LangGraph 分工

- LangGraph：合法节点、条件边、状态、回环预算、取消和恢复。
- Manager：领域上下文、执行前审批、review、revision feedback和终止判断。
- 专业Agent：数据、搜索、候选、模型方案和受控执行。

一句话：

> LangGraph限定“可以怎么走”，Manager决定“根据当前结果应该怎么走”，Agent负责“具体怎么做”。

## Pydantic 参数自修正是另一层

```text
模型生成工具调用
  -> Pydantic校验
  -> 字段错误转为结构化tool error
  -> 允许模型修正一次
  -> 仍不合法则终止
```

这解决的是JSON、字段缺失、类型和范围错误，不是Python训练代码修复。

## 面试时必须主动说的边界

- 30条A/B是真实本地工具+受控故障，不调用LLM。
- Repair Memory真实复用episode当前为0，修复成功率仍为`null`。
- Qwen3 embedding与BM25+ RRF在30条项目内query上表现良好，但不是生产RAG评测。
- 单人完成的32行分块Top-1盲审中，v3/v4直接相关率均为60.0%、至少部分相关率均为83.3%，没有观察到v4准确率收益；它不是双人裁决Gold，也不能推导Top-5质量。
- 200条数据是增强开发集，Table 6是跨域文献锚点，不能当作同域业务Holdout。

## 复现命令

```bash
/opt/anaconda3/envs/amla/bin/python scripts/run_yieldmind_recovery_eval.py
/opt/anaconda3/envs/amla/bin/python scripts/run_yieldmind_repair_memory_eval.py
/opt/anaconda3/envs/amla/bin/python scripts/export_yieldmind_repair_memory_outcomes.py
/opt/anaconda3/envs/amla/bin/python -m pytest -q
```
