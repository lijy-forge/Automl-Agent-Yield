# YieldMind Agent Upgrade Plan

更新时间：2026-09-21

## 输入审计

- 本地仓库：`/Users/lijiayao/面试项目/automl-agent-yield`
- 远端地址：`https://github.com/lijy-forge/Automl-Agent-Yield.git`
- 当前分支：`main`
- 当前本地工作区：开始实施前为干净状态。
- `YieldMind_Agent_Upgrade_Plan.md`：开始实施前仓库中不存在，本文件为本轮新增的任务书、差异审计和进度记录。
- 附件情况：用户消息中的《》没有携带明确附件名；仓库根目录存在 `屈服值-设计方案(1).docx` 与 `高固含浆料屈服值预测-算法设计文档(1).docx`，本轮将其作为领域方案参考，但不把文档中的设想计为已实现能力。

## 真实代码与方案差异

| 方向 | 方案/目标 | 真实代码现状 | 本轮处理 |
| --- | --- | --- | --- |
| 可复现运行 | 能离线跑通基础流程并保存证据 | 原 `run_yield.py` 依赖 LLM/运行数据；默认 `python` 环境缺 pandas，`/opt/anaconda3/envs/amla/bin/python` 可用 | 新增确定性 demo 数据、固定 baseline 离线评测脚本和报告路径 |
| 后端与数据库 | FastAPI + 持久化运行/事件/工具调用 | 原项目为 Flask dashboard + JSONL/文件产物 | 新增 FastAPI API、SQLite 迁移、运行/事件/工具调用/评测表 |
| Agent 工具调用 | Pydantic Schema、Tool Registry、参数/输出约束 | 原项目有自定义 contract/guardrails，但缺统一 Tool Registry | 新增 `yieldmind.tools.ToolRegistry`，封装数据生成、画像、baseline、sandbox、产物摘要、真实 pipeline 入口 |
| LangGraph 编排 | StateGraph 节点、边和状态管理 | 原 `run_yield.py` 为自定义状态机 | 新增 `yieldmind.workflow`，安装并验证 `langgraph<0.3` 后可走真实 `langgraph_stategraph` |
| 执行隔离 | 工具/代码执行受限、超时、轨迹记录 | 原 OperationAgent 有进程超时，dashboard 用 subprocess | 新增 `SubprocessSandbox`：无 shell、Python 入口白名单、仓库内 cwd、最小环境、超时终止 |
| 评测 | 不虚构结果，明确模拟/真实调用 | 原有模型/候选 benchmark，但缺 Agent 工具层回归集 | 新增数据集驱动离线评测；报告明确 `real_llm_calls=0`、`simulated_model_calls=0` |
| RAG/知识库 | Chroma、LangChain 切分、可追溯引用 | 原有 `utils/embeddings.py` 使用 FAISS/BM25/BGE，但没有主流程知识库服务 | 已有版本隔离 Chroma、BM25+、RRF hybrid、30条标注查询；Qwen3已完成独立CPU评测并通过显式`retrieve_evidence`节点接入确定性StateGraph |
| 多轮会话/Memory | 会话、跨轮约束、记忆删除 | 原项目以单次运行与文件产物为主 | 新增 session/turn/memory 表和 API；支持约束继承、幂等消息、后续 run 创建 |
| 安全与证据约束 | 脱敏、上下文预算、引用可校验 | 原主流程有部分 guardrails，但缺统一工具层安全控制 | 新增 `redact_sensitive_payload`、`plan_token_budget`、`validate_evidence_refs`，工具调用日志入库前统一脱敏 |
| 展示 | 可查看工具、运行和事件 | 原 Flask 页面服务于旧主流程 | 新增 FastAPI 静态展示页，保留原 Flask dashboard |
| 领域算法和上游说明 | 保留 AutoML-Agent 上游说明和屈服应力领域算法 | README 已说明上游 AutoML-Agent 和活跃模块 | 不移除原有领域模块；新增层只做包装 |

## 面试重点：从 Manager Agent 到 LangGraph 混合编排

### 一句话结论

前后两种流程确实不一样，但不是“用 LangGraph 删除 Manager Agent”。当前领域主流程采用的是**混合编排**：LangGraph 负责可控、可恢复、可审计的外层流程，原 `YieldAgentManager` 继续负责领域上下文、执行前审批、review、revision feedback 和终止判断，各专业 Agent/工具负责具体执行。也就是说，Manager 从“唯一流程控制器”变成了“受图约束的领域协调者”。

### 前后模式对比

| 维度 | 原始 Manager 中心模式 | 当前离线工作流 | 当前领域工作流 |
| --- | --- | --- | --- |
| 核心入口 | `YieldAgentManager.run()` | `YieldMindWorkflow` | `YieldDomainWorkflow` |
| 谁控制下一步 | Manager 内部的 Python 命令式流程 | StateGraph 的固定节点和条件边 | StateGraph 规定合法路径，Manager 提供领域决策和反馈 |
| Agent 形态 | Manager 调用多个内部方法和 `OperationAgent` | 围绕 Tool Registry 的逻辑角色，不默认调用真实 LLM | 通过 `RealYieldDomainAdapter` 恢复并复用原 Manager/Agent 实现 |
| 状态管理 | 主要保存在进程内对象和 JSON 产物中 | 显式 graph state | 可序列化 graph state + 既有 JSON 产物；节点执行时恢复 Manager 上下文 |
| 回环与终止 | 写在 `run()` 的条件分支和循环里 | 显式条件边 | review 失败后在预算内回到 Candidate/Model；取消走显式 `cancel` 节点 |
| 可观测性 | 主要看到整段运行和最终产物 | 可记录每个节点、工具调用和产物 | 可区分图路由、Manager 决策、专业 Agent 行为和 Operation 执行 |
| 恢复与取消 | 粒度偏整段流程 | 节点边界取消和持久化 | 节点级轨迹、任务租约、取消、恢复；Operation 支持进程组终止 |
| 当前验证边界 | 原有领域流程 | 确定性离线流程已验证 | 已完成单次显式授权的真实 LLM 端到端通过验收；未形成生产成功率统计 |

原流程可以概括为：

```text
用户任务
  -> YieldAgentManager.run()
     -> Data/Search/Candidate/Model/Operation
     -> Review
     -> Manager 内部决定是否修订或结束
```

当前领域流程可以概括为：

```text
用户任务
  -> LangGraph（合法状态、条件边、预算、取消和恢复）
     -> Manager（领域上下文、审批、反馈和终止判断）
        -> Candidate / Model / Operation 等专业执行单元
     -> Review
     -> LangGraph 校验后回环、取消或结束
```

### 为什么没有删除 Manager Agent

Manager 仍然有价值，因为图只能表达“允许怎么走”，不能天然替代“结合实验目标判断应该怎么走”。当前保留下来的 Manager 职责主要包括：

- 维护实验需求、数据上下文和跨节点领域信息。
- 在执行前做审批和约束检查。
- 接收 review 结果，形成 revision feedback，并判断是否继续修订。
- 在预算、质量和终止条件之间做领域层决策。

变化在于：Manager 不再拥有无限制地调用任意 Agent、无限循环或绕过取消状态的权力。所有决策都必须经过图中合法边、前置条件和预算校验。因此这不是弱化智能，而是把“智能决策”和“工程控制”分层。

需要准确说明的是，原项目虽然叫 Manager Agent，但外层调度也主要由 Python 条件分支实现，并不是每一步都由 LLM 自由规划；当前版本也还不是一个可以任意动态选择所有 Agent 的自由路由器。

### 为什么引入 LangGraph

LangGraph 的主要优势不是让模型更聪明，而是让多 Agent 系统更可靠：

- **显式状态机**：合法节点、条件边、回环路径和终止条件可以直接检查。
- **有界回环**：review 失败只能在 revision budget 内回到 Candidate/Model，避免无限自我修订。
- **节点级可观测**：可以记录每个节点的输入摘要、输出、耗时、错误和产物，而不是只看到 Manager 的最终结果。
- **取消与故障隔离**：在节点边界检查取消；Operation 节点还可终止整个受控进程组。
- **恢复与重放**：状态和产物路径可持久化，失败后可以判断从哪里恢复，而不是重跑整个黑盒流程。
- **测试更容易**：可以单独测试路由、节点、review 回环和异常分支，不依赖完整真实 LLM 链路。
- **适配异步任务**：更容易与 PostgreSQL、Redis、Celery 的任务状态、租约和幂等协议对齐。

如果只是把 `YieldAgentManager.run()` 整体包装成一个 LangGraph 节点，那么内部仍然是黑盒，得不到节点级 checkpoint、条件边、取消、恢复和审计能力，所以本项目没有采用这种“形式上用了图、实质仍是单体 Manager”的做法。

### 代价与边界

- 流程自由度下降：新增分支通常要修改图、状态 Schema 和路由校验。
- 工程复杂度上升：需要同时维护 graph state、事件、checkpoint、任务状态和领域产物的一致性。
- LangGraph 节点不等于独立智能体：当前 CandidateAgent、ModelAgent 仍是 `YieldAgentManager` 的领域方法，`OperationAgent` 才是独立类；离线工作流中的 Agent 更多是逻辑角色。
- 已执行带真实 API 凭据的 Candidate/Model/Operation 链路：首次运行暴露 Operation 大结果队列回传死锁，修复后`run_6f40a7d9d715`端到端`passed`。这是一次真实验收，不是多样本效果评测，面试时不能把它外推为生产完成率。

### 当前落地形态

当前没有退回纯 Manager 模式，而是在现有混合架构上增加了结构化 `DomainManagerRouter`。Manager 产生下一位执行者的结构化决策，LangGraph 继续负责校验和执行合法路由：

```json
{
  "next_agent": "CandidateAgent",
  "reason_code": "review_requires_revision",
  "feedback": "重新限制 phi_m 并执行消融",
  "remaining_revision_budget": 1
}
```

每条决策实际保存 `next_node`、`next_agent`、`reason_code`、`feedback`、`remaining_revision_budget`、`allowed_next_nodes` 和 `validated`。图只接受合法集合内的下一节点，并在决策时检查失败、取消、review 结果和修订预算。这样既保留 Manager 的领域判断，又不会牺牲系统的确定性边界。前端事件层已经区分 `manager_decision` 与 `agent_progress`，可以分别展示 StateGraph 路由、Manager 决策和专业 Agent 行为。

### 面试回答模板

#### 30 秒版

> 我们前后两版的控制流确实发生了变化。原来是 Manager Agent 通过一段命令式流程直接串联各 Agent；现在采用 LangGraph 加 Manager 的混合编排。LangGraph 负责状态、合法路由、回环预算、取消和恢复，Manager 仍负责需求理解、执行审批、review 反馈和终止判断。这样做不是删除 Manager，而是把领域决策与工程控制解耦，使流程更可观测、可测试和可恢复。代价是图和状态 Schema 的维护成本更高，自由度也受到约束。

#### 2 分钟版

> 原版系统以 `YieldAgentManager.run()` 为中心，Data、Search、Candidate、Model 和 Operation 的调用顺序、review 以及 revision loop 都写在 Manager 的 Python 控制流中。它的优点是实现直接、动态上下文集中，但缺点是流程偏黑盒：节点级状态、取消、恢复、回环预算和行为审计都不够明确。
>
> 迁移后，我们没有把 Manager 删除，也没有重写原有领域算法，而是采用 LangGraph 作为外层控制面。图明确规定 `prepare -> data -> requirements -> search -> candidate -> model -> pre_execution -> operation -> review` 等合法路径，并处理取消、失败和有界修订；每个节点通过适配器恢复原 Manager 的上下文，继续复用它的需求管理、审批、review 和反馈能力。OperationAgent 也保留原有职责，并增加了进程组级终止协议。
>
> 这个设计的核心是职责分离：LangGraph 回答“系统允许怎么走”，Manager 回答“基于领域结果应该怎么走”，专业 Agent 回答“具体任务怎么执行”。它提高了可观测性、测试性、恢复能力和生产安全性。当前结构化 ManagerRouter 已经输出 `next_agent`、原因、反馈和剩余预算，再由图校验后执行；Manager 决策和专业 Agent 行为也使用不同事件类型输出。真实 LLM 领域链路已有单次端到端通过验收，但还不足以形成真实模型完成率；`16.7%→83.3%`只来自不调用LLM的30条受控故障恢复A/B。

### 常见追问与着重回答

**问：为什么不保留纯 Manager Agent？** 纯 Manager 适合快速原型，但一旦接入异步任务、取消、恢复和人工审计，隐藏在 `run()` 内的控制流很难可靠管理。混合模式保留了 Manager 的领域判断，同时给它加上可验证的工程边界。

**问：LangGraph 会让 Agent 更智能吗？** 不会直接提升模型智能。它提升的是流程可靠性、可解释性和可运维性；模型质量仍取决于提示词、工具、数据、检索和评测。

**问：现在每个节点都是独立 Agent 吗？** 不是。当前既有 CandidateAgent、ModelAgent 主要是 Manager 内的方法，OperationAgent 是独立类；离线工作流的一些 Agent 名称表示职责角色。面试时应区分“图节点”“Agent 角色”和“独立运行服务”。

**问：为什么不把原 Manager 整体作为一个图节点？** 那只能获得一个粗粒度外壳，内部仍无法节点级 checkpoint、路由审计和取消，也无法针对 review 回环做独立测试。

**问：当前方案最值得强调的工程价值是什么？** 是把动态决策约束在一个可恢复、可审计的状态机中，同时复用原有领域实现，降低重写风险；不是单纯把框架名称从 Manager 换成 LangGraph。

## 阶段进度

### 阶段 0：差异审计与任务记录

- 状态：已完成。
- 产物：本文件。
- 验证：确认本地 git 干净、远端地址匹配、仓库无原计划文件、附件文档存在但不是可执行实现。

### 阶段 1：可复现运行基础

- 状态：已完成初版。
- 产物：
  - `scripts/run_yieldmind_eval.py`
  - `yieldmind/evaluation.py`
  - `yieldmind.tools.generate_demo_yield_data`
  - `yieldmind.tools.run_fixed_baseline_eval`
- 说明：离线评测调用现有 `knowledge.yield_synthetic_data` 与 `knowledge.yield_baselines`，不调用 LLM，不伪造模型指标。

### 阶段 2：后端与数据库

- 状态：已完成初版。
- 产物：
  - `yieldmind/api.py`
  - `yieldmind/database.py`
  - `yieldmind/migrations/001_init.sql`
  - `scripts/init_yieldmind_db.py`
- 数据库：本阶段初版为 SQLite；阶段 13 已升级为 PostgreSQL生产路径，SQLite保留为离线测试模式。
- 表：`yieldmind_runs`、`yieldmind_events`、`yieldmind_tool_calls`、`yieldmind_eval_runs`。

### 阶段 3：Agent 工具调用与执行隔离

- 状态：已完成初版。
- 产物：
  - `yieldmind/tools.py`
  - `yieldmind/sandbox.py`
- 工具：
  - `generate_demo_yield_data`
  - `profile_yield_data`
  - `run_fixed_baseline_eval`
  - `run_sandboxed_python`
  - `summarize_run_artifacts`
  - `run_existing_yield_pipeline`
- 真实/模拟边界：
  - 默认工具均为 `offline_deterministic`，`llm_calls=0`。
  - `run_existing_yield_pipeline` 只有 `allow_live_llm=true` 才会运行原 `run_yield.py`，否则拒绝执行并说明未模拟。

### 阶段 4：评测与展示

- 状态：FastAPI实验工作台第一轮重构已完成；Next.js未引入。
- 产物：
  - `yieldmind/static/index.html`
  - `yieldmind/static/app.css`
  - `yieldmind/static/app.js`
  - `tests/test_yieldmind_core.py`
- 评测指标：工具调用是否成功、失败样本是否被拒绝、隔离执行是否通过、是否产生真实报告。
- 注意：这些是 Agent 工具层回归结果，不是最终研究模型效果。
- 展示改造：
  - 主视图从后端按钮/原始JSON控制台改为实验运行工作台，展示全局最终推荐模型的OOF指标、实际值/预测值散点图、候选策略排名、固定基线、选择审计、StateGraph节点、产物和逐步Trace。
  - 候选与固定基线都可成为最终推荐；图表最多展示200条获胜模型OOF预测，只使用真实交叉验证预测，不生成前端模拟点。
  - 增加由工作流真实节点事件驱动的多Agent活动区；DataAgent、SearchAgent、ModelAgent、CandidateAgent、ReviewAgent、ReportAgent与Agent Manager分别显示当前节点、执行状态和时间，历史运行则由已持久化stage重建，不使用前端定时假进度。
  - 默认载入并明确标记200行增强演示CSV；增加10 MB上限的CSV上传入口，文件只写入受控目录，并在运行前使用现有屈服应力Schema校验有效行和目标列。
  - Redis任务取消/恢复和Tool Registry保留在次级区域；Redis不可用时前端明确禁用队列操作，同步离线工作流仍可运行。
- 真实验证：
  - FastAPI根页面、CSS和JS均经真实HTTP返回`200`；浏览器完成健康、工具、运行列表及最近运行结果加载。
  - 60样本/3折真实HTTP离线工作流返回`passed`，9个StateGraph节点全通过；展示3个候选、4个固定基线、OOF预测点和9类产物，`real_llm_calls=0`。
  - 默认200行增强CSV经真实FastAPI流式运行返回`passed`（`run_316bb843f863`）；收到18条节点开始/结束事件、9个阶段和200条获胜模型OOF预测点，`real_llm_calls=0`。候选`strategy_hgb_packing_sp`的OOF RMSE/R2为`3.7748/-0.0031`，固定`baseline_ridge_linear`为`0.3193/0.9928`；页面、JSON报告和Markdown报告均最终推荐Ridge基线，不再只做降级警告。
  - `/api/data/default`对默认文件返回200行、72列和`yield_stress`目标列；同一文件经上传接口复验成功，非CSV扩展名在接口测试中返回400。
  - 局部验收发现Chroma过滤后的向量数少于请求数会返回500；现按过滤后的真实向量数量限制`n_results`，同一Top-5知识查询复验返回`200`。
  - 完整自动化回归：`98 passed`；现有warning来自Chroma/Pydantic弃用提示、joblib核心数探测和sklearn收敛提示，不计为测试通过率提升。

### 阶段 5：LangGraph 与 Function Calling 适配

- 状态：已完成可运行初版。
- 产物：
  - `yieldmind/workflow.py`
  - `yieldmind/function_calling.py`
  - `scripts/run_yieldmind_workflow.py`
- LangGraph：
  - 已安装并验证 `langgraph<0.3`，与项目原 `langchain==0.2.1` 兼容。
  - `scripts/run_yieldmind_workflow.py` 实测返回 `workflow_backend="langgraph_stategraph"`、`langgraph_available=true`。
  - 如果其他环境未安装 LangGraph，代码会明确退回 `local_workflow_fallback`，不会假称 StateGraph 已执行。
- Function Calling：
  - `ToolRegistry` schema 可转换为 OpenAI-compatible Chat Completions `tools`。
  - 默认评测使用 `offline_rule_planner`，不调用 LLM；`allow_live_llm=true` 才走真实 Function Calling。
  - API 新增 `/api/agent/plan`、`/api/agent/execute-plan`、`/api/workflows/offline`。

### 阶段 6：领域 RAG、Memory 与技能证据

- 状态：已完成可运行初版。
- 产物：
  - `yieldmind/knowledge_base.py`
  - `yieldmind/memory.py`
  - `yieldmind/migrations/002_knowledge_memory.sql`
  - `knowledge_sources/yieldmind_domain_seed.md`
  - `docs/skills_evidence.md`
  - `docs/ai_coding_cases.md`
  - `docs/progress.md`
- RAG：
  - 安装并验证 `chromadb<0.6`，`pip check` 无 broken requirements。
  - 使用 LangChain `RecursiveCharacterTextSplitter` 做文档切分。
  - 支持 Markdown/TXT 入库、稳定 `document_id`/`chunk_id`、幂等重复入库、Chroma 检索和引用元数据返回。
  - 本阶段当时的 embedding 为 deterministic `local_hashing_v1`；后续profile隔离和三路检索基线见阶段16，Qwen/Qwen3仍未下载、未压测、未验证。
- Memory：
  - 支持创建会话、提交消息、幂等 key、约束继承、用户纠正、创建 parent run、记忆写入/删除和上下文 token 估算。
  - 当前为规则化上下文管理，不包含真实 LLM 多轮回答生成。
- API 新增：
  - `/api/knowledge/ingest`
  - `/api/knowledge/search`
  - `/api/knowledge/documents`
  - `/api/sessions`
  - `/api/sessions/{session_id}/messages`
  - `/api/sessions/{session_id}/context`
  - `/api/memories`

### 阶段 7：候选评估、产物验收与报告节点迁移

- 状态：已完成可运行初版。
- 产物：
  - `yieldmind.tools.run_candidate_benchmark`
  - `yieldmind.tools.verify_artifacts`
  - `yieldmind.tools.build_run_report`
  - `yieldmind.workflow` 新增 `candidate_benchmark -> verify_artifacts -> report` 节点。
- 说明：
  - `run_candidate_benchmark` 复用原有 `knowledge.yield_candidate_benchmark.run_candidate_benchmark` 和 `apply_benchmark_selection`，不是模拟 benchmark。
  - 当前默认候选报告是离线 seed，用于无 LLM 环境下验证 CandidateBenchmark 节点；真实 CandidateAgent 生成报告仍可通过 `candidate_report_path` 输入。
  - `verify_artifacts` 做工作流产物存在性、非空和 JSON 可读性校验；它不是完整 `operation_agent/yield_guardrails.py` 全量科研验收替代。
  - `build_run_report` 从真实 artifact path 生成 JSON/Markdown 报告。

### 阶段 8：Safety、Evidence Validation 与 Token Budget

- 状态：已完成可运行初版。
- 产物：
  - `yieldmind/safety.py`
  - `yieldmind.tools.redact_sensitive_payload`
  - `yieldmind.tools.plan_token_budget`
  - `yieldmind.tools.validate_evidence_refs`
  - API：`/api/safety/redact`、`/api/budget/plan`、`/api/evidence/validate`
- 说明：
  - ToolRegistry 在记录工具调用参数前会统一脱敏，避免 redaction 工具自身把 secret 写入日志。
  - `redact_sensitive_payload` 支持敏感字段名、Bearer/API key、邮箱、手机号、数据库 URL 的规则脱敏。
  - `plan_token_budget` 按 required/priority 和 token budget 选择或裁剪上下文段；token 估算优先 `tiktoken`，不可用时退回 `chars/4`。
  - `validate_evidence_refs` 从 SQLite `yieldmind_document_chunks` 校验 `chunk_id`，并可检查 `text_hash`、`index_version`、`document_id` 和 `document_version`。
  - 当前属于 deterministic guardrail，不是完整 DLP，也不替代“引用语义是否支撑结论”的人工标注评测。

### 阶段 9：数据集驱动离线评测集

- 状态：已完成可运行初版。
- 产物：
  - `evals/yieldmind_offline_cases.json`
  - `yieldmind.evaluation.load_eval_dataset`
  - `yieldmind.function_calling.local_rule_plan` 新增 safety/budget/evidence 路由规则
- 覆盖：
  - 6 条工具执行 case。
  - 12 条 planner 工具决策 case。
  - 1 条 workflow case。
  - 7 条知识库 case：1 条入库/检索 smoke + 6 条 RAG 查询样本。
  - 9 条 safety case：1 条 evidence/budget aggregate + 4 条 redaction + 4 条 token budget。
  - 1 条 session/memory case。
- 边界：
  - 这些是 deterministic offline regression，不是 LLM 工具选择准确率。
  - planner case 衡量本地规则 planner 的一致性；真实 Function Calling 仍需显式 `allow_live_llm=true` 后单独记录。

### 阶段 10：真实 Function Calling 冒烟机制

- 状态：已完成保护式脚本；真实模型调用尚未执行。
- 产物：
  - `scripts/run_yieldmind_function_calling_smoke.py`
  - pytest：`test_function_calling_smoke_script_skips_live_llm_by_default`
- 说明：
  - 默认运行只生成 `skipped_live_llm_not_enabled` 报告，并附带本地 rule planner 预览，`real_llm_calls=0`。
  - 只有显式传入 `--allow-live-llm` 才会通过 `yieldmind.function_calling.plan_tools` 发起一次真实 Chat Completions tools 调用。
  - 报告会写入 `agent_workspace/yieldmind/function_calling/`，并通过 redaction 逻辑避免 secret 原文落盘。
  - 本轮未把 skipped 报告标记为真实 LLM 成果；真实 Function Calling 准确性仍待单独执行并记录。

### 阶段 11：Docker/无网络强隔离

- 状态：代码、保护策略、确定性测试和真实容器运行时隔离验收已完成；官方基础镜像构建仍受网络条件阻塞。
- 产物：
  - `yieldmind.sandbox.DockerSandbox`
  - `run_docker_sandboxed_python` Tool
  - `docker/yieldmind-sandbox.Dockerfile`
  - `docker/requirements-sandbox.txt`
  - `scripts/check_yieldmind_docker_sandbox.py`
- 约束：显式 `allow_docker=true`、镜像白名单、`--pull=never`、`--network none`、只读根文件系统和源码挂载、独立可写输出挂载、drop all capabilities、no-new-privileges、CPU/内存/PID/超时限制，不传宿主环境变量。
- 验证事实：命令构造、安全路径和默认拒绝已由 pytest 验证；daemon启动后真实容器运行通过并输出 `network-check=blocked`。官方 Python 3.11基础镜像构建因 Docker Hub token 请求超时未完成，运行测试使用本机缓存 Python镜像临时标签。

### 阶段 12：LangGraph 条件路由与有限恢复

- 状态：升级层条件路由及 PostgreSQL持久检查点已完成；原 `run_yield.py` 全流程尚未迁移。
- 改动：
  - 使用 `add_conditional_edges` 根据结构化失败状态进入局部修复、重规划或终止分支。
  - `max_local_repairs`、`max_replans` 限制恢复次数，避免无限循环。
  - 每个节点记录 `stage_execution_id` 和输入哈希；状态记录 failure/route history。
  - `003_stage_executions.sql` 将节点执行轨迹持久化，并提供 `/api/runs/{run_id}/stages` 查询。
  - SQLite离线模式使用 `MemorySaver`；PostgreSQL生产路径使用 `PostgresSaver` 和独立 `thread_id`，初始化脚本负责 checkpointer migration。
  - workflow 为线程、Tool、参数和修复轮次生成稳定幂等键；Tool Registry 执行前持久声明，完成后回写结构化结果。
  - 用户明确提供的数据路径不存在时直接失败，不再静默切换 demo 数据。
  - 修复 baseline 工具最佳结果取值错误，保证工具摘要与原 baseline report 一致。
- 边界：已证明 checkpoint 持久写入和完成 Tool 调用重放复用；崩溃后残留的 `running` 调用会拒绝盲目重跑并等待处置，因此是 at-most-once 防护，不是 exactly-once，中断恢复仍待验收。
- 验证：故障注入测试证明 CandidateBenchmark 首次失败后进入 local repair 并在第二次成功；另有测试证明错误用户路径只走 `start -> prepare_data(failed) -> finish`，不调用 demo 工具。

### 阶段 13：PostgreSQL、Redis 与 Celery

- 状态：本地容器真实集成已完成；生产部署、密钥管理和高可用未验证。
- PostgreSQL：
  - 使用 SQLAlchemy 2.0 metadata 定义 schema，`psycopg` 作为驱动；任务租约阶段revision为`20260918_0004`，Memory升级后当前head为`20260918_0005`。
  - `YieldMindStore` 生产模式读取 `YIELDMIND_DATABASE_URL`，现有参数化查询通过兼容层运行于 PostgreSQL；SQLite只保留离线测试模式。
  - 已真实覆盖 run/event/tool/stage/session/turn/memory/document/chunk/task 读写路径。
  - LangGraph `PostgresSaver` 已接入生产工作流；`scripts/init_yieldmind_db.py` 创建其独立迁移表。
- Redis/Celery：
  - Redis作为 Celery broker 和 result backend；PostgreSQL `yieldmind_tasks` 是对外任务状态权威。
  - 任务先持久化并按 idempotency key 去重，再发布到 Redis；发布失败记录 `dispatch_failed`。
  - 新增 `/api/tasks/workflows/offline`、`/api/tasks/{task_id}` 和 `/health/dependencies`。
- 实际验证：
  - PostgreSQL/Redis 集成脚本十项检查全部通过，包含 Alembic版本、工作流、8条节点轨迹、10条 LangGraph checkpoint、Tool重放幂等、Memory、知识元数据和任务幂等。
  - 既有数据库 `0001 -> 0002` 和临时空数据库从零迁移均通过；临时库验证后已删除。
  - 临时 Celery worker真实消费一条 Redis任务，PostgreSQL终态为 `completed`，Celery状态为 `SUCCESS`，关联 run 可查询；验证后 worker已停止。
- 边界：Compose中的口令仅为本地开发默认值；没有完成生产secret manager、连接池压测、Redis哨兵/集群或 PostgreSQL高可用。

### 阶段 14：任务状态可靠性与恢复闭环

- 状态：14.1任务状态机、14.2排队取消、14.3运行中节点边界协作取消和14.4租约/人工恢复已完成；OperationAgent节点内子进程终止已在阶段15实施，自动恢复仍未实施。
- 14.1改动：
  - Store集中定义合法状态转换，调用方即使传入错误前态也不能将 `completed/failed/dispatch_failed` 改回活动状态。
  - 发布端在发送 Celery 消息前原子执行 `created -> queued`；worker只能执行 `queued -> running`，消除快速worker完成后被发布端覆盖回 `queued` 的竞态。
  - 重复请求仅在 `queued/running/completed/failed` 时报告已发布；`dispatch_failed` 不再误报为发布成功。
  - 新增 `scripts/check_yieldmind_task_transitions.py`，同一检查真实通过 SQLite 和 PostgreSQL。
- 发现与修复记录：首次 PostgreSQL反向测试暴露调用方可显式传入 `completed -> running` 的规则缺口，测试结果为最终状态错误回到 `running`；随后将状态图收口到 Store 并复验，非法回退被 `ValueError` 拒绝且最终状态保持 `completed`。
- 14.1验证：pytest `22 passed`；真实 Redis/Celery任务为 `completed/SUCCESS`；PostgreSQL状态 smoke 三项检查全 true；验证后worker和容器均已停止。
- 14.2排队取消：
  - Alembic revision `20260918_0003` 新增取消请求时间、完成时间、原因和请求方标签；现有库升级及临时空库从零迁移均通过。
  - 14.2落地时 `POST /api/tasks/{task_id}/cancel` 仅接受 `queued`；14.3已扩展为接受 `running` 并返回 `cancel_requested`，终态仍返回409，重复取消保持幂等。
  - PostgreSQL先原子落 `cancelled`，Celery revoke只作辅助；即使Redis控制命令失败，worker也无法执行 `cancelled -> running`。
  - 真实场景按“无worker发布并取消→启动worker”验证：数据库保持 `cancelled`，Celery=`REVOKED`，未创建run、未写结果，`real_llm_calls=0`。
  - FastAPI静态控制台增加依赖检查、任务提交、状态轮询、取消原因和取消按钮；真实HTTP联调返回完整取消审计字段。当前仍不是Next.js，且本机未安装Playwright，因此未声称浏览器截图验收。
- 14.3运行中协作取消：
  - 数据库状态机增加 `running -> cancel_requested -> cancelled`；取消请求、完成时间、原因和发起方保持可审计，重复请求幂等。
  - worker在创建run后立即将run_id回写task；LangGraph在每个节点边界查询取消状态，并通过显式 `cancel` 节点持久化轨迹，最终task和run共同落 `cancelled`。
  - 真实 PostgreSQL/Redis/Celery 场景连续两次通过：均观察到DB `running` 后发起取消，task/run一致为 `cancelled`，轨迹为 `start -> cancel -> finish`，Celery=`REVOKED`，`real_llm_calls=0`。
  - 真实关闭Redis后，Celery查询降级为 `UNKNOWN` 且返回错误，PostgreSQL仍原子保存排队取消和审计；恢复Redis后十项集成检查全部再次通过。
  - 修复了两项验证中发现的问题：LangGraph条件路由内的state修改不会持久化，改为显式cancel节点；静态控制台取消函数遗漏解析HTTP响应，已补回 `responseJson`。
- 14.3当时边界：节点内正在运行的长耗时Tool或OperationAgent子进程不会被即时强杀，只会在其返回后的下一个节点边界停止；其中OperationAgent缺口已由阶段15闭合，普通同步Tool仍保留该边界。Redis下线验证不等于Redis HA，PostgreSQL HA也未实施。
- 14.4任务租约与恢复：
  - Alembic revision `20260918_0004` 增加worker身份、heartbeat、lease到期、中断审计和父任务字段；现有库升级与临时空库全迁移均通过。
  - Celery worker领取任务时写入租约，并由独立心跳线程续租；终态清除有效租约但保留worker审计。
  - `GET /api/task-recovery/stale`只列出已过期活动任务；恢复请求必须显式确认旧worker停止，旧task/run原子终结为`interrupted`，新task/new run保存父链且使用新ID。
  - 真实演练使用FIFO让节点读取确定性阻塞，观察到心跳续租后以PID冷停worker；租约自然过期后恢复，新task=`completed`、新run=`passed`、Celery=`SUCCESS`，11项检查全true。
  - 恢复不是旧run原地回退，也不是训练优化器状态续跑；人工确认是防止网络分区或长节点误判后并发副作用的必要安全闸门。

### 阶段 15：原多Agent主流程StateGraph迁移与OperationAgent终止协议

- 状态：代码与离线/模拟路由验证已完成；真实模型端到端运行未执行，因此不声称真实LLM成功率或模型效果。
- 逻辑审计：
  - CandidateAgent和ModelAgent不是独立服务，而是`run_yield.py`中`YieldAgentManager`的方法；OperationAgent是独立类。迁移继续复用这些领域实现和既有JSON产物，没有复制或替换领域算法。
  - 旧OperationAgent只有固定超时，且生成脚本通过`shell=True`启动；杀外层`multiprocessing.Process`不能充分证明后代进程已回收。
  - 仅把`YieldAgentManager.run()`包成单一LangGraph节点不具备条件边、节点轨迹和节点级取消价值，因此未采用。
- 主流程迁移：
  - 新增`yieldmind/domain_workflow.py`，真实StateGraph节点为`start -> prepare -> data -> requirements -> search -> candidate -> model -> pre_execution -> operation -> review -> finish`。
  - review不通过且仍有预算时，经`revision_feedback`回到CandidateAgent和ModelAgent；Operation取消直接进入显式`cancel`节点，不再执行review。
  - Graph state只保存Pydantic请求、普通字典、计数和产物路径；LLM客户端、DataFrame、进程和manager实例不进入checkpoint。每个真实节点从既有JSON产物恢复manager上下文。
  - 真实适配器默认拒绝运行，必须显式`allow_live_llm=true`；真实调用尚未统一计量时标记`model_call_mode=live_unmetered`，不虚构调用次数为0。
  - 新增同步API `POST /api/workflows/domain`、异步API `POST /api/tasks/workflows/domain`和Celery任务类型`domain_workflow`；复用既有PostgreSQL状态、Redis分发、租约、取消、幂等和人工恢复。
- 终止协议：
  - 新增`yieldmind/process_control.py`。受控目标先创建POSIX session；父进程轮询取消与超时，先向整个进程组发送SIGTERM，宽限后再SIGKILL，并返回`cancelled/timed_out/controller_error`及PID、耗时、终止信号元数据。
  - domain StateGraph将整个Operation节点置于受控进程组内，因此`free_search/plugin/llm_freeform`产生的未主动脱离后代都在同一回收边界内。旧`llm_freeform`入口也接入相同控制器。
  - `operation_agent/execution.py`改为argv与显式env执行，移除`shell=True`，且不为生成脚本另建session，避免其逃离外层Operation进程组。
- 验证：
  - 真实POSIX进程测试覆盖正常返回、超时和外部取消；测试目标会再启动一个30秒子进程，超时/取消后均确认该后代PID消失。
  - StateGraph模拟适配器覆盖正常流程、一次review失败后的有界回环、Operation取消后跳过review、未授权真实模型时在start节点阻断，以及domain任务入队幂等。
  - 新入口默认数据不存在时会沿用旧入口行为自动启用合成数据；显式`--synthetic-data/--no-synthetic-data`仍覆盖自动判断。
  - 模拟适配器结果明确记录`model_call_mode=simulated_test_adapter`、`real_llm_calls=0`，不作为真实模型效果或真实Function Calling结果。
  - 真实PostgreSQL/Redis/Celery提交`domain_workflow`通过九项检查：发布与幂等、任务类型、worker消费、task/run关联、StateGraph轨迹和未授权模型闸门均符合预期。Celery执行为`SUCCESS`，业务task/run有意为`failed`，错误明确要求`allow_live_llm=true`，`real_llm_calls=0`。
- 已知边界：
  - 进程组后代回收依赖POSIX；非POSIX平台只能退回父进程terminate/kill，尚未实现Windows Job Object。
  - 任意主动调用`setsid()`脱离Operation进程组的生成代码不在该回收保证内；当前生成脚本执行器不会这样做。
  - 普通同步Tool仍只支持节点边界取消；训练优化器内部状态不会在取消后续跑。
  - 尚未执行带真实API凭据的完整Candidate/Model/Operation链路，不能宣称真实模型端到端成功。

### 阶段 16：检索基线、Embedding隔离与Function Calling闭环

- 状态：离线实现与验证已完成；Qwen3真实推理、Reranker和真实模型Function Calling仍未执行。
- 前提审计：
  - 原知识库并非空白，已有Chroma、LangChain分块、稳定chunk元数据和6条小样例；但唯一种子文档只有约135词，无法支撑可信的30查询选型。
  - `utils/embeddings.py`中的FAISS/BM25/BGE属于旧组件，不等于已接入当前Chroma主流程。
  - 仓库依赖固定`transformers==4.40.1`（本机实际安装4.40.2），而Qwen3 Embedding需先在兼容隔离环境验证；本阶段没有下载模型，也没有把hashing结果描述为语义embedding效果。
- 检索实现：
  - `EmbeddingProfile`锁定provider、model ID、revision、dimension、normalize、metric、query/document instruction和index version，并以profile指纹隔离Chroma collection，阻止不同维度向量混写。
  - 新增可选`SentenceTransformerEmbeddingFunction`；默认禁止下载，维度与profile不一致时拒绝建索引。
  - `search_knowledge`支持`vector / bm25 / hybrid`；词法路径使用BM25+，hybrid使用RRF。返回检索通道、profile、索引版本与实测延迟。
  - 幂等入库同时检查数据库元数据和当前Chroma collection；数据库显示available但当前collection无向量时会重建，避免profile升级后的空索引。
  - 现有文档表已经持久化`index_version`，本轮通过新collection和index version完成隔离，因此没有无依据地新增数据库迁移。
- 评测集与结果：
  - 新增6份仓库行为/领域约束材料和`evals/yieldmind_retrieval_cases_v1.json`，共30个唯一case、6个标注来源，包含中文跨语言与低词面重合查询。
  - 最终复验文件：`agent_workspace/yieldmind/retrieval_evals/retrieval_eval_20260918_094010.json`。
  - hashing vector：Recall@5=`0.8333`、MRR=`0.8083`、citation accuracy@1=`0.8000`、平均延迟约`12.08ms`。
  - BM25+：Recall@5=`0.8000`、MRR=`0.8000`、citation accuracy@1=`0.8000`、平均延迟约`2.62ms`。
  - RRF hybrid：Recall@5=`0.8333`、MRR=`0.8083`、citation accuracy@1=`0.8000`、平均延迟约`15.54ms`；单次小样本延迟不用于稳定吞吐结论。
  - 这些标签衡量仓库知识检索，不是外部文献benchmark；结果只说明跨语言是当前短板，不证明Qwen3一定优于其他候选。
- Function Calling：
  - 模型返回的工具名和JSON参数在执行前经过Tool Registry与同一Pydantic Schema校验；畸形JSON、未知工具、缺少必填参数和超出调用预算均显式失败，不再静默替换为空参数。
  - 实时执行保留assistant tool-call消息，将脱敏Tool结果按`tool_call_id`回传模型，再以`tool_choice=none`完成第二轮证据约束回答。
  - Fake Client协议测试明确记录`mode=simulated_test_adapter`、`simulated_model_calls=2`、`real_llm_calls=0`；没有冒充真实Function Calling结果。
- 已知边界：
  - 30条case仍是项目内自建集，规模较小且标签由单人构建；生产选型前需要独立复核标签并增加真实用户查询。
  - Qwen3、BGE-M3或其他多语embedding尚未在相同case、相同硬件和相同分块下实测；Reranker应在语义embedding基线后再判断收益与延迟成本。
  - 真实Function Calling端到端仍需显式授权并产生外部模型费用；本阶段只完成可测试协议和模拟适配器回归。
- 本机embedding准入审计：
  - `scripts/check_yieldmind_embedding_capability.py`只检查本地依赖、缓存、硬件与磁盘，不联网、不下载、不执行模型；报告为`agent_workspace/yieldmind/embedding_capability/embedding_capability_20260918_094133.json`。
  - 实际环境为x86_64、32GiB内存、约44.9GB可用磁盘、无CUDA/MPS；安装`transformers=4.40.2`、`sentence-transformers=2.7.0`，Qwen/BGE缓存均不存在。
  - Qwen3 0.6B当前被`model_not_cached`、`transformers<4.51`和CPU-only实测成本阻断；BGE-M3还缺少FlagEmbedding。因而没有修改生产默认embedding。
- PostgreSQL/Redis复验：
  - 扩展真实集成smoke验证PostgreSQL chunk join、BM25+与Chroma向量的hybrid结果，11项检查全部为true，`knowledge_hybrid_postgres=true`。
  - 报告为`agent_workspace/yieldmind/integrations/postgres_redis_smoke_20260918_094729.json`；`real_llm_calls=0`。验证后PostgreSQL和Redis容器均已停止，状态为`exited (0)`。

### 阶段 17：Qwen3独立推理环境与真实Embedding评测

- 状态：已完成。独立环境、固定模型、回环服务、真实CPU推理评测和资源观测均已闭环；评测结束后服务已停止。
- 逻辑审计：
  - Qwen3要求`transformers>=4.51`及`tokenizers>=0.21`，而主环境Chroma 0.5.23要求`tokenizers<=0.20.3`；将两者强装在同一环境会产生不可解依赖冲突。
  - 采用独立推理进程而非升级主环境：Qwen环境只安装Torch、Transformers和Sentence-Transformers，主环境继续负责Chroma、BM25+、RRF、PostgreSQL及评测。
- 已完成：
  - 新增`requirements-qwen3-embedding.txt`和`scripts/setup_yieldmind_qwen3_env.sh`；`.venv-qwen3`约1.1GB，`pip check`通过，版本为Torch 2.2.1、Transformers 4.51.3、Sentence-Transformers 4.1.0。
  - 主环境保持Transformers 4.40.2、Sentence-Transformers 2.7.0，未被修改。
  - 模型固定为`Qwen/Qwen3-Embedding-0.6B@97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3`，维度1024，使用英文领域检索instruction，文档侧不加instruction。
  - 新增仅允许回环地址的embedding服务和带身份握手的HTTP客户端；模型ID、commit或维度不一致时拒绝建索引，支持可选Bearer token、批量限制和向量维度校验。
  - HTTP客户端显式绕过系统代理并拒绝非回环endpoint；真实测试发现全局`HTTP_PROXY`会把本机请求转发到代理并返回502，该失败未计入模型评测。
  - Hugging Face官方域名在首次下载时连接超时；模型通过镜像按同一完整commit下载，最终评测从本地缓存加载且`model_download_allowed=false`，模型缓存和独立环境均不提交Git。
- 真实验证：
  - 完整本地报告：`agent_workspace/yieldmind/retrieval_evals/retrieval_eval_20260918_103507.json`；Git跟踪的精简结果：`evals/results/yieldmind_qwen3_retrieval_20260918.json`。两者记录`embedding_execution_mode=isolated_http_sentence_transformer_real_inference`、`real_llm_calls=0`，精简结果同时保存完整报告SHA-256。
  - Qwen3 vector：Recall@5=`0.9667`、MRR=`0.7789`、Top-1引用准确率=`0.6667`、平均查询延迟=`378.6ms`。
  - 同次BM25+：`0.8000 / 0.8000 / 0.8000 / 2.4ms`；Qwen3+BM25+ RRF：`1.0000 / 0.8639 / 0.7667 / 431.3ms`。
  - 7篇文档切分为20个chunk，入库耗时`15.60s`，完整评测耗时`40.02s`；服务从本地缓存加载耗时`3.95s`，峰值RSS约`3.90GB`，共67次encode、80条文本。
  - Bearer鉴权无令牌返回HTTP 401；评测完成后`8091`连接失败，确认未保留常驻服务。
- 判断：
  - Qwen3显著提高Recall@5，和BM25+融合后本集合Recall@5达到1.0；但纯向量Top-1准确率低于BM25+，且CPU平均延迟约高两个数量级。因此当前不把纯Qwen向量设为生产默认，保留离线hashing回归及BM25+，将Qwen3+RRF作为需扩大数据后复核的质量候选。
  - 30条case是单人标注的项目内集合，不能外推为生产RAG效果；BGE-M3和reranker尚未同口径实测，也不应凭模型榜单宣称优于当前方案。

### 阶段 18：Qwen3检索接入StateGraph主流程

- 状态：已完成显式可选接入和真实PostgreSQL集成验收；默认离线工作流仍不依赖Qwen服务。
- 实现：
  - `ToolRegistry`支持注入指定`KnowledgeBase`，`search_knowledge`和入库工具不再私自固定为hashing profile。
  - `WorkflowRequest`新增显式知识检索开关、查询、Top-K、索引版本和检索模式；StateGraph在数据分析后增加`retrieve_evidence`节点。
  - 检索节点调用注册工具后，以`chunk_id/text_hash/index_version/document_id/document_version`执行证据校验；无命中、服务异常或引用校验失败时进入明确失败终态，不静默回退hashing。
  - 报告工具新增结构化`evidence_refs`，Markdown/JSON报告均可定位chunk来源；Chroma目录不再被错误登记为可下载文件产物。
  - 新增`scripts/run_yieldmind_qwen_workflow_smoke.py`，串联Qwen入库、混合检索、引用校验、确定性建模、产物验收与报告。
- 真实验证：
  - 完整本地报告：`agent_workspace/yieldmind/qwen_workflow_smoke/20260918_110331/qwen_workflow_smoke.json`；Git精简结果：`evals/results/yieldmind_qwen3_workflow_20260918.json`。
  - 11项检查全部为true；工作流为`langgraph_stategraph`，checkpointer为`langgraph_postgres`并写入11条checkpoint，Redis健康检查通过。
  - 7篇文档、20个chunk真实执行8次Qwen encode并编码21条文本；混合检索返回5条引用且元数据校验通过，报告包含结构化证据。
  - 本阶段`real_llm_calls=0`、`simulated_model_calls=0`：证明真实embedding和工具/图编排接线，不代表真实LLM完成了工具选择。
  - 初次定向测试发现检索工具把Chroma目录误登记为文件产物，导致artifact guardrail拒绝流程；修正Tool契约后定向测试和真实集成均通过。

### 阶段 19：Qwen3服务端配置与FastAPI联调

- 状态：已完成。FastAPI知识接口和Tool Registry可由服务端环境显式切换Qwen3，默认仍为离线hashing。
- 实现：
  - 新增`yieldmind/knowledge_runtime.py`，用Pydantic校验profile、回环endpoint、超时、Chroma路径和collection；token只按环境变量名读取，不进入公开配置。
  - Tool Registry支持延迟`knowledge_base_factory`，FastAPI启动时不连接Qwen；只有readiness或实际知识调用才检查服务，避免可选依赖阻止API进程启动。
  - `/api/knowledge/ingest`、`/api/knowledge/search`及`search_knowledge` Tool统一使用服务端配置的profile；`/api/knowledge/config`返回非密钥配置。
  - `/health/dependencies`新增knowledge依赖：Qwen被配置但endpoint缺失、身份不匹配或不可用时返回非就绪；默认hashing不产生网络调用。
  - `.env.example`增加YieldMind专用embedding配置；新增`scripts/run_yieldmind_qwen_api_smoke.py`。
- 真实验证：
  - 完整本地报告：`agent_workspace/yieldmind/qwen_api_smoke/20260918_111812/qwen_api_smoke.json`；Git精简结果：`evals/results/yieldmind_qwen3_api_20260918.json`。
  - PostgreSQL、Redis、Qwen readiness及入库/搜索共9项检查全部为true；直接知识接口和Tool Registry搜索均使用profile指纹`445d787820ef`。
  - 7篇文档、20个chunk；本轮真实增加9次embedding encode、22条文本，两个搜索路径各返回5条结果。
  - 使用FastAPI TestClient执行真实路由函数和真实外部依赖，不宣称独立HTTP进程吞吐；`real_llm_calls=0`，不宣称真实LLM工具选择。
  - 安全审阅发现初版公开配置返回Chroma绝对路径；修正为仅返回`chroma_dir_configured`后重新完成9项真实联调。最终单次直接/Tool搜索延迟分别为`735.9ms/922.7ms`，不选择性保留更低的旧轮次数字。

### 阶段 20：独立FastAPI进程与Qwen3 HTTP验收

- 状态：已完成。补齐阶段19仅使用TestClient的限制，使用独立Uvicorn进程和真实回环TCP请求验证正反两条依赖链路。
- 实现：
  - 新增`scripts/check_yieldmind_qwen_api_http.py`，显式绕过系统代理并拒绝非回环API/embedding地址。
  - 正常模式调用liveness、readiness、knowledge配置、文档入库、直接检索和Tool Registry检索；通过Qwen服务前后计数核对真实embedding调用。
  - `--expect-unready`模式验证Qwen endpoint不可达时liveness仍可用、readiness返回HTTP 503，并独立核对PostgreSQL/Redis仍健康。
- 真实验证：
  - Git精简结果：`evals/results/yieldmind_qwen3_api_http_20260918.json`；完整本地正反报告及SHA-256记录在该文件中。
  - 故障场景7/7检查通过：`/health=200`、`/health/dependencies=503`、PostgreSQL/Redis健康、knowledge不健康，且embedding/LLM调用均为0。
  - 正常场景13/13检查通过：6个HTTP endpoint均返回200，7篇文档/20个chunk入库，两条检索路径各返回5条，profile指纹均为`445d787820ef`。
  - 本轮真实增加9次Qwen encode、22条文本；`real_llm_calls=0`、`simulated_model_calls=0`。
  - 单次直接/Tool检索观测为`453.5ms/566.6ms`，仅记录本次功能验收，不作为吞吐、并发、反向代理或生产网络性能结论。
  - 验收后8072、8091、5432、6379端口均已释放，PostgreSQL/Redis容器为`exited (0)`。

### 阶段 21：检索Bad Case可解释性与BM25Plus候选修正

- 状态：实现与真实Qwen复验、双人独立标注、24条分歧裁决和最终Gold Label均已完成；固定Gold上的Qwen embedding余弦重排消融已完成且结果退化，cross-encoder对照尚未完成，默认检索未变。
- 问题核查：
  - 原评测只保存命中名次和第一来源，无法区分未召回、正确文档错误chunk排前、目标词跨chunk等原因，不能据此合理决定增加reranker。
  - 候选诊断发现`BM25Plus`的delta使零query-token重叠chunk也获得正分；原`score > 0`过滤会让这些无关候选进入RRF。这是检索逻辑问题，不是模型能力问题。
- 实现：
  - `evaluate_retrieval`新增每个case的候选chunk、仓库相对来源、来源/词项匹配、`source_match_rank`、`terms_match_rank`和结构化`failure_reason`；不保存候选全文和本机绝对路径。
  - BM25Plus候选增加规范化query token至少一项重叠约束；保留BM25Plus算法、原tokenizer和原指标口径。
  - 新增回归测试，验证零重叠query不再由delta产生伪候选，并覆盖两类失败原因。
- 真实验证：
  - 修复后30条Qwen真实评测：vector=`0.9667/0.7789/0.6667/538.2ms`，BM25+=`0.8000/0.8000/0.8000/3.0ms`，hybrid=`1.0000/0.8861/0.8000/444.5ms`，依次为Recall@5/MRR/Top-1/单次平均延迟。
  - 对比修复前，hybrid Recall@5不变，MRR增加`0.0222`、Top-1增加`0.0333`；只有`safety_redaction`从第3升至第1，未选择性删除其他Bad Case。
  - 旧自动标签下剩余6个hybrid Top-1 mismatch；双人裁决后的最终人工标签确认5个Top-1 Bad Case，其中4个为正确片段已在Top-5的排序问题，1个为Top-5内无直接相关片段。
  - 完整本地报告：`agent_workspace/yieldmind/retrieval_evals/retrieval_eval_20260918_114855.json`；Git精简结果：`evals/results/yieldmind_qwen3_retrieval_diagnostics_20260918.json`。
  - 决策：不启用Qwen embedding余弦重排；它在最终人工Gold上降低MRR、Top-1和nDCG。正式cross-encoder仍需独立实测，不能用该消融替代。

### 阶段 22：分块版本一致性与盲审标注包

- 状态：已完成分块/RRF稳定性修正、盲审工具、两份各150行的独立人工标注、一致性分析、24条分歧裁决及最终Gold Label生成。
- 执行前核查：
  - 真实检索使用`chunk_size=700/chunk_overlap=80`，旧报告却固定记录`recursive_chars_1200_180_v1`；文本和阶段21指标未因此变化，但索引身份与复现说明不正确。
  - 修正分块版本后首次复跑发现一个RRF平分case会随chunk ID哈希换序；根因是平分时使用不具语义的chunk ID排序，而不是Qwen推理变化。
- 实现：
  - 分块版本改为由真实参数生成；重复入库只有在文档、index和split version均一致时才幂等复用，改变分块配置会重建并清理旧向量。
  - RRF平分依次按通道数、最佳通道名次、名次和、来源路径及chunk序号稳定排序，并在诊断报告中记录各通道名次。
  - 新增`scripts/prepare_yieldmind_retrieval_review.py`及Pydantic行Schema；候选和case固定随机打乱，公开CSV不含来源、排名、分数和原标签，私有映射保存在Git忽略目录。
  - 已生成`evals/review/yieldmind_retrieval_review_reviewer_1.csv`和说明`evals/review/README.md`；30个问题按case ID、20个唯一候选chunk按固定hash映射提供中文辅助译文，英文原文仍是裁决依据。
  - 脚本默认拒绝覆盖已存在CSV；只有显式指定时才能重建完全未标注的表，任一标签或备注存在时均拒绝覆盖。
  - 新增`scripts/analyze_yieldmind_retrieval_reviews.py`，校验行集合、候选文本哈希和不可编辑字段，并生成分标注者指标、一致性统计及空白裁决表；分析不改写任一原始CSV。
  - 新增`scripts/finalize_yieldmind_retrieval_review.py`，重新计算原始分歧后校验裁决表保护字段、标签和说明，输出150行最终Gold Label、最终指标及Top-1 Bad Case；不信任或覆盖原始标注。
- 真实验证：
  - 最终Qwen报告为`agent_workspace/yieldmind/retrieval_evals/retrieval_eval_20260918_121005.json`，SHA-256=`3718871a6dff7f000a953169c5ded5c955a88e2c56fadabe9a0b69d4b34ff59e`。
  - 报告明确记录`recursive_chars_700_80_v1`、7篇文档/20个chunk、67次encode/80条文本；hybrid质量仍为`1.0000/0.8861/0.8000`。
  - 两份CSV均为30个case/150行且候选文本哈希全部匹配；两人relevance精确一致126/150=`84.00%`，线性加权Cohen's kappa=`0.8146`、二次加权kappa=`0.8808`，24条分歧均为相邻等级，无`0↔2`严重分歧。
  - reviewer_1指标为Recall@5=`1.0000`、MRR=`0.9194`、Top-1=`0.8667`、nDCG@5=`0.9435`；reviewer_3分别为`1.0000/0.8972/0.8333/0.9321`。最小标签保守共识仅作为诊断，不代替逐条裁决。
  - 第二份原始CSV误沿用`reviewer_id=reviewer_1`；报告保留原值并按输入角色统计为`reviewer_3`，未静默改写。完整结果见`evals/results/yieldmind_retrieval_dual_review_20260919.json`，24条裁决表见`evals/review/yieldmind_retrieval_review_adjudication.csv`。
  - 24条分歧已全部裁决，其中最终标签`0`和`1`各12条；最终150条标签分布为`0:79/1:38/2:33`。Gold指标为Recall@5=`0.9667`、MRR=`0.8861`、Top-1=`0.8333`、nDCG@5=`0.9409`。
  - 最终5个Top-1 Bad Case中，`agent_multi_role`、`data_leakage_denylist`、`eval_oof`和`safety_argv`属于排序问题；`safety_redaction`在Top-5内无直接相关候选，不能仅依赖reranker修复。最终报告见`evals/results/yieldmind_retrieval_final_gold_20260919.json`。

### 阶段 23：有界多步 Agent Loop、重复调用检测与逐步 Trace

- 状态：代码与模拟协议回归已完成；真实LLM多步调用尚未执行。
- 执行前核查：
  - 旧Function Calling路径固定为“一次选工具、执行、一次禁用工具的回答”，不能根据工具结果继续选择下一个工具。
  - 旧`max_tool_calls`只约束首次模型返回数量；Tool Registry虽支持幂等key，Function Calling路径未传入，因此不能宣称已防止重复副作用。
- 实现：
  - `ToolPlanRequest`新增`max_steps`和`max_repeated_tool_calls`；`max_tool_calls`改为整个loop的总工具请求上限，额度用完后下一步强制`tool_choice=none`以仅生成回答。
  - 工具名与规范化JSON参数生成SHA-256指纹；首次执行使用`run_id + fingerprint`作为持久幂等key，相同调用复用已有结果而不再执行。
  - 允许有上限的一次重复结果回传，让模型有机会修正；持续重复则以`repeated_tool_call_limit`终止，步数用尽以`max_steps_exhausted`终止。
  - API执行结果返回`run_id`、每步outcome、调用指纹、请求/真实执行/重复次数和终止原因；同步持久化`agent_loop_step`与`function_calling_tool`事件。
  - 供应商返回usage时累计prompt/completion/total token；缺失usage时保留未报告状态，不伪造实测token。
- 验证：
  - 注入模拟客户端验证两轮不同工具后第三轮回答、逐步消息回传、usage累计和3条持久Trace。
  - 验证相同调用请求3次但只真正执行1次，第2次复用、第3次以重复上限终止。
  - 验证工具总额度用尽后只能回答，以及连续工具调用在步数上限时明确失败。所有模型次数均记为`simulated_model_calls`，不作为真实LLM成果。

### 阶段 24：短期/长期 Memory 完整化

- 状态：24.1数据迁移、短期约束并发保护、长期记忆准入及Tool调用关联已完成；自动滚动摘要和turn/run累计模型预算尚未实施。
- 短期记忆（session scope）：
  - 保存当前数据集/版本、目标列、评价协议、预算、选定run、取消条件、最近对话和有界滚动摘要。
  - 每次约束更新增加`constraint_version`，并通过乐观并发检查防止两个请求相互覆盖；摘要保留覆盖的turn ID范围。
  - 上下文组装顺序为：固定安全指令、当前显式约束、当前run摘要、已验证证据、适用的长期记忆、滚动摘要、最近消息；每部分记录token和裁剪原因。
- 长期记忆（workspace scope）：
  - 只保存用户明确确认的偏好，或验收通过run中的结构化经验；模型推测、未验证文献和失败方案不能直接写成成功事实。
  - 记忆保留`kind`、来源turn/run、验收状态、适用的数据特征/评估协议、版本、时间和失效状态；失败经验只能以`failure`kind用于避错。
  - 当前应用没有用户身份/权限系统，因此不宣称跨用户个性化；第一版长期记忆限于服务端受控workspace。
- 调用与预算：
  - Function Calling和Tool Registry完整写入`session_id/turn_id`；模型、工具、证据与摘要步骤可回溯到同一turn。
  - 单次上下文裁剪与turn/run累计模型预算分开；优先保留显式约束，再减少历史和证据，累计额度不足时停止下一次模型请求。
- 预计验收：约束继承与纠正、会话隔离、并发版本冲突、摘要溯源、长期记忆准入/失效、失败经验不被当成成功、删除后不再进入上下文，以及累计预算停止。
- 24.1实现：
  - Alembic `20260918_0005`为session增加workspace、约束版本和摘要turn溯源，为turn增加生效约束版本，为memory增加workspace、验收状态、来源run和适用性。SQLite使用等价的可重放增量补列。
  - session约束更新使用`constraint_version`比较更新；陈旧请求返回显式冲突，FastAPI映射为HTTP 409，不静默覆盖。
  - `candidate`长期记忆不进入上下文；确认后只对同workspace会话可见。`successful_experience`只能由已存在且状态为`passed`的run确认。
  - 摘要更新要求`through_turn_id`实际属于当前会话；上下文分开session/workspace memory，仅选择`active + confirmed`记忆并记录裁剪原因。
  - Agent Loop和Tool Registry现已将`session_id/turn_id`写入Tool Call，不属于该session的turn会在模型调用前拒绝。
- 24.1验证：
  - SQLite旧库增量初始化后实查全8个新字段存在；候选/确认记忆可见性、workspace/session隔离、摘要溯源和Tool关联测试通过。
  - 本地PostgreSQL真实从`0004`升级到`0005`，revision、字段和`idx_yieldmind_memories_context`实查通过；真实CAS冲突及候选/确认准入smoke通过，临时数据已清理，PostgreSQL已停止。

### 阶段 25：多格式领域文献库与可追溯入库

- 状态：已完成多格式解析、可复现下载、来源元数据、语料隔离、批量限流、真实Qwen入库及幂等复验；文献查询人工标注尚未开始。
- 前提核查：
  - 原7篇项目文档仅约11KB/20个chunk，150行盲审候选来自20个唯一chunk，平均复用7.5次；“语料太少导致高复用”成立，但pairwise词法相似度均值仅`0.147`，不能简化为“分块重叠导致所有片段高度相似”。
  - 只依据PDF URL成功不能证明论文主题正确；已发现并排除一篇URL可下载但内容是石墨烯锂扩散的无关论文。
  - 文献库是可引用证据，不是自动成为真值标签；在没有外部文献查询标注前，不报文献检索Recall/MRR。
- 实现：
  - 新增Markdown/TXT/PDF/DOCX/HTML格式感知加载；保留标题层级、PDF页码，去除多页重复页眉/页脚，空扫描PDF明确报告需要OCR。
  - 新增8篇开放文献的manifest和受控下载器；固定HTTPS地址、DOI、许可、SHA-256和100MiB上限，拒绝私有地址、格式伪装及哈希变化。
  - PostgreSQL/SQLite增加`corpus/source_url/doi/license/metadata_json/page_start/page_end`，Alembic迁移为`0006/0007`；Chroma与SQL检索都支持`project/literature`隔离。
  - 项目文档保持`700/80`，142页文献使用`1800/180`，避免将长论文碎成过多短片段；分块参数及解析语义纳入`split_version v3`。
  - Qwen HTTP客户默认每批8条，服务端硬上限16条。旧实现一次发送30--75条时峰值约11.4GB；批8实测约5.76GiB，批16约7.0GiB且单条更慢，因此不默认16。
  - SQL是检索事实源；向量命中必须对应SQL中`available`的chunk，旧向量只在新SQL chunk提交后删除，避免跨存储中断暴露未提交内容。
- 已完成验证：
  - 8个PDF的SHA-256全部匹配，共142页；`pypdf`layout提取约44.4万字符，无整页空白，产生326个文献chunk；7篇项目文档产生39个chunk。
  - hashing配置对真实PostgreSQL+Chroma完成7+8篇入库，再次运行`document_encode_calls=0`；两个corpus隔离、文献DOI/URL/页码完整性检查均通过，`real_llm_calls=0`。
  - Qwen索引现实存在7篇项目文档/39 chunks和8篇文献/326 chunks；本次增量补全在新服务进程中真实编码210条文本（其中208个文档chunk、2条验收查询），CPU服务进程峰值RSS=`6,186,856,448 bytes`（约5.76GiB）。
  - Qwen幂等复跑返回`ok=true`、`document_encode_calls=0`、`http_encode_calls=2`（仅两条验收query）；项目/文献corpus隔离及文献DOI/URL/页码完整性均为true，`real_llm_calls=0`、`simulated_model_calls=0`。
  - 首次离线全回归为`36/37`：SQLite评测库与PostgreSQL入库共用Chroma目录，SQL事实源正确拒绝了异库向量，但固定候选深度被无效向量占满。修复为离线评测使用临时独立Chroma，核心检索在遇到未提交/异库向量时自适应扩大候选；复跑恢复`37/37`。
  - 格式解析、来源身份、语料过滤、下载安全、HTTP批处理及跨存储中断隔离均有自动化回归。
- 已知边界：
  - 未做OCR，无法保证复杂表格、公式、旋转图注的无损提取；检索证据仍需回到DOI/页码核对。
  - 8篇文献只是初始语料，覆盖水泥、浆体、高固相膏与yielding liquids，不代表完整领域分布；发表年份和开放获取条件也会引入选择偏差。

### 阶段 26：固定 Gold Label 重排对照

- 状态：可复现评测框架和真实Qwen embedding余弦重排消融已完成；Qwen3 cross-encoder因固定版本模型尚未缓存且下载网络不可达而未执行。
- 实现：
  - 新增`scripts/run_yieldmind_reranker_eval.py`，严格校验Gold行、case内query和连续原始排名；平分时按原始排名稳定排序。
  - 报告同时保存原始/重排Recall@5、MRR、Top-1、nDCG、逐case候选分数、Top-1升降、输入SHA-256、模型固定revision、推理延迟和峰值RSS。
  - 正式Qwen3-Reranker采用官方yes/no causal-LM打分协议并默认禁止下载；另提供已缓存Qwen3-Embedding余弦重排作为明确标注的bi-encoder消融，二者不会混称。
- 真实结果：
  - 固定输入为最终人工Gold的30个case/150个Top-5候选；原始指标Recall@5=`0.9667`、MRR=`0.8861`、Top-1=`0.8333`、nDCG@5=`0.9409`。
  - 已缓存`Qwen/Qwen3-Embedding-0.6B@97b0c614...`在CPU真实编码30个唯一query和20个唯一原文chunk；推理`27.7413s`，按150对摊销`184.94ms/pair`，峰值RSS约`3.46GB`，`real_llm_calls=0`。
  - 余弦重排后Recall@5不变，MRR降至`0.8389`、Top-1降至`0.7667`、nDCG@5降至`0.8921`；5个case更换Top-1，没有把相关片段提升到首位，反而有2个直接相关Top-1被降级，因此明确`do_not_enable`。
  - 完整本地报告SHA-256=`c2ca1fc53aabd35c14d7348d22f9b1ae5de216ecb7fd91b77a28002d2238daef`；Git精简结果为`evals/results/yieldmind_qwen3_embedding_cosine_rerank_20260919.json`。
  - `Qwen/Qwen3-Reranker-0.6B@204631fe...`下载在显式授权后仍返回`No route to host`；未产生或伪造cross-encoder指标，默认检索保持不变。

### 阶段 27：候选与固定基线的全局选择闭环

- 状态：已完成。候选冠军不再自动等于最终推荐，固定基线可在同协议OOF指标更好时获胜。
- 公平性修正：
  - 固定基线与候选统一以OOF RMSE选择；不再用基线折均RMSE与候选OOF RMSE混比。
  - 同轮候选使用相同的KFold切分随机种子，避免策略间由不同折分配引入额外方差。
  - 仅在评估协议一致且两侧OOF RMSE均有效时比较；协议不匹配时拒绝声称谁更好。
  - 候选只在OOF RMSE严格更低时获胜，平局选更简单的固定基线。
- 端到端接线：`final_selection`进入benchmark产物、workflow stage payload、最终JSON/Markdown报告和前端工作台；固定基线新增最多200条OOF预测预览，因此基线获胜时散点图仍显示真实OOF点。
- 默认数据实测：候选相对Ridge基线的OOF RMSE增加`3.4555`，最终选择`baseline_ridge_linear`；选择原因、双方指标和获胜预测点均在结果中可审计。

### 阶段 28：结构感知分块与检索路由

- 状态：第一阶段已完成；新旧分块算法均可复现，现有本地hashing知识库已重建为`section_aware_v4`。
- 分块实现：
  - 保留`recursive_chars_v3`作为历史对照，新增`section_aware_v4`；代码、表格和公式作为原子块，普通内容按标题/段落/句子分块。
  - chunk持久化`parent_id`、`chunk_role`、`token_count`、`split_version`和扩展元数据；新增SQLite增量升级和Alembic `20260919_0008`。
  - 向量索引使用title/section/content type上下文增强文本，引用返回仍使用未注入合成头的原始证据文本。
- 检索实现：
  - 仍以TopK为候选选择策略；hybrid默认从vector/BM25各取`3K`后做RRF，未把不可比的cosine/BM25/RRF分数当作TopP概率。
  - 新增`routing_mode=auto`，对project/literature进行可审计的确定性路由；无强路由证据时搜索两个corpus，路由后corpus无结果时显式降级为全库。
  - 新增父章节展开、父块去重和总上下文字符预算；工作流与前端快速检索显式使用auto routing、parent expansion、dedup和12,000字符预算。
- 离线快速A/B（local hashing，30条旧回归case）：补跑同参数后，`700/80 v3` hybrid为Recall@5=`0.9000`、MRR=`0.8361`、Top-1=`0.8000`，`700/80 v4`为`0.9000/0.8417/0.8000`；旧`0.8333/0.8083/0.8000`报告实际是`recursive_chars_1200_180_v1`，不再当作同参数v3基线。v4的`450/60`为`0.9000/0.8361/0.8000`，`900/120`为`0.9000/0.8417/0.8000`。该轮只用于分块候选筛选，不代替新chunk的池化人工Gold或Qwen复验；精简结果见`evals/results/yieldmind_chunking_hashing_ablation_20260919.json`。
- 真实重建：local hashing配置下项目语料7篇/39 chunks（`700/80 v4`）、文献8篇/326 chunks（`1800/180 v4`）；corpus隔离、文献DOI/URL/页码完整性全部通过，`real_llm_calls=0`。

### 阶段 29：同参数跨分块池化盲审

- 状态：Top-1精简包的第一位人工评审已完成；第二位独立评审和完整168行池化评审尚未完成，因此当前是单评审快速筛选结果，不是最终Gold。
- 对照严格限定为`recursive_chars_v3 700/80` vs `section_aware_v4 700/80`，两侧均为local hashing + hybrid TopK=5；生成器会校验报告内`split_version`和配置，不一致时直接拒绝。
- 30个问题合计300个系统候选映射；按候选原文SHA-256在每个case内去重后为168行，其中132行被两系统共同召回，36行只属于其中一套系统。
- 为降低人工成本，额外生成Top-1两阶段快速筛选包：30个case只需32行/人，28个case的两套Top-1原文相同，只有`agent_multi_role`和`eval_oof`各需评审两条候选。该包只能用于Top-1准确率快速筛选，不会被误用于宣称Recall@5、MRR或nDCG。
- 公开CSV只包含问题、随机候选编号和证据文本，不包含系统身份、原排名、chunk ID或source path；两份CSV除`reviewer_id`外完全一致。私有manifest保留每套系统的排名，分析器可使用同一份Gold分别计算Recall@5、MRR、Top-1和nDCG。
- 完整产物：`evals/review/yieldmind_chunking_review_reviewer_1.csv`、`yieldmind_chunking_review_reviewer_2.csv`、`README_chunking_pool.md`；Top-1精简产物：`yieldmind_chunking_top1_review_reviewer_1.csv`、`yieldmind_chunking_top1_review_reviewer_2.csv`、`README_chunking_top1.md`。私有排名映射位于`agent_workspace/yieldmind/retrieval_reviews/`。
- 标注后处理已接入池化manifest：双人一致性报告和最终Gold报告会直接输出两套系统的Top-1直接相关率、差值、胜/负/平及ta例；对Top-1精简包不计算或宣称不可观测的完整Top-5质量。
- 第一位评审结果：32行全部完成，标签分布为`2:18 / 1:7 / 0:7`，28条high confidence、4条medium confidence、14条带备注；候选原文哈希全部通过manifest校验。
- 单评审Top-1对照：两套系统的直接相关率均为`18/30=60.0%`，至少部分相关率均为`25/30=83.3%`。28/30个case返回同一原文；`agent_multi_role`和`eval_oof`返回不同原文，但两侧都被评为`0`，因此结果为`0胜/0负/30平`、差值`0.0`。
- 当前决策：没有观察到`section_aware_v4`相对`recursive_chars_v3`的Top-1质量收益，不能据此宣称结构感知分块提升检索效果；完整结果见`evals/results/yieldmind_chunking_top1_reviewer_1_20260921.json`。若要形成Gold和一致性指标，仍需第二位评审独立完成同一32行或继续完成完整168行池化评审。

### 阶段 30：检索语料路由专项评测

- 状态：已完成确定性路由用例、词边界修正、文献意图补全和持久化索引端到端复验。
- 新增32条固定路由case：10条project、11条literature、11条混合/无强意图；评测的是语料意图和corpus过滤，不是回答语义相关性。
- 原路由使用简单子串计数，会把`rapid`中的`api`、`redistribution`中的`redis`当作项目信号；现对拉丁词使用token边界，中文仍使用短语匹配。
- 文献意图补全`SAOS`、`BreakPro`、`rheometer`、`shear rate`、`plug flow`、`wall slip`、`yielding liquids`、`solid volume fraction`、DOI和论文检索等当前语料的高置信表达；显式corpus过滤仍优先于auto route，平分/无信号仍搜全库。
- 首轮30 case静态路由准确率为`0.9000`；最终32 case静态路由`32/32`，无强意图强制路由率`0`。同一批case在实际local-hashing持久化索引上检索`32/32`通过，路由后corpus纯度`1.0000`。
- 精简结果见`evals/results/yieldmind_routing_eval_20260919.json`；完整本地报告为`agent_workspace/yieldmind/routing_evals/routing_eval_20260919_191402.json`。

### 阶段 31：结构化 Manager 路由与逐 Agent 行为输出

- 状态：离线/模拟领域链路实现与回归已完成；没有调用真实 LLM。
- 新增`DomainManagerRouter`，为每次领域路由输出可序列化决策：`next_node`、`next_agent`、`reason_code`、`feedback`、`remaining_revision_budget`、`allowed_next_nodes`和`validated`。
- 所有路由先受合法转换表约束，再交给LangGraph条件边执行；失败、取消、pre-execution审批、review通过、需要修订和预算耗尽分别使用明确原因码。路由决策在普通节点内写入state，避免依赖LangGraph条件函数中的不可持久化修改。
- `manager_decisions`保存完整Manager决策，`route_history`保存精简路由审计；review回环会明确记录`review_requires_revision -> revision_feedback -> CandidateAgent`及剩余预算。
- 领域节点新增`agent_progress`开始/结束事件，记录Agent、动作、attempt、耗时、状态和输出摘要；Manager路由使用独立`manager_decision`事件，避免把图路由与专业Agent执行混为一谈。
- Agent角色按真实职责输出：Data/Search/Candidate/Model/Operation分别展示专业行为；requirements、审批、review、revision feedback、取消和结束归属Agent Manager。这里的Agent名称表示领域职责，不宣称每个角色都是独立服务。
- 新增`POST /api/workflows/domain/stream` NDJSON入口；静态工作台事件处理器可识别Manager决策，历史结果优先从`agent_activities`恢复，并补充OperationAgent展示。
- 验证覆盖正常直线流程、一次有界修订、Operation取消、未授权真实模型闸门、异步任务幂等和领域流式事件；全量回归`103 passed`，`real_llm_calls=0`。

### 阶段 32：Agent 行为审计前端闭环

- 状态：实现与自动化回归已完成；未触发真实 LLM。
- 静态工作台新增“Agent 行为”页签，不再只显示角色卡片的最终状态；运行中和历史运行都可以展示完整行为时间线。
- 普通Agent动作展示角色、action、attempt、开始/完成时间、耗时、输入摘要、输出摘要和工具调用；Manager路由单独展示前序节点、下一Agent、原因码、剩余修订预算、反馈和完整结构化决策。
- 领域工作流开始事件补充当前修订轮次、剩余预算、已有产物、是否存在Manager反馈和执行模式；完成事件按白名单提取产物名、调用模式、审批结果、Operation返回码、review结论和修订反馈，避免把不可控的大对象或敏感运行状态直接输出到页面。
- 历史领域运行优先从`agent_activities`恢复；离线历史运行则兼容既有增强`stages`。Manager决策与普通Agent进度继续使用不同事件类型，前端统计角色数、完成动作数和路由决策数。
- 前端补充OperationAgent角色、领域阶段中文标签和行为审计响应式样式；脚本语法检查、领域工作流测试和静态控制台测试均通过。

### 阶段 33：离线/领域双模式工作台与审计交互

- 状态：实现、自动化回归和真实HTTP联调已完成；未执行人工分块标注，也未调用真实LLM。
- 修复领域历史运行沿用离线指标布局的问题：前端通过`manager_args/manager_decisions/workflow_kind`识别领域结果，不再用空OOF指标和空候选表占位。
- 离线模式继续展示OOF R²/RMSE/MAE/MAPE、候选策略和预测散点；领域模式改为展示修订轮次、Manager决策数、完成节点数、模型调用模式，以及执行前审批、Operation返回、Manager review、最后路由、修订反馈和进程控制摘要。
- StateGraph轨迹在领域模式显示Manager修订次数和路由决策数，并统一使用中文阶段标签；加载领域历史运行时自动进入Agent行为页，离线/领域产物、Trace和原始结果仍可查看。
- Agent行为页新增角色、事件类型和状态筛选；支持把当前真实行为事件导出为`yieldmind-agent-audit-v1` JSON，不生成或补齐不存在的Agent事件。
- 响应式布局补充行为筛选工具栏，小屏下自动纵向排列；离线与领域模式共享同一套安全转义和结构化详情组件。
- 真实FastAPI HTTP验收：根页面、JS、CSS、health和默认数据均返回200；20样本/2折离线流式运行`run_d59daa7b12c7`返回18条真实`agent_progress`、7个Agent角色、9个阶段和最终`passed`。本机没有Playwright/Chromium，因此该验收不宣称浏览器截图通过。

### 阶段 34：前端回归真实领域 Agent 与 Redis 连通

- 状态：已完成真实链路切换、Redis连通、前端状态收敛和一次显式授权的真实 LLM 端到端验证；验证运行因两轮 Operation 超时最终失败，没有将其粉饰为成功。
- 前端语义修正：主按钮只调用`/api/workflows/domain/stream`，必须显式确认真实 LLM 与代码执行；确定性 sklearn 评测移到“离线基线评测（非 Agent）”，不再伪造 Data/Model/Reporter Agent 协作。
- 角色对齐：页面只保留真实领域链路存在的`Agent Manager / DataAgent / SearchAgent / CandidateAgent / ModelAgent / OperationAgent`；去掉不存在的 ReviewAgent/ReportAgent 占位。
- 交互合并：将“Agent状态”与“Agent行为”合并为主页滚动协作区；左侧是可筛选角色状态，右侧持续输出 Agent 动作、Manager 路由、耗时、尝试次数、输入/输出摘要和审计详情，支持自动滚动和 JSON 导出。
- 状态闭环：最终结果会将残留`running`收敛为`passed/failed/cancelled`，未调用角色标记`skipped`；历史运行如果 Operation `rcode != 0`或`timed_out=true`，前端会纠正旧事件中的假`passed`。
- 细粒度可观测性：Operation 子进程的代码生成、preflight拒绝、fallback、重试、源码保存和 champion 选择会写入 run-local JSONL，由父流程增量转换为`agent-activity-detail-v1`事件；细节事件进入滚动区，但不伪装成新 StateGraph 节点。
- Redis：Docker Desktop 因`com.docker.vpnkit`后端失败无法启动容器；为避免引入 Homebrew 的大规模系统升级，在`.runtime/`构建 Redis 7.2.5 并仅绑定`127.0.0.1:6379`。`redis-cli PING=PONG`，且`/health/dependencies` 返回`redis.ok=true`。
- 真实运行`run_5461af410bfe`：`live_llm + langgraph_stategraph`，总耗时`1570.486s`；CandidateAgent 两轮约`123.0s/144.2s`，ModelAgent 约`40.8s/60.2s`，OperationAgent 两轮均达`600s`超时。Manager 在第一轮 review 后生成 revision feedback 并重跑 Candidate/Model/Operation，第二轮预算耗尽后以`failed`结束。运行产生多份 LLM 模型/机理源码、`free_search_report.json`、预测、champion 模型和`predict.py`，因 Operation 超时不声称端到端成功。
- 验证：`node --check yieldmind/static/app.js`、`py_compile yieldmind/domain_workflow.py`通过；领域流程回归`10 passed`，其中新增 Operation 详细日志去重、超时失败语义和失败后 Manager review 路由测试；静态前端契约测试另行通过。

### 阶段 35：Operation 大结果回传死锁修复与真实端到端通过

- 根因：首次真实运行的两轮 Operation 其实已写出 champion、predictions 和成功的 round result；子进程在向`multiprocessing.Queue`回传大结果时填满管道，父进程又先等待子进程退出，因此双方互等并在600秒被误判超时。
- 修复：`run_managed_process`在worker存活期间持续排空结果队列；收到结果后再给worker有界清理时间，如仍有子孙进程不退出则清理进程组，但不丢弃已回传结果。
- 可观测性：Operation 滚动区新增`process_started`和`result_returned`事件，后者带真实进程状态、耗时、exitcode以及是否发送terminate/kill。
- 回归：新增8 MB返回值用例，证明大于管道缓冲区的结果不再死锁；全量测试`107 passed`。
- 真实验收`run_6f40a7d9d715`：`live_llm + langgraph_stategraph`，总耗时约`599.19s`；CandidateAgent=`189.86s`、ModelAgent=`45.44s`、OperationAgent=`362.22s`。Operation 回传`1,093,267`bytes的`run_result.json`，`exitcode=0`、未发送terminate/kill；Manager review=`accepted`，最终状态=`passed`。
- 真实行为证据：3个模型代码在第一次尝试通过；3个机理代码被preflight按“phi=0归零”和“必须使用phi”约束拒绝；搜索最终从合格池选出`yodel::mod_resid_hgb::fus_mech_resid_temp#2`，Manager根据真实产物通过复核。

### 阶段 36：可验证的跨任务 Repair Memory

- 状态：第一版已完成。人工分块标注不是它的前置条件，两条工作可以独立进行。
- 三态生命周期：Operation报错时只写入`candidate`；未验证候选不会被后续任务检索；只有后续Operation `rcode=0`且Manager review通过后才升级为`confirmed`。
- 隔离与去重：按`workspace_id + execution_mode`隔离；对路径、行号、数值等易变内容归一化后生成错误指纹，同一run的同类错误幂等写入。取消操作不会记成修复经验。
- 使用方式：已验证记忆作为独立的`repair_memory_context`传给CandidateAgent、ModelAgent和Operation的执行前检查，不会伪装成当前轮Manager feedback。
- 存储：复用现有`yieldmind_memories`表，无需新增数据库迁移；记忆中保留错误签名、原始失败摘要、成功结果、修复建议和来源run。
- 当前边界：第一版按工作区、执行模式和时间顺序取最近已验证经验，还没做语义相似度检索；历史run不自动回填，避免将无法确认因果关系的旧错误冒充已验证修复。
- 回归：Repair Memory与领域工作流定向测试`14 passed`；全量测试`111 passed`。

### 阶段 37：知识库检索接入领域 SearchAgent 主链路

- 状态：已完成。领域`/api/workflows/domain/stream`现在默认启用知识库检索，前端可独立开关；外部搜索仍是可选的第二证据源。
- 查询规划：使用用户需求、显式query和数据列能力确定性生成最多3个query；包含`phi/sp_percent`时增加YODEL/packing查询，包含剪切速率/流动曲线时才增加HB/Bingham查询。
- 检索协议：每个query先做project/literature自动语料路由，vector与BM25+两路并行召回，再用RRF按名次融合；候选仍是TopK，不把分数误当为TopP概率。
- 上下文处理：默认TopK=`5`、父章节展开、父块去重和全局`12,000`字符预算；多query候选按rank轮询合并，避免第一个query耗尽全部上下文。
- 证据门禁：每个主证据的`chunk_id/text_hash/index_version/document_id/document_version`在进入Agent prompt前都会对当前数据库重新校验；校验失败的chunk不会进入Candidate/Model/Operation上下文。
- 协议兼容：知识库hit会转成原CandidateAgent/ModelAgent可消费的`snippets`协议，但额外保留chunk、文档、页码、路由、检索通道和分块版本；知识库与外部证据轮询合并，不会互相覆盖。
- 降级：知识库未就绪或Qwen embedding服务不可用时，SearchAgent会保留`failed/degraded`状态和错误，再根据是否有外部证据以及`require_search_results`决定继续或终止，不伪造本地证据。
- 可观测性：SearchAgent行为的输入摘要显示知识库开关、TopK、上下文预算和外部搜索开关；输出摘要显示路由/检索模式、embedding模型、有效/无效证据数和实际上下文字符数。
- 真实本地验收：`local_hashing_v1`在当前365个chunk索引上对用户需求+YODEL数据能力查询自动路由到`literature`，稳定证据校验`7/7`通过，多query按rank轮询后在全局预算内选入3条、共`11,999`字符，其中1条同时由vector和BM25+命中，`real_llm_calls=0`。真实冷启动曾暴露Chroma/BM25线程同时首次导入NumPy的部分初始化竞态，已改为主线程预热依赖后再并行检索。
- 回归：新增领域查询规划、路由检索、证据校验、旧Agent协议转换和RealDomainAdapter主链路测试；最终全量回归`114 passed, 265 warnings`。

### 阶段 38：数据血缘、Anchor语义与评测口径防混淆

- 状态：已完成第一版。DataAgent不再只输出schema和行列数，而是输出可机读的`dataset_lineage`和`anchor_audit`。
- 当前主数据已确认为`augmented_development`：200条、全部`data_fidelity=augmented_hf`、`is_augmented=true`、67个特征；用于候选搜索和OOF开发评测，不自动冒充独立业务Anchor。
- 多保真数据会被单独标记为`multifidelity_development`；系统明确提示需按`base_hf_id`/批次做grouped或paired评测，两种数字不得混写。
- Lian Table 6的16条数据已标记为`literature_mechanism_anchor`；作用是文献机理一致性检查，不是工艺药浆的同域业务Holdout。
- Anchor门禁现在比较训练/锚点schema、目标单位和完整特征集。当前200条工艺数据与Table 6实测为不兼容：锚点缺少66个训练特征，且属于不同材料/特征域；因此默认禁用Anchor评分并记录具体理由。
- 前端默认数据卡片现在显示“200条增强开发集”及评浏边界；DataAgent完成事件会展示数据角色、保真计数、增强行数、允许的评测用途和Anchor门禁结果。
- 统一对外口径：当前主流程成绩为200条增强开发数据上的5折OOF RMSE=`0.2839 Pa`、R²=`0.9943`，相对固定基线RMSE=`0.3011`下降约`5.73%`；该口径不等于生产泛化或独立业务Anchor成绩。
- 回归：新增200条增强集、20+180多保真集、文献锚点和跨域不兼容判定测试；真实CSV冒烟确认主数据角色正确、Table 6缺少66个训练特征并被禁用；最终全量回归`118 passed, 265 warnings`。

### 阶段 39：多保真评测的严格分组隔离

- 状态：已完成。阶段检查发现原联合搜索会对20条高保真数据做普通KFold，并在每折训练中使用全部180条低保真/增强行；这会让验证批次对应的增强样本进入训练，存在组级泄漏风险。
- 拆分升级：高保真样本改用`GroupKFold`，分组键优先级为`base_hf_id -> raw_batch_id -> batch_id`；普通候选每折只看高保真训练组，多保真候选额外剔除与验证组同ID的所有低保真行。
- 失败门禁：多保真候选如缺少可用分组元数据，直接标记失败，不再回退到会混入同组增强数据的随机KFold。
- 可审计输出：每折记录高/低保真训练数、验证数、因同组被剔除的增强行数和两类组重叠列表；主搜索返回也新增`fixed_baseline`明细，可直接检查基线是否同样使用分组协议。
- 真实数据冒烟：20个高保真组+180条低保真数据做5折评测；每折为16个高保真训练组/4个验证组，保留144条低保真训练行并剔除36条验证同组行；`train_valid_group_overlap=[]`、`lf_train_validation_group_overlap=[]`，严格隔离通过。
- 新协议对照线：20个高保真组上的5折Group OOF中，Ridge固定基线最优，RMSE=`0.6035 Pa`、R²=`0.3461`；这只是严格分组对照线，还不是重跑真实候选后的最终冠军成绩。
- 口径影响：该修改只升级20+180多保真数据的评测协议，不会改写200条增强开发集已报告的5折OOF RMSE=`0.2839 Pa`与R²=`0.9943`；两组数字仍不能直接混比。
- 回归：新增同组低保真剔除、缺少分组元数据拒绝和主搜索基线审计集成测试；最终全量回归`121 passed, 265 warnings`。

### 阶段 40：Session Context 主链路闭环

- 状态：已完成。原先已有session、turn、约束版本、确认记忆、rolling summary和`build_context()`，但Function Calling和领域LangGraph只关联`session_id/turn_id`审计，没有真正消费组合后的上下文。
- Function Calling：`execute_plan` 现在在第一次模型调用前加载有界session context，把当前约束、已确认的session/workspace memory、rolling summary、当前run摘要和最近消息注入system context；审计只保留章节名、预算、裁剪项和约束版本，不把全文复制到事件。
- 领域LangGraph：`DomainWorkflowRequest`新增`session_id/turn_id/session_context_max_tokens`；启动节点验证turn归属与workspace隔离，加载后传给CandidateAgent、ModelAgent和Operation执行合同，并将session绑定到真实run。
- 上下文优先级：当前用户需求/显式约束 > 已验证Repair Memory > session/workspace已确认记忆 > rolling summary/最近消息 > 检索证据；历史内容不能用来绕过当前安全合同。
- 预算修正：不再因“必选章节”无限超出`max_context_tokens`；超长约束按剩余预算裁剪，记录`estimated_tokens/included_tokens/clipped`，总量不超过声明上限。
- 前端：浏览器通过`localStorage`复用session；每次真实Agent运行前先写入turn，再把`session_id/turn_id`传给领域流。Agent行为区可看到已选上下文章节和预算，不展开敏感全文。
- 边界：本阶段完成时Repair Memory仍是`workspace_id + execution_mode + recency`；后续已在阶段44补齐任务预排序和真实执行错误的相似度重排。rolling summary仍通过可追溯API更新，未做自动LLM压缩。
- 回归：新增Function Calling上下文注入、领域run绑定、预算硬上限和跨workspace拒绝测试；最终全量回归`125 passed, 265 warnings`。

### 阶段 41：ToolExecutor、受控参数自修正与风险策略

- 状态：已完成主闭环。新增`yieldmind/tool_execution.py`，实际承担工具查找后的参数准备、Policy判定、幂等占位、handler执行、异常归一化、脱敏和审计。`ToolRegistry.execute()`仅作旧调用方的兼容委托，API、Function Calling和真实离线Workflow已直接走ToolExecutor。
- 执行上下文：新增`ExecutionContext`，保留actor、caller、run/session/turn、执行模式和可信grants；授权从模型可生成的args中分离。
- 风险策略：ToolSpec/ToolDefinition新增`execution_backend` 和`required_grants`。任意Python subprocess由medium提升为high；Docker需`docker_execute`，subprocess需`subprocess_execute`，真实LLM pipeline同时需`live_llm + subprocess_execute`，知识写入需`knowledge_write`。high工具如声明`in_process`会被Policy直接拒绝。
- 授权边界：即使模型在工具args中自行填写`allow_docker=true`，没有外层`ExecutionContext` grant仍会在handler之前被拒绝；拒绝尝试以`denied`状态留痕。
- 结构化错误：未知工具、JSON不合法、参数不是object和Pydantic字段错误统一输出`code/phase/tool_name/field_errors/correction_hint/repairable`；参数校验失败时不调用handler。
- 有限自修正：真实Agent Loop把结构化tool error作为tool message返回模型，默认只允许一次参数修正；第二次仍不合法时以`tool_argument_repair_exhausted`终止。失败批次里即使有其他合法调用也整批不执行，避免部分副作用。
- 审计：ToolResult保留`error_detail` 和`audit`，可查risk、backend、caller、grants和PolicyDecision；参数失败记为`invalid`，策略拒绝记为`denied`，幂等占位只在验证与授权通过后创建。
- 当前边界：本阶段是轻量capability policy，不是完整RBAC；幂等临时错误选择性重试已在阶段43完成，并行工具和MCP/动态插件仍按后续优先级保留。
- 回归：新增结构化校验失败、high风险授权、模型不可自授权、一次修正成功、二次失败终止和混合批次零部分执行测试；最终全量回归`131 passed, 265 warnings`。

### 阶段 42：工具结果有界回传

- 状态：已完成。实时Agent Loop不再把工具完整返回值无上限塞回模型上下文；`ToolPlanRequest.max_tool_result_chars`默认限制单个tool message为12000字符，可在1000～50000之间显式调整。
- 保留策略：优先保留成功/失败状态、结构化错误、指标、选中方案、证据标识、warning和artifact path；列表和长文本做有界摘要，并返回`result_truncated/original_result_chars/model_result_budget_chars`供审计。
- 双路结果：裁剪只发生在“回给模型”的tool message。`ToolExecutionResult.results`、运行事件和数据库工具审计仍保留脱敏后的完整结果，避免因节省上下文而丢失可追溯性。
- 安全边界：裁剪前先递归脱敏；即使返回值进入最小预览分支，仍对最终JSON做严格字符上限约束，不会因超长artifact或error字段突破预算。
- 回归：新增大结果严格预算、敏感字段脱敏、artifact指针保留和“模型裁剪/执行结果完整”集成测试；最终全量回归`133 passed, 265 warnings`。

### 阶段 43：幂等保护下的选择性重试

- 状态：已完成。ToolSpec/ToolDefinition新增`retry_safe`和`max_transient_retries`，重试能力从隐式行为变成可查询的工具合同；当前只为只读的`search_knowledge`显式开启，最多重试2次。
- 三重门禁：只有工具声明`retry_safe=true`、调用带有幂等键、并且本次失败被明确标记`retryable=true`时才会重试。参数错误、Policy拒绝、幂等冲突、普通代码异常和未授权副作用工具均不重试。
- 临时错误边界：识别Timeout/Connection类异常和handler显式返回的retryable错误；使用有上限的指数退避，单次等待最多2秒，不对未知异常猜测性重试。
- 审计：ToolResult.audit新增`execution_attempts/transient_retries/retry_delays_seconds`；本可重试但因缺幂等键而被抑制时，记录`retry_suppressed_reason`。同一逻辑调用的多次尝试只落一条tool-call审计记录。
- 回归：新增“两次临时失败后成功”、缺幂等键不重试和普通RuntimeError不重试测试；最终全量回归`135 passed, 265 warnings`。

### 阶段 44：Repair Memory 相似度检索与错误后重排

- 状态：已完成主链路。仍先按`workspace_id + execution_mode + confirmed + active`做强过滤，不允许跨工作区、跨执行模式或未验证candidate进入排序候选。
- 两阶段检索：任务开始时用用户需求和查询做预排序；OperationAgent真实报错后，再用`error_logs + action_result`归一化错误重排，过滤低相关项，并更新下一轮CandidateAgent、ModelAgent和执行合同的Repair Context。
- 排序信号：组合embedding cosine、词项归一化相似度、时间新鲜度和完全一致的error signature。服务配置Qwen3 Embedding时使用真实语义向量；当前本地默认为确定性hashing embedding，不冒充Qwen语义效果。
- 降级机制：embedding客户端不可用时自动退化为词项相似度+时间排序，仅记录异常类型，不中断建模工作流。内存向量缓存以`memory_id + updated_at`失效，确保修复内容更新后重新编码。
- 可观察性：每条命中记录rank、总分、embedding/词项/时间分量、方法和是否降级；前端Agent行为流会显示Manager在Operation错误后的记忆检索动作，下一轮Candidate/Model输入摘要展示命中数、排序方法和Top score。
- 回归：新增NaN错误与Docker超时的排序对照、embedding失败降级、真实失败后重排及前端事件契约测试；最终全量回归`137 passed, 265 warnings`。

### 阶段 45：Repair Memory 数据与运行环境适用性门禁

- 状态：已完成。DataAgent的数据画像不再只用于页面展示；LangGraph现在持久传递`data_lineage/data_contract`，数据合同包含数据角色、源Schema、目标列、特征数与排序后特征集合的SHA-256签名。
- 运行合同：每次领域工作流记录Python、NumPy、Pandas、scikit-learn版本、操作系统、机器架构、执行后端和`operation_contract_version`；不记录环境变量密文或凭据。
- 错误分类门禁：`data_schema`错误强校验Schema/数据角色/目标列/特征签名；`dependency`错误比较Python与关键依赖主次版本；`resource`错误比较执行后端与平台；`artifact_contract`错误比较产物合同版本。明确不匹配的记忆在向量排序前直接剔除。
- 旧数据兼容：历史Memory如缺少新合同字段，不伪造匹配，标记`legacy_unscoped/missing_memory_fields`并降低适用性分；只有“双方字段都存在且冲突”时才强拒绝，避免升级后无声丢失全部旧经验。
- 检索时机：启动时做运行合同预检索，DataAgent完成后携完整数据合同再过滤，Operation失败后用错误文本+两类合同做最终重排；Candidate/Model不会消费已被拒绝的修复。
- 审计与前端：`repair_memory_retrieval`新增适用性候选数、拒绝数、被拒绝memory ID和逐字段mismatch；随Manager行为与下轮Agent输入摘要直接展示。
- 真实产物冒烟：现有67特征报告生成Schema=`generated_yield_process_202607`、target=`yield_stress`、feature signature=`1815011fef45b24b28887b5d`；当前环境识别为Python 3.11、NumPy 1.26、Pandas 2.1、scikit-learn 1.4、Darwin/x86_64。
- 回归：新增跨Schema/数据角色/目标列/特征集拒绝和Python/依赖主次版本兼容性测试；最终全量回归`139 passed, 265 warnings`。

### 阶段 46：Repair Memory 过期与重新确认闭环

- 状态：已完成。已验证Repair Memory默认有效期为90天，以`last_verified_at -> verified_at -> updated_at`的优先级计算年龄；超期记忆仍保留在数据库用于审计，但不进入排序和Agent Context。
- 旧数据兼容：早期记忆没有`verified_at`时，以`updated_at_legacy_fallback`估算，并在freshness审计中显式标记时间来源，不默认它永久有效。
- 自动重新确认：只有记忆在当前Operation错误后被相似度+适用性门禁选中，工作流实际进入后续修订轮，新Operation返回`rcode=0`，且Manager review接受时，才刷新验证时间。仅预加载记忆或未经修订就成功，不自动续期。
- 显式重新确认：新增`POST /api/memories/{memory_id}/reconfirm`，要求提供已存在且`status=passed`的`evidence_run_id`、reviewer和note；证据必须来自领域StateGraph，且workspace与execution mode与Memory兼容。运行中、失败、跨workspace或非领域run会返400，未知run返404。
- 幂等和追溯：`verification_history`记录run ID、时间、来源、Manager决策、成功结果和note，最多保留20条；同一evidence run重复回调不增加`verification_count`。
- 前端行为：自动重新确认会产生Agent Manager的`reconfirm_verified_repair_memory`实时行为，包含memory IDs、evidence run和修订轮次；不需要用户打开额外面板。
- 阶段性验证：过期判定完成后Repair Memory测试`8 passed`；显式重新确认与API完成后`10 passed`；LangGraph自动续期与前端事件`13 passed`；加入跨workspace/非领域证据拒绝后定向复跑`23 passed`；最终全量回归`142 passed, 265 warnings`。

### 阶段 47：Repair Memory 固定评测集与安全拒答基线

- 状态：已完成第一版可重复评测。评测集包含8条已验证Memory fixture和13条query，其中8条正向排序样例、5条workspace/执行模式/数据契约/依赖版本/过期负向样例。
- 标签边界：当前v1全部是人工构造的契约回归样例，不是生产失败抽样。所有`reuse_outcome=not_observed`，因此报告强制`repair_success.rate=null`、`claimable=false`，不将检索命中率冒充修复成功率。
- 真实代码路径：每条fixture都经过`record_candidate -> confirm -> retrieve`，使用临时SQLite和固定时钟，不直接伪造confirmed行，不污染开发数据库；离线确定性hashing embedding，`real_llm_calls=0`。
- 首轮基线：Top-1/Recall@K/MRR均为`1.0000`，误导命中率`0.0000`，但安全拒答率仅`0.4000`。根因是目标记忆被适用性或过期门禁拦截后，排序器又从其他错误类别中补入无关记忆。
- 修正：Operation出现真实错误后，在embedding排序前增加`error_category`硬门禁；dependency错误再区分NumPy/scikit-learn/Pandas/XGBoost/LightGBM/Torch依赖包族。因此NaN错误不再返回sklearn API修复，NumPy alias错误不再返回sklearn修复。任务启动/数据画像阶段尚无具体错误，继续宽召回，避免用用户需求文本误判错误类别。拒绝原因以`error_category_mismatch/dependency_family_mismatch`进入审计。
- 修正后严格评测：Top-1=`1.0000`、Recall@K=`1.0000`、MRR=`1.0000`、安全拒答率=`1.0000`、护栏拒绝原因准确率=`1.0000`、误导命中率=`0.0000`，全部预设阈值通过。最新报告：`agent_workspace/yieldmind/repair_memory_evals/repair_memory_eval_20260921_075103.json`。
- 阶段性验证：Schema/引用合法性`2 passed`；离线执行器接入后`3 passed`；错误类别和依赖包族门禁后Repair Memory定向回归`14 passed`；修复“任务预加载被过度门控”回归后`15 passed`；最终全量回归`146 passed, 265 warnings`。

### 阶段 48：Repair Memory 真实复用结果采集

- 状态：已完成采集链路，后续真实任务会自动累积样本。新增`yieldmind_repair_memory_reuse_attempts`表及SQLite/PostgreSQL迁移，每条记录保留memory/run、workspace、execution mode、命中轮次、应用轮次、Operation rcode、Manager决策和有界结果摘要。
- 因果边界：任务预加载只说明“看过记忆”，不进入成功率分母。只有Operation真实报错、严格匹配已验证Memory、Manager确实进入下一轮revision时，才创建`applied`尝试；下一轮Operation和Manager review后收敛为`succeeded/failed`。
- 不污染分母：匹配但预算耗尽、没有进入revision的记忆不建尝试；已应用但尚未review的记录保持open，不进入成功率。终态幂等，失败记录不能被后续重复调用改写为成功。
- 跨run去重：阶段检查发现同一错误会在不同run生成重复confirmed Memory，从而让一次revision重复计数。已按`workspace + execution_mode + normalized error fingerprint`阻止重复建档，原记忆通过reconfirm累积证据。
- 统计口径：整体成功率按`run_id + applied_round`的revision episode聚合，不会因同一轮命中多条Memory而放大分母；另保留逐Memory成功/失败统计。有1条以上观测时可计算，但少于30个episode只标记`preliminary`，不作正式效果声称。
- 查询与导出：新增`GET /api/repair-memory/reuse-summary`，可按workspace查询摘要和可选尝试明细；`scripts/export_yieldmind_repair_memory_outcomes.py`生成不调用模型的JSON报告。
- 当前真实库结果：升级后尚新增0个可观测复用episode，因此`success_rate=null`、`computable=false`、`claimable=false`、status=`insufficient_evidence`。测试产生的成功/失败样例使用临时数据库，没有污染真实统计。
- 阶段性验证：表、幂等和终态单测`12 passed`；真实工作流成功/失败各一次与跨run去重`14 passed`；API/导出/数据库生命周期`15 passed`；13条固定检索评测继续全通过；最终全量回归`148 passed, 265 warnings`。

### 阶段 49：Agent Loop 受控故障恢复 A/B 评测

- 状态：已完成收口评测，不新增Agent或改变主架构。新增30条成对任务：5条无故障正常对照、20条可恢复故障、5条不可恢复负对照。
- 可恢复故障：基线评估、候选benchmark、产物校验和报告生成首次失败应路由`local_repair`；数据生成和数据画像首次失败应路由`replan`。不可恢复对照为强制证据不可用，应安全失败而不是绕过门禁。
- 成对协议：每条case的数据、随机种子、真实本地工具和完成判定完全相同；唯一差别是基线组`max_local_repairs=max_replans=0`，实验组各允许1次。故障注入只替换声明工具的第一次返回，恢复后必须真实重跑数据、模型、产物和报告链路。
- 完成判定：同时要求最终`status=passed`、finish节点通过、生成`report_json`且errors为空，不只看某一个节点返回200。
- A/B结果：关闭恢复时完成`5/30=16.7%`；开启有界Agent Loop后完成`25/30=83.3%`，提升`66.7`个百分点。20/20条可恢复故障恢复成功，路由准确率`100%`；5条不可恢复负对照均继续失败，没有伪造成功。
- 代价：平均运行时间由`1.325s`增加到`2.305s`；实验组平均局部修复`0.533`次、重规划`0.133`次。这是恢复能力换取的可量化开销。
- 口径边界：这是“真实本地工具+受控故障注入”的LangGraph恢复评测，`real_llm_calls=0`，不代表真实LLM代码生成或生产流量完成率。简历必须保留“30条受控故障A/B”限定语。
- 分层评测总量：核心Agent/工具/工作流37条+检索30条+语料路由32条+Repair Memory 13条+故障恢复30条，共`142`条。单元/集成工程回归另计，不混入评测集数量。
- 阶段性验证：30条Schema、路由分布和故障次数单测`3 passed`；60次真实本地工作流A/B报告status=`passed`；最终全量回归`151 passed, 265 warnings`。

### 阶段 50：项目收口与演示验收

- 状态：已完成；本阶段不再增加Agent或主链路功能，只统一简历口径、证据位置、复现命令和演示验收记录。
- 面试口径：新增`YieldMind_Interview_Claims.md`，明确区分LangGraph故障恢复循环、领域多Agent修订循环和Pydantic工具参数自修正；`142条`、`16.7% -> 83.3%`、`20/20`和`151 passed`均标出分母、证据与不可外推边界。
- 服务健康：真实HTTP访问根页面和静态JS均返回200；`/health/dependencies`确认SQLite、Redis和本地确定性知识库全部`ok=true`。Repair Memory真实复用仍为0个episode，成功率保持`null`，没有用测试数据填充分母。
- 演示运行：通过`POST /api/workflows/offline`完成`run_eba1fca98c1b`；后端为`langgraph_stategraph`，`start -> prepare_data -> profile -> retrieve_evidence -> evaluate -> candidate_benchmark -> verify_artifacts -> report -> finish`共9个阶段全部`passed`，errors为空。
- 演示产物：报告写入`agent_workspace/yieldmind/reports/yieldmind_report_1789990811.json`；该20样本、2折演示只证明端到端工程链路可运行，不替代200条增强开发集的统一模型指标，也不调用LLM。
- 最终边界：后续只接受缺陷修复、真实任务数据积累和人工标注结果回填；没有新证据时不再扩充简历数字或宣称真实LLM修复率。

## 本轮验证结果

验证环境：`/opt/anaconda3/envs/amla/bin/python`

| 命令 | 结果 |
| --- | --- |
| `python -m py_compile yieldmind/*.py scripts/*.py tests/*.py` | 通过 |
| `python scripts/init_yieldmind_db.py` | 通过，初始化 `agent_workspace/yieldmind/yieldmind.sqlite3` |
| `python -m pytest -q` | 最终全量复跑`151 passed, 265 warnings`；覆盖结构化Manager路由、逐Agent行为与领域流式事件、Session Context主链路、ToolExecutor/ToolPolicy与受控参数自修正、工具结果有界回传与幂等选择性重试、SearchAgent知识库主链路、数据血缘与Anchor兼容门禁、多保真严格分组隔离、带相似度、适用性、过期、重新确认、错误类别/依赖包族安全拒答、固定评测与真实复用结果采集的跨任务Repair Memory、受控故障A/B、Operation大结果回传、多格式文献、评审裁决、跨分块池化盲审与配置防错、语料路由词边界与显式过滤、工作台上传/流式事件和固定Gold重排。warning 来自 joblib CPU core探测、sklearn GPR收敛提示和Chroma/Pydantic deprecation，不影响结果 |
| `python scripts/run_yieldmind_recovery_eval.py` | 通过；30条成对受控故障任务上，关闭恢复`5/30=16.7%`，开启有界恢复`25/30=83.3%`，提升`66.7`个百分点；可恢复故障恢复率和路由准确率均为`100%`，`real_llm_calls=0` |
| 最终真实HTTP演示验收 | `run_eba1fca98c1b`通过；`langgraph_stategraph`的9个阶段全部`passed`，SQLite/Redis/knowledge依赖全部健康，生成JSON/Markdown报告；该演示`real_llm_calls=0` |
| `python scripts/run_yieldmind_repair_memory_eval.py` | 通过；13条固定契约case上Top-1/Recall@K/MRR/安全拒答率/护栏拒绝原因准确率均为`1.0000`，误导命中率`0.0000`，`real_llm_calls=0`；无真实复用outcome，因此修复成功率为`null` |
| `python scripts/export_yieldmind_repair_memory_outcomes.py` | 通过；当前真实库0个已观测revision episode，`success_rate=null`、status=`insufficient_evidence`、`real_llm_calls=0`；报告不混入临时测试数据 |
| `python -m pytest -q tests/test_domain_workflow.py tests/test_process_control.py` | `16 passed`；覆盖Manager合法路由、review回环预算、Operation取消/超时/进程组清理、运行时细节事件与8 MB大结果队列回传 |
| `run_yieldmind_retrieval_eval.py --split-strategy recursive_chars_v3 --chunk-size 700 --chunk-overlap 80` | 通过；同参数v3 hybrid Recall@5=`0.9000`、MRR=`0.8361`、Top-1=`0.8000`，39 chunks |
| `run_yieldmind_retrieval_eval.py --split-strategy section_aware_v4`（hashing分块A/B） | `450/60`、`700/80`、`900/120`的hybrid Recall@5均为`0.9000`；MRR分别为`0.8361/0.8417/0.8417`，Top-1均为`0.8000`。报告保存于`/private/tmp/yieldmind_chunk_eval_*` |
| `prepare_yieldmind_chunking_review.py` | 通过；30 case的300个Top-5系统映射池化为168条盲审候选，132条为两系统共享；双人CSV隐藏系统与排名，标签目前为空 |
| `prepare_yieldmind_chunking_review.py --candidate-scope top1_union` + 第一位人工评审 | 通过；第一位评审32/32行完成且候选哈希校验通过；两套分块Top-1直接相关率均为`60.0%`、至少部分相关率均为`83.3%`，`0胜/0负/30平`。这是单评审Top-1快速筛选，不是最终Gold或Top-5质量评测 |
| `run_yieldmind_routing_eval.py --verify-retrieval` | 通过；32/32静态路由正确，32/32持久化索引检索成功，corpus纯度`1.0000`，无强意图强制路由率`0` |
| `python scripts/ingest_yieldmind_corpora.py`（local hashing） | 通过；实际重建7篇项目文档/39 chunks和8篇文献/326 chunks为`section_aware_v4`，corpus隔离及DOI/URL/页码检查全true |
| `FastAPI TestClient /api/workflows/offline/stream`（默认200行CSV） | 通过，`run_316bb843f863`返回`passed`、18条节点事件和9个阶段；最终推荐Ridge基线，OOF RMSE=`0.3193`、R2=`0.9928`，报告保存200条获胜预测点 |
| FastAPI真实HTTP前端契约与20样本流式运行 | 根页面/JS/CSS/health/默认数据均返回200；`run_d59daa7b12c7`返回18条真实Agent事件、7个角色、9个阶段及最终`passed`，服务验证后已正常停止 |
| `python scripts/run_yieldmind_eval.py` | 通过，`37/37` case passed；`real_llm_calls=0`，`simulated_model_calls=0` |
| `python scripts/run_yieldmind_retrieval_eval.py` | 通过，30条case分别完成hashing vector、BM25+和RRF hybrid；结果如阶段16，`real_llm_calls=0`、`simulated_model_calls=0` |
| `python scripts/run_yieldmind_retrieval_eval.py --preset qwen3-embedding-0.6b --embedding-endpoint http://127.0.0.1:8091` | 通过，真实Qwen3 CPU embedding完成30条case；hybrid Recall@5=`1.0000`、MRR=`0.8639`，服务峰值RSS约`3.90GB`，`real_llm_calls=0` |
| 修正BM25Plus零重叠候选后重跑同一Qwen命令 | 通过；hybrid Recall@5=`1.0000`、MRR=`0.8861`、Top-1=`0.8000`，真实67次encode/80条文本，`real_llm_calls=0` |
| `python scripts/run_yieldmind_qwen_api_smoke.py --embedding-endpoint http://127.0.0.1:8091` | 通过，FastAPI readiness、Qwen入库、直接搜索和Tool搜索9项检查全true；真实9次encode/22条文本，`real_llm_calls=0` |
| `python scripts/check_yieldmind_qwen_api_http.py --api-base-url http://127.0.0.1:8072 --embedding-endpoint http://127.0.0.1:8091` | 通过，独立Uvicorn/TCP正常场景13/13及Qwen不可达场景7/7检查通过；正向真实9次encode/22条文本，反向readiness=503，`real_llm_calls=0` |
| `python scripts/check_yieldmind_embedding_capability.py` | 通过；离线确认Qwen3/BGE-M3均未达到本机零变更运行条件，`network_calls=0`、`model_downloads=0`、`model_inference_calls=0` |
| `python scripts/ingest_yieldmind_corpora.py`（Qwen3 profile） | 通过；PostgreSQL+Chroma实存7篇项目文档/39 chunks和8篇文献/326 chunks，幂等复跑`document_encode_calls=0`、两次query embedding，语料隔离和DOI/URL/页码检查均通过，`real_llm_calls=0` |
| `.venv-qwen3/bin/python scripts/run_yieldmind_reranker_eval.py --scorer qwen3-embedding-cosine --device cpu --batch-size 16` | 通过；固定人工Gold的Qwen embedding余弦重排使MRR `0.8861→0.8389`、Top-1 `0.8333→0.7667`、nDCG `0.9409→0.8921`，结论为不启用；该结果不是cross-encoder指标 |
| `python scripts/run_yieldmind_function_calling_smoke.py` | 通过，生成 skipped 报告；`real_llm_calls=0`，未调用真实模型 |
| `python scripts/run_yieldmind_workflow.py --n-samples 40 --n-splits 2` | 通过，`status=passed`、`workflow_backend=langgraph_stategraph`、`langgraph_available=true` |
| `pip check` | 通过，`No broken requirements found` |
| `FastAPI TestClient /api/safety/redact + /api/budget/plan` | 通过，脱敏结果不含原始 secret/email，budget endpoint 返回可用 token |
| `POST /api/agent/plan` | 通过，返回 `mode=offline_rule_planner`、`llm_calls=0` |
| `POST /api/workflows/offline` | 通过，返回 `passed langgraph_stategraph True` |
| `FastAPI TestClient /health + /api/tools` | 通过，健康检查 `ok=True`，当前工具数 `14` |
| `scripts/run_yieldmind_api.py --port 8070` + `curl /health` | 通过；服务验证后已停止，本轮未保留常驻进程 |
| `scripts/check_yieldmind_docker_sandbox.py --allow-docker --image yieldmind-sandbox:local` | 通过，`container_runs=1`，容器输出 `network-check=blocked`；使用本机缓存 Python 镜像临时标签 |
| `scripts/run_yieldmind_postgres_redis_smoke.py` | 通过，PostgreSQL/Redis/LangGraph及hybrid检索11项真实检查全部为true，持久checkpoint数为10，同幂等键Tool只落一条记录 |
| `scripts/run_yieldmind_queue_smoke.py` | 通过，Redis发布、幂等、Celery消费、PostgreSQL终态、关联 run 和 Celery SUCCESS 全部为 true |
| `scripts/run_yieldmind_running_cancel_smoke.py` | 通过，真实运行中任务在节点边界取消，task/run=`cancelled`、Celery=`REVOKED`、十项检查全 true |
| `scripts/run_yieldmind_redis_outage_smoke.py` | 通过，真实Redis下线时PostgreSQL取消及审计仍成功，Celery状态明确降级为 `UNKNOWN`；恢复后依赖复验通过 |
| `scripts/run_yieldmind_recovery_smoke.py prepare/recover` | 通过，真实worker冷停、租约过期、父任务中断、新任务/新run恢复及父链11项检查全true |
| `scripts/run_yieldmind_domain_queue_smoke.py` | 通过，真实PostgreSQL/Redis/Celery九项检查全true；Celery=`SUCCESS`，未授权domain task/run按设计=`failed`，轨迹`start -> finish`且`real_llm_calls=0` |

离线评测报告：

```text
agent_workspace/yieldmind/evals/yieldmind_offline_eval_20260918_082744.json
agent_workspace/yieldmind/evals/yieldmind_offline_eval_20260918_093727.json
agent_workspace/yieldmind/retrieval_evals/retrieval_eval_20260918_094010.json
agent_workspace/yieldmind/function_calling/yieldmind_function_calling_smoke_20260917_091403.json
agent_workspace/yieldmind/docker_sandbox/yieldmind_docker_sandbox_20260917_110623.json
agent_workspace/yieldmind/integrations/postgres_redis_smoke_20260918_073513.json
agent_workspace/yieldmind/integrations/postgres_redis_smoke_20260918_094729.json
agent_workspace/yieldmind/integrations/postgres_redis_celery_smoke_20260917_114602.json
agent_workspace/yieldmind/integrations/postgres_redis_cancel_20260918_064515.json
agent_workspace/yieldmind/integrations/postgres_redis_running_cancel_20260918_070821.json
agent_workspace/yieldmind/integrations/postgres_redis_outage_20260918_071111.json
agent_workspace/yieldmind/integrations/postgres_redis_recovery_20260918_072715.json
agent_workspace/yieldmind/integrations/postgres_redis_domain_queue_20260918_085450.json
agent_workspace/yieldmind/qwen_api_http/20260918_113601/qwen_api_http_unready.json
agent_workspace/yieldmind/qwen_api_http/20260918_113809/qwen_api_http_ready.json
agent_workspace/yieldmind/repair_memory_evals/repair_memory_eval_20260921_075103.json
agent_workspace/yieldmind/repair_memory_outcomes/repair_memory_outcomes_20260921_075058.json
agent_workspace/yieldmind/recovery_evals/recovery_eval_20260921_130839.json
```

评测 case：

- `generate_demo_data`：真实调用现有合成数据生成器。
- `profile_demo_data`：真实调用 schema adapter 读取 CSV。
- `baseline_demo_data`：真实运行 sklearn 固定基线评估。
- `candidate_benchmark_demo_data`：真实运行原 CandidateBenchmark proxy 评估与 selection audit。
- `sandbox_python_echo`：真实通过受控 Python 子进程执行。
- `docker_sandbox_requires_authorization`：真实验证未授权时不会启动容器；不是容器执行成功率。
- `missing_data_rejected`：真实校验缺失路径并按预期拒绝。
- `planner_cases`：12 条本地规则工具决策一致性评测，非 LLM 结果。
- `offline_workflow_demo_baseline`：真实执行工作流；当前环境已验证为 `langgraph_stategraph`，并产出 baseline、candidate benchmark、selection audit、artifact verification 和 report artifacts。
- `knowledge_ingest_search`：默认离线套件保留真实Chroma入库、检索与6条小样例回归；独立检索套件另含30条标注查询及三种基线。
- `safety_evidence_budget`、`redaction_cases`、`budget_cases`：真实验证脱敏、Token Budget 选择、有效/无效证据引用校验；不调用 LLM。
- `session_memory_constraints`：真实验证约束继承、幂等 turn、创建后续 run 与记忆删除。

## 运行命令

推荐使用项目已有可用环境：

```bash
/opt/anaconda3/envs/amla/bin/python scripts/init_yieldmind_db.py
/opt/anaconda3/envs/amla/bin/python scripts/run_yieldmind_eval.py
/opt/anaconda3/envs/amla/bin/python -m pytest tests/test_yieldmind_core.py -q
/opt/anaconda3/envs/amla/bin/python scripts/run_yieldmind_api.py --port 8070
```

服务启动后访问：

```text
http://127.0.0.1:8070
http://127.0.0.1:8070/health
http://127.0.0.1:8070/api/tools
```

## 待办

- 数据角色与Anchor兼容门禁已接入；下一个数据依赖项是获得与200条增强工艺数据同材料体系、同67特征口径且从未参与生成/训练的真实批次。到位后按批次或时间留出重跑，将OOF开发指标与业务Holdout指标分开固化。
- 完成两份分块池化盲审CSV的独立人工标注，运行一致性分析和分歧裁决；以最终Gold比较同参数v3/v4。若v4的人工指标仍有优势，再执行Qwen3 embedding同参数复验。
- 领域SearchAgent主链路已接入配置化知识库；当前本机API使用`local_hashing_v1`离线profile。后续在Qwen3 embedding HTTP服务就绪时，需再跑一次真实领域端到端，对比SearchAgent延迟、证据变化和最终候选方案；不沿用已被否决的embedding余弦重排。
- Repair Memory的验证、相似度排序、失败后重排、数据Schema/依赖/执行合同适用性门禁、过期/重新确认、13条合成契约评测和真实复用结果自动采集已完成；后续是在真实运行中累积至少30个独立revision episode，再报告可对外声称的修复成功率。人工拒绝机制已由memory review API支持。
- 已完成`yieldmind.domain_workflow`的显式授权真实端到端验收；首次运行暴露的`multiprocessing.Queue`大结果回传死锁已修复，8 MB回归与第二次真实运行均通过，Operation不再被误判600秒超时。
- 在已完成双人标注和24条争议裁决的30条项目内查询基础上，继续增加真实用户查询和外部文献查询标签，避免以当前小规模自建集替代生产效果。
- 对真实 LLM Function Calling 做一次显式 `allow_live_llm=true` 冒烟测试，并记录模型、时间和结果；默认评测仍保持零 LLM 调用。
- Qwen3 embedding余弦重排已在固定Gold上实测并因指标退化而否决；待固定revision的Qwen3-Reranker模型可用后再执行正式cross-encoder对照，同时继续扩充真实查询。FastAPI静态实验工作台已完成第一轮可视化重构，Next.js仍未引入。
- Docker/no-network 运行时已真实通过；官方 `python:3.11-slim` 可复现镜像构建因 Docker Hub token 请求超时未完成，本轮运行验证使用本机缓存 Python 3.8 slim镜像的临时本地标签。
- LangGraph升级层已有条件边、PostgreSQL持久检查点、Tool持久幂等、节点边界取消、Operation进程组终止和任务级人工恢复；自动恢复、普通同步Tool节点内取消和模糊`running` Tool通用处置仍未完成。
