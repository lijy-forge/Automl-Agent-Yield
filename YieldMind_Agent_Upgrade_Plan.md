# YieldMind Agent Upgrade Plan

更新时间：2026-09-18

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

- 状态：已完成初版。
- 产物：
  - `yieldmind/static/index.html`
  - `tests/test_yieldmind_core.py`
- 评测指标：工具调用是否成功、失败样本是否被拒绝、隔离执行是否通过、是否产生真实报告。
- 注意：这些是 Agent 工具层回归结果，不是最终研究模型效果。

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
  - 使用 SQLAlchemy 2.0 metadata 定义 schema，`psycopg` 作为驱动；任务租约升级后当前 Alembic revision 为 `20260918_0004`。
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

## 本轮验证结果

验证环境：`/opt/anaconda3/envs/amla/bin/python`

| 命令 | 结果 |
| --- | --- |
| `python -m py_compile yieldmind/*.py scripts/*.py tests/*.py` | 通过 |
| `python scripts/init_yieldmind_db.py` | 通过，初始化 `agent_workspace/yieldmind/yieldmind.sqlite3` |
| `python -m pytest -q` | 通过，`57 passed, 70 warnings`；新增检索profile隔离/BM25+、索引重建、30条数据集约束、HTTP embedding/API回环与代理隔离、StateGraph证据接入/无证据失败、服务端运行时选择及模拟Function Calling双轮协议测试；warning 来自 joblib CPU core探测、sklearn GPR收敛提示和Chroma/Pydantic deprecation，不影响结果 |
| `python scripts/run_yieldmind_eval.py` | 通过，`37/37` case passed；`real_llm_calls=0`，`simulated_model_calls=0` |
| `python scripts/run_yieldmind_retrieval_eval.py` | 通过，30条case分别完成hashing vector、BM25+和RRF hybrid；结果如阶段16，`real_llm_calls=0`、`simulated_model_calls=0` |
| `python scripts/run_yieldmind_retrieval_eval.py --preset qwen3-embedding-0.6b --embedding-endpoint http://127.0.0.1:8091` | 通过，真实Qwen3 CPU embedding完成30条case；hybrid Recall@5=`1.0000`、MRR=`0.8639`，服务峰值RSS约`3.90GB`，`real_llm_calls=0` |
| `python scripts/run_yieldmind_qwen_api_smoke.py --embedding-endpoint http://127.0.0.1:8091` | 通过，FastAPI readiness、Qwen入库、直接搜索和Tool搜索9项检查全true；真实9次encode/22条文本，`real_llm_calls=0` |
| `python scripts/check_yieldmind_qwen_api_http.py --api-base-url http://127.0.0.1:8072 --embedding-endpoint http://127.0.0.1:8091` | 通过，独立Uvicorn/TCP正常场景13/13及Qwen不可达场景7/7检查通过；正向真实9次encode/22条文本，反向readiness=503，`real_llm_calls=0` |
| `python scripts/check_yieldmind_embedding_capability.py` | 通过；离线确认Qwen3/BGE-M3均未达到本机零变更运行条件，`network_calls=0`、`model_downloads=0`、`model_inference_calls=0` |
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

- 对新`yieldmind.domain_workflow`执行一次显式授权的真实模型端到端验收，并记录模型、时间、各节点结果与失败原因；当前只完成真实领域方法接线和模拟路由回归。
- 在30条项目内查询基础上增加真实用户查询、双人标签复核和外部文献语料，避免以当前小规模自建集替代生产效果。
- 对真实 LLM Function Calling 做一次显式 `allow_live_llm=true` 冒烟测试，并记录模型、时间和结果；默认评测仍保持零 LLM 调用。
- Qwen3已在隔离环境完成同口径实测；下一步先扩充真实查询并双人复核标签，再决定是否切换默认embedding。只有现有失败case仍显示可修复空间时，再投入BGE-M3或Reranker对照；Next.js仍未完成且当前优先级较低。
- Docker/no-network 运行时已真实通过；官方 `python:3.11-slim` 可复现镜像构建因 Docker Hub token 请求超时未完成，本轮运行验证使用本机缓存 Python 3.8 slim镜像的临时本地标签。
- LangGraph升级层已有条件边、PostgreSQL持久检查点、Tool持久幂等、节点边界取消、Operation进程组终止和任务级人工恢复；自动恢复、普通同步Tool节点内取消和模糊`running` Tool通用处置仍未完成。
