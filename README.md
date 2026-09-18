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
prediction. The original domain methods remain on `YieldAgentManager`, while
`yieldmind.domain_workflow` schedules them as explicit LangGraph nodes:
DataAgent, SearchAgent, CandidateAgent, ModelAgent, pre-execution review,
OperationAgent, and post-execution review. A failed review follows a bounded
revision edge back to CandidateAgent/ModelAgent instead of hiding the loop in
one graph node.

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

YieldMind upgrade layer:

```bash
/opt/anaconda3/envs/amla/bin/python scripts/init_yieldmind_db.py
/opt/anaconda3/envs/amla/bin/python scripts/run_yieldmind_eval.py
/opt/anaconda3/envs/amla/bin/python scripts/run_yieldmind_retrieval_eval.py
/opt/anaconda3/envs/amla/bin/python scripts/run_yieldmind_function_calling_smoke.py
/opt/anaconda3/envs/amla/bin/python scripts/run_yieldmind_api.py --port 8070
```

The live domain StateGraph is deliberately opt-in because CandidateAgent and
ModelAgent call the configured model API:

```bash
/opt/anaconda3/envs/amla/bin/python scripts/run_yieldmind_domain_workflow.py \
  --synthetic-data --no-external-search --no-require-search-results \
  --allow-live-llm
```

This additive layer provides FastAPI, PostgreSQL production persistence with a
SQLite offline-test mode, a Pydantic Tool Registry, bounded Python process-group
execution, real LangGraph StateGraphs, a
function-calling adapter, a Chroma-backed domain knowledge base, session/memory
primitives, safety/evidence guardrails, a data-driven offline evaluation suite,
and a small web page at
`http://127.0.0.1:8070`. The default evaluation is deterministic and records
`real_llm_calls=0`; running live LLM function calling or the original LLM
pipeline requires the explicit `allow_live_llm=true` flag.

The knowledge path supports version-isolated Chroma vector indexes, BM25+ and
reciprocal-rank-fusion hybrid retrieval. The default vector profile is
deterministic hashing for offline regression only; it is not presented as a
semantic embedding result. The 30-query benchmark includes Chinese cross-language
queries and reports Recall@5, MRR, citation accuracy at one, and latency:

```bash
/opt/anaconda3/envs/amla/bin/python scripts/run_yieldmind_retrieval_eval.py
/opt/anaconda3/envs/amla/bin/python scripts/check_yieldmind_embedding_capability.py
```

An optional sentence-transformers profile refuses model downloads unless
`--allow-model-download` is supplied. Model ID, revision, dimension,
normalization, metric, instructions, and index version are part of the profile;
the revision must be immutable rather than the floating `main` branch, and
different profiles use different Chroma collections. The shared environment
keeps `transformers==4.40.2`; Qwen3 runs behind the isolated loopback service
described below.

Live Function Calling validates every model-selected tool and argument object
against the Tool Registry before execution. Tool results are redacted, returned
to the model with their `tool_call_id`, and followed by a second constrained
answer call. Injected protocol-test clients are recorded as simulated calls,
never as real model calls.

Qwen3 Embedding runs in a separate Python environment because its required
`tokenizers>=0.21` conflicts with Chroma 0.5.23 in the main environment. The
model process only serves normalized vectors on loopback; Chroma and database
access remain in the main application:

```bash
bash scripts/setup_yieldmind_qwen3_env.sh
YIELDMIND_EMBEDDING_TOKEN=local-only-token \
  .venv-qwen3/bin/python scripts/run_yieldmind_embedding_server.py \
  --hf-home agent_workspace/yieldmind/hf_cache --port 8091

YIELDMIND_EMBEDDING_TOKEN=local-only-token \
  /opt/anaconda3/envs/amla/bin/python scripts/run_yieldmind_retrieval_eval.py \
  --preset qwen3-embedding-0.6b --embedding-endpoint http://127.0.0.1:8091
```

The preset pins the model to an immutable Hugging Face commit. Add
`--allow-model-download` to the server only for the initial controlled download.
On the current CPU-only host, the 30-query real-inference run produced the
following results (Recall@5 / MRR / citation accuracy@1 / mean query latency):

| Retrieval | Metrics |
| --- | --- |
| Qwen3 vector | `0.9667 / 0.7789 / 0.6667 / 378.6 ms` |
| BM25+ | `0.8000 / 0.8000 / 0.8000 / 2.4 ms` |
| Qwen3 + BM25+ RRF | `1.0000 / 0.8639 / 0.7667 / 431.3 ms` |

The model service peaked at about `3.90 GB` RSS; indexing 20 chunks took
`15.60s`, excluding model loading. These are results on a small, single-author,
repository-specific set, not production-quality evidence. Hashing remains the
deterministic offline default; Qwen3 hybrid retrieval is the measured quality
candidate while the benchmark is expanded and independently reviewed.

To exercise Qwen retrieval inside the actual deterministic StateGraph, start
the loopback embedding service and PostgreSQL/Redis, then run:

```bash
YIELDMIND_EMBEDDING_TOKEN=local-only-token \
YIELDMIND_DATABASE_URL='postgresql+psycopg://yieldmind:yieldmind_dev_only@127.0.0.1:5432/yieldmind' \
  /opt/anaconda3/envs/amla/bin/python scripts/run_yieldmind_qwen_workflow_smoke.py \
  --embedding-endpoint http://127.0.0.1:8091 --require-redis
```

This path adds an explicit `retrieve_evidence` node. It invokes the registered
`search_knowledge` tool, validates returned chunk identity/version metadata,
and writes structured evidence references into the final report. Qwen remains
opt-in: ordinary offline workflows do not require the model service.

## Active Modules

- `run_yield.py`: owns `YieldAgentManager`, the yield-specific manager that
  implements the original domain phases and algorithms used by both the legacy
  entry point and the domain StateGraph adapter.
- `yieldmind/domain_workflow.py`: explicitly schedules prepare/data/search,
  CandidateAgent, ModelAgent, OperationAgent, review, revision, cancellation,
  and finish nodes without storing clients, DataFrames, or processes in graph
  state.
- `yieldmind/process_control.py`: polls timeout/cancellation and terminates the
  complete POSIX process group with TERM/KILL escalation.
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
- `yieldmind/`: additive service, database, tool registry, sandbox, evaluation,
  LangGraph workflow, function-calling adapter, knowledge base, session/memory,
  safety/evidence checks, and display layer for the staged Agent upgrade plan.

## YieldMind Upgrade Verification

The additive YieldMind layer keeps the original domain algorithms intact. Its
offline regression suite makes no LLM calls and reports that boundary in every
evaluation artifact.

```bash
/opt/anaconda3/envs/amla/bin/python -m pytest tests/test_yieldmind_core.py -q
/opt/anaconda3/envs/amla/bin/python scripts/run_yieldmind_eval.py
/opt/anaconda3/envs/amla/bin/python scripts/run_yieldmind_workflow.py --n-samples 40 --n-splits 2
```

The Docker sandbox is opt-in, never pulls images automatically, and applies
`--network none`, a read-only root filesystem/repository mount, a dedicated
writable output mount, dropped capabilities, and CPU/memory/PID limits. Build
the local runtime explicitly, then run the one-shot smoke check:

```bash
docker build -f docker/yieldmind-sandbox.Dockerfile -t yieldmind-sandbox:local .
/opt/anaconda3/envs/amla/bin/python scripts/check_yieldmind_docker_sandbox.py \
  --allow-docker --image yieldmind-sandbox:local
```

Without `--allow-docker`, no container is started. A stopped/unavailable Docker
daemon or missing local image is reported as a failed capability check rather
than a successful sandbox run.

PostgreSQL is the production persistence backend and Redis is the Celery
broker/result backend. LangGraph uses `PostgresSaver` in this mode and the
database initialization command creates its checkpoint tables. SQLite remains
available for deterministic offline tests and uses an in-memory checkpointer.
Workflow tool calls use durable idempotency claims: completed calls can be
reused, while an ambiguous in-progress call is not replayed automatically.

```bash
docker compose -f compose.yieldmind.yml up -d --wait postgres redis
export YIELDMIND_DATABASE_URL='postgresql+psycopg://yieldmind:yieldmind_dev_only@127.0.0.1:5432/yieldmind'
export YIELDMIND_REDIS_URL='redis://:yieldmind_redis_dev_only@127.0.0.1:6379/0'
/opt/anaconda3/envs/amla/bin/python scripts/init_yieldmind_db.py
/opt/anaconda3/envs/amla/bin/python scripts/run_yieldmind_postgres_redis_smoke.py
```

Run `scripts/run_yieldmind_worker.py` in a separate process to consume the
`yieldmind` Celery queue. The checked-in credentials are local-development
defaults only; production deployments must inject different secrets.

The deterministic queue endpoint is `POST /api/tasks/workflows/offline`; the
original multi-agent StateGraph endpoint is `POST /api/tasks/workflows/domain`.
The latter rejects execution unless its request explicitly contains
`allow_live_llm=true`.

Queued and running tasks can be cancelled through
`POST /api/tasks/{task_id}/cancel`. PostgreSQL records cancellation time,
reason, and the caller-provided requester label before Celery receives a
best-effort revoke. Queued tasks stop immediately. Running workflows stop
cooperatively at the next LangGraph node boundary and persist a `cancel` stage;
the task and linked run then both become `cancelled`. For tasks submitted to
`POST /api/tasks/workflows/domain`, the OperationAgent node additionally polls
the same cancellation state while running and stops its process group, including
generated-code descendants. Other synchronous tools still cancel only at node
boundaries.

The real integration checks are available in
`scripts/run_yieldmind_running_cancel_smoke.py` and
`scripts/run_yieldmind_redis_outage_smoke.py`. The latter verifies that
PostgreSQL remains the task-status authority when Redis/Celery control is
unavailable; it does not claim Redis or PostgreSQL high availability.

Workers also hold a renewable database lease. Expired active tasks are listed
by `GET /api/task-recovery/stale`. Recovery through
`POST /api/tasks/{task_id}/recover` requires an explicit confirmation that the
old worker has stopped, marks the old task/run `interrupted`, and creates a new
linked task/run. It never moves an old terminal task back to `running` and does
not claim to resume an optimizer or subprocess from its internal state.

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
