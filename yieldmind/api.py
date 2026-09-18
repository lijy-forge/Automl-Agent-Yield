"""FastAPI backend for YieldMind runs, tools, events, and offline evals."""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from yieldmind.database import YieldMindStore
from yieldmind.domain_workflow import DomainWorkflowRequest, run_domain_workflow
from yieldmind.evaluation import run_offline_evaluation
from yieldmind.function_calling import ToolCallProtocolError, ToolPlanRequest, execute_plan, plan_tools
from yieldmind.knowledge_base import KnowledgeBase, KnowledgeIngestRequest, KnowledgeSearchRequest
from yieldmind.knowledge_runtime import (
    KnowledgeRuntimeConfig,
    configured_knowledge_base,
    knowledge_runtime_health,
)
from yieldmind.memory import (
    AddMessageRequest,
    CreateSessionRequest,
    DeleteMemoryRequest,
    MemoryVersionConflict,
    ReviewMemoryRequest,
    SessionMemoryStore,
    UpdateSessionSummaryRequest,
    UpsertMemoryRequest,
)
from yieldmind.safety import (
    BudgetRequest,
    EvidenceValidationRequest,
    RedactRequest,
    plan_token_budget,
    redact_payload,
    validate_evidence_refs,
)
from yieldmind.tools import OFFLINE_MODE, registry_for_workspace
from yieldmind.task_queue import (
    EnqueueWorkflowRequest,
    EnqueueDomainWorkflowRequest,
    CancelTaskRequest,
    RecoverTaskRequest,
    TaskCancellationConflict,
    TaskRecoveryConflict,
    cancel_queued_task,
    enqueue_workflow,
    enqueue_domain_workflow,
    get_task_status,
    recover_stale_task,
    redis_health,
)
from yieldmind.workflow import WorkflowRequest, run_workflow


PROJECT_ROOT = Path(__file__).resolve().parents[1]
STATIC_DIR = Path(__file__).resolve().parent / "static"


class CreateRunRequest(BaseModel):
    mode: str = OFFLINE_MODE
    source: str = "api"
    metadata: dict[str, Any] = Field(default_factory=dict)


class ToolCallRequest(BaseModel):
    args: dict[str, Any] = Field(default_factory=dict)
    run_id: str | None = None


app = FastAPI(
    title="YieldMind Agent API",
    version="0.1.0",
    description="Service/database/tool layer for the yield-stress AutoML agent.",
)
store = YieldMindStore()


def active_knowledge_base() -> KnowledgeBase:
    try:
        return configured_knowledge_base(store)
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Knowledge runtime unavailable: {type(exc).__name__}: {exc}") from exc


registry = registry_for_workspace(store=store, knowledge_base_factory=lambda: configured_knowledge_base(store))


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "ok": True,
        "service": "yieldmind",
        "mode_default": OFFLINE_MODE,
        "database_backend": store.backend,
        "database_location": store.location,
        "tool_count": len(registry.definitions()),
        "knowledge_profile": os.environ.get("YIELDMIND_EMBEDDING_PROFILE", "local_hashing"),
    }


@app.get("/health/dependencies")
def dependency_health() -> JSONResponse:
    database_ok = False
    database_error = ""
    try:
        database_ok = store.ping()
    except Exception as exc:
        database_error = f"{type(exc).__name__}: {exc}"
    redis_status = redis_health()
    knowledge_status = knowledge_runtime_health()
    payload = {
        "ok": database_ok and bool(redis_status.get("ok")) and bool(knowledge_status.get("ok")),
        "database": {"ok": database_ok, "backend": store.backend, "location": store.location, "error": database_error},
        "redis": redis_status,
        "knowledge": knowledge_status,
    }
    return JSONResponse(status_code=200 if payload["ok"] else 503, content=payload)


@app.post("/api/runs")
def create_run(request: CreateRunRequest) -> dict[str, Any]:
    run_id = store.create_run(mode=request.mode, source=request.source, status="created", metadata=request.metadata)
    store.add_event(run_id, stage="api", message="Run created.", payload=request.model_dump())
    return {"run_id": run_id, "run": store.get_run(run_id)}


@app.get("/api/runs")
def list_runs(limit: int = 50) -> dict[str, Any]:
    return {"runs": store.list_runs(limit=limit)}


@app.get("/api/runs/{run_id}")
def get_run(run_id: str) -> dict[str, Any]:
    run = store.get_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="run not found")
    return {"run": run}


@app.get("/api/runs/{run_id}/events")
def list_events(run_id: str, after_event_id: int = 0, limit: int = 200) -> dict[str, Any]:
    if not store.get_run(run_id):
        raise HTTPException(status_code=404, detail="run not found")
    return {"events": store.list_events(run_id, after_event_id=after_event_id, limit=limit)}


@app.get("/api/runs/{run_id}/stages")
def list_stage_executions(run_id: str, limit: int = 500) -> dict[str, Any]:
    if not store.get_run(run_id):
        raise HTTPException(status_code=404, detail="run not found")
    return {"stage_executions": store.list_stage_executions(run_id, limit=limit)}


@app.get("/api/runs/{run_id}/events/stream")
async def stream_events(run_id: str, after_event_id: int = 0) -> StreamingResponse:
    if not store.get_run(run_id):
        raise HTTPException(status_code=404, detail="run not found")

    async def generate():
        last_id = int(after_event_id)
        while True:
            events = store.list_events(run_id, after_event_id=last_id, limit=100)
            for event in events:
                last_id = int(event["event_id"])
                yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
            await asyncio.sleep(1.0)

    return StreamingResponse(generate(), media_type="text/event-stream")


@app.get("/api/tools")
def list_tools() -> dict[str, Any]:
    return {"tools": [definition.model_dump() for definition in registry.definitions()]}


@app.post("/api/tools/{tool_name}/call")
def call_tool(tool_name: str, request: ToolCallRequest) -> JSONResponse:
    if request.run_id and not store.get_run(request.run_id):
        raise HTTPException(status_code=404, detail="run not found")
    result = registry.execute(tool_name, request.args, run_id=request.run_id)
    if request.run_id:
        event_payload = redact_payload(RedactRequest(payload=result.model_dump())).payload
        store.add_event(
            request.run_id,
            stage="tool",
            level="info" if result.ok else "error",
            message=f"Tool {tool_name} {'passed' if result.ok else 'failed'}.",
            payload=event_payload,
        )
    status_code = 200 if result.ok else 400
    return JSONResponse(status_code=status_code, content=result.model_dump())


@app.post("/api/evals/offline")
def run_eval() -> dict[str, Any]:
    report = run_offline_evaluation(store=store, registry=registry, python_executable=sys.executable)
    return report


@app.post("/api/knowledge/ingest")
def ingest_knowledge(request: KnowledgeIngestRequest) -> dict[str, Any]:
    return active_knowledge_base().ingest(request)


@app.post("/api/knowledge/search")
def search_knowledge(request: KnowledgeSearchRequest) -> dict[str, Any]:
    return active_knowledge_base().search(request)


@app.get("/api/knowledge/documents")
def list_knowledge_documents(limit: int = 100) -> dict[str, Any]:
    return {"documents": active_knowledge_base().list_documents(limit=limit)}


@app.get("/api/knowledge/config")
def knowledge_config() -> dict[str, Any]:
    try:
        return KnowledgeRuntimeConfig.from_env().public_summary()
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Knowledge runtime misconfigured: {type(exc).__name__}: {exc}") from exc


@app.post("/api/safety/redact")
def redact_sensitive_payload(request: RedactRequest) -> dict[str, Any]:
    return redact_payload(request).model_dump()


@app.post("/api/budget/plan")
def plan_budget(request: BudgetRequest) -> dict[str, Any]:
    return plan_token_budget(request).model_dump()


@app.post("/api/evidence/validate")
def validate_evidence(request: EvidenceValidationRequest) -> dict[str, Any]:
    return validate_evidence_refs(store, request).model_dump()


@app.post("/api/sessions")
def create_session(request: CreateSessionRequest) -> dict[str, Any]:
    return {"session": SessionMemoryStore(store).create_session(request)}


@app.get("/api/sessions/{session_id}")
def get_session(session_id: str) -> dict[str, Any]:
    session = SessionMemoryStore(store).get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="session not found")
    return {"session": session}


@app.post("/api/sessions/{session_id}/messages")
def add_session_message(session_id: str, request: AddMessageRequest) -> dict[str, Any]:
    if request.session_id != session_id:
        raise HTTPException(status_code=400, detail="path session_id and body session_id differ")
    try:
        return SessionMemoryStore(store).add_message(request)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except MemoryVersionConflict as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@app.get("/api/sessions/{session_id}/messages")
def list_session_messages(session_id: str, limit: int = 50) -> dict[str, Any]:
    return {"messages": SessionMemoryStore(store).list_messages(session_id, limit=limit)}


@app.get("/api/sessions/{session_id}/context")
def get_session_context(session_id: str, max_tokens: int = 1200) -> dict[str, Any]:
    try:
        return SessionMemoryStore(store).build_context(session_id, max_tokens=max_tokens)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.put("/api/sessions/{session_id}/summary")
def update_session_summary(session_id: str, request: UpdateSessionSummaryRequest) -> dict[str, Any]:
    if request.session_id != session_id:
        raise HTTPException(status_code=400, detail="path session_id and body session_id differ")
    try:
        return {"session": SessionMemoryStore(store).update_summary(request)}
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/memories")
def upsert_memory(request: UpsertMemoryRequest) -> dict[str, Any]:
    try:
        return {"memory": SessionMemoryStore(store).upsert_memory(request)}
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/memories")
def search_memories(session_id: str = "", workspace_id: str = "default", limit: int = 20) -> dict[str, Any]:
    try:
        return {
            "memories": SessionMemoryStore(store).search_memories(
                session_id=session_id,
                workspace_id=workspace_id,
                limit=limit,
            )
        }
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post("/api/memories/{memory_id}/review")
def review_memory(memory_id: str, request: ReviewMemoryRequest) -> dict[str, Any]:
    if request.memory_id != memory_id:
        raise HTTPException(status_code=400, detail="path memory_id and body memory_id differ")
    try:
        return {"memory": SessionMemoryStore(store).review_memory(request)}
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.delete("/api/memories/{memory_id}")
def delete_memory(memory_id: str) -> dict[str, Any]:
    return SessionMemoryStore(store).delete_memory(DeleteMemoryRequest(memory_id=memory_id))


@app.post("/api/workflows/offline")
def run_offline_workflow(request: WorkflowRequest) -> dict[str, Any]:
    return run_workflow(request, store=store)


@app.post("/api/workflows/domain")
def run_live_domain_workflow(request: DomainWorkflowRequest) -> dict[str, Any]:
    return run_domain_workflow(request, store=store)


@app.post("/api/tasks/workflows/offline", status_code=202)
def enqueue_offline_workflow(request: EnqueueWorkflowRequest) -> dict[str, Any]:
    try:
        return enqueue_workflow(request, store=store)
    except ConnectionError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.post("/api/tasks/workflows/domain", status_code=202)
def enqueue_live_domain_workflow(request: EnqueueDomainWorkflowRequest) -> dict[str, Any]:
    try:
        return enqueue_domain_workflow(request, store=store)
    except ConnectionError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.get("/api/task-recovery/stale")
def stale_tasks(limit: int = 100) -> dict[str, Any]:
    return {"tasks": store.list_stale_tasks(limit=limit)}


@app.get("/api/tasks/{task_id}")
def task_status(task_id: str) -> dict[str, Any]:
    result = get_task_status(task_id, store=store)
    if not result:
        raise HTTPException(status_code=404, detail="task not found")
    return result


@app.post("/api/tasks/{task_id}/recover", status_code=202)
def recover_task(task_id: str, request: RecoverTaskRequest) -> dict[str, Any]:
    try:
        result = recover_stale_task(task_id, request, store=store)
    except TaskRecoveryConflict as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except ConnectionError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    if result is None:
        raise HTTPException(status_code=404, detail="task not found")
    return result


@app.post("/api/tasks/{task_id}/cancel")
def cancel_task(task_id: str, request: CancelTaskRequest) -> dict[str, Any]:
    try:
        result = cancel_queued_task(
            task_id,
            store=store,
            reason=request.reason,
            requested_by=request.requested_by,
        )
    except TaskCancellationConflict as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    if result is None:
        raise HTTPException(status_code=404, detail="task not found")
    return result


@app.post("/api/agent/plan")
def plan_agent_tools(request: ToolPlanRequest) -> dict[str, Any]:
    try:
        result = plan_tools(request, registry=registry)
    except PermissionError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except (ToolCallProtocolError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return result.model_dump()


@app.post("/api/agent/execute-plan")
def execute_agent_plan(request: ToolPlanRequest) -> dict[str, Any]:
    try:
        result = execute_plan(request, registry=registry, store=store)
    except PermissionError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except (ToolCallProtocolError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return result.model_dump()
