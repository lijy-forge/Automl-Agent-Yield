"""Pydantic tool registry for the YieldMind agent upgrade."""

from __future__ import annotations

import csv
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Literal

from pydantic import BaseModel, Field

from yieldmind.database import YieldMindStore, json_dumps
from yieldmind.knowledge_base import (
    KnowledgeBase,
    KnowledgeIngestRequest,
    KnowledgeSearchRequest,
)
from yieldmind.sandbox import (
    PROJECT_ROOT,
    DockerSandbox,
    DockerSandboxCommand,
    SandboxCommand,
    SubprocessSandbox,
)
from yieldmind.safety import (
    BudgetRequest,
    EvidenceValidationRequest,
    RedactRequest,
    plan_token_budget,
    redact_payload,
    validate_evidence_refs,
)


OFFLINE_MODE = "offline_deterministic"
LIVE_LLM_MODE = "live_llm_explicit"


def _schema(model: type[BaseModel]) -> dict[str, Any]:
    return model.model_json_schema()


def _jsonable(value: Any) -> Any:
    try:
        import numpy as np

        if isinstance(value, np.integer):
            return int(value)
        if isinstance(value, np.floating):
            number = float(value)
            return number if math.isfinite(number) else None
        if isinstance(value, np.ndarray):
            return value.tolist()
    except Exception:
        pass
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


class ToolResult(BaseModel):
    ok: bool
    mode: str = OFFLINE_MODE
    result: dict[str, Any] = Field(default_factory=dict)
    artifacts: dict[str, str] = Field(default_factory=dict)
    error: str = ""
    llm_calls: int = 0
    simulated_model_calls: int = 0


class ToolDefinition(BaseModel):
    name: str
    description: str
    risk_level: str
    input_schema: dict[str, Any]
    output_schema: dict[str, Any]
    requires_live_llm: bool = False


class GenerateDemoDataArgs(BaseModel):
    n_samples: int = Field(default=80, ge=20, le=5000)
    random_state: int = Field(default=42, ge=0)
    overwrite: bool = True


class ProfileYieldDataArgs(BaseModel):
    data_path: str


class RunFixedBaselineArgs(BaseModel):
    data_path: str
    n_splits: int = Field(default=5, ge=2, le=10)
    random_state: int = Field(default=42, ge=0)
    enable_sp_monotonic: bool = False


class RunCandidateBenchmarkArgs(BaseModel):
    data_path: str
    candidate_report_path: str = ""
    anchor_path: str = ""
    n_splits: int = Field(default=3, ge=2, le=10)
    random_state: int = Field(default=42, ge=0)
    max_rows: int = Field(default=600, ge=20, le=5000)


class VerifyArtifactsArgs(BaseModel):
    artifact_paths: dict[str, str] = Field(default_factory=dict)
    require_json: list[str] = Field(default_factory=list)


class ReportEvidenceRef(BaseModel):
    chunk_id: str
    text_hash: str
    index_version: str
    document_id: str = ""
    document_version: str = ""
    source_path: str = ""


class BuildRunReportArgs(BaseModel):
    title: str = "YieldMind Workflow Report"
    run_id: str = ""
    artifact_paths: dict[str, str] = Field(default_factory=dict)
    evidence_refs: list[ReportEvidenceRef] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


class SandboxPythonArgs(BaseModel):
    argv: list[str] = Field(default_factory=lambda: [sys.executable, "-c", "print('yieldmind sandbox ok')"])
    cwd: str = "."
    timeout_seconds: float = Field(default=30.0, ge=1.0, le=600.0)
    env: dict[str, str] = Field(default_factory=dict)


class DockerSandboxPythonArgs(BaseModel):
    argv: list[str] = Field(default_factory=lambda: ["python", "-c", "print('yieldmind docker sandbox ok')"])
    cwd: str = "."
    output_dir: str = "agent_workspace/yieldmind/docker_runs/default"
    image: str = "python:3.11-slim"
    timeout_seconds: float = Field(default=30.0, ge=1.0, le=600.0)
    memory: str = "512m"
    cpus: float = Field(default=1.0, gt=0.0, le=8.0)
    pids_limit: int = Field(default=128, ge=16, le=1024)
    allow_docker: bool = False


class SummarizeRunArtifactsArgs(BaseModel):
    run_dir: str
    prediction_preview_rows: int = Field(default=5, ge=0, le=50)


class IngestKnowledgeArgs(BaseModel):
    paths: list[str] = Field(..., min_length=1)
    index_version: str = "yieldmind-chroma-hashing-v1"
    chunk_size: int = Field(default=1200, ge=200, le=4000)
    chunk_overlap: int = Field(default=180, ge=0, le=1000)


class SearchKnowledgeArgs(BaseModel):
    query: str
    top_k: int = Field(default=5, ge=1, le=20)
    index_version: str = "yieldmind-chroma-hashing-v1"
    document_id: str | None = None
    retrieval_mode: Literal["vector", "bm25", "hybrid"] = "hybrid"


class ExistingPipelineArgs(BaseModel):
    prompt: str = "Build a yield-stress AutoML model."
    data_path: str
    test_path: str = ""
    run_dir: str = "agent_workspace/runs/yieldmind_live_llm_run"
    n_revise: int = Field(default=0, ge=0, le=5)
    operation_attempts: int = Field(default=1, ge=1, le=5)
    allow_live_llm: bool = False
    external_search: bool = False
    timeout_seconds: float = Field(default=900.0, ge=30.0, le=7200.0)


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    args_model: type[BaseModel]
    handler: Callable[[BaseModel], ToolResult]
    risk_level: str = "low"
    requires_live_llm: bool = False


class ToolRegistry:
    """Single source of truth for callable YieldMind tools."""

    def __init__(
        self,
        *,
        workspace_root: str | Path | None = None,
        store: YieldMindStore | None = None,
        knowledge_base: KnowledgeBase | None = None,
        knowledge_base_factory: Callable[[], KnowledgeBase] | None = None,
    ) -> None:
        self.workspace_root = Path(workspace_root or PROJECT_ROOT).resolve()
        self.store = store
        self.knowledge_base = knowledge_base
        self.knowledge_base_factory = knowledge_base_factory
        self._tools: dict[str, ToolSpec] = {}
        self._register_defaults()

    def _knowledge_base(self) -> KnowledgeBase:
        if self.knowledge_base is not None:
            return self.knowledge_base
        if self.knowledge_base_factory is not None:
            return self.knowledge_base_factory()
        return KnowledgeBase(store=self.store or YieldMindStore())

    def _register(self, spec: ToolSpec) -> None:
        self._tools[spec.name] = spec

    def _register_defaults(self) -> None:
        self._register(
            ToolSpec(
                name="generate_demo_yield_data",
                description="Create deterministic yield-stress demo data using the existing domain synthetic-data generator.",
                args_model=GenerateDemoDataArgs,
                handler=self._generate_demo_data,
            )
        )
        self._register(
            ToolSpec(
                name="profile_yield_data",
                description="Load a yield CSV through knowledge.yield_schema and return schema, feature, and target statistics.",
                args_model=ProfileYieldDataArgs,
                handler=self._profile_yield_data,
            )
        )
        self._register(
            ToolSpec(
                name="run_fixed_baseline_eval",
                description="Run the existing deterministic fixed-baseline evaluator and persist the measured report.",
                args_model=RunFixedBaselineArgs,
                handler=self._run_fixed_baseline_eval,
            )
        )
        self._register(
            ToolSpec(
                name="run_candidate_benchmark",
                description="Run the existing CandidateAgent proxy benchmark on concrete model/mechanism strategies.",
                args_model=RunCandidateBenchmarkArgs,
                handler=self._run_candidate_benchmark,
            )
        )
        self._register(
            ToolSpec(
                name="verify_artifacts",
                description="Verify that tool/workflow artifacts exist, are non-empty, and selected JSON artifacts are readable.",
                args_model=VerifyArtifactsArgs,
                handler=self._verify_artifacts,
            )
        )
        self._register(
            ToolSpec(
                name="build_run_report",
                description="Build a compact JSON and Markdown report from real workflow artifact paths.",
                args_model=BuildRunReportArgs,
                handler=self._build_run_report,
            )
        )
        self._register(
            ToolSpec(
                name="run_sandboxed_python",
                description="Run a Python subprocess with no shell, bounded timeout, repository-contained cwd, and minimal env.",
                args_model=SandboxPythonArgs,
                handler=self._run_sandboxed_python,
                risk_level="medium",
            )
        )
        self._register(
            ToolSpec(
                name="run_docker_sandboxed_python",
                description=(
                    "Run an explicitly authorized Python command in an allowlisted local Docker image with "
                    "network disabled, read-only root/repository mounts, a dedicated writable output mount, "
                    "and CPU/memory/PID limits. Images are never pulled automatically."
                ),
                args_model=DockerSandboxPythonArgs,
                handler=self._run_docker_sandboxed_python,
                risk_level="high",
            )
        )
        self._register(
            ToolSpec(
                name="summarize_run_artifacts",
                description="Read real artifacts from an existing run directory and return metric/prediction previews.",
                args_model=SummarizeRunArtifactsArgs,
                handler=self._summarize_run_artifacts,
            )
        )
        self._register(
            ToolSpec(
                name="ingest_knowledge_documents",
                description="Ingest supported domain Markdown/TXT documents into the Chroma-backed YieldMind knowledge base.",
                args_model=IngestKnowledgeArgs,
                handler=self._ingest_knowledge_documents,
                risk_level="medium",
            )
        )
        self._register(
            ToolSpec(
                name="search_knowledge",
                description="Search the domain knowledge base and return chunk IDs, sources, text, scores, and index version for evidence-grounded planning.",
                args_model=SearchKnowledgeArgs,
                handler=self._search_knowledge,
            )
        )
        self._register(
            ToolSpec(
                name="redact_sensitive_payload",
                description="Recursively redact API keys, bearer tokens, emails, phone numbers, database URLs, and sensitive field names before logging or model use.",
                args_model=RedactRequest,
                handler=self._redact_sensitive_payload,
            )
        )
        self._register(
            ToolSpec(
                name="plan_token_budget",
                description="Estimate and select prompt/context sections under an input token budget while preserving required constraints first.",
                args_model=BudgetRequest,
                handler=self._plan_token_budget,
            )
        )
        self._register(
            ToolSpec(
                name="validate_evidence_refs",
                description="Validate cited knowledge chunk IDs against the configured metadata database, including optional text_hash and index_version checks.",
                args_model=EvidenceValidationRequest,
                handler=self._validate_evidence_refs,
            )
        )
        self._register(
            ToolSpec(
                name="run_existing_yield_pipeline",
                description=(
                    "Run run_yield.py through the sandbox. This may call the configured live LLM only when "
                    "allow_live_llm=true; otherwise it refuses instead of simulating a model call."
                ),
                args_model=ExistingPipelineArgs,
                handler=self._run_existing_pipeline,
                risk_level="high",
                requires_live_llm=True,
            )
        )

    def definitions(self) -> list[ToolDefinition]:
        return [
            ToolDefinition(
                name=spec.name,
                description=spec.description,
                risk_level=spec.risk_level,
                input_schema=_schema(spec.args_model),
                output_schema=_schema(ToolResult),
                requires_live_llm=spec.requires_live_llm,
            )
            for spec in self._tools.values()
        ]

    def validate_call(self, name: str, args: dict[str, Any] | None = None) -> BaseModel:
        """Validate a proposed tool call without producing side effects."""
        if name not in self._tools:
            raise KeyError(f"Unknown tool: {name}")
        return self._tools[name].args_model.model_validate(args or {})

    def execute(
        self,
        name: str,
        args: dict[str, Any] | None = None,
        *,
        run_id: str | None = None,
        idempotency_key: str = "",
    ) -> ToolResult:
        parsed = self.validate_call(name, args)
        spec = self._tools[name]
        started = time.time()
        payload_args = redact_payload(RedactRequest(payload=parsed.model_dump())).payload
        claimed_call: dict[str, Any] | None = None
        if self.store is not None and idempotency_key:
            claimed_call, existed = self.store.claim_tool_call(
                tool_name=name,
                mode=OFFLINE_MODE,
                args=payload_args,
                idempotency_key=idempotency_key,
                run_id=run_id,
                started_at=started,
            )
            if existed:
                previous_result = claimed_call.get("result") or {}
                if claimed_call.get("status") in {"passed", "failed"} and previous_result:
                    return ToolResult.model_validate(previous_result)
                return ToolResult(
                    ok=False,
                    mode=str(claimed_call.get("mode") or OFFLINE_MODE),
                    error=(
                        "An earlier execution with this idempotency key is still running or has an "
                        "ambiguous outcome; refusing automatic replay."
                    ),
                )
        try:
            result = spec.handler(parsed)
        except Exception as exc:
            result = ToolResult(ok=False, mode=OFFLINE_MODE, error=f"{type(exc).__name__}: {exc}")

        status = "passed" if result.ok else "failed"
        payload_result = redact_payload(RedactRequest(payload=result.model_dump())).payload
        if self.store is not None:
            completed = time.time()
            if claimed_call is not None:
                self.store.complete_tool_call(
                    str(claimed_call["call_id"]),
                    status=status,
                    mode=result.mode,
                    result=payload_result,
                    error=result.error,
                    completed_at=completed,
                )
            else:
                self.store.record_tool_call(
                    tool_name=name,
                    mode=result.mode,
                    status=status,
                    args=payload_args,
                    result=payload_result,
                    error=result.error,
                    run_id=run_id,
                    started_at=started,
                    completed_at=completed,
                )
        return result

    def _generate_demo_data(self, args: BaseModel) -> ToolResult:
        parsed = args if isinstance(args, GenerateDemoDataArgs) else GenerateDemoDataArgs.model_validate(args)
        from knowledge.yield_synthetic_data import ensure_default_synthetic_yield_data

        paths = ensure_default_synthetic_yield_data(
            n_samples=parsed.n_samples,
            random_state=parsed.random_state,
            overwrite=parsed.overwrite,
        )
        return ToolResult(
            ok=True,
            result={
                "generator": "knowledge.yield_synthetic_data.ensure_default_synthetic_yield_data",
                "n_samples": parsed.n_samples,
                "random_state": parsed.random_state,
                "paths": _jsonable(paths),
                "note": "Deterministic offline data generation; no LLM or model API call.",
            },
            artifacts={key: str(value) for key, value in paths.items() if key.endswith("_path") or key.endswith("path")},
        )

    def _profile_yield_data(self, args: BaseModel) -> ToolResult:
        parsed = args if isinstance(args, ProfileYieldDataArgs) else ProfileYieldDataArgs.model_validate(args)
        from knowledge.yield_schema import load_yield_dataframe

        data_path = Path(parsed.data_path).expanduser()
        if not data_path.exists():
            return ToolResult(ok=False, error=f"Data path does not exist: {data_path}")
        df, meta = load_yield_dataframe(data_path)
        target = df["yield_stress"]
        result = {
            "data_path": str(data_path),
            "row_count": int(len(df)),
            "column_count": int(len(df.columns)),
            "columns": list(df.columns),
            "schema": _jsonable(meta),
            "target_summary": {
                "min": float(target.min()),
                "max": float(target.max()),
                "mean": float(target.mean()),
                "std": float(target.std()) if len(target) > 1 else 0.0,
            },
            "mode_note": "Real CSV parsing via existing schema adapter; no LLM or simulated model call.",
        }
        return ToolResult(ok=True, result=_jsonable(result))

    def _ingest_knowledge_documents(self, args: BaseModel) -> ToolResult:
        parsed = args if isinstance(args, IngestKnowledgeArgs) else IngestKnowledgeArgs.model_validate(args)
        kb = self._knowledge_base()
        result = kb.ingest(
            KnowledgeIngestRequest(
                paths=parsed.paths,
                index_version=parsed.index_version,
                chunk_size=parsed.chunk_size,
                chunk_overlap=parsed.chunk_overlap,
            )
        )
        failed = [item for item in result.get("documents", []) if item.get("status") == "failed"]
        return ToolResult(
            ok=not failed,
            result=_jsonable(result),
            error="; ".join(str(item.get("error", "")) for item in failed),
        )

    def _search_knowledge(self, args: BaseModel) -> ToolResult:
        parsed = args if isinstance(args, SearchKnowledgeArgs) else SearchKnowledgeArgs.model_validate(args)
        kb = self._knowledge_base()
        result = kb.search(
            KnowledgeSearchRequest(
                query=parsed.query,
                top_k=parsed.top_k,
                index_version=parsed.index_version,
                document_id=parsed.document_id,
                retrieval_mode=parsed.retrieval_mode,
            )
        )
        return ToolResult(ok=True, result=_jsonable(result))

    def _redact_sensitive_payload(self, args: BaseModel) -> ToolResult:
        parsed = args if isinstance(args, RedactRequest) else RedactRequest.model_validate(args)
        result = redact_payload(parsed)
        return ToolResult(ok=True, result=_jsonable(result.model_dump()))

    def _plan_token_budget(self, args: BaseModel) -> ToolResult:
        parsed = args if isinstance(args, BudgetRequest) else BudgetRequest.model_validate(args)
        result = plan_token_budget(parsed)
        return ToolResult(ok=not result.over_budget, result=_jsonable(result.model_dump()))

    def _validate_evidence_refs(self, args: BaseModel) -> ToolResult:
        parsed = args if isinstance(args, EvidenceValidationRequest) else EvidenceValidationRequest.model_validate(args)
        result = validate_evidence_refs(self.store or YieldMindStore(), parsed)
        return ToolResult(ok=result.ok, result=_jsonable(result.model_dump()), error="; ".join(item["reason"] for item in result.invalid_refs))

    def _run_fixed_baseline_eval(self, args: BaseModel) -> ToolResult:
        parsed = args if isinstance(args, RunFixedBaselineArgs) else RunFixedBaselineArgs.model_validate(args)
        from knowledge.yield_baselines import run_fixed_yield_baselines
        from knowledge.yield_schema import load_yield_dataframe

        data_path = Path(parsed.data_path).expanduser()
        if not data_path.exists():
            return ToolResult(ok=False, error=f"Data path does not exist: {data_path}")
        df, meta = load_yield_dataframe(data_path)
        if len(df) < parsed.n_splits:
            return ToolResult(ok=False, error=f"Need at least {parsed.n_splits} rows; got {len(df)}.")
        report = run_fixed_yield_baselines(
            df,
            meta.get("feature_columns"),
            n_splits=parsed.n_splits,
            random_state=parsed.random_state,
            enable_sp_monotonic=parsed.enable_sp_monotonic,
        )
        rows = report.get("baseline_results", []) if isinstance(report, dict) else []
        best_name = report.get("best_baseline", "") if isinstance(report, dict) else ""
        best = next((row for row in rows if row.get("name") == best_name), {})
        if not best:
            best = min(rows, key=lambda row: row.get("mean_rmse", float("inf")), default={})
        out_dir = self.workspace_root / "agent_workspace" / "yieldmind" / "baselines"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"fixed_baseline_{int(time.time())}.json"
        payload = {
            "data_path": str(data_path),
            "n_splits": parsed.n_splits,
            "random_state": parsed.random_state,
            "report": _jsonable(report),
            "best_baseline": _jsonable(best),
            "mode_note": "Measured deterministic sklearn baselines; no LLM or simulated model call.",
        }
        out_path.write_text(json_dumps(payload), encoding="utf-8")
        return ToolResult(
            ok=True,
            result=payload,
            artifacts={"baseline_report": str(out_path)},
        )

    def _default_candidate_report(self, data_path: Path) -> dict[str, Any]:
        return {
            "stage": "candidate_generation_offline_seed",
            "data_path": str(data_path),
            "candidate_models": [
                {
                    "id": "ridge_raw",
                    "name": "Ridge raw formulation baseline",
                    "family": "ridge",
                    "modeling_idea": "Linear/Ridge regression over normalized formulation features.",
                    "required_columns": ["phi", "sp_percent"],
                },
                {
                    "id": "hgb_packing_sp",
                    "name": "HistGradientBoosting packing + superplasticizer proxy",
                    "family": "hist_gradient_boosting",
                    "modeling_idea": "Boosted tree model with YODEL packing and SP saturation features.",
                    "required_columns": ["phi", "sp_percent"],
                },
                {
                    "id": "kernel_packing",
                    "name": "Kernel regression packing proxy",
                    "family": "kernel",
                    "modeling_idea": "Kernel model using maximum packing density phi_m / YODEL style nonlinear features.",
                    "required_columns": ["phi"],
                },
            ],
            "candidate_mechanisms": [
                {
                    "id": "yodel_packing",
                    "name": "YODEL packing / maximum packing density",
                    "paper_or_source": "project domain seed and existing yield candidate benchmark",
                    "formula_or_relationship": "Yield stress increases as solid fraction approaches effective maximum packing phi_m.",
                    "required_columns": ["phi"],
                    "role_options": ["feature"],
                    "applicability_to_current_data": "Supported when solid volume fraction exists.",
                },
                {
                    "id": "sp_saturation",
                    "name": "Superplasticizer saturation / dispersant decay",
                    "paper_or_source": "project domain seed and Lian-style formulation schema",
                    "formula_or_relationship": "SP dosage can reduce yield stress until saturation.",
                    "required_columns": ["sp_percent"],
                    "role_options": ["feature"],
                    "applicability_to_current_data": "Supported when sp_percent or equivalent dosage exists.",
                },
            ],
            "candidate_hybrid_strategies": [
                {
                    "id": "strategy_ridge_raw",
                    "name": "Ridge raw features",
                    "model_id": "ridge_raw",
                    "mechanism_ids": [],
                    "combination_method": "raw_features",
                    "data_support": "supported",
                    "deterministic_data_score_1_to_10": 6.0,
                    "implementation_plan": "Evaluate as a transparent no-mechanism candidate.",
                },
                {
                    "id": "strategy_hgb_packing_sp",
                    "name": "HGB with YODEL packing and SP proxy",
                    "model_id": "hgb_packing_sp",
                    "mechanism_ids": ["yodel_packing", "sp_saturation"],
                    "combination_method": "mechanism_features",
                    "data_support": "partial",
                    "deterministic_data_score_1_to_10": 7.5,
                    "implementation_plan": "Evaluate proxy packing/SP features through the fixed candidate benchmark harness.",
                },
                {
                    "id": "strategy_kernel_packing",
                    "name": "Kernel packing features",
                    "model_id": "kernel_packing",
                    "mechanism_ids": ["yodel_packing"],
                    "combination_method": "mechanism_features",
                    "data_support": "partial",
                    "deterministic_data_score_1_to_10": 7.0,
                    "implementation_plan": "Evaluate nonlinear packing signal as a kernel proxy candidate.",
                },
            ],
            "selected_combination": {"strategy_id": "strategy_hgb_packing_sp"},
        }

    def _run_candidate_benchmark(self, args: BaseModel) -> ToolResult:
        parsed = args if isinstance(args, RunCandidateBenchmarkArgs) else RunCandidateBenchmarkArgs.model_validate(args)
        from knowledge.yield_candidate_benchmark import apply_benchmark_selection, run_candidate_benchmark

        data_path = Path(parsed.data_path).expanduser()
        if not data_path.exists():
            return ToolResult(ok=False, error=f"Data path does not exist: {data_path}")
        if parsed.candidate_report_path:
            candidate_path = Path(parsed.candidate_report_path).expanduser()
            if not candidate_path.exists():
                return ToolResult(ok=False, error=f"Candidate report path does not exist: {candidate_path}")
            candidate_report = json.loads(candidate_path.read_text(encoding="utf-8"))
        else:
            candidate_report = self._default_candidate_report(data_path)
        anchor_path = Path(parsed.anchor_path).expanduser() if parsed.anchor_path else None
        benchmark = run_candidate_benchmark(
            data_path,
            candidate_report,
            anchor_path=anchor_path if anchor_path and anchor_path.exists() else None,
            random_state=parsed.random_state,
            max_rows=parsed.max_rows,
            n_splits=parsed.n_splits,
        )
        updated_candidate_report, selection_audit = apply_benchmark_selection(candidate_report, benchmark)
        out_dir = self.workspace_root / "agent_workspace" / "yieldmind" / "candidate_benchmarks"
        out_dir.mkdir(parents=True, exist_ok=True)
        stamp = int(time.time())
        benchmark_path = out_dir / f"candidate_benchmark_{stamp}.json"
        candidate_path = out_dir / f"candidate_report_selected_{stamp}.json"
        audit_path = out_dir / f"candidate_selection_audit_{stamp}.json"
        benchmark_path.write_text(json_dumps(_jsonable(benchmark)), encoding="utf-8")
        candidate_path.write_text(json_dumps(_jsonable(updated_candidate_report)), encoding="utf-8")
        audit_path.write_text(json_dumps(_jsonable(selection_audit)), encoding="utf-8")
        result = {
            "data_path": str(data_path),
            "benchmark_report": _jsonable(benchmark),
            "selection_audit": _jsonable(selection_audit),
            "mode_note": "Real CandidateBenchmark proxy evaluation using knowledge.yield_candidate_benchmark; no LLM call.",
        }
        return ToolResult(
            ok=True,
            result=result,
            artifacts={
                "candidate_benchmark_report": str(benchmark_path),
                "candidate_report": str(candidate_path),
                "candidate_selection_audit": str(audit_path),
            },
        )

    def _verify_artifacts(self, args: BaseModel) -> ToolResult:
        parsed = args if isinstance(args, VerifyArtifactsArgs) else VerifyArtifactsArgs.model_validate(args)
        reasons: list[str] = []
        warnings: list[str] = []
        verified: dict[str, dict[str, Any]] = {}
        for name, raw_path in parsed.artifact_paths.items():
            if not raw_path:
                warnings.append(f"{name}: empty path")
                continue
            path = Path(raw_path).expanduser()
            if not path.exists():
                reasons.append(f"{name}: missing {path}")
                continue
            if not path.is_file():
                reasons.append(f"{name}: not a file {path}")
                continue
            size = path.stat().st_size
            if size <= 0:
                reasons.append(f"{name}: empty file {path}")
            verified[name] = {"path": str(path), "size_bytes": int(size), "mtime": path.stat().st_mtime}
            if name in parsed.require_json or path.suffix.lower() == ".json":
                try:
                    payload = json.loads(path.read_text(encoding="utf-8"))
                    verified[name]["json_type"] = type(payload).__name__
                    if isinstance(payload, dict):
                        verified[name]["top_level_keys"] = sorted(str(key) for key in payload.keys())[:30]
                except Exception as exc:
                    reasons.append(f"{name}: invalid json: {type(exc).__name__}: {exc}")
        return ToolResult(
            ok=not reasons,
            result={"passed": not reasons, "verified": verified, "reasons": reasons, "warnings": warnings},
            error="; ".join(reasons),
        )

    def _build_run_report(self, args: BaseModel) -> ToolResult:
        parsed = args if isinstance(args, BuildRunReportArgs) else BuildRunReportArgs.model_validate(args)
        out_dir = self.workspace_root / "agent_workspace" / "yieldmind" / "reports"
        out_dir.mkdir(parents=True, exist_ok=True)
        stamp = int(time.time())

        def read_json(path_str: str) -> dict[str, Any]:
            if not path_str:
                return {}
            path = Path(path_str).expanduser()
            if not path.exists() or path.suffix.lower() != ".json":
                return {}
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                return payload if isinstance(payload, dict) else {"value": payload}
            except Exception as exc:
                return {"read_error": f"{type(exc).__name__}: {exc}"}

        artifact_summaries = {
            name: {
                "path": path,
                "json": read_json(path),
            }
            for name, path in parsed.artifact_paths.items()
        }
        selected_strategy = None
        benchmark_json = artifact_summaries.get("candidate_benchmark_report", {}).get("json", {})
        if isinstance(benchmark_json, dict):
            selected_strategy = benchmark_json.get("selected_strategy_id")
        evidence_refs = [ref.model_dump(mode="json") for ref in parsed.evidence_refs]
        report = {
            "title": parsed.title,
            "run_id": parsed.run_id,
            "created_at": time.time(),
            "artifact_paths": parsed.artifact_paths,
            "evidence_refs": evidence_refs,
            "selected_strategy_id": selected_strategy,
            "notes": parsed.notes,
            "artifact_summaries": artifact_summaries,
            "mode_note": "Report generated from real artifact files; no model call.",
        }
        json_path = out_dir / f"yieldmind_report_{stamp}.json"
        md_path = out_dir / f"yieldmind_report_{stamp}.md"
        json_path.write_text(json_dumps(_jsonable(report)), encoding="utf-8")
        md_lines = [
            f"# {parsed.title}",
            "",
            f"- run_id: `{parsed.run_id or 'n/a'}`",
            f"- selected_strategy_id: `{selected_strategy or 'n/a'}`",
            "- model calls: `0`",
            "",
            "## Artifacts",
        ]
        for name, path in parsed.artifact_paths.items():
            md_lines.append(f"- `{name}`: `{path}`")
        if evidence_refs:
            md_lines.extend(["", "## Evidence"])
            for ref in evidence_refs:
                md_lines.append(
                    f"- `{ref.get('chunk_id', '')}` from `{ref.get('source_path', '')}` "
                    f"(index `{ref.get('index_version', '')}`)"
                )
        if parsed.notes:
            md_lines.extend(["", "## Notes"])
            md_lines.extend(f"- {note}" for note in parsed.notes)
        md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
        return ToolResult(
            ok=True,
            result=_jsonable(report),
            artifacts={"report_json": str(json_path), "report_md": str(md_path)},
        )

    def _run_sandboxed_python(self, args: BaseModel) -> ToolResult:
        parsed = args if isinstance(args, SandboxPythonArgs) else SandboxPythonArgs.model_validate(args)
        sandbox = SubprocessSandbox(self.workspace_root)
        result = sandbox.run(
            SandboxCommand(
                argv=parsed.argv,
                cwd=parsed.cwd,
                timeout_seconds=parsed.timeout_seconds,
                env=parsed.env,
            )
        )
        return ToolResult(ok=result.ok, mode="isolated_subprocess", result=result.model_dump())

    def _run_docker_sandboxed_python(self, args: BaseModel) -> ToolResult:
        parsed = args if isinstance(args, DockerSandboxPythonArgs) else DockerSandboxPythonArgs.model_validate(args)
        sandbox = DockerSandbox(self.workspace_root)
        result = sandbox.run(DockerSandboxCommand.model_validate(parsed.model_dump()))
        return ToolResult(
            ok=result.ok,
            mode=result.mode,
            result=result.model_dump(),
            error=result.error,
        )

    def _summarize_run_artifacts(self, args: BaseModel) -> ToolResult:
        parsed = args if isinstance(args, SummarizeRunArtifactsArgs) else SummarizeRunArtifactsArgs.model_validate(args)
        run_dir = Path(parsed.run_dir).expanduser()
        if not run_dir.exists():
            return ToolResult(ok=False, error=f"Run directory does not exist: {run_dir}")

        def read_json(path: Path) -> dict[str, Any]:
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                return payload if isinstance(payload, dict) else {}
            except Exception:
                return {}

        metrics_path = run_dir / "metrics" / "metrics.json"
        free_search_path = run_dir / "metrics" / "free_search_report.json"
        mechanism_path = run_dir / "logs" / "mechanism_report.json"
        predictions_path = run_dir / "predictions" / "yield_predictions.csv"
        if not predictions_path.exists():
            predictions_path = run_dir / "metrics" / "predictions.csv"
        prediction_preview: list[dict[str, Any]] = []
        if predictions_path.exists() and parsed.prediction_preview_rows:
            with predictions_path.open("r", encoding="utf-8-sig", newline="") as handle:
                for idx, row in enumerate(csv.DictReader(handle)):
                    if idx >= parsed.prediction_preview_rows:
                        break
                    prediction_preview.append(dict(row))
        result = {
            "run_dir": str(run_dir),
            "exists": True,
            "artifacts": {
                "metrics": str(metrics_path) if metrics_path.exists() else "",
                "free_search_report": str(free_search_path) if free_search_path.exists() else "",
                "mechanism_report": str(mechanism_path) if mechanism_path.exists() else "",
                "predictions": str(predictions_path) if predictions_path.exists() else "",
            },
            "metrics": read_json(metrics_path),
            "free_search_report_summary": {
                key: value
                for key, value in read_json(free_search_path).items()
                if key in {"champion", "best_fixed_baseline_oof_rmse", "baseline_status", "status"}
            },
            "mechanism_report": read_json(mechanism_path),
            "prediction_preview": prediction_preview,
        }
        return ToolResult(ok=True, result=_jsonable(result))

    def _run_existing_pipeline(self, args: BaseModel) -> ToolResult:
        parsed = args if isinstance(args, ExistingPipelineArgs) else ExistingPipelineArgs.model_validate(args)
        if not parsed.allow_live_llm:
            return ToolResult(
                ok=False,
                mode=LIVE_LLM_MODE,
                error=(
                    "run_existing_yield_pipeline can call the configured LLM through run_yield.py. "
                    "Set allow_live_llm=true to make a real model call; no simulated pipeline run was produced."
                ),
                llm_calls=0,
            )
        argv = [
            sys.executable,
            "run_yield.py",
            "--prompt",
            parsed.prompt,
            "--data-path",
            parsed.data_path,
            "--run-dir",
            parsed.run_dir,
            "--n-revise",
            str(parsed.n_revise),
            "--operation-attempts",
            str(parsed.operation_attempts),
            "--external-search" if parsed.external_search else "--no-external-search",
        ]
        if parsed.test_path:
            argv.extend(["--test-path", parsed.test_path])
        sandbox = SubprocessSandbox(self.workspace_root)
        result = sandbox.run(
            SandboxCommand(
                argv=argv,
                cwd=".",
                timeout_seconds=parsed.timeout_seconds,
                env={"YIELDMIND_LIVE_LLM_RUN": "1"},
            )
        )
        return ToolResult(
            ok=result.ok,
            mode=LIVE_LLM_MODE,
            result=result.model_dump(),
            artifacts={"run_dir": parsed.run_dir},
            llm_calls=-1,
        )


def registry_for_workspace(
    store: YieldMindStore | None = None,
    *,
    knowledge_base: KnowledgeBase | None = None,
    knowledge_base_factory: Callable[[], KnowledgeBase] | None = None,
) -> ToolRegistry:
    return ToolRegistry(
        workspace_root=PROJECT_ROOT,
        store=store,
        knowledge_base=knowledge_base,
        knowledge_base_factory=knowledge_base_factory,
    )
