"""Yield-stress AutoML live dashboard server."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

from flask import Flask, Response, jsonify, request


EVENT_LOG = "agent_workspace/live_log.jsonl"
DEFAULT_YIELD_CSV_PATH = (
    "/Users/lijiayao/my-project/automl-agent-yield/"
    "agent_workspace/data/generated_yield_stress/generated_yield_stress_data_engineered.csv"
)
DEFAULT_YIELD_TEST_PATH = (
    "/Users/lijiayao/my-project/automl-agent-yield/"
    "agent_workspace/data/generated_yield_stress/no_anchor.csv"
)

app = Flask(__name__)

_pipeline_thread = None
_pipeline_lock = threading.Lock()
_stop_requested = threading.Event()


def _as_bool_param(value):
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _read_json_file(path, default=None):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default


def _load_config():
    return {
        "title": "屈服值 AutoML Agent",
        "subtitle": "外部搜索 + 自由代码生成 + 机理依据审计",
    }


def _load_template():
    tpl_path = Path(__file__).with_name("templates") / "yield_dashboard.html"
    return tpl_path.read_text(encoding="utf-8")


def _latest_yield_run_dir():
    runs_dir = Path("agent_workspace/runs")
    if not runs_dir.is_dir():
        return ""
    candidates: set[Path] = set()
    useful = {"yield_predictions.csv", "metrics.json", "mechanism_report.json", "run_result.json"}
    for root, _, files in os.walk(runs_dir):
        if any(name in useful for name in files):
            rel = Path(root).relative_to(runs_dir)
            if not rel.parts:
                # A useful file sitting directly in runs/ root (stray leftover),
                # not inside a run directory — skip it so we don't IndexError.
                continue
            candidates.add(runs_dir / rel.parts[0])
    if not candidates:
        return ""

    def newest_mtime(path: Path) -> float:
        mtimes = []
        for root, _, files in os.walk(path):
            for name in files:
                try:
                    mtimes.append(os.path.getmtime(os.path.join(root, name)))
                except OSError:
                    pass
        return max(mtimes) if mtimes else path.stat().st_mtime

    return str(max(candidates, key=newest_mtime))


@app.route("/")
def index():
    cfg = _load_config()
    html = _load_template()
    html = html.replace("{{ title }}", cfg["title"])
    html = html.replace("{{ subtitle }}", cfg["subtitle"])
    return html


@app.route("/events")
def events():
    def stream():
        last_pos = 0
        while True:
            try:
                if os.path.exists(EVENT_LOG):
                    current_size = os.path.getsize(EVENT_LOG)
                    if current_size < last_pos:
                        last_pos = 0
                    with open(EVENT_LOG, "r", encoding="utf-8") as f:
                        f.seek(last_pos)
                        new_lines = f.readlines()
                        last_pos = f.tell()
                    for line in new_lines:
                        line = line.strip()
                        if line:
                            yield f"data: {line}\n\n"
            except Exception:
                pass
            time.sleep(0.15)

    return Response(
        stream(),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.route("/api/events")
def api_events():
    try:
        limit = int(request.args.get("limit", "0") or "0")
    except (TypeError, ValueError):
        limit = 0
    events_payload = []
    if os.path.exists(EVENT_LOG):
        with open(EVENT_LOG, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    events_payload.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    if limit > 0:
        events_payload = events_payload[-limit:]
    return jsonify(events_payload)


@app.route("/api/predictions")
def api_predictions():
    import pandas as pd

    candidates = []
    runs_dir = Path("agent_workspace/runs")
    if runs_dir.is_dir():
        for root, _, files in os.walk(runs_dir):
            if "yield_predictions.csv" in files:
                candidates.append(Path(root) / "yield_predictions.csv")
    if not candidates:
        return jsonify([])
    pred_path = max(candidates, key=lambda path: path.stat().st_mtime)
    try:
        df = pd.read_csv(pred_path, encoding="utf-8-sig")
        return jsonify(df.to_dict(orient="records"))
    except Exception:
        return jsonify([])


@app.route("/api/yield_summary")
def api_yield_summary():
    import pandas as pd

    run_dir = _latest_yield_run_dir()
    if not run_dir:
        return jsonify({
            "exists": False,
            "run_dir": "",
            "metrics": {},
            "mechanism_report": {},
            "search_report": {},
            "candidate_report": {},
            "model_plan": {},
            "predictions": [],
        })

    run_path = Path(run_dir)
    metrics_path = run_path / "metrics" / "metrics.json"
    free_search_report_path = run_path / "metrics" / "free_search_report.json"
    mechanism_path = run_path / "logs" / "mechanism_report.json"
    prediction_path = run_path / "predictions" / "yield_predictions.csv"
    if not prediction_path.exists():
        prediction_path = run_path / "metrics" / "predictions.csv"
    search_path = run_path / "search_report.json"
    if not search_path.exists():
        search_path = free_search_report_path
    candidate_path = run_path / "candidate_report.json"
    model_plan_path = run_path / "model_plan.json"
    run_result_path = run_path / "run_result.json"
    generated_code_path = run_path / "generated_code.py"
    if not generated_code_path.exists():
        generated_code_path = run_path / "predict.py"
    synthetic_report_path = run_path / "logs" / "synthetic_data_report.json"
    if not synthetic_report_path.exists():
        synthetic_report_path = run_path / "data" / "synthetic_data_report.json"

    predictions = []
    if prediction_path.exists():
        try:
            df = pd.read_csv(prediction_path, encoding="utf-8-sig")
            predictions = df.head(30).to_dict(orient="records")
        except Exception:
            predictions = []

    run_result = _read_json_file(run_result_path, {})
    generated_code = ""
    if generated_code_path.exists():
        try:
            generated_code = generated_code_path.read_text(encoding="utf-8")
        except Exception:
            generated_code = ""
    if not generated_code and isinstance(run_result, dict):
        generated_code = str(run_result.get("code") or "")

    metrics_payload = _read_json_file(metrics_path, {})
    free_search_report = _read_json_file(free_search_report_path, {})
    if not metrics_payload and isinstance(free_search_report, dict):
        champion = free_search_report.get("champion") or {}
        oof = champion.get("oof_metrics") or {}
        metrics_payload = {
            "RMSE": oof.get("rmse"),
            "MAE": oof.get("mae"),
            "R2": oof.get("r2"),
            "MAPE": oof.get("mape"),
            "evaluation_protocol": "free_search_5fold_oof",
            "champion": champion.get("name"),
            "baseline_status": champion.get("baseline_status"),
            "best_fixed_baseline_oof_rmse": free_search_report.get("best_fixed_baseline_oof_rmse"),
        }

    return jsonify({
        "exists": True,
        "run_dir": run_dir,
        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(run_path.stat().st_mtime)),
        "metrics": metrics_payload,
        "mechanism_report": _read_json_file(mechanism_path, {}),
        "synthetic_data_report": _read_json_file(synthetic_report_path, {}),
        "search_report": _read_json_file(search_path, {}),
        "free_search_report": free_search_report,
        "candidate_report": _read_json_file(candidate_path, {}),
        "model_plan": _read_json_file(model_plan_path, {}),
        "run_result": run_result,
        "generated_code_path": str(generated_code_path) if generated_code else "",
        "generated_code": generated_code,
        "training_log": str(run_result.get("action_result") or "") if isinstance(run_result, dict) else "",
        "repair_logs": run_result.get("error_logs", []) if isinstance(run_result, dict) else [],
        "predictions": predictions,
    })


@app.route("/api/status")
def api_status():
    with _pipeline_lock:
        is_running = _pipeline_thread is not None and _pipeline_thread.is_alive()
    return jsonify({"running": is_running})


@app.route("/api/stop", methods=["POST"])
def api_stop():
    _stop_requested.set()
    with _pipeline_lock:
        is_running = _pipeline_thread is not None and _pipeline_thread.is_alive()
    return jsonify({"status": "stop_requested", "running": is_running})


@app.route("/api/run", methods=["POST"])
def api_run():
    global _pipeline_thread
    with _pipeline_lock:
        if _pipeline_thread and _pipeline_thread.is_alive():
            return jsonify({"error": "Pipeline is already running"}), 409

    params = request.json or {}
    _stop_requested.clear()
    os.makedirs("agent_workspace", exist_ok=True)
    with open(EVENT_LOG, "w", encoding="utf-8"):
        pass

    _pipeline_thread = threading.Thread(target=_run_yield_pipeline, args=(params,), daemon=True)
    _pipeline_thread.start()
    return jsonify({"status": "started"})


def _run_yield_pipeline(params):
    from utils import _emit_event

    prompt = str(params.get("prompt") or "Build a yield-stress AutoML model.")
    csv_path = str(params.get("csv_path") or DEFAULT_YIELD_CSV_PATH)
    test_path = str(params.get("test_path") or DEFAULT_YIELD_TEST_PATH)
    n_revise = int(params.get("n_revise") if params.get("n_revise") is not None else 0)
    operation_attempts = int(params.get("operation_attempts") if params.get("operation_attempts") is not None else 3)
    budget = str(params.get("budget") or "normal")
    external_search = _as_bool_param(params.get("external_search", False))
    allow_search_fallback = _as_bool_param(params.get("allow_search_fallback", True))
    run_dir = str(params.get("run_dir") or "").strip()
    if not run_dir:
        run_dir = os.path.join("agent_workspace", "runs", f"yield_dashboard_{time.strftime('%Y%m%d_%H%M%S')}")

    _emit_event("manager", "Agent Manager:", "Yield dashboard pipeline started.")
    _emit_event("data", "Data Agent:", f"Training CSV: {csv_path}")
    _emit_event("data", "Data Agent:", f"Temporary test CSV: {test_path}")

    cmd = [
        sys.executable,
        "run_yield.py",
        "--prompt",
        prompt,
        "--data-path",
        csv_path,
        "--test-path",
        test_path,
        "--run-dir",
        run_dir,
        "--llm",
        "ark",
        "--n-revise",
        str(n_revise),
        "--operation-attempts",
        str(max(1, operation_attempts)),
        "--no-synthetic-data",
    ]
    cmd.append("--external-search" if external_search else "--no-external-search")
    cmd.append("--no-require-search-results" if allow_search_fallback else "--require-search-results")

    env = os.environ.copy()
    env["YIELD_RUN_DIR"] = os.path.abspath(run_dir)
    env["YIELD_DATA_PATH"] = csv_path
    env["YIELD_TEST_PATH"] = test_path
    env["AMLA_EVENT_LOG"] = EVENT_LOG
    env["AMLA_MIRROR_EVENTS_TO_STDOUT"] = "0"
    env["YIELD_DASHBOARD"] = "1"
    env["YIELD_EXECUTION_MODE"] = "free_search"
    env["YIELD_SEARCH_BUDGET"] = budget
    env["YIELD_LLM"] = "ark"
    exhausted_ark_models = {
        "qwen-plus",
        "qwen3.7-max-2026-05-17",
        "qwen3.7-max-preview",
        "qwen3.7-plus-2026-05-26",
        "kimi-k2.6",
        "deepseek-v4-pro",
        "qwen3.6-27b",
        "qwen3.6-flash",
    }
    ark_model = os.environ.get("ARK_MODEL", "qwen3.7-flash-2026-07-15").strip()
    if not ark_model or ark_model in exhausted_ark_models:
        ark_model = "qwen3.7-flash-2026-07-15"
    env["ARK_MODEL"] = ark_model
    env["ARK_MODEL_FALLBACKS"] = "qwen3.7-flash"
    env["YIELD_SYNTHETIC_DATA"] = "0"
    env.setdefault("YIELD_AGENT_LLM_TIMEOUT", "240")
    env.setdefault("YIELD_OPERATION_TIMEOUT", "600")
    env.setdefault("YIELD_MIN_LLM_MODEL_SPECS", "2")

    process = subprocess.Popen(
        cmd,
        cwd=os.path.dirname(__file__) or ".",
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
        bufsize=1,
    )
    assert process.stdout is not None
    for raw_line in process.stdout:
        if _stop_requested.is_set():
            process.terminate()
            _emit_event("system", "SYSTEM", "Yield pipeline stop requested.")
            break
        line = raw_line.rstrip()
        if not line:
            continue
        sender = "manager"
        if "YIELD_STAGE: synthetic_data" in line:
            sender = "data"
            line = "Synthetic data stage started: generating Table 6 anchor and physics-guided training CSV."
        elif "YIELD_STAGE: search" in line:
            line = "SearchAgent stage started."
        elif "YIELD_SEARCH_SNIPPETS:" in line:
            count = line.split(":", 1)[1].strip()
            line = f"SearchAgent completed. External snippets: {count}."
        elif "YIELD_STAGE: candidates" in line:
            sender = "model"
            line = "CandidateAgent stage started: generating model and mechanism candidates."
        elif "YIELD_CANDIDATES_SAVED:" in line:
            sender = "model"
            line = "CandidateAgent completed. " + line.split(":", 1)[1].strip() + "."
        elif "YIELD_STAGE: model_plan" in line:
            sender = "model"
            line = "ModelAgent planning stage started."
        elif "YIELD_STAGE: operation" in line:
            sender = "operation"
            line = "OperationAgent code generation/execution stage started."
        elif "YIELD_STAGE: search_failed" in line:
            sender = "system"
            line = "SearchAgent returned zero external snippets; OperationAgent was not executed."
        elif "Deterministic yield verification failed" in line or "Traceback" in line:
            sender = "system"
        elif "Yield pipeline completed successfully" in line:
            sender = "operation"
            line = "Generated training script saved artifacts; waiting for deterministic guardrail verification."
        elif "YIELD_RUN_DIR:" in line or "YIELD_RESULT_SAVED:" in line:
            sender = "manager"
        else:
            continue
        _emit_event(sender, f"{sender.title()}:", line)

    return_code = process.wait()
    run_result = _read_json_file(Path(run_dir) / "run_result.json", {})
    final_rcode = run_result.get("rcode") if isinstance(run_result, dict) else None
    if return_code == 0 and final_rcode == 0:
        _emit_event("manager", "Agent Manager:", f"Yield pipeline completed. Run dir: {run_dir}")
    elif return_code == 2 or final_rcode == 2:
        _emit_event(
            "system",
            "SYSTEM",
            f"Yield pipeline stopped before OperationAgent because external search returned no snippets. Run dir: {run_dir}",
        )
    else:
        detail = f"Yield pipeline failed with return code {return_code}"
        if final_rcode is not None:
            detail += f", run_result.rcode={final_rcode}"
        action_result = ""
        if isinstance(run_result, dict):
            action_result = str(run_result.get("action_result") or "").strip()
        if action_result:
            detail += f". Last result: {action_result[-500:]}"
        _emit_event("system", "SYSTEM", f"{detail}. Run dir: {run_dir}")
        raise RuntimeError(detail)


def start_server(port=5052):
    t = threading.Thread(
        target=lambda: app.run(host="127.0.0.1", port=port, debug=False, threaded=True),
        daemon=True,
    )
    t.start()
    return t
