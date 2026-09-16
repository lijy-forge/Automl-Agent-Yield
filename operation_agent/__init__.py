import os
import re
import time

from configs import AVAILABLE_LLMs, OPERATION_MAX_ERROR_CHARS
from operation_agent.execution import execute_multifile_package, execute_script
from operation_agent.yield_guardrails import (
    build_yield_contract_instruction,
    is_yield_task,
    preflight_yield_source_reasons,
    verify_yield_run,
)
from utils import get_client, print_message


agent_profile = """You are an MLOps engineer for a yield-stress AutoML project.
Your job is to write complete, runnable Python training code that loads the
provided data, avoids target leakage, evaluates with the requested protocol,
saves artifacts, and prints a clear success line only after mandatory outputs
are written."""


class OperationAgent:
    def __init__(
        self,
        user_requirements,
        llm,
        code_path,
        device=0,
        task=None,
        forced_skill=None,
        research_mode=False,
        research_scope=None,
    ):
        self.agent_type = "operation"
        self.llm = llm
        self.model = AVAILABLE_LLMs[llm]["model"]
        self.experiment_logs = []
        self.user_requirements = user_requirements
        self.root_path = "./agent_workspace/exp"
        self.code_path = code_path
        self.device = device
        self.task = task
        self.forced_skill = forced_skill
        self.research_mode = bool(research_mode)
        self.research_scope = str(research_scope or "").strip().lower()
        self.money = {}

    def _save_failed_candidate(self, raw_completion: str, iteration: int, reason: str, code: str | None = None):
        try:
            failed_dir = os.path.join(self.root_path, "failed_candidates")
            os.makedirs(failed_dir, exist_ok=True)
            safe_code_path = self.code_path.strip("/").replace("/", "_") or "candidate"
            base = f"{safe_code_path}_itr{iteration}"
            with open(os.path.join(failed_dir, f"{base}_reason.txt"), "w", encoding="utf-8") as f:
                f.write(str(reason))
            with open(os.path.join(failed_dir, f"{base}_raw.txt"), "w", encoding="utf-8") as f:
                f.write(raw_completion or "")
            if code:
                with open(os.path.join(failed_dir, f"{base}.py"), "w", encoding="utf-8") as f:
                    f.write(code)
        except Exception:
            pass

    @staticmethod
    def _strip_markdown_code_fences(text: str) -> str:
        cleaned_lines = []
        for line in str(text or "").splitlines():
            if re.match(r"^\s*```(?:python|py)?\s*$", line, flags=re.IGNORECASE):
                continue
            cleaned_lines.append(line)
        return "\n".join(cleaned_lines).strip()

    @staticmethod
    def _looks_like_incomplete_python(code: str) -> bool:
        text = str(code or "").strip()
        if len(text.splitlines()) < 8 or len(text) < 200:
            return True
        markers = ("import ", "from ", "def ", "class ", "if __name__")
        return not any(marker in text.lower() for marker in markers)

    @staticmethod
    def _format_yield_preflight_failure(reasons: list[str]) -> str:
        reason_text = "\n".join(f"- {reason}" for reason in reasons)
        return f"Yield preflight static check failed:\n{reason_text}"

    def _parse_multifile_response(self, raw_completion: str):
        markers = list(re.finditer(r"^\s*#\s*===\s*FILE:\s*(.+?)\s*===\s*$", raw_completion or "", flags=re.MULTILINE))
        if not markers:
            return None
        files = {}
        for i, marker in enumerate(markers):
            fname = marker.group(1).strip()
            start = marker.end()
            end = markers[i + 1].start() if i + 1 < len(markers) else len(raw_completion)
            code = self._strip_markdown_code_fences(raw_completion[start:end])
            if code and not self._looks_like_incomplete_python(code):
                files[fname] = code
        return files or None

    @staticmethod
    def _pick_main_file(files_dict: dict) -> str:
        for candidate in ("train.py", "main.py"):
            for fname in files_dict:
                if os.path.basename(fname) == candidate:
                    return fname
        return sorted(files_dict.keys())[0]

    def self_validation(self, filename):
        return execute_script(filename, device=self.device)

    def _build_exec_prompt(self, code_instructions: str, previous_code: str, previous_error: str) -> str:
        downstream_task = "yield_stress_regression"
        if isinstance(self.user_requirements, dict):
            downstream_task = (
                self.user_requirements.get("problem", {}).get("downstream_task")
                or self.user_requirements.get("task")
                or downstream_task
            )
        return f"""Carefully read the following instructions and write complete Python code for the {downstream_task} task.

# Instructions
{code_instructions}

# Previously Written Code
{previous_code}

# Error from the Previously Written Code
{previous_error[-OPERATION_MAX_ERROR_CHARS:]}

Repository requirements:
- Target the current environment; do not install packages or ask for a different interpreter.
- Execute non-interactively from start to finish.
- Save tabular artifacts as CSV, model artifacts as .pt/.pth/.pkl/.joblib, and metadata as JSON.
- Do not use parquet, pyarrow, fastparquet, Gradio, Flask, FastAPI, Streamlit, Docker, MLflow, W&B, or shell installers.
- If writing JSON, convert numpy/pandas scalar and array objects to plain Python JSON types first.

Output format:
- Prefer a single Python code block starting with ```python and ending with ```.
- If multiple files are required, mark each file as '# === FILE: filename.py ===' followed by a Python code block.
"""

    def _extract_single_file_code(self, raw_completion: str) -> str | None:
        if "```python" not in raw_completion:
            return None
        code = raw_completion.split("```python", 1)[1].split("```", 1)[0]
        return code if code.strip() else None

    def generate_yield_plugin(self, code_instructions, n_attempts=3):
        """Ask the LLM for a CONSTRAINED Python candidate plugin (not a full script).

        Returns {ok, plugin_source, attempts, error_logs}. The plugin must pass
        both static (source) and dynamic (module) preflight before it is accepted;
        rejection reasons are fed back for the next attempt.
        """
        from operation_agent.yield_plugin_contract import (
            PLUGIN_API_SPEC,
            load_plugin_from_source,
            validate_plugin_feature_declaration,
            validate_plugin_module,
            validate_plugin_source,
        )

        error_logs: list[str] = []
        log = "Nothing. This is your first attempt."
        for iteration in range(max(1, int(n_attempts))):
            try:
                prompt = (
                    f"{code_instructions}\n\n{PLUGIN_API_SPEC}\n\n"
                    f"# Feedback on your previous attempt\n{log}\n\n"
                    "Return ONLY one ```python code block defining CANDIDATE_SPEC, "
                    "add_features(df, fit_context), and make_model(random_state)."
                )
                res = get_client(self.llm).chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": agent_profile},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.2,
                )
                raw_completion = res.choices[0].message.content.strip()
                self.money[f"Operation_Plugin_{iteration}"] = res.usage.to_dict(mode="json")
                source = self._extract_single_file_code(raw_completion) or self._strip_markdown_code_fences(raw_completion)

                reasons = validate_plugin_source(source)
                if not reasons:
                    module = load_plugin_from_source(source)
                    reasons = validate_plugin_module(module)
                if not reasons:
                    reasons = validate_plugin_feature_declaration(
                        source, getattr(module, "CANDIDATE_SPEC", {}) or {}
                    )
                if not reasons:
                    print_message(self.agent_type, f"Accepted structured plugin on attempt #{iteration}.")
                    return {"ok": True, "plugin_source": source, "attempts": iteration + 1, "error_logs": error_logs}

                log = "Plugin rejected by preflight:\n" + "\n".join(f"- {r}" for r in reasons)
                error_logs.append(log)
                self._save_failed_candidate(raw_completion, iteration, log, source)
                print_message(self.agent_type, f"Plugin attempt #{iteration} rejected:\n{log}")
            except Exception as exc:
                log = f"Plugin generation error: {type(exc).__name__}: {exc}"
                error_logs.append(log)
                print_message(self.agent_type, log)

        return {"ok": False, "plugin_source": None, "attempts": int(n_attempts), "error_logs": error_logs}

    def generate_yield_mechanism(self, mechanism_brief, sample_df, n_attempts=3):
        """Ask the LLM to implement ONE searched mechanism as executable shape code.

        `mechanism_brief` is the searched formula/description (name, source, and
        the relationship to encode). `sample_df` drives the physics preflight.
        Returns {ok, mechanism_source, mechanism, attempts, error_logs}; the
        mechanism must pass static + dynamic + physics preflight (see
        yield_mechanism_contract) before it is accepted, with rejection reasons
        fed back for the next attempt.
        """
        from operation_agent.yield_mechanism_contract import (
            MECHANISM_API_SPEC,
            preflight_mechanism,
        )

        error_logs: list[str] = []
        log = "Nothing. This is your first attempt."
        for iteration in range(max(1, int(n_attempts))):
            try:
                prompt = (
                    f"{MECHANISM_API_SPEC}\n\n"
                    f"# The searched mechanism you must implement\n{mechanism_brief}\n\n"
                    f"# Feedback on your previous attempt\n{log}\n\n"
                    "Return ONLY one ```python code block defining MECHANISM_SPEC and shape(df, params)."
                )
                res = get_client(self.llm).chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": agent_profile},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.2,
                )
                raw_completion = res.choices[0].message.content.strip()
                self.money[f"Operation_Mechanism_{iteration}"] = res.usage.to_dict(mode="json")
                source = self._extract_single_file_code(raw_completion) or self._strip_markdown_code_fences(raw_completion)

                reasons, mechanism = preflight_mechanism(source, sample_df)
                if not reasons and mechanism is not None:
                    print_message(self.agent_type, f"Accepted searched mechanism '{mechanism.id}' on attempt #{iteration}.")
                    return {"ok": True, "mechanism_source": source, "mechanism": mechanism,
                            "attempts": iteration + 1, "error_logs": error_logs}

                log = "Mechanism rejected by preflight:\n" + "\n".join(f"- {r}" for r in reasons)
                error_logs.append(log)
                self._save_failed_candidate(raw_completion, iteration, log, source)
                print_message(self.agent_type, f"Mechanism attempt #{iteration} rejected:\n{log}")
            except Exception as exc:
                log = f"Mechanism generation error: {type(exc).__name__}: {exc}"
                error_logs.append(log)
                print_message(self.agent_type, log)

        return {"ok": False, "mechanism_source": None, "mechanism": None,
                "attempts": int(n_attempts), "error_logs": error_logs}

    def generate_yield_model(self, model_brief, sample_df, y_sample, n_attempts=3):
        """Ask the LLM to implement ONE searched model as a bounded estimator factory.

        The LLM writes only MODEL_SPEC + make_estimator(params, random_state).
        The fixed harness still owns data loading, fold-local CV, anchor scoring,
        artifacts, and leakage controls. Rejection reasons are fed back across
        attempts, mirroring generate_yield_mechanism.
        """
        from operation_agent.yield_model_contract import (
            COMPLEX_MODEL_API_SPEC,
            MODEL_API_SPEC,
            preflight_model,
        )

        error_logs: list[str] = []
        log = "Nothing. This is your first attempt."
        brief_text = str(model_brief or "").lower()
        use_complex_contract = (
            "latent_physics_architecture" in brief_text
            or "complex_model_contract" in brief_text
            or "hidden_parameter_physics" in brief_text
            or "m1_eff" in brief_text
        )
        api_spec = COMPLEX_MODEL_API_SPEC if use_complex_contract else MODEL_API_SPEC
        for iteration in range(max(1, int(n_attempts))):
            try:
                prompt = (
                    f"{api_spec}\n\n"
                    f"# The searched model idea you must implement\n{model_brief}\n\n"
                    f"# Feedback on your previous attempt\n{log}\n\n"
                    "Return ONLY one ```python code block defining MODEL_SPEC and "
                    "make_estimator(params, random_state)."
                )
                res = get_client(self.llm).chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": agent_profile},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.2,
                )
                raw_completion = res.choices[0].message.content.strip()
                self.money[f"Operation_Model_{iteration}"] = res.usage.to_dict(mode="json")
                source = self._extract_single_file_code(raw_completion) or self._strip_markdown_code_fences(raw_completion)

                reasons, model = preflight_model(source, sample_df, y_sample)
                if not reasons and model is not None:
                    print_message(self.agent_type, f"Accepted searched model '{model.id}' on attempt #{iteration}.")
                    return {"ok": True, "model_source": source, "model": model,
                            "attempts": iteration + 1, "error_logs": error_logs}

                log = "Model rejected by preflight:\n" + "\n".join(f"- {r}" for r in reasons)
                error_logs.append(log)
                self._save_failed_candidate(raw_completion, iteration, log, source)
                print_message(self.agent_type, f"Model attempt #{iteration} rejected:\n{log}")
            except Exception as exc:
                log = f"Model generation error: {type(exc).__name__}: {exc}"
                error_logs.append(log)
                print_message(self.agent_type, log)

        return {"ok": False, "model_source": None, "model": None,
                "attempts": int(n_attempts), "error_logs": error_logs}

    def implement_solution(self, code_instructions, full_pipeline=True, code="", n_attempts=5):
        yield_task = is_yield_task(self.task, self.user_requirements)
        if yield_task:
            print_message(
                self.agent_type,
                "I am implementing a yield-stress generated-code task. "
                "Search report, model plan, data profile, and full code instructions are saved in the run directory.",
            )
            code_instructions = code_instructions + build_yield_contract_instruction()
        else:
            print_message(self.agent_type, f"I am implementing the following instruction:\n\r{code_instructions}")

        log = "Nothing. This is your first attempt."
        error_logs = []
        completion = ""
        action_result = ""
        rcode = -1

        for iteration in range(max(1, int(n_attempts))):
            try:
                prompt = self._build_exec_prompt(code_instructions, code, log)
                res = get_client(self.llm).chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": agent_profile},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.3,
                )
                raw_completion = res.choices[0].message.content.strip()
                self.money[f"Operation_Coding_{iteration}"] = res.usage.to_dict(mode="json")

                multifile = self._parse_multifile_response(raw_completion)
                if multifile:
                    main_file = self._pick_main_file(multifile)
                    completion = "\n\n".join(f"# === FILE: {fn} ===\n{fc}" for fn, fc in multifile.items())
                    print_message(self.agent_type, f"Multi-file package detected ({len(multifile)} files). Entry: {main_file}")
                    if yield_task:
                        preflight_reasons = preflight_yield_source_reasons(completion)
                        if preflight_reasons:
                            log = self._format_yield_preflight_failure(preflight_reasons)
                            code = completion
                            action_result = log
                            rcode = -1
                            error_logs.append(log)
                            self._save_failed_candidate(raw_completion, iteration, log, code)
                            print_message(self.agent_type, f"I got this yield preflight error (itr #{iteration}):\n{log}")
                            continue
                    started_at = time.time()
                    rcode, log = execute_multifile_package(
                        files_dict=multifile,
                        main_file=main_file,
                        base_dir=f"{self.root_path}{self.code_path}_pkg",
                        device=str(self.device),
                    )
                    code = completion
                else:
                    single_code = self._extract_single_file_code(raw_completion)
                    if single_code is None:
                        log = "Generated response did not contain a python code block."
                        self._save_failed_candidate(raw_completion, iteration, log)
                        error_logs.append(log)
                        print_message(self.agent_type, f"I got this error (itr #{iteration}): {log}")
                        continue
                    completion = single_code
                    if yield_task:
                        preflight_reasons = preflight_yield_source_reasons(completion)
                        if preflight_reasons:
                            log = self._format_yield_preflight_failure(preflight_reasons)
                            code = completion
                            action_result = log
                            rcode = -1
                            error_logs.append(log)
                            self._save_failed_candidate(raw_completion, iteration, log, code)
                            print_message(self.agent_type, f"I got this yield preflight error (itr #{iteration}):\n{log}")
                            continue
                    filename = f"{self.root_path}{self.code_path}.py"
                    os.makedirs(os.path.dirname(filename), exist_ok=True)
                    with open(filename, "w", encoding="utf-8") as f:
                        f.write(completion)
                    started_at = time.time()
                    rcode, log = self.self_validation(filename)
                    code = completion

                if yield_task:
                    verification = verify_yield_run(
                        source_code=code,
                        action_result=log,
                        return_code=rcode,
                        run_dir=os.getenv("YIELD_RUN_DIR"),
                        started_at=started_at,
                    )
                    if verification.passed:
                        action_result = log
                        rcode = 0
                        break
                    rcode = -1
                    reason_text = "\n".join(f"- {reason}" for reason in verification.reasons)
                    log = (
                        f"{log}\n\nDeterministic yield verification failed:\n"
                        f"{reason_text}\n\nRepair guidance:\n{verification.repair_guidance}"
                    )
                    error_logs.append(log)
                    action_result = log
                    self._save_failed_candidate(raw_completion, iteration, log, code)
                    print_message(
                        self.agent_type,
                        "I got this yield verification error "
                        f"(itr #{iteration}):\n{reason_text}\n\nRepair guidance:\n{verification.repair_guidance}",
                    )
                    continue

                action_result = log
                if rcode == 0:
                    break
                error_logs.append(log)
                print_message(self.agent_type, f"I got this error (itr #{iteration}): {log}")
            except Exception as exc:
                log = f"Execution error occurs: {exc}"
                action_result = log
                error_logs.append(log)
                print_message(self.agent_type, f"===== Retry: {iteration + 1} =====")
                print_message(self.agent_type, log)

        if yield_task:
            if rcode == 0:
                print_message(
                    self.agent_type,
                    "Yield generated code executed and passed deterministic guardrails. "
                    f"Run directory: {os.getenv('YIELD_RUN_DIR', '')}",
                )
            else:
                print_message(
                    self.agent_type,
                    "Yield generated code did not pass execution/guardrails. "
                    f"Last output:\n{action_result[-OPERATION_MAX_ERROR_CHARS:]}",
                )
        else:
            print_message(self.agent_type, f"I executed the given plan and got the follow results:\n\n{action_result}")

        return {"rcode": rcode, "action_result": action_result, "code": completion, "error_logs": error_logs}
