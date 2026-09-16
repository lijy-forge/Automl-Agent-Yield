"""Mechanism API contract for the direction-④ yield AutoML search.

Companion to yield_plugin_contract.py. Where that contract lets the LLM write a
constrained MODEL plugin, this one lets the LLM implement a searched MECHANISM as
executable code — the "external search of mechanisms" leg of the free search.

Design (see yield_direction4_mechanism_model_search_design.md):
  * The LLM writes ONLY the physics SHAPE function `shape(df, params) -> g`
    (unit-scale physics; final tau = m1 * g). It also declares a small numeric
    grid of extra params via MECHANISM_SPEC["params"].
  * The HARNESS (knowledge/yield_mechanisms.DynamicMechanism) owns the fold-local
    fit: it supplies a data-driven phi_m grid (phi_m > phi by construction),
    grid-searches the declared params, and closed-form fits the linear scale m1
    against TRAINING y only.
  * Consequence: the LLM's code NEVER sees y and cannot invert per-sample y, so a
    searched mechanism cannot leak. `shape(df, params)` has exactly this
    signature — no y argument — and that is checked below.

A searched mechanism that is physically wrong is not dangerous: it simply scores
poorly on OOF/anchor and loses to the raw arm. Preflight only rejects code that
is unsafe, non-runnable, or numerically degenerate — not code that is merely a
weak physical hypothesis.

Mechanism module MUST define two top-level symbols:

    MECHANISM_SPEC : dict
        - "id":                  str  (short snake_case)
        - "name":                str
        - "paper":               str  (source citation from the search)
        - "required_columns":    list[str]  (must include "phi")
        - "needs_shear_rate":    bool (optional, default False)
        - "needs_microstructure":bool (optional, default False)
        - "params":              dict[str, list[number]]  (extra fold-fit params;
                                 each a non-empty numeric grid; may be empty {})

    def shape(df, params):
        Return an array g (unit-scale physics), len == len(df). `params` contains
        "phi_m" (harness-supplied) plus every declared param. MUST NOT take or use
        y. Use only df input columns + params. numpy/pandas only.
"""

from __future__ import annotations

import ast
import inspect
import types
from typing import Any

import numpy as np
import pandas as pd

from knowledge.yield_candidate_benchmark import LEAKAGE_COLUMNS
from knowledge.yield_mechanisms import DynamicMechanism

REQUIRED_SPEC_KEYS = ("id", "name", "required_columns", "params")

MAX_DECLARED_PARAMS = 4          # cap search width
MAX_GRID_PER_PARAM = 8
MAX_TOTAL_PARAM_COMBOS = 1000    # phi_m grid (~5) x product(declared grids); shape calls are cheap vectorised numpy

# Static red flags — the module runs in-process, so refuse escape / IO / leakage
# / non-deterministic constructs before importing. numpy/pandas are allowed.
FORBIDDEN_SOURCE_TOKENS = (
    "subprocess", "os.system", "os.popen", "socket", "requests", "urllib",
    "open(", "read_csv", "to_csv", "eval(", "exec(", "__import__",
    "random.random", "np.random",
)


def validate_mechanism_source(source: str) -> list[str]:
    """Static preflight on mechanism source text (before import)."""
    reasons: list[str] = []
    text = str(source or "")
    if not text.strip():
        return ["Mechanism source is empty."]
    if "MECHANISM_SPEC" not in text:
        reasons.append("Mechanism must define a top-level MECHANISM_SPEC dict.")
    if "def shape" not in text:
        reasons.append("Mechanism must define shape(df, params).")
    lowered = text.lower()
    for token in FORBIDDEN_SOURCE_TOKENS:
        if token in lowered:
            reasons.append(f"Mechanism source contains a forbidden construct: '{token}'.")
    for col in sorted(LEAKAGE_COLUMNS):
        if col in ("target", "label", "source"):
            continue  # too generic to flag on substring
        if f'"{col}"' in text or f"'{col}'" in text:
            reasons.append(f"Mechanism references leakage/reference column '{col}'.")
    if not _shape_uses_df_column(text, "phi"):
        reasons.append("shape(df, params) must use df['phi']; yield-stress mechanisms must depend on solid fraction.")
    return reasons


def _shape_uses_df_column(source: str, column: str) -> bool:
    """Return True when the shape() body reads df[column] / df.column / df.get(column)."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return False
    shape_node = next(
        (node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "shape"),
        None,
    )
    if shape_node is None:
        return False
    for node in ast.walk(shape_node):
        if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name) and node.value.id == "df":
            key = node.slice
            if isinstance(key, ast.Constant) and str(key.value) == column:
                return True
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == "df":
            if node.attr == column:
                return True
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if (
                isinstance(node.func.value, ast.Name)
                and node.func.value.id == "df"
                and node.func.attr == "get"
                and node.args
                and isinstance(node.args[0], ast.Constant)
                and str(node.args[0].value) == column
            ):
                return True
    return False


def load_mechanism_from_source(source: str, module_name: str = "yield_searched_mechanism") -> types.ModuleType:
    """Execute mechanism source into an in-memory module and return it.

    numpy/pandas are PRE-INJECTED (as np/numpy and pd/pandas) so a shape that
    forgets the import still runs — the harness owns this boilerplate, the LLM
    only writes physics. An explicit `import numpy as np` in the source is
    harmless (it just rebinds the same module).
    """
    module = types.ModuleType(module_name)
    module.__dict__["__name__"] = module_name
    module.__dict__.update({"np": np, "numpy": np, "pd": pd, "pandas": pd})
    exec(compile(source, f"<{module_name}>", "exec"), module.__dict__)  # noqa: S102 - vetted by validate_mechanism_source
    return module


def _validate_spec(spec: Any) -> list[str]:
    reasons: list[str] = []
    if not isinstance(spec, dict):
        return ["MECHANISM_SPEC must be a dict."]
    for key in REQUIRED_SPEC_KEYS:
        if key not in spec:
            reasons.append(f"MECHANISM_SPEC is missing required key '{key}'.")
    if not str(spec.get("id", "")).strip():
        reasons.append("MECHANISM_SPEC['id'] must be a non-empty string.")
    req = spec.get("required_columns")
    if not isinstance(req, list) or "phi" not in [str(c) for c in (req or [])]:
        reasons.append("MECHANISM_SPEC['required_columns'] must be a list containing 'phi'.")

    params = spec.get("params")
    if not isinstance(params, dict):
        reasons.append("MECHANISM_SPEC['params'] must be a dict (may be empty {}).")
        return reasons
    if "phi_m" in params:
        reasons.append("Do not declare 'phi_m' in params; the harness supplies the phi_m grid.")
    if len(params) > MAX_DECLARED_PARAMS:
        reasons.append(f"Too many declared params ({len(params)} > {MAX_DECLARED_PARAMS}).")
    total = 5  # approx phi_m grid size
    for name, grid in params.items():
        if not isinstance(grid, (list, tuple)) or not grid:
            reasons.append(f"params['{name}'] must be a non-empty list of numbers.")
            continue
        if len(grid) > MAX_GRID_PER_PARAM:
            reasons.append(f"params['{name}'] grid too long ({len(grid)} > {MAX_GRID_PER_PARAM}).")
        if any(not isinstance(v, (int, float)) or isinstance(v, bool) for v in grid):
            reasons.append(f"params['{name}'] must contain only numbers.")
        total *= max(1, len(grid))
    if total > MAX_TOTAL_PARAM_COMBOS:
        reasons.append(f"param grid too large ({total} combos > {MAX_TOTAL_PARAM_COMBOS}); shrink grids.")
    return reasons


def _validate_shape_signature(shape_fn: Any) -> list[str]:
    if not callable(shape_fn):
        return ["shape must be a callable."]
    try:
        sig = inspect.signature(shape_fn)
    except (TypeError, ValueError):
        return []  # builtins / C funcs — skip, dynamic check will catch problems
    params = list(sig.parameters.values())
    positional = [p for p in params if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)]
    reasons: list[str] = []
    if len(positional) != 2:
        reasons.append("shape must take exactly two positional args (df, params).")
    banned = {"y", "target", "label", "y_true", "anchor"}
    for p in params:
        if p.name.lower() in banned:
            reasons.append(f"shape must not take a '{p.name}' argument (anti-leakage: shape never sees y).")
    return reasons


def validate_mechanism_module(module: types.ModuleType, sample_df: pd.DataFrame) -> list[str]:
    """Dynamic + physics preflight on an imported mechanism module."""
    reasons: list[str] = []
    spec = getattr(module, "MECHANISM_SPEC", None)
    reasons.extend(_validate_spec(spec))
    shape_fn = getattr(module, "shape", None)
    reasons.extend(_validate_shape_signature(shape_fn))
    if reasons:
        return reasons

    # physics preflight: build a test params point and run the shape once
    phi = pd.to_numeric(sample_df["phi"], errors="coerce").to_numpy(float)
    max_phi = float(np.nanmax(phi)) if np.isfinite(phi).any() else 0.5
    test_params: dict[str, Any] = {"phi_m": min(max_phi + 0.05, 0.98)}
    for name, grid in (spec.get("params") or {}).items():
        test_params[name] = grid[0]
    try:
        g = np.asarray(shape_fn(sample_df, test_params), dtype=float)
    except Exception as exc:
        return [f"shape(df, params) raised {type(exc).__name__}: {exc}"]
    if g.shape[0] != len(sample_df):
        reasons.append(f"shape returned length {g.shape[0]}, expected {len(sample_df)}.")
    finite_frac = float(np.mean(np.isfinite(g))) if g.size else 0.0
    if finite_frac < 0.5:
        reasons.append(f"shape produced mostly non-finite output ({finite_frac:.0%} finite); numerically degenerate.")
    elif np.allclose(np.nan_to_num(g), 0.0):
        reasons.append("shape produced an all-zero signal; carries no information.")
    reasons.extend(_validate_phi_response(shape_fn, sample_df, test_params, g))
    return reasons


def _validate_phi_response(shape_fn: Any, sample_df: pd.DataFrame, test_params: dict[str, Any], g_ref: np.ndarray) -> list[str]:
    """Cheap physical sanity checks for yield-stress shape functions.

    A high-solid yield-stress mechanism should use solid fraction as the main
    loading coordinate: near phi=0 the unit-scale yield-stress signal should be
    near zero, and with other variables fixed it should not mostly decrease as
    phi increases toward phi_m.
    """
    reasons: list[str] = []
    if "phi" not in sample_df.columns:
        return ["sample_df has no phi column for mechanism physics preflight."]
    phi_m = float(test_params.get("phi_m", 0.0) or 0.0)
    if not np.isfinite(phi_m) or phi_m <= 0.0:
        return ["invalid phi_m for mechanism physics preflight."]

    ref_scale = max(1.0, float(np.nanmedian(np.abs(np.nan_to_num(g_ref, nan=0.0)))))
    low_df = sample_df.head(min(16, len(sample_df))).copy()
    low_df["phi"] = 0.0
    try:
        low_g = np.asarray(shape_fn(low_df, test_params), dtype=float)
    except Exception as exc:
        return [f"shape failed low-phi physical preflight: {type(exc).__name__}: {exc}"]
    low_level = float(np.nanmedian(np.abs(np.nan_to_num(low_g, nan=0.0, posinf=ref_scale, neginf=ref_scale))))
    if low_level > max(1e-6, 1e-4 * ref_scale):
        reasons.append(
            "shape does not approach zero at phi=0; yield-stress mechanism should vanish at zero solids."
        )

    sweep_df = sample_df.head(1).copy()
    if sweep_df.empty:
        return reasons
    upper = max(0.02, min(phi_m - 1e-4, 0.98 * phi_m))
    if upper <= 0.02:
        return reasons
    sweep_df = pd.concat([sweep_df] * 16, ignore_index=True)
    sweep_df["phi"] = np.linspace(0.0, upper, len(sweep_df))
    try:
        sweep_g = np.asarray(shape_fn(sweep_df, test_params), dtype=float)
    except Exception as exc:
        return reasons + [f"shape failed phi-sweep physical preflight: {type(exc).__name__}: {exc}"]
    if np.mean(np.isfinite(sweep_g)) < 1.0:
        reasons.append("shape produced non-finite values during phi-sweep physical preflight.")
    else:
        diffs = np.diff(sweep_g)
        if len(diffs) and float(np.mean(diffs < -1e-8)) > 0.2:
            reasons.append("shape mostly decreases as phi increases; expected non-decreasing packing/yield trend.")
    return reasons


def build_llm_mechanism(module: types.ModuleType, source: str | None = None) -> DynamicMechanism:
    """Wrap a validated mechanism module into a runnable DynamicMechanism.

    `source` (the LLM text) is stored on the mechanism so it can be persisted and
    so the mechanism stays picklable (the exec'd shape is re-derived from source).
    """
    return DynamicMechanism(getattr(module, "MECHANISM_SPEC"), getattr(module, "shape"), source=source)


def preflight_mechanism(source: str, sample_df: pd.DataFrame) -> tuple[list[str], DynamicMechanism | None]:
    """Full pipeline: static -> load -> dynamic/physics. Returns (reasons, mechanism|None)."""
    reasons = validate_mechanism_source(source)
    if reasons:
        return reasons, None
    try:
        module = load_mechanism_from_source(source)
    except Exception as exc:
        return [f"Mechanism source failed to import: {type(exc).__name__}: {exc}"], None
    reasons = validate_mechanism_module(module, sample_df)
    if reasons:
        return reasons, None
    return [], build_llm_mechanism(module, source=source)


# Instruction block injected into the OperationAgent prompt so the LLM writes a
# conforming mechanism instead of prose. Kept in sync with the contract above.
MECHANISM_API_SPEC = f"""You are implementing ONE searched yield-stress MECHANISM as executable Python.
Write ONLY these two top-level symbols and nothing else:

MECHANISM_SPEC = {{
    "id": "<short_snake_case_id>",
    "name": "<human name>",
    "paper": "<citation you based this on>",
    "required_columns": ["phi", ...],          # must include "phi"
    "needs_shear_rate": False,                  # True only if the formula needs a flow curve
    "needs_microstructure": False,              # True if it needs PSD/SSA (d50/psd_width/SSA)
    "params": {{"<param>": [<numeric grid>], ...}},  # extra fold-fit params; may be {{}}
}}

def shape(df, params):
    # Return an array g of length len(df): the UNIT-SCALE physics signal.
    # Final prediction is tau = m1 * g, where the HARNESS fits m1 (and searches
    # params) against TRAINING y for you. `params` already contains "phi_m"
    # (harness-supplied, guaranteed > every phi) plus each param you declared.
    # Use ONLY df input columns and params. Do NOT use y.
    # numpy is available as `np` and pandas as `pd` (ALREADY IMPORTED for you).
    ...

Hard rules:
  * Do NOT declare or overwrite "phi_m" — the harness owns the phi_m grid.
  * `shape` must not take a y/target argument and must not read any target,
    label, or *_reference column (that is leakage).
  * No file/network I/O, no eval/exec, no randomness.
  * `np` (numpy) and `pd` (pandas) are pre-injected — you may use them directly;
    an explicit `import numpy as np` is fine too. No other libraries.
  * Keep the grid SMALL: at most {MAX_DECLARED_PARAMS} params, and the PRODUCT of
    all your grid lengths must be <= 150 (the harness also sweeps ~5 phi_m values,
    and the total combinations must stay under {MAX_TOTAL_PARAM_COMBOS}).
  * Output must be finite for phi in [0, phi_m); divide by (phi_m - phi) only with
    a small epsilon floor, e.g. np.maximum(phi_m - phi, 1e-9).
"""
