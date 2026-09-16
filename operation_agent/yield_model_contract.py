"""Constrained model-code contract for yield free-search.

This is the model-side counterpart to yield_mechanism_contract.py. The LLM is
allowed to write a SMALL model factory, not a full training script:

    MODEL_SPEC = {...}
    def make_estimator(params, random_state):
        return sklearn_like_estimator

The fixed harness still owns data loading, fold-local CV, leakage controls,
anchor evaluation, champion selection, and artifact writing. This gives the
model axis real LLM-written candidates without returning to arbitrary scripts.
"""

from __future__ import annotations

import inspect
import itertools
import os
import types
from typing import Any, Callable

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    ExtraTreesRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
    StackingRegressor,
    VotingRegressor,
)
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, RBF, WhiteKernel
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import BayesianRidge, ElasticNet, Lasso, Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.svm import SVR

from knowledge.yield_candidate_benchmark import LEAKAGE_COLUMNS
from knowledge.yield_joint_search import LatentM1PhysicsLayerRegressor

REQUIRED_SPEC_KEYS = ("id", "name", "params")
MAX_DECLARED_PARAMS = 5
MAX_GRID_PER_PARAM = 8
MAX_TOTAL_PARAM_COMBOS = 36

FORBIDDEN_SOURCE_TOKENS = (
    "import ", "from ", "__import__", "open(", "read_csv", "to_csv",
    "subprocess", "os.", "sys.", "socket", "requests", "urllib",
    "eval(", "exec(", "pickle", "joblib", "pathlib", "shutil",
    "random.random", "np.random", "class ",
)

_MODEL_GLOBALS = {
    "np": np,
    "numpy": np,
    "pd": pd,
    "pandas": pd,
    "make_pipeline": make_pipeline,
    "StandardScaler": StandardScaler,
    "PolynomialFeatures": PolynomialFeatures,
    "Ridge": Ridge,
    "BayesianRidge": BayesianRidge,
    "Lasso": Lasso,
    "ElasticNet": ElasticNet,
    "RandomForestRegressor": RandomForestRegressor,
    "ExtraTreesRegressor": ExtraTreesRegressor,
    "HistGradientBoostingRegressor": HistGradientBoostingRegressor,
    "VotingRegressor": VotingRegressor,
    "StackingRegressor": StackingRegressor,
    "SVR": SVR,
    "KernelRidge": KernelRidge,
    "GaussianProcessRegressor": GaussianProcessRegressor,
    "ConstantKernel": ConstantKernel,
    "RBF": RBF,
    "Matern": Matern,
    "WhiteKernel": WhiteKernel,
    "MLPRegressor": MLPRegressor,
    "LatentM1PhysicsLayerRegressor": LatentM1PhysicsLayerRegressor,
}


def _hist_gradient_boosting_regressor_compat(*args, **kwargs):
    """Small sklearn-version compatibility shim for LLM-written factories."""
    if "monotonic_constraints" in kwargs and "monotonic_cst" not in kwargs:
        kwargs["monotonic_cst"] = kwargs.pop("monotonic_constraints")
    elif "monotonic_constraints" in kwargs:
        kwargs.pop("monotonic_constraints")
    if kwargs.get("loss") in {"least_squares", "l2_loss"}:
        kwargs["loss"] = "squared_error"
    return HistGradientBoostingRegressor(*args, **kwargs)


_MODEL_GLOBALS["HistGradientBoostingRegressor"] = _hist_gradient_boosting_regressor_compat


def _json_id(value: Any) -> str:
    text = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    text = "".join(ch for ch in text if ch.isalnum() or ch == "_").strip("_")
    return text or "llm_model"


def _llm_model_config_cap() -> int:
    mode = str(os.environ.get("YIELD_SEARCH_BUDGET") or "normal").strip().lower()
    defaults = {
        "quick": 1,
        "normal": 3,
        "full": 6,
        "unlimited": MAX_TOTAL_PARAM_COMBOS,
    }
    default = defaults.get(mode, defaults["normal"])
    try:
        return max(1, int(os.environ.get("YIELD_MAX_LLM_MODEL_CONFIGS", str(default))))
    except Exception:
        return int(default)


def _budget_estimator_cap(kind: str) -> int:
    mode = str(os.environ.get("YIELD_SEARCH_BUDGET") or "normal").strip().lower()
    defaults = {
        "quick": {"max_iter": 120, "n_estimators": 80},
        "normal": {"max_iter": 220, "n_estimators": 180},
        "full": {"max_iter": 600, "n_estimators": 500},
        "unlimited": {"max_iter": 0, "n_estimators": 0},
    }.get(mode, {"max_iter": 220, "n_estimators": 180})
    env_name = "YIELD_MAX_ESTIMATOR_ITER" if kind == "max_iter" else "YIELD_MAX_ESTIMATOR_TREES"
    try:
        return max(0, int(os.environ.get(env_name, str(defaults.get(kind, 0)))))
    except Exception:
        return int(defaults.get(kind, 0))


def _cap_estimator_complexity(est):
    if not hasattr(est, "get_params") or not hasattr(est, "set_params"):
        return est
    caps = {
        "max_iter": _budget_estimator_cap("max_iter"),
        "n_estimators": _budget_estimator_cap("n_estimators"),
    }
    updates: dict[str, Any] = {}
    try:
        params = est.get_params(deep=True)
    except Exception:
        return est
    for key, value in params.items():
        tail = str(key).split("__")[-1]
        cap = caps.get(tail, 0)
        if cap <= 0 or isinstance(value, bool):
            continue
        try:
            if int(value) > cap:
                updates[key] = cap
        except Exception:
            continue
    if updates:
        try:
            est.set_params(**updates)
        except Exception:
            pass
    return est


def validate_model_source(source: str) -> list[str]:
    """Static preflight before executing model source."""
    reasons: list[str] = []
    text = str(source or "")
    if not text.strip():
        return ["Model source is empty."]
    if "MODEL_SPEC" not in text:
        reasons.append("Model must define a top-level MODEL_SPEC dict.")
    if "def make_estimator" not in text:
        reasons.append("Model must define make_estimator(params, random_state).")
    lowered = text.lower()
    for token in FORBIDDEN_SOURCE_TOKENS:
        if token in lowered:
            reasons.append(f"Model source contains a forbidden construct: '{token}'.")
    for col in sorted(LEAKAGE_COLUMNS):
        if col in ("source",):
            continue
        if f'"{col}"' in text or f"'{col}'" in text:
            reasons.append(f"Model references leakage/reference column '{col}'.")
    return reasons


def load_model_from_source(source: str, module_name: str = "yield_searched_model") -> types.ModuleType:
    module = types.ModuleType(module_name)
    module.__dict__["__name__"] = module_name
    module.__dict__.update(_MODEL_GLOBALS)
    exec(compile(source, f"<{module_name}>", "exec"), module.__dict__)  # noqa: S102 - vetted above
    return module


def _validate_spec(spec: Any) -> list[str]:
    reasons: list[str] = []
    if not isinstance(spec, dict):
        return ["MODEL_SPEC must be a dict."]
    for key in REQUIRED_SPEC_KEYS:
        if key not in spec:
            reasons.append(f"MODEL_SPEC is missing required key '{key}'.")
    if not str(spec.get("id", "")).strip():
        reasons.append("MODEL_SPEC['id'] must be a non-empty string.")
    label = f"{spec.get('id', '')} {spec.get('name', '')}".lower().replace("-", "_").replace(" ", "_")
    if "group_lasso" in label:
        reasons.append(
            "Do not label a sklearn ElasticNet/Lasso model as group_lasso. "
            "Use a truthful id such as sparse_polynomial_elasticnet unless a true group penalty is implemented."
        )
    params = spec.get("params")
    if not isinstance(params, dict):
        reasons.append("MODEL_SPEC['params'] must be a dict (may be empty {}).")
        return reasons
    if len(params) > MAX_DECLARED_PARAMS:
        reasons.append(f"Too many declared params ({len(params)} > {MAX_DECLARED_PARAMS}).")
    total = 1
    for name, grid in params.items():
        if not isinstance(grid, (list, tuple)) or not grid:
            reasons.append(f"params['{name}'] must be a non-empty list.")
            continue
        if len(grid) > MAX_GRID_PER_PARAM:
            reasons.append(f"params['{name}'] grid too long ({len(grid)} > {MAX_GRID_PER_PARAM}).")
        total *= max(1, len(grid))
    if total > MAX_TOTAL_PARAM_COMBOS:
        reasons.append(f"param grid too large ({total} combos > {MAX_TOTAL_PARAM_COMBOS}); shrink grids.")
    return reasons


def _model_kind(spec: dict[str, Any] | None) -> str:
    if not isinstance(spec, dict):
        return "simple_estimator"
    return str(
        spec.get("model_kind")
        or spec.get("contract")
        or spec.get("implementation_kind")
        or "simple_estimator"
    ).strip().lower()


def _is_latent_physics_spec(spec: dict[str, Any] | None) -> bool:
    kind = _model_kind(spec)
    if kind in {
        "latent_physics_architecture",
        "complex_latent_physics",
        "hidden_parameter_physics",
        "latent_m1_eff",
    }:
        return True
    text = f"{(spec or {}).get('id', '')} {(spec or {}).get('name', '')}".lower()
    return "latent" in text and ("physics" in text or "m1" in text)


def _validate_factory_signature(factory: Any) -> list[str]:
    if not callable(factory):
        return ["make_estimator must be callable."]
    try:
        sig = inspect.signature(factory)
    except (TypeError, ValueError):
        return []
    params = list(sig.parameters.values())
    positional = [p for p in params if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)]
    if len(positional) != 2:
        return ["make_estimator must take exactly two positional args (params, random_state)."]
    banned = {"x", "df", "data", "y", "target", "label", "anchor", "test"}
    reasons = []
    for p in params:
        if p.name.lower() in banned:
            reasons.append(
                f"make_estimator must not take '{p.name}' (the harness supplies data via estimator.fit only)."
            )
    return reasons


def _walk_estimator_tree(est: Any):
    yield est
    for attr in ("steps", "estimators", "transformer_list"):
        items = getattr(est, attr, None)
        if not items:
            continue
        for item in items:
            child = item[-1] if isinstance(item, (list, tuple)) and item else item
            if child is not None:
                yield from _walk_estimator_tree(child)


def _validate_estimator_structure(est: Any, spec: dict[str, Any]) -> list[str]:
    """Generated models should be small structures, not fixed-baseline wrappers."""
    if _is_latent_physics_spec(spec):
        if not isinstance(est, LatentM1PhysicsLayerRegressor):
            return [
                "latent_physics_architecture models must return "
                "LatentM1PhysicsLayerRegressor(...) from make_estimator."
            ]
        return []
    names = {type(obj).__name__ for obj in _walk_estimator_tree(est)}
    structural_names = {
        "PolynomialFeatures",
        "VotingRegressor",
        "StackingRegressor",
        "FeatureUnion",
        "ColumnTransformer",
        "TransformedTargetRegressor",
    }
    regressor_names = {
        "Ridge",
        "BayesianRidge",
        "Lasso",
        "ElasticNet",
        "KernelRidge",
        "RandomForestRegressor",
        "ExtraTreesRegressor",
        "HistGradientBoostingRegressor",
        "SVR",
        "GaussianProcessRegressor",
        "MLPRegressor",
    }
    plain_wrapper_names = regressor_names - {"KernelRidge"}
    has_structure = bool(names & structural_names)
    regressor_count = len(names & regressor_names)
    if has_structure or regressor_count >= 2:
        structure_reasons = []
    else:
        structure_reasons = []
        model_id = str(spec.get("id") or "").lower()
        if any(token in model_id for token in ("baseline", "smoke")):
            return []
        if names & plain_wrapper_names:
            structure_reasons.append(
                "Generated model is only a simple single-estimator wrapper. "
                "Use a bounded structure such as PolynomialFeatures+regularized regression, "
                "or VotingRegressor/StackingRegressor with two complementary regressors."
            )
    label = f"{spec.get('id', '')} {spec.get('name', '')}".lower().replace("-", "_").replace(" ", "_")
    if "monotonic" in label or "_mono_" in f"_{label}_":
        try:
            params = est.get_params(deep=True) if hasattr(est, "get_params") else {}
        except Exception:
            params = {}
        has_monotonic_constraint = any(
            str(key).split("__")[-1] == "monotonic_cst" and value is not None
            for key, value in params.items()
        )
        if not has_monotonic_constraint:
            structure_reasons.append(
                "Model id/name claims monotonic behavior, but the estimator has no monotonic_cst constraint. "
                "Rename the model or implement a supported monotonic estimator."
            )
    return structure_reasons


def _first_params(spec: dict[str, Any]) -> dict[str, Any]:
    return {str(k): list(v)[0] for k, v in (spec.get("params") or {}).items()}


def _param_combo_key(combo: dict[str, Any]) -> str:
    return repr(sorted((str(k), repr(v)) for k, v in combo.items()))


def _sample_param_grid(names: list[str], grids: list[list[Any]], cap: int) -> list[dict[str, Any]]:
    """Small coverage-oriented grid sampler.

    Taking the first N Cartesian products can hide later categorical options
    such as latent_backend="kernel_ridge". Start from the default combo, then
    vary one parameter at a time before falling back to remaining products.
    """
    if not names or not grids:
        return [{}]
    cap = max(1, int(cap))
    base = {n: values[0] for n, values in zip(names, grids)}
    selected: list[dict[str, Any]] = []
    seen: set[str] = set()

    def add(combo: dict[str, Any]) -> bool:
        key = _param_combo_key(combo)
        if key in seen:
            return False
        selected.append(dict(combo))
        seen.add(key)
        return len(selected) >= cap

    if add(base):
        return selected
    max_len = max(len(values) for values in grids)
    for value_idx in range(1, max_len):
        for name, values in zip(names, grids):
            if value_idx >= len(values):
                continue
            combo = dict(base)
            combo[name] = values[value_idx]
            if add(combo):
                return selected
    for combo_values in itertools.product(*grids):
        combo = {n: v for n, v in zip(names, combo_values)}
        if add(combo):
            return selected
    return selected


def validate_model_module(module: types.ModuleType, sample_df: pd.DataFrame, y_sample: np.ndarray) -> list[str]:
    """Dynamic preflight: factory builds a sklearn-like estimator and it fits/predicts."""
    reasons: list[str] = []
    spec = getattr(module, "MODEL_SPEC", None)
    reasons.extend(_validate_spec(spec))
    factory = getattr(module, "make_estimator", None)
    reasons.extend(_validate_factory_signature(factory))
    if reasons:
        return reasons

    X = sample_df.select_dtypes(include=[np.number]).copy()
    if X.empty:
        return ["sample_df has no numeric columns for model preflight."]
    y = np.asarray(y_sample, dtype=float)
    if len(y) != len(X):
        return [f"y_sample length {len(y)} does not match X length {len(X)}."]

    try:
        est = _cap_estimator_complexity(factory(_first_params(spec), 0))
    except Exception as exc:
        return [f"make_estimator(params, random_state) raised {type(exc).__name__}: {exc}"]
    if not hasattr(est, "fit") or not hasattr(est, "predict"):
        return ["make_estimator must return an object with fit(X, y) and predict(X)."]
    reasons.extend(_validate_estimator_structure(est, spec))
    if reasons:
        return reasons
    try:
        if _is_latent_physics_spec(spec):
            physics_shape = np.ones(len(X), dtype=float)
            est.fit(X, y, physics_shape=physics_shape)
            latent = np.asarray(est.predict_latent(X), dtype=float)
            if latent.shape[0] != len(X):
                reasons.append(f"predict_latent returned length {latent.shape[0]}, expected {len(X)}.")
            pred = np.asarray(est.predict(X, physics_shape=physics_shape), dtype=float)
        else:
            est.fit(X, y)
            pred = np.asarray(est.predict(X), dtype=float)
    except Exception as exc:
        return [f"estimator fit/predict smoke test failed: {type(exc).__name__}: {exc}"]
    if pred.shape[0] != len(X):
        reasons.append(f"predict returned length {pred.shape[0]}, expected {len(X)}.")
    finite_frac = float(np.mean(np.isfinite(pred))) if pred.size else 0.0
    if finite_frac < 1.0:
        reasons.append(f"predict produced non-finite values ({finite_frac:.0%} finite).")
    return reasons


class DynamicModel:
    """LLM-written model factory with a bounded hyperparameter grid."""

    def __init__(self, spec: dict[str, Any], factory_fn: Callable[[dict[str, Any], int], Any],
                 source: str | None = None):
        self.spec = dict(spec or {})
        self.id = _json_id(self.spec.get("id"))
        self.name = str(self.spec.get("name", self.id))
        self.model_kind = _model_kind(self.spec)
        self.source = source
        self._factory_fn = factory_fn
        self._param_grids = {str(k): list(v) for k, v in (self.spec.get("params", {}) or {}).items()}

    @staticmethod
    def _factory_from_source(source: str):
        ns = dict(_MODEL_GLOBALS)
        exec(compile(source, "<model_source>", "exec"), ns)  # noqa: S102 - source already preflight-vetted
        return ns.get("make_estimator")

    def __getstate__(self):
        state = self.__dict__.copy()
        if self.source:
            state["_factory_fn"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        if self._factory_fn is None and getattr(self, "source", None):
            self._factory_fn = self._factory_from_source(self.source)

    def hparam_grid(self) -> list[dict[str, Any]]:
        names = list(self._param_grids.keys())
        grids = [self._param_grids[n] for n in names]
        if not grids:
            return [{}]
        return _sample_param_grid(names, grids, _llm_model_config_cap())

    def make_estimator(self, random_state: int, hparams: dict[str, Any] | None = None):
        return _cap_estimator_complexity(self._factory_fn(dict(hparams or {}), int(random_state)))

    def is_latent_physics_model(self) -> bool:
        return _is_latent_physics_spec(self.spec)


def build_llm_model(module: types.ModuleType, source: str | None = None) -> DynamicModel:
    return DynamicModel(getattr(module, "MODEL_SPEC"), getattr(module, "make_estimator"), source=source)


def preflight_model(source: str, sample_df: pd.DataFrame, y_sample: np.ndarray) -> tuple[list[str], DynamicModel | None]:
    reasons = validate_model_source(source)
    if reasons:
        return reasons, None
    try:
        module = load_model_from_source(source)
    except Exception as exc:
        return [f"Model source failed to import: {type(exc).__name__}: {exc}"], None
    reasons = validate_model_module(module, sample_df, y_sample)
    if reasons:
        return reasons, None
    return [], build_llm_model(module, source=source)


MODEL_API_SPEC = f"""You are implementing ONE searched yield-stress MODEL as executable Python.
Write ONLY these two top-level symbols and nothing else:

MODEL_SPEC = {{
    "id": "<short_snake_case_id>",
    "name": "<human name>",
    "params": {{"<param>": [<small grid values>], ...}},  # may be {{}}
}}

def make_estimator(params, random_state):
    # Return a sklearn-like regressor with fit(X, y) and predict(X).
    # The fixed harness owns data loading, CV, anchor evaluation, artifacts, and
    # leakage controls. You only build the estimator object.
    ...

Hard rules:
  * Do NOT load files, inspect paths, call network, run CV, or evaluate metrics.
  * Do NOT read target/test/anchor/reference columns in this source.
  * Do NOT define a training loop that accesses external data. Return an estimator.
  * Do NOT define custom classes. Use the sklearn classes already exposed below
    so the saved champion remains joblib/predict.py compatible.
  * No imports. The following are already available: np, pd, make_pipeline,
    StandardScaler, PolynomialFeatures, Ridge, Lasso, ElasticNet,
    BayesianRidge, RandomForestRegressor, ExtraTreesRegressor, HistGradientBoostingRegressor,
    VotingRegressor, StackingRegressor, SVR, KernelRidge,
    GaussianProcessRegressor, ConstantKernel, RBF, Matern, WhiteKernel,
    MLPRegressor.
  * Do NOT return only StandardScaler + one ordinary regressor. Generated
    models must test a small structure that is different from the fixed model
    floor, such as PolynomialFeatures + Ridge/KernelRidge, or a
    VotingRegressor/StackingRegressor combining two complementary regressors.
  * Read current_data_model_guidance when it is present in the model brief. If
    the data is small/high-dimensional, keep the estimator low-capacity and
    strongly regularized. If phi is constant, do not rely mainly on phi-only
    packing variation.
  * The MODEL_SPEC id/name must truthfully describe the estimator. Do NOT call
    ElasticNet/Lasso "group_lasso" unless a true group penalty is implemented,
    and do NOT call a model "monotonic" unless it uses monotonic_cst or another
    explicit monotonic constraint.
  * Keep params small: at most {MAX_DECLARED_PARAMS} params. Prefer 3-12 total
    hyperparameter combinations; hard maximum is {MAX_TOTAL_PARAM_COMBOS}.
  * Prefer deterministic estimators and pass random_state where supported.
  * Use current sklearn argument names. For HistGradientBoostingRegressor use
    loss="squared_error" and monotonic_cst=...; do not use l2_loss,
    least_squares, or monotonic_constraints.
"""


COMPLEX_MODEL_API_SPEC = f"""You are implementing ONE searched COMPLEX yield-stress MODEL as executable Python.
This contract is for a controlled latent-physics architecture, not a full
training script. Write ONLY these two top-level symbols and nothing else:

MODEL_SPEC = {{
    "id": "<short_snake_case_id>",
    "name": "<human name>",
    "model_kind": "latent_physics_architecture",
    "params": {{
        "latent_backend": ["ridge", "kernel_ridge", "mlp"],
        "latent_arch": ["mlp2_hidden16", "mlp3_hidden16", "branched_hidden16"],
        "alpha": [0.0001, 0.001],
        "learning_rate": [0.001, 0.01],
        "max_iter": [200]
    }},
}}

def make_estimator(params, random_state):
    # Return a controlled latent m1_eff model. The harness fits it only inside
    # train folds using physics_shape from the selected mechanism, then predicts
    # tau = m1_eff(X) * physics_shape(X).
    return LatentM1PhysicsLayerRegressor(...)

Hard rules:
  * Do NOT load files, inspect paths, call network, run CV, or evaluate metrics.
  * Do NOT read target/test/anchor/reference columns in this source.
  * Do NOT define a training loop that accesses external data.
  * Do NOT define custom classes. Use LatentM1PhysicsLayerRegressor only.
  * No imports. LatentM1PhysicsLayerRegressor is already available.
  * MODEL_SPEC['model_kind'] must be "latent_physics_architecture".
  * latent_backend may be "ridge", "kernel_ridge", "mlp", or "auto".
    Prefer "ridge" or "kernel_ridge" for small high-dimensional data; use
    "mlp" only as a bounded comparison when sample size is sufficient.
  * latent_arch must use only: "mlp2_hidden16", "mlp3_hidden16",
    "branched_hidden16", or "mlp2_hidden64". Prefer branched_hidden16 when the
    data has composition/thermal/mixing feature groups.
  * Keep params small: at most {MAX_DECLARED_PARAMS} params. Prefer 3-12 total
    hyperparameter combinations; hard maximum is {MAX_TOTAL_PARAM_COMBOS}.
  * The returned estimator must be LatentM1PhysicsLayerRegressor with arch,
    latent_backend, alpha, learning_rate, max_iter, and random_state taken from
    params/defaults.
"""
