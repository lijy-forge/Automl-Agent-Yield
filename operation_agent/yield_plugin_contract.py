"""Plugin API contract for the deterministic yield AutoML harness.

Paper framing (locked 2026-07-12): the LLM does NOT write a full training
script, and it does NOT emit JSON only. It writes a small, *constrained Python
candidate plugin*. A fixed deterministic harness owns data loading, leakage
removal, OOF cross-validation, anchor validation, baselines, mechanism
ablation, artifacts, and guardrails. The plugin only expresses a single point
in the mechanism-aware AutoML search space:

    axis 1  feature families      -> add_features(...)
    axis 2  physics hyperparams   -> CANDIDATE_SPEC["physics_hparams"] (fold-local only)
    axis 3  model family + constr -> make_model(...) + CANDIDATE_SPEC["constraints"]

This removes the runnability bottleneck (the LLM never re-writes CV / anchor /
baseline / artifact boilerplate) while keeping the "LLM writes Python" story:
add_features is exactly the retrieval-grounded mechanism -> feature translation
that is the paper's core contribution.

Plugin module MUST define three top-level symbols:

    CANDIDATE_SPEC : dict
        Auditable declaration of the search-space point. Required keys:
          - "candidate_id":     str
          - "feature_families": list[str]  (subset of FEATURE_FAMILIES)
          - "physics_hparams":  dict       (must set "phi_max_policy": "fold_local")
          - "model_family":     str        (one of MODEL_FAMILIES)
          - "constraints":      list[str]  (subset of CONSTRAINTS)

    def add_features(df, fit_context):
        Return (feature_df, feature_state).
        df           : pandas.DataFrame of harness-approved INPUT columns only
                       (the harness has already removed leakage/reference cols).
        fit_context  : None => FIT mode (a training fold): the plugin may compute
                       statistics from df (e.g. fold-local phi_max) and MUST
                       return them in feature_state. A dict (possibly empty) =>
                       TRANSFORM mode (validation / anchor / full-fit reuse): the
                       plugin MUST reuse fit_context and MUST NOT recompute
                       statistics from df. This is the anti-leakage contract.
                       (None vs dict is used instead of empty-vs-nonempty so that
                       stateless raw-only plugins are unambiguous.)
        feature_df   : pandas.DataFrame indexed like df, numeric engineered cols.
        feature_state: JSON-serializable dict of fitted params (FIT mode);
                       echo fit_context back in TRANSFORM mode.

    def make_model(random_state):
        Return an UNFITTED scikit-learn-compatible regressor (fit/predict).
        The harness handles CV, prediction clipping for nonnegative_prediction,
        and all persistence.

Hard rules enforced by validation below:
  - phi_max / phi0 / K_sp must be fit fold-locally, NEVER from the anchor.
  - The plugin must not import gradient-boosting wheels directly
    (lightgbm/xgboost/catboost) or touch the filesystem / network / anchor.
"""

from __future__ import annotations

import types
from typing import Any

from knowledge.yield_candidate_benchmark import LEAKAGE_COLUMNS

FEATURE_FAMILIES = {"raw", "packing", "yodel_network", "sp_saturation", "water_film"}
MODEL_FAMILIES = {"ridge", "kernel", "mlp", "rf", "hgb"}
CONSTRAINTS = {"nonnegative_prediction", "sp_monotonic"}
PHI_MAX_POLICIES = {"fold_local"}

REQUIRED_CALLABLES = ("add_features", "make_model")
REQUIRED_SPEC_KEYS = ("candidate_id", "feature_families", "physics_hparams", "model_family", "constraints")

# Static-source red flags. The plugin runs in-process, so we refuse obvious
# leakage / escape / non-deterministic constructs before importing it.
FORBIDDEN_SOURCE_TOKENS = (
    "import lightgbm",
    "import xgboost",
    "import catboost",
    "from lightgbm",
    "from xgboost",
    "from catboost",
    "subprocess",
    "os.system",
    "socket",
    "requests",
    "urllib",
    "open(",
    "read_csv",
    "to_csv",
    "eval(",
    "exec(",
    "__import__",
    "random.random",
    "np.random.rand",
    "np.random.random",
)


def validate_plugin_source(source: str) -> list[str]:
    """Static preflight on plugin source text (before it is imported)."""
    reasons: list[str] = []
    text = str(source or "")
    if not text.strip():
        return ["Plugin source is empty."]
    if "CANDIDATE_SPEC" not in text:
        reasons.append("Plugin must define a top-level CANDIDATE_SPEC dict.")
    if "def add_features" not in text:
        reasons.append("Plugin must define add_features(df, fit_context).")
    if "def make_model" not in text:
        reasons.append("Plugin must define make_model(random_state).")
    lowered = text.lower()
    for token in FORBIDDEN_SOURCE_TOKENS:
        if token in lowered:
            reasons.append(f"Plugin source contains a forbidden construct: '{token}'.")
    for col in sorted(LEAKAGE_COLUMNS):
        # Referencing a leakage/target/reference column name literally is a
        # strong signal of target leakage; the harness never passes these in.
        if col in ("target", "label", "source"):
            continue  # too generic to flag on substring
        if f'"{col}"' in text or f"'{col}'" in text:
            reasons.append(f"Plugin references leakage/reference column '{col}'.")
    return reasons


def load_plugin_from_source(source: str, module_name: str = "yield_candidate_plugin") -> types.ModuleType:
    """Execute plugin source into an in-memory module and return it."""
    module = types.ModuleType(module_name)
    module.__dict__["__name__"] = module_name
    exec(compile(source, f"<{module_name}>", "exec"), module.__dict__)  # noqa: S102 - vetted by validate_plugin_source
    return module


def validate_plugin_module(module: types.ModuleType) -> list[str]:
    """Dynamic preflight on an imported plugin module."""
    reasons: list[str] = []
    for name in REQUIRED_CALLABLES:
        if not callable(getattr(module, name, None)):
            reasons.append(f"Plugin is missing a callable '{name}'.")
    spec = getattr(module, "CANDIDATE_SPEC", None)
    if not isinstance(spec, dict):
        reasons.append("CANDIDATE_SPEC must be a dict.")
        return reasons
    for key in REQUIRED_SPEC_KEYS:
        if key not in spec:
            reasons.append(f"CANDIDATE_SPEC is missing required key '{key}'.")

    families = spec.get("feature_families")
    if not isinstance(families, list) or not families:
        reasons.append("CANDIDATE_SPEC['feature_families'] must be a non-empty list.")
    else:
        bad = [f for f in families if f not in FEATURE_FAMILIES]
        if bad:
            reasons.append(f"Unknown feature_families {bad}; allowed: {sorted(FEATURE_FAMILIES)}.")

    model_family = spec.get("model_family")
    if model_family not in MODEL_FAMILIES:
        reasons.append(f"model_family '{model_family}' not in allowed {sorted(MODEL_FAMILIES)}.")

    constraints = spec.get("constraints", [])
    if not isinstance(constraints, list):
        reasons.append("CANDIDATE_SPEC['constraints'] must be a list.")
    else:
        bad = [c for c in constraints if c not in CONSTRAINTS]
        if bad:
            reasons.append(f"Unknown constraints {bad}; allowed: {sorted(CONSTRAINTS)}.")

    physics = spec.get("physics_hparams")
    if not isinstance(physics, dict):
        reasons.append("CANDIDATE_SPEC['physics_hparams'] must be a dict.")
    else:
        policy = str(physics.get("phi_max_policy") or "")
        if policy not in PHI_MAX_POLICIES:
            reasons.append(
                f"physics_hparams['phi_max_policy'] must be one of {sorted(PHI_MAX_POLICIES)} "
                "(physics params must be fit fold-locally, never from the anchor)."
            )
    return reasons


# Distinctive engineered-feature tokens that signal a mechanism family is
# actually being BUILT (not merely a raw input column being read). Raw input
# names like "sp_percent"/"phi"/"w_b" are intentionally excluded so reading them
# under the "raw" family does not trigger a false positive.
# Tokens must be distinctive of a BUILT engineered feature, not words that also
# appear in comments (e.g. "yodel", which describes a packing feature) or in the
# spec itself (e.g. "phi_max_policy"). Bare mechanism NAMES are deliberately
# excluded to avoid false positives; YODEL packing proxies count as "packing".
FAMILY_FEATURE_SIGNALS = {
    "sp_saturation": ("sp_decay", "sp_sat", "sp_centered", "sp_adsorp", "langmuir_sp"),
    "packing": ("packing_gap", "packing_index", "phi_over_gap", "phi_m_eff", "free_volume", "phi_max_eff"),
    "yodel_network": ("percolation", "contact_network", "contact_number", "network_index"),
    "water_film": ("water_film", "film_thickness", "wft_", "lubric"),
}


def validate_plugin_feature_declaration(source: str, spec: dict) -> list[str]:
    """Audit that CANDIDATE_SPEC.feature_families matches the features actually built.

    Reduces false positives by keying on distinctive engineered-feature tokens
    rather than raw input column names. A mismatch is a preflight failure so the
    report's "which mechanism families were used" stays auditable.
    """
    reasons: list[str] = []
    declared = {str(x) for x in (spec.get("feature_families") or [])}
    low = str(source or "").lower()
    for family, tokens in FAMILY_FEATURE_SIGNALS.items():
        if family in declared:
            continue
        hits = [t for t in tokens if t in low]
        if hits:
            reasons.append(
                f"Source builds {family}-style features {hits} but '{family}' is not declared "
                f"in feature_families {sorted(declared)}. Declare it or remove those features."
            )
    return reasons


# Instruction block injected into the OperationAgent prompt so the LLM emits a
# conforming plugin instead of a full script. Kept in sync with the API above.
PLUGIN_API_SPEC = f"""You are writing a CONSTRAINED Python candidate plugin, NOT a full training script.
A fixed deterministic harness already owns data loading, leakage removal, 5-fold OOF,
anchor validation, baselines, mechanism ablation, artifacts, and guardrails.
Write ONLY these three top-level symbols and nothing else:

CANDIDATE_SPEC = {{
    "candidate_id": "<short_snake_case_id>",
    "feature_families": [subset of {sorted(FEATURE_FAMILIES)}],
    "physics_hparams": {{"phi_max_policy": "fold_local", ...}},   # fold-local ONLY
    "model_family": "<one of {sorted(MODEL_FAMILIES)}>",
    "constraints": [subset of {sorted(CONSTRAINTS)}],
}}

def add_features(df, fit_context):
    # df: harness-approved INPUT columns only (leakage already removed).
    # fit_context is None -> FIT mode (training fold): compute stats from df,
    #   return them in feature_state.
    # fit_context is a dict -> TRANSFORM mode: reuse fit_context, do NOT recompute.
    # Return (feature_df, feature_state).
    ...

def make_model(random_state):
    # Return an unfitted scikit-learn-compatible regressor.
    ...

Rules: no file/network I/O; do not import lightgbm/xgboost/catboost; never read the
anchor or any target/reference column; physics params must be fit fold-locally.
Allowed libraries: numpy, pandas, scikit-learn.
"""
