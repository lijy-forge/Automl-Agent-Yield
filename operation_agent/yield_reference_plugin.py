"""Reference (deterministic) yield candidate plugin.

Per the locked design decision (2026-07-12), the packing/SP feature logic that
used to be hardcoded inside yield_candidate_benchmark is DEMOTED to a reference
plugin. It plays two roles:

  1. The fixed ablation arms of the main paper table (raw / raw+packing /
     raw+SP / raw+packing+SP) and a reproducibility baseline.
  2. A conformance fixture proving the harness <-> plugin contract works,
     against which LLM-generated mechanism-driven plugins are compared.

It intentionally wraps the existing, already-validated fold-local feature code
in yield_candidate_benchmark rather than reimplementing it, so the "reference"
arm is byte-for-byte the same math the benchmark proxy has always used.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from knowledge.yield_candidate_benchmark import (
    _add_packing_features,
    _fit_packing_params,
    _make_model,
)

CANDIDATE_SPEC = {
    "candidate_id": "reference_packing_sp_hgb",
    "feature_families": ["raw", "packing", "sp_saturation"],
    "physics_hparams": {"phi_max_policy": "fold_local"},
    "model_family": "hgb",
    "constraints": ["nonnegative_prediction"],
    "provenance": "deterministic reference; wraps benchmark fold-local packing/sp feature code",
}


def add_features(df: pd.DataFrame, fit_context: dict | None):
    """Fold-local feature construction. See yield_plugin_contract for the API."""
    families = set(CANDIDATE_SPEC["feature_families"])
    is_fit = fit_context is None
    state = {} if fit_context is None else dict(fit_context)

    feat = pd.DataFrame(index=df.index)
    if "raw" in families:
        for col in df.columns:
            feat[col] = df[col]

    if "packing" in families and "phi" in df.columns:
        if is_fit:
            # _fit_packing_params reads only training-fold rows -> fold-local.
            state["packing_params"] = _fit_packing_params(df)
        feat = _add_packing_features(feat, state["packing_params"])

    if "sp_saturation" in families and "sp_percent" in df.columns:
        sp_raw = pd.to_numeric(df["sp_percent"], errors="coerce")
        if is_fit:
            finite = sp_raw[np.isfinite(sp_raw)]
            center = float(np.nanmedian(sp_raw)) if len(finite) else 0.0
            if len(finite):
                q75, q25 = np.nanpercentile(sp_raw, [75, 25])
                scale = float(max(q75 - q25, float(np.nanstd(sp_raw)), 1e-6))
            else:
                scale = 1.0
            state["sp_center"] = center
            state["sp_scale"] = scale
        sp = sp_raw.fillna(state["sp_center"]).astype(float).to_numpy(dtype=float)
        feat["bench_sp_decay"] = np.exp(-np.maximum(sp, 0.0))
        feat["bench_sp_centered"] = (sp - state["sp_center"]) / state["sp_scale"]

    return feat, state


def make_model(random_state: int):
    """Unfitted regressor for the declared model family."""
    return _make_model(CANDIDATE_SPEC["model_family"], random_state, 500)
