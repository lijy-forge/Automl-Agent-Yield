"""Layer 1 — Mechanism library for direction-④ yield-stress AutoML.

Each mechanism is a parameterized physics function with metadata. Design rules
(see docs/yield_direction4_mechanism_model_search_design.md):

  * Global physics params (phi_m, phi_c) are fit FOLD-LOCALLY against TRAINING
    y only, via a deterministic grid + linear scale (NO nonlinear solver, NO
    anchor, NO per-sample inversion of y).
  * `base_predict` gives the physics tau (used by residual / pure-mechanism modes).
  * `features` gives mechanism-derived columns (used by mechanism-feature mode).
  * `physics_layer` computes tau from a per-sample hidden variable phi_m (used by
    the hidden-physical-layer / PINN mode) — the hidden var is predicted from X
    by a sub-model, never inverted from y.
  * `data_adequacy` drives the mechanism-admissibility screening (reject HB w/o
    shear-rate, degrade YODEL when microstructure columns are absent).

Fitting global params against TRAINING y is ordinary supervised learning, NOT
leakage. Leakage would be using a sample's own y (or anchor/test y) to build its
feature — which this module never does.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import pandas as pd

_EPS = 1e-9


def _col(df: pd.DataFrame, name: str, default: float = 0.0) -> np.ndarray:
    if name in df.columns:
        return pd.to_numeric(df[name], errors="coerce").to_numpy(dtype=float)
    return np.full(len(df), default, dtype=float)


def _screen_columns(name, required_columns, needs_shear_rate, needs_microstructure, columns) -> dict[str, Any]:
    """Shared data-adequacy screen (used by both Mechanism and DynamicMechanism)."""
    cols = set(str(c) for c in columns)
    missing = [c for c in required_columns if c not in cols]
    if needs_shear_rate and not ({"shear_rate", "flow_curve", "gamma_dot"} & cols):
        return {"status": "rejected",
                "reason": f"{name} needs a shear-rate/flow curve, absent in {sorted(cols)}."}
    if missing:
        return {"status": "rejected",
                "reason": f"{name} requires {missing} which are absent."}
    if needs_microstructure and not ({"d50_um", "psd_width", "specific_surface_m2kg"} & cols):
        return {"status": "degraded",
                "reason": (f"{name} lacks PSD/SSA; phi_m is fit as an EFFECTIVE global "
                           "parameter (composition-only proxy, not measured microstructure).")}
    return {"status": "supported", "reason": f"{name} inputs present."}


def _linear_scale(y: np.ndarray, g: np.ndarray) -> float:
    """Closed-form best m1 for y ~ m1 * g (nonneg)."""
    denom = float(np.sum(g * g))
    if denom <= _EPS:
        return 0.0
    return max(0.0, float(np.sum(y * g) / denom))


@dataclass
class Mechanism:
    id: str
    name: str
    paper: str
    required_columns: tuple[str, ...]
    kind: str  # "packing" | "constitutive" | ...
    needs_shear_rate: bool = False
    needs_microstructure: bool = False
    phi_c_grid: tuple[float, ...] = (0.15, 0.20, 0.25, 0.30)
    _shape: str = "yodel"  # "yodel" or "lian"

    def data_adequacy(self, columns) -> dict[str, Any]:
        return _screen_columns(self.name, self.required_columns, self.needs_shear_rate,
                               self.needs_microstructure, columns)

    # ---- physics core ----
    def _g(self, phi: np.ndarray, phi_m: float, phi_c: float) -> np.ndarray:
        """Unit-scale physics shape g(phi) so that tau = m1 * g."""
        gap = np.maximum(phi_m - phi, _EPS)
        if self._shape == "lian":
            return np.power(np.clip(phi, 0.0, None), 3) / (phi_m * gap)
        active = np.maximum(phi - phi_c, 0.0)
        return phi * active * active / (phi_m * gap)

    def fit_global(self, df: pd.DataFrame, y: np.ndarray) -> dict[str, Any]:
        """Fold-local grid fit of (phi_m, phi_c) + linear m1 against TRAINING y."""
        phi = _col(df, "phi")
        finite = phi[np.isfinite(phi)]
        max_phi = float(np.max(finite)) if len(finite) else 0.5
        phi_m_grid = [round(max_phi + d, 4) for d in (0.02, 0.05, 0.08, 0.12, 0.18)]
        phi_m_grid = [min(p, 0.95) for p in phi_m_grid]
        phi_c_grid = self.phi_c_grid if self._shape == "yodel" else (0.0,)
        best = None
        yv = np.asarray(y, dtype=float)
        for phi_m in phi_m_grid:
            for phi_c in phi_c_grid:
                g = self._g(phi, phi_m, phi_c)
                m1 = _linear_scale(yv, g)
                resid = yv - m1 * g
                sse = float(np.mean(resid * resid))
                if best is None or sse < best["sse"]:
                    best = {"phi_m": float(phi_m), "phi_c": float(phi_c), "m1": float(m1), "sse": sse}
        best.pop("sse", None)
        return best

    def base_predict(self, df: pd.DataFrame, params: dict[str, Any]) -> np.ndarray:
        phi = _col(df, "phi")
        g = self._g(phi, float(params["phi_m"]), float(params.get("phi_c", 0.0)))
        return np.clip(float(params["m1"]) * g, 0.0, None)

    def physics_layer(self, df: pd.DataFrame, phi_m_sample: np.ndarray, params: dict[str, Any]) -> np.ndarray:
        """tau from a PER-SAMPLE hidden phi_m (for the PINN mode). phi_m_sample must be > phi."""
        phi = _col(df, "phi")
        phi_m = np.maximum(np.asarray(phi_m_sample, dtype=float), phi + 1e-3)
        gap = np.maximum(phi_m - phi, _EPS)
        phi_c = float(params.get("phi_c", 0.0))
        if self._shape == "lian":
            g = np.power(np.clip(phi, 0.0, None), 3) / (phi_m * gap)
        else:
            active = np.maximum(phi - phi_c, 0.0)
            g = phi * active * active / (phi_m * gap)
        return np.clip(float(params.get("m1", 1.0)) * g, 0.0, None)

    def features(self, df: pd.DataFrame, params: dict[str, Any]) -> pd.DataFrame:
        """Mechanism-derived feature columns (fold-local params passed in)."""
        phi = _col(df, "phi")
        phi_m = float(params["phi_m"])
        gap = np.maximum(phi_m - phi, _EPS)
        out = pd.DataFrame(index=df.index)
        out[f"mech_{self.id}_gap"] = gap
        out[f"mech_{self.id}_phi_over_gap"] = phi / gap
        out[f"mech_{self.id}_free_volume"] = -np.log(np.clip(1.0 - phi / phi_m, _EPS, 1.0 - _EPS))
        out[f"mech_{self.id}_base_tau"] = self.base_predict(df, params)
        return out


class DynamicMechanism:
    """A mechanism whose physics SHAPE g(phi, params) is supplied by a callable
    (LLM-written from a searched formula), while the HARNESS owns the fold-local
    parameter fit.

    The LLM writes only `shape(df, params) -> g` (unit-scale physics; tau = m1*g)
    and declares which extra params to grid-fit. The harness always supplies a
    data-driven `phi_m` grid (max_phi + positive offsets, so phi_m > phi by
    construction => no division blow-up), grid-searches the declared params, and
    closed-form fits the linear scale `m1` against TRAINING y only. The LLM's
    shape sees X and params, NEVER y — so a searched mechanism cannot leak.

    Implements the same duck-typed interface as Mechanism (id / data_adequacy /
    fit_global / base_predict / features), so screen_pair and the candidate
    builder treat DynamicMechanism and the seed Mechanisms identically.
    """

    def __init__(self, spec: dict[str, Any], shape_fn: Callable[[pd.DataFrame, dict], Any],
                 source: str | None = None):
        self.spec = dict(spec or {})
        self.id = str(self.spec["id"])
        self.name = str(self.spec.get("name", self.id))
        self.paper = str(self.spec.get("paper", ""))
        self.kind = str(self.spec.get("kind", "searched"))
        self.required_columns = tuple(self.spec.get("required_columns", ("phi",)))
        self.needs_shear_rate = bool(self.spec.get("needs_shear_rate", False))
        self.needs_microstructure = bool(self.spec.get("needs_microstructure", False))
        # declared extra params: name -> list of candidate values (numeric grid)
        self._param_grids = {str(k): list(v) for k, v in (self.spec.get("params", {}) or {}).items()}
        self._shape_fn = shape_fn
        # the LLM source string (when known) — used for persistence AND to make the
        # mechanism picklable: an exec'd shape function cannot be pickled directly,
        # so __getstate__ drops it and __setstate__ re-execs the source.
        self.source = source

    @staticmethod
    def _shape_from_source(source: str):
        ns: dict[str, Any] = {"np": np, "numpy": np, "pd": pd, "pandas": pd}
        exec(compile(source, "<mechanism_source>", "exec"), ns)  # noqa: S102 - source already preflight-vetted
        return ns.get("shape")

    def __getstate__(self):
        state = self.__dict__.copy()
        # an exec'd function is not picklable; persist the source instead
        if self.source:
            state["_shape_fn"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        if self._shape_fn is None and getattr(self, "source", None):
            self._shape_fn = self._shape_from_source(self.source)

    def data_adequacy(self, columns) -> dict[str, Any]:
        return _screen_columns(self.name, self.required_columns, self.needs_shear_rate,
                               self.needs_microstructure, columns)

    def _safe_g(self, df: pd.DataFrame, params: dict[str, Any]) -> np.ndarray:
        g = np.asarray(self._shape_fn(df, params), dtype=float)
        # stability guard: a searched shape must never inject nan/inf downstream
        return np.where(np.isfinite(g), g, 0.0)

    def _phi_m_grid(self, df: pd.DataFrame) -> list[float]:
        phi = _col(df, "phi")
        finite = phi[np.isfinite(phi)]
        max_phi = float(np.max(finite)) if len(finite) else 0.5
        grid = [min(round(max_phi + d, 4), 0.98) for d in (0.02, 0.05, 0.08, 0.12, 0.18)]
        return sorted(set(grid))

    def fit_global(self, df: pd.DataFrame, y: np.ndarray) -> dict[str, Any]:
        """Fold-local grid fit over (phi_m x declared params) + closed-form m1."""
        yv = np.asarray(y, dtype=float)
        names = list(self._param_grids.keys())
        grids = [self._param_grids[n] for n in names]
        best = None
        for phi_m in self._phi_m_grid(df):
            for combo in (itertools.product(*grids) if grids else [()]):
                params = {"phi_m": float(phi_m)}
                params.update({n: v for n, v in zip(names, combo)})
                g = self._safe_g(df, params)
                m1 = _linear_scale(yv, g)
                resid = yv - m1 * g
                sse = float(np.mean(resid * resid))
                if best is None or sse < best["sse"]:
                    best = {**params, "m1": float(m1), "sse": sse}
        if best is None:
            best = {"phi_m": float(self._phi_m_grid(df)[-1]), "m1": 0.0, "sse": float("inf")}
        best.pop("sse", None)
        return best

    def base_predict(self, df: pd.DataFrame, params: dict[str, Any]) -> np.ndarray:
        g = self._safe_g(df, params)
        return np.clip(float(params.get("m1", 1.0)) * g, 0.0, None)

    def features(self, df: pd.DataFrame, params: dict[str, Any]) -> pd.DataFrame:
        phi = _col(df, "phi")
        phi_m = float(params["phi_m"])
        gap = np.maximum(phi_m - phi, _EPS)
        out = pd.DataFrame(index=df.index)
        out[f"mech_{self.id}_gap"] = gap
        out[f"mech_{self.id}_phi_over_gap"] = phi / gap
        out[f"mech_{self.id}_base_tau"] = self.base_predict(df, params)
        return out


YODEL = Mechanism(
    id="yodel",
    name="YODEL yield-stress relation (Flatt & Bowen 2006)",
    paper="Flatt R. J. and Bowen P. (2006), JACerS.",
    required_columns=("phi",),
    kind="packing",
    needs_microstructure=True,
    _shape="yodel",
)
LIAN_PACKING = Mechanism(
    id="lian_packing",
    name="Lian-style packing relation (Lian 2025)",
    paper="Lian et al. (2025), Materials 18, 2983.",
    required_columns=("phi",),
    kind="packing",
    needs_microstructure=True,
    _shape="lian",
)
HERSCHEL_BULKLEY = Mechanism(
    id="herschel_bulkley",
    name="Herschel-Bulkley constitutive relation",
    paper="Herschel & Bulkley (1926).",
    required_columns=("phi",),
    kind="constitutive",
    needs_shear_rate=True,  # -> data_adequacy rejects on shear-rate-free data
)

MECHANISM_REGISTRY = {m.id: m for m in (YODEL, LIAN_PACKING, HERSCHEL_BULKLEY)}


def get_mechanism(mech_id: str) -> Mechanism:
    if mech_id not in MECHANISM_REGISTRY:
        raise KeyError(f"Unknown mechanism '{mech_id}'; known: {sorted(MECHANISM_REGISTRY)}")
    return MECHANISM_REGISTRY[mech_id]
