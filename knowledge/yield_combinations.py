"""Layer 3 — Combination-mode templates for direction-④.

Each combination mode is a scikit-compatible estimator scaffold that fixes the
boilerplate + anti-leakage structure; the mechanism/model specifics are the
fill-in slots (a Mechanism instance + a model family string). See
docs/yield_direction4_mechanism_model_search_design.md.

Modes implemented here (deterministic, no torch):
  C0 raw_model         — model on raw features only (baseline)
  C1 mechanism_feature — mechanism-derived columns appended to raw -> model
  C2 physics_residual  — tau = mechanism.base_predict(fold-fit params) + model(residual)
  C5 pure_mechanism    — mechanism equation only (fit fold-local, no model)
(C3 hidden_physical_layer / PINN and C4 physics_regularized are added in a
follow-up using torch.)

Anti-leakage: every mode fits mechanism params + model on the TRAINING rows given
to fit(); predict() reuses them. No mode ever uses a sample's own y at predict
time, and no mode inverts y into a feature.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from knowledge.yield_mechanisms import Mechanism, get_mechanism

MODEL_FAMILIES = ("ridge", "rf", "hgb", "kernel", "mlp")
COMBINATION_MODES = ("raw_model", "mechanism_feature", "physics_residual", "pure_mechanism")


def _make_regressor(family: str, random_state: int, n_samples: int):
    if family == "ridge":
        return make_pipeline(StandardScaler(), Ridge(alpha=1.0))
    if family == "rf":
        return RandomForestRegressor(n_estimators=200, max_depth=None, min_samples_leaf=3,
                                     random_state=random_state, n_jobs=-1)
    if family == "kernel":
        kernel = ConstantKernel(1.0) * Matern(length_scale=1.0, nu=1.5) + WhiteKernel(1e-3)
        return make_pipeline(StandardScaler(),
                             GaussianProcessRegressor(kernel=kernel, alpha=1e-6, normalize_y=True,
                                                      random_state=random_state))
    if family == "mlp":
        return make_pipeline(StandardScaler(),
                             MLPRegressor(hidden_layer_sizes=(32, 16), alpha=1e-3, max_iter=300,
                                          early_stopping=True, random_state=random_state))
    return HistGradientBoostingRegressor(max_iter=300, learning_rate=0.06, max_depth=6,
                                         min_samples_leaf=10, l2_regularization=1.0,
                                         random_state=random_state)


def _encode(df: pd.DataFrame, columns: list[str] | None = None) -> pd.DataFrame:
    enc = pd.get_dummies(df, dummy_na=False).apply(pd.to_numeric, errors="coerce")
    if columns is not None:
        enc = enc.reindex(columns=columns, fill_value=0.0)
    med = enc.median(numeric_only=True)
    return enc.fillna(med).fillna(0.0)


class _BaseCombination:
    combination_mode = "base"

    def __init__(self, mechanism: Mechanism | None = None, model_family: str = "hgb",
                 constraints: tuple[str, ...] = ("nonnegative",), random_state: int = 0):
        self.mechanism = mechanism
        self.model_family = model_family
        self.constraints = tuple(constraints or ())
        self.random_state = int(random_state)
        self.params_: dict[str, Any] = {}
        self._cols: list[str] | None = None
        self.model_ = None

    def _clip(self, pred: np.ndarray) -> np.ndarray:
        pred = np.asarray(pred, dtype=float)
        return np.clip(pred, 0.0, None) if "nonnegative" in self.constraints else pred

    def spec(self) -> dict[str, Any]:
        return {
            "combination_mode": self.combination_mode,
            "mechanism": self.mechanism.id if self.mechanism else None,
            "model_family": self.model_family,
            "constraints": list(self.constraints),
            "fitted_params": self.params_,
        }


class RawModel(_BaseCombination):
    combination_mode = "raw_model"

    def fit(self, X: pd.DataFrame, y):
        Xn = _encode(X)
        self._cols = list(Xn.columns)
        self.model_ = _make_regressor(self.model_family, self.random_state, len(X))
        self.model_.fit(Xn.to_numpy(dtype=float), np.asarray(y, dtype=float))
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        Xn = _encode(X, self._cols)
        return self._clip(self.model_.predict(Xn.to_numpy(dtype=float)))


class MechanismFeature(_BaseCombination):
    combination_mode = "mechanism_feature"

    def fit(self, X: pd.DataFrame, y):
        y = np.asarray(y, dtype=float)
        self.params_ = self.mechanism.fit_global(X, y)          # fold-local, no per-sample inversion
        feats = pd.concat([X, self.mechanism.features(X, self.params_)], axis=1)
        Xn = _encode(feats)
        self._cols = list(Xn.columns)
        self.model_ = _make_regressor(self.model_family, self.random_state, len(X))
        self.model_.fit(Xn.to_numpy(dtype=float), y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        feats = pd.concat([X, self.mechanism.features(X, self.params_)], axis=1)
        Xn = _encode(feats, self._cols)
        return self._clip(self.model_.predict(Xn.to_numpy(dtype=float)))


class PhysicsResidual(_BaseCombination):
    combination_mode = "physics_residual"

    def fit(self, X: pd.DataFrame, y):
        y = np.asarray(y, dtype=float)
        self.params_ = self.mechanism.fit_global(X, y)          # physics base fit on train
        base = self.mechanism.base_predict(X, self.params_)
        Xn = _encode(X)
        self._cols = list(Xn.columns)
        self.model_ = _make_regressor(self.model_family, self.random_state, len(X))
        self.model_.fit(Xn.to_numpy(dtype=float), y - base)     # model learns the residual
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        base = self.mechanism.base_predict(X, self.params_)
        Xn = _encode(X, self._cols)
        resid = self.model_.predict(Xn.to_numpy(dtype=float))
        return self._clip(base + resid)


class PureMechanism(_BaseCombination):
    combination_mode = "pure_mechanism"

    def fit(self, X: pd.DataFrame, y):
        self.params_ = self.mechanism.fit_global(X, np.asarray(y, dtype=float))
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self._clip(self.mechanism.base_predict(X, self.params_))


_MODE_TO_CLASS = {
    "raw_model": RawModel,
    "mechanism_feature": MechanismFeature,
    "physics_residual": PhysicsResidual,
    "pure_mechanism": PureMechanism,
}


def make_combination(combination_mode: str, *, mechanism_id: str | None = None,
                     model_family: str = "hgb", constraints=("nonnegative",),
                     random_state: int = 0) -> _BaseCombination:
    """Factory: build a combination estimator for a candidate search point."""
    if combination_mode not in _MODE_TO_CLASS:
        raise KeyError(f"Unknown combination_mode '{combination_mode}'; known: {sorted(_MODE_TO_CLASS)}")
    mech = get_mechanism(mechanism_id) if mechanism_id else None
    if combination_mode in ("mechanism_feature", "physics_residual", "pure_mechanism") and mech is None:
        raise ValueError(f"combination_mode '{combination_mode}' requires a mechanism_id.")
    return _MODE_TO_CLASS[combination_mode](
        mechanism=mech, model_family=model_family, constraints=constraints, random_state=random_state)
