"""Expert PINN baseline for high-solid-content slurry yield-stress data.

This module adapts the classmate-style yield PINN into the repository's
leakage-safe OOF protocol. It is intentionally a baseline/evaluation module,
not part of the free-search champion selection path yet.
"""

from __future__ import annotations

import copy
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
except Exception as exc:  # pragma: no cover - exercised when torch is absent.
    torch = None
    nn = None
    F = None
    optim = None
    _TORCH_IMPORT_ERROR = exc
else:
    _TORCH_IMPORT_ERROR = None


TARGET_COLUMN = "yield_stress"
FORWARD_STAGE_IDS = [1, 2, 3, 4, 6, 7, 9, 10, 11]
REVERSE_STAGE_IDS = [4, 10]

RAW_INPUT_COLUMNS = [
    "slurry_temp_c",
    "water_tank_temp_c",
    "jacket_temp_c",
    "internal_pressure_kpa",
    "coarse_ap_mass_kg",
    "fine_ap_mass_kg",
    "rdx_mass_kg",
    *[
        item
        for stage in FORWARD_STAGE_IDS
        for item in (f"forward_rpm_{stage}", f"forward_time_{stage}_min")
    ],
    *[
        item
        for stage in REVERSE_STAGE_IDS
        for item in (f"reverse_rpm_{stage}", f"reverse_time_{stage}_min")
    ],
]

DERIVED_COLUMNS = [
    "forward_revolutions",
    "reverse_revolutions",
    "net_revolutions",
    "total_time_min",
    "reverse_time_ratio",
    "total_revolutions_abs",
    "rpm_abs_weighted_mean",
    "reverse_revolutions_ratio",
    "direction_balance_ratio",
    "mixing_intensity_2_scaled",
    "direction_switch_count",
    "jacket_slurry_delta_c",
    "thermal_drive_abs_c",
    "arrhenius_delta",
    "pressure_barus_feature",
    "fine_ap_vol_frac_in_ap",
    "rdx_vol_frac_in_modeled_solids",
    "rdx_to_ap_vol_ratio",
    "ap_bimodal_balance",
    "log_modeled_solid_mass_kg",
]
EXPERT_PINN_FEATURE_COLUMNS = [*RAW_INPUT_COLUMNS, *DERIVED_COLUMNS]


@dataclass
class FeatureScaler:
    mean_: np.ndarray
    std_: np.ndarray

    @classmethod
    def fit(cls, x: np.ndarray) -> "FeatureScaler":
        mean = x.mean(axis=0)
        std = x.std(axis=0)
        std[std < 1e-8] = 1.0
        return cls(mean_=mean, std_=std)

    def transform(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean_) / self.std_


def _require_torch() -> None:
    if torch is None:
        raise RuntimeError(f"PyTorch is required for expert PINN baseline: {_TORCH_IMPORT_ERROR}")


def _logit(value: float) -> float:
    value = min(max(value, 1e-6), 1.0 - 1.0e-6)
    return math.log(value / (1.0 - value))


def _mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = np.maximum(np.abs(y_true), 1e-8)
    return float(np.mean(np.abs((y_true - y_pred) / denom)) * 100.0)


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_pred = np.clip(np.asarray(y_pred, dtype=float), 0.0, None)
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
        "mape": _mape(y_true, y_pred),
    }


def set_seed(seed: int) -> None:
    _require_torch()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    try:
        torch.set_num_threads(max(1, min(4, torch.get_num_threads())))
    except Exception:
        pass


def build_expert_pinn_features(df: pd.DataFrame, *, pressure_ref_kpa: float | None = None) -> pd.DataFrame:
    """Build the 49-column feature schema used by the expert PINN baseline."""

    out = pd.DataFrame(index=df.index)
    for column in RAW_INPUT_COLUMNS:
        if column in df.columns:
            out[column] = pd.to_numeric(df[column], errors="coerce").fillna(0.0)
        else:
            out[column] = 0.0

    eps = 1e-6
    forward_time_cols = [f"forward_time_{stage}_min" for stage in FORWARD_STAGE_IDS]
    reverse_time_cols = [f"reverse_time_{stage}_min" for stage in REVERSE_STAGE_IDS]
    forward_rpm_cols = [f"forward_rpm_{stage}" for stage in FORWARD_STAGE_IDS]
    reverse_rpm_cols = [f"reverse_rpm_{stage}" for stage in REVERSE_STAGE_IDS]

    out["forward_time_total_min"] = out[forward_time_cols].sum(axis=1)
    out["reverse_time_total_min"] = out[reverse_time_cols].sum(axis=1)
    out["total_time_min"] = out["forward_time_total_min"] + out["reverse_time_total_min"]
    out["reverse_time_ratio"] = out["reverse_time_total_min"] / (out["total_time_min"] + eps)

    out["forward_revolutions"] = sum(out[rpm] * out[t] for rpm, t in zip(forward_rpm_cols, forward_time_cols))
    out["reverse_revolutions"] = sum(out[rpm] * out[t] for rpm, t in zip(reverse_rpm_cols, reverse_time_cols))
    out["net_revolutions"] = out["forward_revolutions"] - out["reverse_revolutions"]
    out["total_revolutions_abs"] = out["forward_revolutions"] + out["reverse_revolutions"]
    out["rpm_abs_weighted_mean"] = out["total_revolutions_abs"] / (out["total_time_min"] + eps)
    out["reverse_revolutions_ratio"] = out["reverse_revolutions"] / (out["total_revolutions_abs"] + eps)
    out["direction_balance_ratio"] = out["net_revolutions"] / (out["total_revolutions_abs"] + eps)

    def _switch_count(row: pd.Series) -> int:
        sequence = [("F", row[col]) for col in forward_time_cols] + [("R", row[col]) for col in reverse_time_cols]
        active = [direction for direction, duration in sequence if duration > 0]
        if len(active) <= 1:
            return 0
        return sum(active[idx] != active[idx - 1] for idx in range(1, len(active)))

    out["direction_switch_count"] = out.apply(_switch_count, axis=1)
    pairs = list(zip(forward_rpm_cols, forward_time_cols)) + list(zip(reverse_rpm_cols, reverse_time_cols))
    out["mixing_intensity_2_scaled"] = sum((out[rpm] / 1000.0) ** 2 * out[t] for rpm, t in pairs)

    out["jacket_slurry_delta_c"] = out["jacket_temp_c"] - out["slurry_temp_c"]
    out["thermal_drive_abs_c"] = out["jacket_slurry_delta_c"].abs()
    slurry_temp_k = np.maximum(out["slurry_temp_c"] + 273.15, 1.0)
    out["arrhenius_delta"] = (1.0 / slurry_temp_k) - (1.0 / 298.15)
    if pressure_ref_kpa is None:
        pressure_ref_kpa = float(out["internal_pressure_kpa"].median())
    out["pressure_barus_feature"] = (out["internal_pressure_kpa"] - float(pressure_ref_kpa)) / 1000.0

    ap_density = 1950.0
    rdx_density = 1800.0
    coarse_ap_vol = out["coarse_ap_mass_kg"] / ap_density
    fine_ap_vol = out["fine_ap_mass_kg"] / ap_density
    rdx_vol = out["rdx_mass_kg"] / rdx_density
    ap_total_vol = coarse_ap_vol + fine_ap_vol
    modeled_solids_vol = ap_total_vol + rdx_vol

    fine_ap_frac = fine_ap_vol / (ap_total_vol + eps)
    coarse_ap_frac = coarse_ap_vol / (ap_total_vol + eps)
    out["fine_ap_vol_frac_in_ap"] = fine_ap_frac
    out["rdx_vol_frac_in_modeled_solids"] = rdx_vol / (modeled_solids_vol + eps)
    out["rdx_to_ap_vol_ratio"] = rdx_vol / (ap_total_vol + eps)
    out["ap_bimodal_balance"] = 4.0 * fine_ap_frac * coarse_ap_frac
    out["log_modeled_solid_mass_kg"] = np.log1p(
        out["coarse_ap_mass_kg"] + out["fine_ap_mass_kg"] + out["rdx_mass_kg"]
    )

    return out[EXPERT_PINN_FEATURE_COLUMNS].apply(pd.to_numeric, errors="coerce").fillna(0.0)


if nn is not None:

    class ThermalBranch(nn.Module):
        def __init__(self, in_dim: int, out_dim: int):
            super().__init__()
            hidden = max(8, out_dim)
            self.net = nn.Sequential(
                nn.Linear(in_dim, hidden),
                nn.LayerNorm(hidden),
                nn.Tanh(),
                nn.Linear(hidden, out_dim),
                nn.Tanh(),
            )

        def forward(self, x):
            return self.net(x)


    class CompositionBranch(nn.Module):
        def __init__(self, in_dim: int, out_dim: int):
            super().__init__()
            hidden = max(8, out_dim)
            self.shortcut = nn.Linear(in_dim, out_dim)
            self.mlp = nn.Sequential(
                nn.Linear(in_dim, hidden),
                nn.SiLU(),
                nn.Linear(hidden, out_dim),
            )
            self.norm = nn.LayerNorm(out_dim)
            self.act = nn.Tanh()

        def forward(self, x):
            z = self.shortcut(x) + self.mlp(x)
            z = self.norm(z)
            return self.act(z)


    class HistoryBranch(nn.Module):
        STAGE_RAW_DIM = 22

        def __init__(self, in_dim: int, out_dim: int):
            super().__init__()
            summary_dim = in_dim - self.STAGE_RAW_DIM
            stage_hidden = max(8, out_dim)
            stage_embed = max(8, out_dim)
            summary_hidden = max(12, out_dim)

            self.stage_encoder = nn.Sequential(
                nn.Linear(2, stage_hidden),
                nn.Tanh(),
                nn.Linear(stage_hidden, stage_embed),
                nn.Tanh(),
            )
            self.summary_mlp = nn.Sequential(
                nn.Linear(summary_dim, summary_hidden),
                nn.LayerNorm(summary_hidden),
                nn.SiLU(),
                nn.Linear(summary_hidden, stage_embed),
                nn.SiLU(),
            )
            self.merge = nn.Sequential(
                nn.Linear(stage_embed * 3, out_dim),
                nn.LayerNorm(out_dim),
                nn.SiLU(),
                nn.Linear(out_dim, out_dim),
                nn.Tanh(),
            )

        def forward(self, x):
            stage_raw = x[:, : self.STAGE_RAW_DIM].reshape(-1, 11, 2)
            summary_raw = x[:, self.STAGE_RAW_DIM :]
            stage_tokens = self.stage_encoder(stage_raw.reshape(-1, 2)).reshape(x.size(0), 11, -1)
            stage_time = torch.clamp(stage_raw[:, :, 1], min=0.0)
            stage_weight = stage_time / (stage_time.sum(dim=1, keepdim=True) + 1e-6)
            stage_weighted = torch.sum(stage_tokens * stage_weight.unsqueeze(-1), dim=1)
            stage_max = stage_tokens.max(dim=1).values
            summary_embed = self.summary_mlp(summary_raw)
            return self.merge(torch.cat([stage_weighted, stage_max, summary_embed], dim=1))


    class GatedFusionHead(nn.Module):
        def __init__(self, branch_dim: int, hidden_dim: int):
            super().__init__()
            gate_hidden = max(8, branch_dim)
            tail_hidden = max(8, hidden_dim // 2)
            self.gate_temperature = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
            self.residual_scale = nn.Parameter(torch.tensor(0.10, dtype=torch.float32))
            self.gate_net = nn.Sequential(
                nn.Linear(branch_dim * 3, gate_hidden),
                nn.LayerNorm(gate_hidden),
                nn.Tanh(),
                nn.Linear(gate_hidden, 3),
            )
            self.base_head = nn.Linear(branch_dim, 1)
            self.delta_head = nn.Sequential(
                nn.Linear(branch_dim * 4 + 3, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, tail_hidden),
                nn.SiLU(),
                nn.Linear(tail_hidden, 1),
            )

        def forward(self, zt, zc, zh):
            context = torch.cat([zt, zc, zh], dim=1)
            temperature = torch.clamp(self.gate_temperature.abs(), min=0.7, max=2.5)
            gate_logits = self.gate_net(context) / temperature
            gates = torch.softmax(gate_logits, dim=1)
            stacked = torch.stack([zt, zc, zh], dim=1)
            fused = torch.sum(stacked * gates.unsqueeze(-1), dim=1)
            head_input = torch.cat([zt, zc, zh, fused, gates], dim=1)
            raw_base = self.base_head(fused).squeeze(-1)
            raw_delta = self.delta_head(head_input).squeeze(-1)
            raw = raw_base + torch.tanh(self.residual_scale) * raw_delta
            return raw, gates


    class ExpertYieldPINN(nn.Module):
        M1_LO = 50.0
        M1_HI = 3000.0
        M1_INIT = 120.0

        PHI0_LO = 0.0
        PHI0_HI = 0.09
        PHI_MAX_LO = 0.52
        PHI_MAX_HI = 0.72

        THERMAL_IDX = [0, 1, 2, 3, 40, 41, 42, 43]
        COMPOSITION_IDX = [4, 5, 6, 44, 45, 46, 47, 48]
        HISTORY_IDX = list(range(7, 40))

        def __init__(self, input_dim: int = 49, hidden_dim: int = 16):
            super().__init__()
            if input_dim != 49:
                raise ValueError(f"ExpertYieldPINN expects 49 input features, got {input_dim}")
            branch_dim = max(8, hidden_dim)
            self.branch_thermal = ThermalBranch(len(self.THERMAL_IDX), branch_dim)
            self.branch_composition = CompositionBranch(len(self.COMPOSITION_IDX), branch_dim)
            self.branch_history = HistoryBranch(len(self.HISTORY_IDX), branch_dim)
            self.fuse = GatedFusionHead(branch_dim=branch_dim, hidden_dim=max(hidden_dim, 16))
            self._init_weights()

            phi0_init = 0.026
            phi0_ratio = (phi0_init - self.PHI0_LO) / (self.PHI0_HI - self.PHI0_LO)
            self.raw_phi0 = nn.Parameter(torch.tensor(_logit(phi0_ratio), dtype=torch.float32))

            phi_max_init = 0.570
            phi_max_ratio = (phi_max_init - self.PHI_MAX_LO) / (self.PHI_MAX_HI - self.PHI_MAX_LO)
            self.raw_phi_max = nn.Parameter(torch.tensor(_logit(phi_max_ratio), dtype=torch.float32))

        def _init_weights(self) -> None:
            for module in self.modules():
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight, gain=0.6)
                    nn.init.zeros_(module.bias)

            ratio = (self.M1_INIT - self.M1_LO) / (self.M1_HI - self.M1_LO)
            self.fuse.base_head.weight.data.zero_()
            self.fuse.base_head.bias.data.fill_(_logit(ratio))
            self.fuse.gate_net[-1].weight.data.zero_()
            self.fuse.gate_net[-1].bias.data.zero_()
            self.fuse.delta_head[-1].weight.data.zero_()
            self.fuse.delta_head[-1].bias.data.zero_()

        def decode_global_params(self):
            phi0 = self.PHI0_LO + torch.sigmoid(self.raw_phi0) * (self.PHI0_HI - self.PHI0_LO)
            phi_max = self.PHI_MAX_LO + torch.sigmoid(self.raw_phi_max) * (self.PHI_MAX_HI - self.PHI_MAX_LO)
            return phi0, phi_max

        def decode_m1(self, raw):
            return self.M1_LO + torch.sigmoid(raw) * (self.M1_HI - self.M1_LO)

        def forward(self, x, phi):
            xt = x[:, self.THERMAL_IDX]
            xc = x[:, self.COMPOSITION_IDX]
            xh = x[:, self.HISTORY_IDX]
            zt = self.branch_thermal(xt)
            zc = self.branch_composition(xc)
            zh = self.branch_history(xh)
            raw_m1, gates = self.fuse(zt, zc, zh)

            m1_pred = self.decode_m1(raw_m1)
            phi0, phi_max = self.decode_global_params()
            eps = 1e-6
            numerator = m1_pred * phi * torch.clamp(phi - phi0, min=eps) ** 2
            denominator = phi_max * torch.clamp(phi_max - phi, min=eps)
            tau_pred = numerator / denominator
            aux = {
                "m1_pred": m1_pred,
                "phi0": phi0,
                "phi_max": phi_max,
                "branch_gates": gates,
            }
            return tau_pred, aux


def _stable_small_sample_loss(pred, target):
    log_loss = F.smooth_l1_loss(torch.log1p(torch.clamp(pred, min=0.0)), torch.log1p(target), beta=0.15)
    scale = torch.clamp(target.mean(), min=1.0)
    rel_loss = torch.mean(((pred - target) / scale) ** 2)
    return log_loss + 0.05 * rel_loss


def _to_tensors(x: np.ndarray, phi: np.ndarray, y: np.ndarray | None = None):
    _require_torch()
    tensors = [
        torch.tensor(x, dtype=torch.float32),
        torch.tensor(phi, dtype=torch.float32),
    ]
    if y is not None:
        tensors.append(torch.tensor(y, dtype=torch.float32))
    return tuple(tensors)


def _train_with_internal_early_stop(
    x_train: np.ndarray,
    phi_train: np.ndarray,
    y_train: np.ndarray,
    *,
    seed: int,
    hidden_dim: int,
    max_epochs: int,
    patience: int,
    lr: float,
    weight_decay: float,
    inner_val_frac: float,
) -> tuple[int, dict[str, float]]:
    set_seed(seed)
    n = len(x_train)
    rng = np.random.default_rng(seed)
    order = rng.permutation(n)
    n_val = max(1, int(round(n * inner_val_frac)))
    val_idx = order[:n_val]
    fit_idx = order[n_val:]
    if len(fit_idx) < 8:
        fit_idx = order
        val_idx = order[: max(1, min(4, n))]

    scaler = FeatureScaler.fit(x_train[fit_idx])
    x_fit = scaler.transform(x_train[fit_idx])
    x_eval = scaler.transform(x_train[val_idx])
    xt, phit, yt = _to_tensors(x_fit, phi_train[fit_idx], y_train[fit_idx])
    xv, phiv, yv = _to_tensors(x_eval, phi_train[val_idx], y_train[val_idx])

    model = ExpertYieldPINN(input_dim=x_train.shape[1], hidden_dim=hidden_dim)
    opt = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=max(8, patience // 5))
    best_state = copy.deepcopy(model.state_dict())
    best_loss = float("inf")
    best_epoch = 1
    wait = 0

    for epoch in range(1, max_epochs + 1):
        model.train()
        opt.zero_grad()
        pred, _ = model(xt, phit)
        loss = _stable_small_sample_loss(pred, yt)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=3.0)
        opt.step()

        model.eval()
        with torch.no_grad():
            pred_eval, _ = model(xv, phiv)
            eval_loss = float(_stable_small_sample_loss(pred_eval, yv).item())
        scheduler.step(eval_loss)

        if np.isfinite(eval_loss) and eval_loss < best_loss - 1e-7:
            best_loss = eval_loss
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            wait = 0
        else:
            wait += 1
            if epoch >= max(60, patience) and wait >= patience:
                break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        pred_eval, aux = model(xv, phiv)
    learned = {
        "inner_best_loss": float(best_loss),
        "inner_best_epoch": float(best_epoch),
        "inner_phi0": float(aux["phi0"].detach().cpu().item()),
        "inner_phi_max": float(aux["phi_max"].detach().cpu().item()),
    }
    return int(best_epoch), learned


def _train_fixed_epochs(
    x_train: np.ndarray,
    phi_train: np.ndarray,
    y_train: np.ndarray,
    *,
    seed: int,
    hidden_dim: int,
    epochs: int,
    lr: float,
    weight_decay: float,
) -> tuple[Any, FeatureScaler, dict[str, float]]:
    set_seed(seed)
    scaler = FeatureScaler.fit(x_train)
    xt, phit, yt = _to_tensors(scaler.transform(x_train), phi_train, y_train)
    model = ExpertYieldPINN(input_dim=x_train.shape[1], hidden_dim=hidden_dim)
    opt = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    epochs = max(1, int(epochs))

    for _ in range(epochs):
        model.train()
        opt.zero_grad()
        pred, _ = model(xt, phit)
        loss = _stable_small_sample_loss(pred, yt)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=3.0)
        opt.step()

    model.eval()
    with torch.no_grad():
        _, aux = model(xt, phit)
    meta = {
        "fit_epochs": float(epochs),
        "phi0": float(aux["phi0"].detach().cpu().item()),
        "phi_max": float(aux["phi_max"].detach().cpu().item()),
    }
    return model, scaler, meta


def _predict(model: Any, scaler: FeatureScaler, x: np.ndarray, phi: np.ndarray) -> np.ndarray:
    xt, phit = _to_tensors(scaler.transform(x), phi)
    model.eval()
    with torch.no_grad():
        pred, _ = model(xt, phit)
    return np.clip(pred.detach().cpu().numpy().astype(float), 0.0, None)


def evaluate_expert_pinn_oof(
    data_path: str | Path,
    *,
    seeds: list[int] | tuple[int, ...] = (0, 1, 2),
    n_splits: int = 5,
    hidden_dim: int = 16,
    max_epochs: int = 500,
    patience: int = 60,
    lr: float = 1e-3,
    weight_decay: float = 1e-5,
    inner_val_frac: float = 0.15,
    out_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Evaluate the expert PINN baseline using leakage-safe KFold OOF."""

    _require_torch()
    data_path = Path(data_path)
    df = pd.read_csv(data_path) if data_path.suffix.lower() == ".csv" else pd.read_excel(data_path)
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Missing target column {TARGET_COLUMN!r} in {data_path}")
    if "phi" not in df.columns:
        raise ValueError("Expert PINN baseline requires a phi column.")

    pressure_ref = float(pd.to_numeric(df["internal_pressure_kpa"], errors="coerce").median())
    features_df = build_expert_pinn_features(df, pressure_ref_kpa=pressure_ref)
    x = features_df.to_numpy(dtype=np.float32)
    phi = pd.to_numeric(df["phi"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
    y = pd.to_numeric(df[TARGET_COLUMN], errors="coerce").to_numpy(dtype=np.float32)
    valid = np.isfinite(y)
    x, phi, y = x[valid], phi[valid], y[valid]
    sample_ids = (
        df.loc[valid, "sample_id"].astype(str).to_numpy()
        if "sample_id" in df.columns
        else np.array([str(i) for i in range(len(y))])
    )

    seed_results: list[dict[str, Any]] = []
    predictions_by_seed: dict[str, list[dict[str, Any]]] = {}

    for seed in seeds:
        splitter = KFold(n_splits=n_splits, shuffle=True, random_state=int(seed))
        oof = np.zeros(len(y), dtype=float)
        fold_rows: list[dict[str, Any]] = []

        for fold_idx, (train_idx, val_idx) in enumerate(splitter.split(x), start=1):
            best_epoch, inner_meta = _train_with_internal_early_stop(
                x[train_idx],
                phi[train_idx],
                y[train_idx],
                seed=int(seed) * 1000 + fold_idx,
                hidden_dim=hidden_dim,
                max_epochs=max_epochs,
                patience=patience,
                lr=lr,
                weight_decay=weight_decay,
                inner_val_frac=inner_val_frac,
            )
            model, scaler, fit_meta = _train_fixed_epochs(
                x[train_idx],
                phi[train_idx],
                y[train_idx],
                seed=int(seed) * 1000 + fold_idx + 100,
                hidden_dim=hidden_dim,
                epochs=best_epoch,
                lr=lr,
                weight_decay=weight_decay,
            )
            pred = _predict(model, scaler, x[val_idx], phi[val_idx])
            oof[val_idx] = pred
            row = {
                "fold": fold_idx,
                "n_train": int(len(train_idx)),
                "n_valid": int(len(val_idx)),
                **_metrics(y[val_idx], pred),
                **inner_meta,
                **fit_meta,
            }
            fold_rows.append(row)

        metrics = _metrics(y, oof)
        seed_key = str(int(seed))
        predictions_by_seed[seed_key] = [
            {
                "sample_id": str(sample_ids[idx]),
                "yield_stress_actual": float(y[idx]),
                "yield_stress_predicted": float(oof[idx]),
            }
            for idx in range(len(y))
        ]
        seed_results.append(
            {
                "seed": int(seed),
                "metrics": metrics,
                "fold_metrics": fold_rows,
            }
        )

    rmse_values = [row["metrics"]["rmse"] for row in seed_results]
    aggregate = {
        "rmse_mean": float(np.mean(rmse_values)),
        "rmse_std": float(np.std(rmse_values)),
        "mae_mean": float(np.mean([row["metrics"]["mae"] for row in seed_results])),
        "r2_mean": float(np.mean([row["metrics"]["r2"] for row in seed_results])),
        "mape_mean": float(np.mean([row["metrics"]["mape"] for row in seed_results])),
    }

    result = {
        "baseline_id": "expert_pinn_gated_three_branch",
        "baseline_name": "Expert gated three-branch PINN baseline",
        "data_path": str(data_path),
        "protocol": {
            "n_splits": int(n_splits),
            "seeds": [int(s) for s in seeds],
            "hidden_dim": int(hidden_dim),
            "max_epochs": int(max_epochs),
            "patience": int(patience),
            "lr": float(lr),
            "weight_decay": float(weight_decay),
            "inner_val_frac": float(inner_val_frac),
            "outer_valid_used_for_early_stopping": False,
            "feature_schema": "classmate-style 49-feature thermal/composition/history schema adapted to this dataset",
        },
        "data_profile": {
            "n_samples": int(len(y)),
            "n_features": int(x.shape[1]),
            "target": TARGET_COLUMN,
            "phi_min": float(np.min(phi)),
            "phi_max": float(np.max(phi)),
            "phi_std": float(np.std(phi)),
            "pressure_ref_kpa": float(pressure_ref),
        },
        "seed_results": seed_results,
        "aggregate": aggregate,
    }

    if out_dir is not None:
        out_path = Path(out_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        (out_path / "expert_pinn_oof_report.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        for seed, rows in predictions_by_seed.items():
            pd.DataFrame(rows).to_csv(out_path / f"expert_pinn_oof_predictions_seed{seed}.csv", index=False)

    return result
