"""Synthetic low-fidelity data support for yield-stress AutoML.

The real industrial yield-stress dataset is not available yet. This module
creates a reproducible, auditable temporary dataset from a real Table 6 anchor
plus physics-inspired perturbations. The LLM may design models, but it should
not directly invent CSV rows.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SYNTHETIC_DIR = PROJECT_ROOT / "agent_workspace" / "data" / "yield_synthetic"
DEFAULT_ANCHOR_PATH = DEFAULT_SYNTHETIC_DIR / "real_table6_anchor.csv"
DEFAULT_SYNTHETIC_PATH = DEFAULT_SYNTHETIC_DIR / "synthetic_yield_v1.csv"
DEFAULT_SYNTHETIC_REPORT_PATH = DEFAULT_SYNTHETIC_DIR / "synthetic_data_report.json"

LIAN_2025_SOURCE = {
    "source_id": "LOCAL_TABLE6_MATERIALS_18_02983",
    "paper": "Lian, C.; Zhang, X.; Han, L.; Lin, W.; Wen, W. Constitutive Modeling of Rheological Behavior of Cement Paste Based on Material Composition. Materials 2025, 18, 2983.",
    "doi": "10.3390/ma18132983",
    "table": "Table 6. Mix proportions and rheological properties of pastes with PCE superplasticizer.",
    "claims_used": [
        "The paper introduces a virtual maximum packing fraction fmax to account for flocculation and entrapped-water effects.",
        "Solid volume fraction is the dominant factor governing paste rheology; yield stress rises strongly as powder volume fraction approaches the maximum packing limit.",
        "PCE superplasticizer action can be modeled as a modulation of virtual packing density fmax through floc dispersion and release of entrapped water.",
    ],
}


TABLE6_ROWS = [
    # mix, Vp_L, Vw_L, phi_pct, sp_pct, cement_kg_m3, fa_kg_m3, water_kg_m3, sp_kg_m3, yield_stress_pa, plastic_viscosity_pa_s
    (1, 459, 533, 45.80, 0.80, 1130, 234, 533, 10.91, 0.35, 11.66),
    (2, 459, 531, 45.90, 0.90, 1131, 234, 531, 12.29, 0.23, 5.77),
    (3, 460, 530, 45.90, 1.00, 1132, 234, 530, 13.67, 0.19, 5.11),
    (4, 503, 493, 50.20, 0.40, 1290, 256, 493, 6.19, 1.95, 62.48),
    (5, 503, 490, 50.30, 0.50, 1293, 257, 490, 8.52, 0.97, 35.48),
    (6, 504, 489, 50.30, 0.60, 1293, 257, 489, 8.83, 0.74, 33.22),
    (7, 504, 489, 50.40, 0.60, 1294, 257, 489, 9.30, 0.67, 28.80),
    (8, 478, 517, 47.80, 0.40, 1229, 244, 517, 5.89, 1.14, 39.57),
    (9, 479, 515, 47.90, 0.50, 1230, 244, 515, 7.37, 0.66, 30.23),
    (10, 503, 491, 50.30, 0.50, 1292, 257, 491, 7.74, 1.29, 49.60),
    (11, 504, 487, 50.40, 0.70, 1295, 257, 487, 10.87, 0.39, 20.68),
    (12, 479, 516, 47.90, 0.40, 1229, 244, 516, 6.63, 0.86, 32.42),
    (13, 479, 516, 47.90, 0.50, 1230, 244, 516, 7.08, 0.67, 31.31),
    (14, 504, 488, 50.40, 0.60, 1294, 257, 488, 10.08, 0.44, 28.48),
    (15, 479, 514, 47.90, 0.50, 1231, 244, 514, 8.11, 0.60, 24.68),
    (16, 480, 513, 47.90, 0.60, 1232, 245, 513, 8.86, 0.46, 21.16),
]


def table6_anchor_dataframe() -> pd.DataFrame:
    columns = [
        "mix_id",
        "Vp_L",
        "Vw_L",
        "phi_percent",
        "sp_percent",
        "cement_kg_m3",
        "fly_ash_kg_m3",
        "water_kg_m3",
        "sp_kg_m3",
        "yield_stress",
        "plastic_viscosity_pa_s",
    ]
    df = pd.DataFrame(TABLE6_ROWS, columns=columns)
    df.insert(0, "sample_id", [f"table6_{int(x):02d}" for x in df["mix_id"]])
    binder = df["cement_kg_m3"] + df["fly_ash_kg_m3"]
    df["phi"] = df["phi_percent"] / 100.0
    df["w_b"] = df["water_kg_m3"] / binder
    df["fa_ratio"] = df["fly_ash_kg_m3"] / binder
    # Composition-schema columns so training and anchor share one feature set
    # and match what the real dataset will report (Table 6 style mix proportions).
    df["binder_kg_m3"] = binder
    df["sp_binder_ratio"] = df["sp_kg_m3"] / binder
    # Reference-only virtual maximum packing fraction, from transferable drivers
    # only (SP dosage + fly-ash fraction). Documented as the target "answer key",
    # never a model input (see feature_policy / guardrails).
    phi_max_eff = (
        0.585
        + 0.030 * np.tanh((df["sp_percent"] - 0.45) / 0.35)
        - 0.012 * (df["fa_ratio"] - df["fa_ratio"].mean())
    )
    df["phi_max_eff_reference"] = np.clip(phi_max_eff, df["phi"] + 0.035, 0.70)
    df["data_fidelity"] = "real_table6_anchor"
    df["source"] = LIAN_2025_SOURCE["source_id"]
    return df


def _safe_range(series: pd.Series, pad: float, low: float | None = None, high: float | None = None) -> tuple[float, float]:
    lo = float(series.min()) - pad
    hi = float(series.max()) + pad
    if low is not None:
        lo = max(lo, low)
    if high is not None:
        hi = min(hi, high)
    return lo, hi


def generate_synthetic_yield_data(
    n_samples: int = 800,
    random_state: int = 20250629,
    anchor_df: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Generate a reproducible synthetic dataset anchored to Lian Table 6.

    The synthetic target is not random labels. It is generated by a packing
    term plus process/structure modifiers, then scaled to match the Table 6
    yield-stress range.
    """

    rng = np.random.default_rng(random_state)
    anchor = table6_anchor_dataframe() if anchor_df is None else anchor_df.copy()

    # Draw only the transferable, physically-independent knobs. w_b is DERIVED
    # from mix design (not drawn) so training and real Table 6 share the same
    # composition relations, and no train-only invented feature can wreck anchor
    # generalization.
    # Sample each driver in a band that OVERLAPS the real Table 6 operating
    # window, so the synthetic set is a denser sampling of the same regime rather
    # than a wider one the model can overfit. Real phi barely moves
    # (0.458-0.504); keep the draw tight around it.
    phi_lo, phi_hi = _safe_range(anchor["phi"], 0.03, 0.40, 0.58)
    sp_lo, sp_hi = _safe_range(anchor["sp_percent"], 0.15, 0.0, 1.4)
    # Real Table 6 holds fly-ash fraction nearly constant (~0.167). Keep the draw
    # tight around it (mildly varied, still realistic) instead of an over-broad
    # band that has almost no range overlap with the anchor.
    fa_lo, fa_hi = _safe_range(anchor["fa_ratio"], 0.008, 0.0, 0.30)

    phi = rng.uniform(phi_lo, phi_hi, n_samples)
    sp_percent = rng.uniform(sp_lo, sp_hi, n_samples)
    fa_ratio = rng.uniform(fa_lo, fa_hi, n_samples)

    # Mix design on a ~1 m^3 (1000 L) basis, with mild total-volume variation so
    # composition columns are not perfectly collinear with phi (as in real mixes).
    rho_cement, rho_fly_ash = 3.15, 2.30  # kg/L
    v_total = rng.uniform(980.0, 1015.0, n_samples)
    vp_l = phi * v_total
    vw_l = (1.0 - phi) * v_total
    water_kg = vw_l * 1.0  # water density ~1 kg/L
    binder_kg = vp_l / ((1.0 - fa_ratio) / rho_cement + fa_ratio / rho_fly_ash)
    cement_kg = (1.0 - fa_ratio) * binder_kg
    fly_ash_kg = fa_ratio * binder_kg
    sp_kg = sp_percent / 100.0 * binder_kg
    w_b = water_kg / binder_kg
    sp_binder_ratio = sp_kg / binder_kg

    # Virtual maximum packing fraction from transferable drivers only.
    phi_max_eff = (
        0.585
        + 0.030 * np.tanh((sp_percent - 0.45) / 0.35)
        - 0.012 * (fa_ratio - anchor["fa_ratio"].mean())
    )
    phi_max_eff = np.clip(phi_max_eff, phi + 0.035, 0.70)

    # Label law calibrated to the ONE real reference we have (Lian Table 6): a
    # 2-driver fit gives log(tau) ~= -1.79 + 6.14*phi - 2.83*sp (R^2=0.84 on the
    # 16 rows). We reproduce that first-order behaviour so a model trained on the
    # synthetic set transfers to the real operating regime, but we deliver the phi
    # effect THROUGH a YODEL/free-volume packing divergence term (not a bare
    # linear phi) so that packing/fmax features are genuinely predictive and the
    # mechanism-feature ablation is meaningful. SP also modulates phi_max_eff, so
    # SP acts partly through packing (dispersion) and partly directly, matching
    # the YODEL view of superplasticizer action.
    #
    # NOTE (honesty): because the constants are fit to Table 6, the anchor is a
    # calibration/consistency reference here, not a fully blind test. A truly
    # independent validation requires the real full dataset (documented in
    # limitations).
    # Real-calibrated log-linear backbone (2-driver fit on Table 6):
    #   log(tau) = -1.79 + 6.14*phi - 2.83*sp   (R^2=0.84 on 16 rows)
    # We keep this exact level/slope so magnitude AND driver hierarchy transfer to
    # the real regime, then add a SMALL zero-mean packing refinement so YODEL/
    # free-volume features still carry independent signal for the ablation without
    # distorting the calibrated level.
    C0, A_PHI, A_SP = -1.794, 6.140, -2.832
    phi_c = 0.33
    packing_gap = np.maximum(phi_max_eff - phi, 0.02)
    # Free-volume / packing divergence (YODEL / Krieger-Dougherty flavour).
    g_pack = -np.log(np.clip(packing_gap / (phi_max_eff + 1e-9), 1e-6, 1.0))
    g_pack_bump = g_pack - float(np.mean(g_pack))          # zero-mean refinement
    fa_term = -0.80 * (fa_ratio - anchor["fa_ratio"].mean())
    log_tau = (
        C0
        + A_PHI * phi
        + A_SP * sp_percent
        + 0.15 * g_pack_bump       # small nonlinear packing signal (mechanism)
        + fa_term
    )
    noise = rng.normal(0.0, 0.12, size=n_samples)          # log-scale noise
    yield_stress = np.clip(np.exp(log_tau + noise), 0.03, None)

    df = pd.DataFrame(
        {
            "sample_id": [f"synthetic_yield_v1_{i:04d}" for i in range(n_samples)],
            "phi": phi,
            "sp_percent": sp_percent,
            "w_b": w_b,
            "fa_ratio": fa_ratio,
            "cement_kg_m3": cement_kg,
            "fly_ash_kg_m3": fly_ash_kg,
            "water_kg_m3": water_kg,
            "sp_kg_m3": sp_kg,
            "binder_kg_m3": binder_kg,
            "sp_binder_ratio": sp_binder_ratio,
            "Vp_L": vp_l,
            "Vw_L": vw_l,
            "phi_max_eff_reference": phi_max_eff,
            "yield_stress": yield_stress,
            "data_fidelity": "synthetic_low_fidelity",
            "source": LIAN_2025_SOURCE["source_id"],
        }
    )

    report = {
        "dataset_kind": "composition_grounded_synthetic_low_fidelity",
        "random_state": random_state,
        "n_samples": int(n_samples),
        "anchor_source": LIAN_2025_SOURCE,
        "anchor_rows": int(len(anchor)),
        "target_column": "yield_stress",
        "feature_policy": {
            "allowed_model_inputs": [
                "phi",
                "sp_percent",
                "w_b",
                "fa_ratio",
                "cement_kg_m3",
                "fly_ash_kg_m3",
                "water_kg_m3",
                "sp_kg_m3",
                "binder_kg_m3",
                "sp_binder_ratio",
                "Vp_L",
                "Vw_L",
            ],
            "reference_only_not_model_input": ["phi_max_eff_reference"],
            "leakage_columns": ["yield_stress", "plastic_viscosity_pa_s", "phi_max_eff_reference"],
            "design_note": (
                "Only composition/mix-proportion features are generated. Invented particle/process "
                "columns (d50, psd_width, specific_surface, temperature, mixing_*, rest_time, "
                "curing_agent) were removed: they varied in training but are constant/absent in the "
                "real Table 6 anchor, causing severe synthetic->real distribution shift and wrecking "
                "anchor generalization. Training and anchor now share one composition schema."
            ),
        },
        "generation_equations": {
            "mix_design": "1 m^3 basis: Vp=phi*Vtot, Vw=(1-phi)*Vtot; binder=Vp/((1-fa)/3.15+fa/2.30); cement=(1-fa)*binder; fly_ash=fa*binder; water=Vw; sp_kg=SP%/100*binder; w_b=water/binder (derived).",
            "phi_max_eff": "0.585 + 0.030*tanh((SP-0.45)/0.35) - 0.012*(fa_ratio-fa_anchor_mean); clipped to phi+0.035..0.70",
            "yield_stress": "log(tau) = -1.794 + 6.140*phi - 2.832*SP + 0.15*(g_pack - mean(g_pack)) - 0.80*(fa-fa_anchor_mean) + N(0,0.12); g_pack=-log((phi_max_eff-phi)/phi_max_eff). Backbone calibrated to Lian Table 6 2-driver fit (R2=0.84); packing term is a small zero-mean mechanism refinement.",
            "anchor_calibration_note": "The log-linear backbone constants (-1.794, 6.140, -2.832) are fit to the Table 6 anchor, so the anchor is a calibration/consistency reference here, NOT a fully blind test. Independent validation requires the real full dataset.",
            "notes": "Low-fidelity, real-calibrated, transferable-feature-only approximation for AutoML workflow development, not final industrial truth.",
        },
        "mechanistic_basis": [
            {
                "name": "Virtual maximum packing fraction",
                "paper_or_source": LIAN_2025_SOURCE["paper"],
                "doi": LIAN_2025_SOURCE["doi"],
                "relationship": "fmax represents flocculation/entrapped-water effects and is modulated by PCE dosage and fly-ash fraction.",
            },
            {
                "name": "Packing-controlled yield stress growth",
                "paper_or_source": LIAN_2025_SOURCE["paper"],
                "doi": LIAN_2025_SOURCE["doi"],
                "relationship": "yield stress rises strongly as solid volume fraction approaches the maximum packing limit.",
            },
        ],
        "limitations": [
            "Synthetic data are for workflow development and model-structure research only.",
            "Only phi/sp_percent/w_b/fa_ratio and derived composition columns are modeled; particle/process history is not simulated.",
            "The real Table 6 anchor (16 rows) remains the only real ground truth; a fabricated set must never replace it as the validation signal.",
            "The real industrial dataset must replace or calibrate this dataset before final claims.",
        ],
    }
    return df, report


def ensure_default_synthetic_yield_data(
    output_dir: str | Path | None = None,
    n_samples: int = 800,
    random_state: int = 20250629,
    overwrite: bool = False,
) -> dict[str, str]:
    out_dir = Path(output_dir) if output_dir is not None else DEFAULT_SYNTHETIC_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    anchor_path = out_dir / DEFAULT_ANCHOR_PATH.name
    synthetic_path = out_dir / DEFAULT_SYNTHETIC_PATH.name
    report_path = out_dir / DEFAULT_SYNTHETIC_REPORT_PATH.name

    if overwrite or not anchor_path.exists():
        table6_anchor_dataframe().to_csv(anchor_path, index=False)
    if overwrite or not synthetic_path.exists() or not report_path.exists():
        df, report = generate_synthetic_yield_data(n_samples=n_samples, random_state=random_state)
        df.to_csv(synthetic_path, index=False)
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    return {
        "anchor_path": str(anchor_path.resolve()),
        "synthetic_path": str(synthetic_path.resolve()),
        "synthetic_report_path": str(report_path.resolve()),
    }


if __name__ == "__main__":
    paths = ensure_default_synthetic_yield_data(overwrite=True)
    print(json.dumps(paths, ensure_ascii=False, indent=2))
