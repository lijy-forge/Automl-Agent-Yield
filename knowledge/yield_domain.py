"""Domain prompt material for yield-stress AutoML generation."""

from __future__ import annotations


YIELD_DOMAIN_CONTEXT = """
# Yield-Stress AutoML Domain Context

Task: predict static yield stress tau0 / yield_stress for high-solid-content
slurries from available formulation and process variables.

The final model is intentionally NOT fixed. The code-generation agent may use
data-driven models, physics-informed neural networks, symbolic/structured
models, multi-fidelity training, ensembles, or other externally searched
approaches, as long as the implementation is runnable and passes the verifier.

Current temporary datasets:
- Lian 2025 layout: Phi, SP_percent -> Tau0_Pa.
- Zhou 1999 layout: phi, d_s_um, optional powder -> tau_Pa.
- Future industrial layout may include phi, d50, sigma_d, Emix, temperature,
  and yield_stress.

Core engineering constraints:
- Never use target or auxiliary true-physics columns as model inputs:
  Tau0_Pa, tau_Pa, yield_stress, phi_max, m1_true, m1_lf.
- Fit preprocessing only on train folds/splits.
- Evaluate with held-out or out-of-fold predictions, not training-set metrics.
- Save a mechanism report. If any mechanistic equation or physics constraint is
  used, list its paper/source, the exact formula/relationship used, the data
  columns it maps to, and why it is appropriate. If no mechanism is used, state
  that explicitly.
- Preserve physical validity: predicted yield stress must be finite and
  non-negative. If a hidden physical variable is predicted, report validity
  diagnostics such as phi_m > phi or bounded m1_eff.

Mechanistic directions that may be useful, but are not mandatory:
- YODEL-type suspension yield-stress relations from Flatt and Bowen.
- Lian 2025 cement-paste relation using phi and superplasticizer-controlled
  packing state.
- Bingham, Herschel-Bulkley, Casson, DLVO-inspired, thixotropic structural
  kinetics, or other rheology-informed relations found by search.
- Physics-guided data augmentation or multi-fidelity training when low-fidelity
  synthetic data is available.
"""


DEFAULT_YIELD_SEARCH_QUERIES = [
    "yield stress model suspensions YODEL Flatt Bowen 2006",
    "physics informed neural network yield stress prediction slurry",
    "multi fidelity neural network yield stress cement paste Lian 2025",
    "rheology informed machine learning yield stress high solid slurry",
]


REFERENCE_MECHANISM_NOTES = [
    {
        "name": "YODEL yield-stress relation",
        "paper": "Flatt R. J. and Bowen P. (2006), YODEL: A Yield Stress Model for Suspensions, Journal of the American Ceramic Society.",
        "basis": "Represents yield stress as a particle-network contribution that rises nonlinearly as solid volume fraction approaches maximum packing.",
        "typical_formula": "tau = m1 * phi * (phi - phi_c)^2 / (phi_m * (phi_m - phi))",
        "status": "reference only; generated code may use, adapt, or reject it based on data columns.",
    },
    {
        "name": "Lian-style packing-state relation",
        "paper": "Lian et al. (2025), Materials 18, 2983.",
        "basis": "Uses phi and superplasticizer dosage to infer an effective packing state, then computes yield stress through a constrained physical layer.",
        "typical_formula": "tau = m1 * phi^3 / (phi_max * (phi_max - phi))",
        "status": "reference only; useful for the temporary Lian dataset.",
    },
    {
        "name": "PI-MFNN hidden-variable physical layer",
        "paper": "Project design documents and multi-fidelity PINN implementations in Yield-value-prediction.",
        "basis": "Neural network predicts a hidden physical variable such as phi_m or m1_eff, while the physical layer computes tau0 and enforces hard bounds.",
        "typical_formula": "phi_m = phi + margin + sigmoid(z) * upper_range; tau0 = physical_layer(phi, phi_m, ...)",
        "status": "reference only; generated code must document any hidden-variable mechanism it uses.",
    },
]

