# Yield Data Contract And Leakage Controls

Scope: project constraints from `knowledge/yield_domain.py`,
`knowledge/yield_schema.py`, and the workflow validation code.

## Supported layouts

The Lian-style layout maps `Phi` and `SP_percent` to target `Tau0_Pa`. The
Zhou-style layout maps `phi`, `d_s_um`, and optional `powder` to target
`tau_Pa`. A future industrial layout may include `phi`, `d50`, `sigma_d`,
`Emix`, temperature, and target `yield_stress`.

## Target leakage denylist

Target and auxiliary true-physics columns must never become model inputs. The
project denylist includes `Tau0_Pa`, `tau_Pa`, `yield_stress`, `phi_max`,
`m1_true`, and `m1_lf`. Hidden-variable predictions must be generated from
allowed features, not copied from true hidden-state columns.

## Split-aware preprocessing

Imputation, scaling, encoding, and learned feature transforms must be fit only
on each training fold or training split. Fitting preprocessing on all rows
before cross-validation leaks validation distribution information and makes
the reported metric optimistic.

## Units and schema checks

Column aliases alone are insufficient. A run should preserve the target unit,
check numeric coercion, report missingness, and reject an unsupported schema
instead of silently guessing the target. Yield-stress predictions are expected
to be finite and non-negative.

## Distribution shift

Random folds do not represent every deployment scenario. When batches,
formulations, powder types, or acquisition periods exist, grouped or temporal
holdouts should be considered. A model can score well under random splits and
still fail on a new formulation domain.
