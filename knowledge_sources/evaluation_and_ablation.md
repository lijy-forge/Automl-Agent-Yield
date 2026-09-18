# Evaluation And Mechanism Ablation

Scope: repository evaluation rules and acceptance semantics. Numerical model
performance must come from an actual run artifact, never from this document.

## Out-of-fold evaluation

Training-set fit metrics are not accepted as generalization evidence. Baseline
and candidate comparisons use held-out or out-of-fold predictions under the
same split definition. The split seed and fold count belong in the run record.

## Metrics

Regression reports may include RMSE, MAE, and R2. RMSE preserves the target
unit and emphasizes larger errors; MAE is more robust to a small number of
large residuals; R2 is scale-relative and can be negative on held-out data.

## Fixed baseline

`run_fixed_baseline_eval` provides a deterministic reference path over the
normalized yield schema. It is an engineering baseline, not an Agent-generated
model and not a substitute for an external holdout.

## Candidate benchmark

`run_candidate_benchmark` evaluates candidate strategies with the repository's
benchmark implementation and records a selection audit. Candidate selection
must compare like-for-like folds and cannot rely only on a generated rationale.

## Mechanism ablation

When a generated model claims a physics mechanism helps, compare
`with_mechanism` and `without_mechanism` under the same folds, preprocessing,
features, and budget. Record delta RMSE and delta R2. A marginal change should
not be presented as strong physical validation.

## Evaluation set boundary

The offline Agent regression set measures workflow behavior such as tool
routing and artifact checks. It does not measure a real LLM when the report
records zero real model calls, and it does not establish production accuracy.
