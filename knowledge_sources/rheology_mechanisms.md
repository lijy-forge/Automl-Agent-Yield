# Rheology Mechanism Candidates

Scope: project knowledge derived from `knowledge/yield_domain.py`. These notes
describe candidate mechanisms and implementation constraints. They are not a
claim that any mechanism has already improved a held-out metric.

## YODEL candidate

The repository records YODEL as a candidate family for dense suspension yield
stress. It links solid volume fraction `phi`, a maximum packing state `phi_m`,
and particle contact-network effects. A YODEL-style relation should only be
used when the available columns and units support the mapped variables.

## Lian-style packing state

For the temporary Lian layout, the known columns are `Phi`, `SP_percent`, and
the target `Tau0_Pa`. The project notes describe a Lian-style physical layer in
which superplasticizer dosage may control an effective packing state. This is
a candidate relation, not permission to use target-derived hidden columns.

## Hidden physical variables

A physics-informed model may predict a hidden variable such as `phi_m` or
`m1_eff`, then compute yield stress through a constrained physical layer.
Validity diagnostics must verify conditions such as `phi_m > phi`, bounded
`m1_eff`, finite output, and non-negative predicted yield stress.

## Alternative rheology families

Bingham, Herschel-Bulkley, Casson, DLVO-inspired relations, and thixotropic
structural kinetics are search directions rather than mandatory choices. The
generated model must document the exact equation, source, column mapping, and
reason for use, or explicitly state that no mechanism was used.

## Mechanism evidence boundary

A mechanism name or plausible formula does not establish predictive benefit.
The claim requires a controlled ablation under the same data split and
preprocessing protocol. Physical plausibility and held-out predictive quality
are separate acceptance dimensions.
