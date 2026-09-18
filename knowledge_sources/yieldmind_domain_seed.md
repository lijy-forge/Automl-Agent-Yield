# YieldMind Domain Seed

YODEL is a yield-stress mechanism family that relates the static yield stress
of dense suspensions to solid volume fraction, maximum packing density `phi_m`,
and contact-network effects. In this repository it is treated as a candidate
mechanism only when the available columns support packing-style features.

Mechanism ablation is required when a generated model claims that a physics
mechanism improves prediction. The same protocol should compare
`with_mechanism` and `without_mechanism`, record delta RMSE and delta R2, and
avoid presenting marginal gains as strong physical validation.

YieldMind keeps external evidence separate from generated claims. A report that
uses a mechanism should cite concrete knowledge chunks, source paths, document
versions, and index versions. A chunk ID proves traceability, not semantic
truth; whether the text truly supports a claim still needs a labeled check or
human review.
