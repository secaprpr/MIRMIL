# MIR-MIL V3 architecture diagnosis and replacement hypothesis

## Scope

This diagnosis uses the frozen BRACS3 UNI protocol, independent seeds
2024/2025/2026, a 4096-instance budget, and the official train/validation/test
split.  The objective is an overall single-model improvement in accuracy,
balanced accuracy, macro-AUC, and macro-F1, rather than an ensemble or an
AUC-only gain.

## Empirical failure

The strongest completed baseline, AC-MIL, has official-test means of 0.6820
accuracy, 0.6670 balanced accuracy, 0.8529 macro-AUC, and 0.6620 macro-F1.
V2 A reaches 0.8511 macro-AUC but only 0.6398 accuracy and 0.5848 macro-F1.
Pairwise dropout 0.05 looks strong on validation (0.9153 AUC, 0.8051 accuracy,
0.7529 balanced accuracy, 0.7548 F1) but falls to 0.8329 AUC, 0.6705 accuracy,
0.6416 balanced accuracy, and 0.6231 F1 on test.

The error is concentrated in the intermediate AT class.  On official test,
mean recalls for V2 A are 0.844/0.246/0.719 for classes 0/1/2.  AC-MIL obtains
0.667/0.522/0.813.  V2 predicts class 1 on only 12.6% of slides although class
1 represents 26.4% of the test set.  Validation-only vector scaling recovers
V2 A macro-F1 to 0.6769, but macro-AUC remains 0.8477.  Calibration is therefore
secondary: it repairs much of the operating point but not the missing ranking
geometry.

## Architectural localization

The frozen MIR-MIL state is class agnostic until after bag compression:

1. a shared encoder maps patches to hidden features;
2. shared response bases form a 128-dimensional composition statistic;
3. eight shared tail scores and twelve shared local routes add 392 dimensions;
4. the resulting 520-dimensional state is passed to a class predictor;
5. moment-token and pairwise branches retrieve separate evidence and add
   residual logits after the main prediction already exists.

This is late class conditioning.  If two empirical patch distributions have
similar shared moments and routes but differ in a small class-relevant mode,
the compression can erase that distinction before the classifier sees class
semantics.  The intermediate class is especially vulnerable because it must
retain evidence against both neighboring extremes.

The V2 pairwise branch does not correct this bottleneck coherently.  It learns
new keys, values, class queries, class factors, and biases in a coordinate
system independent of the main potential, then adds its logits with a scalar
weight.  Dropout and rank sweeps change which coordinate system dominates but
cannot make the two geometries identifiable.  This explains the recurring
trade-off between ranking, decision metrics, and seed stability.

The current architecture also ignores label topology.  BRACS3 labels are
ordered, but the main predictor estimates three unconstrained class logits.
The easy class-0/class-2 separation can dominate training while class 1 has no
explicit interval between two adjacent boundaries.  The previous ordinal
experiment did not test this hypothesis cleanly: it added an ordinal residual
after the same class-agnostic compression, so it retained both root causes.

## Rejected explanations

- Feature quality is not the main limitation: AC-MIL is stronger with the same
  frozen UNI features.
- More residual capacity is not the answer: rank, dropout, residual strength,
  class-conditioned tokens, sparse heads, and ordinal heads repeatedly trade
  one metric for another.
- Pairwise scale is not the primary bug: V2 normalizes incidence aggregation by
  `C-1`, and it preserves PANDA performance.
- Pure calibration is insufficient: it repairs F1 but leaves mean AUC below
  AC-MIL.

## V3 hypothesis: ordinal distributional risk

V3 should replace the shared-state-plus-residual stack with one predictor over
the empirical patch measure.  Let the encoder learn a scalar patch severity
field `s_theta(x)`.  For a bag measure `mu`, define the normalized entropic risk

`R_tau(mu) = tau * log E_mu[exp(s_theta(x) / tau)]`.

This operator is permutation invariant, normalized for bag size, monotone under
first-order severity dominance, and interpolates between average and focal
high-severity evidence through one interpretable temperature.

Ordered thresholds `b_1 < ... < b_(C-1)` define cumulative probabilities

`P(y > k | mu) = sigmoid((R_tau(mu) - b_k) / T)`.

Class probabilities are obtained by adjacent differences of the cumulative
probabilities.  They are valid by construction, and the intermediate class is
represented explicitly as probability mass between two learned boundaries.
There is no main head plus residual head: encoder, risk functional, and ordered
boundaries form the entire deployed classifier.

This is a core architectural replacement, not a module addition.  It also
provides a clean theoretical object for analysis: invariance, monotonicity,
bag-size normalization, limiting behavior in `tau`, and ordered decision
regions can all be stated and tested.

## Falsifiable experiment

Three replacement models isolate the hypothesis:

1. mean severity risk plus cumulative link: tests ordinal structure alone;
2. entropic severity risk plus unconstrained softmax: tests distributional risk
   alone;
3. entropic severity risk plus cumulative link: full V3.

All variants replace the current potential/readout rather than adding to it.
They use the same UNI features, training split, sampler, budget, epochs, and
three seeds.  AC-MIL validation is the gate: 0.9329 AUC, 0.7641 accuracy,
0.7392 balanced accuracy, and 0.7317 macro-F1.  A candidate proceeds directly
to official test only if it is Pareto-competitive with these four means and
does not rely on one exceptional seed.

If the full model does not improve class-1 validation recall and class-1 AUC
over both isolated variants, the core ordinal-risk hypothesis is rejected.
