# MIR-MIL v1 Method

Status: archived method document for the current MIR-MIL v1 line.

Repository tag: `archive/mirmilv1-method`

Primary implementation:

- `modules/MIR_MIL/mir_mil.py`
- `configs/MIR_MIL.yaml`
- `utils/model_utils.py`
- `utils/loop_utils.py`

This document describes MIR-MIL v1 at paper-method level. It is intended to be the stable reference for the current method before moving to a larger architecture redesign.

## 1. Problem setting

Whole-slide image classification is formulated as multiple instance learning. A slide is represented by a bag of pre-extracted patch features

\[
B=\{x_i\}_{i=1}^{N}, \qquad x_i\in \mathbb{R}^{d},
\]

where \(N\) varies across slides. The slide has a single label

\[
y \in \{1,\ldots,C\}.
\]

MIR-MIL v1 assumes that patch features have already been extracted by an external frozen encoder such as ResNet-50 or UNI. The model does not train the feature extractor and does not access raw WSI pixels during MIL training.

The objective is to learn a permutation-invariant slide-level classifier

\[
f_\theta(B) \in \mathbb{R}^{C},
\]

while also providing a closed-form patch-level influence response that explains how infinitesimal changes to the empirical bag measure affect a target class score.

## 2. Core idea

MIR-MIL v1 treats each WSI bag as an empirical measure

\[
\mu_B = \frac{1}{N}\sum_{i=1}^{N}\delta_{x_i}.
\]

Instead of applying standard attention pooling directly to the patch features, MIR-MIL maps the bag measure to a compact neural measure state:

\[
s_\theta(\mu_B) \in \mathbb{R}^{D_s}.
\]

A neural potential then maps this state to slide logits:

\[
z = V_\theta(s_\theta(\mu_B)).
\]

The method is built around three principles:

1. Represent the slide as a learned measure state rather than as an ordered sequence.
2. Preserve both global distributional information and localized routed evidence.
3. Keep the state differentiable with respect to the underlying empirical measure, enabling a closed-form measure influence response.

The accepted v1 extension adds a generic moment-token evidence readout as a residual logit branch. This branch improves evidence preservation without changing the fixed patch features or introducing dataset-specific rules.

## 3. Patch encoder

Each patch feature is first mapped to a hidden representation:

\[
h_i = E_\theta(x_i) \in \mathbb{R}^{H}.
\]

In the implementation, \(E_\theta\) is a lightweight MLP:

\[
E_\theta(x) = \mathrm{Dropout}(\sigma(Wx+b)).
\]

Coordinates are optional. If coordinates are enabled, the input is the concatenation of feature and normalized coordinate vectors. In the current fixed-feature PANDA/BRACS experiments, coordinate dimension is usually zero.

MIR-MIL supports optional grouped \(L_2\) input normalization for pre-extracted features, but this is not part of the accepted BRACS3 v1 improvement.

## 4. Neural measure state

The measure state contains three main components:

1. global composition statistics;
2. smooth tail statistics;
3. local routed measure statistics.

These components are concatenated to form the final state:

\[
s_\theta(\mu_B)
=
\left[
s_{\mathrm{comp}},
s_{\mathrm{tail}},
s_{\mathrm{local}},
s_{\mathrm{anchor}}
\right].
\]

The anchor state is optional and disabled in the main archived configuration.

### 4.1 Composition statistics

A response basis maps each encoded patch to a sketch vector:

\[
\phi_i = \Phi_\theta(h_i) \in \mathbb{R}^{K}.
\]

The first-order composition state is the empirical mean:

\[
s_{\mathrm{comp}} =
\mathbb{E}_{x\sim\mu_B}[\Phi_\theta(E_\theta(x))]
=
\frac{1}{N}\sum_{i=1}^{N}\phi_i.
\]

The implementation also supports a second-order composition state:

\[
s_{\mathrm{var}} =
\mathbb{E}_{x\sim\mu_B}
\left[
(\phi(x)-s_{\mathrm{comp}})^2
\right],
\]

but the main BRACS3 v1 configuration uses first-order composition in the base state and uses second-order statistics in the residual moment-token branch.

### 4.2 Smooth tail statistics

Many WSI labels depend on rare but diagnostic regions. To retain soft extreme evidence, MIR-MIL computes \(M\) learned tail scores:

\[
a_{im}=T_{\theta,m}(h_i).
\]

For each tail channel, the state stores a smooth maximum:

\[
s_{\mathrm{tail},m}
=
\tau_{\mathrm{tail}}
\log
\left(
\frac{1}{N}\sum_{i=1}^{N}
\exp(a_{im}/\tau_{\mathrm{tail}})
\right).
\]

For small \(\tau_{\mathrm{tail}}\), this approaches a maximum; for larger \(\tau_{\mathrm{tail}}\), it behaves more like a soft average. This gives the model a differentiable way to preserve rare high-response evidence without hard top-\(k\) selection.

### 4.3 Local routed measure statistics

Global means and smooth tails can still dilute spatially or morphologically localized evidence. MIR-MIL therefore supports \(R\) learned local routes. Each route \(r\) has:

- a route score \(u_{ir}\);
- a route basis vector \(g_i\in\mathbb{R}^{D_r}\).

The route-specific normalized weight is:

\[
\alpha_{ir}
=
\frac{\exp(u_{ir}/\tau_{\mathrm{local}})}
{\sum_{j=1}^{N}\exp(u_{jr}/\tau_{\mathrm{local}})}.
\]

The local state for route \(r\) is:

\[
m_r =
\sum_{i=1}^{N}\alpha_{ir}g_i.
\]

All route states are flattened and concatenated:

\[
s_{\mathrm{local}} = [m_1,\ldots,m_R].
\]

This mechanism keeps the method permutation-invariant while allowing different learned routes to focus on different local evidence modes.

## 5. Neural potential

The slide logits are produced by a neural potential over the measure state:

\[
z_{\mathrm{base}} = V_\theta(s_\theta(\mu_B)).
\]

MIR-MIL v1 supports several potential variants. The archived BRACS3/PANDA v1 line uses the adaptive multiscale potential:

\[
V_\theta(s)
=
V_{\mathrm{global}}(s_{\mathrm{global}})
+
\gamma(s)
\cdot
\lambda_{\mathrm{local}}
V_{\mathrm{local}}(s_{\mathrm{local}}),
\]

where:

- \(s_{\mathrm{global}}\) contains the composition and tail states;
- \(s_{\mathrm{local}}\) contains routed local states;
- \(\gamma(s)\in(0,1)^C\) is a learned class-wise gate;
- \(\lambda_{\mathrm{local}}\) is a learned or initialized local residual scale.

This design keeps a strong global distributional path while allowing localized routed evidence to contribute when useful.

## 6. Moment-token evidence readout

The accepted MIR-MIL v1 improvement is the moment-token residual readout. It is a dataset-agnostic multi-token attention module over encoded patch representations. It is not tied to BRACS, PANDA, a fixed class count, or any class semantics.

Let \(K_t\) be the number of learned evidence tokens. For token \(t\), the module has a learned token query \(q_t\). Encoded patches are projected to keys and values:

\[
k_i = W_k h_i,\qquad v_i = W_v h_i.
\]

Token attention is computed over patches:

\[
\alpha_{it}
=
\frac{
\exp(k_i^\top q_t / (\sqrt{d_t}\tau_t))
}{
\sum_{j=1}^{N}
\exp(k_j^\top q_t / (\sqrt{d_t}\tau_t))
}.
\]

The original multi-token readout stores one weighted mean per token:

\[
\bar{v}_t = \sum_{i=1}^{N}\alpha_{it}v_i.
\]

The moment-token readout additionally stores a weighted second moment and variance:

\[
\overline{v^2}_t
=
\sum_{i=1}^{N}\alpha_{it}v_i^2,
\]

\[
\mathrm{Var}_t
=
\max(\overline{v^2}_t - \bar{v}_t^2, 0).
\]

The token summary is:

\[
r_t = [\bar{v}_t,\mathrm{Var}_t].
\]

All token summaries are concatenated and passed to a lightweight readout head:

\[
z_{\mathrm{moment}}
=
W_o\,\mathrm{Dropout}
\left(
\mathrm{LayerNorm}
([r_1,\ldots,r_{K_t}])
\right).
\]

The final v1 logits are:

\[
z
=
z_{\mathrm{base}}
+
\lambda_{\mathrm{moment}}z_{\mathrm{moment}}.
\]

In the accepted BRACS3 v1 configuration:

- `moment_token_weight = 0.1`;
- `moment_token_count = 4`;
- `moment_token_dim = 64`;
- `moment_token_readout_dim = 128`;
- `moment_token_temperature = 1.0`;
- `moment_token_dropout = 0.0`.

### Why moment-token is part of v1

The moment-token branch matches the main MIR-MIL philosophy: WSI labels often depend on distributions, not only on a single pooled feature. It preserves token-specific mean evidence and within-token heterogeneity.

This explains its behavior across datasets:

- PANDA benefits because prostate grade is strongly distributional; second-order token statistics preserve grade-related heterogeneity.
- BRACS benefits because fine-grained benign/atypical/malignant categories may depend on localized evidence and heterogeneous tissue patterns, which can be diluted by a single global state.

The branch is residual. Therefore the original measure-state potential remains the primary path, and the moment-token readout adds evidence that the compact state may underrepresent.

## 7. Final prediction

The forward pass is:

1. normalize the input bag if requested;
2. encode patches \(h_i=E_\theta(x_i)\);
3. compute basis, tail scores, and local-route scores;
4. construct the measure state \(s_\theta(\mu_B)\);
5. compute base logits \(z_{\mathrm{base}}=V_\theta(s_\theta(\mu_B))\);
6. optionally compute residual evidence logits;
7. return final logits \(z\).

For the accepted v1 model:

\[
z
=
V_\theta(s_\theta(\mu_B))
+
0.1\,z_{\mathrm{moment}}.
\]

Class probabilities are:

\[
p(y=c\mid B)=\mathrm{softmax}(z)_c.
\]

## 8. Training objective

The default supervised objective is cross-entropy:

\[
\mathcal{L}_{\mathrm{ce}}
=
-\log p(y\mid B).
\]

MIR-MIL v1 also supports optional regularizers. The general training objective is:

\[
\mathcal{L}
=
\mathcal{L}_{\mathrm{ce}}
+
\lambda_{\mathrm{ord}}\mathcal{L}_{\mathrm{ord}}
+
\lambda_{\mathrm{stab}}\mathcal{L}_{\mathrm{stab}}
+
\lambda_{\mathrm{sub}}\mathcal{L}_{\mathrm{sub}}
+
\lambda_{\mathrm{lip}}\mathcal{L}_{\mathrm{lip}}
+
\lambda_{\mathrm{proto}}\mathcal{L}_{\mathrm{proto}}
+
\lambda_{\mathrm{margin}}\mathcal{L}_{\mathrm{margin}}.
\]

Only a subset is active in the archived BRACS3 v1 setting.

### 8.1 Classification loss

The core loss is standard multiclass cross-entropy. This supports any number of classes \(C\ge 2\).

### 8.2 Stability loss

MIR-MIL supports bag augmentation through patch dropout and feature noise. If enabled, the model penalizes prediction changes:

\[
\mathcal{L}_{\mathrm{stab}}
=
\left\|
f_\theta(B)-f_\theta(\tilde{B})
\right\|_2^2.
\]

This is intended to improve robustness to patch sampling variation.

### 8.3 Ordinal loss

For ordered-label tasks, MIR-MIL supports a cumulative-distribution loss:

\[
\mathcal{L}_{\mathrm{ord}}
=
\left\|
\mathrm{CDF}(\mathrm{softmax}(z))
-
\mathrm{CDF}(y)
\right\|_2^2.
\]

This is disabled by default and should only be used when label order is scientifically meaningful. It is not part of the accepted BRACS3 moment-token v1 result.

### 8.4 Subset consistency

MIR-MIL supports a consistency loss between full-bag logits and subset-view logits:

\[
\mathcal{L}_{\mathrm{sub}}
=
\mathrm{KL}
\left(
\mathrm{softmax}(z_{\mathrm{full}}/T)
\;\|\;
\mathrm{softmax}(z_{\mathrm{sub}}/T)
\right).
\]

This was evaluated as an experiment and is not part of the accepted v1 result.

### 8.5 Margin loss

A generic multiclass logit-margin auxiliary loss is implemented:

\[
\mathcal{L}_{\mathrm{margin}}
=
\frac{1}{C-1}
\sum_{c\ne y}
\left[
m + z_c-z_y
\right]_+^2.
\]

This is default-disabled and was rejected as a robust BRACS3 improvement because it increased seed sensitivity.

## 9. Measure influence response

A key property of MIR-MIL is that the model can compute a patch-level influence response in closed form for the base measure-potential path.

For a target class \(c\), define the explained class score as a logit margin:

\[
F_c(\mu_B)
=
z_c
-
\log\sum_{k\ne c}\exp(z_k).
\]

The measure influence response of a point \(x\) is the first-order change in \(F_c\) when the empirical measure is infinitesimally contaminated by a point mass at \(x\):

\[
R_c(x;\mu_B)
=
\left.
\frac{d}{d\epsilon}
F_c\left((1-\epsilon)\mu_B+\epsilon\delta_x\right)
\right|_{\epsilon=0}.
\]

Because the state is differentiable and composed of explicit empirical-measure functionals, MIR-MIL computes:

\[
R_c(x;\mu_B)
=
\left\langle
\nabla_s F_c(s_\theta(\mu_B)),
\frac{d}{d\epsilon}
s_\theta((1-\epsilon)\mu_B+\epsilon\delta_x)
\right\rangle_{\epsilon=0}.
\]

The implementation decomposes this response into:

\[
R_c
=
R_{\mathrm{comp}}
+
R_{\mathrm{var}}
+
R_{\mathrm{tail}}
+
R_{\mathrm{local}}
+
R_{\mathrm{anchor}}.
\]

This gives a patch-level attribution score that is connected to the model's measure-state decision, rather than being an unrelated post-hoc explanation.

The response can be audited by finite differences:

\[
\frac{
F_c((1-\epsilon)\mu_B+\epsilon\delta_x)-F_c(\mu_B)
}{\epsilon}
\approx
R_c(x;\mu_B).
\]

The repository includes finite-difference and integrated functional attribution utilities for this purpose.

Important limitation: the closed-form MIR response applies to the measure-potential path. Residual readout branches such as moment-token contribute to final logits, but the original closed-form response decomposition is centered on the explicit measure state.

## 10. Permutation invariance

MIR-MIL is permutation-invariant because every bag-level operation is based on symmetric reductions over instances:

- empirical means;
- log-sum-exp smooth maxima;
- softmax-normalized route means;
- token attention sums over instances.

For any permutation \(\pi\):

\[
f_\theta(\{x_i\}_{i=1}^{N})
=
f_\theta(\{x_{\pi(i)}\}_{i=1}^{N}).
\]

## 11. Complexity

Let:

- \(N\) be the number of sampled patches;
- \(H\) be hidden dimension;
- \(K\) be sketch dimension;
- \(M\) be number of tail channels;
- \(R\) be number of local routes;
- \(D_r\) be route dimension;
- \(K_t\) be number of moment tokens;
- \(D_t\) be token dimension;
- \(D_v\) be token value dimension.

The base state construction is linear in the bag size:

\[
O(NH + NK + NM + NRD_r).
\]

The moment-token branch adds:

\[
O(NK_tD_t + NK_tD_v).
\]

There is no \(O(N^2)\) self-attention over all patches. This makes MIR-MIL v1 practical for large bags with sampled patch budgets such as 4096 instances per slide.

## 12. V1 archived configuration

The archived v1 BRACS3/PANDA line uses:

- frozen pre-extracted features;
- official dataset splits;
- no raw WSI training;
- no feature extractor fine-tuning;
- AdamW optimizer;
- cross-entropy as the main loss;
- macro-AUC as the primary validation-selection metric;
- adaptive multiscale potential;
- local routed measures;
- moment-token residual readout enabled.

Representative core model settings:

```yaml
Model:
  hidden_dim: 256
  sketch_dim: 128
  moment_order: 1
  num_tail_scores: 8
  tail_temperature: 0.25
  num_local_routes: 12
  local_route_dim: 32
  local_route_temperature: 0.25
  potential_type: adaptive_multiscale
  multiscale_gate_initial_bias: -0.5
  multiscale_local_initial_scale: 0.5
  prototype_regularization_weight: 0.01

  moment_token_weight: 0.1
  moment_token_count: 4
  moment_token_dim: 64
  moment_token_readout_dim: 128
  moment_token_temperature: 1.0
  moment_token_dropout: 0.0
```

The exact experiment commands and results are recorded in:

- `reports/bracs_sota_recovery_report.md`
- `reports/bracs_deep_optimization_log.tsv`
- `reports/general_mil_improvement_plan.md`
- `reports/bracs3_goal_completion_audit.md`

## 13. Experimental status of v1

The accepted v1 result is:

- `UNI + MIR-MIL moment-token w01`
- BRACS3 official test macro-AUC: `0.842568 ± 0.009488`
- PANDA seed-2024 sanity macro-AUC: `0.958328`

The internal BRACS3 target is:

- `UNI + AC_MIL = 0.852852 ± 0.009653`

Thus the v1 line narrows the gap but does not reach the BRACS3 SOTA target.

This is important for paper writing:

- MIR-MIL v1 can be presented as a method with a principled measure-state formulation and a validated moment-token extension.
- It should not be claimed as BRACS3 SOTA.
- The honest claim is that moment-token improves the MIR-MIL family under fixed features and fair validation-first selection, while the remaining BRACS3 gap suggests the need for a larger architecture redesign.

## 14. Ablation boundary

MIR-MIL v1 includes several implemented modules that were explored but are not part of the accepted main method:

- class-aware evidence head;
- fixed multi-token readout;
- gated multi-token readout;
- low-rank class-token readout;
- latent evidence transformer readout;
- ordinal calibration residual head;
- cosine state residual head;
- class-conditioned moment-token readout;
- tail-aware token readout;
- logit-margin auxiliary loss;
- subset consistency loss.

These are useful ablation or future-work components, but the current paper-level v1 main method should be described as:

\[
\text{MIR-MIL v1}
=
\text{measure-state potential}
+
\text{local routed measures}
+
\text{moment-token residual readout}.
\]

Rejected modules should not be silently merged into the method definition.

## 15. Method limitations

MIR-MIL v1 has several known limitations:

1. The compact measure state is strong for distributional tasks but may underrepresent highly localized class-boundary evidence.
2. Moment-token statistics help, but do not fully solve BRACS3 validation/test transfer.
3. The closed-form influence response is strongest for the explicit measure-potential path; residual attention readouts complicate complete attribution.
4. BRACS3 remains seed-sensitive under the official split.
5. Some modules can improve BRACS validation while degrading PANDA, indicating possible validation over-selection.

These limitations motivate a future v2 architecture rather than further small residual-head stacking.

## 16. Minimal pseudocode

```text
Input: bag B = {x_i}_{i=1}^N

for each patch x_i:
    h_i = Encoder(x_i)
    phi_i = ResponseBasis(h_i)
    a_i = TailScorer(h_i)
    g_i, u_i = LocalRouteBasis(h_i), LocalRouteScorer(h_i)

s_comp = mean_i(phi_i)
s_tail[m] = tau_tail * log mean_i exp(a_i[m] / tau_tail)
for each route r:
    alpha_ir = softmax_i(u_ir / tau_local)
    m_r = sum_i alpha_ir * g_i
s_local = concat_r(m_r)
s = concat(s_comp, s_tail, s_local)

z_base = AdaptiveMultiscalePotential(s)

for each moment token t:
    beta_it = softmax_i(key(h_i)^T token_t / sqrt(d_t) / tau_t)
    mean_t = sum_i beta_it * value(h_i)
    second_t = sum_i beta_it * value(h_i)^2
    var_t = clamp(second_t - mean_t^2, min=0)
z_moment = Linear(LayerNorm(concat_t(mean_t, var_t)))

z = z_base + lambda_moment * z_moment
return z
```

## 17. Recommended citation wording

For internal manuscript drafting, the method can be summarized as:

> MIR-MIL represents a WSI bag as a differentiable empirical measure and maps it to a compact neural measure state consisting of learned composition statistics, smooth tail statistics, and local routed measure summaries. A neural multiscale potential predicts slide-level logits from this state. To preserve evidence that may be diluted by the compact state, MIR-MIL v1 adds a dataset-agnostic moment-token residual readout that retrieves multiple evidence modes from encoded patches and summarizes each mode by both weighted mean and variance. The resulting model remains permutation-invariant, works for arbitrary class counts, uses only frozen pre-extracted patch features, and supports closed-form measure influence response analysis for the base measure-potential path.

