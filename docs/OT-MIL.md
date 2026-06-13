> **归档说明（2026-06-13）**：本文是 OT-MIL 的原始理论设想，不代表最终被实验
> 支持的结论。项目最终判断见 `../PROJECT_CLOSEOUT.md`，完整实验记录见
> `EXPERIMENT_REPORT_2026-06-10.md`。本文保留用于追溯设计动机。

> WSI-MIL 的真正目标不是把所有 patch 聚合，而是在经验测度中学习一个**最小、充分、必要、对域伪相关不敏感**的诊断子测度。

也就是说，从：

[
\text{OT as aggregation}
]

升级为：

[
\text{OT as diagnostic submeasure selection}
]

这是和已有 OT-MIL 拉开距离的关键。

---

# 1. 问题重定义：从 bag 到经验测度

传统 MIL 写成：

[
X={x_1,\dots,x_N}, \quad y \in {1,\dots,C}
]

patch encoder 得到：

[
h_i = E(x_i) \in \mathbb{R}^d
]

一般 MIL 学：

[
f({h_i}_{i=1}^N) \to y
]

我们改成测度视角：

[
\mu_X = \frac{1}{N}\sum_{i=1}^{N}\delta_{h_i}
]

其中 (\mu_X) 是 WSI 在特征空间 (\mathcal{H}) 上的经验测度。分类器是测度泛函：

[
F:\mathcal{P}(\mathcal{H}) \to \Delta^{C-1}
]

[
\hat{y}=F(\mu_X)
]

这个设定本身并不是全新，因为 PANTHER 已经将 WSI patch 分布建模为由 morphological prototypes 组成的 mixture distribution，并构造 compact slide representation。([arXiv](https://arxiv.org/abs/2405.11643?utm_source=chatgpt.com "Morphological Prototyping for Unsupervised Slide Representation Learning in Computational Pathology")) 所以我们的新意不能停留在“WSI 是分布”，而要继续往下走。

---

# 2. 核心假设：不是所有测度质量都对诊断有用

一张 WSI 的 patch 测度可以拆成：

[
\mu_X = \mu_c + \mu_s + \mu_n
]

其中：

- (\mu_c)：causal diagnostic submeasure，真正与标签因果相关；

- (\mu_s)：spurious submeasure，例如染色、scanner、center、组织处理伪影；

- (\mu_n)：irrelevant/noise submeasure，例如背景、无关正常组织、低质量 patch。


真实目标不是学习：

[
F(\mu_X)
]

而是学习：

[
F(\mu_c)
]

但是训练时没有 patch-level causal label，也不知道 (\mu_c) 在哪里。于是我们把目标定义为：

# [
\mu_\gamma^*

\arg\min_{\mu \preceq \mu_X}
|\mu|_{\mathrm{TV}}
]

subject to：

[
P(Y|\mu) \approx P(Y|\mu_X)
]

并且：

[
\mu \perp E_{\text{domain}} \mid Y
]

这里 (\mu \preceq \mu_X) 表示 (\mu) 是从原 WSI 测度中选出来的子测度，(|\mu|_{\mathrm{TV}}) 是选中质量。这个定义比普通 attention heatmap 更强，因为 attention 只是“哪里权重大”，而这里要求选出的证据 **最小、充分、必要、跨域稳定**。

---

# 3. 方法总览

CMS-OTMIL 可以分成 6 个模块：

```text
WSI patches
  ↓
frozen pathology encoder
  ↓
patch empirical measure μX
  ↓
causal / style disentangled projection
  ↓
sparse unbalanced OT to pathology prototypes
  ↓
diagnostic submeasure μγ
  ↓
sufficiency + necessity + invariance training
  ↓
slide classifier
```

下面逐步展开。

---

# 4. 模块一：patch feature 与测度构造

给定 WSI：

[
X={x_i}_{i=1}^N
]

用冻结 encoder：

[
h_i = E(x_i)
]

实际建议先不要 end-to-end 微调 encoder。先用强 pathology foundation model / SSL encoder 离线提特征，因为这样显存可控，也能避免方法复杂度和 encoder 训练耦合。很多近期 WSI 方法，包括 ROAM，都在 frozen foundation-model patch embeddings 上做聚合评估，这也是当前 WSI-MIL 实验的常见设置。([arXiv](https://arxiv.org/abs/2604.07298?utm_source=chatgpt.com "Region-Graph Optimal Transport Routing for Mixture-of-Experts Whole-Slide Image Classification"))

构造经验测度：

[
\mu_X=\sum_i a_i\delta_{h_i}
]

最简单：

[
a_i=\frac{1}{N}
]

如果 tissue mask 有置信度，也可以：

[
a_i = \frac{q_i}{\sum_j q_j}
]

其中 (q_i) 是 tissue quality / foreground score。

---

# 5. 模块二：病理原型测度

定义 (K) 个可学习 pathology prototypes：

[
P={p_1,\dots,p_K}, \quad p_k\in\mathbb{R}^{d'}
]

对应 prototype measure：

[
\nu_P=\sum_{k=1}^{K}b_k\delta_{p_k}
]

其中：

[
b_k \ge 0,\quad \sum_k b_k=1
]

这里 (b_k) 有三种设计：

## 5.1 Uniform prototypes

[
b_k=\frac{1}{K}
]

优点：简单。
缺点：容易强行均衡病理模式，不符合 WSI 长尾形态分布。

## 5.2 Learnable global prototype prior

[
b=\mathrm{softmax}(\beta)
]

优点：可学习不同 prototype 的全局重要性。
缺点：可能被训练集形态分布偏差污染。

## 5.3 Class-conditional prototype prior

[
\nu_y=\sum_k b_{y,k}\delta_{p_{y,k}}
]

每个类别有一组原型。优点是解释性强，例如 LUAD、LUSC、normal 分别有自己的 morphology prototypes。缺点是参数多，小数据集容易过拟合。

**推荐初版：共享 prototypes + class-specific classifier。**

即：

[
P={p_k}_{k=1}^K
]

共享原型，分类头自己学习类别组合。

这样与 TPMIL/PANTHER 类工作相比，避免“类别原型设计”过重，也更容易把贡献聚焦到 OT 子测度选择。

---

# 6. 模块三：从普通 OT 到 sparse unbalanced OT

普通 entropic OT：

# [
\gamma^*

\arg\min_{\gamma\ge 0}
\langle \gamma, C\rangle
+
\varepsilon
\sum_{i,k}\gamma_{ik}(\log\gamma_{ik}-1)
]

subject to：

[
\gamma\mathbf{1}_K = a
]

[
\gamma^\top\mathbf{1}_N = b
]

其中：

[
C_{ik}=d(h_i,p_k)
]

这个普通版本不够，因为它强制所有 patch mass 都要运输：

[
\gamma\mathbf{1}=a
]

但 WSI-MIL 里大量 patch 是无关的。我们需要 **unbalanced OT**，允许只运输部分质量。

定义：

# [
\gamma^*

\arg\min_{\gamma\ge 0}
\langle \gamma, C\rangle
+
\varepsilon \mathrm{KL}(\gamma | ab^\top)
+
\tau_x D_{\mathrm{KL}}(\gamma\mathbf{1}|a)
+
\tau_p D_{\mathrm{KL}}(\gamma^\top\mathbf{1}|b)
+
\lambda \Omega(\gamma)
]

这里：

- (\varepsilon)：entropic regularization，保证可微、稳定；

- (\tau_x)：控制是否必须使用所有 patch mass；

- (\tau_p)：控制是否必须覆盖所有 prototypes；

- (\Omega(\gamma))：稀疏/结构正则。


OTSurv 已经把 WSI 生存预测中的 long-tail morphology 和 tile uncertainty 转成 unbalanced OT，并用 matrix scaling 求解，这说明 unbalanced OT 在 WSI-MIL 实验上是可行的。([arXiv](https://arxiv.org/abs/2506.20741?utm_source=chatgpt.com "OTSurv: A Novel Multiple Instance Learning Framework for Survival Prediction with Heterogeneity-aware Optimal Transport")) 但我们的目标不同：不是 survival heterogeneity，而是 minimal sufficient diagnostic submeasure。

---

# 7. 模块四：最小诊断子测度

令：

[
m_i = \sum_{k=1}^{K}\gamma_{ik}
]

(m_i) 表示第 (i) 个 patch 被选入诊断证据的质量。

定义 OT 诱导的子测度：

[
\mu_\gamma = \sum_{i=1}^{N}m_i\delta_{h_i}
]

如果 (\sum_i m_i) 小，说明模型只选少量 patch。我们希望：

[
\mu_\gamma \ll \mu_X
]

并且：

# [
|\mu_\gamma|_{\mathrm{TV}}

\sum_i m_i
]

尽量小。

### 稀疏正则

最简单：

[
\Omega_{\mathrm{mass}}(\gamma)=\sum_{i,k}\gamma_{ik}
]

但如果只用 (\ell_1)，模型可能把质量均匀变小，导致表达能力下降。更好的是 group sparsity：

# [
\Omega_{\mathrm{group}}(\gamma)

\sum_{i=1}^{N}
\left(
\sum_{k=1}^{K}\gamma_{ik}^2
\right)^{1/2}
]

即：

[
|\gamma|_{2,1}
]

它鼓励少数 patch 被选中，但被选中的 patch 可以分配给多个 prototype。

还可以加 entropy control：

# [
\Omega_{\mathrm{sharp}}(\gamma)

\sum_i
H
\left(
\frac{\gamma_{i,:}}{\sum_k\gamma_{ik}}
\right)
]

如果想让每个 patch 明确对应少数 prototype，就最小化该 entropy；如果想保留不确定性，就不加或反向加。

---

# 8. 模块五：充分性与必要性目标

这是方法的关键区分点。

普通 OT-MIL 聚合完就分类：

[
\hat{y}=f(T_\gamma(\mu_X))
]

这还不够。我们要让 (\mu_\gamma) 成为 **minimal sufficient evidence**。

## 8.1 Sufficiency loss

选出的子测度应该足以预测：

# [
\mathcal{L}_{suf}

\mathrm{CE}
(
f(\mu_\gamma), y
)
]

也可以写成 consistency：

[
D_{\mathrm{KL}}
\left(
f(\mu_X)
|
f(\mu_\gamma)
\right)
]

但如果 (f(\mu_X)) 本身不可靠，直接用 label CE 更干净。

## 8.2 Necessity loss

移除选中证据后，模型不应该还能同样自信地预测原标签。

定义 complement submeasure：

# [
\mu_{\bar{\gamma}}

\sum_i
(a_i - \eta m_i)_+
\delta_{h_i}
]

其中 (\eta) 用于归一化，保证 (m_i\le a_i)。

必要性 margin：

# [
\mathcal{L}_{nec}

## \max
\left(
0,
\Delta

s_y(\mu_\gamma)
+
s_y(\mu_{\bar{\gamma}})
\right)
]

其中 (s_y(\cdot)) 是类别 (y) 的 logit。

直觉：

[
s_y(\mu_\gamma)
\ge
s_y(\mu_{\bar{\gamma}})+\Delta
]

也就是说，选中的证据应该比剩余证据更能支持标签。

## 8.3 Minimality loss

# [
\mathcal{L}_{min}

# |\mu_\gamma|_{\mathrm{TV}}

\sum_i m_i
]

总的证据学习目标：

# [
\mathcal{L}_{evi}

\mathcal{L}_{suf}
+
\alpha \mathcal{L}_{nec}
+
\beta \mathcal{L}_{min}
+
\lambda |\gamma|_{2,1}
]

这一步是和 PANTHER / OTSurv / ROAM 拉开距离的核心。已有工作多强调 prototype representation、heterogeneity-aware transport、capacity-constrained routing；这里强调的是 **sufficiency + necessity + minimality**。([arXiv](https://arxiv.org/abs/2405.11643?utm_source=chatgpt.com "Morphological Prototyping for Unsupervised Slide Representation Learning in Computational Pathology"))

---

# 9. 模块六：因果/域不变 transport

WSI-MIL 最危险的是学到伪相关。已有工作已经指出 attention-based WSI-MIL 可能错误关注染色条件、无关组织形态等模式，导致 patch-level prediction 和解释不可靠。([arXiv](https://arxiv.org/abs/2408.09449?utm_source=chatgpt.com "Attention Is Not What You Need: Revisiting Multi-Instance Learning for Whole Slide Image Classification")) 也有 2025 年的 Causal Attention MIL 工作专门针对 dataset bias 问题。([Proceedings of Machine Learning Research](https://proceedings.mlr.press/v260/wu25b.html?utm_source=chatgpt.com "Causal ATTention Multiple Instance Learning for Whole Slide ..."))

所以我们的 OT cost 不能只是：

[
C_{ik}=|h_i-p_k|^2
]

否则会同时匹配病理形态和染色/center/style。

---

## 9.1 表示分解

令：

[
z_i = g_\theta(h_i)
]

进一步分成：

[
z_i^c = g_c(h_i)
]

[
z_i^s = g_s(h_i)
]

其中：

- (z_i^c)：causal morphology representation；

- (z_i^s)：style/domain representation。


OT cost 只使用 (z_i^c)：

# [
C_{ik}^{c}

|z_i^c-p_k|^2
]

同时加入 domain adversarial，使 (z^c) 不能预测 domain：

## [
\min_{g_c,f}
\max_d
\mathcal{L}_{cls}

\lambda_d \mathcal{L}_{domain}(d(z^c), e)
]

其中 (e) 是 center/site/scanner/domain label。

如果没有 domain label，可以用 stain statistics / slide source / batch ID 作为 proxy；如果完全没有，就退化成增强一致性，见下面。

---

## 9.2 Transport plan invariance

比 feature invariance 更直接的是让 transport plan 在风格增强下稳定。

对同一 WSI 做两种 stain augmentation：

[
X^{(1)}, X^{(2)}
]

得到：

[
\gamma^{(1)}, \gamma^{(2)}
]

加入：

# [
\mathcal{L}_{plan}

|\gamma^{(1)}-\gamma^{(2)}|_1
]

但 patch 顺序和采样可能不同，直接比较 (\gamma) 不稳定。更稳妥的是比较 prototype mass：

# [
q_k^{(v)}

\sum_i \gamma_{ik}^{(v)}
]

# [
\mathcal{L}_{proto}

|q^{(1)}-q^{(2)}|_2^2
]

或者比较 transport-induced representation：

# [
\mathcal{L}_{repr}

|z_\gamma^{(1)}-z_\gamma^{(2)}|_2^2
]

这对应主张：

> 同一病理证据在染色扰动下，transport allocation 应稳定。

---

## 9.3 因果 cost 的更强版本

可以定义：

# [
C_{ik}

|z_i^c-p_k|^2
+
\rho \cdot r_i
]

其中 (r_i) 是 patch 的 spuriousness score。怎么得到 (r_i)？

### 方案 A：domain classifier confidence

如果 domain classifier 很容易从某 patch 预测 center：

[
r_i = \max_e d_e(h_i)
]

表示该 patch 含强域信息，应减少 transport。

### 方案 B：augmentation sensitivity

对 patch 做 stain perturbation：

[
r_i =
|g(h_i)-g(\tilde{h}_i)|
]

越敏感越可能是风格特征。

### 方案 C：label-domain correlation penalty

如果 patch score 同时高度预测 label 和 domain，则可能是 spurious。可以用 HSIC / mutual information penalty：

# [
\mathcal{L}_{HSIC}

\mathrm{HSIC}(z^c, e)
]

最终 cost：

# [
C_{ik}^{causal}

|z_i^c-p_k|^2
+
\rho r_i
]

这比普通 OT cost 多了一个关键约束：**运输低伪相关、高诊断性的 patch mass。**

---

# 10. Slide representation 怎么构造？

有三种选择。

## 10.1 Barycentric representation

# [
z_k

\frac{
\sum_i \gamma_{ik} z_i^c
}{
\sum_i \gamma_{ik}+\epsilon
}
]

最终：

[
z_X=[z_1,\dots,z_K,q_1,\dots,q_K]
]

其中：

[
q_k=\sum_i\gamma_{ik}
]

优点：简单，可解释。
缺点：只保留一阶均值。

## 10.2 Residual-to-prototype representation

# [
r_k

\frac{
\sum_i \gamma_{ik}(z_i^c-p_k)
}{
\sum_i\gamma_{ik}+\epsilon
}
]

[
z_X=[q_k,p_k,r_k]_{k=1}^K
]

直觉是：不仅知道分到了哪个 prototype，还知道该 slide 相对 prototype 的偏移。

## 10.3 Second-order representation

# [
\Sigma_k

\frac{
\sum_i \gamma_{ik}(z_i^c-z_k)(z_i^c-z_k)^\top
}{
\sum_i\gamma_{ik}+\epsilon
}
]

完整表示：

[
z_X=[q_k,z_k,\mathrm{diag}(\Sigma_k)]_{k=1}^{K}
]

注意：PANTHER 已经用 mixture parameters 表示 slide，包括 prototype/morphological mixture 思路。([arXiv](https://arxiv.org/abs/2405.11643?utm_source=chatgpt.com "Morphological Prototyping for Unsupervised Slide Representation Learning in Computational Pathology")) 所以如果做 second-order，必须强调这是 **OT-induced diagnostic submeasure 的二阶统计**，不是 unsupervised GMM summary。初版建议先用 10.1 或 10.2，避免和 PANTHER 太像。

---

# 11. 最终训练目标

整体模型：

[
\gamma^* = \mathrm{UOT}_{\varepsilon,\tau}
(\mu_X,\nu_P; C^{causal})
]

[
\mu_\gamma = \sum_i m_i\delta_{z_i^c}
]

[
\hat{y}=f(T(\gamma,z^c,P))
]

总损失：

# [
\mathcal{L}

\mathcal{L}_{cls}
+
\alpha \mathcal{L}_{suf}
+
\beta \mathcal{L}_{nec}
+
\lambda \mathcal{L}_{min}
+
\eta \mathcal{L}_{inv}
+
\zeta \mathcal{L}_{domain}
]

其中：

# [
\mathcal{L}_{cls}

\mathrm{CE}(f(\mu_\gamma),y)
]

其实 (\mathcal{L}_{cls}) 和 (\mathcal{L}_{suf}) 可以合并。

更干净的版本：

# [
\mathcal{L}

## \mathrm{CE}(f(\mu_\gamma),y)
+
\alpha
\max
(0,\Delta-s_y(\mu_\gamma)+s_y(\mu_{\bar{\gamma}}))
+
\beta |\mu_\gamma|_{TV}
+
\eta \mathcal{L}_{inv}

\zeta \mathcal{L}_{adv}
]

这里 (-\zeta \mathcal{L}_{adv}) 通过 gradient reversal 实现。

---

# 12. 算法伪代码

```text
Input:
  WSI patches {x_i}, slide label y, optional domain label e

1. Extract frozen patch features:
   h_i = E(x_i)

2. Project to causal morphology space:
   z_i = g_c(h_i)

3. Compute OT cost:
   C_ik = ||z_i - p_k||^2 + rho * spurious_score_i

4. Solve sparse unbalanced entropic OT:
   gamma = UOT(C, a, b; epsilon, tau_x, tau_p)

5. Compute selected mass:
   m_i = sum_k gamma_ik

6. Build diagnostic submeasure:
   mu_gamma = sum_i m_i delta_{z_i}

7. Build complement submeasure:
   mu_bar = sum_i (a_i - normalized(m_i))_+ delta_{z_i}

8. Compute slide representation:
   z_slide = concat_k [
       sum_i gamma_ik z_i / (sum_i gamma_ik + eps),
       sum_i gamma_ik
   ]

9. Predict:
   y_hat = classifier(z_slide)

10. Optimize:
   CE(y_hat, y)
   + necessity margin
   + mass sparsity
   + domain invariance / augmentation consistency
```

---

# 13. 理论可行性：能证明什么？

ICLR 审稿人会问：你的理论不是装饰吗？所以理论部分要可证明、可服务方法设计。

我建议证明四类命题。

---

## 13.1 命题一：OT 子测度算子是 permutation-invariant

定义：

# [
T(\mu_X)

\left[
\sum_i \gamma_{i 1} z_i,\dots,\sum_i \gamma_{iK} z_i
\right]
]

因为 (\mu_X) 是经验测度，OT 问题只依赖点集和质量，不依赖 patch 顺序。如果对 patch 做 permutation (\pi)，cost matrix 行也同样 permutation，最优 (\gamma) 行相应 permutation，最终：

[
T(\pi X)=T(X)
]

这个证明简单但必要，保证它是合法 MIL aggregator。

---

## 13.2 命题二：entropic OT 诱导稳定映射

想证明：

[
|T(\mu)-T(\nu)|
\le
L_{\varepsilon,C,\psi}
W_1(\mu,\nu)
]

直观上，entropic OT 的解对 cost 和 marginal perturbation 是平滑的；(\varepsilon>0) 越大，解越平滑，Lipschitz 常数越小但 transport 越模糊。

可以把定理写成较保守的版本：

假设：

1. feature space compact：


[
|z|\le R
]

2. cost (c(z,p)) 对 (z) 是 (L_c)-Lipschitz；

3. representation map (\psi(z,p)) 是 (L_\psi)-Lipschitz；

4. entropic OT regularization (\varepsilon>0)。


则 Sinkhorn-induced representation (T_\varepsilon(\mu)) 对输入测度扰动连续：

[
|T_\varepsilon(\mu)-T_\varepsilon(\nu)|
\le
O\left(\frac{L_c L_\psi R}{\varepsilon}\right)
W_1(\mu,\nu)
]

不一定要把常数推得极紧，但要说明：

- (\varepsilon) 提供稳定性；

- (\varepsilon\to 0) 时趋近硬匹配，稳定性变差；

- 这解释实验中为何中等 (\varepsilon) 最稳。


---

## 13.3 命题三：Wasserstein domain shift bound

如果 loss (\ell) 是 (L_\ell)-Lipschitz，分类器 (f) 是 (L_f)-Lipschitz，OT representation (T) 是 (L_T)-Lipschitz，则：

[
|\ell(f(T(\mu)),y)-\ell(f(T(\nu)),y)|
\le
L_\ell L_f L_T W_1(\mu,\nu)
]

对 train center (P) 和 test center (Q)：

[
R_Q(f\circ T)
\le
R_P(f\circ T)
+
L_\ell L_f L_T W_1(P,Q)
]

这给出 WSI domain shift 的数学解释：如果 stain/scanner shift 在 patch-measure Wasserstein 距离上不大，并且 (T) 稳定，那么风险不会大幅上升。

但这还不是我们独有，因为 OT/DRO 文献中类似 bound 很常见。独特点要放在：

[
T = \text{sparse causal OT submeasure operator}
]

即我们证明的是“诊断子测度选择算子”的稳定性，而不是普通 OT distance。

---

## 13.4 命题四：minimal sufficient submeasure 的泛化优势

这是最有潜力但也最难的理论。

可以从信息瓶颈角度写：

# [
\mu_\gamma

S_\theta(\mu_X)
]

目标：

## [
\min I(\mu_\gamma;\mu_X)

\beta I(\mu_\gamma;Y)
]

实际优化中的：

[
|\mu_\gamma|_{TV}
]

可以视为对 (I(\mu_\gamma;\mu_X)) 的 proxy，(\mathcal{L}_{cls}) 促进 (I(\mu_\gamma;Y))，(\mathcal{L}_{nec}) 促进证据必要性。

可提出命题：

若 spurious submeasure (\mu_s) 与 label 的相关性在训练环境和测试环境之间变化，而 causal submeasure (\mu_c) 满足：

[
P_e(Y|\mu_c)=P_{e'}(Y|\mu_c)
]

则在满足 sufficiency 和 invariance 的候选子测度中，选择最小 TV-mass 的子测度可降低包含 (\mu_s) 的概率。

这个命题严格证明需要额外假设，例如：

- causal evidence 比 spurious evidence 更稀疏；

- spurious pattern 在环境间不稳定；

- invariance penalty 能排除环境相关特征。


可以写成 proposition 而非强 theorem。

---

# 14. 理论边界：哪些不能乱承诺？

必须诚实。以下几件事很难严格证明：

## 14.1 不能证明选出的 patch 一定是真正因果病灶

因为只有 slide-level label，没有 patch-level causal annotation。我们只能说：

> under invariance and sufficiency assumptions, selected submeasure is biased toward stable diagnostic evidence.

不能说：

> guaranteed causal lesion localization。

## 14.2 不能证明 OT 一定比 attention 表达力更强

OT 和 attention 都可以很强，特别是 attention 叠多层后。可以证明的是：

- OT 有双边边缘约束；

- OT 可显式控制 prototype capacity / selected mass；

- OT plan 对 measure alignment 有明确优化含义；

- attention softmax 没有这些约束。


不要声称“OT universally dominates attention”。

## 14.3 不能把 Wasserstein bound 说成真实临床泛化保证

bound 只是解释模型设计的稳定性动机。真实 external validation 仍然必须做。

---

# 15. 实验可行性：能不能跑？

可以跑，但必须避免一上来搞太复杂。

---

## 15.1 推荐第一阶段实验设置

### 数据集

优先选公开、常用、能做 patient-level split 的 WSI 分类任务：

1. **CAMELYON 16**
    二分类 / metastasis detection，适合看 heatmap 与 evidence localization。

2. **TCGA-NSCLC**
    LUAD vs LUSC，常见 WSI-MIL benchmark，也适合 external 到 CPTAC。ROAM 报告了 NSCLC generalization TCGA-CPTAC 任务，这说明这条评估线是审稿人熟悉的。([arXiv](https://arxiv.org/abs/2604.07298?utm_source=chatgpt.com "Region-Graph Optimal Transport Routing for Mixture-of-Experts Whole-Slide Image Classification"))

3. **TCGA-BRCA / RCC / CRC subtype**
    可作为补充。


### 特征

先冻结 encoder：

- ResNet 50 ImageNet baseline；

- CTransPath / UNI / CONCH / Virchow 等 pathology FM 中选一个或两个。


不要一开始多 encoder 全做，先证明方法思想。

### Baselines

必须包括：

```text
Mean pooling
Max pooling
ABMIL
CLAM
DSMIL
TransMIL
PANTHER
TPMIL / prototype MIL
RAM-MIL if doing OOD/retrieval comparison
OTSurv-style UOT adaptation if possible
ROAM if MoE routing comparison relevant
```

其中 PANTHER、OTSurv、ROAM 是 OT/prototype 相邻工作的关键参照。PANTHER 是 prototype/GMM slide representation，OTSurv 是 heterogeneity-aware unbalanced OT，ROAM 是 capacity-constrained OT routing。([arXiv](https://arxiv.org/abs/2405.11643?utm_source=chatgpt.com "Morphological Prototyping for Unsupervised Slide Representation Learning in Computational Pathology"))

---

## 15.2 第一版模型不要太复杂

建议初版只实现：

[
\text{UOT} + \text{minimality} + \text{sufficiency/necessity}
]

先不做完整 causal disentanglement。

即：

[
C_{ik}=|g(h_i)-p_k|^2
]


## 15.3 计算复杂度

假设：

- 每张 WSI 采样 (N=4096) patches；

- prototypes (K=16) 或 (32)；

- cost matrix 是 (N\times K)，不是 (N\times N)。


复杂度：

[
O(NK \cdot T)
]

其中 (T) 是 Sinkhorn iterations，通常 20–50。

如果 (N=4096,K=32,T=30)，一次 slide 大概是 400 万级别操作，完全可接受。

关键是不要做 slide-to-slide OT：

[
O(N^2)
]

那会很贵。我们的设计是 patch-to-prototype OT，所以可行。

---

## 15.4 采样策略

每个 epoch 不必用所有 patches。建议：

```text
训练：每张 slide 随机采样 2048–8192 patches
验证：固定采样或全量分块聚合
测试：多次采样 ensemble 或 top tissue patches
```

OT 的 minimal submeasure 学习可能对采样敏感，因此要报告：

- (N=1024,2048,4096,8192) 的敏感性；

- random seed variance；

- selected mass ratio。


---

# 16. 实验指标设计

## 16.1 分类性能

常规：

[
\text{AUC}, \text{ACC}, \text{F 1}, \text{Balanced ACC}
]

医学任务还要：

[
\text{Sensitivity}, \text{Specificity}, \text{PR-AUC}
]

## 16.2 OOD / external validation

这是这篇方法最需要证明的。

例如：

[
\text{Train: TCGA-NSCLC}
]

[
\text{Test: CPTAC-NSCLC}
]

或者 train center A，test center B。

主张是 causal minimal transport 应该更稳，所以必须看：

# [
\Delta \mathrm{AUC}

## \mathrm{AUC}_{internal}

\mathrm{AUC}_{external}
]

越小越好。

## 16.3 证据质量

如果有 CAMELYON 16 pixel-level tumor annotations，可以计算：

- selected patch 是否落在 tumor mask；

- patch-level AUC；

- localization IoU；

- pointing game accuracy；

- top-k hit rate。


这能支持“minimal sufficient evidence”不是胡选。

## 16.4 稳定性

对同一 WSI 做 stain augmentation：

[
X^{(1)}, X^{(2)}
]

计算：

[
|q^{(1)}-q^{(2)}|
]

或者：

[
\mathrm{Jaccard}(S^{(1)},S^{(2)})
]

其中 (S) 是 top selected patches。

对比 ABMIL attention 的稳定性。

## 16.5 稀疏性-性能曲线

这是最能展示方法价值的图。

横轴：

# [
\text{selected mass ratio}

\frac{\sum_i m_i}{\sum_i a_i}
]

纵轴：

[
\mathrm{AUC}
]

理想结果：

- 我们用 10–30% patch mass 达到接近全量性能；

- complement patches 性能显著下降；

- ABMIL top attention patches 不如我们的 selected submeasure 稳定。


---

# 17. 必做消融实验

至少要有这些：

|Variant|目的|
|---|---|
|Balanced OT|看 unbalanced 是否必要|
|UOT only|看 OT 本身效果|
|UOT + minimality|看稀疏证据是否有效|
|UOT + sufficiency|看充分性|
|UOT + necessity|看必要性|
|UOT + invariance|看 OOD 增益|
|no prototype learning|看 prototype 是否学到有用结构|
|random selected mass|证明不是稀疏本身有效|
|attention top-k|与 attention evidence 对比|
|different K|prototype 数敏感性|
|different (\varepsilon,\tau)|OT 超参数敏感性|

特别重要的是：

[
\text{UOT only}
\quad \text{vs}
\quad
\text{UOT + necessity + minimality}
]

如果只是 UOT 有效，创新会被认为和 OTSurv/PANTHER/ROAM 相近；如果 necessity/minimality 提升明显，才说明我们的方法主张成立。

---

# 18. 可能失败点与解决方案

## 18.1 失败点：模型把质量选得太少，性能崩

原因：

[
\beta |\mu_\gamma|_{TV}
]

过强。

解决：

使用 curriculum：

[
\beta_t = \beta_{\max}\cdot \frac{t}{T}
]

先学分类，再逐渐压缩证据。

---

## 18.2 失败点：模型把所有质量分给一个 prototype

这是 prototype collapse。

解决：

加入 prototype usage entropy：

[
q_k=\sum_i\gamma_{ik}
]

# [
\mathcal{L}_{usage}

-\sum_k \bar{q}_k\log \bar{q}_k
]

注意这个 loss 只在 batch/global 层面鼓励使用多个 prototype，不要强制每张 WSI 均匀使用所有 prototypes。OTSurv 已经指出 WSI morphology 可能是 long-tail distribution，因此过度均匀是不合理的。([arXiv](https://arxiv.org/abs/2506.20741?utm_source=chatgpt.com "OTSurv: A Novel Multiple Instance Learning Framework for Survival Prediction with Heterogeneity-aware Optimal Transport"))

---

## 18.3 失败点：necessity loss 训练不稳定

因为 complement submeasure 可能仍然包含另一个诊断区域，尤其癌症 WSI 里病灶很多。

解决：

把 necessity 从“必须完全失败”改成 margin-based soft objective：

[
s_y(\mu_\gamma)
\ge
s_y(\mu_{\bar{\gamma}})+\Delta
]

不要要求 complement 预测错误。

---

## 18.4 失败点：domain adversarial 降低分类性能

如果 domain 与 label 强相关，直接去 domain 信息可能伤害性能。

解决：

先用 augmentation consistency 替代 domain adversarial：

# [
\mathcal{L}_{inv}

|T(X)-T(\mathrm{Aug}(X))|^2
]

这更温和，也不依赖 domain label。

---

## 18.5 失败点：OT 比 attention 慢

因为 (N\times K) 而不是 (N^2)，正常不会太慢。若仍慢：

- 降低 (K) 到 8/16；

- Sinkhorn iteration 设为 10–20；

- 使用 log-domain Sinkhorn；

- 先 spatial binning 到 region tokens，这与 ROAM 的 region-token 思路相似，但我们应避免把贡献说成 region routing。([arXiv](https://arxiv.org/abs/2604.07298?utm_source=chatgpt.com "Region-Graph Optimal Transport Routing for Mixture-of-Experts Whole-Slide Image Classification"))


---


# 21. 最终建议

我建议你不要从“OT 聚合器”写起，而从这个问题写起：

[
\boxed{
\text{Can a weakly supervised MIL model identify the smallest submeasure that is sufficient and necessary for bag-level prediction?}
}
]

然后 OT 只是解决它的工具：

[
\boxed{
\text{sparse unbalanced OT provides a differentiable, measure-theoretic relaxation of submeasure selection}
}
]

这条路线理论上可以证明：

- permutation invariance；

- Sinkhorn/UOT 稳定性；

- Wasserstein perturbation 下风险界；

- sufficiency/necessity 的证据选择意义。


实验上可以验证：

- 分类性能；

- external generalization；

- stain robustness；

- selected mass ratio；

- localization quality；

- complement ablation；

- 与 PANTHER/OTSurv/ROAM/ABMIL/CLAM 的区别。


**最稳妥的论文核心不是 Causal OT-MIL，而是 Minimal Sufficient Submeasure Learning。**
Causal 和 domain invariance 可以作为增强模块与实验亮点，不建议一开始把“因果”作为唯一主张。
