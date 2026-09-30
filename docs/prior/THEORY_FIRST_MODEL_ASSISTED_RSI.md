# Theory-first specification: factual policy improvement with predictable model corrections

**Status:** mathematical research specification, not a validated robotics algorithm and not a certified novelty claim.  
**Literature checked:** 30 September 2026.  
**Main direction:** model-assisted, on-policy improvement of a frozen generative robot policy, using factual outcomes to retain an unbiased improvement direction and online convex optimization to regulate the variance contribution of imperfect model predictions.  
**Do not reopen:** the stopped Q-Planning-lite / restored-snapshot counterfactual sandbox.

## 1. Scientific decision and provenance

The latest repair-router v7 review supports stopping that sandbox as a bounded investment decision. It does not establish a unique root cause of the online learner's failure, does not prove that Q-learning is impossible, and does not validate a new algorithm. Restore/history sensitivity limits interpretation of the cloned-state diagnostics; ordinary factual TD training was a different data path.

This specification deliberately does not use the old MC labels, failed-model checkpoints, or an asserted contact-dynamics root cause as its justification. It proposes a different optimization contract.

The contract is:

> Use a learned model to reduce the noise in an improvement signal whose expectation is determined by actual deployment outcomes, rather than treating the learned model's predictions as ground truth for improvement.

This principle is not new by itself. Q-Prop, action-dependent control variates, TrajCV, and doubly robust policy gradients are essential predecessors. A possible paper must establish a nontrivial extension or a substantive robot-specific result beyond them. Adding the word "RSI" is not novelty.

A second principle is:

> Learn the amount of model assistance from past factual trajectories, so an inaccurate or evolving model does not force persistent increases in gradient variance.

The online-regret result below provides a precise, limited guarantee for this principle. It does **not** say the robot outperforms a separately trained baseline, that every update is beneficial, or that physical exploration is safe.

## 2. Closest prior work and the claim boundary

| Prior | What is already established | What cannot be claimed as new here |
|---|---|---|
| Q-Planning (2026) | Frozen BC proposals, learned Q-guided action selection, Q-only learning from deployment successes and failures | "Frozen BC plus a small trainable selector/value system" |
| DSRL (2025) | Black-box diffusion-policy improvement by learning in its latent noise space | "Improve an implicit pretrained policy without differentiating the base generator" |
| RISE (2026) | Robot self-improvement using imagined outcomes and progress values | "World-model-driven robot self-improvement" |
| Q-Prop (2017) | Off-policy critic as a control variate; conservative/aggressive control-variate use | "Use an inaccurate critic without simply trusting its gradient" or "adapt a model-trust coefficient" |
| LAX/RELAX (2018) | Learned low-variance unbiased gradient estimators | "Train a neural control variate" |
| TrajCV (CoRL 2019 / proceedings 2020) | Trajectory-wise control variates and variance analysis | "Account for temporal gradient covariance" |
| DR-PG (2020) | General doubly robust policy gradients and variance theory | The factual-residual correction identity alone |
| Active-IS PG (2024), AISAC (2026) | Optimize behavior distributions to reduce policy-gradient variance | "Collect actions according to gradient information" |
| Adaptive control variates (Kim & Henderson, 2007) | Adaptively learned control variates in finite-horizon simulation | Adaptive coefficient fitting in isolation |
| SureSim (2025) | Factual correction of imperfect simulator predictions for robot policy evaluation | "Correct model predictions with real outcomes" in evaluation |
| The Mirage of Action-Dependent Baselines (2018) | Action-dependent baselines need not beat strong state baselines; implementation details can create apparent gains | A presumption that a more detailed control variate must help |

**Candidate research gap, not established exclusivity:** repeated improvement of implicit action-chunk robot policies using changing, potentially misspecified model predictions, without counterfactual state restoration, with an explicit finite-sample/online-regret account of how model assistance affects the variance of the *actual return gradient* and the cost per useful update.

The novel delta would have to concern the joint deployment/learning protocol, a sharper useful guarantee, or an empirically necessary robot-specific extension. Merely implementing the mathematics below is not yet a top-conference contribution. If an existing estimator and its adaptive implementation already subsume the proposed method and guarantees, use that work as the baseline and do not rename it.

## 3. Exactly what is being optimized

### 3.1 Environment and outcome

An episode contains at most `T` macro decisions. Each macro action is the exact fixed-length prefix the controller commits to, except for a documented terminal stop. After termination, pad with dummy decisions whose score gradient is zero.

Let

\[
Y\in[0,1],\qquad J(\theta)=\mathbb E_{\pi_\theta}[Y].
\]

For the first specification, `Y` is a factual terminal success indicator. A bounded, prespecified normalized return also works. Define first success, truncation, timeouts, and any safety-termination penalty before collecting data. Do not substitute model progress for `Y` and keep claiming a theorem about task success.

The history `h_t` includes all observations/actions available to the deployed policy before decision `t`, the task/goal, and remaining time. The mathematical formulation may use the full history even if the implemented frozen feature encoder compresses it. Unbiasedness does not require that the compressed feature be a Markov state; poor features can still limit achievable performance.

### 3.2 Frozen implicit proposal policy

At history `h_t`, a frozen generator samples

\[
C_t=(u_{t,1},\ldots,u_{t,K})\sim\mu(\cdot\mid h_t).
\]

It can be a BC/diffusion/flow policy. It need not expose a continuous action density. Its sampling law must not explicitly depend on the trainable selector parameters `theta`. Any internal generator randomness is included in the trajectory law.

A small selector chooses

\[
p_\theta(j\mid x_t),\qquad x_t=(h_t,C_t).
\]

Execute **the selected candidate prefix**, not a weighted average with a different action identity. All training/evaluation uses this same stochastic selection rule.

If the K proposals are identically distributed samples from the BC policy, a uniform selector has the same marginal action-prefix distribution as that BC policy. Thus initialize the selector to uniform rather than deliberately starving the BC of demonstrations to manufacture headroom.

This algorithm cannot create behaviors outside proposal support. If the frozen generator produces no useful alternatives in the relevant situations, no selector-only algorithm can solve that limitation.

### 3.3 Minimal analyzable selector

A suitable theory implementation is

\[
p_\theta(j\mid x)=\operatorname{softmax}_j(\theta^\top\varphi(x,j)),
\qquad \|\varphi(x,j)\|\le B,
\]

with a frozen feature extractor. This makes the trainable selector small and bounds score derivatives. It is not a claim that a linear head is expressive enough for every robot task.

For a fixed realized history/candidate set define

\[
s_{t,j}=\nabla_\theta\log p_\theta(j\mid x_t),
\quad
v_{t,j}=\nabla_\theta p_\theta(j\mid x_t)=p_{t,j}s_{t,j}.
\]

Write `s_t = s_{t,J_t}` for the sampled index.

## 4. Assumptions and non-assumptions

### Required

A1. The outcome, finite macro horizon, initial-state distribution, and controller semantics are specified.

A2. The physical/environment transition law has no direct dependence on `theta` except through the chosen actions. For the convergence theorem, the population objective is stationary during the analyzed run. Policy and model versions may evolve; an adversarially changing physical environment is not covered by the stationary-objective theorem.

A3. All actor-gradient episodes are on-policy under the logged, fixed selector version for that episode. Do not reuse stale-policy samples for actor updates without an additional valid off-policy analysis.

A4. At each action, model predictions for all candidates and the ordinary state baseline are computed from pre-action information. Their versions are fixed before the episode, or changed only by a separately justified predictable rule. No fitting on that episode's outcome before computing its actor gradient.

A5. The model-assistance coefficient used in an episode is chosen before that episode. It can depend arbitrarily on previous episodes. Its update from the current outcome is used only for future episodes.

A6. Scores, outcomes, predictions and coefficient domain are bounded as specified below. The theorem is for SGD-style ascent; Adam, momentum, clipping, reuse for multiple actor epochs, or policy projection require a separate analysis.

A7. Conditional trajectory samples correspond to the intended operational initial/reset distribution. Unknown hidden state *within* an episode is allowed. Uncontrolled cross-episode simulator history that changes that operational law is not magically corrected.

### Not required for gradient correctness

- Correct world model or correct Q estimates.
- Full physical-state observation or exact Markov observation features.
- Repeating several actions from the same physical state.
- Mid-episode restoration/cloning.
- A density for the pretrained continuous generator.
- An expert's corrective action labels.
- Ground-truth root-cause/failure-class labels.

**No state cloning is not no resets.** Ordinary task initialization and operational safety remain necessary.

## 5. Why some real outcome coverage is unavoidable

Consider a one-decision task with actions `a0,a1`. All observed data use `a0`, whose success probability is 1/2 in two possible environments. In environment E+, action `a1` succeeds surely; in E-, it fails surely. The observed data distributions are identical, but the direction of improvement toward `a1` is opposite.

Therefore no algorithm can infer the correct preference in both environments using only those observations and an unconstrained possibly incorrect model. One must obtain action coverage, impose a justified structural assumption, or accept uncertainty.

This does **not** require executing every action from the exact same physical state. Randomized action selection across genuine episodes can identify a population gradient.

## 6. The unbiased factual baseline

Let `b_t=b(h_t,C_t)` be an action-independent baseline fixed before choosing `J_t`. It can be learned on prior data and clipped to [0,1]. Define

\[
G^0=\sum_{t=0}^{T-1}s_t(Y-b_t).
\]

### Proposition 1: factual policy-gradient identity

\[
\mathbb E[G^0]=\nabla_\theta J(\theta).
\]

**Proof.** Write the probability of a complete realized trajectory as the product of environment factors, proposal-generator factors, and the selector probabilities. Only the last factors have an explicit `theta` derivative. Thus the log-likelihood derivative is `sum_t s_t`. Differentiating under the expectation gives `E[Y sum_t s_t]`. For any pre-action information sigma-field F_t,

\[
\mathbb E[s_tb_t\mid\mathcal F_t]
=b_t\sum_jp_{t,j}\nabla\log p_{t,j}
=b_t\nabla\sum_jp_{t,j}=0.
\]

Subtracting the baselines preserves the expectation. This is the ordinary likelihood-ratio/policy-gradient identity, not a new theorem.

## 7. A model correction that cannot change the expected gradient

Let `m_tj` be a model-predicted bounded outcome/value for candidate `j`. It may be generated by a world-model rollout and a value decoder, or by a direct learned predictor. Treat it as a number, not as a trusted label.

Use the residual relative to the baseline already present in `G0`:

\[
z_{t,j}=m_{t,j}-b_t.
\]

Construct

\[
C_t=\sum_jv_{t,j}z_{t,j}-s_tz_{t,J_t},
\qquad C=\sum_tC_t.
\]

For an episode-level scalar `alpha` chosen beforehand,

\[
\boxed{G_\alpha=G^0+\alpha C.}
\]

`alpha=0` recovers the factual baseline. `alpha=1` gives

\[
G_1=\sum_t\left[\sum_jv_{t,j}m_{t,j}
+s_t(Y-m_{t,J_t})\right].
\]

The second expression is the familiar action-dependent model-plus-residual construction. Its ancestry is Q-Prop/action-dependent control variates/DR-PG, not a newly discovered correction identity.

### Proposition 2: model misspecification changes variance, not expectation

For every predictable finite model `m` and coefficient `alpha`,

\[
\mathbb E[G_\alpha]=\nabla J(\theta).
\]

**Proof.** Conditional on F_t, all candidates, predictions, probabilities and derivatives are fixed. Therefore

\[
\mathbb E[C_t\mid\mathcal F_t]
=\sum_jv_{t,j}z_{t,j}-\sum_jp_{t,j}s_{t,j}z_{t,j}=0.
\]

Taking total expectation and using Proposition 1 proves the claim. The model does not need to estimate Q correctly. It does need to be predictable: using the outcome to refit its prediction and then correcting that same sample is not covered.

A negative `alpha` is mathematically allowed. It is a signed control-variate coefficient, not a physical instruction to execute unsafe actions or to reverse rewards.

### What is not claimed

The theorem does not make the model accurate, identify the outcome of an unexecuted action, ensure low variance, or guarantee safe hardware behavior. It identifies the average gradient of the stated deployed stochastic policy.

## 8. Why model accuracy alone does not determine model utility

At fixed `theta,m`, let

\[
\sigma_0^2=\operatorname{tr}\operatorname{Var}(G^0),
\quad c=\mathbb E[(G^0)^\top C],
\quad v=\mathbb E[\|C\|^2].
\]

Because `E[C]=0`,

\[
\boxed{\sigma_\alpha^2=\sigma_0^2+2\alpha c+\alpha^2v.}
\]

If `v>0`, the unconstrained population optimum is

\[
\alpha^*=-c/v,
\qquad
\sigma_{\alpha^*}^2=\sigma_0^2-c^2/v.
\]

With `alpha in [-A,A]`, clip `-c/v` into that interval. Zero remains feasible, so the constrained population optimum cannot have greater variance than the baseline. If `v=0`, C is zero almost surely and set `alpha=0`.

**Consequences.**

- A model whose control variate has no covariance with the residual gradient cannot reduce its variance.
- An arbitrary coefficient such as one can increase variance.
- Prediction MSE/R2, confidence, or inverse-action consistency do not by themselves determine `c^2/v`.
- These covariance facts are control-variate theory, not a novelty claim.
- The true coefficients are unknown. Plugging in noisy estimates does not inherit population optimality automatically.

The comparisons must start from a strong state-baseline estimator, not an artificially noisy REINFORCE implementation. The Mirage paper is a mandatory negative prior here.

### Optional contextual basis, not a required extra module

For a fixed predictable basis `f_r(h_t,C_t)`, define

\[
Z_r=\sum_t f_r(h_t,C_t)C_t.
\]

Then `E[Z_r]=0`; a bounded vector coefficient gives `G=G0+Z alpha`. The convex theory below extends directly. Do not add a learned partition or many context features before the scalar version has a justified need.

## 9. Main proposed control rule: online second-moment regret

Rather than estimating model usefulness from arbitrary features or true counterfactual outcomes, use an observable convex loss on each factual trajectory:

\[
\ell_n(a)=\|G_n^0+aC_n\|^2.
\]

At a fixed policy, minimizing its expectation is exactly minimizing gradient variance, since all coefficients have the same expected gradient. This is not an assumption that a surrogate loss correlates with return.

Algorithm:

\[
\alpha_{n+1}=\Pi_{[-A,A]}
\left[\alpha_n-2\beta C_n^\top(G_n^0+\alpha_nC_n)\right].
\]

Use `alpha_n`, not `alpha_{n+1}`, in the actor update from episode n. Then update the predictor/baseline for future data.

This is projected online gradient descent on a convex quadratic. It imports the standard OCO mechanism, rather than inventing a new optimizer. The research target is the complete predictable robot/model-learning protocol and its useful efficiency regime; merely using OGD is not novelty.

### Proposition 3: pathwise second-moment regret

Let the coefficient domain have diameter D and suppose `|ell'_n(a)| <= F` throughout it. With `beta=D/(F sqrt(N))`, for every fixed comparator a in the domain,

\[
\sum_{n=1}^N\ell_n(\alpha_n)-\ell_n(a)\le DF\sqrt N=:R_N.
\]

**Proof.** Nonexpansiveness of projection gives

\[
|\alpha_{n+1}-a|^2\le
|\alpha_n-a|^2-2\beta\ell_n'(\alpha_n)(\alpha_n-a)
+\beta^2|\ell_n'(\alpha_n)|^2.
\]

Convexity yields `ell_n(alpha_n)-ell_n(a) <= ell'_n(alpha_n)(alpha_n-a)`. Rearrange, sum and telescope:

\[
\sum_n[\ell_n(\alpha_n)-\ell_n(a)]
\le D^2/(2\beta)+\beta NF^2/2.
\]

Substitute beta. No stationary model-prediction quality assumption is needed for this pathwise inequality.

### Corollary 3.1: a precise fallback guarantee

Use comparator `a=0`. Taking expectations conditional on the sequential information structure,

\[
\sum_n\mathbb E\operatorname{trVar}(G_{\alpha_n,n}\mid\mathcal H_{n-1})
\le
\sum_n\mathbb E\operatorname{trVar}(G^0_n\mid\mathcal H_{n-1})+R_N.
\]

`H_(n-1)` includes the current policy, prior training data and the predictable model/coefficient. The two gradients have the same conditional mean, so those squared means cancel.

**Interpretation:** excess variance relative to ignoring the auxiliary model is sublinear along the learner's encountered policy sequence.

**Not implied:** per-update variance dominance, outperforming an independently trained baseline's learning curve, global policy improvement, or physical safety. The baseline comparator here is evaluated on the *same sequence of policies/data-generating conditions*. It is not the counterfactual trajectory of another RL training run.

For an empirical best coefficient selected in hindsight, the pathwise quadratic-loss regret is still valid; do not interpret that data-dependent comparator as automatically an unbiased deployed estimator. The zero comparator avoids this issue.

### Explicit constants for the restricted selector

If `||s_tj|| <= S`, `Y,b,m in [0,1]`, then

\[
\|G^0\|\le TS=:M_0,
\qquad \|C\|\le2TS=:M_C.
\]

Thus a valid meta-gradient bound is

\[
F=2M_C(M_0+AM_C),\qquad D=2A.
\]

These are conservative worst-case constants. The guarantee can be numerically loose; do not equate existence of a bound with practical sample efficiency.

## 10. Connection to the actual return objective

Assume `J` is L-smooth and consider SGD-style ascent

\[
\theta_{n+1}=\theta_n+\eta G_n.
\]

Write `g_n=grad J(theta_n)`. From smoothness,

\[
\mathbb E[J(\theta_{n+1})-J(\theta_n)\mid\mathcal H_{n-1}]
\ge
\eta\|g_n\|^2-\frac{L\eta^2}{2}\mathbb E[\|G_n\|^2\mid\mathcal H_{n-1}].
\]

Since the gradient is conditionally unbiased,

\[
\boxed{
\mathbb E[\Delta J_n\mid\mathcal H_{n-1}]
\ge
\left(\eta-\frac{L\eta^2}{2}\right)\|g_n\|^2
-\frac{L\eta^2}{2}\sigma_n^2.
}
\]

For `eta <= 1/L`, a sufficient condition for the displayed lower bound to be positive is

\[
\|g_n\|^2>L\eta\sigma_n^2.
\]

A batch of n conditionally independent/martingale-difference samples under a frozen policy replaces variance by the appropriate average variance divided by n. Coefficient prediction must remain predictable for each trajectory.

### Proposition 4: a stationary-point bound with model-assistance penalty

Suppose the factual baseline has conditional variance at most `sigma0^2` throughout the run, and `eta <= 1/L`. Combining Proposition 3 with the smoothness inequality gives

\[
\boxed{
\frac1N\sum_n\mathbb E\|\nabla J(\theta_n)\|^2
\le
\frac{2(J_{max}-J(\theta_1))}{\eta N}
+L\eta\sigma_0^2
+\frac{L\eta R_N}{N}.
}
\]

**Proof.** Sum the smoothness inequalities, bound the expected sum of assisted squared-gradient norms by the baseline sum plus `R_N`, substitute `E||G0||^2=||g_n||^2+variance`, and use `eta-L eta^2/2 >= eta/2`. The left-hand telescoping objective is at most `Jmax-J(theta_1)`.

Since `R_N=O(sqrt(N))`, choosing `eta=O(N^-1/2)` makes the additional model-assistance penalty `O(N^-1)`, under the boundedness assumptions. This is a standard nonconvex stationarity-type guarantee with an OCO-derived penalty. It is not an optimal-policy theorem and not a proof of real-robot gains.

### The smoothness assumption can be made explicit in the linear-head model

For bounded frozen features, `||s_t|| <= 2B` and the Hessian of `log p` has norm at most `B^2`. Differentiating the trajectory likelihood twice yields a conservative global bound

\[
L\le B^2(4T^2+T).
\]

This comes from the squared norm of the summed scores and the sum of log-probability Hessians. Contact dynamics need not be differentiable with respect to physical actions for this likelihood-ratio argument; only the selector probabilities are differentiated.

This explicit bound may be far too pessimistic for useful step sizes. It does not transfer unchanged to an arbitrary jointly trained deep encoder, Adam, a deterministic argmax policy, or a changing physical objective.

## 11. Full implementation contract

```text
Inputs:
  frozen generator mu; differentiable selector p_theta
  frozen-for-the-episode scalar predictor m and baseline b
  coefficient alpha in [-A,A], initialized to 0
  declared outcome Y and exact committed-prefix semantics

For each factual episode n:
  snapshot theta_n, predictor version, baseline version, alpha_n
  for every macro decision:
    construct candidate set before selecting any candidate
    compute p, m_j, b from pre-action data
    sample J ~ p and execute exactly candidate J's committed prefix
    log all K probabilities, predictions, selected index and version IDs
  observe the actual bounded outcome Y
  compute G0 and C at theta_n; stop gradients through m, b and alpha
  actor gradient = G0 + alpha_n * C
  perform the declared actor SGD update
  alpha_(n+1) = clip(alpha_n - 2 beta C dot actor_gradient, -A, A)
  update m and b on factual experience, for future episodes only
```

A convenient actor pseudo-objective with detached numerical coefficients is

\[
\sum_t \log p_\theta(J_t|x_t)
\,[Y-b_t-\alpha(m_{t,J_t}-b_t)]
+\alpha\sum_{t,j}p_\theta(j|x_t)(m_{t,j}-b_t).
\]

Differentiate only through `p_theta`. For gradient descent implementations, negate this objective. Cache proposal sets rather than resampling them when computing the gradient.

A neural optimizer that differentiates through the model prediction, uses the just-fitted coefficient for the current episode, selects data only on success, drops failed/early-terminated episodes, or takes several actor epochs from stale probabilities is a different algorithm. It must not inherit these guarantees without further derivation.

A factual failure is not imitated as a successful action. It contributes through a return-centered likelihood-ratio gradient. The expectation statement uses both successes and failures under the actual collection policy.

## 12. Efficiency: when does the mechanism actually deserve to exist?

Unbiasedness and nonpositive population variance excess do not imply practical usefulness.

At a fixed policy/model, the available variance improvement beyond the chosen baseline is

\[
\Delta\sigma^2_{oracle}=c^2/v
\]

when the unconstrained optimum is feasible. If it is tiny, a model-control branch is not worth pursuing, even with a correct proof.

For wall-clock budget B and per-episode effective cost `kappa_method`, a simple fixed-policy comparison of gradient-mean error is proportional to

\[
\sigma^2_{method}\,\kappa_{method}/B.
\]

Thus lower variance must offset model inference, candidate generation, score-gradient calculation and model training costs. Compare both factual interaction count and wall time. OGD calibration itself does not require extra counterfactual episodes, but evaluating predictions for K candidates is not free.

Mandatory empirical baselines:

- factual on-policy PG with a strong state baseline;
- fixed-alpha action-dependent correction;
- Q-Prop/DR-PG/TrajCV-style alternatives adapted faithfully to the same policy interface;
- an independently validated practical pretrained-policy improvement baseline such as DSRL where compatible;
- model-only improvement, explicitly acknowledging its potential bias;
- offline population-oracle coefficients only as a diagnostic upper benchmark.

A strong state baseline is indispensable. The supplied exact example shows that much apparent room can disappear after it is added.

## 13. Exact calculations already executed for this specification

These are **mathematical implementation checks**, not robot training, not benchmark results, and not evidence of publication novelty.

### A. Partially observed three-decision finite process

`exact_checks.py` enumerates latent states, noisy observations, actions and Bernoulli outcomes. The true likelihood-ratio gradient matches finite differences. An auxiliary single-splice DR construction remains unbiased for several inaccurate model predictions, at about 1e-17 numerical error. Logging a wrong propensity breaks that equality.

`extended_checks.py` computes exact second moments. It verifies the optimal time-allocation identity but also shows that the simple one-splice/local-gradient construction can have **higher variance than an all-step on-policy estimator**. Therefore it is not the recommended primary algorithm, and is not smuggled in as an extra "novel" component.

### B. All-step model correction

`calibrated_cv_checks.py` verifies zero-mean controls, cross-time orthogonality, unbiased corrected gradients and the covariance/optimal-coefficient formulas. Relative to a fixed state baseline, some inaccurate model predictions increase variance with coefficient one. Population-optimal coefficients in that particular process recover less than 1% improvement. This is a warning against assuming a useful effect.

### C. Four-candidate illustration and online-regret check

`online_calibration_checks.py` uses a four-candidate Bernoulli-outcome process with its optimal scalar baseline.

For a deliberately wrong predictor, exact variance is approximately:

```text
state-baseline PG:             0.111148
unregulated coefficient one: 0.231368
population-calibrated CV:     0.100139
```

These numbers illustrate the covariance mechanism. The population coefficient is known from enumeration in this check, not learned for free on a robot.

On one synthetic 4,096-sample stream with changing model predictions, the projected OGD sequence satisfies its pathwise regret bound; conditional gradient expectation remains correct to numerical precision. The regret bound is loose. No claim of universal strict improvement follows.

Run all checks with NumPy:

```bash
OPENBLAS_NUM_THREADS=1 python exact_checks.py
OPENBLAS_NUM_THREADS=1 python extended_checks.py
OPENBLAS_NUM_THREADS=1 python calibrated_cv_checks.py
OPENBLAS_NUM_THREADS=1 python online_calibration_checks.py
```

## 14. Theory-derived hypotheses and ablations

| Mathematical statement | Controlled change | Predicted observation | What failure would mean |
|---|---|---|---|
| Zero-mean correction | Deliberately bias/permutate m while holding the factual policy fixed | Expected gradient unchanged; variance may change | Wrong action probabilities, correction term, model timing, or differentiation contract |
| Predictability is required | Fit alpha or m on the same current outcome and apply retroactively | The original unbiasedness theorem no longer applies; test can expose bias | A negative control, not an intended extension |
| Model-only bias | Remove factual residual correction | Gradient bias tracks action-dependent prediction error | Confirms why model predictions cannot simply be trusted |
| Covariance, not prediction MSE, controls utility | Compare models with matched MSE but different gradient covariance | Variance follows c,v, not necessarily MSE | Reassess claimed usefulness metric |
| Online quadratic regret | Alpha fixed at 0/1 versus predictable OGD, including changing model quality | Recorded second-moment regret obeys the bound; may show no strict benefit | Check OGD implementation/assumptions; no tuning until positive |
| Actual return link | Compare equal-budget update gradients with measured variance and independent factual returns | The smoothness/SNR prediction is locally compatible with update behavior | Could be finite-step nonlocality, parameterization, baseline strength or negligible headroom |
| No cloning dependence | Ordinary independently initialized episodes only | No same-snapshot outcome labels are needed | Any accidental dependence on old cloned labels violates the new design |
| Strong baseline control | Improve/optimize the state baseline | Some apparent model benefit can disappear | Stop that proposed contribution rather than weaken the baseline |
| Claimed WM value | Replace WM predictor by a cheaper return predictor | A WM-specific claim requires extra benefit to survive cost matching | Do not label an ordinary critic result a WAM contribution |

Experiments do not prove the algebra. They verify the implementation and test whether the assumptions, signal size, and cost regime make the algebra useful in the target domain.

## 15. Bounded next research sequence

### Step 1: theoretical originality check before robot coding

Write a side-by-side derivation against Q-Prop, TrajCV, DR-PG, adaptive control-variate literature and OCO. Explicitly ask whether the proposed online-regret/predictability statement is already a direct corollary of an existing method. If so, label it an inherited guarantee, not a new theorem.

A positive publication claim must identify a nontrivial remaining delta. Do not infer novelty merely from lack of a matching keyword search.

### Step 2: exact-model correctness and power

Extend the finite enumeration to at least three prespecified regimes: useful model, anti-correlated model, and model redundant with an optimal state baseline. Compare all-step estimators, not only a weak single-time baseline. Estimate the variance reduction that is mathematically possible before any long run.

Stop if the useful regime requires oracle information unavailable to the proposed robot system, or if cost-adjusted improvement disappears against the strongest standard estimator.

### Step 3: independent factual-rollout backbone

Only after Steps 1–2, choose a published policy/environment stack whose factual baseline has independently reproducible non-ceiling performance. Do not resume the stopped `qplite` clone pipeline. Keep the pretrained proposer and evaluator unchanged. No arbitrary reduction of demonstration count to manufacture headroom.

Start with a small selector and a cheap bounded predictor; use a released model rather than training a video WM. One RTX 5090 is a resource constraint, not evidence of an achievable run time. Profile actual inference/simulation first, and preallocate an explicit interaction and occupied-device budget from the measured throughput.

### Step 4: one mechanism comparison, not a search campaign

Compare predictable variance control to the strongest inherited correction method at matched data, encoder, action support, policy-update count and evaluation outcome. Prespecify a practically meaningful return or interaction-efficiency margin and use equivalence/non-inferiority tests when claiming sameness. Failure to reject zero is not equivalence.

Unit of generalization is an independent training seed/task instance, not the number of action pairs. Use fresh confirmatory episodes after design choices, and do not select the best seed or checkpoint post hoc.

### Step 5: scale only with a real delta

Required evidence for a paper-oriented continuation:

1. A distinct contribution beyond ordinary adaptive control variates is articulated.
2. The inherited factual improvement backbone works without the new controller.
3. Model assistance produces useful covariance beyond a strong state baseline.
4. Predictable online calibration improves finite-budget outcomes over simpler corrections.
5. The claimed effect survives independent tasks/seeds and compute/interaction matching.
6. A WM-specific version beats a cheaper predictor after its extra cost is counted.

Otherwise stop the candidate direction. A correct theorem with no useful regime is not a reason to fund a large robot campaign.

## 16. Claim ladder and limitations

**Established analytically in this note, conditional on assumptions:**

- factual gradient and corrected estimator have the same expectation;
- arbitrary predictable model error changes variance rather than creating mean bias;
- exact covariance criterion for when model assistance can help;
- projected OGD has a sublinear quadratic-loss regret against ignoring the model;
- that regret yields a bounded additional term in an SGD stationarity guarantee.

**Checked numerically on finite synthetic processes:**

- expectation/finite-difference identities;
- variance algebra and time-control covariance;
- one explicit OGD pathwise-regret inequality;
- counterexamples to naive model trust and to universal superiority of a single-splice estimator.

**Not established:**

- main-track novelty;
- superiority over the strongest published robotics/DR-PG baselines;
- sample efficiency with a large pretrained robot policy;
- physical safety, reliable resets or success verification;
- monotonic realized improvement;
- global optimality or overcoming a frozen proposal support ceiling;
- the cause of the stopped sandbox's failures.

The immediate recommendation is a mathematics/closest-prior review, not authorization for another full robotics implementation campaign.

## References (primary sources)

1. Q-Planning: Giridhar et al., *Beyond Imitation: Self-Improving Robot Policies via Off-Policy Q-Planning* (2026). https://arxiv.org/abs/2608.21204
2. DSRL: Wagenmaker et al., *Steering Your Diffusion Policy with Latent Space Reinforcement Learning* (2025). https://arxiv.org/abs/2506.15799
3. RISE: Yang et al., *RISE: Self-Improving Robot Policy with Compositional World Model* (2026). https://arxiv.org/abs/2602.11075
4. Q-Prop: Gu et al. (2017). https://arxiv.org/abs/1611.02247
5. LAX/RELAX: Grathwohl et al., *Backpropagation through the Void* (2018). https://arxiv.org/abs/1711.00123
6. TrajCV: Cheng, Yan, Boots, *Trajectory-wise Control Variates for Variance Reduction in Policy Gradient Methods*. https://proceedings.mlr.press/v100/cheng20a.html
7. DR-PG: Huang & Jiang, *From Importance Sampling to Doubly Robust Policy Gradient* (2020). https://proceedings.mlr.press/v119/huang20b.html
8. Papini et al., *Policy Gradient with Active Importance Sampling* (2024). https://rlj.cs.umass.edu/2024/papers/Paper90.html
9. Molaei et al., *Actor-Critic with Active Importance Sampling* (2026). https://arxiv.org/abs/2605.07094
10. Badithela et al., *Reliable and Scalable Robot Policy Evaluation with Imperfect Simulators* / SureSim (2025). https://arxiv.org/abs/2510.04354
11. Tucker et al., *The Mirage of Action-Dependent Baselines in Reinforcement Learning* (2018). https://proceedings.mlr.press/v80/tucker18a.html
12. Hazan, *Introduction to Online Convex Optimization*. https://arxiv.org/abs/1909.05207
13. Kim & Henderson, *Adaptive Control Variates for Finite-Horizon Simulation* (2007). https://pubsonline.informs.org/doi/10.1287/moor.1070.0251
14. Sutton et al., *Policy Gradient Methods for Reinforcement Learning with Function Approximation*. https://papers.nips.cc/paper/1713-policy-gradient-methods-for-reinforcement-learning-with-function-approximation

The primary results above motivate and bound the proposed research; they are not evidence that the exact proposed combination is novel or effective on robotics.
