### A Pluto.jl notebook ###
# v0.20.0

using Markdown
using InteractiveUtils

# ╔═╡ 11111111-0807-4111-8111-111111111111
md"""
# Lecture 08, Notebook 07: Analytic OLG DEQN, Exogenous Cloud in Lux

This smoke-size translation keeps the analytic 6-age OLG validation target and
uses fresh exogenous state clouds rather than a persistent simulated cloud.
"""

# ╔═╡ b4ba56b1-2bf4-40c8-3d7b-d645c4f1a76a
md"""
## Lecture 08, Notebook 07: Analytic 6-Agent OLG — DEQN (Exogenous Sampling)

**Course:** Deep Learning for Solving and Estimating Dynamic Models in Economics and Finance
**Script reference:** §5.2 (the 6-agent analytic Krueger–Kübler OLG); §5.3 (DEQN mapping); §5.4 (KKT)
**Notebook role:** core
**Author:** Simon Scheidegger

*Julia/Lux/Pluto preview of* `lectures/lecture_08_olg_models_deqns/code/lecture_08_07_OLG_Analytic_DEQN_exogenous.ipynb`.

> **Run mode.** The checked-in run uses `RUN_MODE = "smoke"` with fixed `SEED = 0` for a fast structural check. Set `RUN_MODE` to `"teaching"` or `"production"` in the config cell below to reproduce slide-scale accuracy (production switches to the 100/50 hidden-layer architecture of Appendix A.8).

> **Sampling mode.** This is the exogenous-cloud ablation: every training batch is an independent feasible state cloud drawn from broad boxes, with no feedback from the current policy. The persistent-simulation companion `lecture_08_08_OLG_Analytic_DEQN_persistent.ipynb` trains on simulated trajectories under the current policy.
"""

# ╔═╡ d54a2146-8553-06bd-5373-84c7004e7976
md"""
This notebook solves the six-age overlapping-generations model of Krueger and Kübler (2004): log utility, stochastic Cobb–Douglas production, four i.i.d. TFP/depreciation aggregate states, and a closed-form age-specific savings rate that the trained policy can be validated against.

The notebook mirrors the structure of the IRBC and benchmark-OLG notebooks: parameters, augmented state, neural-network policy transform, residual / loss construction, training-data generation, mini-batch training, out-of-sample residual tables, and validation against the closed-form savings rates.

In this Lux/Pluto preview the training cloud is **drawn exogenously** from broad feasible boxes: every batch is an independent feasible state cloud, with no feedback from the current policy. This is useful for checking the residual code without the additional feedback created by simulation.
"""

# ╔═╡ 22222222-0807-4222-8222-222222222222
begin
    import Pkg
    Pkg.activate(joinpath(@__DIR__, "..", "..", "..", "julia"))

    using DLEFJulia
    using Lux
    using NNlib
    using Optimisers
end

# ╔═╡ 33333333-0807-4333-8333-333333333333
begin
    RUN_MODE = "smoke"
    SEED = 0
    budgets = (
        smoke = (steps = 4, batch_size = 16),
        teaching = (steps = 80, batch_size = 128),
        production = (steps = 2_000, batch_size = 512),
    )
    hp = run_mode_budget(RUN_MODE; budgets)
    rng = rng_from_seed(SEED)
end

# ╔═╡ 67d0d5a5-34cd-4ef8-015d-fec7736fe380
md"""
## 1. Economic parameters

The Krueger–Kübler (2004) calibration: \$A = 6\$ overlapping agents, log utility, Cobb–Douglas production with capital share \$\alpha = 0.3\$, discount factor \$\beta = 0.7\$, only agent 1 supplies labor, and four i.i.d. aggregate states combining TFP \$\eta \in \{0.95, 1.05\}\$ and depreciation \$\delta \in \{0.5, 0.9\}\$ with equal probability (\$\pi_{ss'} = 0.25\$). Under these choices the model has a closed-form age-specific savings rate (see §8). Here `AnalyticOLGParams()` carries this calibration and `analytic_olg_closed_form_savings_rates(params)` returns the closed-form rates used for validation.
"""

# ╔═╡ 22f9055e-bca0-549f-e0c0-f2aa977101f5
md"""
### 2. State representation and augmented network inputs

The minimal state is the cohort capital vector together with the aggregate shock indicator (\$1 + A = 7\$ numbers for \$A = 6\$). To make learning easier we feed the network an **extended state** that pre-computes aggregates, per-agent income blocks, and the shock-transition probabilities:

\$\$
  \dim(\mathbf{s}_t) = \underbrace{1 + 4 + 2 + 5}_{12\text{ aggregate}} + \underbrace{4A}_{\text{per-agent}} + \underbrace{4}_{\boldsymbol{\pi}(z_t)} = 16 + 4A,
\$\$

which is \$40\$ for the analytic model (\$A = 6\$). In Julia the feature width is `analytic_olg_feature_dim(params)` and the augmentation is built by `analytic_olg_features`; it is pure feature engineering — the equilibrium is unchanged, only the network input is enriched.
"""

# ╔═╡ 5885caa8-c92b-1740-68f1-aa6b6677423d
md"""
### 3. Neural network and policy transformation

The policy network is an MLP from the 40-dimensional augmented state to the savings of cohorts \$1,\ldots,A-1\$ (cohort \$A\$ saves nothing),

\$\$
  \mathcal{N}_\theta(\mathbf{s}_t) \;\longrightarrow\; \bigl(\hat a_t^1, \hat a_t^2, \ldots, \hat a_t^{A-1}\bigr), \qquad \hat a_t^h = k_{t+1}^{h+1}.
\$\$

The output head is a **sigmoid savings-fraction transform**: the raw network output is squashed to a fraction \$\hat\beta_h \in (0, 1)\$ of current income, and savings are \$\hat a_t^h = \hat\beta_h \cdot \mathrm{inc}_t^h\$. This guarantees \$\hat a_t^h \ge 0\$ *and* current consumption \$c_t^h = (1 - \hat\beta_h)\,\mathrm{inc}_t^h \ge 0\$ by construction, and it matches the closed-form structure ("save a fixed fraction \$\beta_h\$ of income"). Aggregate next-period capital \$K_{t+1} = \sum_h \hat a_t^h\$ then satisfies capital-market clearing by construction, and no Lagrange multipliers appear (borrowing constraints are non-binding under this calibration).

In Lux the network is `make_mlp(analytic_olg_feature_dim(params), (32, 16), params.n_ages - 1; activation = NNlib.tanh)`; the sigmoid savings-fraction transform lives inside `analytic_olg_policy_from_raw` / `analytic_olg_residual`, and `setup_training` initializes Float64 parameters with an `Optimisers.Adam` optimizer. The model is called with the explicit `y, st = model(x, ps, st)` pattern on feature-by-batch arrays.
"""

# ╔═╡ 44444444-0807-4444-8444-444444444444
begin
    params = AnalyticOLGParams()
    rates = analytic_olg_closed_form_savings_rates(params)
    model = make_mlp(analytic_olg_feature_dim(params), (32, 16), params.n_ages - 1; activation = NNlib.tanh)
    state = setup_training(rng_from_seed(SEED; offset = 1), model, Optimisers.Adam(0.002); parameter_type = Float64)
end

# ╔═╡ 1f342403-9455-cd5f-1878-6f6376a79daf
md"""
## 4. Residuals and loss

For each cohort \$h = 1, \ldots, A-1\$ the relative Euler residual is

\$\$
  \mathrm{EulerErr}_h(\mathbf{s}_t) \;=\;
  \frac{\beta\, \mathbb{E}_t\!\left[\, u'(c_{t+1}^{h+1})\, \bigl(R_{t+1} + 1 - \delta_{t+1}\bigr)\,\right]}
       {u'(c_t^h)} \;-\; 1,
\$\$

with \$u(c) = \log c\$. Reporting is in *relative* form, so \$10^{-3}\$ corresponds to a 0.1% Euler-equation wedge. The total loss is

\$\$
  \ell_\theta \;=\; \frac{1}{N_s}\sum_{i=1}^{N_s} \sum_{h=1}^{A-1} \bigl(\mathrm{EulerErr}_h(\mathbf{s}_i)\bigr)^2,
\$\$

with the conditional expectation evaluated by direct summation over the four discrete shock states. No market-clearing term is needed — capital-market clearing holds by construction (§3). In Julia this is `analytic_olg_residual`, wrapped by the `analytic_loss(model, ps, st, batch)` closure so the trainer sees `(loss, state)`; parameter gradients come from `Zygote` inside `train_step!`.
"""

# ╔═╡ c538b2d8-c811-5bf8-600d-e9fa057820d1
md"""
### Training-data construction and the training loop

In this exogenous version, training states are drawn from broad feasible boxes via `sample_analytic_olg_states(rng, params, n)`; the clouds are deliberately not centered on a steady state, a lifecycle profile, or the closed-form policy.

The compact preview draws a **fresh independent batch on every step** and applies one `train_step!` (Adam with gradient clipping `max_grad_norm = 10`). This is the exogenous-ablation analog of the Python notebook's explicit segment-continuation loop (`X_segment, X_end = get_training_segment(X_start, model); X_start = X_end`): because each batch is independent there is no `X_start → X_end` state to carry, which is exactly what "no feedback from the current policy" means. The persistent companion `08_08` restores the continuation loop.
"""

# ╔═╡ 55555555-0807-4555-8555-555555555555
begin
    analytic_loss(model, ps, st, batch) = begin
        pieces, st_new = analytic_olg_residual(model, ps, st, batch; params)
        return pieces.loss, st_new
    end

    initial_batch = sample_analytic_olg_states(rng, params, hp.batch_size)
    initial_loss = loss_value(state, analytic_loss, initial_batch)
    history = NamedTuple[]
    for _ in 1:hp.steps
        local batch = sample_analytic_olg_states(rng, params, hp.batch_size)
        metrics = train_step!(state, analytic_loss, batch; max_grad_norm = 10.0)
        append_metric!(history; step = metrics.step, loss = metrics.loss)
    end
end

# ╔═╡ 1f0dabfc-97da-41f5-0717-a3e6ba7146f0
md"""
## 6. Final diagnostics

The diagnostics report relative Euler residuals on an out-of-sample cloud. Mean and max absolute Euler residuals are dimensionless, so a *mean* of, say, \$2 \times 10^{-3}\$ corresponds to a 0.2% average wedge. This compact preview evaluates on a single exogenous test cloud of 64 states (`sample_analytic_olg_states(rng, params, 64)`) for off-trajectory robustness; the persistent companion `08_08` additionally reports the simulated ergodic-cloud residuals of the full Python notebook.
"""

# ╔═╡ b520a59f-872c-16ce-cc7d-0778ff38c849
md"""
### 8. Closed-form validation

The trained policy is validated against the closed-form age-specific savings rates of Krueger and Kübler (2004),

\$\$
  \beta_h \;=\; \beta \cdot \frac{1 - \beta^{A - h}}{1 - \beta^{A - h + 1}},
  \qquad h = 1, \ldots, A - 1.
\$\$

Under log utility and i.i.d. shocks the optimal policy reduces to \$\hat a_t^h = \beta_h \cdot \mathrm{inc}_t^h\$: each cohort saves a fixed fraction of income, regardless of the shock. Here `analytic_olg_exact_policy` supplies the exact savings and `analytic_olg_policy_error` measures both the exact policy's residual (a sanity floor) and the learned network's deviation from \$\beta_h\$ — the one-line validation that gives this model its pedagogical role. The training loss itself stays unsupervised: the closed-form rates are reserved for validation, never used as training targets.

> The full Python notebook also runs a **policy-stability check** (§7): it evaluates the policy on a fixed holdout cloud after each monitoring interval and treats the run as stable once the RMS/max policy drift falls below tolerance. That drift machinery is omitted in this compact exogenous preview and carried by the persistent companion `08_08`. The Python notebook additionally draws loss/residual plots, replaced here by the returned diagnostics.
"""

# ╔═╡ 66666666-0807-4666-8666-666666666666
begin
    eval_states = sample_analytic_olg_states(rng, params, 64)
    diagnostics, _ = analytic_olg_residual(state.model, state.ps, state.st, eval_states; params)
    exact_policy = analytic_olg_exact_policy(eval_states; params)
    exact_error = analytic_olg_policy_error(exact_policy.savings, eval_states; params).summary
    learned_error = residual_summary(diagnostics.policy_error)
end

# ╔═╡ 332f3041-b4ac-0239-e9ba-38e57e872ee0
md"""
## Conclusion

This Lux/Pluto preview mapped the analytic 6-age OLG onto a DEQN: the Krueger–Kübler calibration (`AnalyticOLGParams`), the augmented network state, the sigmoid savings-fraction policy head, the relative-Euler loss (`analytic_olg_residual`), exogenous-cloud training, and validation against the closed-form savings rates \$\beta_h\$.

For a quick check keep `RUN_MODE = "smoke"`; use `"teaching"` for a classroom run and `"production"` for a paper-style run. The cell below returns a machine-checkable summary (closed-form rates, initial/final loss, max Euler residual, and the exact-vs-learned policy errors) for this notebook's run.
"""

# ╔═╡ 77777777-0807-4777-8777-777777777777
(
    sampling = :exogenous,
    closed_form_savings_rates = rates,
    initial_loss = initial_loss,
    final_loss = history[end].loss,
    max_abs_euler = residual_summary(diagnostics.euler).max_abs,
    exact_policy_check = exact_error.max_abs,
    learned_policy_error = learned_error.mean_abs,
)

# ╔═╡ Cell order:
# ╟─11111111-0807-4111-8111-111111111111
# ╟─b4ba56b1-2bf4-40c8-3d7b-d645c4f1a76a
# ╟─d54a2146-8553-06bd-5373-84c7004e7976
# ╠═22222222-0807-4222-8222-222222222222
# ╠═33333333-0807-4333-8333-333333333333
# ╟─67d0d5a5-34cd-4ef8-015d-fec7736fe380
# ╟─22f9055e-bca0-549f-e0c0-f2aa977101f5
# ╟─5885caa8-c92b-1740-68f1-aa6b6677423d
# ╠═44444444-0807-4444-8444-444444444444
# ╟─1f342403-9455-cd5f-1878-6f6376a79daf
# ╟─c538b2d8-c811-5bf8-600d-e9fa057820d1
# ╠═55555555-0807-4555-8555-555555555555
# ╟─1f0dabfc-97da-41f5-0717-a3e6ba7146f0
# ╟─b520a59f-872c-16ce-cc7d-0778ff38c849
# ╠═66666666-0807-4666-8666-666666666666
# ╟─332f3041-b4ac-0239-e9ba-38e57e872ee0
# ╠═77777777-0807-4777-8777-777777777777
