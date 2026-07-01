### A Pluto.jl notebook ###
# v0.20.0

using Markdown
using InteractiveUtils

# ╔═╡ 11111111-0809-4111-8111-111111111111
md"""
# Lecture 08, Notebook 09: Benchmark OLG DEQN, Exogenous Cloud in Lux

Smoke mode uses fewer cohorts for speed. Teaching and production modes switch
back to the 56-cohort benchmark helper.
"""

# ╔═╡ bbf2495d-1a94-a09a-f51f-bbd9f6354d76
md"""
## Lecture 08, Notebook 09: 56-Agent OLG Benchmark — DEQN (Exogenous Sampling)

**Course:** Deep Learning for Solving and Estimating Dynamic Models in Economics and Finance
**Script reference:** §5.5 (the 56-agent Azinovic–Gaegauf–Scheidegger 2022 benchmark; two assets, borrowing/collateral constraints, persistent shocks); §5.3–§5.4 (DEQN mapping, product-form KKT)
**Notebook role:** core
**Author:** Simon Scheidegger

*Julia/Lux/Pluto preview of* `lectures/lecture_08_olg_models_deqns/code/lecture_08_09_OLG_Benchmark_DEQN_exogenous.ipynb`.

> **Run mode.** The checked-in run uses `RUN_MODE = "smoke"` with fixed `SEED = 0`. In smoke mode the 56-cohort problem is **reduced to `n_ages = 8`** for a fast structural check — this is not convergence or production validation. `"teaching"` and `"production"` restore the full \$A = 56\$ benchmark (production uses ~1000 hidden units and a long two-stage schedule intended for a GPU/distributed run).

> **Sampling mode.** This is the exogenous-cloud ablation: every training batch is an independent feasible state cloud drawn from broad boxes, useful for stress-testing the residual code without simulation feedback. The primary classroom variant is the persistent-simulation companion `lecture_08_10_OLG_Benchmark_DEQN_persistent.ipynb`.
"""

# ╔═╡ a0072ef1-abec-f2dd-472b-ad051be4afba
md"""
This notebook solves the 56-period overlapping-generations benchmark of Azinovic, Gaegauf and Scheidegger (2022): risky illiquid capital, a risk-free one-period bond, cohort-specific borrowing and collateral constraints, lifecycle labor income, and aggregate TFP / depreciation shocks. The equilibrium conditions are the cohort-stacked capital and bond Euler equations, product-form KKT complementarity residuals for the borrowing and collateral inequalities (\$\lambda_b^h\,k'^h\$ and \$\mu^h\,q^h\$, both squared in the loss), and an explicit bond-market clearing residual.

The notebook mirrors the structure of the IRBC and analytic-OLG notebooks: parameters, augmented state, neural-network policy transform, residual / loss construction, training-data generation, mini-batch training, and out-of-sample residual tables. This is a teaching preview, not a full replacement for the Python benchmark artifact.
"""

# ╔═╡ 22222222-0809-4222-8222-222222222222
begin
    import Pkg
    Pkg.activate(joinpath(@__DIR__, "..", "..", "..", "julia"))

    using DLEFJulia
    using Lux
    using NNlib
    using Optimisers
    using Statistics
end

# ╔═╡ 33333333-0809-4333-8333-333333333333
begin
    RUN_MODE = "smoke"
    SEED = 0
    budgets = (
        smoke = (steps = 3, batch_size = 8, n_ages = 8),
        teaching = (steps = 60, batch_size = 64, n_ages = 56),
        production = (steps = 2_000, batch_size = 256, n_ages = 56),
    )
    hp = run_mode_budget(RUN_MODE; budgets)
    rng = rng_from_seed(SEED)
end

# ╔═╡ 371088a2-7ddd-d61d-e201-796e2e563a50
md"""
## 1. Economic parameters

The Azinovic–Gaegauf–Scheidegger (2022) benchmark calibration: \$A = 56\$ overlapping agents (ages 25–80), Cobb–Douglas production, a lifecycle labor profile, cohort-specific borrowing and collateral constraints, convex capital adjustment costs, and four persistent (Markov) aggregate-shock states combining TFP \$\eta\$ and depreciation \$\delta\$. Here `BenchmarkOLGParams(n_ages = hp.n_ages)` carries this calibration; in smoke mode `hp.n_ages = 8`, so the reported dimensions below are for the reduced structural check rather than the full \$A = 56\$.
"""

# ╔═╡ 22f9055e-bca0-549f-e0c0-f2aa977101f5
md"""
### 2. State representation and augmented network inputs

The minimal state is \$(z_t, \mathbf{k}_t, \mathbf{b}_t)\$ — the aggregate shock index plus the cohort capital and bond holdings — of dimension \$1 + 2A = 113\$ at \$A = 56\$. For learning, the network is fed an engineered state of dimension \$240\$,

\$\$
  \mathbf{s}_t = \bigl(z_t,\ \mathrm{onehot}(z_t),\ \eta_t,\ \delta_t,\ K_t,\ L_t,\ r_t,\ w_t,\ Y_t,\ \mathbf{k}_t,\ \mathrm{fininc}_t,\ \mathrm{labinc}_t,\ \mathrm{cash}_t,\ \boldsymbol{\pi}(z_t,\cdot)\bigr) \in \mathbb{R}^{240},
\$\$

where \$\mathrm{fininc}_t^h = r_t k_t^h + b_t^h\$, \$\mathrm{labinc}_t^h = w_t e^h\$, \$\mathrm{cash}_t^h = r_t k_t^h + b_t^h + w_t e^h\$, and \$Y_t = \eta_t K_t^\alpha L_t^{1-\alpha} + (1-\delta_t)K_t\$. The bond vector is recoverable from financial income and is not passed separately; the bond price is a network **output**, not an input. In Julia the feature width is `benchmark_olg_feature_dim(params)` (which scales with `n_ages`, so it is smaller in smoke mode) and the augmentation is feature engineering only.
"""

# ╔═╡ 5885caa8-c92b-1740-68f1-aa6b6677423d
md"""
### 3. Neural network and policy transformation

The policy network outputs four cohort-specific blocks plus the bond price:

\$\$
  \mathcal{N}_\theta(\mathbf{s}_t) \;\longrightarrow\; \bigl(\hat{k}_{t+1}^h,\, \hat{\lambda}_b^{h},\, \hat{q}^h,\, \hat{\mu}^h\bigr)_{h=1}^{A-1} \;\cup\; \{\hat{q}_t\},
\$\$

for a total of \$4(A-1) + 1\$ outputs (`raw_dim = 4 * (params.n_ages - 1) + 1`). Softplus heads enforce \$\hat{k}_{t+1}^h, \hat{\lambda}_b^h, \hat{\mu}^h \ge 0\$ by construction (the bond holding is recovered as \$\hat{b}_{t+1}^h = (\hat{q}^h - \hat{k}_{t+1}^h)/\kappa\$). The borrowing and collateral inequalities are then enforced softly via the squared product-form KKT residuals of §4. In Lux the network is `make_mlp(benchmark_olg_feature_dim(params), (32, 32), raw_dim; activation = NNlib.tanh)`, the policy transform is in `benchmark_olg_policy_from_raw` / `benchmark_olg_residual`, and `setup_training` uses `Optimisers.Adam` with Float64 parameters.
"""

# ╔═╡ 44444444-0809-4444-8444-444444444444
begin
    params = BenchmarkOLGParams(n_ages = hp.n_ages)
    raw_dim = 4 * (params.n_ages - 1) + 1
    model = make_mlp(benchmark_olg_feature_dim(params), (32, 32), raw_dim; activation = NNlib.tanh)
    state = setup_training(rng_from_seed(SEED; offset = 1), model, Optimisers.Adam(0.001); parameter_type = Float64)
end

# ╔═╡ 1f342403-9455-cd5f-1878-6f6376a79daf
md"""
## 4. Residuals and loss

For each cohort \$h = 1, \ldots, A-1\$ there is a capital Euler, a bond Euler, and two complementarity residuals. The capital Euler in relative form is

\$\$
  \mathrm{EulerErr}_h^{(k)}(\mathbf{s}_t) \;=\;
  \frac{\beta\, \mathbb{E}_t\!\left[\, u'(c_{t+1}^{h+1})\, \bigl(R_{t+1} + 1 - \delta_{t+1}\bigr)\,\right]}
       {u'(c_t^h)\,\bigl(1 + \zeta(k_{t+1}^h - r\,k_t^h)\bigr)} \;-\; 1,
\$\$

with the analogous residual for the bond. Borrowing- and collateral-constraint complementarity is enforced softly by adding *squared product residuals* to the loss:

\$\$
  \mathrm{KKT}_{\mathrm{borrow}}^h \;=\; \lambda_b^h \cdot k'^h, \qquad
  \mathrm{KKT}_{\mathrm{coll}}^h \;=\; \mu^h \cdot q^h.
\$\$

Both vanish exactly at any KKT point and have a differentiable squared form. (The Fischer–Burmeister alternative \$\Phi(a,b) = a + b - \sqrt{a^2 + b^2}\$ is used in the IRBC notebook of Chapter 3, where the irreversibility constraint binds more often.) Bond-market clearing is enforced as a residual, \$\sum_{h} b_{t+1}^h = 0\$. The total loss is the mean-squared sum of all \$4(A-1) + 1\$ residuals; `KKT_WEIGHT` and `MARKET_WEIGHT` put the complementarity and market-clearing residuals on the same scale as the Euler errors. In Julia this is `benchmark_olg_residual`, wrapped by `benchmark_loss`.
"""

# ╔═╡ c538b2d8-c811-5bf8-600d-e9fa057820d1
md"""
### Training-data construction and the training loop

In this exogenous version, training states are drawn from broad feasible boxes via `sample_benchmark_olg_states(rng, params, n)`, deliberately not centered on a steady state. The compact preview draws a **fresh independent batch on every step** and applies one `train_step!` (Adam, gradient clipping `max_grad_norm = 10`) — the feedback-free analog of the Python notebook's explicit segment-continuation loop, with no `X_start → X_end` state to carry. The persistent companion `08_10` restores the continuation loop.
"""

# ╔═╡ 55555555-0809-4555-8555-555555555555
begin
    benchmark_loss(model, ps, st, batch) = begin
        pieces, st_new = benchmark_olg_residual(model, ps, st, batch; params)
        return pieces.loss, st_new
    end

    initial_batch = sample_benchmark_olg_states(rng, params, hp.batch_size)
    initial_loss = loss_value(state, benchmark_loss, initial_batch)
    history = NamedTuple[]
    for _ in 1:hp.steps
        local batch = sample_benchmark_olg_states(rng, params, hp.batch_size)
        metrics = train_step!(state, benchmark_loss, batch; max_grad_norm = 10.0)
        append_metric!(history; step = metrics.step, loss = metrics.loss)
    end
end

# ╔═╡ a9c3aba2-cae9-dd50-6ddc-a9f175e1fbf8
md"""
## 6. Final diagnostics

The final diagnostics report the constraint- and equilibrium-condition residuals on an out-of-sample cloud: max absolute capital and bond Euler residuals, the minimum collateral slack, and the mean absolute bond-market-clearing residual (`benchmark_olg_residual` on `sample_benchmark_olg_states(rng, params, hp.batch_size)`). Mean and max absolute Euler residuals are dimensionless.

> The full Python notebook also runs a **policy-stability check** (§7, drift of the policy on a fixed holdout cloud) and draws **lifecycle plots** (cohort capital/consumption by shock state, and the bond-market residual along a simulated trajectory). Those are carried by the persistent companion `08_10` / the Python ground truth; this compact exogenous preview reports the residual summary instead.
"""

# ╔═╡ 66666666-0809-4666-8666-666666666666
begin
    eval_states = sample_benchmark_olg_states(rng, params, hp.batch_size)
    diagnostics, _ = benchmark_olg_residual(state.model, state.ps, state.st, eval_states; params)
end

# ╔═╡ 0367ce56-32a6-3a5a-3a7d-ed51d5bf1827
md"""
## Conclusion

This Lux/Pluto preview mapped the 56-cohort AGS (2022) benchmark onto a DEQN: two assets (illiquid capital + bond), softplus policy heads for non-negativity, product-form KKT residuals (\$\lambda_b^h k'^h\$, \$\mu^h q^h\$) for the borrowing/collateral inequalities, and a bond-market-clearing residual — all trained on exogenous feasible clouds.

In smoke mode this runs at `n_ages = 8` as a **structural check, not a convergence run**; set `RUN_MODE = "teaching"` or `"production"` to restore the full \$A = 56\$ benchmark. The cell below returns a machine-checkable summary (cohort count, initial/final loss, max capital/bond Euler residuals, minimum collateral slack, and mean bond-market residual) for this run.
"""

# ╔═╡ 77777777-0809-4777-8777-777777777777
(
    sampling = :exogenous,
    n_ages = params.n_ages,
    initial_loss = initial_loss,
    final_loss = history[end].loss,
    max_abs_capital_euler = residual_summary(diagnostics.euler_capital).max_abs,
    max_abs_bond_euler = residual_summary(diagnostics.euler_bond).max_abs,
    min_collateral_slack = minimum(diagnostics.collateral),
    mean_bond_market = mean(abs, diagnostics.bond_market),
)

# ╔═╡ Cell order:
# ╟─11111111-0809-4111-8111-111111111111
# ╟─bbf2495d-1a94-a09a-f51f-bbd9f6354d76
# ╟─a0072ef1-abec-f2dd-472b-ad051be4afba
# ╠═22222222-0809-4222-8222-222222222222
# ╠═33333333-0809-4333-8333-333333333333
# ╟─371088a2-7ddd-d61d-e201-796e2e563a50
# ╟─22f9055e-bca0-549f-e0c0-f2aa977101f5
# ╟─5885caa8-c92b-1740-68f1-aa6b6677423d
# ╠═44444444-0809-4444-8444-444444444444
# ╟─1f342403-9455-cd5f-1878-6f6376a79daf
# ╟─c538b2d8-c811-5bf8-600d-e9fa057820d1
# ╠═55555555-0809-4555-8555-555555555555
# ╟─a9c3aba2-cae9-dd50-6ddc-a9f175e1fbf8
# ╠═66666666-0809-4666-8666-666666666666
# ╟─0367ce56-32a6-3a5a-3a7d-ed51d5bf1827
# ╠═77777777-0809-4777-8777-777777777777
