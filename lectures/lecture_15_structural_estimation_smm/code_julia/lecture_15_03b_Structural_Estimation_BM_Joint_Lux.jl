### A Pluto.jl notebook ###
# v0.20.0

using Markdown
using InteractiveUtils

# ╔═╡ 11111111-1531-4111-8111-111111111111
md"""
# Lecture 15, Notebook 03b: Joint Brock-Mirman SMM in Lux

The policy surrogate takes `(z,K,beta,rho)` and the SMM criterion is evaluated
on a two-dimensional grid under one common shock path.
"""

# ╔═╡ ebe0ed77-7a27-e37c-4aa7-cecadefc9297
md"""
## Lecture 15, Notebook 03b: Joint structural estimation via SMM — Brock–Mirman with \$(\beta, \varrho)\$

**Course:** Deep Learning for Solving and Estimating Dynamic Models in Economics and Finance
**Script reference:** Chapter 10 (Structural estimation via SMM)
**Notebook role:** core
**Author:** Simon Scheidegger

*Julia/Lux/Pluto preview of* `lectures/lecture_15_structural_estimation_smm/code/lecture_15_03b_Structural_Estimation_BM_Joint.ipynb`.

> **Compute budget.** Lecture 15 uses fixed CPU budgets rather than a `RUN_MODE` switch. This Julia preview runs a deliberately tiny smoke budget (`N_TRAIN = 4`, short burn-in and horizon, coarse 3×3 parameter grid), so it demonstrates the joint surrogate-SMM pipeline end to end but is not a reproduction of the Python classroom-budget estimates. `SEED = 0` and common random numbers keep the run deterministic across the \$(\beta, \varrho)\$ grid.
"""

# ╔═╡ b33dfcf3-02df-55b1-8fc6-062547e4b86d
md"""
This joint notebook keeps the same minimal structure as the scalar exercise, now estimating two parameters at once:

1. train a single Lux policy surrogate with both parameters as pseudo-states;
2. generate synthetic data at \$(\beta_{\mathrm{true}}, \varrho_{\mathrm{true}}) = (0.96, 0.90)\$;
3. evaluate simulated moments over a two-dimensional parameter grid under common random numbers;
4. recover the parameters by SMM and inspect identification through the criterion surfaces and a moment Jacobian.

The active-learning and BoTorch material from the longer version has been removed from the main notebook; it is better treated as an optional research extension once the basic SMM estimator is clear.
"""

# ╔═╡ 22222222-1531-4222-8222-222222222222
begin
    import Pkg
    Pkg.activate(joinpath(@__DIR__, "..", "..", "..", "julia"))

    using DLEFJulia
    using Lux
    using NNlib
    using Optimisers
end

# ╔═╡ f43e08a0-6b93-ac52-df0c-8ee1c8522eb6
md"""
## 1. Joint pseudo-state surrogate

The surrogate now takes \$(z, K, \beta, \varrho)\$ as inputs and returns the savings rate \$s \in (0,1)\$. The underlying economy is the same partial-depreciation Brock–Mirman model as the scalar notebook — \$Y_t = z_t K_t^\alpha\$, \$K_{t+1} = (1-\delta)K_t + s_t Y_t\$, \$C_t = (1-s_t)Y_t\$, with \$\log z_{t+1} = \varrho \log z_t + \sigma_z \varepsilon_{t+1}\$ — but the discount factor \$\beta\$ now varies across the training batch alongside \$\varrho\$, so one surrogate covers the whole \$(\beta, \varrho)\$ rectangle.

The configuration cell below fixes the economic parameters (`SMMBrockMirmanParams`), the Gauss–Hermite quadrature rule, the synthetic truth \$(\beta_{\mathrm{true}}, \varrho_{\mathrm{true}}) = (0.96, 0.90)\$, and the coarse \$3\times 3\$ candidate grid (`beta_grid` × `rho_grid`) — deliberately built to contain the truth exactly — together with the tiny smoke budgets.
"""

# ╔═╡ 33333333-1531-4333-8333-333333333333
begin
    SEED = 0
    N_TRAIN = 4
    BATCH_SIZE = 32
    T_BURN = 8
    T_SIM = 24
    params = SMMBrockMirmanParams()
    rule = gauss_hermite_rule(5)
    beta_true = 0.96
    rho_true = 0.90
    beta_grid = [0.94, 0.96, 0.98]
    rho_grid = [0.85, 0.90, 0.95]
    theta_grid = hcat([[beta, rho] for rho in rho_grid for beta in beta_grid]...)
    rng = rng_from_seed(SEED)
end

# ╔═╡ 89215c9b-bb0a-be55-92cd-d6afbf7c5d38
md"""
We build the \$(z, K, \beta, \varrho) \mapsto s\$ surrogate as a Lux MLP (swish hidden layers, sigmoid head) and train it on the joint Euler-equation residual (`smm_joint_euler_loss`), using the same Gauss–Hermite quadrature as the scalar notebook. Both \$\beta\$ and \$\varrho\$ vary across the sampled batch (`sample_smm_joint_states`). Parameter gradients come from Zygote and are applied with `Optimisers.Adam` via `train_step!`, threading parameters and state explicitly. The checked-in budget is a tiny smoke run (`N_TRAIN = 4`); the full Python notebook also plots the training-loss curve to confirm convergence at the classroom budget.
"""

# ╔═╡ 44444444-1531-4444-8444-444444444444
begin
    model = make_mlp(4, (24, 24), 1; activation = NNlib.swish, final_activation = NNlib.sigmoid)
    state = setup_training(rng_from_seed(SEED; offset = 11), model, Optimisers.Adam(5e-4); parameter_type = Float64)
    loss_fn(model, ps, st, batch) = begin
        pieces, st_new = smm_joint_euler_loss(model, ps, st, batch, rule; params)
        return pieces.loss, st_new
    end
    initial_batch = sample_smm_joint_states(rng, BATCH_SIZE; params)
    initial_loss = loss_value(state, loss_fn, initial_batch)
    history = NamedTuple[]
    for step in 1:N_TRAIN
        local batch = sample_smm_joint_states(rng, BATCH_SIZE; params)
        metrics = train_step!(state, loss_fn, batch; max_grad_norm = 20.0)
        append_metric!(history; step, loss = metrics.loss)
    end
end

# ╔═╡ fbcdd25f-70f8-7375-6a5a-cb085e7063e7
md"""
## 2. Simulate the moment map on a two-dimensional grid

`simulate_smm_joint_theta` rolls the trained surrogate forward for every \$(\beta, \varrho)\$ candidate under one common shock path (`common_random_shocks`), and `smm_joint_moments` reduces each simulated \$(C, I, Y)\$ path to four summary moments. The grid deliberately contains the synthetic truth exactly, which makes the classroom result transparent: evaluated at the true parameter under the same shocks, the simulated moments line up with the synthetic data moments.
"""

# ╔═╡ c9810c73-351f-3e21-a2d6-f00996006adf
md"""
## 3. SMM criterion surfaces

We compare two moment sets, both scored with the identity-weighted SMM criterion:

- a **weak** two-moment set (`mask_weak = [false, true, true, false]`), dominated by dynamic-persistence information, which produces a shallow, ridge-like direction in \$\beta\$;
- the **over-identified** four-moment set (`mask_over = trues(4)`), which adds the level of the savings rate and produces a localized minimum.

This is the identification lesson: the estimator is only as informative as the moments used to build the criterion. `smm_grid_estimate` returns the grid minimizer for each specification (`estimate_weak`, `estimate_over`). The full Python notebook draws both criterion surfaces as contour plots.
"""

# ╔═╡ de3ec9ca-2b90-f456-c6b0-4da84f6927c7
md"""
## 4. Moment match and local identification diagnostics

The over-identified estimate is the preferred specification here. `smm_moment_jacobian_2d` builds the finite-difference Jacobian \$J = \partial m / \partial(\beta, \varrho)\$ at the truth — how the selected moments move locally with the parameters — and `smm_identification_svd` takes its singular value decomposition. A small singular value flags a weakly identified direction, so the ratio of largest to smallest singular value acts as a local identification condition number.
"""

# ╔═╡ 55555555-1531-4555-8555-555555555555
begin
    shocks = common_random_shocks(rng_from_seed(SEED; offset = 123), T_BURN + T_SIM)
    sim = simulate_smm_joint_theta(state.model, state.ps, state.st, theta_grid, shocks;
        params, T_burn = T_BURN, T_sim = T_SIM)
    moments_flat = smm_joint_moments(sim.C, sim.I, sim.Y)
    moments = reshape(moments_flat, length(beta_grid), length(rho_grid), 4)
    moments = permutedims(moments, (2, 1, 3))
    idx_beta_true = findfirst(x -> isapprox(x, beta_true), beta_grid)
    idx_rho_true = findfirst(x -> isapprox(x, rho_true), rho_grid)
    target_moments = moments[idx_rho_true, idx_beta_true, :]
    mask_weak = [false, true, true, false]
    mask_over = trues(4)
    estimate_over = smm_grid_estimate(theta_grid, moments_flat, target_moments; mask = mask_over)
    estimate_weak = smm_grid_estimate(theta_grid, moments_flat, target_moments; mask = mask_weak)
    J = smm_moment_jacobian_2d(moments, beta_grid, rho_grid, idx_beta_true, idx_rho_true)
    ids = smm_identification_svd(J; mask = mask_over)
end

# ╔═╡ 1ed81232-2fbf-6956-3f57-a6b04f73bd0e
md"""
The full Python notebook closes with a set of policy-match plots connecting the SMM estimate back to the trained surrogate, comparing the estimated policy against the truth across \$K\$. This compact preview reports the recovered parameters and identification diagnostics numerically instead.
"""

# ╔═╡ 749c0ee0-e423-9c04-c957-86ad4b1b8f97
md"""
## Summary

The joint notebook focuses on the essential structural-estimation message:

- a single four-input policy surrogate covers the whole \$(\beta, \varrho)\$ rectangle;
- common random numbers make the simulated moment map deterministic;
- the criterion surfaces reveal which moments identify which parameter directions;
- the over-identified SMM criterion recovers the synthetic truth accurately, while the weak two-moment set leaves \$\beta\$ poorly pinned down.

The removed GP / active-learning machinery is valuable for high-throughput estimation, but it belongs after this core workflow rather than inside the introductory notebook. The cell below returns a machine-checkable summary — training loss, the over-identified \$(\widehat{\beta}, \widehat{\varrho})\$ and their absolute errors, the weak-moment estimates, and the smallest singular value of the moment Jacobian.
"""

# ╔═╡ 66666666-1531-4666-8666-666666666666
(
    initial_loss = initial_loss,
    final_loss = history[end].loss,
    beta_true = beta_true,
    rho_true = rho_true,
    beta_hat_over = estimate_over.theta[1],
    rho_hat_over = estimate_over.theta[2],
    beta_abs_error = abs(estimate_over.theta[1] - beta_true),
    rho_abs_error = abs(estimate_over.theta[2] - rho_true),
    weak_beta_hat = estimate_weak.theta[1],
    weak_rho_hat = estimate_weak.theta[2],
    smallest_singular_value = minimum(ids.singular_values),
)

# ╔═╡ Cell order:
# ╟─11111111-1531-4111-8111-111111111111
# ╟─ebe0ed77-7a27-e37c-4aa7-cecadefc9297
# ╟─b33dfcf3-02df-55b1-8fc6-062547e4b86d
# ╠═22222222-1531-4222-8222-222222222222
# ╟─f43e08a0-6b93-ac52-df0c-8ee1c8522eb6
# ╠═33333333-1531-4333-8333-333333333333
# ╟─89215c9b-bb0a-be55-92cd-d6afbf7c5d38
# ╠═44444444-1531-4444-8444-444444444444
# ╟─fbcdd25f-70f8-7375-6a5a-cb085e7063e7
# ╟─c9810c73-351f-3e21-a2d6-f00996006adf
# ╟─de3ec9ca-2b90-f456-c6b0-4da84f6927c7
# ╠═55555555-1531-4555-8555-555555555555
# ╟─1ed81232-2fbf-6956-3f57-a6b04f73bd0e
# ╟─749c0ee0-e423-9c04-c957-86ad4b1b8f97
# ╠═66666666-1531-4666-8666-666666666666
