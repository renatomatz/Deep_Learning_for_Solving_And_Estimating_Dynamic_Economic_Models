### A Pluto.jl notebook ###
# v0.20.0

using Markdown
using InteractiveUtils

# ╔═╡ 11111111-1503-4111-8111-111111111111
md"""
# Lecture 15, Notebook 03: Scalar Brock-Mirman SMM in Lux

A single Lux policy surrogate treats persistence `rho` as a pseudo-state. The
SMM grid uses common random numbers, so the simulated moment map is deterministic
across candidate values.
"""

# ╔═╡ 2809f225-0925-9277-4934-c209f517df83
md"""
## Lecture 15, Notebook 03: Structural estimation via SMM — Brock–Mirman, scalar persistence

**Course:** Deep Learning for Solving and Estimating Dynamic Models in Economics and Finance
**Script reference:** Chapter 10 (Structural estimation via SMM)
**Notebook role:** core
**Author:** Simon Scheidegger

*Julia/Lux/Pluto preview of* `lectures/lecture_15_structural_estimation_smm/code/lecture_15_03_Structural_Estimation_BM.ipynb`.

> **Compute budget.** Lecture 15 uses fixed CPU budgets rather than a `RUN_MODE` switch. This Julia preview runs a deliberately tiny smoke budget (`N_TRAIN = 4`, short burn-in and horizon), so it demonstrates the surrogate-SMM pipeline end to end but is not a reproduction of the Python classroom-budget estimates. `SEED = 0` and common random numbers keep the run deterministic across candidate \$\varrho\$.
"""

# ╔═╡ b6a9f78f-6997-58e4-3d97-b205cd1d1f0b
md"""
This compact preview keeps the surrogate-SMM demonstration deliberately minimal. The workflow is:

1. train one Lux neural policy surrogate with the structural parameter \$\varrho\$ as an input;
2. generate synthetic data at \$\varrho_{\mathrm{true}}=0.90\$;
3. simulate the same surrogate for candidate values of \$\varrho\$ under common random numbers;
4. estimate \$\varrho\$ by minimizing a simple SMM criterion.

The Gaussian-process moment-map and active-learning layer from the longer version is omitted here. That layer is useful for research-scale bootstraps and high-dimensional search, but it obscures the basic estimation idea in a first pass.
"""

# ╔═╡ 22222222-1503-4222-8222-222222222222
begin
    import Pkg
    Pkg.activate(joinpath(@__DIR__, "..", "..", "..", "julia"))

    using DLEFJulia
    using Lux
    using NNlib
    using Optimisers
end

# ╔═╡ de8731e4-86c5-de23-35d4-a35a82a4f982
md"""
## 1. Model, quadrature, and the pseudo-state policy surrogate

The model is the partial-depreciation Brock–Mirman economy used in Chapter 10:

\$\$
Y_t = z_t K_t^\alpha, \qquad
K_{t+1} = (1-\delta)K_t + s_t Y_t, \qquad
C_t = (1-s_t)Y_t,
\$\$

with \$\log z_{t+1} = \varrho \log z_t + \sigma_z \varepsilon_{t+1}\$. The network learns the savings-rate policy

\$\$
(z_t, K_t, \varrho) \mapsto s_t \in (0,1),
\$\$

treating the structural parameter \$\varrho\$ as a pseudo-state, so a single trained surrogate covers the whole family of models.

The configuration cell below fixes the economic parameters (`SMMBrockMirmanParams`), the Gauss–Hermite quadrature rule (`gauss_hermite_rule(5)`) used to evaluate the conditional expectation in the Euler equation, the candidate grid \$\varrho \in \{0.80, 0.85, 0.90, 0.95\}\$, and the tiny smoke budgets. A short steady-state anchor fixes the economically relevant low-saving branch during classroom-scale training; the Euler residual remains the training objective.
"""

# ╔═╡ 33333333-1503-4333-8333-333333333333
begin
    SEED = 0
    N_TRAIN = 4
    BATCH_SIZE = 32
    T_BURN = 8
    T_SIM = 24
    params = SMMBrockMirmanParams()
    rule = gauss_hermite_rule(5)
    rho_true = 0.90
    rho_grid = collect(0.80:0.05:0.95)
    rng = rng_from_seed(SEED)
end

# ╔═╡ b5020e8a-8727-dab8-b738-3d8d3e886efc
md"""
We instantiate the surrogate as a Lux MLP \$(z, K, \varrho) \mapsto s \in (0,1)\$ — two hidden layers of swish units with a sigmoid output head — and train it on the Euler-equation residual (`smm_scalar_euler_loss`), integrating the conditional expectation with the Gauss–Hermite rule. Parameter gradients flow through Zygote and are applied with `Optimisers.Adam`; the model, parameters, and state are threaded explicitly through `train_step!` in the Lux `y, st = model(x, ps, st)` style, and training states are drawn feature-by-batch with `sample_smm_scalar_states`. The checked-in budget is a tiny smoke run (`N_TRAIN = 4`); the slide figures use the longer classroom budget.

After training, the full Python notebook inspects the trained policy family, plotting \$s(K)\$ across \$\varrho\$ at \$z=1\$ to show that a single surrogate — trained once over the whole \$(K,\varrho)\$ rectangle — is then reused inside the estimator. This compact preview skips that figure and proceeds directly to estimation.
"""

# ╔═╡ 44444444-1503-4444-8444-444444444444
begin
    model = make_mlp(3, (24, 24), 1; activation = NNlib.swish, final_activation = NNlib.sigmoid)
    state = setup_training(rng_from_seed(SEED; offset = 1), model, Optimisers.Adam(5e-4); parameter_type = Float64)
    loss_fn(model, ps, st, batch) = begin
        pieces, st_new = smm_scalar_euler_loss(model, ps, st, batch, rule; params)
        return pieces.loss, st_new
    end
    initial_batch = sample_smm_scalar_states(rng, BATCH_SIZE; params)
    initial_loss = loss_value(state, loss_fn, initial_batch)
    history = NamedTuple[]
    for step in 1:N_TRAIN
        local batch = sample_smm_scalar_states(rng, BATCH_SIZE; params)
        metrics = train_step!(state, loss_fn, batch; max_grad_norm = 20.0)
        append_metric!(history; step, loss = metrics.loss)
    end
end

# ╔═╡ b4d62998-3df4-ccf7-beec-2b5b4526c227
md"""
## 3. Synthetic data, common random numbers, and moments

The synthetic data are generated at \$\varrho_{\mathrm{true}}=0.90\$. Every candidate \$\varrho\$ on the grid is then evaluated with the same initial condition, burn-in, horizon, and shock sequence: `common_random_shocks` draws one shock path that is reused across all candidates. These common random numbers make the SMM criterion a deterministic and smooth function of \$\varrho\$.

`simulate_smm_scalar_rho` rolls the surrogate forward for each candidate, and `smm_scalar_moments` reduces each simulated path \$(C, I, Y)\$ to four summary moments. The full Python notebook plots each moment against \$\varrho\$ to show how the dynamic moments respond to persistence. It also prints an analytic identification aside — for the AR(1) shock, \$\mathrm{Std}(\Delta \log z) = \sigma_z\sqrt{2/(1+\varrho)}\$ while the level variance is \$\sigma_z^2/(1-\varrho^2)\$ — which anchors the identifying consumption-growth moment to the underlying shock structure.
"""

# ╔═╡ a0400ba2-a0b6-fabb-86ed-737b5a78fe5a
md"""
## 4. SMM estimation

The SMM criterion compares simulated to data moments under identity weighting:

\$\$
Q(\varrho) = \left[m(\varrho) - \widehat{m}^{\,\text{data}}\right]' \left[m(\varrho) - \widehat{m}^{\,\text{data}}\right].
\$\$

Only the three dynamic moments enter scalar persistence estimation (`used = [true, true, true, false]`); the fourth moment, the mean savings rate, is displayed as a diagnostic but left out of the criterion. `smm_grid_estimate` returns the grid minimizer \$\widehat{\varrho}\$, and `smm_moment_sensitivity_1d` reports the finite-difference sensitivity \$\partial m / \partial \varrho\$ at the truth — the local slope that identifies \$\varrho\$.
"""

# ╔═╡ 55555555-1503-4555-8555-555555555555
begin
    shocks = common_random_shocks(rng_from_seed(SEED; offset = 123), T_BURN + T_SIM)
    sim = simulate_smm_scalar_rho(state.model, state.ps, state.st, rho_grid, shocks;
        params, T_burn = T_BURN, T_sim = T_SIM)
    moments = smm_scalar_moments(sim.C, sim.I, sim.Y)
    idx_true = findfirst(x -> isapprox(x, rho_true), rho_grid)
    target_moments = moments[idx_true, :]
    used = [true, true, true, false]
    estimate = smm_grid_estimate(rho_grid, moments, target_moments; mask = used)
    sensitivity = smm_moment_sensitivity_1d(moments, rho_grid, idx_true)
end

# ╔═╡ dbeebaba-ecca-7f9e-fcae-45a872912449
md"""
## Summary

The scalar notebook contains only the core teaching pipeline:

- one pseudo-state policy surrogate is trained with \$\varrho\$ as an input (\$\varrho\in[0.50,0.99]\$);
- synthetic data are generated at \$\varrho_{\mathrm{true}}=0.90\$;
- common random numbers turn the simulated moment map into a deterministic function of \$\varrho\$;
- the SMM profile has a clear interior minimum and recovers the synthetic truth.

This is the simplest useful classroom demonstration; the Gaussian-process moment-map layer can be reintroduced later as an optional extension once students understand the surrogate-SMM estimator itself. The cell below returns a machine-checkable summary of this smoke run — initial and final training loss, the recovered \$\widehat{\varrho}\$ and its absolute error, the criterion minimum, and the norm of the moment sensitivity.
"""

# ╔═╡ 66666666-1503-4666-8666-666666666666
(
    initial_loss = initial_loss,
    final_loss = history[end].loss,
    rho_true = rho_true,
    rho_hat = estimate.theta,
    rho_abs_error = abs(estimate.theta - rho_true),
    criterion_min = estimate.value,
    moment_sensitivity_norm = sqrt(sum(abs2, sensitivity[used])),
)

# ╔═╡ Cell order:
# ╟─11111111-1503-4111-8111-111111111111
# ╟─2809f225-0925-9277-4934-c209f517df83
# ╟─b6a9f78f-6997-58e4-3d97-b205cd1d1f0b
# ╠═22222222-1503-4222-8222-222222222222
# ╟─de8731e4-86c5-de23-35d4-a35a82a4f982
# ╠═33333333-1503-4333-8333-333333333333
# ╟─b5020e8a-8727-dab8-b738-3d8d3e886efc
# ╠═44444444-1503-4444-8444-444444444444
# ╟─b4d62998-3df4-ccf7-beec-2b5b4526c227
# ╟─a0400ba2-a0b6-fabb-86ed-737b5a78fe5a
# ╠═55555555-1503-4555-8555-555555555555
# ╟─dbeebaba-ecca-7f9e-fcae-45a872912449
# ╠═66666666-1503-4666-8666-666666666666
