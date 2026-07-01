### A Pluto.jl notebook ###
# v0.20.0

using Markdown
using InteractiveUtils

# ╔═╡ 11111111-0304-4111-8111-111111111111
md"""
# Lecture 03, Notebook 04: DEQN Exercises (Solutions)

This solution notebook keeps the exercises compact and Lux-native. It shows the
same building blocks as the Python solution: stochastic Brock-Mirman, a second
policy head for labor, Fischer-Burmeister complementarity, and a tiny life-cycle
residual check.
"""

# ╔═╡ 64d1604e-92a9-7fd9-3f63-45553171f299
md"""
## Lecture 03, Notebook 04: DEQN Exercises (Solutions)

**Course:** Deep Learning for Solving and Estimating Dynamic Models in Economics and Finance
**Script reference:** §2.4–2.5 (Brock–Mirman benchmark; KKT + Fischer–Burmeister complementarity); previews the IRBC model of Ch. 3 and the OLG model of Ch. 5
**Notebook role:** solution
**Author:** Simon Scheidegger

*Julia/Lux/Pluto preview of* `lectures/lecture_03_deep_equilibrium_nets/code/lecture_03_04_DEQN_Exercises_Solutions.ipynb`.

> **This is the solutions notebook** for the four exercises in Notebook 03. Each cell implements one exercise in a **compact, Lux-native** way; it does not reproduce every detail of the Python solutions. In particular the labor, constrained-labor, and life-cycle cells use a **one-step Euler proxy** (deterministic, no Gauss–Hermite expectation over next-period shocks) so they stay smoke-sized and fast.
"""

# ╔═╡ 23ccefbd-9193-47e4-3579-5698e4698b2e
md"""
## Simple Introduction to Deep Equilibrium Nets

### Notebook 4: worked solutions to the coding session

We solve the **four exercises** of Notebook 03:

1. **Stochastic Brock–Mirman** — the Notebook 2 model, trained with the full quadrature expectation.
2. **Endogenous labor supply** — a two-head policy \$[K_{t+1}, L_t]\$ with a savings Euler residual plus an intratemporal labor FOC.
3. **Occasionally binding labor constraint** — the labor time constraint \$L_t \le 1.01\$ encoded with Fischer–Burmeister complementarity.
4. **A small life-cycle (OLG) economy** — age-specific savings with borrowing constraints, shown as a compact residual check.

Each solution builds on `DLEFJulia` helpers (`stochastic_bm_residual`, `split_output_heads`, `fischer_burmeister`, `positive_softplus`, `sigmoid_bounds`) with explicit Lux `model(x, ps, st)` calls.
"""

# ╔═╡ 22222222-0304-4222-8222-222222222222
begin
    import Pkg
    Pkg.activate(joinpath(@__DIR__, "..", "..", "..", "julia"))

    using CairoMakie
    using DLEFJulia
    using Lux
    using NNlib
    using Optimisers
    using Statistics
end

# ╔═╡ 57b733cb-0038-ecaf-e915-5613b82f1617
md"""
`RUN_MODE` selects the training budget and `SEED = 0` fixes the RNG; this preview keeps `RUN_MODE = "smoke"` and `SEED = 0`. Smoke budgets check loadability and finite residuals, not production-quality policies.
"""

# ╔═╡ 33333333-0304-4333-8333-333333333333
begin
    RUN_MODE = "smoke"
    SEED = 0
    budgets = (
        smoke = (steps = 25, batch_size = 48),
        teaching = (steps = 400, batch_size = 128),
        production = (steps = 2_000, batch_size = 256),
    )
    hp = run_mode_budget(RUN_MODE; budgets)
    rng = rng_from_seed(SEED)
end

# ╔═╡ 24b41796-0574-30fe-4fd0-af8952a61f2a
md"""
### Exercise 1: stochastic Brock–Mirman

The model of Notebook 2: state \$\mathbf{X}_t = [z_t, K_t]\$, policy \$K_{t+1} = \mathcal{N}(\mathbf{X}_t)\$, and the relative-consumption Euler residual

\$\$0 = \frac{1}{C_t\,\beta\, E\!\left[\frac{1}{C_{t+1}}(1 - \delta + r_{t+1})\right]} - 1, \qquad r_t = \alpha z_t K_t^{\alpha-1},\$\$

with the expectation taken by Gauss–Hermite quadrature. The cell below wires `BrockMirmanParams`, `gauss_hermite_rule(5)`, and `stochastic_bm_residual` together, maps the raw output to a feasible savings rate with a **sigmoid** (the hard feasibility constraint), samples states uniformly, and trains with `Optimisers.Adam(0.001)`.

The full Python notebook also covers *simulating the policy forward* and *iterating between training and simulation* — building an ergodic set with `simulate_single_step`/`sim_periods` and retraining the DEQN on those simulated states. This preview trains on uniformly sampled states only and omits the simulation-forward loop.
"""

# ╔═╡ 44444444-0304-4444-8444-444444444444
begin
    params = BrockMirmanParams(alpha = 0.36, beta = 0.99, delta = 0.1, rho_z = 0.9, sigma_z = 0.04)
    rule = gauss_hermite_rule(5)

    sample_states(rng, n) = vcat(reshape(0.75 .+ 0.5 .* rand(rng, n), 1, :), reshape(0.9 .+ 11.1 .* rand(rng, n), 1, :))

    bm_model = make_mlp(2, (24, 24), 1; activation = NNlib.tanh)
    bm_state = setup_training(rng_from_seed(SEED; offset = 1), bm_model, Optimisers.Adam(0.001); parameter_type = Float64)
    bm_loss(model, ps, st, states) = begin
        pieces, st_new = stochastic_bm_residual(model, ps, st, states, rule; params, transform = NNlib.sigmoid)
        return pieces.loss, st_new
    end

    bm_initial = loss_value(bm_state, bm_loss, sample_states(rng, hp.batch_size))
    bm_history = NamedTuple[]
    for _ in 1:hp.steps
        metrics = train_step!(bm_state, bm_loss, sample_states(rng, hp.batch_size); max_grad_norm = 10.0)
        append_metric!(bm_history; step = metrics.step, loss = metrics.loss)
    end
end

# ╔═╡ 85909a4b-da15-47ff-2ee7-bdb083726739
md"""
### Exercise 2: endogenous labor supply

Labor is now a choice, so the policy \$\mathbf{f}(\mathbf{X}_t) = [K_{t+1}, L_t]\$ is two-dimensional and output is \$Y_t = z_t L_t^{1-\alpha} K_t^{\alpha}\$. There are two optimality conditions: the capital Euler equation and the intratemporal labor FOC

\$\$0 = \psi\, L_t^{\gamma} - \frac{w_t}{C_t}, \qquad r_t = \alpha K_t^{\alpha-1} L_t^{1-\alpha}, \quad w_t = (1-\alpha)K_t^{\alpha} L_t^{-\alpha},\$\$

where \$\gamma\$ is the labor-supply curvature (the \$\theta\$ of the write-up). The cell splits the network's two heads with `split_output_heads`, bounds savings with a **sigmoid** and labor with **`positive_softplus`** (labor must be positive), and sums the squared Euler and labor residuals into the loss. *(For speed the solution uses a compact one-step Euler proxy rather than the full expectation. This preview also sets the curvature to \$\gamma = 2.0\$, whereas the Python write-up uses \$\theta = 1.0\$, so the implied Frisch elasticity differs.)*
"""

# ╔═╡ 55555555-0304-4555-8555-555555555555
begin
    labor_params = (alpha = 0.36, beta = 0.99, delta = 0.1, psi = 1.0, gamma = 2.0)
    two_head_model = make_mlp(2, (24, 24), 2; activation = NNlib.tanh)
    two_head_state = setup_training(rng_from_seed(SEED; offset = 2), two_head_model, Optimisers.Adam(0.001); parameter_type = Float64)

    function labor_deqn_residual(model, ps, st, states; constrained = false, labor_cap = 1.01)
        x = assert_feature_batch(states, 2)
        z = @view x[1:1, :]
        k = @view x[2:2, :]
        raw, st_new = model(x, ps, st)
        heads = split_output_heads(raw, (savings = 1, labor = 1))
        savings = NNlib.sigmoid.(heads.savings)
        labor_unbounded = positive_softplus(heads.labor; floor = 0.15, scale = 0.8)
        labor = constrained ? sigmoid_bounds(heads.labor, 0.15, labor_cap) : labor_unbounded

        output = z .* k .^ labor_params.alpha .* labor .^ (1 - labor_params.alpha)
        consumption = output .* (1 .- savings)
        k_next = (1 - labor_params.delta) .* k .+ output .* savings
        wage = (1 - labor_params.alpha) .* z .* k .^ labor_params.alpha .* labor .^ (-labor_params.alpha)
        rental = labor_params.alpha .* z .* k .^ (labor_params.alpha - 1) .* labor .^ (1 - labor_params.alpha)
        labor_foc = labor_params.psi .* labor .^ labor_params.gamma .- wage ./ consumption

        # A compact one-step Euler proxy for the exercise solution.
        euler_residual = 1 .- 1 ./ (consumption .* labor_params.beta .* (1 .- labor_params.delta .+ rental))
        labor_residual = constrained ? fischer_burmeister(labor_cap .- labor, -labor_foc) : labor_foc
        loss = mean(abs2, euler_residual) + mean(abs2, labor_residual)
        return (loss = loss, euler = euler_residual, labor = labor_residual, savings = savings, labor_policy = labor, next_capital = k_next), st_new
    end

    labor_loss(model, ps, st, states) = begin
        pieces, st_new = labor_deqn_residual(model, ps, st, states)
        return pieces.loss, st_new
    end

    labor_initial = loss_value(two_head_state, labor_loss, sample_states(rng, hp.batch_size))
    for _ in 1:hp.steps
        train_step!(two_head_state, labor_loss, sample_states(rng, hp.batch_size); max_grad_norm = 10.0)
    end
    labor_diagnostics, _ = labor_deqn_residual(two_head_state.model, two_head_state.ps, two_head_state.st, sample_states(rng, hp.batch_size))
end

# ╔═╡ 3e2f8e7f-c102-c886-b838-5390d99e4e12
md"""
### Exercise 3: occasionally binding labor constraint

With a time constraint \$L_t \le 1.01\$ the interior labor FOC becomes a Kuhn–Tucker system. A single **Fischer–Burmeister** equation encodes the complementarity:

\$\$f^{FB}(a, b) = \sqrt{a^2 + b^2} - a - b = 0 \iff (a = 0, b \ge 0)\ \text{or}\ (a \ge 0, b = 0),\$\$

with \$a\$ the slack \$1.01 - L_t\$ and \$b\$ the labor-FOC wedge. The cell reuses `labor_deqn_residual` with `constrained = true`: labor is squashed into its admissible range with **`sigmoid_bounds`**, and the labor residual becomes `fischer_burmeister(labor_cap - labor, -labor_foc)`.
"""

# ╔═╡ 66666666-0304-4666-8666-666666666666
begin
    constrained_model = make_mlp(2, (20, 20), 2; activation = NNlib.tanh)
    constrained_state = setup_training(rng_from_seed(SEED; offset = 3), constrained_model, Optimisers.Adam(0.001); parameter_type = Float64)
    constrained_loss(model, ps, st, states) = begin
        pieces, st_new = labor_deqn_residual(model, ps, st, states; constrained = true, labor_cap = 1.01)
        return pieces.loss, st_new
    end
    constrained_initial = loss_value(constrained_state, constrained_loss, sample_states(rng, hp.batch_size))
    for _ in 1:hp.steps
        train_step!(constrained_state, constrained_loss, sample_states(rng, hp.batch_size); max_grad_norm = 10.0)
    end
    constrained_diagnostics, _ = labor_deqn_residual(constrained_state.model, constrained_state.ps, constrained_state.st, sample_states(rng, hp.batch_size); constrained = true, labor_cap = 1.01)
end

# ╔═╡ 4d010cfd-4541-535f-985b-4e99e915b437
md"""
### Exercise 4: small life-cycle (OLG) economy

Households live \$H = 6\$ periods with a borrowing constraint \$k_t^h \ge 0\$ and age-specific labor endowments \$l^h\$ (lower in retirement). For each age the Euler equation

\$\$\frac{1}{c_t^h} \ge \beta\, E\!\left[\frac{1}{c_{t+1}^{h+1}}(1 - \delta + r_{t+1})\right], \qquad c_t^h = l^h w_t + k_t^h(1-\delta+r_t) - k_{t+1}^{h+1},\$\$

holds with equality whenever \$k_{t+1}^{h+1} > 0\$ — again Fischer–Burmeister complementarity. The cell below is a **compact deterministic residual check**: it builds a life-cycle asset/consumption path from a fixed savings guess and evaluates the per-age Fischer–Burmeister residual, illustrating the OLG structure without training a full network.
"""

# ╔═╡ 77777777-0304-4777-8777-777777777777
begin
    H = 6
    beta_life = 0.96^10
    labor_endowment = [1.0, 1.0, 0.95, 0.9, 0.35, 0.2]
    lifecycle_raw_savings = fill(0.18, H - 1)
    lifecycle_assets = cumsum(vcat(1.0, lifecycle_raw_savings))
    lifecycle_consumption = max.(0.05, labor_endowment .+ lifecycle_assets .- vcat(lifecycle_assets[2:end], 0.0))
    lifecycle_euler = 1 ./ lifecycle_consumption[1:(H - 1)] .- beta_life ./ lifecycle_consumption[2:H]
    lifecycle_fb = fischer_burmeister.(lifecycle_assets[2:H], -lifecycle_euler)
    lifecycle_loss = mean(abs2, lifecycle_fb)
end

# ╔═╡ abf1fb42-fa54-aa12-7a35-f4540bc97632
md"""
### Checking the Brock–Mirman solution

As a sanity check we scatter the trained Exercise 1 residual against capital over freshly sampled states: a well-trained policy keeps the relative-consumption error tight around zero across the sampled range.
"""

# ╔═╡ 88888888-0304-4888-8888-888888888888
begin
    eval_states = sample_states(rng, 80)
    bm_diag, _ = stochastic_bm_residual(bm_state.model, bm_state.ps, bm_state.st, eval_states, rule; params, transform = NNlib.sigmoid)
    fig = Figure(size = figure_size(RUN_MODE))
    ax = Axis(fig[1, 1], xlabel = "K_t", ylabel = "residual")
    scatter!(ax, vec(eval_states[2:2, :]), vec(bm_diag.residual); color = (:dodgerblue3, 0.65))
    fig
end

# ╔═╡ 138bfab6-aa75-8c0d-a8c4-c12f6f8fd2a4
md"""
### Conclusion

These four solutions show the DEQN loss-design toolkit in one place: quadrature for conditional expectations (Exercise 1), multi-head policies with an added intratemporal FOC (Exercise 2), Fischer–Burmeister complementarity for occasionally binding constraints (Exercise 3), and the OLG residual structure with borrowing constraints (Exercise 4). The compact one-step and deterministic proxies keep the notebook smoke-sized; the full Python solutions train each model with the complete stochastic expectation. The cell below returns a machine-checkable summary of every exercise's initial/final loss and finiteness.
"""

# ╔═╡ 99999999-0304-4999-8999-999999999999
(
    bm_initial_loss = bm_initial,
    bm_final_loss = bm_history[end].loss,
    endogenous_labor_initial = labor_initial,
    endogenous_labor_final = labor_diagnostics.loss,
    constrained_labor_initial = constrained_initial,
    constrained_labor_final = constrained_diagnostics.loss,
    constrained_labor_max = maximum(constrained_diagnostics.labor_policy),
    lifecycle_loss = lifecycle_loss,
    lifecycle_finite = all(isfinite, lifecycle_fb),
)

# ╔═╡ Cell order:
# ╟─11111111-0304-4111-8111-111111111111
# ╟─64d1604e-92a9-7fd9-3f63-45553171f299
# ╟─23ccefbd-9193-47e4-3579-5698e4698b2e
# ╠═22222222-0304-4222-8222-222222222222
# ╟─57b733cb-0038-ecaf-e915-5613b82f1617
# ╠═33333333-0304-4333-8333-333333333333
# ╟─24b41796-0574-30fe-4fd0-af8952a61f2a
# ╠═44444444-0304-4444-8444-444444444444
# ╟─85909a4b-da15-47ff-2ee7-bdb083726739
# ╠═55555555-0304-4555-8555-555555555555
# ╟─3e2f8e7f-c102-c886-b838-5390d99e4e12
# ╠═66666666-0304-4666-8666-666666666666
# ╟─4d010cfd-4541-535f-985b-4e99e915b437
# ╠═77777777-0304-4777-8777-777777777777
# ╟─abf1fb42-fa54-aa12-7a35-f4540bc97632
# ╠═88888888-0304-4888-8888-888888888888
# ╟─138bfab6-aa75-8c0d-a8c4-c12f6f8fd2a4
# ╠═99999999-0304-4999-8999-999999999999
