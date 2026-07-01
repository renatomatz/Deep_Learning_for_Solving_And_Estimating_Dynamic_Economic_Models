### A Pluto.jl notebook ###
# v0.20.0

using Markdown
using InteractiveUtils

# ╔═╡ 11111111-0401-4111-8111-111111111111
md"""
# Lecture 04, Notebook 01: Smooth IRBC DEQN in Lux

This Julia/Pluto preview follows the Python smooth IRBC notebook's economic
objects directly.  The state is ordered as `[k_1,...,k_N,z_1,...,z_N]`, the
policy returns `[k'_1,...,k'_N,lambda]`, and the residuals use the Python
complete-markets lambda policy, adjustment-cost Euler wedge, aggregate resource
constraint, monomial Stroud expectation, persistent-simulation sampler, and
zero-shock stochastic-steady-state diagnostic.

The shared `IRBCParams` and `irbc_smooth_residual` helpers are intentionally not
used here because they implement a compact teaching variant with different
state ordering and residual equations.
"""

# ╔═╡ ed503c03-65b7-e192-07e6-b26255ac2eda
md"""
## Lecture 04, Notebook 01: IRBC with DEQNs — Smooth Benchmark

**Course:** Deep Learning for Solving and Estimating Dynamic Models in Economics and Finance
**Script reference:** §3.2 (model setup), §3.3 (Euler equation and aggregate resource constraint), §3.4 (smooth DEQN loss, network architecture), §3.5 (persistent-simulation training; time-invariance and zero-shock stochastic-steady-state diagnostics)
**Notebook role:** core
**Author:** Simon Scheidegger

*Julia/Lux/Pluto preview of* `lectures/lecture_04_irbc_with_deqns/code/lecture_04_01_IRBC_DEQN_smooth.ipynb`.

> **Run mode.** The checked-in run uses `RUN_MODE = "smoke"` with `SEED = 0` for fast execution; the accuracy figures in the slides and the companion script use the longer `teaching` / `production` budgets defined in the next configuration cell. Set `RUN_MODE` there to reproduce them.
"""

# ╔═╡ 8c30b5d7-a332-cfcd-84fd-2fd1acfcdecd
md"""
This notebook solves the smooth \$N\$-country international real business-cycle model with complete markets, productivity risk, and convex capital-adjustment costs. It is deliberately written as a teaching notebook: one model, one training loop, and one switch between simulation-based and exogenous training states.

The central design choice is in the construction of the training data. In simulation mode the notebook does **not** restart from the steady state in every episode. Instead, it keeps a vector of current trajectory heads. Each training segment simulates these trajectory heads forward for `SIMULATION_LENGTH` stochastic periods, flattens the simulated states into a training cloud, performs the chosen stochastic-gradient updates, and then continues from the terminal states of the same trajectories.

The state is

\$\$
  s_t=(k_t^1,\ldots,k_t^N,z_t^1,\ldots,z_t^N),\qquad z_t^j=\log a_t^j.
\$\$

The policy network returns only

\$\$
  p(s_t)=\big(k_{t+1}^1,\ldots,k_{t+1}^N,\lambda_t\big),
\$\$

because this smooth benchmark has no irreversible-investment multipliers.

Two additional diagnostics are included. The first monitors whether the learned policy has stabilized across training iterations by measuring policy drift on a fixed holdout cloud. The second computes the zero-shock stochastic steady state: the fixed point of the learned stochastic policy when realized shocks are set to zero.
"""

# ╔═╡ 22222222-0401-4222-8222-222222222222
begin
    import Pkg
    Pkg.activate(joinpath(@__DIR__, "..", "..", "..", "julia"))

    using DLEFJulia
    using Lux
    using NNlib
    using Optimisers
    using Random
    using Statistics
end

# ╔═╡ 33333333-0401-4333-8333-333333333333
begin
    RUN_MODE = "smoke"
    SEED = 0
    SAMPLING_MODE = "simulation"

    budgets = (
        smoke = (
            num_segments = 25,
            n_trajectories = 4,
            simulation_length = 32,
            batch_size = 128,
            learning_rate = 3e-4,
            monitor_every = 5,
            time_anchor_states = 256,
            zero_shock_starts = 8,
            zero_shock_max_steps = 250,
        ),
        teaching = (
            num_segments = 301,
            n_trajectories = 10,
            simulation_length = 256,
            batch_size = 512,
            learning_rate = 2e-4,
            monitor_every = 10,
            time_anchor_states = 2_048,
            zero_shock_starts = 32,
            zero_shock_max_steps = 750,
        ),
        production = (
            num_segments = 1_201,
            n_trajectories = 32,
            simulation_length = 512,
            batch_size = 1_024,
            learning_rate = 1e-4,
            monitor_every = 25,
            time_anchor_states = 4_096,
            zero_shock_starts = 64,
            zero_shock_max_steps = 1_500,
        ),
    )

    hp = run_mode_budget(RUN_MODE; budgets)
    rng = rng_from_seed(SEED)

    PASSES_PER_SEGMENT = 1
    SHUFFLE_STATES_WITHIN_SEGMENT = true
    CLIP_NORM = 10.0
    EULER_WEIGHT = 1.0
    ARC_WEIGHT = 1.0

    TIME_INVARIANCE_TOL_RMS = 1.0e-3
    TIME_INVARIANCE_TOL_MAX = 1.0e-2
    RUN_ZERO_SHOCK_STEADY_STATE_CHECK = true
    ZERO_SHOCK_TOL = 1.0e-7
    ZERO_SHOCK_FIXED_POINT_TOL = 1.0e-4
    ZERO_SHOCK_Z_TOL = 1.0e-4
    ZERO_SHOCK_CROSS_TRACK_TOL = 5.0e-3
    SSS_MEAN_RESIDUAL_TOL = 1.0e-2
    SSS_K_MIN_OK = 0.20
    SSS_K_MAX_OK = 5.00
end

# ╔═╡ a87c6a14-24f4-4c4d-7bfa-6e6b71a9304d
md"""
## 1. Economic parameters

Productivity is represented by \$z_t^j=\log a_t^j\$ and follows

\$\$
  z_{t+1}^j=\rho z_t^j+\sigma(\varepsilon_{t+1}^j+\varepsilon_{t+1}^{agg}).
\$\$

The innovation variance of \$z^j\$ is therefore \$2\sigma^2\$, which is used below to create reasonable productivity boxes for exogenous sampling and for the initial trajectory heads.

The deterministic normalization sets the reference capital level to one. The simulated training data below does **not** start from that reference point; the reference level is used only to scale neural-network inputs and complete-markets weights.
"""

# ╔═╡ 44444444-0401-4444-8444-444444444444
begin
    N_COUNTRIES = 2

    beta = 0.99
    zeta = 0.36
    delta = 0.01
    rho_z = 0.95
    sigma_e = 0.01
    kappa = 0.50

    gamma_min = 0.25
    gamma_max = 1.00

    K_REF = 1.0
    LAMBDA_REF = 1.0

    A_tfp = (1.0 / beta - 1.0 + delta) / (zeta * K_REF^(zeta - 1.0))
    Y_ref = A_tfp * K_REF^zeta
    C_ref = Y_ref - delta * K_REF

    gammas_vec = N_COUNTRIES == 1 ? [gamma_min] : collect(range(gamma_min, gamma_max; length = N_COUNTRIES))
    taus_vec = LAMBDA_REF .* C_ref .^ (1.0 ./ gammas_vec)
    gammas = reshape(gammas_vec, N_COUNTRIES, 1)
    taus = reshape(taus_vec, N_COUNTRIES, 1)

    n_states = 2 * N_COUNTRIES
    n_policies = N_COUNTRIES + 1
    n_shocks = N_COUNTRIES + 1

    z_std = sigma_e * sqrt(2.0 / (1.0 - rho_z^2))
    z_bound = 3.0 * z_std

    EXOGENOUS_K_LOW = 0.55
    EXOGENOUS_K_HIGH = 1.80
    EXOGENOUS_Z_LOW = -z_bound
    EXOGENOUS_Z_HIGH = z_bound

    INITIAL_K_LOW = 0.65
    INITIAL_K_HIGH = 1.45
    INITIAL_Z_LOW = -1.5 * z_std
    INITIAL_Z_HIGH = 1.5 * z_std

    EMERGENCY_REPAIR_BAD_STATES = true
    SIM_REPAIR_K_MIN = 0.05
    SIM_REPAIR_K_MAX = 8.0
    SIM_REPAIR_ABS_Z_MAX = 8.0 * z_std

    EPS = 1e-8
end

# ╔═╡ 3b2789bd-5423-a183-0059-1e31b10f6ef8
md"""
## 2. Integration rule

The expectation in the Euler equations is evaluated with a monomial rule. It uses \$2(N+1)\$ nodes instead of a tensor-product Gauss--Hermite rule with \$Q^{N+1}\$ nodes, so it remains cheap when the number of countries increases.

The Julia cell builds this Stroud-3 rule with `make_monomial_rule` and reports `stroud_checks` (node count, weight sum, and the first and second moments) to confirm the weights sum to one and the rule reproduces the moments of a standard normal.
"""

# ╔═╡ 55555555-0401-4555-8555-555555555555
begin
    function make_monomial_rule(dim::Integer)
        dim > 0 || throw(ArgumentError("dim must be positive"))
        nodes = zeros(Float64, dim, 2dim)
        weights = fill(1.0 / (2.0 * dim), 2dim)
        radius = sqrt(float(dim))
        for i in 1:dim
            nodes[i, 2i - 1] = radius
            nodes[i, 2i] = -radius
        end
        return QuadratureRule(nodes, weights)
    end

    quad_rule = make_monomial_rule(n_shocks)
    stroud_checks = (
        n_nodes = length(quad_rule.weights),
        weight_sum = sum(quad_rule.weights),
        first_moment = quadrature_expectation(x -> x[1], quad_rule),
        second_moment = quadrature_expectation(x -> x[1]^2, quad_rule),
        idio_aggregate_cross = quadrature_expectation(x -> x[1] * x[end], quad_rule),
    )
end

# ╔═╡ 28d74d18-1383-e9ce-60c7-ed4d4307defb
md"""
## 3. Neural network and policy transformation

The network is initialized so that \$k_{t+1}=k_t\$ and \$\lambda_t=1\$ before training. This is not a training-data assumption; it is only a stable initial policy. Next-period capital is parameterized relative to current capital:

\$\$
   k_{t+1}^j=k_t^j\exp\{\bar g\tanh r_j(s_t)\}.
\$\$

Thus \$k_{t+1}^j\$ remains positive and per-period capital growth is bounded during simulation.

In Lux, `scale_states` normalizes the log-capital and productivity inputs, `zero_output_head` zeros the final layer so the initial policy is exactly \$k_{t+1}=k_t\$, \$\lambda_t=1\$, and `smooth_policy` applies the \$\exp\{\bar g\tanh(\cdot)\}\$ capital map (with \$\bar g=\$ `KP_GROWTH_SCALE`) and the \$\lambda\$ transform.
"""

# ╔═╡ 66666666-0401-4666-8666-666666666666
begin
    INPUT_K_LOG_SCALE = 0.50
    INPUT_Z_SCALE = max(z_bound, 1e-6)
    KP_GROWTH_SCALE = 0.30
    LAMBDA_LOG_SCALE = 1.25
    NUM_HIDDEN_1 = 128
    NUM_HIDDEN_2 = 128

    function python_ordered_blocks(states)
        x = assert_feature_batch(states, n_states)
        k = x[1:N_COUNTRIES, :]
        z = x[(N_COUNTRIES + 1):n_states, :]
        return k, z
    end

    function scale_states(states)
        k, z = python_ordered_blocks(states)
        k_scaled = log.(max.(k, EPS) ./ K_REF) ./ INPUT_K_LOG_SCALE
        z_scaled = z ./ INPUT_Z_SCALE
        return vcat(k_scaled, z_scaled)
    end

    function zero_output_head(ps)
        final = ps.layer_3
        final_zero = (weight = zero.(final.weight), bias = zero.(final.bias))
        return merge(ps, (layer_3 = final_zero,))
    end

    function smooth_policy(model, ps, st, states)
        k, _ = python_ordered_blocks(states)
        raw, st_new = model(scale_states(states), ps, st)
        raw_k = raw[1:N_COUNTRIES, :]
        raw_lambda = raw[(N_COUNTRIES + 1):(N_COUNTRIES + 1), :]
        kp = max.(k, EPS) .* exp.(KP_GROWTH_SCALE .* tanh.(raw_k))
        lamb = LAMBDA_REF .* exp.(LAMBDA_LOG_SCALE .* tanh.(raw_lambda))
        return (next_capital = kp, lambda = lamb, raw = raw), st_new
    end

    function reference_states(n::Integer)
        return vcat(ones(Float64, N_COUNTRIES, n), zeros(Float64, N_COUNTRIES, n))
    end
end

# ╔═╡ 7a3dbfb7-971d-3dd1-d03b-c3155b97bf71
md"""
## 4. Residuals and loss

The Euler residual is a relative wedge:

\$\$
   \text{EulerErr}_{j,t}
   =\frac{\beta E_t\left[\lambda_{t+1}\left(MPK_{t+1}^j+1-\delta+
   \frac{\kappa}{2}g_{t+2}^j(g_{t+2}^j+2)\right)\right]}
   {\lambda_t\left(1+\kappa g_{t+1}^j\right)}-1.
\$\$

A value of \$10^{-2}\$ means a one-percent Euler-equation wedge. The aggregate-resource residual is also relative: it is the resource surplus divided by contemporaneous aggregate resources.

`smooth_residual` assembles this Euler wedge, evaluates the conditional expectation with the monomial `quad_rule`, adds the relative aggregate-resource-constraint (ARC) residual, and returns the training loss `EULER_WEIGHT · mean(euler²) + ARC_WEIGHT · mean(arc²)`.
"""

# ╔═╡ 77777777-0401-4777-8777-777777777777
begin
    production(k, z) = A_tfp .* exp.(z) .* max.(k, EPS) .^ zeta
    production_k(k, z) = zeta * A_tfp .* exp.(z) .* max.(k, EPS) .^ (zeta - 1.0)

    function adjustment_cost(k, kp)
        k_safe = max.(k, EPS)
        growth = kp ./ k_safe .- 1.0
        return 0.5 * kappa .* k_safe .* growth .^ 2
    end

    function adjustment_cost_kp(k, kp)
        k_safe = max.(k, EPS)
        return kappa .* (kp ./ k_safe .- 1.0)
    end

    function consumption_from_lambda(lamb)
        return (max.(lamb, EPS) ./ taus) .^ (-gammas)
    end

    function next_state_from_shock(states, kp, shock)
        _, z = python_ordered_blocks(states)
        eps_idio = reshape(shock[1:N_COUNTRIES], N_COUNTRIES, 1)
        eps_agg = shock[N_COUNTRIES + 1]
        z_next = rho_z .* z .+ sigma_e .* (eps_idio .+ eps_agg)
        return vcat(kp, z_next)
    end

    function smooth_residual(model, ps, st, states; rule = quad_rule)
        k, z = python_ordered_blocks(states)
        current_policy, st_new = smooth_policy(model, ps, st, states)
        kp = current_policy.next_capital
        lamb = current_policy.lambda

        lhs = lamb .* (1.0 .+ adjustment_cost_kp(k, kp))
        expectation = zero(kp)

        for q in eachindex(rule.weights)
            state_next = next_state_from_shock(states, kp, @view rule.nodes[:, q])
            next_policy, _ = smooth_policy(model, ps, st_new, state_next)
            k_next, z_next = python_ordered_blocks(state_next)
            kp_next = next_policy.next_capital
            lambda_next = next_policy.lambda
            g_next = kp_next ./ max.(k_next, EPS) .- 1.0
            return_next = production_k(k_next, z_next) .+ 1.0 .- delta .+
                0.5 .* kappa .* g_next .* (g_next .+ 2.0)
            expectation = expectation .+ rule.weights[q] .* lambda_next .* return_next
        end

        rhs = beta .* expectation
        euler_res = rhs ./ max.(lhs, EPS) .- 1.0

        y = production(k, z)
        c = consumption_from_lambda(lamb)
        gamma_cost = adjustment_cost(k, kp)
        arc_level = sum(y .+ (1.0 - delta) .* k .- kp .- gamma_cost .- c; dims = 1)
        arc_scale = sum(y .+ (1.0 - delta) .* k; dims = 1)
        arc_res = arc_level ./ max.(arc_scale, EPS)

        loss_euler = mean(abs2, euler_res)
        loss_arc = mean(abs2, arc_res)
        loss = EULER_WEIGHT * loss_euler + ARC_WEIGHT * loss_arc

        return (
            loss = loss,
            loss_euler = loss_euler,
            loss_arc = loss_arc,
            euler = euler_res,
            arc = arc_res,
            next_capital = kp,
            lambda = lamb,
            consumption = c,
            adjustment_cost = gamma_cost,
            resource_level = arc_level,
        ), st_new
    end
end

# ╔═╡ d22e2d4f-2b5d-8951-b051-429920337fef
md"""
## 5. Training-data construction

This is the central cell. In simulation mode, the training data for one segment is created as follows.

1. Start from the current trajectory heads `X_start`, with shape `(N_TRAJECTORIES, n_states)`.
2. For `SIMULATION_LENGTH` periods, record the current states and then simulate one stochastic transition under the current policy network.
3. Flatten the recorded path into a matrix with shape `(N_TRAJECTORIES * SIMULATION_LENGTH, n_states)`.
4. Return the terminal states `X_end`. After the optimizer has updated the policy on the simulated segment, these terminal states become the next `X_start`.

Thus the simulated trajectories continue across training segments. They do not repeatedly restart from the steady state.

In Julia this is `get_training_segment` / `simulate_path`; `SAMPLING_MODE = "simulation"` continues the trajectories, while `SAMPLING_MODE = "exogenous"` draws a fresh state box each segment. (At the Lux boundary states are stored feature-by-batch, so the shapes above are transposed to `(n_states, N_TRAJECTORIES * SIMULATION_LENGTH)`.)
"""

# ╔═╡ 88888888-0401-4888-8888-888888888888
begin
    function sample_feasible_initial_states(local_rng, n_tracks::Integer)
        log_k = log(INITIAL_K_LOW) .+
            (log(INITIAL_K_HIGH) - log(INITIAL_K_LOW)) .* rand(local_rng, N_COUNTRIES, n_tracks)
        k0 = exp.(log_k)
        z0 = INITIAL_Z_LOW .+ (INITIAL_Z_HIGH - INITIAL_Z_LOW) .* rand(local_rng, N_COUNTRIES, n_tracks)
        return vcat(k0, z0)
    end

    function get_training_data_exogenous(local_rng, n_data::Integer)
        k = EXOGENOUS_K_LOW .+ (EXOGENOUS_K_HIGH - EXOGENOUS_K_LOW) .* rand(local_rng, N_COUNTRIES, n_data)
        z = EXOGENOUS_Z_LOW .+ (EXOGENOUS_Z_HIGH - EXOGENOUS_Z_LOW) .* rand(local_rng, N_COUNTRIES, n_data)
        return vcat(k, z)
    end

    function simulate_single_step(local_rng, states, model, ps, st)
        current_policy, st_new = smooth_policy(model, ps, st, states)
        _, z = python_ordered_blocks(states)
        batch = size(states, 2)
        shocks = randn(local_rng, n_shocks, batch)
        eps_idio = shocks[1:N_COUNTRIES, :]
        eps_agg = shocks[(N_COUNTRIES + 1):(N_COUNTRIES + 1), :]
        z_next = rho_z .* z .+ sigma_e .* (eps_idio .+ eps_agg)
        return vcat(current_policy.next_capital, z_next), st_new
    end

    function bad_state_mask(states)
        k, z = python_ordered_blocks(states)
        mask = falses(size(states, 2))
        for col in axes(states, 2)
            mask[col] = !all(isfinite, @view states[:, col]) ||
                any((@view k[:, col]) .< SIM_REPAIR_K_MIN) ||
                any((@view k[:, col]) .> SIM_REPAIR_K_MAX) ||
                any(abs.(@view z[:, col]) .> SIM_REPAIR_ABS_Z_MAX)
        end
        return mask
    end

    function repair_bad_states(local_rng, states)
        EMERGENCY_REPAIR_BAD_STATES || return states
        mask = bad_state_mask(states)
        any(mask) || return states
        repaired = copy(states)
        repaired[:, mask] .= sample_feasible_initial_states(local_rng, count(mask))
        return repaired
    end

    function simulate_path(local_rng, x_start, model, ps, st, n_steps::Integer)
        current = Matrix{Float64}(x_start)
        n_tracks = size(current, 2)
        path = Matrix{Float64}(undef, n_states, n_tracks * n_steps)
        st_current = st

        for t in 1:n_steps
            cols = ((t - 1) * n_tracks + 1):(t * n_tracks)
            path[:, cols] .= current
            current, st_current = simulate_single_step(local_rng, current, model, ps, st_current)
            current = repair_bad_states(local_rng, current)
        end

        return path, current, st_current
    end

    function get_training_segment(local_rng, x_start, train_state)
        if SAMPLING_MODE == "simulation"
            x_segment, x_end, _ = simulate_path(
                local_rng,
                x_start,
                train_state.model,
                train_state.ps,
                train_state.st,
                hp.simulation_length,
            )
            return x_segment, x_end
        elseif SAMPLING_MODE == "exogenous"
            x_segment = get_training_data_exogenous(local_rng, hp.n_trajectories * hp.simulation_length)
            return x_segment, x_start
        end
        throw(ArgumentError("SAMPLING_MODE must be simulation or exogenous"))
    end
end

# ╔═╡ 44d016a8-91f2-24ed-3833-363f09d0a5d5
md"""
## 6. Mini-batches and training loop

A training segment may contain many states, for example \$10\times 256=2560\$ states. The variables `batch_size`, `PASSES_PER_SEGMENT`, and the optimizer (`Optimisers.Adam`) determine how the optimizer uses these states.

The continuation logic is intentionally explicit in the loop (`run_persistent_training!`):
```julia
x_segment, x_end = get_training_segment(local_rng, x_start, train_state)
train_on_segment_states!(train_state, local_rng, x_segment)   # optimizer updates
x_start = x_end                                               # continue the trajectories
```
The model itself is instantiated with `make_mlp` and `setup_training`; the demo cell simulates a short path to show the segment shapes before the full run.
"""

# ╔═╡ 99999999-0401-4999-8999-999999999999
begin
    function minibatch_columns(local_rng, n::Integer, batch_size::Integer)
        idx = collect(1:n)
        SHUFFLE_STATES_WITHIN_SEGMENT && Random.shuffle!(local_rng, idx)
        return (idx[start:min(start + batch_size - 1, n)] for start in 1:batch_size:n)
    end

    smooth_loss(model, ps, st, states) = begin
        pieces, st_new = smooth_residual(model, ps, st, states)
        return pieces.loss, st_new
    end

    function train_on_segment_states!(train_state, local_rng, x_segment)
        losses = Float64[]
        grad_norms = Float64[]
        n_updates = 0
        for _ in 1:PASSES_PER_SEGMENT
            for cols in minibatch_columns(local_rng, size(x_segment, 2), hp.batch_size)
                metrics = train_step!(train_state, smooth_loss, x_segment[:, cols]; max_grad_norm = CLIP_NORM)
                push!(losses, Float64(metrics.loss))
                push!(grad_norms, Float64(metrics.grad_norm))
                n_updates += 1
            end
        end
        return (loss = mean(losses), grad_norm = mean(grad_norms), n_updates = n_updates)
    end

    function sample_time_invariance_anchor_states(n_anchor::Integer)
        local_rng = rng_from_seed(SEED; offset = 91_017)
        n_exog = ceil(Int, 0.75 * n_anchor)
        n_init = n_anchor - n_exog

        log_k_exog = log(EXOGENOUS_K_LOW) .+
            (log(EXOGENOUS_K_HIGH) - log(EXOGENOUS_K_LOW)) .* rand(local_rng, N_COUNTRIES, n_exog)
        z_exog = EXOGENOUS_Z_LOW .+ (EXOGENOUS_Z_HIGH - EXOGENOUS_Z_LOW) .* rand(local_rng, N_COUNTRIES, n_exog)
        x_exog = vcat(exp.(log_k_exog), z_exog)

        if n_init > 0
            x_init = sample_feasible_initial_states(local_rng, n_init)
            x = hcat(x_exog, x_init)
        else
            x = x_exog
        end
        perm = randperm(local_rng, size(x, 2))
        return x[:, perm]
    end

    function policy_fingerprint(model, ps, st, states)
        pol, st_new = smooth_policy(model, ps, st, states)
        log_kp = log.(max.(pol.next_capital, EPS) ./ K_REF)
        log_lamb = log.(max.(pol.lambda, EPS) ./ LAMBDA_REF)
        return vcat(log_kp, log_lamb), st_new
    end

    function relative_policy_drift(previous, current)
        diff = current .- previous
        scale = 1.0 + sqrt(mean(previous .^ 2))
        return (
            rms = sqrt(mean(diff .^ 2)) / scale,
            max_abs = maximum(abs.(diff)) / scale,
        )
    end
end

# ╔═╡ aaaaaaaa-0401-4aaa-8aaa-aaaaaaaaaaaa
begin
    model = make_mlp(n_states, (NUM_HIDDEN_1, NUM_HIDDEN_2), n_policies; activation = NNlib.tanh)
    ps0, st0 = setup_model(rng_from_seed(SEED; offset = 1), model; parameter_type = Float64)
    ps0 = zero_output_head(ps0)
    state = setup_training(model, ps0, st0, Optimisers.Adam(hp.learning_rate))

    x_check = reference_states(3)
    policy_check, _ = smooth_policy(state.model, state.ps, state.st, x_check)
    initial_loss = loss_value(state, smooth_loss, x_check)
    initial_policy_check = (
        max_abs_kp_minus_k = maximum(abs.(policy_check.next_capital .- x_check[1:N_COUNTRIES, :])),
        max_abs_lambda_minus_one = maximum(abs.(policy_check.lambda .- 1.0)),
    )
end

# ╔═╡ bbbbbbbb-0401-4bbb-8bbb-bbbbbbbbbbbb
begin
    x_start = sample_feasible_initial_states(rng, hp.n_trajectories)
    x_demo, x_end_demo, _ = simulate_path(
        rng_from_seed(SEED; offset = 2),
        x_start,
        state.model,
        state.ps,
        state.st,
        min(hp.simulation_length, 8),
    )
    demo_shapes = (
        x_start = size(x_start),
        training = size(x_demo),
        x_end = size(x_end_demo),
    )
end

# ╔═╡ cccccccc-0401-4ccc-8ccc-cccccccccccc
begin
    function run_persistent_training!(train_state, local_rng, x_start_initial, x_anchor)
        anchor_fingerprint_previous, _ = policy_fingerprint(train_state.model, train_state.ps, train_state.st, x_anchor)
        local_history = NamedTuple[]
        x_start_local = copy(x_start_initial)
        x_segment_last_local = x_check

        for seg in 0:(hp.num_segments - 1)
            x_segment, x_end = get_training_segment(local_rng, x_start_local, train_state)
            train_metrics = train_on_segment_states!(train_state, local_rng, x_segment)
            SAMPLING_MODE == "simulation" && (x_start_local = x_end)
            x_segment_last_local = x_segment

            if seg % hp.monitor_every == 0 || seg == hp.num_segments - 1
                pieces, _ = smooth_residual(train_state.model, train_state.ps, train_state.st, x_segment)
                k_segment, z_segment = python_ordered_blocks(x_segment)
                anchor_fingerprint_now, _ = policy_fingerprint(train_state.model, train_state.ps, train_state.st, x_anchor)
                drift = relative_policy_drift(anchor_fingerprint_previous, anchor_fingerprint_now)
                anchor_fingerprint_previous = anchor_fingerprint_now

                append_metric!(
                    local_history;
                    segment = seg,
                    loss = Float64(pieces.loss),
                    mean_abs_euler = mean(abs, pieces.euler),
                    mean_abs_arc = mean(abs, pieces.arc),
                    policy_drift_rms = drift.rms,
                    policy_drift_max = drift.max_abs,
                    k_min = minimum(k_segment),
                    k_max = maximum(k_segment),
                    z_min = minimum(z_segment),
                    z_max = maximum(z_segment),
                    n_updates = train_metrics.n_updates,
                    mean_train_loss = train_metrics.loss,
                    grad_norm = train_metrics.grad_norm,
                )
            end
        end

        return (
            history = local_history,
            x_start = x_start_local,
            x_segment_last = x_segment_last_local,
            anchor_fingerprint = anchor_fingerprint_previous,
        )
    end

    x_anchor = sample_time_invariance_anchor_states(hp.time_anchor_states)
    training_result = run_persistent_training!(state, rng, x_start, x_anchor)
    history = training_result.history
    x_start = training_result.x_start
    x_segment_last = training_result.x_segment_last
    anchor_fingerprint_previous = training_result.anchor_fingerprint
end

# ╔═╡ 3701d15d-5270-56f3-4564-f4d996c7e9f3
md"""
## 7. Final diagnostics

The final diagnostics report dimensionless, interpretable residuals on two state clouds:

- an exogenous test cloud, useful for checking off-trajectory robustness;
- a simulated test cloud, useful for checking accuracy on the ergodic region induced by the learned policy.

The mean absolute Euler error is a relative Euler wedge. For example, `mean = 2e-3` means an average wedge of roughly 0.2 percent. The ARC error is a relative resource violation.

The next section adds two convergence checks: policy drift on a fixed holdout cloud and the zero-shock stochastic steady state.
"""

# ╔═╡ dddddddd-0401-4ddd-8ddd-dddddddddddd
begin
    function summarize_abs(values)
        x = vec(abs.(Float64.(values)))
        return (
            mean = mean(x),
            median = median(x),
            p95 = quantile(x, 0.95),
            p99 = quantile(x, 0.99),
            max = maximum(x),
        )
    end

    function simulated_evaluation_states(model, ps, st; n_eval_tracks = 64, burn_in = 64, eval_length = 256)
        local_rng = rng_from_seed(SEED; offset = 24_681)
        x0 = sample_feasible_initial_states(local_rng, n_eval_tracks)
        path, _, _ = simulate_path(local_rng, x0, model, ps, st, burn_in + eval_length)
        n_tracks = size(x0, 2)
        keep_cols = (burn_in * n_tracks + 1):((burn_in + eval_length) * n_tracks)
        return path[:, keep_cols]
    end

    function residual_report(x, model, ps, st)
        pieces, _ = smooth_residual(model, ps, st, x)
        k, _ = python_ordered_blocks(x)
        return (
            loss = Float64(pieces.loss),
            euler = summarize_abs(pieces.euler),
            arc = summarize_abs(pieces.arc),
            capital_state_min = minimum(k),
            capital_state_max = maximum(k),
            capital_policy_min = minimum(pieces.next_capital),
            capital_policy_max = maximum(pieces.next_capital),
        )
    end

    n_eval_exogenous = RUN_MODE == "smoke" ? 1_024 : 8_192
    n_eval_tracks = RUN_MODE == "smoke" ? 8 : 64
    eval_burn_in = RUN_MODE == "smoke" ? 8 : 64
    eval_length = RUN_MODE == "smoke" ? 32 : 256

    x_eval_exog = get_training_data_exogenous(rng_from_seed(SEED; offset = 31_415), n_eval_exogenous)
    x_eval_sim = simulated_evaluation_states(
        state.model,
        state.ps,
        state.st;
        n_eval_tracks,
        burn_in = eval_burn_in,
        eval_length,
    )

    report_exogenous = residual_report(x_eval_exog, state.model, state.ps, state.st)
    report_simulated = residual_report(x_eval_sim, state.model, state.ps, state.st)
end

# ╔═╡ b4956cd7-fd5e-cc2c-cc55-a0d7cb1544bd
md"""
## 8. Time-invariance and zero-shock stochastic steady state

There are two different notions that are useful to keep separate.

First, the policy network has no calendar-time input, so a fixed set of weights always defines a time-homogeneous recursive policy. The nontrivial numerical question is whether the policy function has stopped changing as training proceeds. The notebook checks this by evaluating the policy on a fixed holdout cloud `X_anchor` and reporting the monitor-to-monitor policy drift.

Second, the stochastic steady state reported below is the fixed point of the learned stochastic policy when realized shocks are set to zero. It is not imposed during training. Starting from several dispersed feasible states, the notebook simulates the learned policy with zero shocks and checks whether all paths converge to a common economically meaningful point.
"""

# ╔═╡ eeeeeeee-0401-4eee-8eee-eeeeeeeeeeee
begin
    function sample_zero_shock_start_states(n_tracks::Integer)
        local_rng = rng_from_seed(SEED; offset = 51_031)
        return sample_feasible_initial_states(local_rng, n_tracks)
    end

    function simulate_single_step_zero_shock(states, model, ps, st)
        pol, st_new = smooth_policy(model, ps, st, states)
        _, z = python_ordered_blocks(states)
        z_next = rho_z .* z
        return vcat(pol.next_capital, z_next), st_new
    end

    function scaled_transition_distance(x, x_next)
        k, z = python_ordered_blocks(x)
        kp, z_next = python_ordered_blocks(x_next)
        d_k = maximum(abs.(log.(max.(kp, EPS) ./ max.(k, EPS))))
        d_z = maximum(abs.(z_next .- z)) / max(INPUT_Z_SCALE, EPS)
        return max(d_k, d_z)
    end

    function compute_zero_shock_stochastic_steady_state(model, ps, st)
        x = sample_zero_shock_start_states(hp.zero_shock_starts)
        distances = Float64[]
        converged = false
        steps = 0
        st_current = st

        for step in 1:hp.zero_shock_max_steps
            x_next, st_current = simulate_single_step_zero_shock(x, model, ps, st_current)
            all(isfinite, x_next) || break
            dist = scaled_transition_distance(x, x_next)
            push!(distances, dist)
            x = x_next
            steps = step
            if dist <= ZERO_SHOCK_TOL
                converged = true
                break
            end
        end

        return (states = x, distances = distances, steps = steps, converged = converged)
    end

    function zero_shock_stochastic_steady_state_report(model, ps, st)
        RUN_ZERO_SHOCK_STEADY_STATE_CHECK || return nothing
        result = compute_zero_shock_stochastic_steady_state(model, ps, st)
        pieces, _ = smooth_residual(model, ps, st, result.states)
        k, z = python_ordered_blocks(result.states)
        capital_fixed_error = maximum(abs.(pieces.next_capital .- k) ./ max.(k, EPS))
        max_abs_z = maximum(abs.(z))
        log_k_spread = maximum(std(log.(max.(k, EPS)); dims = 2))
        mean_k = vec(mean(k; dims = 2))
        e_abs = abs.(pieces.euler)
        a_abs = abs.(pieces.arc)

        checks = (
            finite_state_and_policy = all(isfinite, result.states) &&
                all(isfinite, pieces.next_capital) &&
                all(isfinite, pieces.consumption),
            positive_capital = minimum(k) > 0.0,
            zero_shock_iteration_converged = result.converged,
            capital_fixed_point = capital_fixed_error <= ZERO_SHOCK_FIXED_POINT_TOL,
            productivity_fixed_point = max_abs_z <= ZERO_SHOCK_Z_TOL,
            starts_converge_to_common_point = log_k_spread <= ZERO_SHOCK_CROSS_TRACK_TOL,
            positive_consumption = minimum(pieces.consumption) > 0.0,
            capital_in_broad_range = minimum(k) >= SSS_K_MIN_OK * K_REF &&
                maximum(k) <= SSS_K_MAX_OK * K_REF,
            euler_residual_small = mean(e_abs) <= SSS_MEAN_RESIDUAL_TOL,
            resource_residual_small = mean(a_abs) <= SSS_MEAN_RESIDUAL_TOL,
        )

        return (
            states = result.states,
            distances = result.distances,
            steps = result.steps,
            converged = result.converged,
            mean_k = mean_k,
            relative_to_deterministic_k = mean_k ./ K_REF .- 1.0,
            capital_fixed_error = capital_fixed_error,
            max_abs_z = max_abs_z,
            log_k_spread = log_k_spread,
            mean_abs_euler = mean(e_abs),
            mean_abs_arc = mean(a_abs),
            checks = checks,
        )
    end

    final_fingerprint_1, _ = policy_fingerprint(state.model, state.ps, state.st, x_anchor)
    final_fingerprint_2, _ = policy_fingerprint(state.model, state.ps, state.st, x_anchor)
    same_state_repeatability_max = maximum(abs.(final_fingerprint_2 .- final_fingerprint_1))
    zero_shock_result = zero_shock_stochastic_steady_state_report(state.model, state.ps, state.st)
end

# ╔═╡ 0dd27b86-57bd-f688-5d89-1c1edb9bc329
md"""
## 9. Simple plots

These plots are intentionally minimal. They show whether training is moving in the right direction and whether the simulated state cloud is well behaved.

This Julia preview assembles the plotting arrays in `plot_data` — the training-loss curve on a \$\log_{10}\$ scale, the mean absolute Euler and ARC residuals over training, and a current-versus-next capital scatter for country 1 — rather than rendering figures inline (the smooth notebook does not import a plotting backend).
"""

# ╔═╡ ffffffff-0401-4fff-8fff-ffffffffffff
begin
    hist_segments = [row.segment for row in history]
    hist_loss = [row.loss for row in history]
    hist_euler = [row.mean_abs_euler for row in history]
    hist_arc = [row.mean_abs_arc for row in history]

    eval_policy, _ = smooth_policy(state.model, state.ps, state.st, x_eval_sim)
    k_eval, _ = python_ordered_blocks(x_eval_sim)

    plot_data = (
        training_segments = hist_segments,
        log10_loss = log10.(max.(hist_loss, 1e-30)),
        mean_abs_euler = hist_euler,
        mean_abs_arc = hist_arc,
        capital_country1 = vec(k_eval[1, :]),
        next_capital_country1 = vec(eval_policy.next_capital[1, :]),
    )
end

# ╔═╡ 2cfbc2a2-2436-2710-df54-f12fb9caeb45
md"""
### How to change the training data (and a closing summary)

To experiment with the sampler, edit the active `budgets` entry and the switches in the configuration cell:

- **One long trajectory:** set `n_trajectories = 1`, `simulation_length = 1024`.
- **Several shorter trajectories:** set `n_trajectories = 10`, `simulation_length = 256`.
- **Exogenous sampling:** set `SAMPLING_MODE = "exogenous"` to draw a fresh state box each segment instead of continuing simulated trajectories.

The optimizer and update schedule are set by `learning_rate` (feeding `Optimisers.Adam`), `batch_size`, and `PASSES_PER_SEGMENT`. The policy-stability check is controlled by `time_anchor_states`, `TIME_INVARIANCE_TOL_RMS`, and `TIME_INVARIANCE_TOL_MAX`; the zero-shock stochastic steady-state check by `RUN_ZERO_SHOCK_STEADY_STATE_CHECK`, `zero_shock_starts`, `zero_shock_max_steps`, and `ZERO_SHOCK_TOL`.

### Conclusion

This Lux/Pluto preview reproduces the smooth IRBC DEQN pipeline end to end: the complete-markets \$\lambda\$-policy, the adjustment-cost Euler wedge and relative ARC residual, the \$2(N+1)\$-node Stroud monomial expectation, the persistent-simulation sampler that continues trajectories across training segments, and the time-invariance and zero-shock stochastic-steady-state diagnostics. The cell below returns a machine-checkable summary of this notebook's smoke run.
"""

# ╔═╡ abababab-0401-4aba-8aba-abababababab
(
    run_mode = RUN_MODE,
    sampling_mode = SAMPLING_MODE,
    state_order = "[k_1,...,k_N,z_1,...,z_N]",
    policy_order = "[k'_1,...,k'_N,lambda]",
    deterministic_reference_capital = K_REF,
    A_tfp = A_tfp,
    C_ref = C_ref,
    gammas = gammas_vec,
    taus = taus_vec,
    stroud = stroud_checks,
    initial_policy_check = initial_policy_check,
    initial_loss_at_reference_states = initial_loss,
    demo_shapes = demo_shapes,
    final_loss = history[end].loss,
    final_mean_abs_euler = history[end].mean_abs_euler,
    final_mean_abs_arc = history[end].mean_abs_arc,
    final_policy_drift = (
        rms = history[end].policy_drift_rms,
        max = history[end].policy_drift_max,
        same_state_repeatability_max = same_state_repeatability_max,
    ),
    exogenous_report = report_exogenous,
    simulated_report = report_simulated,
    zero_shock = isnothing(zero_shock_result) ? nothing : (
        converged = zero_shock_result.converged,
        steps = zero_shock_result.steps,
        final_distance = isempty(zero_shock_result.distances) ? NaN : zero_shock_result.distances[end],
        mean_k = zero_shock_result.mean_k,
        capital_fixed_error = zero_shock_result.capital_fixed_error,
        max_abs_z = zero_shock_result.max_abs_z,
        mean_abs_euler = zero_shock_result.mean_abs_euler,
        mean_abs_arc = zero_shock_result.mean_abs_arc,
        checks = zero_shock_result.checks,
    ),
)

# ╔═╡ Cell order:
# ╟─11111111-0401-4111-8111-111111111111
# ╟─ed503c03-65b7-e192-07e6-b26255ac2eda
# ╟─8c30b5d7-a332-cfcd-84fd-2fd1acfcdecd
# ╠═22222222-0401-4222-8222-222222222222
# ╠═33333333-0401-4333-8333-333333333333
# ╟─a87c6a14-24f4-4c4d-7bfa-6e6b71a9304d
# ╠═44444444-0401-4444-8444-444444444444
# ╟─3b2789bd-5423-a183-0059-1e31b10f6ef8
# ╠═55555555-0401-4555-8555-555555555555
# ╟─28d74d18-1383-e9ce-60c7-ed4d4307defb
# ╠═66666666-0401-4666-8666-666666666666
# ╟─7a3dbfb7-971d-3dd1-d03b-c3155b97bf71
# ╠═77777777-0401-4777-8777-777777777777
# ╟─d22e2d4f-2b5d-8951-b051-429920337fef
# ╠═88888888-0401-4888-8888-888888888888
# ╟─44d016a8-91f2-24ed-3833-363f09d0a5d5
# ╠═99999999-0401-4999-8999-999999999999
# ╠═aaaaaaaa-0401-4aaa-8aaa-aaaaaaaaaaaa
# ╠═bbbbbbbb-0401-4bbb-8bbb-bbbbbbbbbbbb
# ╠═cccccccc-0401-4ccc-8ccc-cccccccccccc
# ╟─3701d15d-5270-56f3-4564-f4d996c7e9f3
# ╠═dddddddd-0401-4ddd-8ddd-dddddddddddd
# ╟─b4956cd7-fd5e-cc2c-cc55-a0d7cb1544bd
# ╠═eeeeeeee-0401-4eee-8eee-eeeeeeeeeeee
# ╟─0dd27b86-57bd-f688-5d89-1c1edb9bc329
# ╠═ffffffff-0401-4fff-8fff-ffffffffffff
# ╟─2cfbc2a2-2436-2710-df54-f12fb9caeb45
# ╠═abababab-0401-4aba-8aba-abababababab
