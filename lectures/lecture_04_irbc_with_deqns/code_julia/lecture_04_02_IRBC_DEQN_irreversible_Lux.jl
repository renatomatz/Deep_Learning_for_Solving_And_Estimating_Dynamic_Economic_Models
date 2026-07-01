### A Pluto.jl notebook ###
# v0.20.0

using Markdown
using InteractiveUtils

# ╔═╡ 11111111-0402-4111-8111-111111111111
md"""
# Lecture 04, Notebook 02: Irreversible-Investment IRBC in Lux

This Pluto/Lux translation follows the Python irreversible-investment notebook
directly.  The state is ordered as capital then productivity,
`[k_1, ..., k_N, z_1, ..., z_N]`, and the policy outputs are raw investment
controls, one world marginal-utility multiplier, and country KKT multipliers.
"""

# ╔═╡ 9c9b32da-90e5-69c0-465f-9591a1fda4e6
md"""
## Lecture 04, Notebook 02: IRBC with DEQNs — Irreversible Investment

**Course:** Deep Learning for Solving and Estimating Dynamic Models in Economics and Finance
**Script reference:** §3.3 (Fischer–Burmeister complementarity), §3.4 (irreversible DEQN loss), §3.5 (persistent-simulation training; time-invariance and zero-shock stochastic-steady-state diagnostics)
**Notebook role:** core
**Author:** Simon Scheidegger

*Julia/Lux/Pluto preview of* `lectures/lecture_04_irbc_with_deqns/code/lecture_04_02_IRBC_DEQN_irreversible.ipynb`.

> **Run mode.** The checked-in run uses `RUN_MODE = "smoke"` with `SEED = 0`; the longer `teaching` / `production` budgets in the next configuration cell reproduce the accuracy figures in the slides and companion script.
"""

# ╔═╡ ece79b31-4cf9-d51e-39db-ef32ca83d832
md"""
This notebook solves the \$N\$-country international real business-cycle model with complete markets, productivity risk, convex adjustment costs, and irreversible investment. The recursive policy is

\$\$
  p(s_t)=\big(k_{t+1}^1,\ldots,k_{t+1}^N,\mu_t^1,\ldots,\mu_t^N,\lambda_t\big),
\$\$

with the complementarity condition

\$\$
  0\leq \mu_t^j\perp I_t^j=k_{t+1}^j-(1-\delta)k_t^j\geq 0.
\$\$

The notebook is intentionally organized around a single training-data switch. In the default simulation mode it keeps a set of continuing stochastic trajectories. It simulates a segment, trains on that segment, then continues from the segment's terminal states. It does not restart from the steady state in every episode.

Two additional diagnostics are included. The first monitors whether the learned policy has stabilized across training iterations by measuring policy drift on a fixed holdout cloud. The second computes the zero-shock stochastic steady state and checks that, at that point, investment replaces depreciation and the irreversibility multiplier is close to zero.
"""

# ╔═╡ 22222222-0402-4222-8222-222222222222
begin
    import Pkg
    Pkg.activate(joinpath(@__DIR__, "..", "..", "..", "julia"))

    using CairoMakie
    using DLEFJulia
    using Lux
    using NNlib
    using Optimisers
    using Random
    using Statistics
end

# ╔═╡ 33333333-0402-4333-8333-333333333333
begin
    RUN_MODE = "smoke"
    SEED = 0
    budgets = (
        smoke = (
            num_segments = 25,
            n_trajectories = 4,
            simulation_length = 32,
            batch_size = 128,
            learning_rate = 3e-4,
            monitor_every = 5,
            time_invariance_anchor_states = 256,
            zero_shock_n_starts = 8,
            zero_shock_max_steps = 250,
        ),
        teaching = (
            num_segments = 401,
            n_trajectories = 10,
            simulation_length = 256,
            batch_size = 512,
            learning_rate = 2e-4,
            monitor_every = 10,
            time_invariance_anchor_states = 2_048,
            zero_shock_n_starts = 32,
            zero_shock_max_steps = 750,
        ),
        production = (
            num_segments = 1_601,
            n_trajectories = 32,
            simulation_length = 512,
            batch_size = 1_024,
            learning_rate = 1e-4,
            monitor_every = 25,
            time_invariance_anchor_states = 4_096,
            zero_shock_n_starts = 64,
            zero_shock_max_steps = 1_500,
        ),
    )
    hp = run_mode_budget(RUN_MODE; budgets)
    rng = rng_from_seed(SEED)
end

# ╔═╡ a0c95810-e9c5-5679-66b2-8f71c4b7b397
md"""
## 1. Economic parameters

We use \$z_t^j=\log a_t^j\$ and

\$\$
  z_{t+1}^j=\rho z_t^j+\sigma(\varepsilon_{t+1}^j+\varepsilon_{t+1}^{agg}).
\$\$

The model equations and the error measures follow the IRBC section of Brumm and Scheidegger. The main additional objects relative to the smooth notebook are the KKT multipliers \$\mu_t^j\$ and the complementarity residuals.

In this Julia preview every parameter is collected in one place by `lecture0402_parameters()`, which returns an immutable NamedTuple `p` threaded through every function below.
"""

# ╔═╡ 303e6fae-3f96-9d0c-2aa9-b13bbe7338b0
md"""
## 2. Integration rule

The monomial rule uses \$2(N+1)\$ nodes for the \$N\$ idiosyncratic shocks and the one aggregate shock.

The Julia cell builds it with `stroud3_normal_rule(p.n_shocks)` and reports `stroud_checks` (weight sum and first/second moments) to confirm it integrates a standard normal correctly.
"""

# ╔═╡ 44444444-0402-4444-8444-444444444444
begin
    function lecture0402_parameters()
        n_countries = 2
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
        A_tfp = (1 / beta - 1 + delta) / (zeta * K_REF^(zeta - 1))
        Y_ref = A_tfp * K_REF^zeta
        C_ref = Y_ref - delta * K_REF
        gammas = collect(range(gamma_min, gamma_max; length = n_countries))
        taus = LAMBDA_REF .* C_ref .^ (1 ./ gammas)
        z_std = sigma_e * sqrt(2 / (1 - rho_z^2))
        z_bound = 3 * z_std
        investment_max_fraction = 0.25
        initial_investment_fraction = min(max(delta / investment_max_fraction, 1e-5), 1 - 1e-5)
        investment_bias = log(initial_investment_fraction / (1 - initial_investment_fraction))

        return (
            n_countries = n_countries,
            n_states = 2 * n_countries,
            n_policies = 2 * n_countries + 1,
            n_shocks = n_countries + 1,
            beta = beta,
            zeta = zeta,
            delta = delta,
            rho_z = rho_z,
            sigma_e = sigma_e,
            kappa = kappa,
            gammas = gammas,
            taus = taus,
            K_REF = K_REF,
            LAMBDA_REF = LAMBDA_REF,
            A_tfp = A_tfp,
            Y_ref = Y_ref,
            C_ref = C_ref,
            z_std = z_std,
            z_bound = z_bound,
            exogenous_k_low = 0.55,
            exogenous_k_high = 1.80,
            exogenous_z_low = -z_bound,
            exogenous_z_high = z_bound,
            initial_k_low = 0.65,
            initial_k_high = 1.45,
            initial_z_low = -1.5 * z_std,
            initial_z_high = 1.5 * z_std,
            sim_repair_k_min = 0.05,
            sim_repair_k_max = 8.0,
            sim_repair_abs_z_max = 8.0 * z_std,
            input_k_log_scale = 0.50,
            input_z_scale = max(z_bound, 1e-6),
            lambda_log_scale = 1.25,
            investment_max_fraction = investment_max_fraction,
            investment_bias = investment_bias,
            mu_softplus_bias = -10.0,
            mu_scale = 1.0,
            eps = 1e-8,
            fb_eps = 1e-4,
            euler_weight = 1.0,
            arc_weight = 1.0,
            fb_weight = 1.0,
            binding_tol_investment_fraction = 1e-4,
            zero_shock_tol = 1e-7,
            zero_shock_fixed_point_tol = 1e-4,
            zero_shock_z_tol = 1e-4,
        )
    end

    p = lecture0402_parameters()
    rule = stroud3_normal_rule(p.n_shocks)
    stroud_checks = (
        weight_sum = sum(rule.weights),
        first_moment = quadrature_expectation(x -> x[1], rule),
        second_moment = quadrature_expectation(x -> x[1]^2, rule),
    )
end

# ╔═╡ 34745ad6-6b45-ba3d-058a-ebc985f20c61
md"""
## 3. Neural network and policy transformation

The irreversible model is numerically delicate because simulation must not generate infeasible capital states. We therefore hard-code investment feasibility:

\$\$
   I_t^j\geq 0,\qquad k_{t+1}^j=(1-\delta)k_t^j+I_t^j.
\$\$

The network chooses an investment fraction with a sigmoid transformation. The initial policy sets \$I_t^j\approx\delta k_t^j\$, hence \$k_{t+1}^j\approx k_t^j\$, but the training trajectories themselves start from dispersed feasible states.

In Lux this is `irreversible_policy_from_raw`: the raw outputs pass through `sigmoid` (the investment fraction, biased by `investment_bias` so \$I\approx\delta k\$ at initialization), \$\exp\circ\tanh\$ (the marginal-utility level \$\lambda\$), and `softplus` (the KKT multiplier \$\mu\geq 0\$). The final Dense layer is initialized to zeros so the untrained policy is exactly this reference point.

In this preview the policy network uses two 32-unit `tanh` hidden layers to keep the smoke run light. The full Python notebook uses two 128-unit hidden layers (`NUM_HIDDEN_1 = NUM_HIDDEN_2 = 128`), as does the smooth companion notebook `lecture_04_01`; that wider network has more approximation capacity when reproducing the slide/script accuracy figures.
"""

# ╔═╡ 55555555-0402-4555-8555-555555555555
begin
    zero_weight(rng, out, in) = zeros(Float64, out, in)
    zero_bias(rng, out) = zeros(Float64, out)

    function build_irreversible_policy_network(p)
        return Lux.Chain(
            Lux.Dense(p.n_states => 32, NNlib.tanh),
            Lux.Dense(32 => 32, NNlib.tanh),
            Lux.Dense(32 => p.n_policies; init_weight = zero_weight, init_bias = zero_bias),
        )
    end

    function scale_states(states, p)
        x = assert_feature_batch(states, p.n_states)
        n = p.n_countries
        k = max.(@view(x[1:n, :]), p.eps)
        z = @view x[(n + 1):(2 * n), :]
        k_scaled = log.(k ./ p.K_REF) ./ p.input_k_log_scale
        z_scaled = z ./ p.input_z_scale
        return vcat(k_scaled, z_scaled)
    end

    function irreversible_policy_from_raw(raw, states, p)
        n = p.n_countries
        size(raw, 1) == p.n_policies ||
            throw(DimensionMismatch("policy output must have $(p.n_policies) rows"))
        k = max.(@view(states[1:n, :]), p.eps)
        raw_inv = @view raw[1:n, :]
        raw_lambda = @view raw[(n + 1):(n + 1), :]
        raw_mu = @view raw[(n + 2):(2 * n + 1), :]

        investment_fraction = p.investment_max_fraction .* NNlib.sigmoid.(raw_inv .+ p.investment_bias)
        investment = k .* investment_fraction
        kp = (1 - p.delta) .* k .+ investment
        lambda = p.LAMBDA_REF .* exp.(p.lambda_log_scale .* tanh.(raw_lambda))
        mu = p.mu_scale .* lambda .* NNlib.softplus.(raw_mu .+ p.mu_softplus_bias)
        return (
            next_capital = kp,
            lambda = lambda,
            mu = mu,
            investment = investment,
            investment_fraction = investment_fraction,
        )
    end

    function policy(model, ps, st, states, p)
        raw, st_new = model(scale_states(states, p), ps, st)
        return irreversible_policy_from_raw(raw, states, p), st_new
    end
end

# ╔═╡ 90cbc543-4b00-08a3-2a38-49af42f2d502
md"""
## 4. Residuals and loss

The Euler equation with irreversibility is

\$\$
  \lambda_t(1+\kappa g_{t+1}^j)-\mu_t^j
  =\beta E_t\left[\lambda_{t+1}\left(MPK_{t+1}^j+1-\delta+
  \frac{\kappa}{2}g_{t+2}^j(g_{t+2}^j+2)\right)-(1-\delta)\mu_{t+1}^j\right].
\$\$

We report a relative Euler wedge. Complementarity is monitored with a dimensionless Fischer--Burmeister residual applied to \$(\mu_t^j/\lambda_t, I_t^j/k_t^j)\$.

The primitives (production, `MPK`, adjustment cost, and `python_fischer_burmeister`, \$a+b-\sqrt{a^2+b^2+\varepsilon^2}\$) are defined in this cell; `irreversible_residual` in the next cell evaluates the expectation with the monomial `rule` and assembles the Euler wedge, the relative aggregate-resource residual, and the FB complementarity residual into the total loss `euler_weight · mean(euler²) + arc_weight · mean(arc²) + fb_weight · mean(fb²)`.
"""

# ╔═╡ 66666666-0402-4666-8666-666666666666
begin
    production(k, z, p) = p.A_tfp .* exp.(z) .* max.(k, p.eps) .^ p.zeta

    production_k(k, z, p) =
        p.zeta .* p.A_tfp .* exp.(z) .* max.(k, p.eps) .^ (p.zeta - 1)

    function adjustment_cost(k, kp, p)
        k_safe = max.(k, p.eps)
        g = kp ./ k_safe .- 1
        return 0.5 * p.kappa .* k_safe .* g .^ 2
    end

    adjustment_cost_kp(k, kp, p) = p.kappa .* (kp ./ max.(k, p.eps) .- 1)

    function consumption_from_lambda(lambda, p)
        taus = reshape(p.taus, :, 1)
        gammas = reshape(p.gammas, :, 1)
        return (max.(lambda, p.eps) ./ taus) .^ (-gammas)
    end

    python_fischer_burmeister(a, b, p) =
        a .+ b .- sqrt.(a .^ 2 .+ b .^ 2 .+ p.fb_eps^2)

    function next_state_from_shock(states, kp, node, p)
        n = p.n_countries
        z = @view states[(n + 1):(2 * n), :]
        eps_idio = reshape(node[1:n], n, 1)
        eps_agg = node[n + 1]
        z_next = p.rho_z .* z .+ p.sigma_e .* (eps_idio .+ eps_agg)
        return vcat(kp, z_next)
    end
end

# ╔═╡ 77777777-0402-4777-8777-777777777777
begin
    function irreversible_residual(model, ps, st, states; p = p, rule = rule)
        x = assert_feature_batch(states, p.n_states)
        n = p.n_countries
        k = @view x[1:n, :]
        z = @view x[(n + 1):(2 * n), :]
        pol, st_new = policy(model, ps, st, x, p)

        lhs = pol.lambda .* (1 .+ adjustment_cost_kp(k, pol.next_capital, p))
        expectation = zero(pol.next_capital)
        for q in eachindex(rule.weights)
            state_next = next_state_from_shock(x, pol.next_capital, @view(rule.nodes[:, q]), p)
            pol_next, _ = policy(model, ps, st_new, state_next, p)
            k_next = @view state_next[1:n, :]
            z_next = @view state_next[(n + 1):(2 * n), :]
            g_next = pol_next.next_capital ./ max.(k_next, p.eps) .- 1
            return_next = production_k(k_next, z_next, p) .+ 1 .- p.delta .+
                0.5 * p.kappa .* g_next .* (g_next .+ 2)
            integrand = pol_next.lambda .* return_next .- (1 - p.delta) .* pol_next.mu
            expectation = expectation .+ rule.weights[q] .* integrand
        end

        rhs = p.beta .* expectation .+ pol.mu
        euler = rhs ./ max.(lhs, p.eps) .- 1

        y = production(k, z, p)
        c = consumption_from_lambda(pol.lambda, p)
        gamma_cost = adjustment_cost(k, pol.next_capital, p)
        arc_level = sum(y .+ (1 - p.delta) .* k .- pol.next_capital .- gamma_cost .- c; dims = 1)
        arc_scale = sum(y .+ (1 - p.delta) .* k; dims = 1)
        arc = arc_level ./ max.(arc_scale, p.eps)

        mu_rel = pol.mu ./ max.(pol.lambda, p.eps)
        I_rel = pol.investment ./ max.(k, p.eps)
        fb = python_fischer_burmeister(mu_rel, I_rel, p)
        comp_product = mu_rel .* I_rel

        loss_euler = mean(abs2, euler)
        loss_arc = mean(abs2, arc)
        loss_fb = mean(abs2, fb)
        loss = p.euler_weight * loss_euler + p.arc_weight * loss_arc + p.fb_weight * loss_fb
        return (
            loss = loss,
            euler = euler,
            resource = arc,
            complementarity = fb,
            next_capital = pol.next_capital,
            lambda = pol.lambda,
            mu = pol.mu,
            investment = pol.investment,
            investment_fraction = pol.investment_fraction,
            comp_product = comp_product,
            loss_euler = loss_euler,
            loss_arc = loss_arc,
            loss_fb = loss_fb,
        ), st_new
    end

    irreversible_loss(model, ps, st, states) = begin
        pieces, st_new = irreversible_residual(model, ps, st, states)
        return pieces.loss, st_new
    end
end

# ╔═╡ 7063d01b-ed38-bebd-0665-7e8ce1cc78e9
md"""
## 5. Training-data construction

The simulation sampler below mirrors the baseline DEQN notebook.

1. Start from current trajectory heads `X_start`.
2. Simulate a stochastic segment of length `SIMULATION_LENGTH` under the current policy.
3. Flatten the simulated states into training data.
4. Keep the terminal states `X_end`.
5. Train the network on the segment and then set `X_start = X_end`.

Feasibility is maintained by construction because the policy always satisfies \$I_t^j\geq0\$ and \$k_{t+1}^j=(1-\delta)k_t^j+I_t^j\$. In Julia the sampler is `get_training_data_simulation` / `simulate_path`, with `repair_bad_states` resetting any track that leaves the feasible box.
"""

# ╔═╡ 88888888-0402-4888-8888-888888888888
begin
    function sample_feasible_initial_states(rng, p, n_tracks::Integer)
        log_k = log(p.initial_k_low) .+
            (log(p.initial_k_high) - log(p.initial_k_low)) .* rand(rng, p.n_countries, n_tracks)
        k0 = exp.(log_k)
        z0 = p.initial_z_low .+
            (p.initial_z_high - p.initial_z_low) .* rand(rng, p.n_countries, n_tracks)
        return vcat(k0, z0)
    end

    function sample_exogenous_states(rng, p, n_data::Integer)
        k = p.exogenous_k_low .+
            (p.exogenous_k_high - p.exogenous_k_low) .* rand(rng, p.n_countries, n_data)
        z = p.exogenous_z_low .+
            (p.exogenous_z_high - p.exogenous_z_low) .* rand(rng, p.n_countries, n_data)
        return vcat(k, z)
    end

    function simulate_single_step(rng, train_state, states, p)
        pol, _ = policy(train_state.model, train_state.ps, train_state.st, states, p)
        shocks = randn(rng, p.n_shocks, size(states, 2))
        z = @view states[(p.n_countries + 1):(2 * p.n_countries), :]
        eps_idio = @view shocks[1:p.n_countries, :]
        eps_agg = @view shocks[(p.n_countries + 1):(p.n_countries + 1), :]
        z_next = p.rho_z .* z .+ p.sigma_e .* (eps_idio .+ eps_agg)
        return vcat(pol.next_capital, z_next)
    end

    function repair_bad_states(rng, states, p)
        k = @view states[1:p.n_countries, :]
        z = @view states[(p.n_countries + 1):(2 * p.n_countries), :]
        bad = vec(
            .!all(isfinite.(states); dims = 1) .|
            any(k .< p.sim_repair_k_min; dims = 1) .|
            any(k .> p.sim_repair_k_max; dims = 1) .|
            any(abs.(z) .> p.sim_repair_abs_z_max; dims = 1)
        )
        any(bad) || return states
        repaired = copy(states)
        repaired[:, bad] .= sample_feasible_initial_states(rng, p, count(identity, bad))
        return repaired
    end

    function simulate_path(rng, train_state, X_start, p, n_steps::Integer)
        current = copy(X_start)
        n_tracks = size(current, 2)
        path = Matrix{Float64}(undef, p.n_states, n_tracks * n_steps)
        for t in 1:n_steps
            path[:, ((t - 1) * n_tracks + 1):(t * n_tracks)] .= current
            current = simulate_single_step(rng, train_state, current, p)
            current = repair_bad_states(rng, current, p)
        end
        return path, current
    end

    function get_training_data_simulation(rng, train_state, X_start, p, n_steps)
        path, X_end = simulate_path(rng, train_state, X_start, p, n_steps)
        return path, X_end
    end
end

# ╔═╡ 0ae102eb-90f8-b441-550e-4db98d6fb0fc
md"""
## 6. Mini-batches and training loop

The training loop (`train_irreversible_segments!`) makes the continuation step explicit: each segment is used once and then the simulation continues from its terminal states (`X_start = X_end`). `batch_size` controls whether the segment is used as one full batch or as several stochastic mini-batches (`iterate_minibatches`).
"""

# ╔═╡ 99999999-0402-4999-8999-999999999999
begin
    function iterate_minibatches(rng, X, batch_size; shuffle::Bool = true)
        n = size(X, 2)
        if batch_size <= 0 || batch_size >= n
            return (X,)
        end
        idx = shuffle ? Random.randperm(rng, n) : collect(1:n)
        return tuple((@view X[:, idx[start:min(start + batch_size - 1, n)]] for start in 1:batch_size:n)...)
    end

    function train_on_segment_states!(rng, train_state, X_segment, batch_size)
        losses = Float64[]
        n_updates = 0
        for X_batch in iterate_minibatches(rng, X_segment, batch_size)
            metrics = train_step!(train_state, irreversible_loss, Matrix(X_batch); max_grad_norm = 10.0)
            push!(losses, metrics.loss)
            n_updates += 1
        end
        return mean(losses), n_updates
    end

    function policy_fingerprint(train_state, states, p)
        pol, _ = policy(train_state.model, train_state.ps, train_state.st, states, p)
        log_kp = log.(max.(pol.next_capital, p.eps) ./ p.K_REF)
        log_lambda = log.(max.(pol.lambda, p.eps) ./ p.LAMBDA_REF)
        mu_rel = pol.mu ./ max.(pol.lambda, p.eps)
        return vcat(log_kp, log_lambda, mu_rel, pol.investment_fraction)
    end

    function relative_policy_drift(previous, current)
        diff = current .- previous
        scale = 1 + sqrt(mean(previous .^ 2))
        rms = sqrt(mean(diff .^ 2)) / scale
        max_abs = maximum(abs.(diff)) / scale
        return rms, max_abs
    end
end

# ╔═╡ aaaaaaaa-0402-4aaa-8aaa-aaaaaaaaaaaa
begin
    function train_irreversible_segments!(rng, state, p, hp)
        X_start = sample_feasible_initial_states(rng, p, hp.n_trajectories)
        X_anchor = sample_exogenous_states(rng_from_seed(SEED; offset = 91_017), p, hp.time_invariance_anchor_states)
        anchor_previous = policy_fingerprint(state, X_anchor, p)
        history = NamedTuple[]

        for seg in 0:(hp.num_segments - 1)
            X_segment, X_end = get_training_data_simulation(rng, state, X_start, p, hp.simulation_length)
            mean_train_loss, n_updates = train_on_segment_states!(rng, state, X_segment, hp.batch_size)
            X_start = X_end
            if seg % hp.monitor_every == 0 || seg == hp.num_segments - 1
                diagnostics, _ = irreversible_residual(state.model, state.ps, state.st, X_segment)
                anchor_now = policy_fingerprint(state, X_anchor, p)
                drift_rms, drift_max = relative_policy_drift(anchor_previous, anchor_now)
                anchor_previous = anchor_now
                append_metric!(
                    history;
                    segment = seg,
                    loss = diagnostics.loss,
                    mean_train_loss = mean_train_loss,
                    n_updates = n_updates,
                    mean_abs_euler = mean(abs, diagnostics.euler),
                    mean_abs_resource = mean(abs, diagnostics.resource),
                    mean_abs_fb = mean(abs, diagnostics.complementarity),
                    policy_drift_rms = drift_rms,
                    policy_drift_max = drift_max,
                    min_investment_fraction = minimum(diagnostics.investment_fraction),
                    share_near_binding = mean(diagnostics.investment_fraction .< p.binding_tol_investment_fraction),
                )
            end
        end
        return history
    end

    model = build_irreversible_policy_network(p)
    state = setup_training(rng_from_seed(SEED; offset = 1), model, Optimisers.Adam(hp.learning_rate); parameter_type = Float64)

    reference_states = vcat(ones(p.n_countries, 3), zeros(p.n_countries, 3))
    initial_reference, _ = irreversible_residual(state.model, state.ps, state.st, reference_states)
    initial_loss = initial_reference.loss
    history = train_irreversible_segments!(rng, state, p, hp)
end

# ╔═╡ 78297a77-7be1-9558-ada9-e56de5c6982f
md"""
## 8. Time-invariance and zero-shock stochastic steady state

The architecture has no calendar-time input, so a fixed parameter vector defines a time-homogeneous recursive policy. What still has to be checked numerically is whether the learned policy function has stopped moving across SGD updates. The notebook checks this by comparing policies on a fixed holdout cloud `X_anchor` after each monitoring interval.

In this preview `X_anchor` is drawn entirely from the exogenous box. The full Python notebook instead builds this holdout cloud from roughly 75% exogenous-box and 25% initial-box states (`n_exog = ceil(0.75 * n_anchor)`) and then shuffles them, so the reported policy-drift numbers are not directly comparable to the Python run even though the diagnostic itself is unchanged.

The zero-shock stochastic steady state is the fixed point of the learned stochastic policy when realized shocks are set to zero. In the irreversible model this is especially useful: at a sensible steady state, investment should equal depreciation, the irreversibility constraint should not bind, and the KKT multiplier should be close to zero.

The policy-drift (time-invariance) monitor was accumulated inside the training loop above; this cell iterates the zero-shock map to its fixed point and checks that \$k_{t+1}=k_t\$, that \$z=0\$, that the investment fraction sits at \$\delta\$, and that the relative multiplier \$\mu/\lambda\$ is near zero. In this Julia preview the zero-shock check runs just before the general final-diagnostics report of §7.
"""

# ╔═╡ bbbbbbbb-0402-4bbb-8bbb-bbbbbbbbbbbb
begin
    function simulate_single_step_zero_shock(train_state, states, p)
        pol, _ = policy(train_state.model, train_state.ps, train_state.st, states, p)
        z = @view states[(p.n_countries + 1):(2 * p.n_countries), :]
        z_next = p.rho_z .* z
        return vcat(pol.next_capital, z_next)
    end

    function scaled_transition_distance(x, x_next, p)
        k = max.(@view(x[1:p.n_countries, :]), p.eps)
        kp = max.(@view(x_next[1:p.n_countries, :]), p.eps)
        z = @view x[(p.n_countries + 1):(2 * p.n_countries), :]
        z_next = @view x_next[(p.n_countries + 1):(2 * p.n_countries), :]
        d_k = maximum(abs.(log.(kp ./ k)))
        d_z = maximum(abs.(z_next .- z)) / max(p.input_z_scale, p.eps)
        return max(d_k, d_z)
    end

    function compute_zero_shock_stochastic_steady_state(train_state, p, hp)
        X = sample_feasible_initial_states(rng_from_seed(SEED; offset = 51_031), p, hp.zero_shock_n_starts)
        distances = Float64[]
        converged = false
        step = 0
        for i in 1:hp.zero_shock_max_steps
            X_next = simulate_single_step_zero_shock(train_state, X, p)
            all(isfinite, X_next) || break
            dist = scaled_transition_distance(X, X_next, p)
            push!(distances, dist)
            X = X_next
            step = i
            if dist <= p.zero_shock_tol
                converged = true
                break
            end
        end
        return X, distances, step, converged
    end

    X_ss, zero_shock_distances, zero_shock_steps, zero_shock_converged =
        compute_zero_shock_stochastic_steady_state(state, p, hp)
    ss_diagnostics, _ = irreversible_residual(state.model, state.ps, state.st, X_ss)
    ss_k = @view X_ss[1:p.n_countries, :]
    ss_z = @view X_ss[(p.n_countries + 1):(2 * p.n_countries), :]
    ss_capital_fixed_error = maximum(abs.(ss_diagnostics.next_capital .- ss_k) ./ max.(ss_k, p.eps))
    ss_investment_fraction_delta_error = maximum(abs.(ss_diagnostics.investment_fraction .- p.delta))
    ss_mean_mu_rel = mean(ss_diagnostics.mu ./ max.(ss_diagnostics.lambda, p.eps))
end

# ╔═╡ ec591cf1-1a91-3325-7986-d69c388e2190
md"""
## 7. Final diagnostics

The final report uses dimensionless errors:

- `|Euler relative|`: relative Euler-equation wedge;
- `|ARC relative|`: resource residual divided by aggregate resources;
- `|FB normalized|`: Fischer--Burmeister complementarity residual for \$(\mu/\lambda,I/k)\$;
- `|mu*I normalized|`: normalized complementarity product \$(\mu/\lambda)(I/k)\$.

A mean Euler error of \$2\times10^{-3}\$ is approximately a 0.2 percent Euler wedge.

Here the residuals are summarized with `residual_summary` on an exogenous evaluation cloud (`eval_states`).

In this preview only the exogenous-cloud evaluation is kept. The full Python notebook §7 additionally reports residuals on a burn-in simulated on-distribution cloud (`residual_report` "Out-of-sample simulated states"), an accuracy check on where the policy actually spends its time that is omitted here.
"""

# ╔═╡ cccccccc-0402-4ccc-8ccc-cccccccccccc
begin
    eval_states = sample_exogenous_states(rng, p, 256)
    diagnostics, _ = irreversible_residual(state.model, state.ps, state.st, eval_states)
    fb_stats = residual_summary(diagnostics.complementarity)
    euler_stats = residual_summary(diagnostics.euler)
    resource_stats = residual_summary(diagnostics.resource)
end

# ╔═╡ 26488e9e-7836-37fa-606c-48656f1ab2ab
md"""
## 9. Simple plots

The irreversible model can fail silently if complementarity is ignored. The diagnostics therefore center on the Fischer--Burmeister structure and the investment-versus-multiplier relationship.

This Julia preview renders, with CairoMakie, the investment-fraction \$I/k\$ versus relative-multiplier \$\mu/\lambda\$ scatter for country 1; a healthy solution clusters at positive investment fractions with \$\mu/\lambda\approx 0\$ (the irreversibility constraint slack).
"""

# ╔═╡ dddddddd-0402-4ddd-8ddd-dddddddddddd
begin
    fig = Figure(size = figure_size(RUN_MODE))
    ax = Axis(fig[1, 1], xlabel = "investment fraction I/k", ylabel = "relative KKT multiplier mu/lambda")
    scatter!(
        ax,
        vec(diagnostics.investment_fraction[1, :]),
        vec(diagnostics.mu[1, :] ./ max.(diagnostics.lambda[1, :], p.eps));
        color = (:darkorange, 0.65),
    )
    fig
end

# ╔═╡ e3aa2751-e2eb-8d7d-6966-03991a75083d
md"""
### How to change the training data and optimizer (and a closing summary)

To experiment, edit the active `budgets` entry in the configuration cell:

- **One long trajectory:** set `n_trajectories = 1`, `simulation_length = 1024`.
- **Several shorter trajectories:** set `n_trajectories = 10`, `simulation_length = 256`.
- **Optimizer / schedule:** set `learning_rate` (feeding `Optimisers.Adam`) and `batch_size`; each segment is used once before the trajectories continue.

The policy-stability check is controlled by `time_invariance_anchor_states`; the zero-shock stochastic steady-state check by `zero_shock_n_starts`, `zero_shock_max_steps`, and `zero_shock_tol`.

This preview provides only the persistent-simulation training sampler. The full Python notebook also exposes an exogenous training-data mode (`SAMPLING_MODE = "exogenous"`), together with a §10 instruction to switch to it; that alternative sampler and its how-to are omitted here.

### Conclusion

This Lux/Pluto preview reproduces the irreversible-investment IRBC DEQN pipeline: the KKT-augmented Euler equation, the hard-coded feasibility \$k_{t+1}=(1-\delta)k_t+I_t\$ with \$I_t\geq0\$, the Fischer--Burmeister complementarity residual on \$(\mu/\lambda, I/k)\$, the \$2(N+1)\$-node Stroud monomial expectation, the persistent-simulation sampler, and the time-invariance and zero-shock stochastic-steady-state diagnostics. The cell below returns a machine-checkable summary of this smoke run.
"""

# ╔═╡ eeeeeeee-0402-4eee-8eee-eeeeeeeeeeee
(
    run_mode = RUN_MODE,
    seed = SEED,
    state_order = "k_then_z",
    policy_order = "raw_investment_lambda_raw_mu",
    n_states = p.n_states,
    n_policies = p.n_policies,
    stroud = stroud_checks,
    initial_reference_loss = initial_loss,
    final_loss = history[end].loss,
    final_mean_abs_euler = history[end].mean_abs_euler,
    final_mean_abs_resource = history[end].mean_abs_resource,
    final_mean_abs_fb = history[end].mean_abs_fb,
    max_abs_euler = euler_stats.max_abs,
    max_abs_resource = resource_stats.max_abs,
    max_abs_complementarity = fb_stats.max_abs,
    min_investment_fraction = minimum(diagnostics.investment_fraction),
    share_near_binding = mean(diagnostics.investment_fraction .< p.binding_tol_investment_fraction),
    zero_shock_steps = zero_shock_steps,
    zero_shock_converged = zero_shock_converged,
    zero_shock_final_distance = isempty(zero_shock_distances) ? Inf : zero_shock_distances[end],
    zero_shock_capital_fixed_error = ss_capital_fixed_error,
    zero_shock_max_abs_z = maximum(abs.(ss_z)),
    zero_shock_investment_fraction_delta_error = ss_investment_fraction_delta_error,
    zero_shock_mean_mu_rel = ss_mean_mu_rel,
    finite_share = minimum((
        euler_stats.finite_share,
        resource_stats.finite_share,
        fb_stats.finite_share,
        mean(isfinite.(ss_diagnostics.euler)),
        mean(isfinite.(ss_diagnostics.resource)),
        mean(isfinite.(ss_diagnostics.complementarity)),
    )),
)

# ╔═╡ Cell order:
# ╟─11111111-0402-4111-8111-111111111111
# ╟─9c9b32da-90e5-69c0-465f-9591a1fda4e6
# ╟─ece79b31-4cf9-d51e-39db-ef32ca83d832
# ╠═22222222-0402-4222-8222-222222222222
# ╠═33333333-0402-4333-8333-333333333333
# ╟─a0c95810-e9c5-5679-66b2-8f71c4b7b397
# ╟─303e6fae-3f96-9d0c-2aa9-b13bbe7338b0
# ╠═44444444-0402-4444-8444-444444444444
# ╟─34745ad6-6b45-ba3d-058a-ebc985f20c61
# ╠═55555555-0402-4555-8555-555555555555
# ╟─90cbc543-4b00-08a3-2a38-49af42f2d502
# ╠═66666666-0402-4666-8666-666666666666
# ╠═77777777-0402-4777-8777-777777777777
# ╟─7063d01b-ed38-bebd-0665-7e8ce1cc78e9
# ╠═88888888-0402-4888-8888-888888888888
# ╟─0ae102eb-90f8-b441-550e-4db98d6fb0fc
# ╠═99999999-0402-4999-8999-999999999999
# ╠═aaaaaaaa-0402-4aaa-8aaa-aaaaaaaaaaaa
# ╟─78297a77-7be1-9558-ada9-e56de5c6982f
# ╠═bbbbbbbb-0402-4bbb-8bbb-bbbbbbbbbbbb
# ╟─ec591cf1-1a91-3325-7986-d69c388e2190
# ╠═cccccccc-0402-4ccc-8ccc-cccccccccccc
# ╟─26488e9e-7836-37fa-606c-48656f1ab2ab
# ╠═dddddddd-0402-4ddd-8ddd-dddddddddddd
# ╟─e3aa2751-e2eb-8d7d-6966-03991a75083d
# ╠═eeeeeeee-0402-4eee-8eee-eeeeeeeeeeee
