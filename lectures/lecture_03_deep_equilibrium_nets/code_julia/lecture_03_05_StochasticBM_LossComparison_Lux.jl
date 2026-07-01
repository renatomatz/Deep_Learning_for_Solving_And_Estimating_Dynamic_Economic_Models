### A Pluto.jl notebook ###
# v0.20.0

using Markdown
using InteractiveUtils

# ╔═╡ 11111111-0305-4111-8111-111111111111
md"""
# Lecture 03, Notebook 05: Stochastic Brock-Mirman Loss Kernels

This Pluto translation compares MSE, MAE, Huber, quantile, CVaR, and log-cosh
losses on the same stochastic Brock-Mirman residual batches. The state is the
Python notebook's `(z_t, log K_t)` pair, all loss kernels start from common Lux
parameters, and all path diagnostics reuse the same random innovations.
"""

# ╔═╡ 8216dcc1-2f52-c4be-017f-c052445e05c2
md"""
## Lecture 03, Notebook 05: Loss-Kernel Comparison on Brock–Mirman

**Course:** Deep Learning for Solving and Estimating Dynamic Models in Economics and Finance
**Script reference:** §2.9 (choice of loss kernel: MSE / MAE / Huber / pinball / CVaR / log-cosh)
**Notebook role:** core
**Author:** Simon Scheidegger

*Julia/Lux/Pluto preview of* `lectures/lecture_03_deep_equilibrium_nets/code/lecture_03_05_StochasticBM_LossComparison.ipynb`.
"""

# ╔═╡ 38618cbf-9360-821f-0b67-12ad24df1e64
md"""
## Brock–Mirman with Stochastic TFP: Loss Functions and Convergence

This notebook studies how the **choice of training loss** in a DEQN affects the **economic** convergence of the trained policy. Two distinct objects are at play:

- **Training loss** — the scalar the optimizer minimises (MSE, MAE, Huber, quantile pinball, CVaR, log-cosh, …).
- **Economic loss** — what we actually care about: the *relative Euler equation error along a simulated path*, which maps directly to a consumption-equivalent welfare loss.

These are not the same. A loss that minimises the *mean* squared residual can leave a heavy right tail in rarely visited but economically consequential parts of the state space. We use the stochastic Brock–Mirman model with **full depreciation**, where the optimal savings rate is known in closed form,

\$\$s^\star = \alpha\beta,\$\$

to measure the discrepancy exactly. The **same** network is trained six times, once per loss kernel, with identical initialisation, optimizer, and training states (common random numbers). Only the *reduction* applied to the Euler residual differs. (Companion to Notebook 02, which uses MSE only with partial depreciation; we switch to full depreciation here purely to expose the closed-form benchmark.)
"""

# ╔═╡ e70ffba9-a572-743d-70c3-a4cd79bc022e
md"""
### Setup

Where the Python notebook uses NumPy and TensorFlow/Keras, the Julia preview loads `Lux`, `Optimisers`, `NNlib`, `Statistics`, and `CairoMakie`, plus the shared `DLEFJulia` loss kernels (`mse_loss`, `mae_loss`, `huber_loss`, `pinball_loss`, `logcosh_loss`) and quadrature helpers.
"""

# ╔═╡ 22222222-0305-4222-8222-222222222222
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

# ╔═╡ f02afa8d-5933-99f2-1ed0-e0df9cac5d26
md"""
### Training-budget switch

All six losses share the same number of episodes; only the reduction kernel varies. `RUN_MODE` selects the Julia budget — `smoke` is tiny (a handful of episodes, just a load check), while `teaching` (300 episodes/kernel) and `production` (5000) run the real comparison; `SEED = 0` fixes the RNG. The flag `SAVE_FIGS = (RUN_MODE == "production")` gates PNG output: the slide figures are written only under `production`, so `smoke` and `teaching` runs leave the checked-in figures untouched. This preview keeps `RUN_MODE = "smoke"` and `SEED = 0`.
"""

# ╔═╡ 33333333-0305-4333-8333-333333333333
begin
    RUN_MODE = "smoke"
    SEED = 0
    SAVE_FIGS = RUN_MODE == "production"
    budgets = (
        smoke = (episodes = 8, batch_size = 32, eval_every = 2, eval_grid_points = 7, path_length = 120, burn_in = 20),
        teaching = (episodes = 300, batch_size = 64, eval_every = 25, eval_grid_points = 21, path_length = 2_000, burn_in = 200),
        production = (episodes = 5_000, batch_size = 128, eval_every = 100, eval_grid_points = 21, path_length = 2_000, burn_in = 200),
    )
    hp = run_mode_budget(RUN_MODE; budgets)
end

# ╔═╡ a8374111-d48c-a8c2-33da-95930fe959e1
md"""
### 1. Model: Stochastic Brock–Mirman (1972)

- Production: \$Y_t = e^{z_t} K_t^{\alpha}\$
- AR(1) for log TFP: \$z_{t+1} = \rho\, z_t + \sigma\, \varepsilon_{t+1}\$, \$\varepsilon \sim \mathcal{N}(0,1)\$
- **Full depreciation**, so capital evolves as \$K_{t+1} = s_t Y_t\$
- Log utility, discount \$\beta\$
- Euler equation: \$1 = \beta\, \mathbb{E}_t\!\left[\dfrac{C_t}{C_{t+1}}\, \alpha e^{z_{t+1}} K_{t+1}^{\alpha-1}\right]\$
- **Closed-form optimal savings rate**: \$s^\star = \alpha\beta\$

The closed form is why we set \$\delta = 1\$: it gives an exact reference for any learned policy, including a single-number consumption-equivalent welfare loss. This cell also builds the Gauss–Hermite `rule`, the deterministic steady state \$K_{ss} = (s^\star)^{1/(1-\alpha)}\$, the ergodic productivity dispersion, and the list of six `LOSS_KERNELS`.
"""

# ╔═╡ 44444444-0305-4444-8444-444444444444
begin
    params = BrockMirmanParams(alpha = 0.36, beta = 0.99, delta = 1.0, rho_z = 0.9, sigma_z = 0.04)
    params.delta == 1.0 || error("Lecture 03.05 matches the Python full-depreciation notebook.")

    s_star = params.alpha * params.beta
    K_ss = s_star^(1 / (1 - params.alpha))
    lk_ss = log(K_ss)
    z_sd_ergodic = params.sigma_z / sqrt(1 - params.rho_z^2)

    rule = gauss_hermite_rule(5)
    loss_kinds = LOSS_KERNELS
    kernel_label(kind) = Dict(
        :mse => "MSE",
        :mae => "MAE",
        :huber => "Huber",
        :quantile => "Quantile",
        :cvar => "CVaR",
        :logcosh => "LogCosh",
    )[kind]
end

# ╔═╡ cc501cc2-5b7b-e15d-564a-4d96f7751076
md"""
### 6. Common random numbers

We pre-generate **every** training batch once, so each loss run sees exactly the same sequence of training states — the only difference between runs is the reduction applied to the residual, which is what makes the comparison fair. `make_training_states` draws productivity from its ergodic spread and log-capital around the steady state; `make_eval_states` builds a fixed \$(z, \log K)\$ grid spanning the ergodic support; and `PATH_INNOVATIONS` fixes the shock sequence used later for path simulation.
"""

# ╔═╡ 55555555-0305-4555-8555-555555555555
begin
    function make_training_states(rng, total; params, lk_ss)
        z = z_sd_ergodic .* randn(rng, total)
        lk = (lk_ss - 0.5) .+ rand(rng, total)
        return vcat(reshape(z, 1, :), reshape(lk, 1, :))
    end

    function make_eval_states(n; params, lk_ss)
        z_grid = collect(range(-3z_sd_ergodic, 3z_sd_ergodic; length = n))
        lk_grid = collect(range(lk_ss - 0.4, lk_ss + 0.4; length = n))
        states = reduce(hcat, ([z, lk] for z in z_grid for lk in lk_grid))
        return states, z_grid, lk_grid
    end

    rng_data = rng_from_seed(SEED)
    total_training_points = hp.episodes * hp.batch_size
    TRAIN_STATES = make_training_states(rng_data, total_training_points; params, lk_ss)
    training_batches = [
        TRAIN_STATES[:, ((episode - 1) * hp.batch_size + 1):(episode * hp.batch_size)]
        for episode in 1:hp.episodes
    ]
    EVAL_STATES, z_eval_grid, lk_eval_grid = make_eval_states(hp.eval_grid_points; params, lk_ss)
    PATH_INNOVATIONS = randn(rng_data, hp.path_length)
end

# ╔═╡ 015efd3a-abcd-d4be-1d1e-2531e27d206e
md"""
### 4. Euler residual and ground truth

We work with the **relative** residual

\$\$r(k_t, z_t) = \beta\, \mathbb{E}_t\!\left[\frac{C_t}{C_{t+1}}\, \alpha e^{z_{t+1}} K_{t+1}^{\alpha-1}\right] - 1,\$\$

evaluated by Gauss–Hermite quadrature over next-period productivity. `python_stochastic_bm_residual` mirrors the Python `euler_residual` exactly: it returns the vector \$r\$ (plus savings, consumption, and next-period capital). Every loss kernel below is a different *reduction* of this same \$r\$ to a scalar — the residual itself is identical across all six runs.
"""

# ╔═╡ 66666666-0305-4666-8666-666666666666
begin
    function python_stochastic_bm_residual(model, ps, st, state_feature_batch, rule; params)
        x = assert_feature_batch(state_feature_batch, 2)
        z_t = @view x[1:1, :]
        lk_t = @view x[2:2, :]

        Z_t = exp.(z_t)
        K_t = exp.(lk_t)
        Y_t = Z_t .* K_t .^ params.alpha
        s_t, st_new = model(x, ps, st)
        K_tp1 = s_t .* Y_t
        C_t = (1 .- s_t) .* Y_t
        lk_tp1 = log.(K_tp1)

        expectation = zero.(z_t)
        for (node, weight) in zip(rule.nodes, rule.weights)
            z_tp1 = params.rho_z .* z_t .+ params.sigma_z .* node
            Z_tp1 = exp.(z_tp1)
            state_tp1 = vcat(z_tp1, lk_tp1)
            s_tp1, _ = model(state_tp1, ps, st_new)
            Y_tp1 = Z_tp1 .* K_tp1 .^ params.alpha
            C_tp1 = (1 .- s_tp1) .* Y_tp1
            R_tp1 = params.alpha .* Z_tp1 .* K_tp1 .^ (params.alpha - 1)
            expectation = expectation .+ weight .* (C_t ./ C_tp1) .* R_tp1
        end

        residual = params.beta .* expectation .- 1
        diagnostics = (
            loss = mse_loss(residual, zero.(residual)),
            residual = residual,
            savings = s_t,
            consumption = C_t,
            next_capital = K_tp1,
            next_log_capital = lk_tp1,
            expectation = expectation,
        )
        return diagnostics, st_new
    end
end

# ╔═╡ d7a8722a-668b-8ccc-0454-4571474f5040
md"""
### 5. The six loss kernels

All operate on the same residual vector \$r \in \mathbb{R}^B\$; only the reduction differs:

| Kernel | Formula | Emphasis |
|---|---|---|
| **MSE** | \$\frac{1}{B}\sum r^2\$ | mean of squared residuals (canonical DEQN choice) |
| **MAE** | \$\frac{1}{B}\sum \lvert r\rvert\$ | median, robust to outliers; **constant gradient magnitude**, so progress stalls at small residuals |
| **Huber** | quadratic for \$\lvert r\rvert \le \delta\$, linear above | smooth MSE/MAE hybrid, but a gradient **kink** at \$\lvert r\rvert = \delta\$ |
| **Quantile** | pinball at \$\tau = 0.9\$: \$\max(\tau r, (\tau-1)r)\$ | trains the upper \$\tau\$-quantile of \$r\$ |
| **CVaR** | mean of the top \$10\%\$ of \$\lvert r\rvert\$ | tail mean — explicitly trains the bad regions |
| **LogCosh** | \$\frac{1}{B}\sum \log\cosh(r)\$ | \$C^\infty\$ hybrid: \$\tfrac{1}{2}r^2\$ near zero, \$\lvert r\rvert - \log 2\$ in the tails, **no kink anywhere** |

`kernel_objective` dispatches on the kernel symbol; CVaR uses a small hand-written `python_cvar_abs_loss` (mean of the worst \$10\%\$ of \$\lvert r\rvert\$). `residual_stats` reports the mean, \$p_{90}\$, \$p_{99}\$, RMSE, and max of \$\lvert r\rvert\$ — the **economic** metric tracked during training.
"""

# ╔═╡ 77777777-0305-4777-8777-777777777777
begin
    function python_cvar_abs_loss(r; alpha = 0.9)
        values = abs.(vec(r))
        k = max(floor(Int, length(values) * (1 - alpha)), 1)
        idx = partialsortperm(values, 1:k; rev = true)
        return mean(values[idx])
    end

    function kernel_objective(kind)
        kind == :mse && return r -> mean(abs2, r)
        kind == :mae && return r -> mean(abs, r)
        kind == :huber && return r -> huber_loss(r, zero.(r); delta = 1e-2)
        kind == :quantile && return r -> pinball_loss(zero.(r), r; quantile = 0.9)
        kind == :cvar && return r -> python_cvar_abs_loss(r; alpha = 0.9)
        kind == :logcosh && return r -> logcosh_loss(r, zero.(r))
        error("unknown loss kernel: $kind")
    end

    function residual_stats(model, ps, st, states)
        pieces, _ = python_stochastic_bm_residual(model, ps, st, states, rule; params)
        ar = abs.(vec(pieces.residual))
        return (
            mean_abs = mean(ar),
            p90 = quantile(ar, 0.90),
            p99 = quantile(ar, 0.99),
            rmse = sqrt(mean(abs2, pieces.residual)),
            max_abs = maximum(ar),
        )
    end
end

# ╔═╡ 4c250bd4-7ae7-433b-0133-0f66936113db
md"""
### 3. Network, and 7. Training rig

An identical \$2\times32\$ **swish** MLP with a sigmoid output head is used across all six runs; the weights are initialised once and each kernel starts from the *same* copy (`deepcopy`), which is what makes the comparison fair. Each kernel then gets its own `train_step!` closure over `Optimisers.Adam(1e-3)` with gradient-norm clipping, and the economic metric (relative Euler error on the eval grid) is logged every `eval_every` episodes. `comparison_results` collects the trained state and convergence history for each kernel.
"""

# ╔═╡ 88888888-0305-4888-8888-888888888888
begin
    make_loss_network() = make_mlp(2, (32, 32), 1; activation = NNlib.swish, final_activation = NNlib.sigmoid)
    initial_model = make_loss_network()
    initial_ps, initial_st = setup_model(rng_from_seed(SEED), initial_model; parameter_type = Float64)

    function run_one_kernel(kind)
        model = make_loss_network()
        state = setup_training(model, deepcopy(initial_ps), deepcopy(initial_st), Optimisers.Adam(1e-3))
        objective = kernel_objective(kind)
        kernel_loss(model, ps, st, states) = begin
            pieces, st_new = python_stochastic_bm_residual(model, ps, st, states, rule; params)
            return objective(pieces.residual), st_new
        end

        history = NamedTuple[]
        for (episode, batch) in enumerate(training_batches)
            metrics = train_step!(state, kernel_loss, batch; max_grad_norm = 10.0)
            if ((episode - 1) % hp.eval_every == 0) || episode == hp.episodes
                stats = residual_stats(state.model, state.ps, state.st, EVAL_STATES)
                append_metric!(history;
                    episode = episode - 1,
                    training_loss = metrics.loss,
                    mean_abs = stats.mean_abs,
                    p90 = stats.p90,
                    p99 = stats.p99,
                    rmse = stats.rmse,
                    max_abs = stats.max_abs)
            end
        end

        final_stats = residual_stats(state.model, state.ps, state.st, EVAL_STATES)
        return (kind = kind, label = kernel_label(kind), history = history, final = final_stats, state = state)
    end

    comparison_results = Dict(kind => run_one_kernel(kind) for kind in loss_kinds)
end

# ╔═╡ 7483ab46-3ed0-b1e2-421e-69548b4ad185
md"""
### 8. Convergence of the *economic* metric

Three panels — the mean, \$p_{90}\$, and \$p_{99}\$ of the absolute relative Euler error on the evaluation grid — plotted against the training episode on a log scale, so vertical distance is multiplicative.

> **Reproducing the published figure.** The PNG shipped with the slides was produced under `RUN_MODE = "production"` (5000 episodes/kernel). Under `smoke` or `teaching` the curves show the qualitative ordering but will not match pixel-for-pixel. Because `SAVE_FIGS = (RUN_MODE == "production")`, this cell writes `slides/figures/loss_kernel_convergence_julia.png` only in production; the smoke run just displays the figure.
"""

# ╔═╡ 99999999-0305-4999-8999-999999999999
begin
    fig = Figure(size = (900, 320))
    axes = [
        Axis(fig[1, 1], xlabel = "Training episode", ylabel = "mean |r|", yscale = log10),
        Axis(fig[1, 2], xlabel = "Training episode", ylabel = "p90 |r|", yscale = log10),
        Axis(fig[1, 3], xlabel = "Training episode", ylabel = "p99 |r|", yscale = log10),
    ]
    palette = [:dodgerblue3, :darkorange, :seagreen, :purple3, :firebrick, :black]
    for (color, kind) in zip(palette, loss_kinds)
        hist = comparison_results[kind].history
        episodes = [row.episode for row in hist]
        lines!(axes[1], episodes, [row.mean_abs for row in hist]; label = kernel_label(kind), color, linewidth = 2)
        lines!(axes[2], episodes, [row.p90 for row in hist]; color, linewidth = 2)
        lines!(axes[3], episodes, [row.p99 for row in hist]; color, linewidth = 2)
    end
    axislegend(axes[1]; position = :rt)
    if SAVE_FIGS
        save(joinpath(@__DIR__, "..", "slides", "figures", "loss_kernel_convergence_julia.png"), fig)
    end
    fig
end

# ╔═╡ 60609e2c-c61f-4da3-c398-4cc56687f59f
md"""
### 9. Final economic metric: along an actual simulated path

After training we forward-simulate each learned policy (and the closed-form optimum \$s^\star\$) under the **same** fixed innovation sequence `PATH_INNOVATIONS`. At every visited state we compute the absolute relative Euler residual, and we also compute the **consumption-equivalent welfare loss** of the learned policy versus \$s^\star\$,

\$\$\lambda = 1 - \exp\!\left((1-\beta)\,(U_{\text{learned}} - U^\star)\right),\$\$

the fraction of consumption the agent would forgo to switch from the learned policy to the optimum. `simulate_with_policy` and `discounted_utility` implement the roll-out and lifetime utility.
"""

# ╔═╡ aaaaaaaa-0305-4aaa-8aaa-aaaaaaaaaaaa
begin
    policy_from_state(train_state) = states -> first(train_state.model(states, train_state.ps, train_state.st))
    optimal_policy(states) = fill(s_star, 1, size(states, 2))

    function simulate_with_policy(policy_fn, T, z0, lk0, innovations; params)
        log_c = Vector{Float64}(undef, T)
        states_hist = Matrix{Float64}(undef, 2, T)
        z = z0
        lk = lk0
        for t in 1:T
            state_t = reshape([z, lk], 2, 1)
            states_hist[:, t] .= vec(state_t)
            s_t = only(policy_fn(state_t))
            Z_t = exp(z)
            K_t = exp(lk)
            Y_t = Z_t * K_t^params.alpha
            C_t = (1 - s_t) * Y_t
            log_c[t] = log(C_t)
            K_tp1 = s_t * Y_t
            lk = log(K_tp1)
            z = params.rho_z * z + params.sigma_z * innovations[t]
        end
        return log_c, states_hist
    end

    function discounted_utility(log_c; params)
        discounts = params.beta .^ collect(0:(length(log_c) - 1))
        return sum(discounts .* log_c)
    end

    z0 = 0.0
    lk0 = lk_ss
    log_c_opt, _ = simulate_with_policy(optimal_policy, hp.path_length, z0, lk0, PATH_INNOVATIONS; params)
    U_opt = discounted_utility(log_c_opt; params)
end

# ╔═╡ bbbbbbbb-0305-4bbb-8bbb-bbbbbbbbbbbb
begin
    function path_summary(kind)
        train_state = comparison_results[kind].state
        policy = policy_from_state(train_state)
        log_c, states_hist = simulate_with_policy(policy, hp.path_length, z0, lk0, PATH_INNOVATIONS; params)
        U = discounted_utility(log_c; params)
        ce_loss = 1 - exp((1 - params.beta) * (U - U_opt))
        pieces, _ = python_stochastic_bm_residual(train_state.model, train_state.ps, train_state.st, states_hist, rule; params)
        ar = abs.(vec(pieces.residual))
        return (
            kind = kind,
            label = kernel_label(kind),
            mean_abs = mean(ar),
            p50 = median(ar),
            p90 = quantile(ar, 0.90),
            p99 = quantile(ar, 0.99),
            max_abs = maximum(ar),
            ce_loss = ce_loss,
            path_residuals = ar,
            states = states_hist,
        )
    end

    path_summaries = Dict(kind => path_summary(kind) for kind in loss_kinds)
end

# ╔═╡ f4090ac7-7a61-8e11-4c57-2c126899bdeb
md"""
The figure histograms \$\lvert r\rvert\$ along the simulated path for each kernel (log-spaced bins), exposing where each loss leaves its residual mass. In `production` this is saved to `slides/figures/loss_kernel_path_residuals_julia.png`.
"""

# ╔═╡ cccccccc-0305-4ccc-8ccc-cccccccccccc
begin
    path_fig = Figure(size = figure_size(RUN_MODE))
    ax = Axis(path_fig[1, 1],
        xlabel = "|r| along simulated path",
        ylabel = "count",
        xscale = log10,
        title = "Path Euler residuals by training loss")
    bins = 10 .^ range(-7, 0; length = 60)
    for (color, kind) in zip(palette, loss_kinds)
        hist!(ax, path_summaries[kind].path_residuals; bins, label = kernel_label(kind),
            color = (color, 0.0), strokecolor = color, strokewidth = 1.6)
    end
    axislegend(ax; position = :rt)
    if SAVE_FIGS
        save(joinpath(@__DIR__, "..", "slides", "figures", "loss_kernel_path_residuals_julia.png"), path_fig)
    end
    path_fig
end

# ╔═╡ 04ba6ab0-70a8-e676-9b21-eba8db4c5ef6
md"""
### 10. Do all six policies reach the same stochastic steady state?

Each kernel yields a slightly different policy, hence a slightly different ergodic distribution under the same shocks. Each policy is forward-simulated under `PATH_INNOVATIONS`, an initial burn-in is dropped, and the remaining ergodic window gives the mean and standard deviation of \$\log K_t\$, of the realised savings rate \$s_t\$, and of output. All six rows should sit close to the `OPTIMAL` row (\$s^\star = \alpha\beta\$); the gaps quantify how much each kernel's residual error perturbs the simulated economy.
"""

# ╔═╡ dddddddd-0305-4ddd-8ddd-dddddddddddd
begin
    function steady_state_moments(label, policy_fn)
        _, states = simulate_with_policy(policy_fn, hp.path_length, z0, lk0, PATH_INNOVATIONS; params)
        idx = (hp.burn_in + 1):hp.path_length
        z = states[1, idx]
        lk = states[2, idx]
        savings = vec(policy_fn(states[:, idx]))
        Y = exp.(z) .* exp.(lk) .^ params.alpha
        return (
            policy = label,
            mean_logK = mean(lk),
            std_logK = std(lk),
            mean_s = mean(savings),
            std_s = std(savings),
            mean_Y = mean(Y),
        )
    end

    steady_state_rows = vcat(
        [steady_state_moments("OPTIMAL", optimal_policy)],
        [steady_state_moments(kernel_label(kind), policy_from_state(comparison_results[kind].state)) for kind in loss_kinds],
    )
end

# ╔═╡ a0aa1cb6-3bca-3b21-b51e-2ad9edc8460a
md"""
### 11. Where in the state space does each loss leave error?

We measure \$\lvert s_{\text{learned}}(z, \log K) - s^\star\rvert\$ across the evaluation grid to see where each learned policy disagrees with the closed-form optimum. *(The Python notebook renders this as a per-kernel heat-map; this Julia preview reports the same information as a compact summary table of the grid error.)*
"""

# ╔═╡ eeeeeeee-0305-4eee-8eee-eeeeeeeeeeee
begin
    function state_error_summary(kind)
        train_state = comparison_results[kind].state
        savings = vec(policy_from_state(train_state)(EVAL_STATES))
        err = abs.(savings .- s_star)
        return (
            kind = kind,
            label = kernel_label(kind),
            mean_abs_s_error = mean(err),
            max_abs_s_error = maximum(err),
            error_map = reshape(err, length(lk_eval_grid), length(z_eval_grid))',
        )
    end

    state_error_summaries = Dict(kind => state_error_summary(kind) for kind in loss_kinds)
    state_error_table = [
        (kernel = row.label, mean_abs_s_error = row.mean_abs_s_error, max_abs_s_error = row.max_abs_s_error)
        for row in (state_error_summaries[kind] for kind in loss_kinds)
    ]
end

# ╔═╡ aaf510e8-8071-e2bb-b1e4-a9f1c3977e86
md"""
### Final metrics

`final_table` collects, per kernel, the end-of-training economic metrics (mean, \$p_{90}\$, \$p_{99}\$, max of \$\lvert r\rvert\$) alongside the path welfare loss \$\lambda\$ — a single comparable summary of all six runs.
"""

# ╔═╡ ffffffff-0305-4fff-8fff-ffffffffffff
begin
    final_table = [(
        kernel = kernel_label(kind),
        eval_mean_abs = comparison_results[kind].final.mean_abs,
        eval_p90 = comparison_results[kind].final.p90,
        eval_p99 = comparison_results[kind].final.p99,
        eval_rmse = comparison_results[kind].final.rmse,
        eval_max_abs = comparison_results[kind].final.max_abs,
        path_mean_abs = path_summaries[kind].mean_abs,
        path_p99 = path_summaries[kind].p99,
        ce_loss = path_summaries[kind].ce_loss,
    ) for kind in loss_kinds]
end

# ╔═╡ 8385b318-b45d-1142-fd51-46c95bb7cdd5
md"""
### 12. Discussion

- **MSE** drives the bulk down fast, but its quadratic gradient amplifies outliers (smooth-on-average curves with occasional jitter).
- **MAE stalls at a finite floor.** The gradient of \$\lvert r\rvert\$ has *constant* magnitude, so once residuals are small the optimizer keeps taking the same step and levels off — the textbook MAE pathology.
- **Huber is robust but kinked.** MSE-like in the bulk, MAE-like on outliers, with a gradient jump at \$\lvert r\rvert = \delta\$ that can induce mild oscillation.
- **Quantile and CVaR target the tail.** Narrowest mean-to-max spread, at a small cost in the mean — the right trade when worst-case policy quality matters.
- **LogCosh converges most smoothly.** The only kernel here that is both \$C^\infty\$ everywhere and combines \$\tfrac{1}{2}r^2\$ near zero with \$\lvert r\rvert - \log 2\$ in the tails: MSE's fine convergence, MAE's outlier taming, no kink.
- **The CE-welfare ranking is not the training-loss ranking.** Log utility is sensitive to outlier consumption, so a tail-aware loss can win on welfare even with a larger mean residual.

**Take-away.** *We do not actually care about MSE.* We care about policy behaviour along simulated equilibrium paths, in consumption-equivalent welfare terms; the training loss is just an instrument. Tail-aware kernels pay off for sharp non-linearities or rare-event regions (occasionally binding constraints, tipping risks, fat tails); for smooth problems the difference is mostly cosmetic and MSE is fine — but log-cosh is rarely a bad default.

**Practical guidance.** Default to **MSE** (**LogCosh** is a lower-risk drop-in for heavy tails); use **Huber** for a few unstable outlier states; use **Quantile** \$\tau = 0.9\$ or **CVaR** \$\alpha = 0.9\$ when worst-case quality matters; **avoid MAE** as the sole loss for fine convergence; and always evaluate on the *economic* metric, not the training loss. Further exercises: sweep \$\tau \in \{0.5, 0.9, 0.99\}\$, combine \$\ell_{\text{MSE}} + \eta\,\ell_{\text{CVaR}}\$, re-run with \$\delta = 0.1\$ (partial depreciation, no closed form), or repeat on the higher-dimensional IRBC model. The cell below returns a machine-checkable summary of all six runs.
"""

# ╔═╡ abababab-0305-4aba-8aba-abababababab
(
    kernels = [kernel_label(kind) for kind in loss_kinds],
    state_representation = "(z, log K)",
    residual = "beta * E[(C_t / C_{t+1}) * alpha * Z_{t+1} * K_{t+1}^{alpha-1}] - 1",
    full_depreciation = params.delta == 1.0,
    steps_per_kernel = hp.episodes,
    common_batches = length(training_batches),
    eval_states = size(EVAL_STATES, 2),
    path_innovations = length(PATH_INNOVATIONS),
    final_table = final_table,
    steady_state_rows = steady_state_rows,
    state_error_table = state_error_table,
    save_figs = SAVE_FIGS,
    all_finite = all(row -> all(isfinite, (
            row.eval_mean_abs,
            row.eval_p90,
            row.eval_p99,
            row.eval_rmse,
            row.eval_max_abs,
            row.path_mean_abs,
            row.path_p99,
            row.ce_loss,
        )), final_table),
)

# ╔═╡ Cell order:
# ╟─11111111-0305-4111-8111-111111111111
# ╟─8216dcc1-2f52-c4be-017f-c052445e05c2
# ╟─38618cbf-9360-821f-0b67-12ad24df1e64
# ╟─e70ffba9-a572-743d-70c3-a4cd79bc022e
# ╠═22222222-0305-4222-8222-222222222222
# ╟─f02afa8d-5933-99f2-1ed0-e0df9cac5d26
# ╠═33333333-0305-4333-8333-333333333333
# ╟─a8374111-d48c-a8c2-33da-95930fe959e1
# ╠═44444444-0305-4444-8444-444444444444
# ╟─cc501cc2-5b7b-e15d-564a-4d96f7751076
# ╠═55555555-0305-4555-8555-555555555555
# ╟─015efd3a-abcd-d4be-1d1e-2531e27d206e
# ╠═66666666-0305-4666-8666-666666666666
# ╟─d7a8722a-668b-8ccc-0454-4571474f5040
# ╠═77777777-0305-4777-8777-777777777777
# ╟─4c250bd4-7ae7-433b-0133-0f66936113db
# ╠═88888888-0305-4888-8888-888888888888
# ╟─7483ab46-3ed0-b1e2-421e-69548b4ad185
# ╠═99999999-0305-4999-8999-999999999999
# ╟─60609e2c-c61f-4da3-c398-4cc56687f59f
# ╠═aaaaaaaa-0305-4aaa-8aaa-aaaaaaaaaaaa
# ╠═bbbbbbbb-0305-4bbb-8bbb-bbbbbbbbbbbb
# ╟─f4090ac7-7a61-8e11-4c57-2c126899bdeb
# ╠═cccccccc-0305-4ccc-8ccc-cccccccccccc
# ╟─04ba6ab0-70a8-e676-9b21-eba8db4c5ef6
# ╠═dddddddd-0305-4ddd-8ddd-dddddddddddd
# ╟─a0aa1cb6-3bca-3b21-b51e-2ad9edc8460a
# ╠═eeeeeeee-0305-4eee-8eee-eeeeeeeeeeee
# ╟─aaf510e8-8071-e2bb-b1e4-a9f1c3977e86
# ╠═ffffffff-0305-4fff-8fff-ffffffffffff
# ╟─8385b318-b45d-1142-fd51-46c95bb7cdd5
# ╠═abababab-0305-4aba-8aba-abababababab
