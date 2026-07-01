### A Pluto.jl notebook ###
# v0.20.0

using Markdown
using InteractiveUtils

# ╔═╡ 11111111-0912-4111-8111-111111111111
md"""
# Lecture 09, Notebook 12: Krusell-Smith with All-in-One Deep Learning in Lux

Compact Julia/Lux translation of the Python notebook. The policy network maps
individual capital, employment, aggregate capital, and aggregate productivity to
a consumption share. Euler expectations are exact sums over the four next-period
aggregate/idiosyncratic states. Phase B follows the Python ground truth: train
on a running stochastic Monte Carlo panel, then advance aggregate and
idiosyncratic states with seeded draws under the updated policy.
"""

# ╔═╡ 8ec9815f-4ef5-33ac-3b69-855feae9af13
md"""
## Lecture 09, Notebook 12: Krusell–Smith with all-in-one deep learning

**Course:** Deep Learning for Solving and Estimating Dynamic Models in Economics and Finance
**Script reference:** §6.6 (Alternative deep-learning approaches to Krusell–Smith)
**Notebook role:** extension
**Author:** Simon Scheidegger

*Julia/Lux/Pluto preview of* `lectures/lecture_09_heterogeneous_agents_youngs_method/code/lecture_09_12_KrusellSmith_DeepLearning.ipynb`.
"""

# ╔═╡ 2668ff05-75a1-3d9e-6a03-3bde7c80d685
md"""
## Krusell-Smith with All-in-One Deep Learning

Classroom-scale deep-learning solver for the Krusell-Smith (1998) economy.

**References:** Maliar, Maliar & Winant (2021), *Journal of Monetary Economics* 122, 76–101 (method); Krusell & Smith (1998), *Journal of Political Economy* 106(5) (model).

### What this notebook does

This notebook solves the canonical Krusell-Smith (1998) heterogeneous-agent model — a continuum of ex-ante identical, infinitely-lived agents facing idiosyncratic employment risk and aggregate productivity shocks — by stochastic gradient descent on the Euler-equation residual. The network parameterises the agent's savings policy as a function of \$(k, \varepsilon, \bar{K}, a)\$, and we train it directly against the structural Euler equation rather than iterating on a separate forecasting rule.

### Pedagogical goals

1. The DEQN template from Chapter 2 (loss = squared equilibrium residual, network outputs policies, SGD) transfers verbatim to a full heterogeneous-agent problem.
2. Reproduce the **approximate aggregation** insight of Krusell & Smith (1998): the mean of the wealth distribution is a nearly sufficient statistic for aggregate prices. We regress next-period mean capital on current mean capital and confirm \$R^2 > 0.99\$ in both aggregate states in a fuller run.
3. Verify that a single panel-trained policy network matches the canonical KS log-utility benchmark.

> **About this Julia preview.** The Lux translation preserves the model and the two-phase training design; it is sized for fast execution. The `smoke` budget runs only 8 Phase-A and 30 Phase-B iterations at constant learning rate, so the convergence targets below (percent-level Euler errors, \$R^2 > 0.99\$) are met only under `teaching`/`production`, not the checked-in smoke run — which exercises the full pipeline end to end.
"""

# ╔═╡ 36390a51-ba79-9cac-2e05-f8acfe2dcd45
md"""
### 1. Imports and reproducibility

The Julia stack is `DLEFJulia` (the course kit) plus `Lux` / `NNlib` (network), `Optimisers` (Adam), and `Random` / `Statistics`. Reproducibility is fixed in the next cell via `SEED = 0` and `rng_from_seed`, mirroring the Python notebook's explicit seeding.
"""

# ╔═╡ 22222222-0912-4222-8222-222222222222
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

# ╔═╡ 3f4dda9f-1d1a-6169-222e-b2db04c6ae20
md"""
### Run mode and hyperparameters

`RUN_MODE` / `SEED` and a `budgets` NamedTuple set the run size. Where the Python notebook fixes 2000 Phase-A and 3000 Phase-B iterations with cosine-annealed learning rate, the Julia preview dispatches everything on `RUN_MODE`:

| `RUN_MODE`   | n_agents | grid_size | phase_a | phase_b | hidden | lr |
|--------------|----------|-----------|---------|---------|--------|------|
| `smoke`      | 100      | 25        | 8       | 30      | 16     | 8e-4 |
| `teaching`   | 120      | 80        | 300     | 500     | 48     | 1e-3 |
| `production` | 1000     | 240       | 2000    | 3000    | 64     | 1e-3 |

The learning rate is held constant (Adam) rather than cosine-annealed. `run_mode_budget` selects the row; `rng_from_seed(SEED)` seeds sampling and the panel draws.
"""

# ╔═╡ 33333333-0912-4333-8333-333333333333
begin
    RUN_MODE = "smoke"
    SEED = 0
    budgets = (
        smoke = (n_agents = 100, grid_size = 25, phase_a = 8, phase_b = 30, hidden = 16, lr = 8e-4, max_grad_norm = 10.0, phi_floor = 1e-3),
        teaching = (n_agents = 120, grid_size = 80, phase_a = 300, phase_b = 500, hidden = 48, lr = 1e-3, max_grad_norm = 10.0, phi_floor = 1e-3),
        production = (n_agents = 1_000, grid_size = 240, phase_a = 2_000, phase_b = 3_000, hidden = 64, lr = 1e-3, max_grad_norm = 10.0, phi_floor = 1e-3),
    )
    hp = run_mode_budget(RUN_MODE; budgets)
    rng = rng_from_seed(SEED)
end

# ╔═╡ 793c8ce1-4d10-3df5-453c-e87f52f3f8e0
md"""
### 2. Model parameters

Standard Krusell–Smith (1998) *quarterly* calibration with **log utility** (CRRA with \$\sigma = 1\$), which admits the particularly clean Euler equation \$1/c = \beta\,\mathbb{E}[(1+r'-\delta)/c']\$.

| symbol | meaning | value |
|:-------|:--------|:-----:|
| \$\beta\$ | discount factor (quarterly) | 0.99 |
| \$\alpha\$ | capital share in Cobb–Douglas production | 0.36 |
| \$\delta\$ | depreciation rate (quarterly) | 0.025 |
| \$\bar{l}\$ | employed-agent labour endowment | \$1/(1-u_{\text{ss}})\$ |
| \$z_g, z_b\$ | aggregate TFP in good / bad state | 1.01, 0.99 |
| \$u_g, u_b\$ | unemployment rate in good / bad state | 0.04, 0.10 |

#### What does the deterministic steady state look like?

Ignoring shocks, log utility with \$\beta(1+r-\delta) = 1\$ gives
\$\$r^* = \beta^{-1} - 1 + \delta \approx 0.0351, \qquad \frac{K^*}{L^*} = \left(\frac{\alpha}{r^*}\right)^{\frac{1}{1-\alpha}} \approx 38, \qquad K^* = \frac{K^*}{L^*}\,L^* \approx 35\$\$
with \$L^* \approx 0.93\$ average labour. The level \$K^* \approx 35\$ — computed here as `K_SS` — is the ballpark \$\bar{K}\$ fluctuates around in the ergodic distribution, and it anchors the training bounds `K_MIN`/`K_MAX` and the capital grid.
"""

# ╔═╡ 44444444-0912-4444-8444-444444444444
begin
    beta = 0.99
    sigma = 1.0
    alpha = 0.36
    delta = 0.025

    z_levels = [0.99, 1.01]       # bad, good
    u_rates = [0.10, 0.04]        # bad, good
    u_ss = 0.07
    l_bar = 1 / (1 - u_ss)

    r_ss_det = 1 / beta - 1 + delta
    K_ss_det = (alpha / r_ss_det)^(1 / (1 - alpha))
    L_ss = 1 - u_ss
    K_SS = K_ss_det * L_ss

    K_MIN = 0.5 * K_SS
    K_MAX = 1.5 * K_SS
    k_floor = 1e-3
    k_ind_min = 0.2 * K_SS
    k_ind_max = 4.0 * K_SS
    capital_grid = collect(range(k_floor, 4.5 * K_SS; length = hp.grid_size))
end

# ╔═╡ 90c50ba7-45a8-aad6-1de7-93cab2738fa4
md"""
### 3. Shock transition matrix

Following Krusell & Smith (1998), the joint aggregate–idiosyncratic shock \$(a,\varepsilon) \in \{\text{bad},\text{good}\}\times\{\text{unempl.},\text{empl.}\}\$ evolves as a first-order Markov chain. The numbers are the standard KS calibration (Table 1 of KS 1998):

- Each aggregate state lasts on average \$8\$ quarters, \$P(a'=a\mid a) = 0.875\$.
- The conditional unemployment rate in state \$a'\$ equals \$u_{a'}\$.
- Unemployment spells last on average ~2.4 quarters in good times and ~4 quarters in bad times.

`P_agg` is the \$2\times 2\$ aggregate transition and `P_eps` the \$2\times2\times2\times2\$ conditional idiosyncratic transition; `validate_transition_matrix` and `transition_checks` confirm the rows sum to one.
"""

# ╔═╡ 55555555-0912-4555-8555-555555555555
begin
    P_agg = [
        0.875 0.125
        0.125 0.875
    ]
    validate_transition_matrix(P_agg)

    P_eps = zeros(Float64, 2, 2, 2, 2)
    P_eps[1, 1, :, :] .= [0.525 0.475; 0.350 0.650]  # bad -> bad
    P_eps[1, 2, :, :] .= [0.093 0.907; 0.038 0.962]  # bad -> good
    P_eps[2, 1, :, :] .= [0.840 0.160; 0.200 0.800]  # good -> bad
    P_eps[2, 2, :, :] .= [0.292 0.708; 0.042 0.958]  # good -> good

    transition_checks = (
        aggregate = maximum(abs.(sum(P_agg; dims = 2) .- 1)),
        idiosyncratic = maximum(abs.(sum(P_eps; dims = 4) .- 1)),
    )
end

# ╔═╡ a9b70fa5-06e9-7156-bebb-ce8317523228
md"""
### 5. Policy network

We parameterise the **consumption share** policy
\$\$\pi_\rho : (k, \varepsilon, \bar{K}, a) \mapsto \phi \in (0, 1),\$\$
with consumption \$c = \phi\,y\$ and savings \$k' = (1-\phi)\,y\$. The sigmoid output ensures \$0 \leq c \leq y\$, and the saved fraction is automatically non-negative, which **enforces the borrowing constraint by construction**.

#### Inputs to the network
- `k` — individual capital (log-normalised: \$\log(k / K^*)\$).
- `eps` — 2-dim one-hot for (unempl., empl.).
- `K` — aggregate capital (log-normalised).
- `a` — 2-dim one-hot for (bad, good).

Log-normalisation gives stable inputs over the whole training range and improves convergence markedly. The architecture is `make_mlp(6, (hidden, hidden), 1)` with a **Swish** activation (\$x\,\sigma(x)\$) and a sigmoid-mapped scalar output — the same family as every other DEQN / NAS network in this course.
"""

# ╔═╡ 66666666-0912-4666-8666-666666666666
begin
    swish_activation(x) = x .* NNlib.sigmoid.(x)
    model = make_mlp(6, (hp.hidden, hp.hidden), 1; activation = swish_activation)
    state = setup_training(rng_from_seed(SEED; offset = 1), model, Optimisers.Adam(hp.lr); parameter_type = Float64)
end

# ╔═╡ afefd6d0-27d3-e71a-dd6b-5c224884d18f
md"""
### 4. Prices and budget constraint · 6. Mapping the policy network to consumption and savings

Under Cobb–Douglas production the firm's first-order conditions give
\$\$w(\bar{K}, a) = (1-\alpha) z(a)\, \bar{K}^{\alpha} L^{-\alpha}, \qquad r(\bar{K}, a) = \alpha z(a)\, \bar{K}^{\alpha-1} L^{1-\alpha},\$\$
with \$L = 1 - u(a)\$ (`ks_prices`). The individual budget is
\$\$c + k' = \underbrace{w\,\bar{l}\,\varepsilon + (1 + r - \delta)\,k}_{=: y,\ \text{cash on hand}},\$\$
with \$\bar{l}\varepsilon = \bar{l}\$ when employed and \$0\$ when unemployed (the KS convention: unemployed agents get zero labour income, `cash_on_hand_panel`).

Given a batch of \$(k_i, \varepsilon_i)\$ states and an aggregate \$(\bar{K}, a)\$, `policy_features` builds the log-normalised, one-hot input and `apply_ks_policy` returns the consumption share \$\phi\$, consumption, and next-period capital for every agent. `transition_probabilities` supplies the joint \$(a',\varepsilon')\$ weights used by the Euler expectation below.
"""

# ╔═╡ 77777777-0912-4777-8777-777777777777
begin
    function ks_prices(K_agg, a_idx)
        K_safe = max(K_agg, k_floor)
        z = z_levels[a_idx]
        u = u_rates[a_idx]
        L = 1 - u
        w = (1 - alpha) * z * K_safe^alpha * L^(-alpha)
        r = alpha * z * K_safe^(alpha - 1) * L^(1 - alpha)
        return (w = w, r = r)
    end

    function cash_on_hand_panel(k, eps_idx, prices)
        eps_float = Float64.(eps_idx .- 1)
        return prices.w .* l_bar .* eps_float .+ (1 + prices.r - delta) .* k
    end

    function policy_features(k, eps_idx, K_agg, a_idx)
        n = length(k)
        log_k = log.(max.(k, k_floor) ./ K_SS)
        eps_unemployed = Float64.(eps_idx .== 1)
        eps_employed = Float64.(eps_idx .== 2)
        log_K = fill(log(max(K_agg, k_floor) / K_SS), n)
        a_bad = fill(a_idx == 1 ? 1.0 : 0.0, n)
        a_good = fill(a_idx == 2 ? 1.0 : 0.0, n)
        return vcat(
            reshape(log_k, 1, n),
            reshape(eps_unemployed, 1, n),
            reshape(eps_employed, 1, n),
            reshape(log_K, 1, n),
            reshape(a_bad, 1, n),
            reshape(a_good, 1, n),
        )
    end

    function apply_ks_policy(model, ps, st, k, eps_idx, K_agg, a_idx)
        prices = ks_prices(K_agg, a_idx)
        cash = cash_on_hand_panel(k, eps_idx, prices)
        raw, st_new = model(policy_features(k, eps_idx, K_agg, a_idx), ps, st)
        phi = clamp.(vec(NNlib.sigmoid.(raw)), hp.phi_floor, 1 - hp.phi_floor)
        consumption = phi .* cash
        savings = (1 .- phi) .* cash
        return (
            phi = phi,
            consumption = consumption,
            savings = savings,
            cash = cash,
            prices = prices,
        ), st_new
    end

    weighted_average(x, ::Nothing) = sum(x) / length(x)
    weighted_average(x, weights) = sum(weights .* x) / max(sum(weights), eps(Float64))

    function transition_probabilities(eps_idx, a_idx, a_next, eps_next)
        idio = [P_eps[a_idx, a_next, eps, eps_next] for eps in eps_idx]
        return P_agg[a_idx, a_next] .* idio
    end
end

# ╔═╡ a3aae79b-468a-073b-6ba4-630c9f95280d
md"""
### 7. Euler-equation residual (the loss)

For log utility, the Euler equation is
\$\$\frac{1}{c_t} \;=\; \beta\,\mathbb{E}_t\!\left[\frac{1 + r_{t+1} - \delta}{c_{t+1}}\right].\$\$
Because the aggregate and idiosyncratic shocks each take only two values, the expectation is a *deterministic* sum over the 4 \$(a',\varepsilon')\$ pairs, weighted by the joint transition probability. **No Monte Carlo needed** — this is the cleanest version of the MMW "all-in-one integration operator".

#### Relative Euler error
We report the **relative Euler error**
\$\$e^{\text{REE}}_i = \beta\,c_i\,\mathbb{E}_t\!\left[\frac{1+r'-\delta}{c'_i}\right] - 1.\$\$
A residual of \$10^{-3}\$ means the agent's Euler equation is violated by 0.1% of consumption, independent of units.

#### Important caveat: aggregate \$\bar{K}_{t+1}\$
Tomorrow's aggregate capital is endogenous — it equals the mean of \$k'_i\$ across agents. The all-in-one formulation takes \$\bar{K}' = \frac{1}{N}\sum_i k'_i\$ from the network's own savings decisions (`ks_residual` computes `K_next` inside the loss), so the aggregate law of motion is automatically consistent with the individual policy. No separate law of motion is fitted.
"""

# ╔═╡ 88888888-0912-4888-8888-888888888888
begin
    function ks_residual(model, ps, st, batch)
        policy, st_work = apply_ks_policy(model, ps, st, batch.k, batch.eps, batch.K, batch.a)
        K_next = weighted_average(policy.savings, batch.weights)

        expected_marginal_value = zeros(eltype(policy.consumption), length(policy.consumption))
        for a_next in 1:2, eps_next in 1:2
            eps_next_vec = fill(eps_next, length(batch.eps))
            next_policy, st_work = apply_ks_policy(model, ps, st_work, policy.savings, eps_next_vec, K_next, a_next)
            continuation = (1 + next_policy.prices.r - delta) ./ next_policy.consumption
            weights = transition_probabilities(batch.eps, batch.a, a_next, eps_next)
            expected_marginal_value = expected_marginal_value .+ weights .* continuation
        end

        relative_euler = beta .* policy.consumption .* expected_marginal_value .- 1
        euler_loss = weighted_average(relative_euler .^ 2, batch.weights)
        current_capital_gap = weighted_average(batch.k, batch.weights) - batch.K
        next_capital_gap = weighted_average(policy.savings, batch.weights) - K_next
        loss = euler_loss

        return (
            loss = loss,
            euler = relative_euler,
            euler_loss = euler_loss,
            current_capital_gap = current_capital_gap,
            next_capital_gap = next_capital_gap,
            consumption = policy.consumption,
            savings = policy.savings,
            phi = policy.phi,
            K_next = K_next,
        ), st_work
    end

    ks_loss(model, ps, st, batch) = begin
        pieces, st_new = ks_residual(model, ps, st, batch)
        return pieces.loss, st_new
    end
end

# ╔═╡ 5a7b8a53-5cdc-a395-1667-3e974adafae1
md"""
### 8. Panel transition

Between training steps in Phase B, we advance the panel one period: draw the next aggregate shock and, conditional on it, the next idiosyncratic shock for each agent; individual capital follows the network's savings policy deterministically (`advance_panel`, using `draw_categorical` on `P_agg` / `P_eps`).

This cell also defines the exogenous state sampler `sample_state_batch` (Phase A) and the initial panel `initial_panel`. The Young-histogram helpers here (`initial_ks_histogram`, `histogram_batch`, `ks_young_step`) are **retained for distribution diagnostics only** — the Phase B parity path below trains and reports from the stochastic Monte Carlo panel, matching the Python ground truth.
"""

# ╔═╡ 99999999-0912-4999-8999-999999999999
begin
    function sample_state_batch(rng, n)
        k = k_ind_min .+ (k_ind_max - k_ind_min) .* rand(rng, n)
        eps_idx = ifelse.(rand(rng, n) .< u_ss, 1, 2)
        K_agg = K_MIN + (K_MAX - K_MIN) * rand(rng)
        a_idx = rand(rng) < 0.5 ? 1 : 2
        return (k = k, eps = eps_idx, K = K_agg, a = a_idx, weights = nothing)
    end

    # Optional Young histogram helpers, retained for distribution diagnostics only.
    # The Phase B parity path below trains and reports from the stochastic panel.
    function initial_ks_histogram()
        hist = zeros(Float64, 2, length(capital_grid))
        hist[1, :] .= redistribute_mass(capital_grid, K_SS, u_ss)
        hist[2, :] .= redistribute_mass(capital_grid, K_SS, 1 - u_ss)
        return hist
    end

    function histogram_batch(hist)
        n_idio, n_grid = size(hist)
        k_mat = repeat(reshape(capital_grid, 1, n_grid), n_idio, 1)
        eps_mat = repeat(reshape(collect(1:n_idio), n_idio, 1), 1, n_grid)
        return (k = vec(k_mat), eps = vec(eps_mat), weights = vec(hist))
    end

    function ks_policy_grid(model, ps, st, hist, a_idx)
        batch = histogram_batch(hist)
        K_agg = young_mean(capital_grid, hist)
        policy, st_new = apply_ks_policy(model, ps, st, batch.k, batch.eps, K_agg, a_idx)
        return (
            savings = reshape(policy.savings, size(hist)),
            consumption = reshape(policy.consumption, size(hist)),
            phi = reshape(policy.phi, size(hist)),
            prices = policy.prices,
        ), st_new
    end

    function ks_young_step(hist, savings, a_idx, a_next)
        transition = P_eps[a_idx, a_next, :, :]
        validate_transition_matrix(transition)
        return young_step(capital_grid, hist, savings; transition, clip = true)
    end

    function draw_categorical(rng, probabilities)
        u = rand(rng)
        cumulative = 0.0
        for i in eachindex(probabilities)
            cumulative += probabilities[i]
            u <= cumulative && return i
        end
        return length(probabilities)
    end

    function initial_panel(rng)
        k = fill(K_SS, hp.n_agents)
        eps_idx = ifelse.(rand(rng, hp.n_agents) .< u_ss, 1, 2)
        return (k = k, eps = eps_idx, K = mean(k), a = 2, weights = nothing)
    end

    function advance_panel(model, ps, st, rng, k, eps_idx, a_idx)
        K_agg = mean(k)
        policy, _ = apply_ks_policy(model, ps, st, k, eps_idx, K_agg, a_idx)
        k_next = copy(policy.savings)
        a_next = draw_categorical(rng, @view P_agg[a_idx, :])
        eps_next = similar(eps_idx)
        for i in eachindex(eps_idx)
            eps_next[i] = draw_categorical(rng, @view P_eps[a_idx, a_next, eps_idx[i], :])
        end
        return (k = k_next, eps = eps_next, K = mean(k_next), a = a_next, weights = nothing)
    end
end

# ╔═╡ b18099f6-a8ba-3981-59c7-1036ff4cfb51
md"""
### 9. Training loop — Phase A (exogenous sampling)

The solve runs in two phases.

**Phase A — exogenous sampling.** Draw random states from the bounded training domain and take Adam steps on the mean-squared relative Euler error. The aim is a policy that is approximately correct *everywhere* in the domain, so that Phase B's simulated panel does not diverge. This cell runs `hp.phase_a` iterations, logging loss, max relative Euler error, and the aggregate-capital gaps into the shared history arrays.
"""

# ╔═╡ aaaaaaaa-0912-4aaa-8aaa-aaaaaaaaaaaa
begin
    initial_batch = sample_state_batch(rng, hp.n_agents)
    initial_loss = loss_value(state, ks_loss, initial_batch)

    n_total = hp.phase_a + hp.phase_b
    loss_history = fill(NaN, n_total)
    ree_max_history = fill(NaN, n_total)
    K_history = fill(NaN, n_total)
    K_next_history = fill(NaN, n_total)
    a_history = fill(0, n_total)
    phase_history = fill(-1, n_total)

    history_log = NamedTuple[]
    for step in 1:hp.phase_a
        batch = sample_state_batch(rng, hp.n_agents)
        metrics = train_step!(state, ks_loss, batch; max_grad_norm = hp.max_grad_norm)
        pieces, _ = ks_residual(state.model, state.ps, state.st, batch)
        loss_history[step] = metrics.loss
        ree_max_history[step] = maximum(abs.(pieces.euler))
        K_history[step] = batch.K
        K_next_history[step] = pieces.K_next
        a_history[step] = batch.a
        phase_history[step] = 0
        append_metric!(
            history_log;
            step = step,
            phase = 0,
            loss = metrics.loss,
            ree_max = maximum(abs.(pieces.euler)),
            K = batch.K,
            a = batch.a,
            current_capital_gap = pieces.current_capital_gap,
            next_capital_gap = pieces.next_capital_gap,
            mass = 1.0,
        )
    end
end

# ╔═╡ 2cedce35-6fba-1839-7ad8-5b91e025262e
md"""
### 9 (cont.). Training loop — Phase B (simulated-panel sampling)

**Phase B — simulated-panel sampling.** Maintain a running panel of `n_agents` agents; every iteration evaluates the Euler residual on the *current* panel and then advances the panel by one period under the updated policy (`advance_panel`). The training distribution therefore concentrates on the model's ergodic set. The end-of-loop panel and per-step `panel_log` (aggregate state, \$\bar{K}\$, network-implied \$\bar{K}'\$, simulated next \$\bar{K}\$, and consistency gaps) are carried forward for the diagnostics. This running-panel Phase B is the core semantics preserved from the Python notebook.

*(The Python notebook additionally cosine-anneals the learning rate from \$10^{-3}\$ to \$10^{-5}\$; the Julia preview holds it constant.)*
"""

# ╔═╡ bbbbbbbb-0912-4bbb-8bbb-bbbbbbbbbbbb
begin
    phase_b_result = let panel_local = initial_panel(rng), panel_log_local = NamedTuple[]
        for phase_step in 1:hp.phase_b
            batch = panel_local

            metrics = train_step!(state, ks_loss, batch; max_grad_norm = hp.max_grad_norm)
            pieces, _ = ks_residual(state.model, state.ps, state.st, batch)
            panel_next = advance_panel(state.model, state.ps, state.st, rng, batch.k, batch.eps, batch.a)
            idx = hp.phase_a + phase_step

            loss_history[idx] = metrics.loss
            ree_max_history[idx] = maximum(abs.(pieces.euler))
            K_history[idx] = batch.K
            K_next_history[idx] = pieces.K_next
            a_history[idx] = batch.a
            phase_history[idx] = 1

            append_metric!(
                panel_log_local;
                phase_step = phase_step,
                a = batch.a,
                a_next = panel_next.a,
                K = batch.K,
                K_next = pieces.K_next,
                simulated_K_next = panel_next.K,
                panel_mean_gap = mean(batch.k) - batch.K,
                policy_mean_gap = pieces.K_next - mean(pieces.savings),
            )
            append_metric!(
                history_log;
                step = idx,
                phase = 1,
                loss = metrics.loss,
                ree_max = maximum(abs.(pieces.euler)),
                K = batch.K,
                a = batch.a,
                current_capital_gap = pieces.current_capital_gap,
                next_capital_gap = pieces.next_capital_gap,
                mass = 1.0,
            )

            panel_local = panel_next
        end
        (panel = panel_local, panel_log = panel_log_local)
    end
    panel = phase_b_result.panel
    panel_log = phase_b_result.panel_log
end

# ╔═╡ 163c6d62-d85d-8c58-3d5f-95156e0c2322
md"""
### 10. Convergence diagnostics

This cell computes everything the Python notebook plots in section 10:

- **10.1 Loss and Euler errors.** `phase_b_tail` summarises mean loss and mean/median max relative Euler error over the last iterations.
- **10.2 Ergodic behaviour of \$\bar{K}\$.** `state_moments` reports the mean and spread of aggregate capital by aggregate state on the post-burn-in Phase-B trajectory — expected to sit slightly higher in good times and lower in bad times.
- **10.3 Krusell–Smith approximate aggregation.** `fit_law_of_motion` runs OLS of \$\log \bar{K}_{t+1}\$ on \$\log \bar{K}_t\$ separately for each aggregate state, reporting the \$R^2\$ of the log-linear forecasting rule \$\log \bar{K}_{t+1} = A(a_t) + B(a_t)\log \bar{K}_t\$ — which approaches \$1\$ when the policy has converged, without any separate forecasting-rule fit.
- **10.4 Learned savings policy.** `final_policy` evaluates \$k'(k,\varepsilon,\bar{K},a)\$ for the final panel; a converged policy is monotone in \$k\$, has employed agents saving more than unemployed, and crosses the \$45°\$ line so the ergodic distribution is interior.
"""

# ╔═╡ cccccccc-0912-4ccc-8ccc-cccccccccccc
begin
    function fit_law_of_motion(K_seq, a_seq, a_idx)
        if length(K_seq) < 2
            return (slope = NaN, intercept = NaN, r2 = NaN, n = 0)
        end
        logK_t = log.(K_seq[1:(end - 1)])
        logK_tp1 = log.(K_seq[2:end])
        a_t = a_seq[1:(end - 1)]
        keep = findall((a_t .== a_idx) .& isfinite.(logK_t) .& isfinite.(logK_tp1))
        n = length(keep)
        if n < 2
            return (slope = NaN, intercept = NaN, r2 = NaN, n = n)
        end
        x = logK_t[keep]
        y = logK_tp1[keep]
        xbar = mean(x)
        ybar = mean(y)
        ssx = sum((x .- xbar) .^ 2)
        if ssx <= eps(Float64)
            return (slope = NaN, intercept = ybar, r2 = NaN, n = n)
        end
        slope = sum((x .- xbar) .* (y .- ybar)) / ssx
        intercept = ybar - slope * xbar
        y_hat = intercept .+ slope .* x
        ss_res = sum((y .- y_hat) .^ 2)
        ss_tot = sum((y .- ybar) .^ 2)
        r2 = ss_tot <= eps(Float64) ? 1.0 : 1 - ss_res / ss_tot
        return (slope = slope, intercept = intercept, r2 = r2, n = n)
    end

    function state_moments(K_seq, a_seq, a_idx)
        vals = K_seq[a_seq .== a_idx]
        n = length(vals)
        if n == 0
            return (mean = NaN, std = NaN, n = 0)
        end
        return (mean = mean(vals), std = n == 1 ? 0.0 : std(vals), n = n)
    end

    phase_b_idx = findall(phase_history .== 1)
    K_B = K_history[phase_b_idx]
    a_B = a_history[phase_b_idx]
    tail_n = min(200, length(phase_b_idx))
    tail_idx = phase_b_idx[(end - tail_n + 1):end]
    burn = min(500, max(0, length(K_B) ÷ 5))
    post_burn_range = (burn + 1):length(K_B)
    K_post_burn = K_B[post_burn_range]
    a_post_burn = a_B[post_burn_range]

    final_batch = panel
    final_diagnostics, _ = ks_residual(state.model, state.ps, state.st, final_batch)
    final_policy, _ = apply_ks_policy(state.model, state.ps, state.st, final_batch.k, final_batch.eps, final_batch.K, final_batch.a)
    law_of_motion = (
        bad = fit_law_of_motion(K_post_burn, a_post_burn, 1),
        good = fit_law_of_motion(K_post_burn, a_post_burn, 2),
    )
    phase_b_tail = (
        n = tail_n,
        mean_loss = mean(loss_history[tail_idx]),
        mean_ree_max = mean(ree_max_history[tail_idx]),
        median_ree_max = median(ree_max_history[tail_idx]),
    )
    aggregate_capital_moments = (
        bad = state_moments(K_post_burn, a_post_burn, 1),
        good = state_moments(K_post_burn, a_post_burn, 2),
    )
end

# ╔═╡ 95db4d12-5a08-b0e8-1737-0e1cb274900c
md"""
### 11. Summary and validation checklist

**Success criteria** for a classroom-scale (`teaching`/`production`) solve:

- Median max \$|\text{REE}|\$ over the last Phase-B iterations below \$5\times 10^{-2}\$ (not the 1% MMW target — that needs longer training and larger networks — but economically meaningful).
- Mean loss over the last Phase-B iterations below \$10^{-3}\$.
- \$\bar{K}\$ trajectory stationary and clearly split by aggregate state.
- Log-linear KS forecasting rule fitted ex post has \$R^2 > 0.99\$ in both aggregate states.
- Savings policy monotone in \$k\$, crossing the \$45°\$ line, with employed \$>\$ unemployed.

#### If something doesn't converge
Raise `phase_a` (more uniform warm-up), raise `n_agents` (less Monte Carlo noise in the per-iteration residual), lower `lr` if Phase B oscillates, or widen the initial panel spread.

#### Next steps
- Histogram-DEQN alternative: `11_Continuum_of_Agents_DEQN.ipynb` (Azinovic–Gaegauf–Scheidegger 2022, Appendix A.5).
- Production-scale all-in-one DL for KS: [marcmaliar/deep-learning-euler-method-krusell-smith](https://github.com/marcmaliar/deep-learning-euler-method-krusell-smith).
- DeepHAM (learned generalized moments): Han, Yang & E (*Quantitative Economics*).

The cell below returns the machine-checkable diagnostic NamedTuple for this notebook's run — the two-phase history, the Phase-B panel log, the fitted laws of motion, and the policy summaries.
"""

# ╔═╡ dddddddd-0912-4ddd-8ddd-dddddddddddd
(
    run_mode = RUN_MODE,
    seed = SEED,
    phase_a_steps = hp.phase_a,
    phase_b_steps = hp.phase_b,
    initial_loss = initial_loss,
    final_loss = final_diagnostics.loss,
    euler = residual_summary(final_diagnostics.euler),
    current_capital_gap = final_diagnostics.current_capital_gap,
    next_capital_gap = final_diagnostics.next_capital_gap,
    final_panel_K = final_batch.K,
    final_panel_K_next = final_diagnostics.K_next,
    final_panel_a = final_batch.a,
    final_panel_eps_counts = (unemployed = count(==(1), final_batch.eps), employed = count(==(2), final_batch.eps)),
    phase_b_tail = phase_b_tail,
    transition_checks = transition_checks,
    law_of_motion = law_of_motion,
    aggregate_capital_moments = aggregate_capital_moments,
    phase_b_burn = burn,
    history = (
        loss = loss_history,
        ree_max = ree_max_history,
        K = K_history,
        K_next = K_next_history,
        aggregate_state = a_history,
        phase = phase_history,
    ),
    panel_log = panel_log,
    policy_savings_minmax = extrema(final_policy.savings),
    policy_consumption_minmax = extrema(final_policy.consumption),
    consumption_share_minmax = extrema(final_policy.phi),
    borrowing_slack_min = minimum(final_policy.savings),
)

# ╔═╡ Cell order:
# ╟─11111111-0912-4111-8111-111111111111
# ╟─8ec9815f-4ef5-33ac-3b69-855feae9af13
# ╟─2668ff05-75a1-3d9e-6a03-3bde7c80d685
# ╟─36390a51-ba79-9cac-2e05-f8acfe2dcd45
# ╠═22222222-0912-4222-8222-222222222222
# ╟─3f4dda9f-1d1a-6169-222e-b2db04c6ae20
# ╠═33333333-0912-4333-8333-333333333333
# ╟─793c8ce1-4d10-3df5-453c-e87f52f3f8e0
# ╠═44444444-0912-4444-8444-444444444444
# ╟─90c50ba7-45a8-aad6-1de7-93cab2738fa4
# ╠═55555555-0912-4555-8555-555555555555
# ╟─a9b70fa5-06e9-7156-bebb-ce8317523228
# ╠═66666666-0912-4666-8666-666666666666
# ╟─afefd6d0-27d3-e71a-dd6b-5c224884d18f
# ╠═77777777-0912-4777-8777-777777777777
# ╟─a3aae79b-468a-073b-6ba4-630c9f95280d
# ╠═88888888-0912-4888-8888-888888888888
# ╟─5a7b8a53-5cdc-a395-1667-3e974adafae1
# ╠═99999999-0912-4999-8999-999999999999
# ╟─b18099f6-a8ba-3981-59c7-1036ff4cfb51
# ╠═aaaaaaaa-0912-4aaa-8aaa-aaaaaaaaaaaa
# ╟─2cedce35-6fba-1839-7ad8-5b91e025262e
# ╠═bbbbbbbb-0912-4bbb-8bbb-bbbbbbbbbbbb
# ╟─163c6d62-d85d-8c58-3d5f-95156e0c2322
# ╠═cccccccc-0912-4ccc-8ccc-cccccccccccc
# ╟─95db4d12-5a08-b0e8-1737-0e1cb274900c
# ╠═dddddddd-0912-4ddd-8ddd-dddddddddddd
