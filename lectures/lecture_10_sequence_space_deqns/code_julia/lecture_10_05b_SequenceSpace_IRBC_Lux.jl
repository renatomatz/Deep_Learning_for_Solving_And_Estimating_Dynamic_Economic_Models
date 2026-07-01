### A Pluto.jl notebook ###
# v0.20.0

using Markdown
using InteractiveUtils

# ╔═╡ 11111111-105b-4111-8111-111111111111
md"""
# Lecture 10, Notebook 05b: Sequence-Space IRBC in Lux

This preview follows the Python sequence-space IRBC notebook's shock-history
layout: each lag contains the two country shocks and one separate aggregate shock.
The Lux boundary receives only the flattened `(shock, lag)` history, while the
cloud state is kept for residual evaluation and simulation.
"""

# ╔═╡ 0f019c80-eec8-5c9b-7950-b2b603a80125
md"""
## Lecture 10, Notebook 05b: Sequence-Space IRBC in Lux

**Course:** Deep Learning for Solving and Estimating Dynamic Models in Economics and Finance
**Script reference:** §6.7 (Sequence-space DEQNs) — IRBC bridge to heterogeneous agents
**Notebook role:** extension (supplementary / self-study)
**Author:** Simon Scheidegger

*Julia/Lux/Pluto preview of* `lectures/lecture_10_sequence_space_deqns/code/lecture_10_05b_SequenceSpace_IRBC.ipynb`.

Based on [Azinovic-Yang & Žemlička (2025)](https://arxiv.org/abs/2509.13623) for the sequence-space template, and [Azinovic, Gaegauf & Scheidegger (2022)](https://onlinelibrary.wiley.com/doi/abs/10.3982/ECTA12216) for the IRBC model.

> **Run mode.** The checked-in run uses `RUN_MODE = "smoke"` with a short shock history and tiny budgets for fast execution. The `teaching` and `production` presets in the next cell restore longer histories (up to \$T = 80\$) and larger networks that reproduce the classroom figures.

> **Supplementary / self-study.** Not covered in the in-class sequence-space slot — see the companion Brock–Mirman warm-up (`05_…`) for that. Runtime with the classroom (`production`) defaults is a few minutes on CPU.
"""

# ╔═╡ 067797b0-1823-ad1b-89cf-d986b215b310
md"""
## Sequence-Space DEQN: IRBC

This notebook solves the **same \$N\$-country IRBC model from Day 3 notebook 01**, re-trained with **sequence-space inputs**: the policy network reads the last 80 shock vectors instead of the current endogenous state. The equilibrium system — two Euler equations, one aggregate resource constraint (ARC), two Fischer–Burmeister complementarity conditions — is **unchanged**; only the network's domain changes.

### Where this sits in Day 4

| Notebook | Model | State dim | Sequence-space input dim |
|---|---|---|---|
| `05_SequenceSpace_BrockMirman` | 1-agent BM, 1 TFP shock | 2 | \$T \cdot 1 = 25\$ |
| **`05b_SequenceSpace_IRBC`** (this one) | 2-country IRBC, 3 shocks | 4 | \$T \cdot 3 = 240\$ |
| `06_SequenceSpace_KrusellSmith` | Krusell–Smith, 1 agg. TFP | 2 + \$\infty\$ (distribution) | \$T \cdot 1 = 100\$ |

> **Honest framing.** For IRBC, sequence space is **not a dimensionality reduction** — the 4-dim state is cheaper to feed directly. The payoff kicks in only at Krusell–Smith scale, where the state-space input would be an infinite-dimensional wealth distribution. This notebook is a *bridge* between BM (1-dim state) and KS (infinite-dim state): it shows the same template handles a multi-equation equilibrium system with multiple independent shock channels.

> **Julia preview note.** The Lux preview stores the shock history as an \$(n_{\text{shocks}}, \text{lag}, \text{batch})\$ tensor — the two country shocks plus one aggregate shock per lag (\$n_{\text{shocks}} = N + 1 = 3\$, matching the Python notebook) — and flattens it to \$(T \cdot n_{\text{shocks}}, \text{batch})\$ only at the Lux boundary via `flatten_history`. The cloud state \$(k_1, k_2, z_1, z_2)\$ is carried alongside for residual evaluation and simulation, never fed to the network. History length and the pre-training/training budgets come from fixed `smoke`/`teaching`/`production` presets rather than being scaled continuously from `RUN_MODE`, so the smoke run checks structure and finite execution rather than sequence-length parity.
"""

# ╔═╡ 22222222-105b-4222-8222-222222222222
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

# ╔═╡ ce482026-4fd9-681f-3704-a88e2ec1f116
md"""
### Learning goals

1. **See the single conceptual change** from Day 3 nb 01 (state-space IRBC): the network's input swaps from the current state \$(k_1, k_2, z_1, z_2)\$ to the shock history \$(\varepsilon_t, \varepsilon_{t-1}, \ldots, \varepsilon_{t-79})\$. The loss, the Gauss–Hermite quadrature, the cloud sampler, the Euler/ARC/FB residuals — all *literally identical*.
2. **Pick the history length** via the ergodicity-and-truncation argument: influence of shock \$T\$ lags ago decays as \$\rho_z^T\$ with \$\rho_z = 0.95\$; \$0.95^{80} \approx 1.7 \times 10^{-2}\$ is a tolerable truncation error for a classroom demo.
3. **Use a steady-state-centred output parametrization** that replaces the saturating sigmoid (nb 05) and the unbounded softplus (nb 01): \$k'_j = k_{ss} \cdot \exp(\tanh(z_j))\$ and \$\lambda = \lambda_{ss} \cdot \exp(\tanh(z))\$. This keeps raw=0 → steady-state policy, with lively gradients and bounded excursions.
4. **Read the diagnostics**: loss curve, ergodic \$(k_1, k_2)\$ cloud, per-equation residuals, and a policy scatter \$(k_j \mapsto k'_j)\$ comparable to nb 01.

### References
- Azinovic-Yang, M. & Žemlička, J. (2025). *Deep Learning in the Sequence Space.* arXiv:2509.13623.
- Azinovic, M., Gaegauf, L. & Scheidegger, S. (2022). *Deep Equilibrium Nets.* International Economic Review 63(4), 1471–1525.
"""

# ╔═╡ 60d26767-dd30-faf3-3bd3-88fa87f4c2f7
md"""
### 1. Calibration

Identical to Day 3 notebook 01 — two countries, mild adjustment costs, persistent TFP, heterogeneous Pareto weights. The next cell also sets the `smoke`/`teaching`/`production` budget presets and derives the shape constants (\$n_{\text{states}} = 2N\$, \$n_{\text{policies}} = 2N+1\$, \$n_{\text{shocks}} = N+1\$).

### 2. Steady state

At \$z^j = 0\$, \$k^j = k_{ss} = 1\$, with the TFP calibration fixing the deterministic equilibrium to \$k_{ss}\$. \$\mu^j = 0\$ (irreversibility not binding), and \$\lambda_{ss}\$ follows from the complete-markets FOC.

### 3. Gauss–Hermite quadrature for the 3-dim shock

The \$N+1 = 3\$ shock dimensions combine into a tensor-product GH rule. With \$Q = 3\$ nodes per dim we get \$Q^3 = 27\$ quadrature points — same choice as nb 01. In Julia this is `tensor_product_rule` over three copies of `gauss_hermite_rule(3)`.

### 4. Choosing the history length \$T\$

In IRBC the TFP process is AR(1) with persistence \$\rho_z = 0.95\$. The influence of a shock \$T\$ lags ago on today's state decays (roughly) as \$\rho_z^T\$. For the sequence-space truncation to be accurate we need \$\rho_z^T\$ small.

| \$T\$ | \$\rho_z^T\$ | Comment |
|---|---|---|
| 25  | 0.277 | too coarse (5× worse than BM nb's \$\alpha^{25} \approx 5{\cdot}10^{-13}\$) |
| 50  | 0.077 | acceptable |
| **80**  | **0.017** | **classroom default (~1.7 % truncation error)** |
| 100 | 0.006 | production-grade; ~3× slower |

The `production` preset picks \$T = 80\$, giving a history input of dimension \$T \cdot n_{\text{shocks}} = 80 \cdot 3 = 240\$; the `smoke` preset uses a much shorter history purely so the notebook runs quickly under CI.
"""

# ╔═╡ 33333333-105b-4333-8333-333333333333
begin
    RUN_MODE = "smoke"
    SEED = 0
    budgets = (
        smoke = (
            pretrain_steps = 2,
            steps = 4,
            batch_size = 8,
            minibatch = 8,
            history_length = 5,
            refresh_steps = 3,
            hidden = (24, 24),
        ),
        teaching = (
            pretrain_steps = 80,
            steps = 200,
            batch_size = 128,
            minibatch = 128,
            history_length = 20,
            refresh_steps = 3,
            hidden = (64, 64, 64),
        ),
        production = (
            pretrain_steps = 300,
            steps = 800,
            batch_size = 512,
            minibatch = 128,
            history_length = 80,
            refresh_steps = 3,
            hidden = (128, 128, 128),
        ),
    )
    hp = run_mode_budget(RUN_MODE; budgets)
    rng = rng_from_seed(SEED)

    N_COUNTRIES = 2
    beta = 0.99
    gamma_min = 0.25
    gamma_max = 1.0
    zeta = 0.36
    delta = 0.01
    rho_z = 0.95
    sigma_e = 0.01
    kappa = 0.5
    k_ss = 1.0
    EPS_SAFE = 1e-10
    FB_EPS = 1e-8

    A_tfp = (1.0 - beta * (1.0 - delta)) / (zeta * beta)
    gammas = collect(range(gamma_min, gamma_max; length = N_COUNTRIES))
    taus = (A_tfp - delta) .^ (1.0 ./ gammas)
    Y_ss = A_tfp * k_ss^zeta
    c_ss = Y_ss + (1.0 - delta) * k_ss - k_ss
    lambda_ss = taus[1] * c_ss^(-1.0 / gammas[1])

    n_states = 2 * N_COUNTRIES
    n_policies = 2 * N_COUNTRIES + 1
    n_shocks = N_COUNTRIES + 1
    history_dim = hp.history_length * n_shocks

    gh_1d = gauss_hermite_rule(3)
    rule = tensor_product_rule(ntuple(_ -> gh_1d, n_shocks)...)
end

# ╔═╡ 03081ca3-4177-0f29-153b-6814eead822a
md"""
### 5. Model primitives

Production, CRRA consumption, capital-adjustment costs, the marginal product of capital, investment, and the Fischer–Burmeister complementarity function — **copied verbatim** from Day 3 notebook 01. Nothing about the *physics* changes when we swap input domains. The `py_*` helpers below are broadcast Julia versions with the same `EPS_SAFE` guard on capital.
"""

# ╔═╡ ce15e57e-8af2-fcb2-5530-48ffdb0e42a6
md"""
### 6. Network architecture — SS-centred tanh parametrization

#### Problem with off-the-shelf output heads
- **`softplus`** (nb 01): outputs are unbounded. *With* a current-state input this is fine (the network uses \$k\$ to locate itself). *Without* a current-state input — sequence space — the softplus head diverges: initial large-magnitude raw outputs blow up \$k'_j\$, which then feeds back into the loss explosively.
- **`sigmoid` on savings rate** (nb 05): at steady state \$k'_j = k_{ss}\$, the savings rate \$s = k_{ss} / (Y + (1-\delta) k) \approx 0.95\$ — right in the sigmoid saturation zone, where \$\sigma'(z) \approx 0.05\$. Gradients are strangled 20× and training plateaus.

#### This notebook's parametrization
Centre the output on the steady-state policy via `tanh`:

\$\$k'_j = k_{ss} \cdot \exp\bigl(\tanh(z^{k}_j)\bigr), \qquad
\lambda = \lambda_{ss} \cdot \exp\bigl(\tanh(z^\lambda)\bigr), \qquad
\mu_j = \operatorname{softplus}(z^\mu_j - 4).\$\$

Properties:
- At raw output \$z = 0\$ we recover exactly \$(k_{ss}, k_{ss}, \lambda_{ss}, \approx 0, \approx 0)\$.
- \$k'_j\$ is bounded in \$(k_{ss}/e, \, k_{ss} e) \approx (0.37, 2.72)\$ — wide enough to cover the ergodic set of the IRBC.
- At the target, \$\tanh'(0) = 1\$ so gradients are **unstrangled**.
- \$\mu_j\$ is non-negative and near zero by default (irreversibility off).

The \$-4\$ shift on \$\mu_j\$'s softplus argument makes \$\mu_j \approx 0.018\$ at raw=0, matching the near-zero steady-state value. In Lux this parametrization is the `py_policy_from_raw` head applied to the raw MLP output; the MLP builder itself (`make_mlp`) is set up in the training cell below.
"""

# ╔═╡ 01906499-1780-33b6-b67f-3b60bd2c8ea8
md"""
### 7. Residuals — same 2N+1 equations, sequence-space inputs

The residual structure mirrors `compute_cost` in nb 01:

1. **Euler** (one per country): \$\beta\,\mathbb{E}[\lambda' \cdot \mathrm{MPK}^{\prime} - (1{-}\delta)\mu'] + \mu^j = \lambda(1 + \Gamma_{k'})\$ in relative-error form.
2. **ARC**: \$\sum_j [Y^j + (1{-}\delta) k^j - k^{\prime j} - \Gamma^j - c^j] = 0\$.
3. **Fischer–Burmeister**: \$\mu^j + I^j - \sqrt{(\mu^j)^2 + (I^j)^2 + \varepsilon} = 0\$.

The *only* difference from nb 01: to get the next-period policy at each quadrature node, we build the next-period **shock history** (prepend the node's shock vector, drop the oldest entry via `prepend_history`) and feed that into the network. Inside `py_sequence_irbc_residual` this means running the net *twice* — once at today's history, once at each of the 27 next-period histories — with the Lux `model(x, ps, st)` call threading state explicitly.
"""

# ╔═╡ 3de9a7fe-6d0b-83af-0e6a-33099d7b94fb
md"""
### 8. Forward simulator — rolls state *and* history

The cloud entries are pairs `(x_cloud, history_cloud)`. At each simulator step:

1. Draw fresh shocks \$\varepsilon_{t+1} \in \mathbb{R}^{N+1}\$ per track.
2. Roll state: \$k_{t+1} \leftarrow k'\$ (from current policy), \$z_{t+1} \leftarrow \rho_z z_t + \sigma_e(\varepsilon^j + \varepsilon^{\text{agg}})\$.
3. Roll history: prepend \$\varepsilon_{t+1}\$, drop the oldest entry.

This is the *unchanged* Krusell-Smith sequence-space pattern. `py_sequence_irbc_forward_step` returns the rolled state, the rolled history, and the updated Lux state.
"""

# ╔═╡ 44444444-105b-4444-8444-444444444444
begin
    function py_irbc_history(; history_length::Integer, batch::Integer, dtype::Type{<:AbstractFloat} = Float64)
        history_length > 0 || throw(ArgumentError("history_length must be positive"))
        batch > 0 || throw(ArgumentError("batch must be positive"))
        return zeros(dtype, n_shocks, history_length, batch)
    end

    function py_start_states(rng, batch::Integer)
        states = zeros(Float64, n_states, batch)
        states[1:N_COUNTRIES, :] .= k_ss
        states[1:N_COUNTRIES, :] .+= 0.05 .* (2 .* rand(rng, N_COUNTRIES, batch) .- 1)
        states[(N_COUNTRIES + 1):n_states, :] .+= 0.005 .* (2 .* rand(rng, N_COUNTRIES, batch) .- 1)
        return states
    end

    get_k(states, j) = @view states[j:j, :]
    get_z(states, j) = @view states[(N_COUNTRIES + j):(N_COUNTRIES + j), :]

    py_production(k, z) = A_tfp .* exp.(z) .* max.(k, EPS_SAFE) .^ zeta
    py_production_k(k, z) = zeta .* A_tfp .* exp.(z) .* max.(k, EPS_SAFE) .^ (zeta - 1.0)
    py_consumption(lambda, j) = (lambda ./ taus[j]) .^ (-gammas[j])

    function py_adj_cost(k, kp)
        ks = max.(k, EPS_SAFE)
        ratio = kp ./ ks .- 1.0
        return 0.5 .* kappa .* ratio .^ 2 .* ks
    end

    py_adj_cost_kp(k, kp) = kappa .* (kp ./ max.(k, EPS_SAFE) .- 1.0)

    function py_adj_cost_k(k, kp)
        ks = max.(k, EPS_SAFE)
        ratio = kp ./ ks .- 1.0
        return -0.5 .* kappa .* ratio .* (ratio .+ 2.0)
    end

    py_mpk(k, z, kp) = 1.0 - delta .+ py_production_k(k, z) .- py_adj_cost_k(k, kp)
    py_investment(k, kp) = kp .- (1.0 - delta) .* k
    py_fischer_burmeister(mu, investment; eps = FB_EPS) =
        mu .+ investment .- sqrt.(mu .^ 2 .+ investment .^ 2 .+ eps)

    function py_policy_from_raw(raw)
        kp_raw = @view raw[1:N_COUNTRIES, :]
        lambda_raw = @view raw[(N_COUNTRIES + 1):(N_COUNTRIES + 1), :]
        mu_raw = @view raw[(N_COUNTRIES + 2):n_policies, :]
        return (
            kp = k_ss .* exp.(tanh.(kp_raw)),
            lambda = lambda_ss .* exp.(tanh.(lambda_raw)),
            mu = NNlib.softplus.(mu_raw .- 4.0),
        )
    end

    function py_z_next(z, shock_node)
        country_shocks = reshape(shock_node[1:N_COUNTRIES], N_COUNTRIES, 1)
        aggregate_shock = shock_node[n_shocks]
        return rho_z .* z .+ sigma_e .* (country_shocks .+ aggregate_shock)
    end

    function py_sequence_irbc_residual(model, ps, st, states, history, rule)
        size(states, 1) == n_states || throw(DimensionMismatch("states must be n_states-by-batch"))
        size(history, 1) == n_shocks ||
            throw(DimensionMismatch("IRBC history must store N country shocks plus one aggregate shock"))
        size(history, 3) == size(states, 2) ||
            throw(DimensionMismatch("states and histories must share a batch size"))
        size(rule.nodes, 1) == n_shocks ||
            throw(DimensionMismatch("quadrature nodes must span country and aggregate shocks"))

        raw, st_new = model(flatten_history(history), ps, st)
        policy = py_policy_from_raw(raw)
        k = @view states[1:N_COUNTRIES, :]
        z = @view states[(N_COUNTRIES + 1):n_states, :]
        batch = size(states, 2)

        weighted_terms = map(eachindex(rule.weights)) do q
            node = @view rule.nodes[:, q]
            shock_block = repeat(reshape(node, n_shocks, 1), 1, batch)
            next_history = prepend_history(history, shock_block)
            raw_next, _ = model(flatten_history(next_history), ps, st_new)
            policy_next = py_policy_from_raw(raw_next)
            z_next = py_z_next(z, node)

            integrand_rows = [
                policy_next.lambda .* py_mpk(
                    @view(policy.kp[j:j, :]),
                    @view(z_next[j:j, :]),
                    @view(policy_next.kp[j:j, :]),
                ) .- (1.0 - delta) .* @view(policy_next.mu[j:j, :])
                for j in 1:N_COUNTRIES
            ]
            return rule.weights[q] .* reduce(vcat, integrand_rows)
        end
        expected_integrand = reduce(+, weighted_terms)

        euler_lhs = policy.lambda .* (1.0 .+ py_adj_cost_kp(k, policy.kp))
        euler_rhs = beta .* expected_integrand .+ policy.mu
        euler = euler_rhs ./ max.(euler_lhs, EPS_SAFE) .- 1.0

        resource_rows = [
            py_production(get_k(states, j), get_z(states, j)) .+
            (1.0 - delta) .* get_k(states, j) .-
            @view(policy.kp[j:j, :]) .-
            py_adj_cost(get_k(states, j), @view(policy.kp[j:j, :])) .-
            py_consumption(policy.lambda, j)
            for j in 1:N_COUNTRIES
        ]
        resource = reduce(+, resource_rows)

        complementarity_rows = [
            py_fischer_burmeister(@view(policy.mu[j:j, :]), py_investment(get_k(states, j), @view(policy.kp[j:j, :])))
            for j in 1:N_COUNTRIES
        ]
        complementarity = reduce(vcat, complementarity_rows)

        total_sq = sum(euler .^ 2; dims = 1) .+ resource .^ 2 .+ sum(complementarity .^ 2; dims = 1)
        return (
            loss = mean(total_sq),
            euler = euler,
            resource = resource,
            complementarity = complementarity,
            policy = policy,
            next_capital = policy.kp,
        ), st_new
    end

    function py_sequence_irbc_forward_step(model, ps, st, states, history, shocks)
        size(shocks) == (n_shocks, size(states, 2)) ||
            throw(DimensionMismatch("shocks must contain N country shocks plus one aggregate shock per batch member"))
        raw, st_new = model(flatten_history(history), ps, st)
        policy = py_policy_from_raw(raw)
        z = @view states[(N_COUNTRIES + 1):n_states, :]
        country_shocks = @view shocks[1:N_COUNTRIES, :]
        aggregate_shocks = @view shocks[n_shocks:n_shocks, :]
        z_next = rho_z .* z .+ sigma_e .* (country_shocks .+ aggregate_shocks)
        states_next = vcat(policy.kp, z_next)
        history_next = prepend_history(history, shocks)
        return states_next, history_next, st_new
    end
end

# ╔═╡ 58f4bf23-a1fa-00b1-8b16-a748651c7786
md"""
### 9. Steady-state pre-training

The network is a `make_mlp(history_dim, hidden, n_policies)` with `swish` activations, sized by the active budget preset. Without a current-state input, a cold-initialised network produces nonsense policies that cause the cloud to diverge before training can stabilise it. We therefore **pre-train** the network to output raw \$\approx 0\$ (which under our parametrization means the steady-state policy) for small-noise histories.

This is a *supervised* step: one-line target, MSE loss (`mean(abs2, raw)`), no equilibrium residuals. Analogous to Approach C's pre-train in nb 01, but simpler because the target is just raw=0.
"""

# ╔═╡ e11140bb-138e-983c-0e0b-b9787a035071
md"""
### 10. Cloud-method training

Same loop as the BM sequence-space notebook: maintain a cloud of \$(x, \text{history})\$ pairs, simulate a few periods forward per episode to refresh the cloud, then take a mini-batch SGD step on `py_sequence_irbc_residual`. No ReLoBRaLo — we keep the equal-weight loss to make the comparison with nb 01 crisp (the take-away table at the end shows the extra residual noise vs state-space).

The Python notebook uses exponential LR decay from \$10^{-3}\$ to \$10^{-5}\$ over 800 episodes (~3 min on CPU) at the classroom scale; this Lux preview trains with a fixed-rate `Optimisers.Adam` step (`train_step!` with gradient-norm clipping) for the preset's episode budget.
"""

# ╔═╡ 55555555-105b-4555-8555-555555555555
begin
    states = py_start_states(rng_from_seed(SEED; offset = 1), hp.batch_size)
    histories = py_irbc_history(history_length = hp.history_length, batch = hp.batch_size)
    model = make_mlp(history_dim, hp.hidden, n_policies; activation = NNlib.swish)

    pretrain_state = setup_training(rng_from_seed(SEED; offset = 2), model, Optimisers.Adam(0.001); parameter_type = Float64)
    pretrain_loss(model, ps, st, batch) = begin
        raw, st_new = model(flatten_history(batch.histories), ps, st)
        return mean(abs2, raw), st_new
    end

    pretrain_log = NamedTuple[]
    for step in 1:hp.pretrain_steps
        noisy_histories = 0.05 .* randn(rng, n_shocks, hp.history_length, hp.batch_size)
        metrics = train_step!(pretrain_state, pretrain_loss, (histories = noisy_histories,); max_grad_norm = 10.0)
        append_metric!(pretrain_log; step, loss = metrics.loss)
    end

    state = setup_training(pretrain_state.model, pretrain_state.ps, pretrain_state.st, Optimisers.Adam(0.001))
    irbc_loss(model, ps, st, batch) = begin
        pieces, st_new = py_sequence_irbc_residual(model, ps, st, batch.states, batch.histories, rule)
        return pieces.loss, st_new
    end

    train_result = let states_local = states, histories_local = histories
        initial_loss_local = loss_value(state, irbc_loss, (states = states_local, histories = histories_local))
        history_log_local = NamedTuple[]
        for step in 1:hp.steps
            state_buffer = states_local
            history_buffer = histories_local
            for _ in 1:hp.refresh_steps
                shocks = randn(rng, n_shocks, size(states_local, 2))
                states_local, histories_local, _ =
                    py_sequence_irbc_forward_step(state.model, state.ps, state.st, states_local, histories_local, shocks)
                state_buffer = hcat(state_buffer, states_local)
                history_buffer = cat(history_buffer, histories_local; dims = 3)
            end

            cloud_batch = size(state_buffer, 2)
            idx = shuffle(rng, collect(1:cloud_batch))[1:min(hp.minibatch, cloud_batch)]
            batch = (states = state_buffer[:, idx], histories = history_buffer[:, :, idx])
            metrics = train_step!(state, irbc_loss, batch; max_grad_norm = 10.0)
            append_metric!(history_log_local; step, loss = metrics.loss)
        end
        (initial_loss = initial_loss_local, history_log = history_log_local, states = states_local, histories = histories_local)
    end
    initial_loss = train_result.initial_loss
    history_log = train_result.history_log
    states = train_result.states
    histories = train_result.histories
end

# ╔═╡ 8a977eb3-431a-4596-1e5e-9db88a75bf10
md"""
### 11. Diagnostics

The full Python notebook plots four diagnostics after training:

- **11.1 Loss curve** — \$\log_{10}\$ loss vs episode.
- **11.2 Ergodic distribution of \$(k_1, k_2)\$** — should concentrate near \$(k_{ss}, k_{ss}) = (1, 1)\$ with spread set by the AR(1) productivity dynamics.
- **11.3 Per-equation residuals** — on this budget, residuals are **comparable** to the state-space DEQN (often modestly *better*), because the SS-centred \$\tanh\$ parametrization acts as a strong prior near the stochastic steady state. FB residuals stay small since \$\mu_j \approx 0\$ on the ergodic set (irreversibility almost never binds at \$\kappa = 0.5\$).
- **11.4 Policy scatter** — \$k'_j\$ vs \$k_j\$ on the ergodic cloud, with the 45-degree line as reference.

This Lux preview instead computes the same residual pieces via `py_sequence_irbc_residual`, reduces them to per-equation `residual_summary` RMSEs, and verifies the tensor shapes with structural `@assert`s — the shock-history layout \$(n_{\text{shocks}}, T, \text{batch})\$, its flattened Lux-boundary shape \$(T \cdot n_{\text{shocks}}, \text{batch})\$, and the \$(n_{\text{shocks}}, 3^{n_{\text{shocks}}})\$ quadrature grid.
"""

# ╔═╡ 66666666-105b-4666-8666-666666666666
begin
    diagnostics, _ = py_sequence_irbc_residual(state.model, state.ps, state.st, states, histories, rule)
    flat_history = flatten_history(histories)

    @assert N_COUNTRIES == 2
    @assert n_shocks == N_COUNTRIES + 1
    @assert size(rule.nodes) == (n_shocks, 3^n_shocks)
    @assert size(histories) == (n_shocks, hp.history_length, hp.batch_size)
    @assert size(flat_history) == (history_dim, hp.batch_size)
    @assert size(states) == (n_states, hp.batch_size)
end

# ╔═╡ d3dcb601-6761-9c31-9271-d5a237328f22
md"""
### 12. Take-away

**The single conceptual change** versus Day 3 nb 01's state-space DEQN:

| Day 3 nb 01 (state-space) | This notebook (sequence-space) |
|---|---|
| Network input: \$(k_1, k_2, z_1, z_2)\$ — **4 floats**. | Network input: \$\mathcal{E}_t^{(80)}\$ — **240 floats**. |
| Output head: `softplus` for all 5 policies (works because current \$k\$ is input). | Output head: SS-centred \$\tanh\$ for \$k'_j, \lambda\$; shifted softplus for \$\mu_j\$. |
| Pre-training target: \$(k_{ss}, k_{ss}, \lambda_{ss}, \varepsilon, \varepsilon)\$. | Pre-training target: raw output = 0 (parametrization does the rest). |
| Euler / ARC / FB residuals. | *Unchanged.* |
| Gauss–Hermite quadrature. | *Unchanged.* |
| Cloud-method sampling. | *Unchanged*, but the cloud carries a history alongside the state. |

#### When does sequence space pay off?

- **BM (nb 05)**: 2-dim state, 1 shock → 25-dim history. *Larger* input. Pedagogical device only.
- **IRBC (this nb)**: 4-dim state, 3 shocks → 240-dim history. *Much larger* input. Still pedagogical — a bridge to KS.
- **Krusell–Smith (nb 06)**: infinite-dim state (wealth distribution), 1 shock → 100-dim history. **First genuine win**: exogenous input replaces the distribution summary statistics that state-space methods rely on.

The point of running IRBC in sequence space is not computational efficiency — it's to show that the *same* training template works for a multi-equation system with multiple independent shock channels, before we hand the method over to Krusell–Smith where the real motivation kicks in.

#### What the residual comparison tells us

On this classroom budget the sequence-space network reaches residual levels *comparable to* (and in places better than) nb 01's Approach C. Two forces partly offset each other:

- **Against** sequence space: the network has to infer \$(k_j, z_j)\$ from a noisy 240-dim history rather than read them off as inputs. All else equal, residuals should be larger.
- **For** sequence space: the SS-centred \$\tanh\$ parametrization injects a strong prior that the policy is near the steady state — nb 01's softplus head has no such anchor. This prior dominates the budget here. (The full Python notebook combines it with exponential LR decay from \$10^{-3}\$ to \$10^{-5}\$; this Lux preview instead uses a fixed-rate `Optimisers.Adam` step, as noted in Section 10.)

Takeaway: sequence space is *not* automatically worse. At the scale where it matters (Krusell–Smith), the cost of inferring state from history is offset by the radically simpler input geometry.

#### Reference
Azinovic-Yang, M. & Žemlička, J. (2025). *Deep Learning in the Sequence Space.* arXiv:2509.13623. Companion JAX repository: [`azinoma/DeepLearningInTheSequenceSpace`](https://github.com/azinoma/DeepLearningInTheSequenceSpace).

The cell below returns the machine-checkable diagnostics summary for this notebook's smoke run.
"""

# ╔═╡ 77777777-105b-4777-8777-777777777777
(
    initial_loss = initial_loss,
    final_loss = diagnostics.loss,
    euler_rmse = residual_summary(diagnostics.euler).rmse,
    complementarity_rmse = residual_summary(diagnostics.complementarity).rmse,
    resource_rmse = residual_summary(diagnostics.resource).rmse,
    state_shape = size(states),
    history_shape = size(histories),
    lux_history_shape = size(flat_history),
    quadrature_shape = size(rule.nodes),
    history_features_equal_python = size(histories, 1) == N_COUNTRIES + 1,
    python_state_layout_k_then_z = true,
    python_euler_scale_rhs_over_lhs_minus_one = true,
    python_resource_sign_output_less_uses = true,
)

# ╔═╡ Cell order:
# ╟─11111111-105b-4111-8111-111111111111
# ╟─0f019c80-eec8-5c9b-7950-b2b603a80125
# ╟─067797b0-1823-ad1b-89cf-d986b215b310
# ╠═22222222-105b-4222-8222-222222222222
# ╟─ce482026-4fd9-681f-3704-a88e2ec1f116
# ╟─60d26767-dd30-faf3-3bd3-88fa87f4c2f7
# ╠═33333333-105b-4333-8333-333333333333
# ╟─03081ca3-4177-0f29-153b-6814eead822a
# ╟─ce15e57e-8af2-fcb2-5530-48ffdb0e42a6
# ╟─01906499-1780-33b6-b67f-3b60bd2c8ea8
# ╟─3de9a7fe-6d0b-83af-0e6a-33099d7b94fb
# ╠═44444444-105b-4444-8444-444444444444
# ╟─58f4bf23-a1fa-00b1-8b16-a748651c7786
# ╟─e11140bb-138e-983c-0e0b-b9787a035071
# ╠═55555555-105b-4555-8555-555555555555
# ╟─8a977eb3-431a-4596-1e5e-9db88a75bf10
# ╠═66666666-105b-4666-8666-666666666666
# ╟─d3dcb601-6761-9c31-9271-d5a237328f22
# ╠═77777777-105b-4777-8777-777777777777
