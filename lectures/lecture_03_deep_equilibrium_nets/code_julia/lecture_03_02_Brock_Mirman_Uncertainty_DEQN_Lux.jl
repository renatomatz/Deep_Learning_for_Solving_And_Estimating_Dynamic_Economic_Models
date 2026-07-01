### A Pluto.jl notebook ###
# v0.20.0

using Markdown
using InteractiveUtils

# ╔═╡ 11111111-0302-4111-8111-111111111111
md"""
# Lecture 03, Notebook 02: Stochastic Brock-Mirman DEQN in Lux

This translation adds Gauss-Hermite expectations through the shared quadrature
helper. The quadrature nodes are standard-normal nodes, matching the Python
notebook's Hermite scaling.
"""

# ╔═╡ 18b81e88-c9c7-3e4a-91d6-1a8fcc4d82af
md"""
## Lecture 03, Notebook 02: Stochastic Brock–Mirman DEQN

**Course:** Deep Learning for Solving and Estimating Dynamic Models in Economics and Finance
**Script reference:** §2.4 (stochastic Brock–Mirman), §2.6 (Gauss–Hermite quadrature for the conditional expectation)
**Notebook role:** core
**Author:** Simon Scheidegger

*Julia/Lux/Pluto preview of* `lectures/lecture_03_deep_equilibrium_nets/code/lecture_03_02_Brock_Mirman_Uncertainty_DEQN.ipynb`.
"""

# ╔═╡ e356a814-2fc8-f877-1466-70b03f2c70b1
md"""
## Simple Introduction to Deep Equilibrium Nets

### Notebook 2: uncertainty and sampling states from the simulated path of the economy

The previous notebook introduced DEQNs with a single state (\$\mathbf{X}_t = K_t\$), no uncertainty, and states sampled from a fixed interval. This notebook adds two things:

1. **Aggregate uncertainty** — an AR(1) process for log total-factor-productivity \$\log z_t\$. The equilibrium condition now contains an **expectation operator** requiring numerical integration, and the state becomes two-dimensional, \$\mathbf{X}_t = [z_t, K_t]\$.
2. **Sampling from simulated paths** — instead of a hyper-cube, we draw training states from simulated trajectories of the economy. Because state variables are correlated, a rectangular domain is exponentially wasteful in high dimensions (see [Maliar et al. (2011)](https://www.sciencedirect.com/science/article/pii/S0165188910002186)).

The planner now maximizes expected discounted utility:

\$\$
\begin{aligned}
&\max_{\{C_t\}_{t=0}^{\infty}} \sum_{t=0}^{\infty} \beta^{t} E\!\left[\ln(C_t)\right]\\
\text{s.t.}\quad & K_{t+1} + C_t = Y_t + (1-\delta)K_t,
\end{aligned}
\$\$

with production and productivity

\$\$Y_t = z_t K_t^{\alpha}, \qquad \log z_t = \varrho \log z_{t-1} + \sigma \epsilon_t, \qquad \epsilon_t \sim N(0,1).\$\$

The certainty case of Notebook 1 is recovered when \$\varrho = 1\$ and \$\sigma = 0\$.
"""

# ╔═╡ 1ae8285b-1674-ba25-a81f-f8381ed06fb7
md"""
Recursively, the Bellman equation is \$V(z_t, K_t) = \max_{K_{t+1}} \ln(C_t) + \beta E[V(z_{t+1}, K_{t+1})]\$ with \$C_t = Y_t + (1-\delta)K_t - K_{t+1}\$. The state \$\mathbf{X}_t = [z_t, K_t]\$ is 2-dimensional; the policy \$f(\mathbf{X}_t) = K_{t+1}\$ is 1-dimensional and approximated by \$\mathcal{N}(\mathbf{X}_t)\$. The first-order condition plus the envelope theorem give the **stochastic Euler equation**

\$\$0 = -\frac{1}{C_t} + \beta\, E\!\left[\frac{1}{C_{t+1}}\left(1 - \delta + r_{t+1}\right)\right], \qquad r_{t} = \alpha z_t K_t^{\alpha-1}.\$\$

The only change from Notebook 1 is the expectation on the right, whose integrand varies with the realization of \$z_{t+1}\$. Rearranged into a **relative consumption error**:

\$\$0 = \frac{1}{C_t\,\beta\, E\!\left[\frac{1}{C_{t+1}}(1 - \delta + r_{t+1})\right]} - 1,\$\$

which becomes the **loss**. Three remarks:

1. As before we output a savings rate \$s_t \in (0,1)\$ (sigmoid) with \$K_{t+1} = (1-\delta)K_t + Y_t s_t\$, a **hard** feasibility constraint guaranteeing \$C_t > 0\$ and \$K_{t+1} > 0\$.
2. The expectation must be integrated numerically. Since \$\epsilon_{t+1}\$ is Gaussian we use **Gauss–Hermite quadrature** (detailed at the quadrature cell below). Alternatives include (Quasi-)Monte Carlo, monomial rules, or discretizing the AR(1) into a Markov chain à la [Tauchen (1986)](https://www.sciencedirect.com/science/article/pii/0165176586901680) / Rouwenhorst.
3. We first sample states from an exogenous rectangle \$\mathcal{X} = [\underline{z}, \overline{z}] \times [\underline{K}, \overline{K}]\$, then — because \$z\$ and \$K\$ are correlated — switch to sampling from **simulated paths**, solving the model where it matters.
"""

# ╔═╡ 853d814f-1845-132b-2882-61c7fa8f2498
md"""
### Implementing the loss function

Where the Python notebook imports NumPy and TensorFlow/Keras, the Julia preview loads `Lux` (explicit `model(x, ps, st)` networks), `Optimisers` (Adam), `NNlib` (activations), and `CairoMakie` (plots), on top of the shared `DLEFJulia` helpers.
"""

# ╔═╡ 22222222-0302-4222-8222-222222222222
begin
    import Pkg
    Pkg.activate(joinpath(@__DIR__, "..", "..", "..", "julia"))

    using CairoMakie
    using DLEFJulia
    using Lux
    using NNlib
    using Optimisers
end

# ╔═╡ b34e6e25-d988-de34-631e-86915b007a07
md"""
`RUN_MODE` picks the training budget (`smoke` here for a fast sanity check; `teaching` / `production` for the longer runs behind the slide figures) and `SEED = 0` fixes the RNG. This preview keeps `RUN_MODE = "smoke"` and `SEED = 0`.
"""

# ╔═╡ 33333333-0302-4333-8333-333333333333
begin
    RUN_MODE = "smoke"
    SEED = 0
    budgets = (
        smoke = (steps = 25, batch_size = 64),
        teaching = (steps = 500, batch_size = 128),
        production = (steps = 3_000, batch_size = 256),
    )
    hp = run_mode_budget(RUN_MODE; budgets)
    rng = rng_from_seed(SEED)
end

# ╔═╡ 52b0ca32-64a6-44c6-3aa2-f84c5d98d1a8
md"""
### Economic parameters and the steady state

We fix \$\alpha = 0.36\$, \$\beta = 0.99\$, and now **partial** depreciation \$\delta = 0.1\$, plus AR(1) parameters \$\varrho = 0.9\$ and \$\sigma = 0.04\$. With \$\delta = 0.1\$ there is **no closed-form policy**, so unlike Notebook 1 we cannot benchmark against an analytical solution. We can still pin down the deterministic steady state: at \$z = 1\$ the Euler equation collapses to \$1/\beta = 1 - \delta + \alpha k^{\alpha-1}\$, giving

\$\$k^\star = \left(\frac{1/\beta - 1 + \delta}{\alpha}\right)^{1/(\alpha-1)},\$\$

which is useful for choosing a sensible sampling range. `BrockMirmanParams` packages all of these.
"""

# ╔═╡ a93982e9-c712-1d7c-7fcf-d06142d12045
md"""
### Evaluating the expectation operator

Gauss–Hermite quadrature approximates integrals against \$e^{-x^2}\$ by a weighted sum over \$n\$ nodes,

\$\$\int_{-\infty}^{+\infty} h(x)\, e^{-x^2}\, dx \approx \sum_{i=1}^{n} w_i\, h(x_i),\$\$

exact when \$h\$ is a polynomial of degree \$\le 2n - 1\$; the integrand here is smooth enough for a good approximation with a handful of nodes. One subtlety: the raw nodes/weights integrate against \$e^{-x^2}\$, so their weights sum to \$\sqrt{\pi}\$. To integrate against the \$N(0,1)\$ density we divide the weights by \$\sqrt{\pi}\$ and scale the nodes by \$\sqrt{2}\$. `gauss_hermite_rule(5)` bakes this in, and `quadrature_checks` verifies the rule reproduces the standard-normal mean \$0\$ and variance \$1\$.
"""

# ╔═╡ 44444444-0302-4444-8444-444444444444
begin
    params = BrockMirmanParams(alpha = 0.36, beta = 0.99, delta = 0.1, rho_z = 0.9, sigma_z = 0.04)
    rule = gauss_hermite_rule(5)
    quadrature_checks = (
        weight_sum = sum(rule.weights),
        normal_mean = quadrature_expectation(x -> x, rule),
        normal_variance = quadrature_expectation(x -> x^2, rule),
    )
end

# ╔═╡ 01b4bfa2-7d63-e319-0d55-f4b26045c9aa
md"""
#### Deep neural network

The network again approximates the **savings rate** \$s_t\$, but the input is now the 2-dimensional state \$\mathbf{X}_t = [z_t, K_t]\$ and the output the 1-dimensional savings rate, with \$K_{t+1} = (1-\delta)K_t + Y_t s_t\$. The Python notebook uses two hidden ReLU layers with a sigmoid output head; this Lux preview uses `make_mlp(2, (24, 24), 1; activation = tanh)` and applies the sigmoid separately via `savings_transform`.

**The batch dimension.** In Lux, a batch of \$N\$ states is a **feature-by-batch** \$2 \times N\$ matrix (productivity row, capital row), and the network returns a \$1 \times N\$ matrix of savings rates — the transpose of the Python samples-on-rows convention.
"""

# ╔═╡ 1474e2d2-cf68-b24d-ab3b-b18b48a1f434
md"""
##### Hard vs. soft constraints — the central design choice in DEQNs

- **Inequality / feasibility constraints** — \$C_t > 0\$, \$K_{t+1} > 0\$, the resource constraint \$C_t + K_{t+1} = Y_t\$ — must hold *exactly*.
- **Optimality conditions** — the Euler equation — hold in equilibrium but not at every intermediate guess.

| | Hard constraint (architecture) | Soft constraint (loss) |
|--|--|--|
| **What** | Built into the network output | Penalised in the cost function |
| **How** | Activation choice + algebraic identities | Squared residuals in \$\mathcal{L}\$ |
| **Cost** | Always satisfied — even at random init | Only satisfied at convergence |

The **sigmoid** savings share (\$s_t \in (0,1)\$, with \$K_{t+1} = s_t Y_t\$ and \$C_t = (1-s_t)Y_t\$) **guarantees \$C_t > 0\$ and \$K_{t+1} > 0\$ at every iteration** — a hard constraint. The Euler equation is enforced softly through the loss. This removes a class of bad local minima and is one reason DEQNs converge where naive penalty methods do not (Azinovic, Gaegauf & Scheidegger, 2022, §4.2.2).
"""

# ╔═╡ 5cbf5fc9-b3b7-7a4b-cbc4-e7b9738eb42a
md"""
#### Cost function with the expectation operator

`stochastic_bm_residual` evaluates the relative-consumption residual \$\frac{1}{C_t\,\beta\, E[\frac{1}{C_{t+1}}(1-\delta+r_{t+1})]} - 1\$ on a batch of states. The conditional expectation is taken by summing the integrand over the Gauss–Hermite nodes for next-period productivity \$z_{t+1}\$, exactly the helper the Python notebook builds by iterating over the `n_int` states in \$t+1\$. The mean squared residual is the loss. Gradients come from `Zygote` inside `train_step!` (the analogue of TensorFlow's `GradientTape`, with gradient-norm clipping at 10), and `Optimisers.Adam(0.001)` updates the parameters.
"""

# ╔═╡ aed58310-9f7e-6930-143e-a81682cb65d8
md"""
#### From exogenous sampling to simulated states

This cell trains in two phases. **First**, `sample_states` draws states uniformly from the exogenous rectangle and we run the standard training loop. **Then** we switch to simulated sampling: `simulate_training_states` rolls the economy forward under the current policy — \$K_{t+1} = Y_t s_t\$, then \$\log z_{t+1} = \varrho \log z_t + \sigma \epsilon_{t+1}\$ from a fresh Gaussian innovation — and we retrain on those states, **iterating between simulating and training** as in [Azinovic et al. (2022)](https://onlinelibrary.wiley.com/doi/full/10.1111/iere.12575). Simulating many short parallel tracks (rather than one long path) is faster and keeps the samples more independent.

The motivation: a simulated economy lives on a **cloud around the diagonal** in \$(z, K)\$ space — it never visits low-productivity/high-capital or high-productivity/low-capital corners, so training there is wasted, exponentially so in higher dimensions. *(The full Python notebook draws the simulated time path and the ergodic scatter cloud in standalone plots; this preview folds forward simulation directly into the training loop and reports the loss from both phases.)*
"""

# ╔═╡ 55555555-0302-4555-8555-555555555555
begin
    model = make_mlp(2, (24, 24), 1; activation = NNlib.tanh)
    train_state = setup_training(rng_from_seed(SEED; offset = 1), model, Optimisers.Adam(0.001); parameter_type = Float64)
    savings_transform = NNlib.sigmoid

    function sample_states(rng, n)
        z = 0.7 .+ 0.6 .* rand(rng, n)
        k = 0.9 .+ 11.1 .* rand(rng, n)
        return vcat(reshape(z, 1, :), reshape(k, 1, :))
    end

    function simulate_training_states(rng, train_state, start_states, periods)
        n_tracks = size(start_states, 2)
        current = start_states
        simulated = Matrix{Float64}(undef, 2, n_tracks * periods)
        for t in 1:periods
            z = @view current[1:1, :]
            k = @view current[2:2, :]
            raw_savings, _ = train_state.model(current, train_state.ps, train_state.st)
            savings = savings_transform(raw_savings)
            output = z .* k .^ params.alpha
            k_next = (1 - params.delta) .* k .+ output .* savings
            innovations = reshape(randn(rng, n_tracks), 1, :)
            z_next = exp.(params.rho_z .* log.(z) .+ params.sigma_z .* innovations)
            current = vcat(z_next, k_next)
            simulated[:, ((t - 1) * n_tracks + 1):(t * n_tracks)] .= current
        end
        return simulated, current
    end

    bm_loss(model, ps, st, states) = begin
        pieces, st_new = stochastic_bm_residual(model, ps, st, states, rule; params, transform = savings_transform)
        return pieces.loss, st_new
    end

    initial_batch = sample_states(rng, hp.batch_size)
    initial_loss = loss_value(train_state, bm_loss, initial_batch)
    history = NamedTuple[]
    for _ in 1:hp.steps
        local batch = sample_states(rng, hp.batch_size)
        metrics = train_step!(train_state, bm_loss, batch; max_grad_norm = 10.0)
        append_metric!(history; phase = :uniform, step = metrics.step, loss = metrics.loss)
    end

    simulated_history = let simulated_start = sample_states(rng, min(hp.batch_size, 24))
        local sim_history = NamedTuple[]
        for _ in 1:max(1, hp.steps ÷ 5)
            local simulated_batch
            simulated_batch, simulated_start = simulate_training_states(rng, train_state, simulated_start, 3)
            metrics = train_step!(train_state, bm_loss, simulated_batch; max_grad_norm = 10.0)
            append_metric!(sim_history; phase = :simulated, step = metrics.step, loss = metrics.loss)
        end
        sim_history
    end
end

# ╔═╡ 4087ed63-6aca-dc9d-4eda-f3cd6ededbd3
md"""
### Inspecting the learned policy

With no closed-form benchmark, we evaluate the trained policy along a capital slice at \$z = 1\$, summarise the residual over that slice, and plot \$K_{t+1}\$ against the 45-degree line to see how saving responds to capital at mean productivity.
"""

# ╔═╡ 66666666-0302-4666-8666-666666666666
begin
    z_line = fill(1.0, 1, 100)
    k_line = reshape(collect(range(0.9, 12.0; length = 100)), 1, :)
    eval_states = vcat(z_line, k_line)
    diagnostics, _ = stochastic_bm_residual(train_state.model, train_state.ps, train_state.st, eval_states, rule; params, transform = savings_transform)
    residual_stats = residual_summary(diagnostics.residual)
end

# ╔═╡ 77777777-0302-4777-8777-777777777777
begin
    fig = Figure(size = figure_size(RUN_MODE))
    ax = Axis(fig[1, 1], xlabel = "K_t at Z_t = 1", ylabel = "K_{t+1}")
    lines!(ax, vec(k_line), vec(k_line); color = :gray55, linestyle = :dash, label = "45 degree")
    lines!(ax, vec(k_line), vec(diagnostics.next_capital); color = :dodgerblue3, linewidth = 3, label = "DEQN")
    axislegend(ax; position = :lt)
    fig
end

# ╔═╡ 3209b939-a85e-d8a5-03d9-3f7c79619525
md"""
### Conclusion

This notebook extended the DEQN to a stochastic, two-dimensional Brock–Mirman model: the Euler equation gained an expectation, which we integrated with Gauss–Hermite quadrature, and we moved from a wasteful exogenous rectangle to sampling states from simulated paths of the economy.

**Final remark on simulation-based methods.** Simulation-based sampling is a huge advantage — exponentially so in high dimensions — but it introduces fragility: the training distribution shifts as the policy changes. That is exactly what we want, yet moving too quickly to unseen states can destabilize training, so parameters (**especially the learning rate**) may need careful tuning, and early random policies can predict infeasible states. Azinovic et al. (2022) and [Azinovic and Žemlička (2023)](https://arxiv.org/abs/2303.14802) discuss stabilization and market-clearing architectures. The cell below returns a machine-checkable summary of this run.
"""

# ╔═╡ 88888888-0302-4888-8888-888888888888
(
    quadrature = quadrature_checks,
    initial_loss = initial_loss,
    final_uniform_loss = history[end].loss,
    final_simulated_loss = simulated_history[end].loss,
    max_abs_residual = residual_stats.max_abs,
    finite_share = residual_stats.finite_share,
)

# ╔═╡ Cell order:
# ╟─11111111-0302-4111-8111-111111111111
# ╟─18b81e88-c9c7-3e4a-91d6-1a8fcc4d82af
# ╟─e356a814-2fc8-f877-1466-70b03f2c70b1
# ╟─1ae8285b-1674-ba25-a81f-f8381ed06fb7
# ╟─853d814f-1845-132b-2882-61c7fa8f2498
# ╠═22222222-0302-4222-8222-222222222222
# ╟─b34e6e25-d988-de34-631e-86915b007a07
# ╠═33333333-0302-4333-8333-333333333333
# ╟─52b0ca32-64a6-44c6-3aa2-f84c5d98d1a8
# ╟─a93982e9-c712-1d7c-7fcf-d06142d12045
# ╠═44444444-0302-4444-8444-444444444444
# ╟─01b4bfa2-7d63-e319-0d55-f4b26045c9aa
# ╟─1474e2d2-cf68-b24d-ab3b-b18b48a1f434
# ╟─5cbf5fc9-b3b7-7a4b-cbc4-e7b9738eb42a
# ╟─aed58310-9f7e-6930-143e-a81682cb65d8
# ╠═55555555-0302-4555-8555-555555555555
# ╟─4087ed63-6aca-dc9d-4eda-f3cd6ededbd3
# ╠═66666666-0302-4666-8666-666666666666
# ╠═77777777-0302-4777-8777-777777777777
# ╟─3209b939-a85e-d8a5-03d9-3f7c79619525
# ╠═88888888-0302-4888-8888-888888888888
