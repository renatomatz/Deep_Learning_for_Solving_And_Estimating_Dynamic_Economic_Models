### A Pluto.jl notebook ###
# v0.20.0

using Markdown
using InteractiveUtils

# ╔═╡ 11111111-10ff-4111-8111-111111111111
md"""
# Lecture 10: Krusell-Smith CPU Tutorial in Lux

A CPU-only companion to the original JAX tutorial. It keeps the shape-preserving
idea: the actor emits idiosyncratic MPC heads, so consumption is feasible and
monotone on the capital grid by construction.
"""

# ╔═╡ 87008503-438e-da0f-7dc2-4a17db187b18
md"""
## Lecture 10: Sequence-space DEQNs — Krusell-Smith CPU Tutorial

**Course:** Deep Learning for Solving and Estimating Dynamic Models in Economics and Finance
**Script reference:** §6.7 (Sequence-space DEQNs) — JAX/Optax port of the upstream pedagogical KS tutorial
**Notebook role:** extension
**Author:** Simon Scheidegger (course adaptation; upstream attribution preserved below)

*Julia/Lux/Pluto preview of* `lectures/lecture_10_sequence_space_deqns/code/lecture_10_KrusellSmith_Tutorial_CPU.ipynb`.

> **Run mode.** The checked-in run uses `RUN_MODE = "smoke"`, which maps to a compact CPU budget (short shock history, small capital grid, few training steps). Set `RUN_MODE` to `"teaching"` or `"production"` in the budgets cell for the larger, slower runs — the Python original calls these the `quick` and `full` `TUTORIAL_MODE`s.

> **Shape-preserving CPU port.** This is a CPU-only Lux/Optimisers/Zygote companion to the JAX/Optax tutorial. It keeps the same mathematics and the same shape-preserving idea; only the framework wording changes (JAX `jit`/`grad`/`vmap` → Lux with explicit `model(x, ps, st)` state threading and Zygote/ForwardDiff; Optax → `Optimisers.jl`).

> **Note on `K_ss`.** The deterministic representative-agent benchmark `K_ss` is a convenient reference point for plots and initial conditions. It is *not* the stochastic heterogeneous-agent equilibrium capital stock. *In this CPU preview* we do not actually compute `K_ss` or draw a benchmark line; the initial household distribution is simply centered at a fixed target capital (`K_target = 5.0`).
"""

# ╔═╡ 7f99fc7c-ab8e-7a72-92d6-4ed915b9fb02
md"""
## Upstream attribution

This notebook is a Julia/Lux port of the upstream pedagogical tutorial `01_KrusellSmith_Tutorial_CPU.ipynb` released with the paper *Deep Learning in the Sequence Space* by Marlon Azinovic-Yang and Jan Žemlička (arXiv:2509.13623).

- **Upstream repository:** <https://github.com/azinoma/DeepLearningInTheSequenceSpace>
- **Paper:** Azinovic-Yang & Žemlička (2025), arXiv:2509.13623

The course adaptation adds a `RUN_MODE`-driven budget switch, an explicit shape-guarantee diagnostic, and clarifying commentary. The mathematics and the algorithm are unchanged from the upstream tutorial.
"""

# ╔═╡ 22222222-10ff-4222-8222-222222222222
begin
    import Pkg
    Pkg.activate(joinpath(@__DIR__, "..", "..", "..", "julia"))

    using DLEFJulia
    using Lux
    using NNlib
    using Optimisers
end

# ╔═╡ 33333333-10ff-4333-8333-333333333333
begin
    RUN_MODE = "smoke"
    SEED = 0
    budgets = (
        smoke = (steps = 4, grid_size = 20, history_length = 5),
        teaching = (steps = 150, grid_size = 56, history_length = 25),
        production = (steps = 1_000, grid_size = 128, history_length = 80),
    )
    hp = run_mode_budget(RUN_MODE; budgets)
end

# ╔═╡ accd9dfa-0ef9-72d7-a385-1c1be5858d33
md"""
---
## Part 1: The Economic Model

### The Krusell-Smith Economy

We consider an economy with:
- A **continuum of households** who save and consume
- **Aggregate risk**: productivity \$Z_t \in \{Z_L, Z_H\}\$ follows a Markov chain
- **Idiosyncratic risk**: each household's efficiency \$\varepsilon_t \in \{\varepsilon_L, \varepsilon_H\}\$ also follows a Markov chain

### Prices
A representative firm uses Cobb-Douglas technology \$Y = Z K^\alpha L^{1-\alpha}\$, giving:

\$\$R_t = 1 - \delta + \alpha Z_t (K_t / L_t)^{\alpha - 1}\$\$
\$\$w_t = (1 - \alpha) Z_t (K_t / L_t)^{\alpha}\$\$

### Household Problem
Each household chooses consumption \$c_t\$ to maximize:
\$\$\max \; \mathbb{E}_0 \sum_{t=0}^{\infty} \beta^t \log(c_t)\$\$
subject to:
\$\$c_t + k_{t+1} = w_t \varepsilon_t + R_t k_t, \qquad k_{t+1} \geq 0\$\$

### Optimality Conditions
At an **interior solution** (\$k_{t+1} > 0\$), the Euler equation holds:
\$\$u'(c_t) = \beta \, \mathbb{E}_t [R_{t+1} \, u'(c_{t+1})]\$\$

At the **borrowing constraint** (\$k_{t+1} = 0\$), the Euler gap can be nonzero, but the Kuhn-Tucker conditions still have to hold.

For teaching purposes, this notebook keeps the Markov structure deliberately simple: both aggregate and idiosyncratic shocks have **two states**.

### Calibration

The next cell sets up the model parameters via `SequenceKSParams`, the shock history, the initial household distribution, and the neural-network actor. In this Julia preview the calibration, the capital grid, and the actor are all constructed in one cell; the `RUN_MODE` budget picks the capital-grid size and the shock-history length.
"""

# ╔═╡ 1bf58e2e-ba4f-2bf0-1100-fe83827052ef
md"""
---
## Part 2: The I-Spline MPC Representation

### Shape-preserving operator learning
For a given aggregate history of shocks, \$Z^t\$, we want to predict a consumption *function* \$c(\varepsilon, k)\$. Following the *operator learning* idea, we predict the consumption values on a grid of idiosyncratic states and use piecewise-linear interpolation in between.

We want a consumption function \$c(\varepsilon, k)\$ that is:
- **Increasing** in \$k\$ (richer people consume more)
- **Concave** in \$k\$ (the marginal propensity to consume decreases with wealth)
- **Feasible**: \$0 < c < w\varepsilon + Rk\$ (no borrowing, positive consumption)

Instead of parameterizing \$c\$ directly, we parameterize the **marginal propensity to consume (MPC)**:

\$\$\text{MPC}(\varepsilon, k) = \alpha(\varepsilon) \left(1 - \sum_{j=1}^J \tilde{w}_j(\varepsilon) \, I_j(k)\right)\$\$

where \$\alpha \in (0,1)\$ is the MPC at the borrowing constraint (from a sigmoid), \$I_j(k)\$ are **I-spline basis functions** (monotonically increasing from 0 to 1), and \$\tilde{w}_j \geq 0\$ with \$\sum \tilde{w}_j < 1\$ are weights from a "phantom-zero" softmax. Since each \$I_j\$ is increasing and \$\tilde{w}_j \geq 0\$, the MPC is **decreasing by construction**; since \$\sum \tilde{w}_j < 1\$ it is **positive everywhere**; and since \$\alpha < 1\$, consumption is **feasible**.

Consumption is recovered by **cumulation** on a fixed grid:
\$\$c(k_0) = \text{MPC}(k_0) \cdot m(k_0), \qquad c(k_n) = c(k_{n-1}) + \text{MPC}(k_n) \cdot R \cdot \Delta k_n\$\$

**In this Julia preview** we do **not** build the I-spline basis. The shared `DLEFJulia` sequence-space helpers (`SequenceKSParams`, `sequence_ks_policy_grid`) use a simpler shape-preserving surrogate: the actor emits one **constant per-income-state MPC head** \$\alpha(\varepsilon)\$, squashed into \$(0.05,\,0.95)\$ by a sigmoid (`mpc = 0.05 + 0.90 * sigmoid(raw)`), and consumption on the grid is recovered as \$c(\varepsilon, k) = \text{MPC}(\varepsilon)\cdot\text{cash}(\varepsilon, k)\$ with \$\text{cash} = R k + w\varepsilon\$. Because the MPC is constant in \$k\$ and cash is affine in \$k\$, consumption is increasing and (weakly) concave in \$k\$ and feasible by the same \$\alpha < 1\$ argument — just without the cubic I-spline curvature. The capital grid here is a **uniform** grid on \$[0, 20]\$; the full Python notebook instead builds the cubic I-spline basis \$B_{j,n} = I_j(\log(\text{BASIS\_SHIFT} + k_n))\$ on a **log-spaced** grid (more resolution near \$k = 0\$ where the policy is steepest). The actor is a `make_mlp` MLP, and the shape guarantees are **verified after the fact** by the `shape_report` diagnostic (feasible / monotone / concave) rather than asserted silently.
"""

# ╔═╡ 44444444-10ff-4444-8444-444444444444
begin
    params = SequenceKSParams(
        beta = 0.93, delta = 0.025, gamma = 1.0,
        idio_income = [0.5, 1.5], idio_transition = [0.9 0.1; 0.1 0.9],
        aggregate_z = [0.93, 1.07], aggregate_transition = [0.7 0.3; 0.3 0.7],
        capital_grid = collect(range(0.0, 20.0; length = hp.grid_size)),
    )
    history = sequence_ks_history(params; history_length = hp.history_length, z_index = 2)
    distribution = sequence_ks_initial_distribution(params; K_target = 5.0)
    history_dim = size(history, 1) * hp.history_length
    model = make_mlp(history_dim, (16, 16), length(params.idio_income); activation = NNlib.tanh)
    state = setup_training(rng_from_seed(SEED; offset = 1), model, Optimisers.Adam(0.003); parameter_type = Float64)
end

# ╔═╡ c93a86eb-e394-62ee-81dc-09f25adeb0f3
md"""
---
## Part 4: The Fischer-Burmeister Loss

### Why focus on Euler / KKT conditions?

In practice, directly maximizing discounted utility often gives weaker Euler-equation accuracy than directly training on the first-order conditions. We therefore target the **Euler / KKT residual** itself.

### The Fischer-Burmeister function

The household's optimality has two cases:
- **Interior** (\$k' > 0\$): the Euler equation holds with equality, i.e. \$g = 0\$ where \$g = (c_\text{Euler} - c)/c\$
- **Constrained** (\$k' = 0\$): the Euler gap \$g \geq 0\$

The Fischer-Burmeister function encodes both in one smooth formula:
\$\$\text{FB}(g, s) = \sqrt{g^2 + s^2 + \epsilon} - g - s,\$\$
where \$s = k'/c\$ is the relative savings slack. For log utility, \$u'(c) = 1/c\$, so \$u'^{-1}(q) = 1/q\$. Then \$\text{FB} = 0\$ if and only if the KKT conditions are satisfied: either \$g = 0, s \geq 0\$ (interior) or \$g \geq 0, s = 0\$ (constrained).

> **In this preview.** The shared `sequence_ks_residual` helper evaluates the Euler residual in **absolute** units — the marginal-utility gap \$u'(c) - \beta\,\mathbb{E}_t[R'\,u'(c')]\$ and the absolute savings slack \$k' - k_{\min}\$ — rather than the consumption-relative \$g = (c_\text{Euler}-c)/c\$ and \$s = k'/c\$ written above (which follow the Python original); next-period consumption \$c'\$ is interpolated at each household's chosen savings \$k'\$ with Young lottery weights. It also reports a **mass-weighted** mean \$\text{FB}^2\$ (weighted by the household distribution \$\mu\$) plus a small capital-market-clearing penalty \$\big((K' - K)/(1 + |K|)\big)^2\$, whereas the full Python notebook averages \$\text{FB}^2\$ uniformly with no market-clearing term.

---
## Part 5: Training

The next cell defines the shape diagnostic and the training loss, then runs the optimisation. Each step evaluates the Fischer-Burmeister / Euler residual through `sequence_ks_residual` and takes an `Optimisers.jl` Adam step via `train_step!` (parameter gradients from Zygote); the loss is threaded through Lux with the explicit `model(x, ps, st)` state pattern.

In the full Python tutorial each epoch (i) samples aggregate histories from a **replay buffer**, (ii) simulates them forward with Young's method to evolve the buffer, and (iii) takes several FB gradient steps. This compact CPU preview instead trains on a single coupled `(history, distribution)` pair at smoke-size budgets, so the loop is short and classroom-friendly; the replay-buffer / Young simulation is described in Part 3 below.
"""

# ╔═╡ 55555555-10ff-4555-8555-555555555555
begin
    function shape_report(consumption, cash)
        first_diff = diff(consumption; dims = 2)
        second_diff = size(first_diff, 2) > 1 ? diff(first_diff; dims = 2) : zero(first_diff)
        return (
            feasible = all(consumption .> 0) && all(consumption .< cash),
            monotone = all(first_diff .>= -1e-8),
            concave = all(second_diff .<= 1e-8),
        )
    end

    tutorial_loss(model, ps, st, batch) = begin
        pieces, st_new = sequence_ks_residual(model, ps, st, batch.history, batch.distribution; params)
        return pieces.loss, st_new
    end

    initial_loss = loss_value(state, tutorial_loss, (history = history, distribution = distribution))
    for _ in 1:hp.steps
        train_step!(state, tutorial_loss, (history = history, distribution = distribution); max_grad_norm = 10.0)
    end
end

# ╔═╡ b519533b-67e5-4ddc-5f9e-814fa1d70678
md"""
---
## Part 3: Simulation with Young's (2010) Non-Stochastic Method

To evaluate the Euler equation we need aggregate capital \$K_t\$ and labor \$L_t\$, which depend on the cross-sectional wealth distribution. Instead of simulating a panel of individual households (Monte Carlo), we track the **full distribution** \$\mu(\varepsilon, k)\$ as a histogram on the \$(\varepsilon, k)\$ grid.

Each period, for every grid point \$(\varepsilon, k)\$:
1. Compute next-period capital: \$k' = w\varepsilon + Rk - c(\varepsilon, k)\$
2. \$k'\$ generally falls *between* two grid points. Split the mass \$\mu(\varepsilon, k)\$ to the two neighbors proportionally (**lottery**).
3. Apply the idiosyncratic transition matrix \$\Pi_\varepsilon\$ across income states.

**Key advantage**: aggregates \$K = \sum_{(\varepsilon,k)} k \cdot \mu(\varepsilon,k)\$ and \$L = \sum \varepsilon \cdot \mu\$ are exact weighted sums — **no sampling noise**. This eliminates the Monte Carlo noise floor (~\$10^{-6}\$) that limits accuracy with household panels.

The next cell recomputes the trained residual, extracts the policy grid, checks the shape guarantees, and applies **one** Young forward step (`sequence_ks_forward_step`), then reports the propagated aggregates (`sequence_ks_distribution_aggregates`). The Python tutorial stores coupled `(Z-history, μ)` pairs in a replay buffer and simulates forward \$T\$ periods; here we take a single step to demonstrate the Young propagation.

### The full Python notebook also covers (Parts 6 & 7)

- **Results (Part 6).** Convergence of FB² over epochs and the learned consumption / MPC policy functions for each income state.
- **Euler-error diagnostics (Part 7).** Per-grid-point Euler gap \$|g|\$ and Fischer-Burmeister residual \$|FB|\$: consumption and MPC scatter clouds versus aggregate capital \$K\$ (a visual test of approximate aggregation), and median with 10th–90th-percentile error bands of \$|FB|\$ across the individual capital grid for each income state \$\varepsilon\$.

This Julia preview does not render those figures; instead the final cell returns the underlying numbers — the Euler residual RMSE, the shape-guarantee flags, and the propagated aggregates — as a machine-checkable `NamedTuple`.
"""

# ╔═╡ 66666666-10ff-4666-8666-666666666666
begin
    diagnostics, _ = sequence_ks_residual(state.model, state.ps, state.st, history, distribution; params)
    tutorial_policy, _ = sequence_ks_policy_grid(state.model, state.ps, state.st, history, distribution; params)
    checks = shape_report(tutorial_policy.consumption, tutorial_policy.cash)
    history_next, distribution_next, _ = sequence_ks_forward_step(state.model, state.ps, state.st, history, distribution, 1; params)
    agg_next = sequence_ks_distribution_aggregates(distribution_next, params)
end

# ╔═╡ 18770a99-ddbf-ce77-28c3-4d080a2a97f4
md"""
---
## Summary

In this notebook we:

1. **Represented** the consumption policy with a constant per-income-state MPC head (sigmoid-bounded) in this preview — the full Python notebook uses an I-spline MPC basis — so the grid policy is increasing, concave, and feasible by construction.
2. **Evaluated** equilibrium conditions with a Fischer-Burmeister residual that handles both interior and constrained households.
3. **Trained** on an aggregate shock history coupled with a household distribution.
4. **Diagnosed** solution quality with the Euler residual and the shape-guarantee flags.

**Takeaways**
- The MPC representation (a constant per-income-state head here, an I-spline basis in the full notebook) is a concrete example of how economic shape restrictions can be built directly into a neural network.
- The distribution, evolved by Young's method, is the bridge between individual policies and aggregate prices.
- Consumption vs \$K\$ at fixed \$k\$ visualizes approximate aggregation — tight clouds mean the policy depends mainly on prices, not on the full distribution.
- The Euler/FB error is best in the interior and worst near the constraint and at high \$k\$ with little household mass.

**A few intentional simplifications**
- two-state aggregate and idiosyncratic Markov chains,
- log utility,
- the initial household distribution centered at a fixed target capital (`K_target = 5.0`) rather than a computed deterministic \$K_{ss}\$ benchmark line (not rendered in this preview),
- and, in this CPU preview, smoke-size budgets with a single `(history, distribution)` pair rather than the full replay-buffer training loop.

The cell below returns the machine-checkable diagnostics summary for this run — the initial and final loss, the feasible / monotone / concave shape flags, the Euler RMSE, the propagated aggregates, and the flattened shock-history dimensions at the Lux boundary.
"""

# ╔═╡ 77777777-10ff-4777-8777-777777777777
(
    initial_loss = initial_loss,
    final_loss = diagnostics.loss,
    feasible = checks.feasible,
    monotone = checks.monotone,
    concave = checks.concave,
    euler_rmse = residual_summary(diagnostics.euler).rmse,
    next_mass = agg_next.mass,
    next_capital = agg_next.capital,
    history_shape = size(history_next),
    history_feature_dim = size(history_next, 1),
    flattened_history_dim = size(flatten_history(history_next), 1),
)

# ╔═╡ Cell order:
# ╟─11111111-10ff-4111-8111-111111111111
# ╟─87008503-438e-da0f-7dc2-4a17db187b18
# ╟─7f99fc7c-ab8e-7a72-92d6-4ed915b9fb02
# ╠═22222222-10ff-4222-8222-222222222222
# ╠═33333333-10ff-4333-8333-333333333333
# ╟─accd9dfa-0ef9-72d7-a385-1c1be5858d33
# ╟─1bf58e2e-ba4f-2bf0-1100-fe83827052ef
# ╠═44444444-10ff-4444-8444-444444444444
# ╟─c93a86eb-e394-62ee-81dc-09f25adeb0f3
# ╠═55555555-10ff-4555-8555-555555555555
# ╟─b519533b-67e5-4ddc-5f9e-814fa1d70678
# ╠═66666666-10ff-4666-8666-666666666666
# ╟─18770a99-ddbf-ce77-28c3-4d080a2a97f4
# ╠═77777777-10ff-4777-8777-777777777777
