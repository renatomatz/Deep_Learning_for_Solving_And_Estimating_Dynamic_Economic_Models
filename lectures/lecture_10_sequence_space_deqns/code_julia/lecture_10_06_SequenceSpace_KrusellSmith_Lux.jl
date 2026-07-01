### A Pluto.jl notebook ###
# v0.20.0

using Markdown
using InteractiveUtils

# ╔═╡ 11111111-1006-4111-8111-111111111111
md"""
# Lecture 10, Notebook 06: Sequence-Space Krusell-Smith in Lux

A classroom-scale Krusell-Smith sequence-space actor. The network maps aggregate
shock histories to idiosyncratic MPC heads; Young's method propagates the
cross-sectional distribution outside the gradient step.
"""

# ╔═╡ 9f0bd55d-0137-5b1a-4e4a-6b044cdaff9f
md"""
## Lecture 10: Sequence-space DEQNs

**Course:** Deep Learning for Solving and Estimating Dynamic Models in Economics and Finance
**Script reference:** §6.7 (Sequence-space DEQNs) — Krusell-Smith implementation
**Notebook role:** core
**Author:** Simon Scheidegger

*Julia/Lux/Pluto preview of* `lectures/lecture_10_sequence_space_deqns/code/lecture_10_06_SequenceSpace_KrusellSmith.ipynb`.
"""

# ╔═╡ 0eb526e0-9b46-6483-3481-e4f292d76ec2
md"""
## Sequence-Space DEQN: Krusell-Smith Economy

This notebook solves the **Krusell-Smith (1998)** heterogeneous-agent economy using the **sequence-space Deep Equilibrium Net** approach:

| Component | Method |
|---|---|
| **Policy** | I-spline MPC basis \$\Rightarrow\$ monotone, concave \$c(k)\$ by construction |
| **Distribution** | Young (2010) non-stochastic simulation on a fixed grid |
| **Optimality** | Fischer-Burmeister KKT residual for Euler + borrowing constraint |
| **Training** | Replay buffer of (Z-history, \$\mu\$) pairs; mini-batch SGD on FB\$^2\$ |

**Framework:** this Julia preview uses **Lux** for the actor, **Zygote** for the policy-gradient path, and **Optimisers.jl** (`Optimisers.Adam`, `train_step!`) for the updates; Young's-method distribution propagation runs in plain Julia arrays outside the gradient step.

**References:** Azinovic-Yang & Žemlička (2025), *Deep learning in the sequence space* (arXiv:2509.13623) — the sequence-space DEQN method this notebook implements; Azinovic, Gaegauf & Scheidegger (2022), *International Economic Review* 63(4) — the underlying deep-equilibrium-net framework.
"""

# ╔═╡ 22222222-1006-4222-8222-222222222222
begin
    import Pkg
    Pkg.activate(joinpath(@__DIR__, "..", "..", "..", "julia"))

    using DLEFJulia
    using Lux
    using NNlib
    using Optimisers
end

# ╔═╡ 33333333-1006-4333-8333-333333333333
begin
    RUN_MODE = "smoke"
    SEED = 0
    budgets = (
        smoke = (steps = 5, grid_size = 24, history_length = 6),
        teaching = (steps = 200, grid_size = 64, history_length = 25),
        production = (steps = 1_500, grid_size = 128, history_length = 80),
    )
    hp = run_mode_budget(RUN_MODE; budgets)
end

# ╔═╡ 14e6a1b0-8991-087b-5ac4-1aef9cc6d356
md"""
### Calibration and capital grid

The economy uses capital share \$\alpha = 0.36\$, depreciation \$\delta = 0.025\$, log utility, and Cobb-Douglas production, with two-state aggregate and idiosyncratic Markov chains. These live in `SequenceKSParams`.

**Capital grid.** The Python notebook uses a **log-spaced** grid to concentrate points near \$k = 0\$, where the policy is steepest (high MPC for poor households). This compact Julia preview uses a uniform grid `range(0.0, 25.0; length = hp.grid_size)` whose size scales with `RUN_MODE` (24 points in `smoke`, up to 128 in `production`).
"""

# ╔═╡ 2ba0146f-03ec-39a7-a96a-9fdf6655a71c
md"""
### Neural network actor and the shape-restricted policy

A small MLP maps the encoded aggregate shock **history** to the coefficients that define the consumption policy on the capital grid. This is the sequence-space idea: the network reads the last \$H\$ aggregate shocks — a history tensor — rather than the current endogenous state. In Lux the history is a feature-by-history tensor that is **flattened at the model boundary** (`input_dim = size(history, 1) * hp.history_length`), and the actor is a plain `make_mlp` (`Lux.Chain`) called with the explicit `y, st = model(x, ps, st)` state-threading pattern.

**I-spline MPC basis (Python).** In the Python ground truth the MPC function \$\alpha(k)\$ is built from **I-spline** basis functions,
\$\$\text{MPC}(k) = \alpha \cdot \left(1 - \sum_j \tilde{w}_j \, I_j(k)\right),\$\$
where each \$I_j\$ increases monotonically from 0 to 1, so the MPC is decreasing (consumption concave) **by construction**.

> **Julia parity note.** This preview lets the Lux actor emit MPC heads directly (one per idiosyncratic income state) and **checks monotonicity/concavity after the fact** through the diagnostics, rather than guaranteeing them by an I-spline construction.
"""

# ╔═╡ 44444444-1006-4444-8444-444444444444
begin
    params = SequenceKSParams(
        beta = 0.93,
        delta = 0.025,
        gamma = 1.0,
        idio_income = [0.5, 1.5],
        idio_transition = [0.9 0.1; 0.1 0.9],
        aggregate_z = [0.93, 1.07],
        aggregate_transition = [0.7 0.3; 0.3 0.7],
        capital_grid = collect(range(0.0, 25.0; length = hp.grid_size)),
    )
    history = sequence_ks_history(params; history_length = hp.history_length, z_index = 1)
    distribution = sequence_ks_initial_distribution(params; K_target = 6.0)
    input_dim = size(history, 1) * hp.history_length
    output_dim = length(params.idio_income)
    model = make_mlp(input_dim, (20, 20), output_dim; activation = NNlib.tanh)
    state = setup_training(rng_from_seed(SEED; offset = 1), model, Optimisers.Adam(0.003); parameter_type = Float64)
end

# ╔═╡ b26f9025-3b9e-1633-e9b0-b33350a7cf40
md"""
### Simulation: Young's method

Young's (2010) **non-stochastic simulation** evolves the full distribution \$\mu(\varepsilon, k)\$ on the grid. Aggregates \$K, L\$ are exact weighted sums, eliminating Monte Carlo sampling noise. Here `sequence_ks_forward_step` performs this propagation in plain Julia arrays, **outside** the gradient step (the JAX/TF reference uses `stop_gradient` on the same paths), so the distribution update is never differentiated.
"""

# ╔═╡ 1073b855-234e-4e4e-077d-a38a66126177
md"""
### Fischer-Burmeister loss

The household's optimality has two cases:
- **Interior** (\$k' > 0\$): the Euler equation holds with equality, i.e. \$g = 0\$ where \$g = (c_{\text{Euler}} - c)/c\$.
- **Constrained** (\$k' = 0\$): the Euler gap \$g \geq 0\$.

The Fischer-Burmeister function encodes both in one smooth formula,
\$\$\text{FB}(g, s) = \sqrt{g^2 + s^2 + \epsilon} - g - s, \qquad s = k'/c,\$\$
and \$\text{FB} = 0\$ iff the KKT conditions are satisfied.

Prices and aggregates enter as fixed numbers (the `stop_gradient` analogue): only the actor calls are traced, so **Zygote** flows gradients through the policy network alone. `sequence_ks_residual` returns the pieces of this residual, and `pieces.loss` is what training minimizes.

> **Julia parity note.** The shared residual also computes a borrowing-constraint **complementarity diagnostic**, but that term is reported for inspection only and is **not** included in the `pieces.loss` that the optimizer sees.
"""

# ╔═╡ 91544713-42a2-375e-2455-c3f827f1b814
md"""
### Training

Each training step:
1. Evaluates the FB loss on the current (Z-history, \$\mu\$) pair and updates the actor with `train_step!` (Adam, gradient-norm clipping).
2. Advances the aggregate shock one period and propagates \$\mu\$ forward with Young's method (`sequence_ks_forward_step`), producing the next state.
3. Logs the loss and the resulting aggregate capital and distribution mass.

The full Python notebook maintains a **replay buffer** of (Z-history, \$\mu\$) pairs with FIFO eviction and draws mini-batches from it. This compact preview instead threads a single distribution forward at smoke-size budgets (`hp.steps` = 5 in `smoke`), reusing the same loss and propagation machinery.
"""

# ╔═╡ 55555555-1006-4555-8555-555555555555
begin
    ks_loss(model, ps, st, batch) = begin
        pieces, st_new = sequence_ks_residual(model, ps, st, batch.history, batch.distribution; params)
        return pieces.loss, st_new
    end

    train_result = let history_local = history, distribution_local = distribution
        initial_loss_local = loss_value(state, ks_loss, (history = history_local, distribution = distribution_local))
        history_log_local = NamedTuple[]
        for step in 1:hp.steps
            metrics = train_step!(state, ks_loss, (history = history_local, distribution = distribution_local); max_grad_norm = 10.0)
            z_next = isodd(step) ? 2 : 1
            history_local, distribution_local, _ = sequence_ks_forward_step(state.model, state.ps, state.st, history_local, distribution_local, z_next; params)
            agg = sequence_ks_distribution_aggregates(distribution_local, params)
            append_metric!(history_log_local; step, loss = metrics.loss, capital = agg.capital, mass = agg.mass)
        end
        (initial_loss = initial_loss_local, history_log = history_log_local, history = history_local, distribution = distribution_local)
    end
    initial_loss = train_result.initial_loss
    history_log = train_result.history_log
    history = train_result.history
    distribution = train_result.distribution
end

# ╔═╡ cb8d866b-538f-71ac-3094-e8a99b0636b6
md"""
### Results and learned policy

After training we read the learned policy off the grid with `sequence_ks_policy_grid` and evaluate the equilibrium residual one more time. The Python notebook plots convergence of the FB\$^2\$ loss and the learned consumption/MPC policy functions \$c(k)\$ against the capital grid; this preview instead returns the same quantities as machine-checkable numbers in the final cell.
"""

# ╔═╡ dd081c5e-9b80-cb09-5960-d6714a8bcc8a
md"""
### Diagnostics

We evaluate per-grid-point Euler gaps and FB residuals across the replay-buffer state(s). The full Python notebook renders two diagnostic views:
1. **Consumption and MPC vs \$K\$**: scatter plots at selected \$k\$ values (tight clouds = approximate aggregation holds).
2. **Euler / FB error bands**: median and 10th–90th percentile of \$|FB|\$ across the capital grid.

Here the diagnostics feed `residual_summary` (Euler RMSE), the aggregate capital/mass, the capital-market gap, and the implied prices \$R, w\$ — all returned by the final cell below.
"""

# ╔═╡ 66666666-1006-4666-8666-666666666666
begin
    diagnostics, _ = sequence_ks_residual(state.model, state.ps, state.st, history, distribution; params)
    ks_policy, _ = sequence_ks_policy_grid(state.model, state.ps, state.st, history, distribution; params)
end

# ╔═╡ a43027a9-dc3f-a4c1-4cd8-084ef0e0e9ba
md"""
### Summary

This notebook solved the Krusell-Smith (1998) heterogeneous-agent economy using the sequence-space DEQN approach — here in Julia/Lux:

1. **Represented** the consumption policy with an MLP actor reading the aggregate shock history (an I-spline MPC basis in the Python ground truth).
2. **Evaluated** equilibrium conditions with a Fischer-Burmeister residual that handles both interior and constrained households.
3. **Trained** on aggregate histories and household distributions propagated by Young's method.
4. **Diagnosed** solution quality with per-grid-point Euler errors and aggregate/price checks.

**Takeaways**
- The MPC representation is a concrete example of how economic shape restrictions can be built into (or checked on) a neural-network policy.
- The distribution propagation is the bridge between individual policies and aggregate prices.
- Consumption vs \$K\$ at fixed \$k\$ visualizes approximate aggregation — tight clouds mean the policy depends mainly on prices, not on the full distribution.
- The Euler/FB error bands reveal where accuracy is best (interior \$k\$) and worst (near constraints and at high \$k\$ with zero household mass).

**A few intentional simplifications**
- two-state aggregate and idiosyncratic Markov chains,
- a fixed-length Z-history (rather than an RNN encoder),
- Cobb-Douglas production and log utility.

Each could be relaxed without changing the algorithm's structure. The cell below returns the machine-checkable diagnostics summary for this notebook's smoke run.
"""

# ╔═╡ 77777777-1006-4777-8777-777777777777
(
    initial_loss = initial_loss,
    final_loss = diagnostics.loss,
    euler_rmse = residual_summary(diagnostics.euler).rmse,
    distribution_mass = diagnostics.aggregate.mass,
    aggregate_capital = diagnostics.aggregate.capital,
    capital_market = diagnostics.capital_market,
    interest_rate = ks_policy.prices.R,
    wage = ks_policy.prices.w,
    history_shape = size(history),
    history_feature_dim = size(history, 1),
    flattened_history_dim = size(flatten_history(history), 1),
    policy_shape = size(ks_policy.consumption),
)

# ╔═╡ Cell order:
# ╟─11111111-1006-4111-8111-111111111111
# ╟─9f0bd55d-0137-5b1a-4e4a-6b044cdaff9f
# ╟─0eb526e0-9b46-6483-3481-e4f292d76ec2
# ╠═22222222-1006-4222-8222-222222222222
# ╠═33333333-1006-4333-8333-333333333333
# ╟─14e6a1b0-8991-087b-5ac4-1aef9cc6d356
# ╟─2ba0146f-03ec-39a7-a96a-9fdf6655a71c
# ╠═44444444-1006-4444-8444-444444444444
# ╟─b26f9025-3b9e-1633-e9b0-b33350a7cf40
# ╟─1073b855-234e-4e4e-077d-a38a66126177
# ╟─91544713-42a2-375e-2455-c3f827f1b814
# ╠═55555555-1006-4555-8555-555555555555
# ╟─cb8d866b-538f-71ac-3094-e8a99b0636b6
# ╟─dd081c5e-9b80-cb09-5960-d6714a8bcc8a
# ╠═66666666-1006-4666-8666-666666666666
# ╟─a43027a9-dc3f-a4c1-4cd8-084ef0e0e9ba
# ╠═77777777-1006-4777-8777-777777777777
