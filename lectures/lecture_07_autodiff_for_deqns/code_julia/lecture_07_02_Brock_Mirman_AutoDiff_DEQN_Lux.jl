### A Pluto.jl notebook ###
# v0.20.0

using Markdown
using InteractiveUtils

# ╔═╡ 11111111-0702-4111-8111-111111111111
md"""
# Lecture 07, Notebook 02: Brock-Mirman Autodiff DEQN in Julia

The Python notebook uses two TensorFlow `GradientTape` blocks on a single
period-payoff primitive. This Pluto translation keeps that teaching object:
write `Pi(K_in, K_out)`, take derivatives with respect to the choice slot and
state slot, and compare the resulting Euler residual with the hand-derived
Brock-Mirman residual used elsewhere in the course.
"""

# ╔═╡ 77a78cab-97f6-5778-37e6-d6b3c31bb2dd
md"""
## Lecture 07, Notebook 02: Brock–Mirman via Autodiff (Deterministic)

**Course:** Deep Learning for Solving and Estimating Dynamic Models in Economics and Finance
**Script reference:** §2.7.3 (the autodiff Euler residual; Listing for the deterministic Brock–Mirman case)
**Notebook role:** core
**Author:** Simon Scheidegger

*Julia/Lux/Pluto preview of* `lectures/lecture_07_autodiff_for_deqns/code/lecture_07_02_Brock_Mirman_AutoDiff_DEQN.ipynb`.

> **Run mode.** The checked-in run uses `RUN_MODE = "smoke"` and `SEED = 0` (here 20 training steps); set `RUN_MODE` to `"teaching"` or `"production"` for the longer budgets. The two TensorFlow `GradientTape` calls of the Python notebook become `ForwardDiff` slot derivatives of the period payoff for the Euler residual, plus `Zygote` for the parameter gradient during training.
"""

# ╔═╡ 3b4bf433-c545-de6e-9cf6-180f5f0f0cd6
md"""
## Deep Equilibrium Nets via Automatic Differentiation

### Notebook 2: replacing the FOC and envelope theorem with autodiff

#### Purpose of the notebook
Notebooks **01** and **02** solve the Brock–Mirman (1972) growth model with [Deep Equilibrium Nets](https://onlinelibrary.wiley.com/doi/full/10.1111/iere.12575) (Azinovic et al., 2022). In the hand-derived version, the user does *two* derivations on paper before writing any code:

1. **First-order condition (FOC)** of the Bellman objective with respect to \$K_{t+1}\$.
2. **Envelope theorem** to eliminate the unknown derivative \$V'(K_{t+1})\$.

The resulting Euler equation is then hard-coded as the loss. If one changes the utility function (log \$\to\$ CRRA) or the production function (Cobb–Douglas \$\to\$ CES), the algebra has to be redone.

**This notebook removes both manual steps.** The user writes only the *period payoff* \$\Pi(K_t, K_{t+1}) = u(C_t)\$ as a single primitive function. Automatic differentiation then computes both the FOC term and the envelope term as gradients of \$\Pi\$ — in Python via `tf.GradientTape`, here via `ForwardDiff`.

We solve the **same model as notebook 01** (deterministic, full depreciation, log utility) so that we can verify the autodiff loss against (i) the closed-form analytical policy and (ii) the hand-derived loss of notebook 01 — at machine precision.

The setup cell below loads Lux, DLEFJulia, `ForwardDiff`, and the optimiser stack that replaces the Python notebook's NumPy/TensorFlow/Keras imports.
"""

# ╔═╡ 22222222-0702-4222-8222-222222222222
begin
    import Pkg
    Pkg.activate(joinpath(@__DIR__, "..", "..", "..", "julia"))

    using CairoMakie
    using DLEFJulia
    using ForwardDiff
    using Lux
    using NNlib
    using Optimisers
    using Statistics
end

# ╔═╡ 33333333-0702-4333-8333-333333333333
begin
    RUN_MODE = "smoke"
    SEED = 0
    budgets = (
        smoke = (steps = 20, batch_size = 32, grid_size = 25),
        teaching = (steps = 300, batch_size = 64, grid_size = 100),
        production = (steps = 2_000, batch_size = 128, grid_size = 200),
    )
    hp = run_mode_budget(RUN_MODE; budgets)
    rng = rng_from_seed(SEED)
end

# ╔═╡ 47c38eed-b8c8-7888-b5ce-a12b292d4909
md"""
#### The model (recap of notebook 01)
The planner solves
\$\$\max_{\{C_t\}} \sum_{t=0}^{\infty} \beta^t \ln(C_t) \quad\text{s.t.}\quad K_{t+1} + C_t = Y_t + (1-\delta)K_t,\qquad Y_t = K_t^{\alpha}.\$\$
With full depreciation \$\delta = 1\$ this admits the analytical solution \$K_{t+1} = \alpha\beta\,K_t^{\alpha}\$.

Recursively,
\$\$V(K_t) = \max_{K_{t+1}} \;\underbrace{\ln(Y_t + (1-\delta)K_t - K_{t+1})}_{=\,\Pi(K_t,\,K_{t+1})} \;+\; \beta\, V(K_{t+1}).\$\$

The economic parameters (\$\alpha\$, \$\beta\$, \$\delta\$) are constants throughout, held in a `BrockMirmanParams`. Because the full-depreciation case is solvable in closed form, the cell below also builds `analytic_policy` and the steady state `k_star`, so we can later check the neural-network solution against them.
"""

# ╔═╡ 44444444-0702-4444-8444-444444444444
begin
    params = BrockMirmanParams(alpha = 0.36, beta = 0.99, delta = 1.0)
    analytic_policy(k) = bm_full_depreciation_policy(k, params)
    k_star = bm_steady_state(params)
end

# ╔═╡ cbbb3e91-1f22-7250-8ed4-8682101a6dd2
md"""
#### Deep neural network

We want the network to approximate the savings rate \$s_t\$, so that \$K_{t+1} = Y_t\, s_t \approx Y_t\,\mathcal{N}(K_t)\$. The input is the 1-dimensional state \$K_t\$ and the output is the 1-dimensional savings rate \$s_t\$.

Following [Azinovic et al. (2022)](https://onlinelibrary.wiley.com/doi/full/10.1111/iere.12575), we use a densely connected feed-forward network with two hidden layers (ReLU), and — because we approximate a savings *rate* — a **sigmoid** output so that \$s_t \in (0, 1)\$. This encodes economic prior knowledge directly into the architecture. The Python notebook builds this in Keras; the Julia preview builds it with `make_mlp` (Lux) plus a `NNlib.sigmoid` `savings_transform`, and creates the Adam optimiser through `setup_training` (the `Optimisers.jl` replacement for `tf.keras.optimizers.Adam`).

> **The full Python notebook also** points to related work on baking economic structure into the network architecture: Kahou et al. (2021) and Han et al. (2022) show how *symmetry* can be encoded into the neural-network architecture, and Azinovic and Zemlicka (2023) introduce *market-clearing* neural-network architectures. This preview keeps only the Azinovic et al. (2022) pointer above.

**The batch dimension.** Networks are highly parallelisable, so we evaluate on a whole vector of capital levels at once rather than a single \$K\$. At the Lux boundary these are **feature-by-batch** arrays (features on the first axis, samples on the second) — the transpose of the deep-learning convention of samples on the 0-axis. The explicit `y, st = model(x, ps, st)` call threads Lux parameters and state through every evaluation.
"""

# ╔═╡ 1e7b6d8c-d1e1-7258-19d9-ced342092772
md"""
#### Hard vs. soft constraints — the central design choice in DEQNs

Two kinds of equilibrium conditions appear in any dynamic stochastic model:

- **Inequality / feasibility constraints** — e.g. \$C_t > 0\$, \$K_{t+1} > 0\$, the resource constraint \$C_t + K_{t+1} = Y_t\$. These must hold *exactly*.
- **Optimality conditions** — e.g. the Euler equation. These hold in the equilibrium but not at every intermediate guess of the policy.

Azinovic, Gaegauf & Scheidegger (2022, §4.2.2; lecture script Fig. 2.3) make this distinction explicit and treat the two kinds very differently:

| | Hard constraint (architecture) | Soft constraint (loss) |
|--|--|--|
| **What** | Built into the network output | Penalised in the cost function |
| **How** | Activation choice + algebraic identities | Squared residuals in \$\mathcal{L}\$ |
| **Cost** | Always satisfied — even at random init | Only satisfied at convergence |

**Why this matters here.** The `savings_transform` parameterises the savings *share* \$s_t \in (0, 1)\$ via a **sigmoid** output. Combined with the resource constraint \$K_{t+1} = s_t Y_t\$ and \$C_t = (1 - s_t) Y_t\$, this **guarantees \$C_t > 0\$ and \$K_{t+1} > 0\$ simultaneously**, at every iteration of training. We never have to penalise infeasibility — the architecture rules it out. The Euler equation, by contrast, is enforced softly through the loss.

This split removes a whole class of bad local minima (network outputs that would imply \$C_t < 0\$) and is one reason DEQNs converge in regions where naive penalty methods do not.
"""

# ╔═╡ 55555555-0702-4555-8555-555555555555
begin
    model = make_mlp(1, (24, 24), 1; activation = NNlib.relu)
    train_state = setup_training(rng_from_seed(SEED; offset = 1), model, Optimisers.Adam(0.005); parameter_type = Float64)
    savings_transform = NNlib.sigmoid
end

# ╔═╡ 87633e08-c10d-87fb-fc6c-dafc5b448afc
md"""
#### Why we no longer need pen-and-paper FOC + envelope

> **Ordering note.** In the full Python notebook this FOC + envelope motivation appears right after the model recap, before the network is introduced; here it sits adjacent to its payoff-implementation code for readability.

Define the **period payoff**
\$\$\Pi(K_{\text{in}},\,K_{\text{out}}) \;=\; u\!\big(\,Y(K_{\text{in}}) + (1-\delta)K_{\text{in}} - K_{\text{out}}\,\big),\$\$
where \$K_{\text{in}}\$ is the *state* (today's capital) and \$K_{\text{out}}\$ is the *choice* (tomorrow's capital). This is the *only primitive* the user writes — here `bm_payoff` / the `BMPayoff` functor.

> **Notation — what do \$\partial_1\Pi\$ and \$\partial_2\Pi\$ mean?**
> The subscript names the *slot being differentiated*, not a time index.
> - \$\partial_1 \Pi \;=\; \dfrac{\partial \Pi}{\partial K_{\text{in}}}\$  — derivative w.r.t. the **1st argument** of `Pi`, i.e. the *state*.
> - \$\partial_2 \Pi \;=\; \dfrac{\partial \Pi}{\partial K_{\text{out}}}\$ — derivative w.r.t. the **2nd argument** of `Pi`, i.e. the *choice*.
>
> So \$\partial_2\Pi(K_t,K_{t+1})\$ is the derivative of \$\Pi\$ in its **second slot**, evaluated with \$K_t\$ in slot 1 and \$K_{t+1}\$ in slot 2, i.e. \$\partial\Pi/\partial K_{t+1}\$.
> And \$\partial_1\Pi(K_{t+1},K_{t+2})\$ is the derivative of \$\Pi\$ in its **first slot**, evaluated with \$K_{t+1}\$ in slot 1 and \$K_{t+2}\$ in slot 2, i.e. \$\partial\Pi/\partial K_{t+1}\$ (different slot, *same* physical variable!).

Two clean facts then deliver everything:

| Step | Hand derivation (notebook 01) | Autodiff |
|---|---|---|
| FOC w.r.t. the *choice* \$K_{t+1}\$ | \$-u'(C_t) + \beta V'(K_{t+1}) = 0\$ | \$\partial_2 \Pi(K_t, K_{t+1}) \equiv \dfrac{\partial \Pi}{\partial K_{\text{out}}}\Big|_{(K_t,K_{t+1})}\$ via `ForwardDiff` |
| Envelope: \$V'(K_t)\$ at the *state* \$K_t\$ | \$u'(C_t)\,(Y'(K_t) + 1 - \delta)\$ | \$\partial_1 \Pi(K_t, K_{t+1}) \equiv \dfrac{\partial \Pi}{\partial K_{\text{in}}}\Big|_{(K_t,K_{t+1})}\$ via `ForwardDiff` |

Substituting the envelope at \$K_{t+1}\$ into the FOC gives the **autodiff-ready Euler residual**
\$\$\boxed{\;\underbrace{\partial_2 \Pi(K_t, K_{t+1})}_{\text{derivative w.r.t. }K_{t+1}\text{ (the choice)}} \;+\; \beta\,\underbrace{\partial_1 \Pi(K_{t+1}, K_{t+2})}_{\text{derivative w.r.t. }K_{t+1}\text{ (now the state of the next period)}} \;=\; 0\;}\$\$

Both terms are partial derivatives of the **same function \$\Pi\$**, just differentiated in different *slots* and evaluated at different time pairs. `ForwardDiff` computes them; we never write \$u'\$ or \$Y'\$ explicitly. *The envelope theorem is delivered operationally*, by evaluating \$\partial_1 \Pi\$ (the derivative w.r.t. the state slot) at the optimal \$K_{t+1}\$.

The cell below writes \$\Pi\$ once and exposes the two slot derivatives as `partial_state` (\$\partial_1\Pi\$) and `partial_choice` (\$\partial_2\Pi\$), both read off a single `ForwardDiff.gradient` of the payoff.

> **Why this matters pedagogically.** This is the cleanest demonstration of why automatic differentiation makes DEQN-style methods so powerful: the user only writes the model primitives; *all* derivative-based optimality conditions emerge from the same `Pi`.
"""

# ╔═╡ 66666666-0702-4666-8666-666666666666
begin
    function bm_payoff(k_in, k_out; params)
        output = k_in^params.alpha
        consumption = output + (1 - params.delta) * k_in - k_out
        return log(consumption)
    end

    struct BMPayoff{Alpha,Delta} end

    (::BMPayoff{Alpha,Delta})(v) where {Alpha,Delta} =
        log(v[1]^Alpha + (1 - Delta) * v[1] - v[2])

    payoff_partials(k_in, k_out; params) =
        ForwardDiff.gradient(BMPayoff{params.alpha,params.delta}(), [k_in, k_out])

    partial_state(k_in, k_out; params) = payoff_partials(k_in, k_out; params)[1]

    partial_choice(k_in, k_out; params) = payoff_partials(k_in, k_out; params)[2]
end

# ╔═╡ b89c092e-1091-070b-5391-847433069ad6
md"""
#### Implementing the autodiff cost function
We build the cost in two steps:

**Step A — write \$\Pi\$.** This is the *only* place the model enters. For CRRA utility, change `log(C)` to \$(C^{1-\gamma}-1)/(1-\gamma)\$; for CES production, change the output line. Nothing else moves.

**Step B — take the two slot derivatives** of \$\Pi\$ (recall: subscript = *slot of* \$\Pi\$):
- \$\partial_2 \Pi(K_t, K_{t+1}) \equiv \partial \Pi / \partial K_{\text{out}}\$ at the pair \$(K_t, K_{t+1})\$ — the **FOC term** (derivative w.r.t. the choice).
- \$\partial_1 \Pi(K_{t+1}, K_{t+2}) \equiv \partial \Pi / \partial K_{\text{in}}\$ at the pair \$(K_{t+1}, K_{t+2})\$ — the **envelope term** (derivative w.r.t. the state, evaluated at the optimum).

Then `loss = mean( (FOC term + β · envelope term)² )`. In Julia, `bm_policy_path` rolls the network forward one and two steps (all feasibility hard-coded through the sigmoid), and `autodiff_bm_residual` assembles \$\partial_2\Pi + \beta\,\partial_1\Pi'\$ from the `ForwardDiff` slot derivatives, returning both the raw residual and a relative form.
"""

# ╔═╡ 77777777-0702-4777-8777-777777777777
begin
    function bm_policy_path(model, ps, st, k_feature_batch; params, transform)
        k = assert_feature_batch(k_feature_batch, 1)

        raw, st1 = model(k, ps, st)
        savings = transform(raw)
        output = k .^ params.alpha
        next_capital = (1 - params.delta) .* k .+ output .* savings
        consumption = output .- next_capital .+ (1 - params.delta) .* k

        raw_next, st2 = model(next_capital, ps, st1)
        next_savings = transform(raw_next)
        next_output = next_capital .^ params.alpha
        next_next_capital = (1 - params.delta) .* next_capital .+ next_output .* next_savings
        next_consumption = next_output .- next_next_capital .+ (1 - params.delta) .* next_capital
        next_return = params.alpha .* next_capital .^ (params.alpha - 1)

        return (
            savings = savings,
            consumption = consumption,
            next_capital = next_capital,
            next_savings = next_savings,
            next_consumption = next_consumption,
            next_next_capital = next_next_capital,
            next_return = next_return,
        ), st2
    end

    function autodiff_bm_residual(model, ps, st, k_feature_batch; params, transform)
        k = assert_feature_batch(k_feature_batch, 1)
        path, st_new = bm_policy_path(model, ps, st, k; params, transform)

        dPi_dKout = reshape([
            partial_choice(k[i], path.next_capital[i]; params)
            for i in eachindex(k)
        ], size(k))

        dPi_dKin_next = reshape([
            partial_state(path.next_capital[i], path.next_next_capital[i]; params)
            for i in eachindex(k)
        ], size(k))

        raw_residual = dPi_dKout .+ params.beta .* dPi_dKin_next
        relative_residual = 1 .- (-dPi_dKout) ./ (params.beta .* dPi_dKin_next)

        diagnostics = merge(path, (
            loss = mean(abs2, raw_residual),
            residual = raw_residual,
            relative_residual = relative_residual,
            dPi_dKout = dPi_dKout,
            dPi_dKin_next = dPi_dKin_next,
        ))
        return diagnostics, st_new
    end
end

# ╔═╡ 41ca007e-b7cc-4de5-940f-a875eaa23628
md"""
#### Cross-check 1: autodiff vs hand-derived Euler residual
The hand-derived residual from notebook 01 is
\$\$r^{\text{hand}}(K_t) = -\frac{1}{C_t} + \frac{\beta}{C_{t+1}}\,(1 - \delta + r_{t+1}).\$\$
The autodiff residual is mathematically identical. The cell below evaluates both on the **same network** and reports the maximum absolute difference — and additionally checks the relative form against the shared library residual `deterministic_bm_residual`. We expect machine-precision agreement (~\$10^{-6}\$ in float32; tighter still here in `Float64`).
"""

# ╔═╡ 88888888-0702-4888-8888-888888888888
begin
    function hand_raw_bm_residual(model, ps, st, k_feature_batch; params, transform)
        path, st_new = bm_policy_path(model, ps, st, k_feature_batch; params, transform)
        residual = -1 ./ path.consumption .+
            params.beta .* (1 .- params.delta .+ path.next_return) ./ path.next_consumption
        return merge(path, (residual = residual, loss = mean(abs2, residual))), st_new
    end

    fixed_grid = reshape(collect(range(0.1, 1.0; length = hp.grid_size)), 1, :)
    auto0, _ = autodiff_bm_residual(train_state.model, train_state.ps, train_state.st, fixed_grid;
        params, transform = savings_transform)
    hand0, _ = hand_raw_bm_residual(train_state.model, train_state.ps, train_state.st, fixed_grid;
        params, transform = savings_transform)
    shared0, _ = deterministic_bm_residual(train_state.model, train_state.ps, train_state.st, fixed_grid;
        params, transform = savings_transform)

    raw_check_error = maximum(abs.(auto0.residual .- hand0.residual))
    shared_relative_check_error = maximum(abs.(auto0.relative_residual .- shared0.residual))
end

# ╔═╡ bed525e6-f94f-29c4-7c7f-4666838f8e9d
md"""
### Sampling data
We generate training data by sampling capital levels exogenously from an interval — here `sample_k` draws \$K \in [0.1, 1.0]\$ as a feature-by-batch row.
"""

# ╔═╡ f20dde88-7a45-2842-7772-8b5b9190ce54
md"""
### Training
We iteratively generate a fresh batch and update the network. Each step wraps the autodiff cost in `autodiff_loss` and calls `train_step!`, which computes the gradient of the loss w.r.t. the **network parameters** with `Zygote` (the reverse-mode counterpart to the Python notebook's outer `tf.GradientTape`) and applies an Adam update. Note the two distinct autodiff roles: `ForwardDiff` for the economic slot derivatives *inside* the residual, `Zygote` for the parameter gradient *of* the loss.
"""

# ╔═╡ 99999999-0702-4999-8999-999999999999
begin
    sample_k(rng, n) = reshape(0.1 .+ 0.9 .* rand(rng, n), 1, :)

    autodiff_loss(model, ps, st, k_batch) = begin
        pieces, st_new = autodiff_bm_residual(model, ps, st, k_batch; params, transform = savings_transform)
        return pieces.loss, st_new
    end

    initial_loss = loss_value(train_state, autodiff_loss, sample_k(rng, hp.batch_size))
    history = NamedTuple[]
    for _ in 1:hp.steps
        local batch = sample_k(rng, hp.batch_size)
        metrics = train_step!(train_state, autodiff_loss, batch; max_grad_norm = 10.0)
        append_metric!(history; step = metrics.step, loss = metrics.loss)
    end
end

# ╔═╡ 828491c4-3ba9-997e-7046-10be964704d3
md"""
### Cross-check 2: trained policy vs the analytical solution
With \$\delta = 1\$ the closed-form planner policy is \$K_{t+1} = \alpha\beta\,K_t^{\alpha}\$. We evaluate the trained network on a dense grid and report:

- the **policy curves** overlaid (NN vs analytical, plotted just below),
- the **relative \$L^2\$ error** of the policy against \$\alpha\beta K_t^{\alpha}\$,
- residual summary statistics over the test grid,
- and the trained-network re-run of both cross-check-1 comparisons.

Because the autodiff residual *is* the same Euler equation as in notebook 01, we expect the trained policy to match notebook 01's accuracy.
"""

# ╔═╡ aaaaaaaa-0702-4aaa-8aaa-aaaaaaaaaaaa
begin
    trained_auto, _ = autodiff_bm_residual(train_state.model, train_state.ps, train_state.st, fixed_grid;
        params, transform = savings_transform)
    trained_hand, _ = hand_raw_bm_residual(train_state.model, train_state.ps, train_state.st, fixed_grid;
        params, transform = savings_transform)
    trained_shared, _ = deterministic_bm_residual(train_state.model, train_state.ps, train_state.st, fixed_grid;
        params, transform = savings_transform)

    trained_raw_check_error = maximum(abs.(trained_auto.residual .- trained_hand.residual))
    trained_shared_relative_check_error = maximum(abs.(trained_auto.relative_residual .- trained_shared.residual))

    analytic_next = analytic_policy(fixed_grid)
    policy_relative_l2 = relative_l2_error(trained_auto.next_capital, analytic_next)
    residual_stats = residual_summary(trained_auto.residual)
end

# ╔═╡ bbbbbbbb-0702-4bbb-8bbb-bbbbbbbbbbbb
begin
    fig = Figure(size = figure_size(RUN_MODE))
    ax = Axis(fig[1, 1], xlabel = "K_t", ylabel = "K_{t+1}")
    lines!(ax, vec(fixed_grid), vec(fixed_grid); color = :gray55, linestyle = :dash, label = "45 degree")
    lines!(ax, vec(fixed_grid), vec(analytic_next); color = :black, linewidth = 3, label = "analytic")
    lines!(ax, vec(fixed_grid), vec(trained_auto.next_capital); color = :dodgerblue3, linewidth = 3, label = "autodiff DEQN")
    axislegend(ax; position = :lt)
    fig
end

# ╔═╡ 41004b91-28b5-7d93-c39c-ffe76dffee3d
md"""
### Takeaway

- We replaced two pages of pen-and-paper algebra (FOC + envelope theorem) with **two slot derivatives** of a single primitive `Pi(K_in, K_out)` (`ForwardDiff` here, two `tf.GradientTape` calls in Python), where the two slots are the *state* and the *choice*.
- Cross-check 1 verified that the autodiff residual is the same number as the hand-derived one (machine precision).
- Cross-check 2 verified that the trained policy matches the closed-form analytical solution to comparable accuracy as notebook 01.
- The pattern generalizes: any model whose period payoff can be written as \$\Pi(\text{state},\text{choice})\$ admits an autodiff Euler residual of the form
\$\$\partial_2 \Pi + \beta\,\mathbb{E}[\partial_1 \Pi] = 0,\$\$
where \$\partial_2\Pi\$ is the derivative in the **choice slot** and \$\partial_1\Pi\$ is the derivative in the **state slot**. Notebook 03 applies exactly this template to the stochastic version with AR(1) productivity.

The cell below returns this notebook's machine-checkable diagnostics NamedTuple.
"""

# ╔═╡ cccccccc-0702-4ccc-8ccc-cccccccccccc
(
    steady_state = k_star,
    initial_loss = initial_loss,
    final_loss = history[end].loss,
    raw_autodiff_vs_hand = raw_check_error,
    relative_autodiff_vs_shared = shared_relative_check_error,
    trained_raw_autodiff_vs_hand = trained_raw_check_error,
    trained_relative_autodiff_vs_shared = trained_shared_relative_check_error,
    policy_relative_l2 = policy_relative_l2,
    max_abs_raw_residual = residual_stats.max_abs,
)

# ╔═╡ Cell order:
# ╟─11111111-0702-4111-8111-111111111111
# ╟─77a78cab-97f6-5778-37e6-d6b3c31bb2dd
# ╟─3b4bf433-c545-de6e-9cf6-180f5f0f0cd6
# ╠═22222222-0702-4222-8222-222222222222
# ╠═33333333-0702-4333-8333-333333333333
# ╟─47c38eed-b8c8-7888-b5ce-a12b292d4909
# ╠═44444444-0702-4444-8444-444444444444
# ╟─cbbb3e91-1f22-7250-8ed4-8682101a6dd2
# ╟─1e7b6d8c-d1e1-7258-19d9-ced342092772
# ╠═55555555-0702-4555-8555-555555555555
# ╟─87633e08-c10d-87fb-fc6c-dafc5b448afc
# ╠═66666666-0702-4666-8666-666666666666
# ╟─b89c092e-1091-070b-5391-847433069ad6
# ╠═77777777-0702-4777-8777-777777777777
# ╟─41ca007e-b7cc-4de5-940f-a875eaa23628
# ╠═88888888-0702-4888-8888-888888888888
# ╟─bed525e6-f94f-29c4-7c7f-4666838f8e9d
# ╟─f20dde88-7a45-2842-7772-8b5b9190ce54
# ╠═99999999-0702-4999-8999-999999999999
# ╟─828491c4-3ba9-997e-7046-10be964704d3
# ╠═aaaaaaaa-0702-4aaa-8aaa-aaaaaaaaaaaa
# ╠═bbbbbbbb-0702-4bbb-8bbb-bbbbbbbbbbbb
# ╟─41004b91-28b5-7d93-c39c-ffe76dffee3d
# ╠═cccccccc-0702-4ccc-8ccc-cccccccccccc
