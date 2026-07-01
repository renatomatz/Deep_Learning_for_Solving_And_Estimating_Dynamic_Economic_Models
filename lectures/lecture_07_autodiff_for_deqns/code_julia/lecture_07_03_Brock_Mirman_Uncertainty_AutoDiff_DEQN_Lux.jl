### A Pluto.jl notebook ###
# v0.20.0

using Markdown
using InteractiveUtils

# ╔═╡ 11111111-0703-4111-8111-111111111111
md"""
# Lecture 07, Notebook 03: Stochastic Brock-Mirman via Autodiff in Julia

The Python notebook writes a three-slot payoff primitive
`Pi(K_in, K_out, z_in)` and obtains the stochastic Euler equation from two
slot derivatives:

```math
\partial_2 \Pi(K_t,K_{t+1},z_t) +
\beta E[\partial_1 \Pi(K_{t+1},K_{t+2},z_{t+1})] = 0.
```

This compact Pluto translation keeps the same residual mechanics on fixed
smoke grids. The Lux policy uses feature-by-batch states `[z; K]`, explicit
`model(x, ps, st)` calls, and Gauss-Hermite common-shock expectations.
"""

# ╔═╡ a330efd1-9fd1-047e-5ca6-e9183f8b519f
md"""
## Lecture 07, Notebook 03: Brock–Mirman via Autodiff (Stochastic)

**Course:** Deep Learning for Solving and Estimating Dynamic Models in Economics and Finance
**Script reference:** §2.7.3 (the autodiff Euler residual, stochastic case), §2.6 (Gauss–Hermite quadrature)
**Notebook role:** core
**Author:** Simon Scheidegger

*Julia/Lux/Pluto preview of* `lectures/lecture_07_autodiff_for_deqns/code/lecture_07_03_Brock_Mirman_Uncertainty_AutoDiff_DEQN.ipynb`.

> **Run mode.** The checked-in run uses `RUN_MODE = "smoke"` and `SEED = 0`. The two TensorFlow `GradientTape` calls of the Python notebook become `ForwardDiff` slot derivatives of the three-argument payoff `Pi(K_in, K_out, z_in)`. This compact preview validates the **residual mechanics** (autodiff vs hand-coded, and against the closed-form full-depreciation policy) on a fixed smoke grid with a random Lux policy; it does not run the parameter-training loop — see the `note` field in the final cell.
"""

# ╔═╡ 3b4bf433-c545-de6e-9cf6-180f5f0f0cd6
md"""
## Deep Equilibrium Nets via Automatic Differentiation

### Notebook 3: stochastic Brock-Mirman via autodiff

#### Purpose
This notebook is the stochastic counterpart of **notebook 02**. We solve the Brock-Mirman model with AR(1) total-factor-productivity and partial depreciation \$\delta = 0.1\$, but we replace the hand-derived FOC + envelope theorem by two slot derivatives of a single primitive `Pi(K_in, K_out, z_in)` — `tf.GradientTape` in Python, `ForwardDiff` here. `K_in` is today's *capital state*, `K_out` is the *choice* (tomorrow's capital), and `z_in` is the *exogenous shock*.

#### Model (recap of notebook 02)
\$\$\max_{\{C_t\}} \;\mathbb{E}\!\left[ \sum_{t=0}^{\infty} \beta^t \ln(C_t) \right] \quad \text{s.t.}\quad K_{t+1} + C_t = Y_t + (1-\delta)K_t,\;\; Y_t = z_t K_t^{\alpha},\;\; \log z_{t+1} = \rho \log z_t + \sigma \epsilon_{t+1}.\$\$
The state is \$\mathbf{X}_t = (z_t, K_t)\$, the policy is \$K_{t+1}=g(\mathbf{X}_t)\$. Recursively,
\$\$V(z_t, K_t) = \max_{K_{t+1}} \;\underbrace{\ln\!\big(\,z_t K_t^{\alpha} + (1-\delta)K_t - K_{t+1}\big)}_{=\,\Pi(K_t,K_{t+1},z_t)} + \beta\,\mathbb{E}\!\left[V(z_{t+1}, K_{t+1})\right].\$\$

#### What we will verify
1. **Cross-check vs the hand-derived residual** of notebook 02, to machine precision (\$\sim 10^{-6}\$ in float32; tighter in `Float64`). Confirms the autodiff loss *is* the same Euler equation.
2. **Side-experiment vs an analytical solution**: under full depreciation \$\delta=1\$, the stochastic Brock-Mirman model with log utility has the closed-form policy \$K_{t+1} = \alpha\beta\,z_t K_t^{\alpha}\$. Here we plug that closed form into a constant-savings policy and confirm the autodiff residual vanishes on it.

The setup cell below loads Lux, DLEFJulia, and `ForwardDiff` in place of the Python notebook's NumPy/TensorFlow/Keras imports.
"""

# ╔═╡ 22222222-0703-4222-8222-222222222222
begin
    import Pkg
    Pkg.activate(joinpath(@__DIR__, "..", "..", "..", "julia"))

    using CairoMakie
    using DLEFJulia
    using ForwardDiff
    using Lux
    using NNlib
    using Statistics
end

# ╔═╡ 33333333-0703-4333-8333-333333333333
begin
    RUN_MODE = "smoke"
    SEED = 0
    budgets = (
        smoke = (batch_size = 16, simulation_tracks = 4, simulation_periods = 5),
        teaching = (batch_size = 128, simulation_tracks = 32, simulation_periods = 50),
        production = (batch_size = 512, simulation_tracks = 128, simulation_periods = 200),
    )
    hp = run_mode_budget(RUN_MODE; budgets)
    rng = rng_from_seed(SEED)
end

# ╔═╡ 9fe050cb-4bb6-d21e-e24c-f336b7957b12
md"""
#### Model parameters and the expectation operator

The economic parameters (\$\alpha\$, \$\beta\$, \$\delta\$, and the AR(1) coefficients \$\rho_z\$, \$\sigma_z\$) are constants throughout, held in a `BrockMirmanParams`.

To evaluate the conditional expectation \$\mathbb{E}_{z_{t+1}|z_t}[\cdot]\$ we use **Gauss–Hermite quadrature** with 5 nodes. The Python notebook builds the standard nodes and weights with `np.polynomial.hermite.hermgauss` and rescales the nodes by the innovation standard deviation; the Julia preview builds the same rule with `gauss_hermite_rule(5)`. The `quadrature_checks` confirm the weights sum to one and reproduce the standard-normal mean (0) and variance (1).
"""

# ╔═╡ 44444444-0703-4444-8444-444444444444
begin
    params = BrockMirmanParams(alpha = 0.36, beta = 0.99, delta = 0.1,
        rho_z = 0.9, sigma_z = 0.04)
    rule = gauss_hermite_rule(5)
    quadrature_checks = (
        nodes = length(rule.weights),
        weight_sum = sum(rule.weights),
        normal_mean = quadrature_expectation(x -> x, rule),
        normal_variance = quadrature_expectation(x -> x^2, rule),
    )
end

# ╔═╡ bcae41c5-2cd5-6c0e-2923-4cdb1add60b1
md"""
#### Helper utilities

Before the model, the cell below defines three small helpers used later: `sample_states` draws exogenous \$(z, K)\$ states as a feature-by-batch matrix (the exogenous-sampling scheme), `logit` inverts the sigmoid, and `ConstantRawPolicy` is a fixed-output Lux-style policy used in the full-depreciation cross-check to inject the *known* closed-form savings rate.
"""

# ╔═╡ 55555555-0703-4555-8555-555555555555
begin
    function logit(p)
        return log(p / (1 - p))
    end

    function sample_states(rng, n; z_bounds = (0.7, 1.3), k_bounds = (0.9, 12.0))
        z = z_bounds[1] .+ (z_bounds[2] - z_bounds[1]) .* rand(rng, n)
        k = k_bounds[1] .+ (k_bounds[2] - k_bounds[1]) .* rand(rng, n)
        return vcat(reshape(z, 1, :), reshape(k, 1, :))
    end

    struct ConstantRawPolicy{T}
        raw::T
    end

    function (m::ConstantRawPolicy)(x, ps, st)
        return fill(m.raw, 1, size(x, 2)), st
    end
end

# ╔═╡ a940f0a1-c7b9-a99a-4925-73a15324f6f3
md"""
#### Deep neural network

The network approximates the savings rate \$s_t\$, so that \$K_{t+1} = Y_t\, s_t \approx Y_t\,\mathcal{N}(\mathbf{X}_t)\$. The input is now the **2-dimensional** state \$\mathbf{X}_t = [z_t, K_t]\$ and the output is the 1-dimensional savings rate \$s_t\$.

Following [Azinovic et al. (2022)](https://onlinelibrary.wiley.com/doi/full/10.1111/iere.12575) we use a densely connected feed-forward network with two hidden layers (ReLU) and a **sigmoid** output so that \$s_t \in (0, 1)\$ — economic prior knowledge encoded directly in the architecture. The Python notebook builds this in Keras; the Julia preview uses `make_mlp(2, (12, 12), 1)` (Lux) with a `NNlib.sigmoid` inside `savings_policy`.

**The batch dimension.** We evaluate on a whole matrix of states at once, not a single \$\mathbf{X}\$. At the Lux boundary these are **feature-by-batch** arrays \$[z; K]\$ (features on the first axis, samples on the second) — the transpose of the deep-learning convention of samples on the 0-axis — threaded through the explicit `model(x, ps, st)` call.
"""

# ╔═╡ 77a7526e-e470-8c65-627c-e9530452afef
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

**Why this matters here.** `savings_policy` parameterises the savings *share* \$s_t \in (0, 1)\$ via a **sigmoid** output. Combined with the resource constraint \$K_{t+1} = s_t Y_t\$ and \$C_t = (1 - s_t) Y_t\$, this **guarantees \$C_t > 0\$ and \$K_{t+1} > 0\$ simultaneously**, at every iteration. We never have to penalise infeasibility — the architecture rules it out. The Euler equation, by contrast, is enforced softly through the loss. This is one reason DEQNs converge in regions where naive penalty methods do not.
"""

# ╔═╡ 66666666-0703-4666-8666-666666666666
begin
    model = make_mlp(2, (12, 12), 1; activation = NNlib.relu)
    ps, st = setup_model(rng_from_seed(SEED; offset = 1), model; parameter_type = Float64)

    function savings_policy(model, ps, st, states; params = params)
        x = assert_feature_batch(states, 2)
        raw, st_new = model(x, ps, st)
        savings = NNlib.sigmoid.(raw)
        z = x[1:1, :]
        k = x[2:2, :]
        output = z .* k .^ params.alpha
        next_capital = (1 - params.delta) .* k .+ output .* savings
        consumption = output .* (1 .- savings)
        return (
            savings = savings,
            next_capital = next_capital,
            consumption = consumption,
            output = output,
        ), st_new
    end

    X_test = [1.0 0.5; 1.0 2.0]
    test_policy, _ = savings_policy(model, ps, st, X_test)
end

# ╔═╡ 637096d1-1215-2805-924a-59cf2596b2dc
md"""
#### Why we no longer need pen-and-paper FOC + envelope
With period payoff
\$\$\Pi(K_{\text{in}},\,K_{\text{out}},\,z_{\text{in}}) \;=\; u\!\big(\,Y(K_{\text{in}},z_{\text{in}}) + (1-\delta)K_{\text{in}} - K_{\text{out}}\,\big),\$\$
we apply the **same autodiff notation as in notebook 02**:

> **Notation — what do \$\partial_1\Pi\$ and \$\partial_2\Pi\$ mean here?**
> The subscript names the *slot being differentiated*, not a time index. `Pi` now has three slots, but we only ever differentiate slots 1 and 2:
> - \$\partial_1 \Pi \;=\; \dfrac{\partial \Pi}{\partial K_{\text{in}}}\$  — derivative w.r.t. the **1st argument** of `Pi`, i.e. the *capital state*.
> - \$\partial_2 \Pi \;=\; \dfrac{\partial \Pi}{\partial K_{\text{out}}}\$ — derivative w.r.t. the **2nd argument** of `Pi`, i.e. the *capital choice*.
> The shock \$z_{\text{in}}\$ sits in slot 3 and is treated as an exogenous parameter, not differentiated.
>
> So \$\partial_2\Pi(K_t,K_{t+1},z_t)\$ differentiates in slot 2, evaluated at \$(K_t,K_{t+1},z_t)\$, giving \$\partial\Pi/\partial K_{t+1}\$.
> \$\partial_1\Pi(K_{t+1},K_{t+2},z_{t+1})\$ differentiates in slot 1, evaluated at \$(K_{t+1},K_{t+2},z_{t+1})\$, giving \$\partial\Pi/\partial K_{t+1}\$ (different slot, *same* physical variable).

The hand vs. autodiff correspondence:

| | Hand derivation (notebook 02) | Autodiff |
|---|---|---|
| FOC w.r.t. the *choice* \$K_{t+1}\$ | \$-u'(C_t) + \beta\, \mathbb{E}[V'(z_{t+1},K_{t+1})] = 0\$ | \$\partial_2 \Pi(K_t, K_{t+1}, z_t) \equiv \dfrac{\partial \Pi}{\partial K_{\text{out}}}\Big|_{(K_t,K_{t+1},z_t)}\$ via `ForwardDiff` |
| Envelope: \$V'(K_t,z_t)\$ at the *state* \$K_t\$ | \$u'(C_t)\,(\alpha z_t K_t^{\alpha-1} + 1 - \delta)\$ | \$\partial_1 \Pi(K_t, K_{t+1}, z_t) \equiv \dfrac{\partial \Pi}{\partial K_{\text{in}}}\Big|_{(K_t,K_{t+1},z_t)}\$ via `ForwardDiff` |

Substituting the envelope at \$K_{t+1}\$ into the FOC delivers the **autodiff stochastic Euler residual**
\$\$\boxed{\;\underbrace{\partial_2 \Pi(K_t, K_{t+1}, z_t)}_{\text{derivative w.r.t. }K_{t+1}\text{ (the choice)}} \;+\; \beta\,\mathbb{E}_{z_{t+1}\,|\,z_t}\!\left[\underbrace{\partial_1 \Pi(K_{t+1}, K_{t+2}, z_{t+1})}_{\text{derivative w.r.t. }K_{t+1}\text{ (now the state of }t+1\text{)}}\right] \;=\; 0\;}\$\$
The expectation is approximated by Gauss-Hermite quadrature (5 nodes). Only the user-written piece of the loss changes: now it is just `Pi`.

> **Why this matters.** The pattern \$\,\partial_2 \Pi + \beta\,\mathbb{E}[\partial_1 \Pi] = 0\,\$ is the autodiff template for *any* recursive optimization problem with a single control. Change the utility, the production function, or the depreciation law inside `Pi`, and nothing else in the loss code needs to change. This is the practical reason Deep Equilibrium Nets scale so well to high-dimensional macro models.
"""

# ╔═╡ 5029b94e-770b-3871-2e53-102aa9257008
md"""
**Step A — write the period payoff `Pi(K_in, K_out, z_in)`.** This is the only place the model enters the loss; change \$u\$, \$Y\$, or \$\delta\$ here and nothing else moves. The cell below writes `Pi` once and reads off the two slot derivatives `partial_1_Pi` (\$\partial_1\Pi\$) and `partial_2_Pi` (\$\partial_2\Pi\$) with `ForwardDiff.derivative`. It also defines the closed-form `hand_partial_1` and `hand_partial_2` used for the slot-gradient cross-check that follows.
"""

# ╔═╡ 77777777-0703-4777-8777-777777777777
begin
    function Pi(K_in, K_out, z_in; params = params)
        output = z_in * K_in^params.alpha
        consumption = output + (1 - params.delta) * K_in - K_out
        return log(consumption)
    end

    partial_1_Pi(K_in, K_out, z_in; params = params) =
        ForwardDiff.derivative(k -> Pi(k, K_out, z_in; params), K_in)

    partial_2_Pi(K_in, K_out, z_in; params = params) =
        ForwardDiff.derivative(kout -> Pi(K_in, kout, z_in; params), K_out)

    hand_partial_1(K_in, K_out, z_in; params = params) = begin
        c = z_in * K_in^params.alpha + (1 - params.delta) * K_in - K_out
        (params.alpha * z_in * K_in^(params.alpha - 1) + 1 - params.delta) / c
    end

    hand_partial_2(K_in, K_out, z_in; params = params) = begin
        c = z_in * K_in^params.alpha + (1 - params.delta) * K_in - K_out
        -1 / c
    end
end

# ╔═╡ 0b0701d2-1a14-fe04-5c36-0752b5fe0767
md"""
#### Cross-check: slot derivatives vs closed forms

Before assembling the full residual, we verify the two slot derivatives directly. `slot_gradient_errors` evaluates \$\partial_1\Pi\$ and \$\partial_2\Pi\$ from `ForwardDiff` at a handful of \$(K_{\text{in}}, K_{\text{out}}, z)\$ points and compares them to the hand-coded closed forms \$\partial_1\Pi = (\alpha z K^{\alpha-1} + 1 - \delta)/C\$ and \$\partial_2\Pi = -1/C\$. Agreement is to machine precision.
"""

# ╔═╡ 88888888-0703-4888-8888-888888888888
begin
    slot_points = (
        (K_in = 1.0, K_out = 0.8, z = 1.0),
        (K_in = 3.0, K_out = 2.0, z = 0.9),
        (K_in = 8.0, K_out = 5.0, z = 1.2),
    )
    slot_gradient_errors = (
        partial_1 = maximum(abs(partial_1_Pi(p.K_in, p.K_out, p.z) -
            hand_partial_1(p.K_in, p.K_out, p.z)) for p in slot_points),
        partial_2 = maximum(abs(partial_2_Pi(p.K_in, p.K_out, p.z) -
            hand_partial_2(p.K_in, p.K_out, p.z)) for p in slot_points),
    )
end

# ╔═╡ 930599f8-10da-c516-0f07-7b9b4c59755d
md"""
#### Implementing the autodiff cost function — Step B

**Step B — assemble the Euler residual** from the two slot derivatives of `Pi`:
- \$\partial_2 \Pi(K_t, K_{t+1}, z_t)\$ at \$(K_t, K_{t+1}, z_t)\$ — the **FOC term** (choice slot).
- \$\partial_1 \Pi(K_{t+1}, K_{t+2}, z_{t+1})\$ inside the expectation at \$(K_{t+1}, K_{t+2}, z_{t+1})\$ — the **envelope term** (state slot).

`autodiff_stochastic_residual` rolls the policy forward, then loops over the Gauss-Hermite nodes to approximate \$\mathbb{E}_{z_{t+1}|z_t}[\partial_1\Pi]\$ (the same common-shock quadrature as notebook 02), and returns \$\partial_2\Pi + \beta\,\mathbb{E}[\partial_1\Pi]\$. `hand_stochastic_residual` builds the same quantity from the hand-derived formula for comparison.
"""

# ╔═╡ 99999999-0703-4999-8999-999999999999
begin
    function autodiff_stochastic_residual(model, ps, st, states, rule; params = params)
        x = assert_feature_batch(states, 2)
        z = x[1:1, :]
        k = x[2:2, :]

        current, st_new = savings_policy(model, ps, st, x; params)
        dPi_dKout = partial_2_Pi.(k, current.next_capital, z; params)

        expectation = zero(k)
        for (node, weight) in zip(rule.nodes, rule.weights)
            # The same quadrature node is broadcast across all batch columns,
            # matching the Python notebook's common-shock expectation loop.
            z_next = exp.(params.rho_z .* log.(z) .+ params.sigma_z .* node)
            x_next = vcat(z_next, current.next_capital)
            next_policy, _ = savings_policy(model, ps, st_new, x_next; params)
            dPi_dKin = partial_1_Pi.(current.next_capital, next_policy.next_capital, z_next; params)
            expectation = expectation .+ weight .* dPi_dKin
        end

        residual = dPi_dKout .+ params.beta .* expectation
        lhs = .-dPi_dKout
        rhs = params.beta .* expectation
        return (
            loss = mean(abs2, residual),
            residual = residual,
            savings = current.savings,
            consumption = current.consumption,
            next_capital = current.next_capital,
            lhs = lhs,
            rhs = rhs,
        ), st_new
    end

    function hand_stochastic_residual(model, ps, st, states, rule; params = params)
        x = assert_feature_batch(states, 2)
        z = x[1:1, :]
        k = x[2:2, :]

        current, st_new = savings_policy(model, ps, st, x; params)
        expectation = zero(k)
        for (node, weight) in zip(rule.nodes, rule.weights)
            z_next = exp.(params.rho_z .* log.(z) .+ params.sigma_z .* node)
            x_next = vcat(z_next, current.next_capital)
            next_policy, _ = savings_policy(model, ps, st_new, x_next; params)
            return_next = params.alpha .* z_next .* current.next_capital .^ (params.alpha - 1)
            inside = (1 ./ next_policy.consumption) .* (1 - params.delta .+ return_next)
            expectation = expectation .+ weight .* inside
        end

        residual = -1 ./ current.consumption .+ params.beta .* expectation
        return (
            loss = mean(abs2, residual),
            residual = residual,
            consumption = current.consumption,
            next_capital = current.next_capital,
        ), st_new
    end
end

# ╔═╡ ac88babf-33f4-f695-0cdd-2082c5183f8a
md"""
### Cross-check 1: autodiff vs hand-derived stochastic Euler residual

The hand-derived residual from notebook 02 is
\$\$r^{\text{hand}}(\mathbf{X}_t) \;=\; -\frac{1}{C_t} + \beta\,\mathbb{E}\!\left[\,\frac{1}{C_{t+1}}\,(1 - \delta + r_{t+1})\,\right].\$\$
The autodiff residual is mathematically identical. The cell below evaluates both on the **same (random) network** over a 2-D \$(z, K)\$ grid and reports the maximum absolute difference, plus a link to the shared library residual `stochastic_bm_residual` in relative form. We expect machine-precision agreement.
"""

# ╔═╡ 0e3e1d30-e4a1-a0de-8286-29515e05fc75
md"""
> **Convention note.** This notebook reports the autodiff Euler residual in **absolute** form `-1/C_t + β·E[(1/C_{t+1})(1 - δ + r_{t+1})]`, while `lecture_03_02_Brock_Mirman_Uncertainty_DEQN` and the script's eq. `eq:ree_bm` report the **relative** form `1 - 1/(C_t · β · E[...])`. The two are related by an overall factor of `C_t`; the absolute form isolates float-precision arithmetic for the autodiff-vs-hand cross-check, while the relative form is the natural diagnostic for the trained-policy comparison. The `residual_checks` cell reports both.
"""

# ╔═╡ aaaaaaaa-0703-4aaa-8aaa-aaaaaaaaaaaa
begin
    z_grid = repeat(collect(range(0.8, 1.2; length = 5)), 5)
    k_grid = repeat(collect(range(1.0, 11.0; length = 5)); inner = 5)
    X_grid = vcat(reshape(z_grid, 1, :), reshape(k_grid, 1, :))

    auto_grid, _ = autodiff_stochastic_residual(model, ps, st, X_grid, rule; params)
    hand_grid, _ = hand_stochastic_residual(model, ps, st, X_grid, rule; params)
    shared_relative, _ = stochastic_bm_residual(model, ps, st, X_grid, rule;
        params, transform = NNlib.sigmoid)

    residual_checks = (
        max_abs_autodiff_minus_hand = maximum(abs.(auto_grid.residual .- hand_grid.residual)),
        max_abs_relative_link = maximum(abs.(shared_relative.residual .- auto_grid.residual ./ auto_grid.rhs)),
        finite_autodiff_loss = isfinite(auto_grid.loss),
    )
end

# ╔═╡ 547d23f2-f7ad-7f36-46ba-bb0b00032d6d
md"""
### Simulating the model from the policy

Given a policy, we can simulate the model forward. From a state \$\mathbf{X}_t=[z_t, K_t]\$ we get \$K_{t+1} = Y_t \cdot \mathcal{N}(\mathbf{X}_t)\$, draw an innovation \$\epsilon_t\sim N(0, 1)\$, set \$z_{t+1}=\exp(\rho \log z_t + \sigma \epsilon_t)\$, and repeat. `simulate_single_step` advances one period and `simulate_periods` rolls several tracks forward in parallel; `simulation_summary` reports the start/end means of \$z\$ and \$K\$. (Here the Lux policy is untrained, so this illustrates the simulation *mechanics*.)
"""

# ╔═╡ 8661e829-065f-1902-3855-c9c0bf3c6378
md"""
> **The full Python notebook also covers** the *ergodic distribution* and *simulation-based training*, which this compact preview only describes:
>
> - **Ergodic cloud.** Simulated states settle on a cloud around the diagonal in \$(z, K)\$; the model never visits extreme corners (low \$z\$, high \$K\$ or vice versa), so effort spent training there is wasted. With more state variables this curse of dimensionality worsens (Maliar et al., 2011), and the location of the cloud depends on the very policy being solved for.
> - **Iterating training and simulation.** [Azinovic et al. (2022)](https://onlinelibrary.wiley.com/doi/full/10.1111/iere.12575) address this by alternating between simulating new states from the current policy and training on those states (many short tracks in parallel rather than one long one). Only the *sampling* changes; the autodiff loss is unchanged.
> - **A caution.** Simulation-based methods scale far better in high dimensions but add fragility: the training-data distribution shifts as the policy changes, so the learning rate must be tuned carefully, and early random policies can propose infeasible states. Azinovic & Žemlička (2023) introduce market-clearing architectures that stabilise this.
"""

# ╔═╡ bbbbbbbb-0703-4bbb-8bbb-bbbbbbbbbbbb
begin
    function simulate_single_step(model, ps, st, states, innovations; params = params)
        current, st_new = savings_policy(model, ps, st, states; params)
        z = states[1:1, :]
        z_next = exp.(params.rho_z .* log.(z) .+ params.sigma_z .* innovations)
        return vcat(z_next, current.next_capital), st_new
    end

    function simulate_periods(rng, model, ps, st, start_states, periods; params = params)
        n_tracks = size(start_states, 2)
        history = Array{Float64}(undef, 2, n_tracks, periods)
        current = start_states
        current_st = st
        for t in 1:periods
            history[:, :, t] .= current
            shocks = reshape(randn(rng, n_tracks), 1, :)
            current, current_st = simulate_single_step(model, ps, current_st, current, shocks; params)
        end
        return history, current
    end

    simulation_start = sample_states(rng, hp.simulation_tracks;
        z_bounds = (0.9, 1.1), k_bounds = (2.0, 6.0))
    simulation_path, simulation_end = simulate_periods(rng, model, ps, st,
        simulation_start, hp.simulation_periods; params)
    simulation_summary = (
        start_mean_z = mean(simulation_start[1, :]),
        end_mean_z = mean(simulation_end[1, :]),
        start_mean_k = mean(simulation_start[2, :]),
        end_mean_k = mean(simulation_end[2, :]),
    )
end

# ╔═╡ 19e80def-db1d-0530-6000-be340aa503d7
md"""
### Cross-check 2: side-experiment vs the analytical solution under full depreciation

When \$\delta = 1\$ and \$u(C) = \ln C\$, the stochastic Brock-Mirman model with log utility admits a **closed-form policy** for *any* productivity process:
\$\$K_{t+1} = \alpha\beta\, z_t\, K_t^{\alpha}, \qquad C_t = (1-\alpha\beta)\, z_t\, K_t^{\alpha}.\$\$
The Python notebook re-trains a fresh network under \$\delta=1\$ and compares. This compact preview takes the complementary route: it plugs the known closed-form savings rate \$s = \alpha\beta\$ into a `ConstantRawPolicy` and checks on a 2-D \$(z, K)\$ grid that (i) the implied policy equals \$\alpha\beta z_t K_t^{\alpha}\$ and (ii) the **autodiff residual vanishes** there. This verifies that minimizing the loss would recover the true policy — the residual is exactly zero at the analytical optimum.
"""

# ╔═╡ cccccccc-0703-4ccc-8ccc-cccccccccccc
begin
    full_params = BrockMirmanParams(alpha = params.alpha, beta = params.beta,
        delta = 1.0, rho_z = params.rho_z, sigma_z = params.sigma_z)
    exact_savings = full_params.alpha * full_params.beta
    exact_model = ConstantRawPolicy(logit(exact_savings))
    exact_ps = nothing
    exact_st = NamedTuple()

    z_eval = repeat(collect(range(0.7, 1.3; length = 7)), 7)
    k_eval = repeat(collect(range(0.1, 1.0; length = 7)); inner = 7)
    X_exact = vcat(reshape(z_eval, 1, :), reshape(k_eval, 1, :))
    exact_auto, _ = autodiff_stochastic_residual(exact_model, exact_ps, exact_st,
        X_exact, rule; params = full_params)

    exact_policy = full_params.alpha .* full_params.beta .* X_exact[1:1, :] .*
        X_exact[2:2, :] .^ full_params.alpha
    exact_policy_checks = (
        max_abs_policy_error = maximum(abs.(exact_auto.next_capital .- exact_policy)),
        max_abs_autodiff_residual = maximum(abs.(exact_auto.residual)),
        mean_abs_autodiff_residual = mean(abs.(exact_auto.residual)),
    )
end

# ╔═╡ b376040b-e1a0-dca8-c100-68360fbae31a
md"""
### Policy and residual on a capital slice

The figure below fixes \$z_t = 1\$ and plots the (untrained, random) Lux policy \$K_{t+1}\$ against the 45-degree line, alongside the absolute autodiff residual across the capital slice.
"""

# ╔═╡ dddddddd-0703-4ddd-8ddd-dddddddddddd
begin
    z_line = fill(1.0, 1, 80)
    k_line = reshape(collect(range(0.9, 12.0; length = 80)), 1, :)
    line_states = vcat(z_line, k_line)
    line_auto, _ = autodiff_stochastic_residual(model, ps, st, line_states, rule; params)

    fig = Figure(size = figure_size(RUN_MODE))
    ax1 = Axis(fig[1, 1], xlabel = "K_t at z_t = 1", ylabel = "K_{t+1}")
    lines!(ax1, vec(k_line), vec(k_line); color = :gray55, linestyle = :dash, label = "45 degree")
    lines!(ax1, vec(k_line), vec(line_auto.next_capital); color = :dodgerblue3, linewidth = 3,
        label = "random Lux policy")
    axislegend(ax1; position = :lt)

    ax2 = Axis(fig[1, 2], xlabel = "K_t at z_t = 1", ylabel = "absolute residual")
    lines!(ax2, vec(k_line), vec(abs.(line_auto.residual)); color = :darkorange, linewidth = 3)
    fig
end

# ╔═╡ b3aa6bed-87f5-9079-64b1-2b009628cc1e
md"""
### Takeaway

- We set up the **same stochastic Brock-Mirman model as notebook 02** without ever writing the FOC or invoking the envelope theorem on paper. The user only writes the three-argument primitive `Pi(K_in, K_out, z_in)`; both derivatives come out of `ForwardDiff` (two `tf.GradientTape` calls in Python).
- **Cross-check 1** confirmed that the autodiff stochastic Euler residual is numerically identical to the hand-derived residual at machine precision.
- **Cross-check 2** confirmed that, when an analytical solution is available (full depreciation), the autodiff residual vanishes exactly at the closed-form policy \$K_{t+1} = \alpha\beta z_t K_t^{\alpha}\$.
- The pattern \$\,\partial_2 \Pi + \beta\,\mathbb{E}[\partial_1 \Pi] = 0\,\$ (differentiate in the *choice* slot, plus \$\beta\times\$ expected derivative in the *state* slot) generalizes verbatim to richer models: arbitrary utility, multi-good production, occasionally binding constraints (with a Fischer-Burmeister wrap), and high-dimensional state spaces.

The cell below returns this notebook's machine-checkable diagnostics NamedTuple; as its `note` field records, parameter training through these `ForwardDiff` slot closures is intentionally not attempted in this compact preview.
"""

# ╔═╡ eeeeeeee-0703-4eee-8eee-eeeeeeeeeeee
(
    run_mode = RUN_MODE,
    seed = SEED,
    quadrature = quadrature_checks,
    policy_test_savings = test_policy.savings,
    slot_gradient_errors = slot_gradient_errors,
    residual_checks = residual_checks,
    simulation = simulation_summary,
    full_depreciation_exact_policy = exact_policy_checks,
    note = "The residual is faithful to the Python nested-tape formula. Parameter training through these ForwardDiff slot closures is not claimed in this compact preview.",
)

# ╔═╡ Cell order:
# ╟─11111111-0703-4111-8111-111111111111
# ╟─a330efd1-9fd1-047e-5ca6-e9183f8b519f
# ╟─3b4bf433-c545-de6e-9cf6-180f5f0f0cd6
# ╠═22222222-0703-4222-8222-222222222222
# ╠═33333333-0703-4333-8333-333333333333
# ╟─9fe050cb-4bb6-d21e-e24c-f336b7957b12
# ╠═44444444-0703-4444-8444-444444444444
# ╟─bcae41c5-2cd5-6c0e-2923-4cdb1add60b1
# ╠═55555555-0703-4555-8555-555555555555
# ╟─a940f0a1-c7b9-a99a-4925-73a15324f6f3
# ╟─77a7526e-e470-8c65-627c-e9530452afef
# ╠═66666666-0703-4666-8666-666666666666
# ╟─637096d1-1215-2805-924a-59cf2596b2dc
# ╟─5029b94e-770b-3871-2e53-102aa9257008
# ╠═77777777-0703-4777-8777-777777777777
# ╟─0b0701d2-1a14-fe04-5c36-0752b5fe0767
# ╠═88888888-0703-4888-8888-888888888888
# ╟─930599f8-10da-c516-0f07-7b9b4c59755d
# ╠═99999999-0703-4999-8999-999999999999
# ╟─ac88babf-33f4-f695-0cdd-2082c5183f8a
# ╟─0e3e1d30-e4a1-a0de-8286-29515e05fc75
# ╠═aaaaaaaa-0703-4aaa-8aaa-aaaaaaaaaaaa
# ╟─547d23f2-f7ad-7f36-46ba-bb0b00032d6d
# ╟─8661e829-065f-1902-3855-c9c0bf3c6378
# ╠═bbbbbbbb-0703-4bbb-8bbb-bbbbbbbbbbbb
# ╟─19e80def-db1d-0530-6000-be340aa503d7
# ╠═cccccccc-0703-4ccc-8ccc-cccccccccccc
# ╟─b376040b-e1a0-dca8-c100-68360fbae31a
# ╠═dddddddd-0703-4ddd-8ddd-dddddddddddd
# ╟─b3aa6bed-87f5-9079-64b1-2b009628cc1e
# ╠═eeeeeeee-0703-4eee-8eee-eeeeeeeeeeee
