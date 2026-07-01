### A Pluto.jl notebook ###
# v0.20.0

using Markdown
using InteractiveUtils

# ╔═╡ 11111111-0911-4111-8111-111111111111
md"""
# Lecture 09, Notebook 11: Continuum-Agent DEQN in Lux

A smoke-size Bewley-style economy with two idiosyncratic income states. The Lux
network has policy and price heads; the loss uses Euler and bond-market residuals,
and Young's histogram is advanced inside the training loop after each gradient
step.
"""

# ╔═╡ cfe82a8d-4b6b-407d-ac23-a5891b93d209
md"""
## Lecture 09, Notebook 11: Continuum-of-agents DEQN (Bewley endowment economy)

**Course:** Deep Learning for Solving and Estimating Dynamic Models in Economics and Finance
**Script reference:** §6.4 (DEQN with a continuum of agents)
**Notebook role:** core
**Author:** Simon Scheidegger

*Julia/Lux/Pluto preview of* `lectures/lecture_09_heterogeneous_agents_youngs_method/code/lecture_09_11_Continuum_of_Agents_DEQN.ipynb`.
"""

# ╔═╡ 7a754a46-04d8-2598-f130-f4f9d1d2562b
md"""
## Solving the Continuum-of-Agents Model with Deep Equilibrium Nets

**Bewley Endowment Economy with Heterogeneous Agents**

### Overview

We solve a **Bewley endowment economy** with a **continuum of agents**, aggregate and idiosyncratic shocks, **Epstein–Zin preferences**, and a **borrowing constraint**. The distribution of agents is tracked using **Young's (2010) non-stochastic histogram method**, which enters the neural network as a high-dimensional input.

#### How this notebook fits the lecture
The lecture first uses the **Krusell-Smith benchmark** to explain why distribution tracking matters. The Python ground truth is the **Appendix A.5 teaching implementation** from Azinovic, Gaegauf, and Scheidegger (2022). The common ingredient is Young's histogram update: the benchmark motivates **why** it matters, while the DEQN shows **how** it enters training.

In the paper's formulation two neural networks are trained jointly:
- A **policy network** \$\mathcal{N}_{pol}\$: maps individual + aggregate state \$\to\$ savings \$b'\$, KKT multiplier \$\lambda\$, value \$V\$
- A **price network** \$\mathcal{N}_{price}\$: maps aggregate state \$\to\$ bond price \$p\$

> **About this Julia preview.** This is a compact, smoke-size **Bewley simplification** of the Appendix A.5 model, built to make the Young-histogram-inside-DEQN loop readable end to end. It differs from the Python ground truth in a few deliberate ways: it uses **CRRA utility** (\$\gamma = 2\$) in place of Epstein–Zin recursive utility; **two idiosyncratic income states** with a Markov transition and no separate aggregate-shock states; and a **single Lux MLP** with a savings head and a price head in place of two separate networks. The two equilibrium residuals that survive — the **Euler equation** and **bond-market clearing** — are exactly the objects the histogram feeds, so the pedagogy of "distribution propagation vs. policy training" is preserved.
"""

# ╔═╡ 22222222-0911-4222-8222-222222222222
begin
    import Pkg
    Pkg.activate(joinpath(@__DIR__, "..", "..", "..", "julia"))

    using DLEFJulia
    using Lux
    using NNlib
    using Optimisers
end

# ╔═╡ 3c5e1f71-0a47-45c2-0f7c-2b26a0325136
md"""
### Run mode and training budget

The kit-mandated `RUN_MODE` / `SEED` constants and a `budgets` NamedTuple set the size of the run. Where the Python notebook dispatches \$N_b\$, hidden width, and episode counts on `RUN_MODE`, the Julia preview exposes two knobs — gradient `steps` and histogram `grid_size`:

| `RUN_MODE`   | steps | grid_size |
|--------------|-------|-----------|
| `smoke`      | 8     | 32        |
| `teaching`   | 200   | 120       |
| `production` | 2000  | 400       |

The checked-in default is `smoke`, so the full pipeline is inspectable in class. `run_mode_budget` selects the row; `rng_from_seed(SEED)` seeds the run.
"""

# ╔═╡ 33333333-0911-4333-8333-333333333333
begin
    RUN_MODE = "smoke"
    SEED = 0
    budgets = (
        smoke = (steps = 8, grid_size = 32),
        teaching = (steps = 200, grid_size = 120),
        production = (steps = 2_000, grid_size = 400),
    )
    hp = run_mode_budget(RUN_MODE; budgets)
    rng = rng_from_seed(SEED)
end

# ╔═╡ 82aa170b-b37c-979c-f32d-46539c824289
md"""
### 1. Model description, parameters, and network

#### Preferences (paper)
Each agent in the Appendix A.5 model has **Epstein–Zin recursive utility** (separating risk aversion from IES):
\$\$V(x_t) = \max_{b_{t+1} \geq 0} \left[(1-\beta)\, c_t^{1-\rho} + \beta\, \chi_t\!\left(V(x_{t+1})\right)^{1-\rho}\right]^{\frac{1}{1-\rho}}\$\$
where \$\chi_t(V_{t+1}) = \mathbb{E}_t\!\left[V_{t+1}^{1-\sigma}\right]^{\frac{1}{1-\sigma}}\$ is the certainty equivalent, with \$\beta = 0.95\$, \$\rho = 2\$ (IES \$= 0.5\$), \$\sigma = 8\$.

#### Budget constraint
\$\$c_t + p_t \cdot b_{t+1} = b_t + \eta_t \cdot w(a_t), \qquad b_{t+1} \geq 0\$\$
with bond holdings \$b_t\$, endogenous bond price \$p_t\$, idiosyncratic labour endowment \$\eta_t\$, and aggregate wage \$w(a_t)\$.

#### Equilibrium conditions (paper)
1. **Euler equation** (intertemporal optimality)
2. **Bellman equation** (value-function consistency)
3. **Market clearing** \$\int b_{t+1}\, d\mu_t = 1\$ (bonds in unit net supply)
4. **KKT complementarity** \$\lambda_t \cdot b_{t+1} = 0\$
5. **Budget feasibility** \$c_t > 0\$

#### What this cell actually builds (Julia preview)
The compact preview replaces Epstein–Zin with **CRRA** (\$\gamma = 2\$), sets \$\beta = 0.96\$, and uses **two idiosyncratic income states** \$\eta \in \{0.65, 1.35\}\$ with Markov transition \$\Pi_\eta\$ (validated by `validate_transition_matrix`). Bonds are in **unit net supply**. The asset grid has `grid_size` points on \$[0, 12]\$, and the initial histogram places mass on both income blocks via `redistribute_mass`. A single `make_mlp(3, (16, 16), 2)` with `tanh` activation carries **two output heads** — a savings head and a price head — trained with `Optimisers.Adam`. Consumption positivity is enforced by construction with a `consumption_floor`.
"""

# ╔═╡ 44444444-0911-4444-8444-444444444444
begin
    asset_grid = collect(range(0.0, 12.0; length = hp.grid_size))
    idio_income = [0.65, 1.35]
    idio_transition = [0.90 0.10; 0.08 0.92]
    validate_transition_matrix(idio_transition)

    beta = 0.96
    gamma = 2.0
    consumption_floor = 1e-5
    bond_supply = 1.0

    hist0 = vcat(
        permutedims(redistribute_mass(asset_grid, 0.6, 0.5)),
        permutedims(redistribute_mass(asset_grid, 1.4, 0.5)),
    )
    model = make_mlp(3, (16, 16), 2; activation = NNlib.tanh)
    state = setup_training(rng_from_seed(SEED; offset = 1), model, Optimisers.Adam(0.006); parameter_type = Float64)
end

# ╔═╡ e4103735-99b0-5642-19ff-5de7859edce5
md"""
### Model equations, cost function, and market clearing

This cell defines the map from network outputs to an economic policy and the equilibrium residual.

#### Features and policy
`continuum_features` stacks, per \$(\eta, b)\$ cell, the asset level, the income level, and the current histogram mean — the finite-dimensional encoding of the cross-sectional state. `continuum_policy` evaluates the MLP, splits the raw output into savings and price heads with `split_output_heads`, maps the price head to a bounded bond price \$p = 0.90 + 0.15\,\sigma(\cdot)\$, and reads savings from a sigmoid scaled to the feasible band \$[0,\,(\text{cash}-\underline{c})/p]\$. Consumption follows from the budget constraint \$c = b + \eta - p\,b'\$, floored at `consumption_floor`.

#### Euler residual and bond-market clearing
`continuum_residual` forms marginal utility \$c^{-\gamma}\$, takes the idiosyncratic-shock expectation \$\Pi_\eta\, c^{-\gamma}\$, and builds the **Euler residual**
\$\$\text{EE} = \frac{c^{-\gamma}}{\tfrac{\beta}{p}\,\mathbb{E}_\eta\!\left[c'^{-\gamma}\right]} - 1,\$\$
alongside **bond-market clearing** \$\sum h \cdot b' - 1\$ (unit supply). The loss is the histogram-weighted mean squared Euler residual plus the squared normalised market-clearing gap. Of the paper's five conditions, the preview enforces these two (Euler + market clearing); Bellman/KKT are dropped and budget feasibility holds by construction.
"""

# ╔═╡ 55555555-0911-4555-8555-555555555555
begin
    function continuum_features(grid, hist, income)
        n_idio, n_grid = size(hist)
        assets = repeat(reshape(grid, 1, n_grid), n_idio, 1)
        income_grid = repeat(reshape(income, n_idio, 1), 1, n_grid)
        mean_grid = fill(young_mean(grid, hist), n_idio, n_grid)
        return vcat(reshape(assets, 1, :), reshape(income_grid, 1, :), reshape(mean_grid, 1, :))
    end

    function continuum_policy(model, ps, st, hist)
        n_idio, n_grid = size(hist)
        raw, st_new = model(continuum_features(asset_grid, hist, idio_income), ps, st)
        heads = split_output_heads(raw, (saving = 1, price = 1))
        raw_saving = reshape(heads.saving, n_idio, n_grid)
        raw_price = reshape(heads.price, n_idio, n_grid)
        weights = hist ./ young_mass(hist)
        price = 0.90 + 0.15 * NNlib.sigmoid(sum(raw_price .* weights))
        cash = repeat(reshape(idio_income, n_idio, 1), 1, n_grid) .+ repeat(reshape(asset_grid, 1, n_grid), n_idio, 1)
        upper = min.(last(asset_grid), max.(first(asset_grid), (cash .- consumption_floor) ./ price))
        savings = first(asset_grid) .+ NNlib.sigmoid.(raw_saving) .* (upper .- first(asset_grid))
        consumption = max.(cash .- price .* savings, consumption_floor)
        return (price = price, savings = savings, consumption = consumption, cash = cash), st_new
    end

    function continuum_residual(model, ps, st, hist)
        policy, st_new = continuum_policy(model, ps, st, hist)
        marginal_utility = policy.consumption .^ (-gamma)
        expected_marginal_utility = map(CartesianIndices(policy.savings)) do idx
            i_cur = idx[1]
            bp = policy.savings[idx]
            w = young_weights(asset_grid, bp; clip = true)
            acc = zero(eltype(policy.consumption))
            for i_next in eachindex(idio_income)
                c_next = w.lower_weight * policy.consumption[i_next, w.lower] +
                         w.upper_weight * policy.consumption[i_next, w.upper]
                acc += idio_transition[i_cur, i_next] * c_next ^ (-gamma)
            end
            acc
        end
        euler = marginal_utility ./ max.(beta / policy.price .* expected_marginal_utility, consumption_floor) .- 1
        bond_market = sum(hist .* policy.savings) - bond_supply
        weighted_euler = sum(hist .* euler .^ 2) / young_mass(hist)
        loss = weighted_euler + abs2(bond_market / (1 + bond_supply))
        return (
            loss = loss,
            euler = euler,
            bond_market = bond_market,
            price = policy.price,
            savings = policy.savings,
            consumption = policy.consumption,
        ), st_new
    end

    continuum_loss(model, ps, st, batch) = begin
        pieces, st_new = continuum_residual(model, ps, st, batch.hist)
        return pieces.loss, st_new
    end
end

# ╔═╡ b0ec0176-3f7a-7b6b-bdfa-92c2cf3c5a2e
md"""
### Training loop with Young's histogram co-evolution

This is the DEQN training loop, and the place where **distribution propagation** and **policy training** meet — the distinction this lecture emphasises:

1. Take one `train_step!` on the residual loss at the current histogram (this updates the **policy**).
2. Evaluate the updated policy's savings and advance the histogram one **Young step** with `young_step(asset_grid, hist, savings; transition = idio_transition)` (this propagates the **distribution**).
3. Carry the new histogram forward as the starting point for the next step, recording mass and mean via `append_metric!`.

Young's update is deterministic and differentiable, so the histogram co-evolves with the network exactly as in the paper's episode-based training — here compressed to `hp.steps` gradient steps. Mass and mean are logged each step so histogram conservation is visible in the diagnostics.
"""

# ╔═╡ 66666666-0911-4666-8666-666666666666
begin
    train_result = let hist_local = hist0
        initial_loss_local = loss_value(state, continuum_loss, (hist = hist_local,))
        history_log_local = NamedTuple[]
        for step in 1:hp.steps
            metrics = train_step!(state, continuum_loss, (hist = hist_local,); max_grad_norm = 10.0)
            policy, _ = continuum_policy(state.model, state.ps, state.st, hist_local)
            hist_local = young_step(asset_grid, hist_local, policy.savings; transition = idio_transition)
            append_metric!(history_log_local; step, loss = metrics.loss, mass = young_mass(hist_local), mean_asset = young_mean(asset_grid, hist_local))
        end
        (initial_loss = initial_loss_local, history_log = history_log_local, hist = hist_local)
    end
    initial_loss = train_result.initial_loss
    history_log = train_result.history_log
    final_hist = train_result.hist
end

# ╔═╡ 764e38ed-ce3b-5baa-25b3-bf21187d41b9
md"""
### Diagnostics: policy, prices, and accuracy

With training complete, we re-evaluate the residual and the policy on the **final** histogram. `continuum_residual` returns the Euler residual, the bond-market gap, and the equilibrium bond price; `continuum_policy` returns the savings, consumption, and price maps across the \$(\eta, b)\$ grid. In the Python ground truth these back the loss-convergence, policy-function, bond-price / wealth-distribution, and Euler-accuracy plots (sections 11–14); here they feed the machine-checkable summary in the final cell.
"""

# ╔═╡ 77777777-0911-4777-8777-777777777777
begin
    diagnostics, _ = continuum_residual(state.model, state.ps, state.st, final_hist)
    final_policy, _ = continuum_policy(state.model, state.ps, state.st, final_hist)
end

# ╔═╡ affe062f-dd22-b3cd-32e6-798d7f8caebf
md"""
### Summary

We solved a compact **Bewley endowment economy** with a **continuum of agents** using a Deep Equilibrium Net:

1. **Young's histogram method** provides a finite-dimensional, differentiable encoding of the wealth distribution.
2. A **single Lux MLP** (savings + price heads) is trained by minimising equilibrium residuals — the Euler equation and bond-market clearing.
3. The **histogram co-evolves** with the network during training via `young_step`, so the network trains on equilibrium-consistent distributions.
4. **No separate forecasting rule** is needed — the price head conditions directly on the histogram mean.

#### Key architectural choices
- Input dimension scales with the histogram encoding — resolution (`grid_size`) is a tunable knob.
- Bounded sigmoid maps ensure positive prices and feasible, non-negative savings.
- Step-by-step histogram carry-over keeps training distributions on the model's own ergodic set.

> **Note.** The checked-in run is smoke-size (`grid_size = 32`, 8 gradient steps, CRRA, two income states). Raise `RUN_MODE` to `teaching`/`production` for larger histograms and longer training; the paper's Appendix A.5 configuration (Epstein–Zin, 6 aggregate states, two full networks, \$N_b = 100\$, 65,000 episodes) lives in the Python ground truth.

#### References
- Azinovic, M., Gaegauf, L., & Scheidegger, S. (2022). Deep equilibrium nets. *International Economic Review*, 63(4), 1471–1525.
- Young, E.R. (2010). Solving the incomplete markets model with aggregate uncertainty using the Krusell–Smith algorithm and non-stochastic simulations. *J. Econ. Dynamics & Control*, 34(1), 36–41.
- Krusell, P., & Smith, A.A. (1998). Income and wealth heterogeneity in the macroeconomy. *J. Political Economy*, 106(5), 867–896.

The cell below returns the machine-checkable diagnostics for this notebook's run.
"""

# ╔═╡ 88888888-0911-4888-8888-888888888888
(
    initial_loss = initial_loss,
    final_loss = diagnostics.loss,
    euler_rmse = residual_summary(diagnostics.euler).rmse,
    bond_market = diagnostics.bond_market,
    distribution_mass = young_mass(final_hist),
    mean_asset = young_mean(asset_grid, final_hist),
    price = diagnostics.price,
    policy_minmax = extrema(final_policy.savings),
)

# ╔═╡ Cell order:
# ╟─11111111-0911-4111-8111-111111111111
# ╟─cfe82a8d-4b6b-407d-ac23-a5891b93d209
# ╟─7a754a46-04d8-2598-f130-f4f9d1d2562b
# ╠═22222222-0911-4222-8222-222222222222
# ╟─3c5e1f71-0a47-45c2-0f7c-2b26a0325136
# ╠═33333333-0911-4333-8333-333333333333
# ╟─82aa170b-b37c-979c-f32d-46539c824289
# ╠═44444444-0911-4444-8444-444444444444
# ╟─e4103735-99b0-5642-19ff-5de7859edce5
# ╠═55555555-0911-4555-8555-555555555555
# ╟─b0ec0176-3f7a-7b6b-bdfa-92c2cf3c5a2e
# ╠═66666666-0911-4666-8666-666666666666
# ╟─764e38ed-ce3b-5baa-25b3-bf21187d41b9
# ╠═77777777-0911-4777-8777-777777777777
# ╟─affe062f-dd22-b3cd-32e6-798d7f8caebf
# ╠═88888888-0911-4888-8888-888888888888
