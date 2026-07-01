### A Pluto.jl notebook ###
# v0.20.0

using Markdown
using InteractiveUtils

# ╔═╡ 11111111-0811-4111-8111-111111111111
md"""
# Lecture 08, Notebook 11: OLG Exercise - Closed-Form Savings Rates and Lifecycle Profiles

This compact Pluto translation preserves the solved-exercise structure from the
Python notebook. The TODO cells are intentional student blanks; the following
solution cells compute the same analytical savings rates, lifecycle profiles,
and patience comparison.
"""

# ╔═╡ ed0973b4-e5e9-ca37-767f-003e3b7e39a6
md"""
## Lecture 08, Notebook 11: OLG Exercise — Closed-Form Savings Rates and Lifecycle Profiles

**Course:** Deep Learning for Solving and Estimating Dynamic Models in Economics and Finance
**Script reference:** §5.2 (the 6-agent analytic Krueger–Kübler OLG); a self-contained warm-up complementing Chapter 5 Exercises 5.1–5.2
**Notebook role:** exercise
**Author:** Simon Scheidegger

*Julia/Lux/Pluto preview of* `lectures/lecture_08_olg_models_deqns/code/lecture_08_11_OLG_Exercise.ipynb`.

> **Run mode.** The exercise is analytic and deterministic; `RUN_MODE = "smoke"` and `SEED = 0` are kept for consistency with the other notebooks, and `RUN_MODE` only affects figure sizing here (the calibration `n_ages = 6`, `beta = 0.7`, `r = 0.15`, `wage = 1.0` is the same across modes).
"""

# ╔═╡ 9100fb6c-aca2-ccee-56dc-343bd3759bed
md"""
# Exercise: OLG Savings Rates and Lifecycle Profiles

In this exercise you work with the **analytical OLG model** of Notebook 08 (the persistent-simulation primary).

**Goals (approx. 30 min):**
1. Compute closed-form savings rates \$\beta_h\$ for each cohort
2. Simulate a lifecycle consumption profile
3. Study how the discount factor \$\beta\$ shapes lifecycle behavior

**Prerequisites:** Notebook `lecture_08_08_OLG_Analytic_DEQN_persistent.ipynb` (the analytical 6-agent OLG model).

Each task cell below holds a TODO scaffold (returning a small placeholder NamedTuple) followed by a worked solution cell; the DLEFJulia helpers `exercise_savings_rates` and `simulate_lifecycle` in the next code cell implement the closed-form logic once so every task reuses it.
"""

# ╔═╡ 22222222-0811-4222-8222-222222222222
begin
    import Pkg
    Pkg.activate(joinpath(@__DIR__, "..", "..", "..", "julia"))

    using CairoMakie
    using DLEFJulia
    using Lux
    using Statistics
end

# ╔═╡ 33333333-0811-4333-8333-333333333333
begin
    RUN_MODE = "smoke"
    SEED = 0
    budgets = (
        smoke = (n_ages = 6, beta = 0.70, r = 0.15, wage = 1.0),
        teaching = (n_ages = 6, beta = 0.70, r = 0.15, wage = 1.0),
        production = (n_ages = 6, beta = 0.70, r = 0.15, wage = 1.0),
    )
    hp = run_mode_budget(RUN_MODE; budgets)
    rng = rng_from_seed(SEED)
end

# ╔═╡ 44444444-0811-4444-8444-444444444444
begin
    function exercise_savings_rates(A::Integer, beta)
        params = AnalyticOLGParams(n_ages = A, beta = beta, gamma = 1.0)
        rates = analytic_olg_closed_form_savings_rates(params)
        return vcat(rates, zero(eltype(rates)))
    end

    function simulate_lifecycle(savings_rates; r = 0.15, wage = 1.0)
        A = length(savings_rates)
        income = Vector{Float64}(undef, A)
        consumption = Vector{Float64}(undef, A)
        capital = Vector{Float64}(undef, A - 1)
        income[1] = wage
        for h in 1:A
            consumption[h] = (1 - savings_rates[h]) * income[h]
            if h < A
                capital[h] = savings_rates[h] * income[h]
                income[h + 1] = r * capital[h]
            end
        end
        return (income = income, capital = capital, consumption = consumption)
    end

    function rates_to_raw_logits(rates, params::AnalyticOLGParams)
        scaled = (rates .- params.min_saving_fraction) ./
            (params.max_saving_fraction - params.min_saving_fraction)
        clipped = clamp.(scaled, eps(Float64), 1 - eps(Float64))
        return log.(clipped ./ (1 .- clipped))
    end

    struct ConstantSavingsRaw{V}
        values::V
    end

    function (m::ConstantSavingsRaw)(features, ps, st)
        return repeat(reshape(m.values, :, 1), 1, size(features, 2)), st
    end
end

# ╔═╡ 55555555-0811-4555-8555-555555555555
md"""
## Background

Consider an **\$A\$-agent OLG economy** with log utility \$u(c) = \ln(c)\$ and discount factor \$\beta \in (0,1)\$. Each agent \$h \in \{1, \dots, A\}\$ maximizes remaining lifetime utility. The **optimal savings rate** for agent \$h\$ (the fraction of income saved) is given in closed form by

\$\$\beta_h = \frac{\beta(1 - \beta^{A-h})}{1 - \beta^{A-h+1}}.\$\$

- Agent \$h = A\$ (the oldest) has \$\beta_A = 0\$: they consume everything and die.
- The savings rate determines capital passed to the next period: \$k'_h = \beta_h \cdot \mathrm{income}_h\$.
- Consumption is the residual: \$c_h = (1 - \beta_h) \cdot \mathrm{income}_h\$.

In Julia, `exercise_savings_rates(A, beta)` wraps `analytic_olg_closed_form_savings_rates` (with \$\gamma = 1\$, i.e. log utility) and appends the oldest cohort's zero rate; `simulate_lifecycle` carries saved capital into next-period income.
"""

# ╔═╡ 66666666-0811-4666-8666-666666666666
md"""
## Task 1: Compute Analytical Savings Rates (10 min)

Use the formula above to compute \$\beta_h\$ for \$h = 1, \dots, 5\$ (agent 6 dies and has \$\beta_6 = 0\$). Set \$A = 6\$ and \$\beta = 0.7\$, then create a **bar chart** of savings rates by age.

The next code cell is the TODO scaffold; the cell after it is the worked solution (`savings_rates = exercise_savings_rates(A_task1, beta_task1)`), followed by the CairoMakie bar chart.

> **Preview note.** The full Python notebook embeds the calibration annotation \$(A=6, \beta=0.7)\$ directly in the bar-chart title; this preview keeps the shorter title *"Analytical Savings Rates by Age"* since \$A\$ and \$\beta\$ are fixed and already stated above.
"""

# ╔═╡ 77777777-0811-4777-8777-777777777777
begin
    # TODO: Compute savings rates for each cohort h = 1, ..., A - 1.
    # TODO: Append 0.0 for agent A.
    # TODO: Plot savings rate against age.
    task1_blank = (
        A = hp.n_ages,
        beta = hp.beta,
        todo = "Fill savings_rates, append the oldest cohort's zero rate, and plot a bar chart.",
    )
end

# ╔═╡ 88888888-0811-4888-8888-888888888888
begin
    A_task1 = hp.n_ages
    beta_task1 = hp.beta
    savings_rates = exercise_savings_rates(A_task1, beta_task1)
    ages = collect(1:A_task1)

    task1_table = [
        (age = h, savings_rate = savings_rates[h])
        for h in ages
    ]
end

# ╔═╡ 99999999-0811-4999-8999-999999999999
begin
    fig_rates = Figure(size = figure_size(RUN_MODE))
    ax_rates = Axis(fig_rates[1, 1],
        xlabel = "Age h",
        ylabel = "Savings rate",
        title = "Analytical Savings Rates by Age")
    barplot!(ax_rates, ages, savings_rates; color = :steelblue, strokecolor = :black, strokewidth = 1)
    ylims!(ax_rates, 0, 0.7)
    fig_rates
end

# ╔═╡ aaaaaaaa-0811-4aaa-8aaa-aaaaaaaaaaaa
md"""
### Solution Check

Young agents save the most because they have the longest remaining horizon. The
oldest agent saves nothing because they die next period.
"""

# ╔═╡ bbbbbbbb-0811-4bbb-8bbb-bbbbbbbbbbbb
md"""
## Task 2: Lifecycle Consumption Profile (10 min)

Given the savings rates from Task 1, simulate the **lifecycle of a single cohort** as it ages from \$h = 1\$ to \$h = A\$. Assume steady-state prices — wage \$w = 1.0\$ (income of the youngest agent) and interest rate \$r = 0.15\$:

- Agent \$h = 1\$ earns wage \$w\$, saves \$k_2 = \beta_1 \cdot w\$, consumes \$c_1 = (1 - \beta_1) \cdot w\$.
- Agent \$h = 2\$ earns \$r \cdot k_2\$, saves \$k_3 = \beta_2 \cdot (r \cdot k_2)\$, consumes \$c_2 = (1 - \beta_2) \cdot (r \cdot k_2)\$.
- ... and so on until agent \$A\$ consumes everything.

The TODO scaffold is next; the worked solution calls `simulate_lifecycle(savings_rates; r = hp.r, wage = hp.wage)` and plots consumption and capital by age.
"""

# ╔═╡ cccccccc-0811-4ccc-8ccc-cccccccccccc
begin
    # TODO: Simulate income, capital, and consumption over the lifecycle.
    # TODO: Plot consumption by age and capital holdings for ages 2, ..., A.
    task2_blank = (
        r = hp.r,
        wage = hp.wage,
        todo = "Iterate over ages, carrying saved capital into next period income.",
    )
end

# ╔═╡ dddddddd-0811-4ddd-8ddd-dddddddddddd
begin
    lifecycle = simulate_lifecycle(savings_rates; r = hp.r, wage = hp.wage)

    task2_table = [
        (
            age = h,
            income = lifecycle.income[h],
            consumption = lifecycle.consumption[h],
            next_capital = h < A_task1 ? lifecycle.capital[h] : 0.0,
        )
        for h in ages
    ]
end

# ╔═╡ eeeeeeee-0811-4eee-8eee-eeeeeeeeeeee
begin
    fig_lifecycle = Figure(size = (900, 360))
    ax_cons = Axis(fig_lifecycle[1, 1],
        xlabel = "Age h",
        ylabel = "Consumption c_h",
        title = "Lifecycle Consumption")
    lines!(ax_cons, ages, lifecycle.consumption; color = :steelblue, linewidth = 2)
    scatter!(ax_cons, ages, lifecycle.consumption; color = :steelblue, markersize = 8)

    ax_cap = Axis(fig_lifecycle[1, 2],
        xlabel = "Age h",
        ylabel = "Capital k_h",
        title = "Capital Holdings by Age")
    lines!(ax_cap, ages[2:end], lifecycle.capital; color = :firebrick, linewidth = 2)
    scatter!(ax_cap, ages[2:end], lifecycle.capital; color = :firebrick, markersize = 8)
    fig_lifecycle
end

# ╔═╡ ffffffff-0811-4fff-8fff-ffffffffffff
md"""
### Solution Check

Consumption falls over the lifecycle because capital income shrinks rapidly.
With `r < 1`, each generation saves a smaller fraction of a smaller income.
"""

# ╔═╡ a1111111-0811-4a11-8a11-a11111111111
md"""
## Task 3: Effect of Patience (10 min)

How does the discount factor \$\beta\$ affect lifecycle behavior? Compare lifecycle **consumption profiles** for \$\beta \in \{0.5, 0.7, 0.9\}\$, keeping \$A = 6\$, \$r = 0.15\$, \$w = 1.0\$ fixed. Plot all three profiles on the same axes — what happens as \$\beta\$ increases?

The TODO scaffold is next; the worked solution loops over `beta_values = [0.5, 0.7, 0.9]`, recomputes `exercise_savings_rates` and `simulate_lifecycle` for each, and overlays the three consumption curves.

> **Preview note.** The full Python notebook draws the three \$\beta\$ curves with the palette `['#e74c3c', 'steelblue', '#2ecc71']`; this preview uses the idiomatic CairoMakie palette `(:firebrick, :steelblue, :seagreen)`, so the red and green hues differ slightly while the curve ordering and labels are identical.
"""

# ╔═╡ a2222222-0811-4a22-8a22-a22222222222
begin
    # TODO: Loop over beta_values = [0.5, 0.7, 0.9].
    # TODO: Compute savings rates and lifecycle consumption for each beta.
    # TODO: Plot the three consumption profiles on the same axes.
    task3_blank = (
        beta_values = [0.5, 0.7, 0.9],
        todo = "For each beta, recompute rates, simulate the lifecycle, and overlay consumption.",
    )
end

# ╔═╡ a3333333-0811-4a33-8a33-a33333333333
begin
    beta_values = [0.5, 0.7, 0.9]
    lifecycle_by_beta = Dict(
        beta_val => simulate_lifecycle(exercise_savings_rates(A_task1, beta_val);
            r = hp.r, wage = hp.wage)
        for beta_val in beta_values
    )
end

# ╔═╡ a4444444-0811-4a44-8a44-a44444444444
begin
    fig_patience = Figure(size = figure_size(RUN_MODE))
    ax_patience = Axis(fig_patience[1, 1],
        xlabel = "Age h",
        ylabel = "Consumption c_h",
        title = "Effect of Patience on Lifecycle Consumption")
    colors = (:firebrick, :steelblue, :seagreen)
    for (beta_val, color) in zip(beta_values, colors)
        cons = lifecycle_by_beta[beta_val].consumption
        lines!(ax_patience, ages, cons; color, linewidth = 2, label = "beta = $(beta_val)")
        scatter!(ax_patience, ages, cons; color, markersize = 8)
    end
    axislegend(ax_patience; position = :rt)
    fig_patience
end

# ╔═╡ a5555555-0811-4a55-8a55-a55555555555
md"""
### Solution Check

More patient agents consume less when young and save more. Because `r < 1`,
capital income still declines with age, but higher patience flattens the
consumption profile relative to lower patience.
"""

# ╔═╡ a6666666-0811-4a66-8a66-a66666666666
begin
    validation_params = AnalyticOLGParams(n_ages = A_task1, beta = beta_task1, gamma = 1.0)
    validation_k = reshape(vcat(0.0, lifecycle.capital), :, 1)
    validation_states = analytic_olg_assemble_states(1, validation_k)
    validation_features = analytic_olg_features(validation_states; params = validation_params)
    raw_exact = rates_to_raw_logits(savings_rates[1:(end - 1)], validation_params)
    exact_model = ConstantSavingsRaw(raw_exact)
    validation_diagnostics, validation_state = analytic_olg_residual(
        exact_model, nothing, NamedTuple(), validation_states; params = validation_params)
    exact_policy_error = analytic_olg_policy_error(
        validation_diagnostics.savings, validation_states; params = validation_params).summary
end

# ╔═╡ a7777777-0811-4a77-8a77-a77777777777
md"""
## Summary

**Key takeaways from this exercise:**

1. **Savings rates decline with age.** Young agents save a large fraction of income because they have many periods left to enjoy returns; the oldest agent saves nothing.
2. **Lifecycle consumption** depends on both savings behavior and capital returns. With \$r < 1\$, capital income shrinks rapidly across generations, producing a declining consumption profile.
3. **The discount factor \$\beta\$ governs the patience–savings tradeoff.** Higher \$\beta\$ means agents value future consumption more, so they save more when young, flattening the lifecycle profile.
4. **Connection to DEQN.** In Notebook `lecture_08_08_OLG_Analytic_DEQN_persistent.ipynb` a neural network is trained to learn these savings rates from the Euler equations. The analytical formulas above are the *exact* solution the network approximates — comparing DEQN output to these closed-form values is a key validation step.

The final cell below re-uses this closed-form policy as an exact-solution check: it feeds the closed-form savings logits through `analytic_olg_residual` (via a tiny `ConstantSavingsRaw` model with the Lux-style `model(features, ps, st)` signature) and returns a machine-checkable diagnostic NamedTuple.
"""

# ╔═╡ a8888888-0811-4a88-8a88-a88888888888
(
    run_mode = RUN_MODE,
    seed = SEED,
    n_ages = A_task1,
    savings_rates = savings_rates,
    youngest_saves_most = argmax(savings_rates) == 1,
    oldest_saving_rate = savings_rates[end],
    lifecycle_consumption = lifecycle.consumption,
    patience_consumption_first_age = [lifecycle_by_beta[b].consumption[1] for b in beta_values],
    patience_consumption_last_age = [lifecycle_by_beta[b].consumption[end] for b in beta_values],
    feature_batch_shape = size(validation_features),
    lux_style_state_preserved = validation_state == NamedTuple(),
    exact_policy_error_max = exact_policy_error.max_abs,
    validation_loss = validation_diagnostics.loss,
)

# ╔═╡ Cell order:
# ╟─11111111-0811-4111-8111-111111111111
# ╟─ed0973b4-e5e9-ca37-767f-003e3b7e39a6
# ╟─9100fb6c-aca2-ccee-56dc-343bd3759bed
# ╠═22222222-0811-4222-8222-222222222222
# ╠═33333333-0811-4333-8333-333333333333
# ╠═44444444-0811-4444-8444-444444444444
# ╟─55555555-0811-4555-8555-555555555555
# ╟─66666666-0811-4666-8666-666666666666
# ╠═77777777-0811-4777-8777-777777777777
# ╠═88888888-0811-4888-8888-888888888888
# ╠═99999999-0811-4999-8999-999999999999
# ╟─aaaaaaaa-0811-4aaa-8aaa-aaaaaaaaaaaa
# ╟─bbbbbbbb-0811-4bbb-8bbb-bbbbbbbbbbbb
# ╠═cccccccc-0811-4ccc-8ccc-cccccccccccc
# ╠═dddddddd-0811-4ddd-8ddd-dddddddddddd
# ╠═eeeeeeee-0811-4eee-8eee-eeeeeeeeeeee
# ╟─ffffffff-0811-4fff-8fff-ffffffffffff
# ╟─a1111111-0811-4a11-8a11-a11111111111
# ╠═a2222222-0811-4a22-8a22-a22222222222
# ╠═a3333333-0811-4a33-8a33-a33333333333
# ╠═a4444444-0811-4a44-8a44-a44444444444
# ╟─a5555555-0811-4a55-8a55-a55555555555
# ╠═a6666666-0811-4a66-8a66-a66666666666
# ╟─a7777777-0811-4a77-8a77-a77777777777
# ╠═a8888888-0811-4a88-8a88-a88888888888
