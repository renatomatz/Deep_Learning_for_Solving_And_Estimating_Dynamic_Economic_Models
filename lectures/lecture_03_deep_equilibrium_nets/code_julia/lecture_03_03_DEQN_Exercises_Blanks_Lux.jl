### A Pluto.jl notebook ###
# v0.20.0

using Markdown
using InteractiveUtils

# ╔═╡ 11111111-0303-4111-8111-111111111111
md"""
# Lecture 03, Notebook 03: DEQN Exercises (Blanks)

This Pluto notebook preserves the exercise structure from the Python blank
notebook. The TODO cells are intentional: they ask students to complete policy
transforms, residuals, and complementarity terms without changing the shared
Brock-Mirman helpers.
"""

# ╔═╡ 13a8a3b1-9252-e5b4-2bd9-28fbf8420194
md"""
## Lecture 03, Notebook 03: DEQN Exercises (Blanks)

**Course:** Deep Learning for Solving and Estimating Dynamic Models in Economics and Finance
**Script reference:** §2.4–2.5 (Brock–Mirman benchmark; KKT + Fischer–Burmeister complementarity); previews the IRBC model of Ch. 3 and the OLG model of Ch. 5
**Notebook role:** exercise
**Author:** Simon Scheidegger

*Julia/Lux/Pluto preview of* `lectures/lecture_03_deep_equilibrium_nets/code/lecture_03_03_DEQN_Exercises_Blanks.ipynb`.

> **This is the blanks notebook.** The code cells below deliberately contain `TODO` placeholders and `error(...)` stubs — filling them in is the exercise. Do not expect it to run end-to-end as shipped. The companion **Notebook 04 (Solutions)** gives one compact Lux-native implementation.
"""

# ╔═╡ 533ed44b-20a7-4156-4405-6b43f0f8d0bd
md"""
## Simple Introduction to Deep Equilibrium Nets

### Notebook 3: exercise / coding session on Deep Equilibrium Nets

Building on Notebooks 1 and 2, this session works through **four exercises** of increasing richness:

1. **Stochastic Brock–Mirman** — the model of Notebook 2 (recap below).
2. **Endogenous labor supply** — adds an intratemporal Euler equation, so the loss now sums two optimality conditions.
3. **Occasionally binding labor constraint** — a time constraint \$L_t \le 1.01\$ handled with Fischer–Burmeister complementarity.
4. **A small overlapping-generations (OLG) economy** — five age-specific savings policies with borrowing constraints.

Each exercise is set up in Lux with `DLEFJulia` helpers; the blanks are marked with `TODO`. Work them in order — later exercises reuse the machinery of earlier ones.
"""

# ╔═╡ 22222222-0303-4222-8222-222222222222
begin
    import Pkg
    Pkg.activate(joinpath(@__DIR__, "..", "..", "..", "julia"))

    using DLEFJulia
    using NNlib
end

# ╔═╡ 57b733cb-0038-ecaf-e915-5613b82f1617
md"""
`RUN_MODE` selects the training budget and `SEED = 0` fixes the RNG; this preview keeps `RUN_MODE = "smoke"` and `SEED = 0`. The `smoke` budget is tiny (a few steps) — just enough to check that a filled-in exercise loads and produces finite residuals, not to reproduce production-quality policies.
"""

# ╔═╡ 33333333-0303-4333-8333-333333333333
begin
    RUN_MODE = "smoke"
    SEED = 0
    budgets = (
        smoke = (batch_size = 16, steps = 5),
        teaching = (batch_size = 64, steps = 100),
        production = (batch_size = 128, steps = 1_000),
    )
    hp = run_mode_budget(RUN_MODE; budgets)
    rng = rng_from_seed(SEED)
end

# ╔═╡ 44444444-0303-4444-8444-444444444444
md"""
## Exercise 1: stochastic Brock–Mirman

We start from the exact model of Notebook 2. The planner maximizes \$\sum_t \beta^t E[\ln C_t]\$ subject to \$K_{t+1} + C_t = Y_t + (1-\delta)K_t\$, with \$Y_t = z_t K_t^{\alpha}\$ and \$\log z_t = \varrho \log z_{t-1} + \sigma \epsilon_t\$, \$\epsilon_t \sim N(0,1)\$. The state \$\mathbf{X}_t = [z_t, K_t]\$ is 2-dimensional; the policy \$K_{t+1} = f(\mathbf{X}_t)\$ is approximated by \$\mathcal{N}(\mathbf{X}_t)\$. The stochastic Euler equation, in relative-consumption-error form, is

\$\$0 = \frac{1}{C_t\,\beta\, E\!\left[\frac{1}{C_{t+1}}(1 - \delta + r_{t+1})\right]} - 1, \qquad r_t = \alpha z_t K_t^{\alpha-1},\$\$

and the expectation is taken with Gauss–Hermite quadrature over next-period productivity.

> **Your task.** Use `BrockMirmanParams`, `gauss_hermite_rule`, and `stochastic_bm_residual`. Complete the missing **savings transform** (map the raw network output into a feasible savings rate \$s_t \in (0,1)\$ — the sigmoid is the natural, feasibility-guaranteeing choice) and train the network on simulated states.
"""

# ╔═╡ 55555555-0303-4555-8555-555555555555
begin
    params_ex1 = BrockMirmanParams(alpha = 0.36, beta = 0.99, delta = 0.1)
    rule_ex1 = gauss_hermite_rule(5)
    states_ex1 = vcat(ones(1, hp.batch_size), reshape(range(0.9, 12.0; length = hp.batch_size), 1, :))

    # TODO: choose a bounded transform for the raw network output.
    savings_transform_todo(raw) = error("TODO: map raw network output into a feasible savings rate")

    exercise_1_prompt = (
        state_shape = size(states_ex1),
        quadrature_weight_sum = sum(rule_ex1.weights),
        todo = "Define savings_transform_todo and pass it to stochastic_bm_residual.",
    )
end

# ╔═╡ 66666666-0303-4666-8666-666666666666
md"""
## Exercise 2: endogenous labor supply

Now add endogenous labor. The planner maximizes

\$\$\sum_{t=0}^{\infty} \beta^{t} E\!\left[\ln(C_t) - \psi\frac{L_t^{1+\theta}}{1+\theta}\right], \qquad Y_t = z_t L_t^{1-\alpha} K_t^{\alpha},\$\$

so the policy \$\mathbf{f}(\mathbf{X}_t) = [K_{t+1}, L_t]\$ is now **two-dimensional**. This yields **two optimality conditions**. With rental rate and wage

\$\$r_t = \alpha K_t^{\alpha-1} L_t^{1-\alpha}, \qquad w_t = (1-\alpha)K_t^{\alpha} L_t^{-\alpha},\$\$

the capital Euler equation is as in Exercise 1, and the new intratemporal labor FOC is

\$\$0 = \frac{w_t}{C_t\,\psi\, L_t^{\theta}} - 1.\$\$

Savings again uses a sigmoid share; labor must be **positive**, so a softplus output head is natural.

> **Your task.** The network should output two policies: savings and labor. Split the two output heads with `split_output_heads`, bound each head with economics-inspired transforms, and add the intratemporal labor residual to the Euler residual.
"""

# ╔═╡ 77777777-0303-4777-8777-777777777777
begin
    raw_two_head_example = reshape(range(-1.0, 1.0; length = 2 * hp.batch_size), 2, hp.batch_size)
    heads_ex2 = split_output_heads(raw_two_head_example, (savings = 1, labor = 1))

    # TODO: replace these placeholders with feasible transforms and residuals.
    savings_todo = "TODO: transform heads_ex2.savings into K_{t+1} or a savings rate"
    labor_todo = "TODO: transform heads_ex2.labor into positive labor supply"
    labor_residual_todo = "TODO: implement the intratemporal FOC residual"

    exercise_2_prompt = (head_names = keys(heads_ex2), savings_todo, labor_todo, labor_residual_todo)
end

# ╔═╡ 88888888-0303-4888-8888-888888888888
md"""
## Exercise 3: occasionally binding labor constraint

Impose a time constraint \$L_t \le 1.01\$. The interior labor FOC \$\frac{w_t}{C_t\psi L_t^{\theta}} - 1 = 0\$ now holds only when the choice is interior; otherwise the Kuhn–Tucker (KKT) conditions apply:

\$\$0 = \frac{w_t}{C_t \psi L_t^{\theta}} - 1 + \lambda_t, \quad 0 \le \lambda_t, \quad 0 \le 1.01 - L_t, \quad \lambda_t(1.01 - L_t) = 0.\$\$

A single **Fischer–Burmeister** equation encodes this complementarity exactly:

\$\$f^{FB}(a, b) = \sqrt{a^2 + b^2} - a - b,\$\$

which is zero precisely when \$a = 0, b \ge 0\$ or \$a \ge 0, b = 0\$. Take \$a = \frac{w_t}{C_t \psi L_t^{\theta}} - 1\$ and \$b = 1.01 - L_t\$. (What economics-inspired output activation keeps labor within its bound?)

> **Your task.** Use `fischer_burmeister(a, b)` to encode the Kuhn-Tucker condition for the time constraint. The residual should be zero when the multiplier is positive and the constraint binds, or when the multiplier is zero and the unconstrained FOC holds.
"""

# ╔═╡ 99999999-0303-4999-8999-999999999999
begin
    labor_cap = 1.01
    labor_guess = fill(0.95, 1, hp.batch_size)
    fb_demo = fischer_burmeister(labor_cap .- labor_guess, fill(0.0, 1, hp.batch_size))

    # TODO: define fb_lab_a and fb_lab_b for the constrained labor FOC.
    fb_lab_a_todo = "TODO: slack in the labor time constraint"
    fb_lab_b_todo = "TODO: nonnegative multiplier or FOC wedge"

    exercise_3_prompt = (labor_cap = labor_cap, fb_demo_max_abs = maximum(abs.(fb_demo)), fb_lab_a_todo, fb_lab_b_todo)
end

# ╔═╡ aaaaaaaa-0303-4aaa-8aaa-aaaaaaaaaaaa
md"""
## Exercise 4: small life-cycle (OLG) economy

Households live deterministically for \$H = 6\$ periods (one period ≈ 10 years, ages 20–80). Let \$h \in \{0,\dots,5\}\$ index age groups with capital \$k_t^h\$ and a borrowing constraint \$k_t^h \ge 0\$; each supplies exogenous efficiency labor \$l^h\$ (lower in the last two, retirement) periods. Aggregates are \$K_t = \sum_h k_t^h\$ and \$L = \sum_h l^h\$, with \$r_t = \alpha K_t^{\alpha-1} L^{1-\alpha}\$ and \$w_t = (1-\alpha)K_t^{\alpha} L^{-\alpha}\$. The state stacks productivity and the capital distribution, and there are **five** savings policies (all ages but the last, who consume everything). For each age the Euler equation

\$\$\frac{1}{c_t^h} \ge \beta\, E\!\left[\frac{1}{c_{t+1}^{h+1}}(1 - \delta + r_{t+1})\right], \qquad c_t^h = l^h w_t + k_t^h(1-\delta+r_t) - k_{t+1}^{h+1},\$\$

holds with equality whenever \$k_{t+1}^{h+1} > 0\$ — again a Fischer–Burmeister complementarity (with \$b = k_{t+1}^{h+1}\$). Feasibility is enforced with a sigmoid savings rate out of cash-at-hand \$cah_t^h = l^h w_t + k_t^h(1-\delta+r_t)\$.

> **Your task — keep the exercise blank.** Construct age-specific savings policies, form aggregate capital and labor, and use Fischer-Burmeister residuals for borrowing constraints. The solution notebook gives one compact implementation.
"""

# ╔═╡ d14b9dcd-e71f-40d7-0d48-5b6d05a1238c
md"""
### Wrapping up

The cell below returns a machine-checkable summary of the exercise **prompts** — state shapes, quadrature checks, and the `TODO` strings — together with `blanks_are_intentional = true`. It is a scaffold for your work, not a solved model; compare against **Notebook 04 (Solutions)** once you have attempted the blanks.
"""

# ╔═╡ bbbbbbbb-0303-4bbb-8bbb-bbbbbbbbbbbb
(
    run_mode = RUN_MODE,
    seed = SEED,
    exercise_1 = exercise_1_prompt,
    exercise_2 = exercise_2_prompt,
    exercise_3 = exercise_3_prompt,
    blanks_are_intentional = true,
)

# ╔═╡ Cell order:
# ╟─11111111-0303-4111-8111-111111111111
# ╟─13a8a3b1-9252-e5b4-2bd9-28fbf8420194
# ╟─533ed44b-20a7-4156-4405-6b43f0f8d0bd
# ╠═22222222-0303-4222-8222-222222222222
# ╟─57b733cb-0038-ecaf-e915-5613b82f1617
# ╠═33333333-0303-4333-8333-333333333333
# ╟─44444444-0303-4444-8444-444444444444
# ╠═55555555-0303-4555-8555-555555555555
# ╟─66666666-0303-4666-8666-666666666666
# ╠═77777777-0303-4777-8777-777777777777
# ╟─88888888-0303-4888-8888-888888888888
# ╠═99999999-0303-4999-8999-999999999999
# ╟─aaaaaaaa-0303-4aaa-8aaa-aaaaaaaaaaaa
# ╟─d14b9dcd-e71f-40d7-0d48-5b6d05a1238c
# ╠═bbbbbbbb-0303-4bbb-8bbb-bbbbbbbbbbbb
