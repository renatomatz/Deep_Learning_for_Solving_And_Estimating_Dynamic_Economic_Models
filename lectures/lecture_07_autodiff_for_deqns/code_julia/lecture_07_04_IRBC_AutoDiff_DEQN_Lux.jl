### A Pluto.jl notebook ###
# v0.20.0

using Markdown
using InteractiveUtils

# ╔═╡ 11111111-0704-4111-8111-111111111111
md"""
# Lecture 07, Notebook 04: IRBC via Autodiff in Julia

This compact Pluto translation mirrors the Python notebook's derivative
exercise: define the per-country Lagrangian primitive `Pi`, recover the Euler
residual from slot derivatives, and compare those derivatives with the
hand-derived Day 3 formulas. It also keeps smoke-fast Adam training paths for
the Python notebook's Approach A (exogenous sampling) and Approach B
(simulation-based sampling). The shared Lecture 04 IRBC helpers use a different
compact state and policy parameterization, so the economic residuals here stay
notebook-local.
"""

# ╔═╡ 66f8e120-b474-3ea7-9a32-ce7e2458221f
md"""
## Lecture 07, Notebook 04: IRBC via Autodiff (Self-Study)

**Course:** Deep Learning for Solving and Estimating Dynamic Models in Economics and Finance
**Script reference:** §2.7 (autodiff template) applied to the IRBC model of Ch. 3 (§3.x)
**Notebook role:** extension
**Author:** Simon Scheidegger

*Julia/Lux/Pluto preview of* `lectures/lecture_07_autodiff_for_deqns/code/lecture_07_04_IRBC_AutoDiff_DEQN.ipynb`.

> **Run mode.** The checked-in run uses `RUN_MODE = "smoke"` and `SEED = 0` (3 training episodes — a diagnostic budget, not a convergence run). The per-country slot derivatives that the Python notebook obtains from two `tf.GradientTape` calls are computed with `ForwardDiff` here; the parameter gradient of the loss uses `Zygote`. Because nested `ForwardDiff`-inside-`Zygote` is heavy, training differentiates a residual built from the verified closed-form slot derivatives, and `residual_training_parity` checks it matches the `ForwardDiff` residual to ~1e-10.
"""

# ╔═╡ 53b3c4b9-723f-35ae-d0d6-4a444744444c
md"""
## Notebook 04 — IRBC (basic model) via Automatic Differentiation

> **Status: additional materials / self-study.** This notebook is *not* covered in the Day 4 lecture slot. It revisits the Day 3 baseline IRBC model (`lectures/lecture_04_irbc_with_deqns/code/lecture_04_01_IRBC_DEQN_smooth.ipynb`) using the autodiff template developed in Day 4 notebooks 02 and 03, scaled up from a single-state Brock–Mirman to a multi-country, multi-equation planner problem with adjustment costs and an irreversibility KKT complementarity.

### Learning goals

By the end of this notebook you will be able to:

1. **Identify the planner Lagrangian as the autodiff handle.** In the IRBC model, the per-country contribution \$\Pi^j(k_{\text{in}}, k'_{\text{out}}, z; \lambda, \mu^j)\$ to the planner Lagrangian is the *only* user-supplied object — every Euler residual comes out of two slot derivatives (`tf.GradientTape` in Python, `ForwardDiff` here).
2. **Derive on paper, then implement, the exact correspondence**
\$\$\partial_2 \Pi^j + \beta\,\mathbb{E}[\partial_1 \Pi^j] \;=\; 0\$\$
and verify it agrees with the Day 3 *hand-derived* Euler residual to machine precision.
3. **See what stays hand-coded.** The aggregate resource constraint (ARC) and the Fischer–Burmeister complementarity for irreversibility are *not* envelope outputs, so autodiff has nothing to add. We document why and keep them as simple algebra.
4. **Compare two sampling strategies** under the autodiff loss with the *same* architecture, *same* training budget, and *no* loss balancing (no ReLoBRaLo):
   - **Approach A** — exogenous uniform sampling.
   - **Approach B** — simulation-based (endogenous) sampling.

### What is *not* in this notebook
- ReLoBRaLo loss balancing — covered in Day 3 nb 01 Approach D and the lecture script. The point here is the autodiff template, not loss balancing.
- Pre-training (Day 3 Approach C). Approaches A and B already make the sampling-strategy comparison clean.

The setup cell below loads Lux, DLEFJulia, `ForwardDiff`, `Zygote`, and `Optimisers` in place of the Python notebook's NumPy/TensorFlow/Keras imports.
"""

# ╔═╡ 22222222-0704-4222-8222-222222222222
begin
    import Pkg
    Pkg.activate(joinpath(@__DIR__, "..", "..", "..", "julia"))

    using DLEFJulia
    using ForwardDiff
    using LinearAlgebra
    using Lux
    using NNlib
    using Optimisers
    using Statistics
    using Zygote
end

# ╔═╡ 33333333-0704-4333-8333-333333333333
begin
    RUN_MODE = "smoke"
    SEED = 0
    budgets = (
        smoke = (batch = 32, sim_tracks = 12, sim_periods = 3, width = 32, q = 3,
            train_episodes = 3, n_data_A = 8, n_tracks_B = 4, n_sim_periods_B = 2,
            learning_rate = 1e-3),
        teaching = (batch = 128, sim_tracks = 64, sim_periods = 5, width = 64, q = 3,
            train_episodes = 100, n_data_A = 128, n_tracks_B = 64, n_sim_periods_B = 5,
            learning_rate = 1e-3),
        production = (batch = 256, sim_tracks = 128, sim_periods = 8, width = 64, q = 5,
            train_episodes = 400, n_data_A = 256, n_tracks_B = 128, n_sim_periods_B = 5,
            learning_rate = 1e-3),
    )
    hp = run_mode_budget(RUN_MODE; budgets)
    rng = rng_from_seed(SEED)
end

# ╔═╡ f61d8f4c-de27-96e3-205c-c9e4f1a5a5fd
md"""
### 1. The planner problem

A Pareto-weighted social planner with weights \$\tau^j\$ chooses \$\{c^j_t,\, k^{j\prime}_t,\, \mu^j_t,\, \lambda_t\}_{j=1\ldots N}\$ to maximize

\$\$\mathbb{E}_0 \sum_{t=0}^\infty \beta^t \sum_{j=1}^N \tau^j\, u^j(c^j_t), \qquad u^j(c) = \frac{c^{1-1/\gamma_j}}{1-1/\gamma_j},\$\$

subject to

| Constraint | Equation |
|---|---|
| Aggregate resource constraint (ARC) | \$\sum_j\!\bigl[c^j_t + k^{j\prime}_t + \Gamma(k^j_t,k^{j\prime}_t)\bigr] = \sum_j\!\bigl[Y^j(k^j_t,z^j_t) + (1-\delta)k^j_t\bigr]\$ |
| Irreversibility | \$I^j_t \;\equiv\; k^{j\prime}_t - (1-\delta)k^j_t \;\ge\; 0\$ |
| Production | \$Y^j_t = A_{\rm tfp}\, e^{z^j_t}\, (k^j_t)^\zeta\$ |
| Adjustment cost | \$\Gamma(k, k') = \tfrac{\kappa}{2}\, k\, (k'/k - 1)^2\$ |
| Exogenous TFP | \$z^{j\prime}_t = \rho_z z^j_t + \sigma_e\,(\varepsilon^j_{t+1} + \varepsilon^{\rm agg}_{t+1})\$, \$\varepsilon \sim \mathcal{N}(0, 1)\$ |

The TFP calibration \$A_{\rm tfp} = (1-\beta(1-\delta)) / (\zeta\beta)\$ pins the deterministic steady state at \$k_{\rm ss} = 1\$.

**The same model as Day 3 nb 01.** Our goal is to re-solve it without ever writing down a partial derivative on paper. The cell below builds these parameters (`irbc_autodiff_params`) and the steady-state reference `steady_state_checks`.
"""

# ╔═╡ fce66f8f-116d-804c-74fc-a83f699574db
md"""
### 2. Lagrangian

Attach \$\lambda_t\$ to the ARC and \$\mu^j_t\$ to each irreversibility constraint:

\$\$\mathcal{L}_t \;=\; \sum_j \tau^j\, u^j(c^j_t) \;+\; \lambda_t \sum_j\!\Bigl[Y^j_t + (1-\delta)k^j_t - k^{j\prime}_t - \Gamma^j_t - c^j_t\Bigr] \;+\; \sum_j \mu^j_t\,I^j_t.\$\$

Both multipliers are **policy-network outputs**, not auxiliary variables — exactly the layout of Day 3 nb 01:

- State: \$\mathbf{s}_t = [k^1_t, \ldots, k^N_t,\; z^1_t, \ldots, z^N_t]\$ — dimension \$2N\$ (`n_states`).
- Policy: \$\mathcal{N}_\rho(\mathbf{s}_t) = [k^{1\prime}_t, \ldots, k^{N\prime}_t,\; \lambda_t,\; \mu^1_t, \ldots, \mu^N_t]\$ — dimension \$2N+1\$ (`n_policies`).

The consumption FOC \$\partial \mathcal{L}_t / \partial c^j_t = \tau^j u^{j\prime}(c^j) - \lambda = 0\$ gives

\$\$c^j(\lambda) \;=\; \bigl(\lambda/\tau^j\bigr)^{-\gamma_j},\$\$

so \$c^j\$ is *not* a separate network output — it is recovered from \$\lambda\$ in closed form (the `consumption` helper).
"""

# ╔═╡ 44444444-0704-4444-8444-444444444444
begin
    function irbc_autodiff_params(; n_countries = 2)
        beta = 0.99
        gamma_min = 0.25
        gamma_max = 1.0
        zeta = 0.36
        delta = 0.01
        rho_z = 0.95
        sigma_e = 0.01
        kappa = 0.5
        k_ss = 1.0
        eps_safe = 1e-10
        a_tfp = (1 - beta * (1 - delta)) / (zeta * beta)
        gammas = n_countries == 1 ?
            [gamma_min] :
            [gamma_min + (j - 1) * (gamma_max - gamma_min) / (n_countries - 1)
             for j in 1:n_countries]
        taus = [(a_tfp - delta)^(1 / g) for g in gammas]
        return (
            n_countries = n_countries,
            beta = beta,
            gamma_min = gamma_min,
            gamma_max = gamma_max,
            zeta = zeta,
            delta = delta,
            rho_z = rho_z,
            sigma_e = sigma_e,
            kappa = kappa,
            k_ss = k_ss,
            eps_safe = eps_safe,
            a_tfp = a_tfp,
            gammas = gammas,
            taus = taus,
            n_states = 2 * n_countries,
            n_policies = 2 * n_countries + 1,
            n_shocks = n_countries + 1,
        )
    end

    params = irbc_autodiff_params()
    y_ss = params.a_tfp * params.k_ss^params.zeta
    c_ss = y_ss - params.delta * params.k_ss
    lambda_ss = [params.taus[j] * c_ss^(-1 / params.gammas[j])
                 for j in 1:params.n_countries]
    steady_state_checks = (
        a_tfp = params.a_tfp,
        c_ss = c_ss,
        mpk_ss = 1 - params.delta + params.zeta * params.a_tfp,
        target_mpk = 1 / params.beta,
        lambda_ss = lambda_ss,
        arc_ss = params.n_countries * (y_ss - params.delta * params.k_ss - c_ss),
    )
end

# ╔═╡ 60ee87b7-befb-9690-f4c2-28e6ecd0573c
md"""
### 3. Gauss–Hermite quadrature for the conditional expectation

Each TFP innovation is \$\mathcal{N}(0,1)\$; the AR(1) law of motion turns \$\mathbb{E}_t[f(z_{t+1})]\$ into a Gaussian integral that we approximate by Gauss–Hermite. With \$Q\$ nodes per dimension and \$N+1\$ shocks (\$N\$ country-specific + 1 aggregate), the tensor product gives \$Q^{N+1}\$ nodes.

We use \$Q = 3 \Rightarrow Q^{N+1} = 27\$ nodes for \$N = 2\$ — the same setup as Day 3. In Julia this is `gauss_hermite_rule(hp.q)` combined with `tensor_product_rule`; `quadrature_checks` confirms the weights sum to one and reproduce the standard-normal first and second moments.
"""

# ╔═╡ 55555555-0704-4555-8555-555555555555
begin
    gh_1d = gauss_hermite_rule(hp.q; standard_normal = true, normalize = true)
    quad_rule = tensor_product_rule(ntuple(_ -> gh_1d, params.n_shocks)...)
    quadrature_checks = (
        nodes = size(quad_rule.nodes, 2),
        weight_sum = sum(quad_rule.weights),
        first_moment = quadrature_expectation(x -> x[1], quad_rule),
        second_moment = quadrature_expectation(x -> x[1]^2, quad_rule),
    )
end

# ╔═╡ bcf92f44-681a-5de0-5d2c-151bd3477866
md"""
### 6. Hard helpers (algebraic functions, no derivatives needed)

Production, adjustment-cost *value*, investment, consumption-from-FOC, utility, and the Fischer–Burmeister function. We do **not** code `production_k`, `mpk`, `adj_cost_k`, or `adj_cost_kp` here — those are exactly the hand-derived helpers we are *replacing* with autodiff. (They reappear only later, for the cross-check.)
"""

# ╔═╡ 66666666-0704-4666-8666-666666666666
begin
    production(k, z, p = params) =
        p.a_tfp * exp(z) * max(k, p.eps_safe)^p.zeta

    function adj_cost(k, kp, p = params)
        k_safe = max(k, p.eps_safe)
        ratio = kp / k_safe - 1
        return 0.5 * p.kappa * ratio^2 * k_safe
    end

    investment(k, kp, p = params) = kp - (1 - p.delta) * k

    consumption(lamb, country, p = params) =
        (lamb / p.taus[country])^(-p.gammas[country])

    function utility(c, country, p = params)
        g = p.gammas[country]
        c_safe = max(c, p.eps_safe)
        return abs(g - 1) < 1e-12 ?
            log(c_safe) :
            c_safe^(1 - 1 / g) / (1 - 1 / g)
    end

    fischer_burmeister(mu, i_val; eps = 1e-8) =
        mu + i_val - sqrt(mu^2 + i_val^2 + eps)
end

# ╔═╡ 0a579ef5-4aba-28a4-1b57-3e9161e1ec6a
md"""
### 5. The autodiff handle — a per-country Lagrangian primitive

The Lagrangian decomposes additively across countries:

\$\$\mathcal{L}_t \;=\; \sum_{j=1}^N \Pi^j\!\bigl(k^j_t,\, k^{j\prime}_t,\, z^j_t;\, \lambda_t,\, \mu^j_t\bigr),\$\$

where the *single user-supplied object* is

\$\$\boxed{\;\Pi^j\!\bigl(\underbrace{k^j_{\text{in}}}_{\text{slot 1: STATE}},\, \underbrace{k^{j\prime}_{\text{out}}}_{\text{slot 2: CHOICE}},\, z^j_{\text{in}};\; \lambda,\, \mu^j\bigr) \;\;=\;\; \tau^j\, u^j\!\bigl(c^j(\lambda)\bigr) \;+\; \lambda\!\Bigl[Y^j(k^j_{\text{in}}, z^j_{\text{in}}) + (1-\delta)k^j_{\text{in}} - k^{j\prime}_{\text{out}} - \Gamma(k^j_{\text{in}}, k^{j\prime}_{\text{out}}) - c^j(\lambda)\Bigr] \;+\; \mu^j\!\bigl[k^{j\prime}_{\text{out}} - (1-\delta)k^j_{\text{in}}\bigr]\;}\$\$

with \$c^j(\lambda) = (\lambda/\tau^j)^{-\gamma_j}\$. This is implemented as `pi_contribution`, and the two slot derivatives \$\partial_1\Pi^j\$ (`dpi_dkin`) and \$\partial_2\Pi^j\$ (`dpi_dkout`) are single `ForwardDiff.derivative` calls. It is the *only* place the economics of the IRBC model enters the loss: change \$u^j\$, \$Y^j\$, or \$\Gamma\$ here and nothing else moves.

#### Why we encode it this way

- **Slot 1 = state.** Differentiating \$\Pi^j\$ in slot 1, evaluated at *next period*'s \$(k', k'', z')\$, returns the envelope term — see §7.
- **Slot 2 = choice.** Differentiating in slot 2, evaluated at *this period*'s \$(k, k', z)\$, returns the FOC term — see §7.
- **Shock and multipliers** enter as fixed parameters held outside the differentiation. We *do not* differentiate \$\Pi^j\$ in \$\lambda\$ or \$\mu^j\$ to obtain the Euler residual: the multipliers' own FOCs are encoded by the ARC and the Karush–Kuhn–Tucker complementarity, not by the Euler equation. See §9.

> **Pedagogical note.** The Pareto-weight utility term \$\tau^j u^j(c^j(\lambda))\$ contributes \$0\$ to both \$\partial_1 \Pi^j\$ and \$\partial_2 \Pi^j\$ (no \$k\$/\$k'\$ dependence). We *keep* it in \$\Pi^j\$ for clarity — it makes the connection to "this is the planner Lagrangian" explicit, and it is the term that *would* matter if you instead differentiated in \$\lambda\$ to recover the ARC.
"""

# ╔═╡ 2e398ce7-2cd6-8149-7aba-2ae1c7c33487
md"""
#### Code-level slot table

| Math symbol | `pi_contribution` argument | Differentiated? |
|---|---|---|
| \$k^j_{\text{in}}\$ | 1 | yes (`dpi_dkin`, envelope) |
| \$k^{j\prime}_{\text{out}}\$ | 2 | yes (`dpi_dkout`, FOC) |
| \$z^j_{\text{in}}\$ | 3 | no — parameter |
| \$\lambda\$ | 4 | no — parameter |
| \$\mu^j\$ | 5 | no — parameter |
| `country` (\$j\$) | 6 | index |

`ForwardDiff.derivative` differentiates in only the relevant capital slot; the multipliers and shocks are passed in as constants. This guarantees the gradient computed is the Euler residual, not some other partial of the Lagrangian.
"""

# ╔═╡ 77777777-0704-4777-8777-777777777777
begin
    k_row(country, p = params) = country
    z_row(country, p = params) = p.n_countries + country
    kp_row(country, p = params) = country
    lambda_row(p = params) = p.n_countries + 1
    mu_row(country, p = params) = p.n_countries + 1 + country

    function pi_contribution(k_in, kp_out, z_in, lamb, mu_j, country, p = params)
        c_j = consumption(lamb, country, p)
        util_term = p.taus[country] * utility(c_j, country, p)
        y_j = production(k_in, z_in, p)
        gamma_j = adj_cost(k_in, kp_out, p)
        arc_term = lamb * (y_j + (1 - p.delta) * k_in - kp_out - gamma_j - c_j)
        irr_term = mu_j * (kp_out - (1 - p.delta) * k_in)
        return util_term + arc_term + irr_term
    end

    dpi_dkin(k_in, kp_out, z_in, lamb, mu_j, country, p = params) =
        ForwardDiff.derivative(x -> pi_contribution(x, kp_out, z_in, lamb, mu_j, country, p), k_in)

    dpi_dkout(k_in, kp_out, z_in, lamb, mu_j, country, p = params) =
        ForwardDiff.derivative(x -> pi_contribution(k_in, x, z_in, lamb, mu_j, country, p), kp_out)
end

# ╔═╡ b3b8dd2a-3da2-accc-0061-c86ffa02ecb4
md"""
### 7. From `Pi` to the Euler residual, term by term

#### 7.1 Pen-and-paper FOC for \$k^{j\prime}_t\$

Differentiating \$\mathcal{L}_t + \beta\,\mathbb{E}_t \mathcal{L}_{t+1}\$ in \$k^{j\prime}_t\$ — a choice today, a state tomorrow:

\$\$\frac{\partial \mathcal{L}_t}{\partial k^{j\prime}_t} \;+\; \beta\,\mathbb{E}_t\!\left[\frac{\partial \mathcal{L}_{t+1}}{\partial k^j_{t+1}}\right] \;=\; 0.\$\$

Inside \$\mathcal{L}_t = \sum_i \Pi^i\$, only \$\Pi^j\$ depends on \$k^{j\prime}_t\$, so the FOC collapses to

\$\$\boxed{\;\partial_2 \Pi^j\bigl(k^j_t,\, k^{j\prime}_t,\, z^j_t;\, \lambda_t,\, \mu^j_t\bigr) \;+\; \beta\,\mathbb{E}_t\!\left[\partial_1 \Pi^j\bigl(k^{j\prime}_t,\, k^{j\prime\prime}_t,\, z^j_{t+1};\, \lambda_{t+1},\, \mu^j_{t+1}\bigr)\right] \;=\; 0.\;}\$\$

This is the **same `partial_2 + beta * E[partial_1]` template as Day 4 nb 02–03**, applied \$N\$ times.

#### 7.2 What each slot derivative returns — term by term

**FOC term** \$\;\partial_2 \Pi^j\$ at \$(k^j_t,\, k^{j\prime}_t,\, z^j_t;\, \lambda_t,\, \mu^j_t)\$:

| term in \$\Pi^j\$ | \$\partial / \partial k^{j\prime}_{\text{out}}\$ |
|---|---|
| \$\tau^j u^j(c^j(\lambda))\$ | \$0\$ |
| \$\lambda\bigl(Y^j(k_{\text{in}}, z) + (1-\delta)k_{\text{in}} - c^j(\lambda)\bigr)\$ | \$0\$ |
| \$-\lambda\, k^{j\prime}_{\text{out}}\$ | \$-\lambda\$ |
| \$-\lambda\,\Gamma(k_{\text{in}}, k^{j\prime}_{\text{out}})\$ | \$-\lambda\,\Gamma_{k'}\$ |
| \$\mu^j\, k^{j\prime}_{\text{out}}\$ | \$+\mu^j\$ |

\$\$\Rightarrow\quad \partial_2 \Pi^j \;=\; -\lambda\bigl(1 + \Gamma_{k'}\bigr) + \mu^j.\$\$

**Envelope term** \$\;\partial_1 \Pi^j\$ at \$(k^{j\prime}_t,\, k^{j\prime\prime}_t,\, z^j_{t+1};\, \lambda_{t+1},\, \mu^j_{t+1})\$:

| term in \$\Pi^j\$ | \$\partial / \partial k^{j}_{\text{in}}\$ |
|---|---|
| \$\tau^j u^j(c^j(\lambda))\$ | \$0\$ |
| \$\lambda\,Y^j(k_{\text{in}},z_{\text{in}})\$ | \$\lambda\,F_k\$ |
| \$\lambda(1-\delta)k_{\text{in}}\$ | \$\lambda(1-\delta)\$ |
| \$-\lambda\,\Gamma(k_{\text{in}}, k^{j\prime}_{\text{out}})\$ | \$-\lambda\,\Gamma_k\$ |
| \$-\mu^j(1-\delta) k_{\text{in}}\$ | \$-(1-\delta)\mu^j\$ |

\$\$\Rightarrow\quad \partial_1 \Pi^j \;=\; \lambda\bigl(1-\delta + F_k - \Gamma_k\bigr) - (1-\delta)\mu^j \;=\; \lambda \cdot \mathrm{MPK}^j - (1-\delta)\mu^j.\$\$

Substituting into the boxed Euler equation,

\$\$-\lambda_t\bigl(1+\Gamma_{k'}^t\bigr) + \mu^j_t \;+\; \beta\,\mathbb{E}_t\!\bigl[\lambda_{t+1}\,\mathrm{MPK}^j_{t+1} - (1-\delta)\mu^j_{t+1}\bigr] \;=\; 0,\$\$

which is *exactly* the Day 3 Euler equation, in absolute (not relative) form.

**The cross-check.** The cell below computes these two closed forms in `hand_slot_gradient` and compares them to the `ForwardDiff` slot derivatives of `pi_contribution` at a test point; `slot_gradient_error` is the max-abs difference and should sit at machine precision.
"""

# ╔═╡ 88888888-0704-4888-8888-888888888888
begin
    production_k_hand(k, z, p = params) =
        p.zeta * p.a_tfp * exp(z) * max(k, p.eps_safe)^(p.zeta - 1)

    adj_cost_kp_hand(k, kp, p = params) =
        p.kappa * (kp / max(k, p.eps_safe) - 1)

    function adj_cost_k_hand(k, kp, p = params)
        k_safe = max(k, p.eps_safe)
        ratio = kp / k_safe - 1
        return -0.5 * p.kappa * ratio * (ratio + 2)
    end

    mpk_hand(k, z, kp, p = params) =
        1 - p.delta + production_k_hand(k, z, p) - adj_cost_k_hand(k, kp, p)

    function hand_slot_gradient(k, kp, z, lamb, mu, country, p = params)
        return [
            lamb * mpk_hand(k, z, kp, p) - (1 - p.delta) * mu,
            -lamb * (1 + adj_cost_kp_hand(k, kp, p)) + mu,
        ]
    end

    slot_point = (k = 1.1, kp = 1.04, z = 0.02, lamb = lambda_ss[1], mu = 1e-6, country = 1)
    slot_ad = ForwardDiff.gradient(
        v -> pi_contribution(v[1], v[2], slot_point.z, slot_point.lamb,
            slot_point.mu, slot_point.country, params),
        [slot_point.k, slot_point.kp],
    )
    slot_hand = hand_slot_gradient(slot_point.k, slot_point.kp, slot_point.z,
        slot_point.lamb, slot_point.mu, slot_point.country, params)
    slot_gradient_error = norm(slot_ad - slot_hand, Inf)
end

# ╔═╡ fe61a524-c5e0-afef-d8b5-1adcf6d1c395
md"""
### 11. Sampling: exogenous (Approach A) vs simulation-based (Approach B)

Later we train two *fresh* networks under two sampling strategies, with the same architecture, optimiser, and number of episodes. The only thing that differs is how training data are drawn — and the helpers below implement both:

- **Approach A — exogenous** (`sample_exogenous_states`). Each batch is a uniform draw from the rectangular domain \$[k_{\rm lb}, k_{\rm ub}]^N \times [z_{\rm lb}, z_{\rm ub}]^N\$. Cheap, but spends effort in regions the economy may never visit.
- **Approach B — simulation-based** (`make_start_states` + `simulated_state_cloud`). We start `n_tracks` parallel simulations near the steady state and propagate them forward `n_sim_periods` steps using the *current* network as the policy. The training batch is the resulting cloud of states. Adapts to the current policy and concentrates on the ergodic set, but conditions on the (possibly bad) current policy.
"""

# ╔═╡ 99999999-0704-4999-8999-999999999999
begin
    function sample_exogenous_states(rng, n, p = params)
        k_lb, k_ub = 0.5, 1.5
        z_bound = 3 * p.sigma_e / (1 - p.rho_z)
        k = k_lb .+ (k_ub - k_lb) .* rand(rng, p.n_countries, n)
        z = -z_bound .+ 2z_bound .* rand(rng, p.n_countries, n)
        return vcat(k, z)
    end

    function make_start_states(rng, n_tracks, p = params)
        k = fill(p.k_ss, p.n_countries, n_tracks) .+
            0.2 .* (rand(rng, p.n_countries, n_tracks) .- 0.5)
        z = 0.02 .* (rand(rng, p.n_countries, n_tracks) .- 0.5)
        return vcat(k, z)
    end

    function next_state_from_policy(states, policies, shock, p = params)
        batch = size(states, 2)
        eps_agg = shock[p.n_shocks]
        z_next = similar(states, p.n_countries, batch)
        for country in 1:p.n_countries
            z_next[country, :] .= p.rho_z .* states[z_row(country, p), :] .+
                                  p.sigma_e .* (shock[country] + eps_agg)
        end
        return vcat(@view(policies[1:p.n_countries, :]), z_next)
    end

    function next_state_from_policy_shocks(states, policies, shocks, p = params)
        batch = size(states, 2)
        size(shocks) == (p.n_shocks, batch) ||
            throw(DimensionMismatch("simulation shocks must be n_shocks-by-batch"))
        eps_agg = @view shocks[p.n_shocks, :]
        z_next = similar(states, p.n_countries, batch)
        for country in 1:p.n_countries
            z_next[country, :] .= p.rho_z .* states[z_row(country, p), :] .+
                                  p.sigma_e .* (shocks[country, :] .+ eps_agg)
        end
        return vcat(@view(policies[1:p.n_countries, :]), z_next)
    end

    function simulate_single_step(rng, model, ps, st, states, p = params)
        policies, st_new = model(states, ps, st)
        eps = randn(rng, p.n_shocks, size(states, 2))
        return next_state_from_policy_shocks(states, policies, eps, p), st_new
    end

    function simulated_state_cloud(rng, model, ps, st, start_states, periods, p = params)
        states = start_states
        clouds = [copy(states)]
        st_cur = st
        for _ in 2:periods
            policies, st_cur = model(states, ps, st_cur)
            eps = randn(rng, p.n_shocks, size(states, 2))
            states = next_state_from_policy_shocks(states, policies, eps, p)
            push!(clouds, copy(states))
        end
        return hcat(clouds...), states, st_cur
    end
end

# ╔═╡ 86f5edec-b957-2e66-b75a-33c754c9eed7
md"""
### 4. Neural network architecture

Identical to Day 3 nb 01: 2 hidden layers × 64 units, swish activation, softplus output — here `make_mlp(n_states, (width, width), n_policies; activation = NNlib.swish, final_activation = NNlib.softplus)`. The softplus output enforces \$k^{j\prime}_t > 0\$, \$\lambda_t > 0\$, \$\mu^j_t > 0\$ at every iteration of training (a *hard constraint*, in the sense of Day 4 nb 02). The Euler equation, the ARC, and the FB complementarity are then enforced *softly* through the loss. `lux_boundary_checks` confirms the feature-by-batch shapes and the positivity of the raw policy outputs.
"""

# ╔═╡ aaaaaaaa-0704-4aaa-8aaa-aaaaaaaaaaaa
begin
    model = make_mlp(params.n_states, (hp.width, hp.width), params.n_policies;
        activation = NNlib.swish, final_activation = NNlib.softplus)
    ps, st = setup_model(rng_from_seed(SEED; offset = 1), model; parameter_type = Float64)

    exogenous_states = sample_exogenous_states(rng, hp.batch, params)
    policies_preview, st_preview = model(exogenous_states, ps, st)
    lux_boundary_checks = (
        states_shape = size(exogenous_states),
        policies_shape = size(policies_preview),
        min_policy_output = minimum(policies_preview),
    )
end

# ╔═╡ 5fbc8801-2201-28e8-9e5a-ea76546eda3f
md"""
### 8. The autodiff cost function

Two slot derivatives per country, plus the hand-coded ARC and FB. `autodiff_irbc_residual` builds the Euler residual for country \$j\$ in three steps:

1. Roll \$(k_t, z_t)\$ forward to \$k_{t+1}\$ via the network, then take the FOC slot derivative \$\partial_2\Pi^j(k^j_t, k^{j\prime}_t, z^j_t)\$ (`dpi_dkout`).
2. For each Gauss–Hermite node, draw \$z^j_{t+1}\$, roll to \$k_{t+2}\$, take the envelope slot derivative \$\partial_1\Pi^j(k^{j\prime}_t, k^{j\prime\prime}_t, z^j_{t+1})\$ (`dpi_dkin`), and accumulate the weighted sum.
3. Combine: `euler = dpi_dkout + beta * E[dpi_dkin]`.

The same function also assembles the hand-coded ARC and Fischer–Burmeister residuals and the total squared loss.
"""

# ╔═╡ d5dd4b46-ae1d-76c2-c6bc-9dbc269e0aad
md"""
### 9. Why ARC and FB stay hand-coded

The Euler equation is the only part of the equilibrium that comes from an *envelope-theorem-style* derivation. The remaining two pieces are different mathematical objects:

#### ARC = \$\partial_\lambda \mathcal{L}_t\$ — algebraic, not envelope

Differentiating \$\mathcal{L}_t\$ in the multiplier \$\lambda_t\$ recovers the constraint itself (a standard property of the Lagrangian). The hand-coded one-liner \$\sum_j[Y^j + (1-\delta)k^j - k^{j\prime} - \Gamma^j - c^j(\lambda)]\$ is identical to what differentiating \$\sum_j \Pi^j\$ in \$\lambda\$ would return — but with no autodiff machinery to set up. We use the algebraic form for clarity. (Optional exercise: verify the equivalence; the key cancellation is between the consumption-FOC term and the \$-\lambda c^j\$ term.)

#### FB encodes a KKT triple — not a derivative at all

The irreversibility constraint \$I^j_t \ge 0\$ together with \$\mu^j_t \ge 0\$ generates a Karush–Kuhn–Tucker triple

\$\$\mu^j_t \;\ge\; 0, \qquad I^j_t \;\ge\; 0, \qquad \mu^j_t\, I^j_t \;=\; 0.\$\$

The Fischer–Burmeister function \$\phi(\mu, I) = \mu + I - \sqrt{\mu^2 + I^2 + \varepsilon}\$ satisfies \$\phi(\mu, I) = 0 \iff (\mu \ge 0,\ I \ge 0,\ \mu I = 0)\$ for \$\varepsilon \to 0\$, so penalising \$\phi^2\$ in the loss enforces the entire KKT triple as a single smooth equation. This is a *complementarity* statement, not a partial derivative — autodiff has no role here.

> **Autodiff replaces FOCs and envelope conditions, not algebraic constraints.** In the IRBC model that means: \$N\$ Euler equations come from slot derivatives, the ARC and the \$N\$ FB conditions stay as one-liners.
"""

# ╔═╡ bbbbbbbb-0704-4bbb-8bbb-bbbbbbbbbbbb
begin
    function autodiff_irbc_residual(model, ps, st, states, rule; p = params)
        states = assert_feature_batch(states, p.n_states)
        policies, st_new = model(states, ps, st)
        batch = size(states, 2)

        next_states = Vector{Matrix{eltype(states)}}(undef, length(rule.weights))
        next_policies = Vector{typeof(policies)}(undef, length(rule.weights))
        for q in eachindex(rule.weights)
            ns = next_state_from_policy(states, policies, @view(rule.nodes[:, q]), p)
            np, _ = model(ns, ps, st_new)
            next_states[q] = ns
            next_policies[q] = np
        end

        euler = zeros(eltype(states), p.n_countries, batch)
        for country in 1:p.n_countries, b in 1:batch
            k_t = states[k_row(country, p), b]
            kp_t = policies[kp_row(country, p), b]
            z_t = states[z_row(country, p), b]
            lamb_t = policies[lambda_row(p), b]
            mu_t = policies[mu_row(country, p), b]

            foc = dpi_dkout(k_t, kp_t, z_t, lamb_t, mu_t, country, p)
            expectation = zero(foc)
            for q in eachindex(rule.weights)
                ns = next_states[q]
                np = next_policies[q]
                env = dpi_dkin(kp_t, np[kp_row(country, p), b], ns[z_row(country, p), b],
                    np[lambda_row(p), b], np[mu_row(country, p), b], country, p)
                expectation += rule.weights[q] * env
            end
            euler[country, b] = foc + p.beta * expectation
        end

        arc = zeros(eltype(states), batch)
        fb = zeros(eltype(states), p.n_countries, batch)
        for b in 1:batch, country in 1:p.n_countries
            k_t = states[k_row(country, p), b]
            kp_t = policies[kp_row(country, p), b]
            z_t = states[z_row(country, p), b]
            lamb_t = policies[lambda_row(p), b]
            mu_t = policies[mu_row(country, p), b]
            arc[b] += production(k_t, z_t, p) + (1 - p.delta) * k_t -
                      kp_t - adj_cost(k_t, kp_t, p) - consumption(lamb_t, country, p)
            fb[country, b] = fischer_burmeister(mu_t, investment(k_t, kp_t, p))
        end

        total_sq = vec(sum(abs2, euler; dims = 1) .+
                       reshape(arc .^ 2, 1, :) .+
                       sum(abs2, fb; dims = 1))
        loss = mean(total_sq)
        return (
            loss = loss,
            euler = euler,
            arc = arc,
            fb = fb,
            policies = policies,
        ), st_new
    end
end

# ╔═╡ ee714568-d482-cfb7-b841-32eba8392edd
md"""
### 10. Cross-check 1 — autodiff vs hand-derived (Day 3 nb 01) Euler residual

We re-implement the *hand-derived* Euler residual from Day 3 nb 01 (`hand_irbc_euler_residual`), call it on the same untrained network, and report the maximum absolute difference. We expect machine-precision agreement (roughly \$10^{-5}\$ in float32; tighter in `Float64`).

Day 3's residual is the *relative-error* form
\$\$r^{\text{Day 3},j} \;=\; \frac{\beta\,\mathbb{E}[\lambda'\mathrm{MPK}^{j\prime} - (1-\delta)\mu^{j\prime}] + \mu^j}{\lambda(1+\Gamma_{k'})} \;-\; 1,\$\$

which is the *autodiff residual* (here in absolute form \$\partial_2\Pi^j + \beta\,\mathbb{E}[\partial_1\Pi^j]\$) divided by \$-\lambda(1+\Gamma_{k'})\$. We compare the *absolute* form on both sides for an apples-to-apples check.
"""

# ╔═╡ 080491ab-789c-f9f8-be15-9b60d36acc8c
md"""
**Sign convention note.** The autodiff residual is in *absolute* form (\$\partial_2 \Pi + \beta \mathbb{E}[\partial_1 \Pi]\$, equal to zero at the solution). Day 3 nb 01's training loop uses the *relative-error* form \$r^{\text{Day 3}} = \mathrm{RHS}/\mathrm{LHS} - 1\$, which is the same equation divided by \$-\partial_2\Pi^j = \lambda(1+\Gamma_{k'}) - \mu^j\$. The two losses are *not* literally equal at intermediate iterates — they re-weight the per-state contribution differently — but they vanish on the same set of policies. We use the absolute autodiff form throughout the rest of this notebook; the cross-check above verifies the underlying equation is the right one.
"""

# ╔═╡ cccccccc-0704-4ccc-8ccc-cccccccccccc
begin
    function hand_irbc_euler_residual(model, ps, st, states, rule; p = params)
        states = assert_feature_batch(states, p.n_states)
        policies, st_new = model(states, ps, st)
        batch = size(states, 2)
        euler = zeros(eltype(states), p.n_countries, batch)

        for country in 1:p.n_countries, b in 1:batch
            k_t = states[k_row(country, p), b]
            kp_t = policies[kp_row(country, p), b]
            mu_t = policies[mu_row(country, p), b]
            lamb_t = policies[lambda_row(p), b]
            lhs = -lamb_t * (1 + adj_cost_kp_hand(k_t, kp_t, p)) + mu_t

            expectation = zero(lhs)
            for q in eachindex(rule.weights)
                ns = next_state_from_policy(states, policies, @view(rule.nodes[:, q]), p)
                np, _ = model(ns, ps, st_new)
                k_n = ns[k_row(country, p), b]
                z_n = ns[z_row(country, p), b]
                kp_n = np[kp_row(country, p), b]
                lamb_n = np[lambda_row(p), b]
                mu_n = np[mu_row(country, p), b]
                expectation += rule.weights[q] *
                               (lamb_n * mpk_hand(k_n, z_n, kp_n, p) -
                                (1 - p.delta) * mu_n)
            end
            euler[country, b] = lhs + p.beta * expectation
        end
        return euler, st_new
    end

    auto_pieces, st_after_auto = autodiff_irbc_residual(model, ps, st, exogenous_states, quad_rule; p = params)
    hand_euler, _ = hand_irbc_euler_residual(model, ps, st, exogenous_states, quad_rule; p = params)
    residual_cross_check = (
        max_abs_euler_difference = maximum(abs.(auto_pieces.euler .- hand_euler)),
        autodiff_loss = auto_pieces.loss,
        euler = residual_summary(auto_pieces.euler),
        arc = residual_summary(auto_pieces.arc),
        fb = residual_summary(auto_pieces.fb),
    )
end

# ╔═╡ b76a6b94-3bd3-6501-bb9b-7356c111f7ed
md"""
### Residual cross-check on a simulated state cloud

Cross-check 1 used exogenously sampled states. Here we repeat the autodiff-vs-hand residual comparison on a **simulated** cloud of states — rolled forward from near the steady state with `simulated_state_cloud` — which reflects the ergodic region where the policy is actually used. `simulation_checks` reports the loss, the max-abs autodiff-vs-hand Euler difference, and residual summaries on this cloud.
"""

# ╔═╡ dddddddd-0704-4ddd-8ddd-dddddddddddd
begin
    start_states = make_start_states(rng, hp.sim_tracks, params)
    sim_states, sim_end_states, st_after_sim = simulated_state_cloud(
        rng, model, ps, st, start_states, hp.sim_periods, params)
    sim_pieces, _ = autodiff_irbc_residual(model, ps, st_after_sim, sim_states, quad_rule; p = params)
    sim_hand_euler, _ = hand_irbc_euler_residual(model, ps, st_after_sim, sim_states, quad_rule; p = params)
    simulation_checks = (
        simulated_cloud_shape = size(sim_states),
        end_state_shape = size(sim_end_states),
        loss = sim_pieces.loss,
        max_abs_euler_difference = maximum(abs.(sim_pieces.euler .- sim_hand_euler)),
        euler = residual_summary(sim_pieces.euler),
        arc = residual_summary(sim_pieces.arc),
        fb = residual_summary(sim_pieces.fb),
    )
end

# ╔═╡ f9e8e4a9-9f85-0f41-a758-8732ebc53475
md"""
### 12. Training budget

Both approaches use *identical* settings — the only difference is how states are sampled. The `smoke` budget here runs just a handful of episodes as a diagnostic (finite losses and gradient norms, residual parity), not a convergence run; the `teaching` and `production` budgets in the setup cell scale this up.

| | Smoke | Teaching | Production |
|---|---|---|---|
| `train_episodes` | 3 | 100 | 400 |
| `n_data_A` (Approach A) | 8 | 128 | 256 |
| `n_tracks_B` (Approach B) | 4 | 64 | 128 |
| `n_sim_periods_B` (Approach B) | 2 | 5 | 5 |
| Optimiser | Adam (`Optimisers.jl`), lr = 1e-3 | | |

> Reaching the Day 3 baseline (~\$10^{-7}\$ loss) typically requires thousands of episodes plus pre-training on the steady state — not done here so that the A-vs-B comparison stays clean.

**How the parameter gradient is taken (a Julia note).** The pedagogical residual above uses `ForwardDiff` for the slot derivatives. For *training* we need the gradient of the loss w.r.t. the **network parameters**, taken with `Zygote` (the outer tape). Nested `ForwardDiff`-inside-`Zygote` is expensive, so `zygote_irbc_residual` rebuilds the identical residual from the verified closed-form slot derivatives \$\partial_2\Pi = -\lambda(1+\Gamma_{k'})+\mu\$ and \$\partial_1\Pi = \lambda\,\mathrm{MPK} - (1-\delta)\mu\$ (`dpi_dkout_training` / `dpi_dkin_training`). `residual_training_parity` verifies the two residuals agree to ~1e-10, so `grad_autodiff` (a `Zygote.pullback`) differentiates a residual that is provably the same equation.
"""

# ╔═╡ ffffffff-0704-4fff-8fff-ffffffffffff
begin
    function next_state_from_policy_diff(states, policies, shock, p = params)
        z_rows = [reshape(p.rho_z .* states[z_row(country, p), :] .+
                          p.sigma_e .* (shock[country] .+ shock[p.n_shocks]), 1, :)
                  for country in 1:p.n_countries]
        return vcat(policies[1:p.n_countries, :], z_rows...)
    end

    dpi_dkin_training(k_in, kp_out, z_in, lamb, mu_j, country, p = params) =
        lamb * mpk_hand(k_in, z_in, kp_out, p) - (1 - p.delta) * mu_j

    dpi_dkout_training(k_in, kp_out, lamb, mu_j, p = params) =
        -lamb * (1 + adj_cost_kp_hand(k_in, kp_out, p)) + mu_j

    function zygote_irbc_residual(model, ps, st, states, rule; p = params)
        states = assert_feature_batch(states, p.n_states)
        policies, st_new = model(states, ps, st)
        batch = size(states, 2)

        next_pairs = [begin
            ns = next_state_from_policy_diff(states, policies, rule.nodes[:, q], p)
            np, _ = model(ns, ps, st_new)
            (states = ns, policies = np)
        end for q in eachindex(rule.weights)]

        euler = [begin
            k_t = states[k_row(country, p), b]
            kp_t = policies[kp_row(country, p), b]
            z_t = states[z_row(country, p), b]
            lamb_t = policies[lambda_row(p), b]
            mu_t = policies[mu_row(country, p), b]
            foc = dpi_dkout_training(k_t, kp_t, lamb_t, mu_t, p)
            expectation = sum(
                rule.weights[q] * dpi_dkin_training(
                    kp_t,
                    next_pairs[q].policies[kp_row(country, p), b],
                    next_pairs[q].states[z_row(country, p), b],
                    next_pairs[q].policies[lambda_row(p), b],
                    next_pairs[q].policies[mu_row(country, p), b],
                    country,
                    p,
                )
                for q in eachindex(rule.weights)
            )
            foc + p.beta * expectation
        end for country in 1:p.n_countries, b in 1:batch]

        arc = [sum(
            production(states[k_row(country, p), b], states[z_row(country, p), b], p) +
            (1 - p.delta) * states[k_row(country, p), b] -
            policies[kp_row(country, p), b] -
            adj_cost(states[k_row(country, p), b], policies[kp_row(country, p), b], p) -
            consumption(policies[lambda_row(p), b], country, p)
            for country in 1:p.n_countries
        ) for b in 1:batch]

        fb = [fischer_burmeister(
            policies[mu_row(country, p), b],
            investment(states[k_row(country, p), b], policies[kp_row(country, p), b], p),
        ) for country in 1:p.n_countries, b in 1:batch]

        total_sq = [sum(abs2, euler[:, b]) + arc[b]^2 + sum(abs2, fb[:, b])
                    for b in 1:batch]
        loss = mean(total_sq)
        return (
            loss = loss,
            euler = euler,
            arc = arc,
            fb = fb,
            policies = policies,
        ), st_new
    end

    autodiff_training_loss(model, ps, st, states) = begin
        pieces, st_new = zygote_irbc_residual(model, ps, st, states, quad_rule; p = params)
        return pieces.loss, st_new
    end

    function grad_autodiff(model, ps, st, states)
        (loss, st_new), back = Zygote.pullback(ps) do ps_local
            autodiff_training_loss(model, ps_local, st, states)
        end
        grads = only(back((one(loss), nothing)))
        return loss, grads, st_new
    end

    make_optimizer(lr = hp.learning_rate) = Optimisers.Adam(lr)

    function adam_autodiff_step!(train_state, states; max_grad_norm = 10.0)
        loss, grads, st_new = grad_autodiff(train_state.model, train_state.ps, train_state.st, states)
        finite_loss(loss) || throw(DomainError(loss, "autodiff loss is not finite"))
        if isfinite(max_grad_norm)
            grads, grad_norm = clip_gradient_norm(grads, max_grad_norm)
        else
            grad_norm = sqrt(tree_sum_abs2(grads))
        end
        train_state.opt_state, train_state.ps = Optimisers.update(train_state.opt_state, train_state.ps, grads)
        train_state.st = st_new
        train_state.step += 1
        return (loss = loss, grad_norm = grad_norm, step = train_state.step)
    end

    function loss_history_diagnostics(history)
        losses = [h.loss for h in history]
        grad_norms = [h.grad_norm for h in history]
        return (
            losses = losses,
            grad_norms = grad_norms,
            n_updates = length(losses),
            all_losses_finite = all(isfinite, losses),
            all_grad_norms_finite = all(isfinite, grad_norms),
            initial_loss = first(losses),
            final_loss = last(losses),
            min_loss = minimum(losses),
            loss_decreased = last(losses) <= first(losses),
            note = last(losses) <= first(losses) ?
                "Smoke Adam updates decreased the recorded loss." :
                "Smoke budget is diagnostic only; no monotone loss decrease is asserted.",
        )
    end

    function residual_training_parity(model, ps, st, states)
        auto_pieces, _ = autodiff_irbc_residual(model, ps, st, states, quad_rule; p = params)
        train_pieces, _ = zygote_irbc_residual(model, ps, st, states, quad_rule; p = params)
        hand_euler, _ = hand_irbc_euler_residual(model, ps, st, states, quad_rule; p = params)
        return (
            autodiff_vs_training_loss = abs(auto_pieces.loss - train_pieces.loss),
            autodiff_vs_training_euler = maximum(abs.(auto_pieces.euler .- train_pieces.euler)),
            autodiff_vs_training_arc = maximum(abs.(auto_pieces.arc .- train_pieces.arc)),
            autodiff_vs_training_fb = maximum(abs.(auto_pieces.fb .- train_pieces.fb)),
            autodiff_vs_hand_euler = maximum(abs.(auto_pieces.euler .- hand_euler)),
        )
    end

    parity_ok(report; tol = 1e-10) = all(abs(v) <= tol for v in values(report))
end

# ╔═╡ 714880c1-47c7-d582-63b6-3a3d4a66f0df
md"""
### Training Approaches A and B

`train_approach_A` draws each batch with `sample_exogenous_states`; `train_approach_B` draws each batch from `simulated_state_cloud`, seeded near the steady state and advanced with the current policy. Both start from the *same* network initialisation (`initial_train_ps`) and use identical optimiser settings, so any difference is due to sampling alone. Each episode calls `adam_autodiff_step!` (a `Zygote` gradient, gradient-norm clipping, and an Adam update).
"""

# ╔═╡ 50bce4f1-9c12-ffe1-18e4-d950ef96998a
md"""
> **The full Python notebook also plots**, where this compact preview returns numbers in `training_diagnostics`:
>
> - **§13 Side-by-side comparison.** Python overlays the two log-loss training curves; here `approach_A` / `approach_B` carry the loss and gradient-norm histories, validation losses, and residual-parity flags.
> - **§14 Solution diagnostics.** Python evaluates per-equation residuals and policy plots on a simulated test set under each trained network; here the diagnostics NamedTuple reports the validation loss and residual parity for each approach.
> - **§15 Ergodic distribution.** Python scatters the long-run \$(k_1, k_2)\$ cloud — centred on \$k_{\rm ss}=1\$ with a "cigar" along the diagonal driven by the aggregate shock. The Julia preview reports the simulated-cloud shapes rather than the scatter.
>
> At the smoke budget these are *diagnostic* checks (finite, parity-exact), not a convergence comparison.
"""

# ╔═╡ 12121212-0704-4121-8121-121212121212
begin
    initial_train_ps, initial_train_st = setup_model(
        rng_from_seed(SEED; offset = 11), model; parameter_type = Float64)

    function train_approach_A()
        local_rng = rng_from_seed(SEED; offset = 12)
        state_A = setup_training(model, deepcopy(initial_train_ps), deepcopy(initial_train_st),
            make_optimizer(hp.learning_rate))
        history = NamedTuple[]
        for _ in 1:hp.train_episodes
            X = sample_exogenous_states(local_rng, hp.n_data_A, params)
            metrics = adam_autodiff_step!(state_A, X; max_grad_norm = 10.0)
            append_metric!(history; step = metrics.step, loss = Float64(metrics.loss),
                grad_norm = Float64(metrics.grad_norm))
        end
        return state_A, history
    end

    function train_approach_B()
        local_rng = rng_from_seed(SEED; offset = 13)
        state_B = setup_training(model, deepcopy(initial_train_ps), deepcopy(initial_train_st),
            make_optimizer(hp.learning_rate))
        start_B = make_start_states(local_rng, hp.n_tracks_B, params)
        history = NamedTuple[]
        last_cloud = start_B
        for _ in 1:hp.train_episodes
            X, end_states, _ = simulated_state_cloud(local_rng, state_B.model, state_B.ps,
                state_B.st, start_B, hp.n_sim_periods_B, params)
            metrics = adam_autodiff_step!(state_B, X; max_grad_norm = 10.0)
            append_metric!(history; step = metrics.step, loss = Float64(metrics.loss),
                grad_norm = Float64(metrics.grad_norm))
            start_B = end_states
            last_cloud = X
        end
        return state_B, history, last_cloud, start_B
    end

    state_A, history_A = train_approach_A()
    state_B, history_B, last_cloud_B, end_states_B = train_approach_B()

    validation_states = sample_exogenous_states(rng_from_seed(SEED; offset = 14), hp.n_data_A, params)
    approach_A_eval, _ = zygote_irbc_residual(state_A.model, state_A.ps, state_A.st,
        validation_states, quad_rule; p = params)
    approach_B_eval, _ = zygote_irbc_residual(state_B.model, state_B.ps, state_B.st,
        validation_states, quad_rule; p = params)

    initial_parity = residual_training_parity(model, initial_train_ps, initial_train_st,
        validation_states)
    approach_A_history = loss_history_diagnostics(history_A)
    approach_B_history = loss_history_diagnostics(history_B)
    approach_A_parity = residual_training_parity(state_A.model, state_A.ps, state_A.st, validation_states)
    approach_B_parity = residual_training_parity(state_B.model, state_B.ps, state_B.st, validation_states)

    training_diagnostics = (
        budget = (
            episodes = hp.train_episodes,
            n_data_A = hp.n_data_A,
            n_tracks_B = hp.n_tracks_B,
            n_sim_periods_B = hp.n_sim_periods_B,
            learning_rate = hp.learning_rate,
            max_grad_norm = 10.0,
        ),
        approach_A = merge(approach_A_history, (
            validation_loss = Float64(approach_A_eval.loss),
            residual_parity = approach_A_parity,
        )),
        approach_B = merge(approach_B_history, (
            validation_loss = Float64(approach_B_eval.loss),
            residual_parity = approach_B_parity,
            last_cloud_shape = size(last_cloud_B),
            end_state_shape = size(end_states_B),
        )),
        initial_residual_parity = initial_parity,
        checks = (
            histories_finite = approach_A_history.all_losses_finite &&
                approach_A_history.all_grad_norms_finite &&
                approach_B_history.all_losses_finite &&
                approach_B_history.all_grad_norms_finite,
            initial_residual_parity_ok = parity_ok(initial_parity),
            approach_A_residual_parity_ok = parity_ok(approach_A_parity),
            approach_B_residual_parity_ok = parity_ok(approach_B_parity),
        ),
    )
end

# ╔═╡ 74ca67a5-39c2-6540-d21a-57031d4118db
md"""
### 16. Take-away

What changed between Day 3 nb 01 and this notebook:

| Day 3 nb 01 | This notebook |
|---|---|
| 4 hand-derived analytic helpers (`production_k`, `mpk`, `adj_cost_k`, `adj_cost_kp`) | 0 in the pedagogical residual — replaced by two slot derivatives per country |
| Euler residual written out by hand | Comes out of the autodiff template `partial_2 Pi + beta * E[partial_1 Pi]` |
| 4 sampling/regularisation strategies (A, B, C, D) | Just A and B — focused comparison of sampling alone |

What stayed the same:

- Network architecture (2 × 64 swish + softplus),
- ARC and Fischer–Burmeister (algebraic / KKT, not envelope outputs),
- Quadrature (Gauss–Hermite tensor product, \$Q = 3\$ per dimension),
- Cross-check 1 verifies the autodiff Euler residual *is* the Day 3 hand-derived Euler residual, to machine precision.

The autodiff template generalises trivially: any planner problem whose period contribution can be written as \$\Pi^j(k_{\text{in}}, k_{\text{out}}, z; \lambda, \mu^j)\$ admits Euler residuals of the form \$\partial_2 \Pi^j + \beta\,\mathbb{E}[\partial_1 \Pi^j]\$. Algebraic constraints (ARC) and KKT complementarity (FB) stay hand-coded — autodiff has nothing to add there.

#### Suggested follow-on exercises

1. **Add steady-state pre-training** (Day 3 Approach C) on top of either A or B; recover the Day 3 final loss of \$\sim 10^{-7}\$.
2. **Add ReLoBRaLo** to balance the \$2N+1\$ component losses. Compare against Day 3 Approach D.
3. **Increase \$N\$.** With \$N \ge 4\$ the Gauss–Hermite tensor product (\$Q^{N+1}\$ nodes) becomes expensive; replace it with a monomial cubature or quasi-Monte Carlo — the autodiff template needs no other change, only the quadrature rule moves.
4. **CES production / Epstein–Zin preferences.** Edit the `Pi` primitive (`pi_contribution`) and re-run.

The cell below returns this notebook's machine-checkable diagnostics NamedTuple (steady state, quadrature, Lux boundary, the \$\Pi\$ slot-gradient error, both residual cross-checks, and the Approach A/B training diagnostics).
"""

# ╔═╡ eeeeeeee-0704-4eee-8eee-eeeeeeeeeeee
(
    run_mode = RUN_MODE,
    seed = SEED,
    steady_state = steady_state_checks,
    quadrature = quadrature_checks,
    lux_boundary = lux_boundary_checks,
    pi_slot_gradient_error = slot_gradient_error,
    exogenous_residual_check = residual_cross_check,
    simulation_residual_check = simulation_checks,
    training = training_diagnostics,
)

# ╔═╡ Cell order:
# ╟─11111111-0704-4111-8111-111111111111
# ╟─66f8e120-b474-3ea7-9a32-ce7e2458221f
# ╟─53b3c4b9-723f-35ae-d0d6-4a444744444c
# ╠═22222222-0704-4222-8222-222222222222
# ╠═33333333-0704-4333-8333-333333333333
# ╟─f61d8f4c-de27-96e3-205c-c9e4f1a5a5fd
# ╟─fce66f8f-116d-804c-74fc-a83f699574db
# ╠═44444444-0704-4444-8444-444444444444
# ╟─60ee87b7-befb-9690-f4c2-28e6ecd0573c
# ╠═55555555-0704-4555-8555-555555555555
# ╟─bcf92f44-681a-5de0-5d2c-151bd3477866
# ╠═66666666-0704-4666-8666-666666666666
# ╟─0a579ef5-4aba-28a4-1b57-3e9161e1ec6a
# ╟─2e398ce7-2cd6-8149-7aba-2ae1c7c33487
# ╠═77777777-0704-4777-8777-777777777777
# ╟─b3b8dd2a-3da2-accc-0061-c86ffa02ecb4
# ╠═88888888-0704-4888-8888-888888888888
# ╟─fe61a524-c5e0-afef-d8b5-1adcf6d1c395
# ╠═99999999-0704-4999-8999-999999999999
# ╟─86f5edec-b957-2e66-b75a-33c754c9eed7
# ╠═aaaaaaaa-0704-4aaa-8aaa-aaaaaaaaaaaa
# ╟─5fbc8801-2201-28e8-9e5a-ea76546eda3f
# ╟─d5dd4b46-ae1d-76c2-c6bc-9dbc269e0aad
# ╠═bbbbbbbb-0704-4bbb-8bbb-bbbbbbbbbbbb
# ╟─ee714568-d482-cfb7-b841-32eba8392edd
# ╟─080491ab-789c-f9f8-be15-9b60d36acc8c
# ╠═cccccccc-0704-4ccc-8ccc-cccccccccccc
# ╟─b76a6b94-3bd3-6501-bb9b-7356c111f7ed
# ╠═dddddddd-0704-4ddd-8ddd-dddddddddddd
# ╟─f9e8e4a9-9f85-0f41-a758-8732ebc53475
# ╠═ffffffff-0704-4fff-8fff-ffffffffffff
# ╟─714880c1-47c7-d582-63b6-3a3d4a66f0df
# ╟─50bce4f1-9c12-ffe1-18e4-d950ef96998a
# ╠═12121212-0704-4121-8121-121212121212
# ╟─74ca67a5-39c2-6540-d21a-57031d4118db
# ╠═eeeeeeee-0704-4eee-8eee-eeeeeeeeeeee
