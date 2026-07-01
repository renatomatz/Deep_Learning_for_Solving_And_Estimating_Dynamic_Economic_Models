### A Pluto.jl notebook ###
# v0.20.0

using Markdown
using InteractiveUtils

# ╔═╡ 11111111-1308-4111-8111-111111111111
md"""
# Lecture 13, Notebook 08: Continuous-Time Aiyagari FD and PINN in Lux

This smoke-first translation keeps the finite-difference benchmark as validation
logic and trains a two-network Lux PINN on the stationary HJB/KFE residual pieces.
Smoke mode checks structure only; the saving `L^inf` gate may fail without the
long production run.
"""

# ╔═╡ e1c08b21-dda6-c03d-55b8-39193a93ee28
md"""
## Lecture 13, Notebook 08: Continuous-time Aiyagari — Steady-State PINN (marginal-value form) with an FD benchmark

**Course:** Deep Learning for Solving and Estimating Dynamic Models in Economics and Finance
**Script reference:** Chapter 8, §§8.3–8.5 (HJB, KFE, stationary Aiyagari/Huggett equilibrium) and §8.7 (EMINNs).
**Notebook role:** core
**Author:** Simon Scheidegger and Pavel Ievlev

*Julia/Lux/Pluto preview of* `lectures/lecture_13_continuous_time_ha_numerics/code/lecture_13_08_Aiyagari_Continuous_Time_FD_and_PINN_PyTorch.ipynb`.

> The Python ground truth is a faithful, self-contained port of the JAX `ss/` soft-penalty solver behind the EMINN Krusell–Smith model (Gu, Laurière, Merkel, Payne, 2024, App. B.1), evaluated at its **deterministic steady state**. There is **no finite-difference solve anywhere in training**: the FD code is the validation reference only. This Julia/Lux preview keeps that split — the FD solve validates, it never trains the nets — and runs both nets in `Float64` for the delicate HJB/KFE derivatives (the Python original uses float32 nets against a float64 NumPy FD reference).

> **In-class notebook** (Chapter 8 / Lecture 13 deck Part II, the in-class mesh-free solver). Full training is about 9 min on CPU at `RUN_MODE = "production"`; this preview ships `RUN_MODE = "smoke"` for a fast structural pass that intentionally does **not** clear the saving gate (the aggregate-saving ramp needs a converged policy — see the failure-modes note). Set `RUN_MODE` in the setup cell to `"teaching"` or `"production"` to approach the reference numbers.
"""

# ╔═╡ 74a05e29-39b8-7748-3ad0-b9d1f794eb50
md"""
## A Pedagogical Walkthrough: A Mesh-Free Steady-State PINN, Checked Against Finite Differences

This notebook is designed for teaching. It is intentionally explicit and heavily annotated.

We solve the stationary Aiyagari equilibrium two ways and compare them:

1. **Finite Differences (FD)**, the classic Achdou et al. (2022a) upwind scheme with outer bisection on the interest rate. This is the trusted reference (the "FD gospel"), and nothing it computes ever enters PINN training.
2. **A steady-state PINN**, two small neural nets trained on the HJB and KFE residuals plus a handful of exact integral identities, with **prices and market clearing built in by construction** (no outer bisection).

By the end we compare the marginal value \$W=\partial_a V\$, consumption \$c(a,l)\$, saving \$s(a,l)\$, the stationary density \$g(a,l)\$, and the equilibrium aggregates \$K, r, w\$, and we score the PINN with a single \$L^\infty\$ saving gate against a dense FD policy.
"""

# ╔═╡ 9c9b90ea-2a52-8757-507d-0cc69ef9440f
md"""
## Learning Objectives

- Understand the stationary **HJB + KFE + market-clearing** fixed point, written in **marginal-value form** \$W=\partial_a V\$.
- See how a PINN can make **market clearing hold by construction**: prices are the firm's marginal products at the density net's *own* aggregates, so there is no outer price loop.
- Learn the role of each loss term — the HJB residual, the KFE (in two forms), and the exact integral identities that close the gaps a pointwise residual leaves open.
- Understand the **continuous Gauss–Seidel** gradient gating (two detachments) that decouples the policy net from the distribution net.
- Diagnose solution quality with residuals, moments, and a saving \$L^\infty\$ gate against a dense FD benchmark.
"""

# ╔═╡ 22222222-1308-4222-8222-222222222222
begin
    import Pkg
    Pkg.activate(joinpath(@__DIR__, "..", "..", "..", "julia"))

    using DLEFJulia
    using Lux
    using NNlib
    using Optimisers
    using Statistics
    using Zygote
end

# ╔═╡ 8f70b8bc-0f5a-1e57-536c-a04379078f08
md"""
## Economic Environment and Equilibrium

The EMINN Krusell–Smith model at its **deterministic steady state**: aggregate TFP is pinned at \$z=\bar z\$ (no aggregate shocks), so the economy collapses to the classic Aiyagari/Huggett stationary equilibrium.

**Households.** Hold wealth \$a\in[a_{\min},a_{\max}]=[10^{-6},20]\$, earn labor income \$w\,l\$ with \$l\in\{l_1,l_2\}=\{0.3,1.7\}\$ switching by a symmetric Poisson process (\$\lambda_1=\lambda_2=0.4\$), and choose consumption \$c\$ with CRRA utility \$u(c)=(c^{1-\gamma}-1)/(1-\gamma)\$, \$\gamma=2.1\$, discount rate \$\rho=0.05\$. The budget is \$\dot a = s = w\,l + r\,a - c\$.

**Borrowing constraint (soft form).** Instead of a hard \$a\ge 0\$, the flow utility carries the paper's penalty \$\psi(a)=-\tfrac12\kappa(a-a_{lb})^2\$ for \$a\le a_{lb}\$ (zero above), with \$\kappa=3,\ a_{lb}=1\$. Below \$a_{lb}\$ the penalty's *gradient* \$\psi'(a)=-\kappa(a-a_{lb})>0\$ raises the marginal value of wealth, hence raises saving, so households endogenously avoid the region and the constraint never literally binds.

**Firms / prices.** Cobb–Douglas \$Y=e^z K^\alpha L^{1-\alpha}\$ (\$\alpha=1/3\$, depreciation \$\delta=0.1\$) with competitive factor markets,
\$\$
r = \alpha\,e^z (K/L)^{\alpha-1} - \delta,\qquad w=(1-\alpha)\,e^z (K/L)^\alpha.
\$\$

**Equilibrium** is a triple (policy, distribution, prices) such that the policy is optimal given prices (HJB), the distribution is stationary under that policy (KFE), and prices are the marginal products at the distribution's own aggregates \$K=\sum_j\int a\,g_j\,da\$, \$L=\sum_j l_j\int g_j\,da\$ (market clearing).
"""

# ╔═╡ 9a6b9280-8d27-135e-f6cb-4b28f9aa3cbc
md"""
## 1) Calibration and Numerics

The calibration follows the paper's Table 5 at the deterministic steady state (\$z=\bar z=0\$). The numerical hyperparameters follow the JAX `ss/` recipe: two small nets (width 64, depth 4 at production), a fixed trapezoid quadrature grid for normalization and aggregates, and fresh random collocation each step.

The setup cell below fixes `RUN_MODE = "smoke"`, `SEED = 0`, and `KFE_FORM = "fv"`, then dispatches every grid size and training length off `RUN_MODE` through the `budgets` NamedTuple (the Julia counterpart of the Python `RUN_CFG`). The calibration itself lives in `CTAiyagariParams`, built here with the run-mode grid sizes so the smoke run stays fast.
"""

# ╔═╡ 33333333-1308-4333-8333-333333333333
begin
    RUN_MODE = "smoke"
    SEED = 0
    KFE_FORM = "fv"
    budgets = (
        smoke = (steps = 2, pretrain_steps = 1, n_col = 8, n_quad = 8, n_a = 8,
            a_max = 4.0, fd_outer = 2, fd_inner = 2, lr0 = 1e-5, lr1 = 1e-5,
            hidden = (16, 16), pretrain_abar = 5.0, frac_focus = 0.5,
            a_focus = 4.0, frac_bnd = 0.1, a_bnd_top = 1.0, agg_ramp = (0.0, 1.0)),
        teaching = (steps = 800, pretrain_steps = 100, n_col = 128, n_quad = 120,
            n_a = 120, a_max = 20.0, fd_outer = 12, fd_inner = 80, lr0 = 3e-4,
            lr1 = 1e-6, hidden = (32, 32), pretrain_abar = 5.0, frac_focus = 0.5,
            a_focus = 6.0, frac_bnd = 0.1, a_bnd_top = 2.0, agg_ramp = (0.2, 0.5)),
        production = (steps = 40_000, pretrain_steps = 1_000, n_col = 256,
            n_quad = 400, n_a = 400, a_max = 20.0, fd_outer = 40, fd_inner = 200,
            lr0 = 3e-4, lr1 = 1e-6, hidden = (64, 64, 64, 64),
            pretrain_abar = 5.0, frac_focus = 0.5, a_focus = 6.0, frac_bnd = 0.1,
            a_bnd_top = 2.0, agg_ramp = (0.2, 0.5)),
    )
    hp = run_mode_budget(RUN_MODE; budgets)
    rng = rng_from_seed(SEED)
    params = CTAiyagariParams(n_quad = hp.n_quad, n_a = hp.n_a, a_max = hp.a_max)
end

# ╔═╡ c2738022-aabf-e860-7bf2-2617d6d2dc03
md"""
## 3) Finite-Difference Reference (validation only)

This is the trusted benchmark, the "FD gospel": a standard Achdou et al. (2022a) upwind scheme on a uniform grid, with **outer bisection on the interest rate** \$r\$. Nothing here ever enters PINN training; the FD objects are used only as ground truth for the diagnostics and the saving gate.

- The inner solve: implicit-Euler upwind HJB at fixed prices, then the stationary KFE as the null vector of the generator transpose (one row replaced by the normalization).
- The outer solve: bisect \$r\$ to clear the capital market (invert the firm FOC for \$K_r\$, solve the partial-equilibrium problem, compare with \$K_g=\int a\,g\$).
- The gate reference: the FD **equilibrium** saving policy on the `n_a = n_quad` grid, evaluated at the FD-equilibrium prices — the reference for the \$L^\infty\$ gate. *The full Python notebook* instead re-solves the fixed-price FD HJB on a separate dense grid (`n_dense = 1000`) at those prices, so \$s_{\text{NN}}\$ and \$s_{\text{FD}}\$ share identical prices on a fine mesh that resolves the saving kink a coarse policy would smear; this preview gates against the coarse `n_a`-grid FD saving directly, which coincides with that reference once FD has converged.

The FD reference deliberately uses `n_a = n_quad`, because the FD equilibrium \$K\$ is grid-sensitive and the SS solver bakes its own equilibrium into \$W\$; gating against a coarser FD would measure discretization mismatch, not solver error.

In Julia this is `ct_aiyagari_fd_solve(params; max_outer, inner_max_iter)` from `DLEFJulia`. The same cell also builds the fixed trapezoid quadrature rule and a flat-density normalization/aggregates check (`ct_aiyagari_normalized_density`, `ct_aiyagari_distribution_aggregates`) used later as a sanity anchor for the density net.
"""

# ╔═╡ 44444444-1308-4444-8444-444444444444
begin
    fd = ct_aiyagari_fd_solve(params; max_outer = hp.fd_outer, inner_max_iter = hp.fd_inner)
    quad = ct_aiyagari_trapezoid_rule(params)
    normalized_flat = ct_aiyagari_normalized_density(zeros(length(quad.nodes), 2), quad.weights)
    flat_aggregates = ct_aiyagari_distribution_aggregates(normalized_flat.density, quad.nodes, quad.weights; params)
end

# ╔═╡ 411f6609-f432-9b68-aa9d-9745e56d6313
md"""
## 2) The Equations the Solver Uses

### 2.1 HJB in marginal-value form

The solver never represents \$V\$; it works with \$W=\partial_a V\$ (as the whole repo does). Starting from the stationary HJB
\$\$
\rho V = \max_c\; u(c) + \psi(a) + (w\,l + r\,a - c)\,\partial_a V + \lambda_j\,(V_{\text{other}} - V),
\$\$
the FOC gives \$u'(c)=W\$, i.e. \$c=W^{-1/\gamma}\$ (this is why \$W>0\$ is structural). Differentiating the HJB in \$a\$, the \$\partial c/\partial a\$ terms cancel by the envelope theorem, leaving the **HJB residual**
\$\$
0 = (\rho - r + \lambda_j)\,W - s\,\partial_a W - \lambda_j\,W_{\text{other}} - \psi'(a),\qquad s = w\,l_j + r\,a - W^{-1/\gamma}.
\$\$
All \$z\$-derivative terms and the distribution-coupling term of the full master equation vanish at the deterministic steady state, which is exactly what makes the SS problem self-contained. One extra structural fact used by the strong-form KFE is \$\partial_a s = r - c'(a) = r + (1/\gamma)(c/W)\,\partial_a W\$.

### 2.2 Stationary KFE

For each labor state \$j\$ (with \$\tilde\jmath\$ the other state),
\$\$
0 = -\partial_a(s_j g_j) + \lambda_{\tilde\jmath} g_{\tilde\jmath} - \lambda_j g_j,\qquad\text{zero flux at } a_{\min}, a_{\max}.
\$\$
Two exact integral consequences (both used as losses): **labor-mass balance** \$\lambda_1 M_1 = \lambda_2 M_2\$ with \$M_j=\int g_j\$ (with \$\lambda_1=\lambda_2\$ this forces the 50/50 split, hence \$L=1\$), and the **aggregate-saving identity** \$dK/dt = \int \sum_j s_j g_j\,da = 0\$ (at stationarity the population saves nothing in aggregate).

The code cell below specializes these CRRA/switching equations of Chapter 8 to this calibration: `marginal_value_policy` implements the FOC \$c=W^{-1/\gamma}\$ and the budget \$s = w\,l + r\,a - c\$, `both_marginal_value` / `both_log_density` evaluate \$W\$ and the log-density in one batched Lux forward over both labor states, and `aiyagari_pinn_loss` assembles the HJB and KFE residuals.
"""

# ╔═╡ 013c31a4-1b14-0563-ee2a-646ea5f015e5
md"""
## 4) The Two Neural Nets

Both unknowns are plain MLPs in \$(a_{\text{scaled}}, \text{onehot}(l))\$: 3 inputs, tanh hidden layers, a linear head, Glorot/Xavier init, with \$a\$ affinely scaled to \$[-1,1]\$ (`aiyagari_features`). The two labor states are evaluated in one batched forward — `both_marginal_value` and `both_log_density` stack the two one-hots and return \$(B,2)\$ arrays, the Lux / feature-by-batch analogue of the Python `both(net, a)` (which used `a.repeat(2)`).

### 4.1 The policy side, \$W\$
\$\$ W(a,l_j) = \text{softplus}\big(\text{MLP}(a,l_j)\big). \$\$
Softplus (`NNlib.softplus`, plus a tiny `eps_safe`) enforces \$W>0\$ (marginal utility is positive; \$c=W^{-1/\gamma}\$ requires it) and blocks the *parametrization* form of the \$W\to 0\$ cheat. But \$W\to 0\$ is still an *attractor* the optimizer can approach (\$W\$ small \$\Rightarrow c\$ huge \$\Rightarrow s\$ very negative \$\Rightarrow\$ the transport term washes out the HJB residual); what actually guards against it is the pretrain level (§7) and the shape penalty. The trained net stays well away from the corner, with \$\partial_a W < 0\$ everywhere on a fine grid. The interior derivative \$\partial_a W\$ comes from **`ForwardDiff`** (via `tanh_mlp_scalar_derivatives`, chained with the \$a\$-scaling factor) — the Julia counterpart of the Python `torch.autograd.grad(..., create_graph=True)`.

### 4.2 The distribution side, \$g\$
\$\$ f(a,l_j) = \text{MLP}(a,l_j) - \text{softplus}(\beta_j)\,(a-a_{\min}),\qquad g(a,l_j)=\exp\!\big(f(a,l_j)-\log Z\big). \$\$
Three deliberate choices: **log-space** (positivity for free, and densities spanning 4+ orders of magnitude become \$O(1)\$ quantities); a **trainable tail slope \$\beta_j\$** (the true density decays exponentially, but a bounded tanh-MLP cannot steepen its own tail fast enough where the KFE residual is proportional to the tiny local density, so the decay *rate* is made a single parameter per labor state — here `ps_g.beta`, one per state, initialized at the pretrain prior's rate \$1/\bar a\$ via the softplus inverse); and **normalization by construction** via \$\log Z\$ = logsumexp over (nodes \$\times\$ states) on the fixed trapezoid quadrature (`ct_aiyagari_normalized_density`), so \$\sum_j\int g = 1\$ identically, removing the KFE's scale invariance without a tradeable penalty.

### 4.3 Prices by construction
Every loss evaluation recomputes, from the g-net's current weights, \$f_q\to\log Z\to g_q\to K=\sum a\,w_q\,g,\ L=\sum l\,w_q\,g\to r,w\$ (`detached_prices_from_density`). There is **no stored price variable and no outer bisection**: market clearing holds identically at every step because \$r,w\$ are defined as marginal products at the g-net's own aggregates. The classical nested algorithm (solve a full partial-equilibrium problem per candidate \$r\$, then bisect, as the FD solver does) is replaced by one flat loop in which the distribution shifting *is* the price adjustment. This relies on the Aiyagari fixed point being stable (\$K\$ too high \$\Rightarrow r\$ low \$\Rightarrow\$ saving down \$\Rightarrow\$ mass flows down \$\Rightarrow K\$ falls); an unstable GE feedback would break the joint scheme.
"""

# ╔═╡ afd0684e-661f-d63b-9bd6-03e3534448b8
md"""
## 5) The Joint Loss, Term by Term

`aiyagari_pinn_loss` builds a closure over the quadrature grid. Each call computes:

**HJB residual** (trains the W-net) at `n_col` fresh random collocation points, both labor states. \$\partial_a W\$ is taken by **exact `ForwardDiff`** through `both_marginal_value` (one forward, one scalar derivative per input; the Jacobian is diagonal because each output depends only on its own input, and the derivative is itself differentiable so Zygote can train through it). The policy inside is *live* (the net is asked to satisfy its own FOC-consistent equation); prices are *detached* via `Zygote.dropgrad`. This term is fully mesh-free in both KFE modes.

**Shape penalty** \$L_{\text{shape}}=\text{mean}[\max(\partial_a W,0)^2]\$: the monotonicity prior \$V\$ concave. It costs nothing once satisfied (reads \$0.0\$ after warm-up) but blocks sign-flipped or oscillating \$W\$ during the transient, and is one of the two guards against the \$W\to 0\$ corner.

**KFE, form `"fv"` (default)**: the paper's conservative upwind finite-volume operator (`ct_aiyagari_kfe_drift`) applied to the g-net masses on the quadrature grid. This is **not an FD solve**: no generator is assembled and no linear system is solved; the drift operator is an algebraic map (shifts, clamps, multiplies) from the net's values to the rate at which mass would flow, and the gradient flows through the masses into the g-net. Because it is conservative/upwind, total mass is conserved for any policy, the zero-flux BCs are built in, and the strong form's structural cheats are impossible by construction. The \$\mu/da\$ scaling makes the residual a density-rate, so the loss magnitude is grid-resolution-independent. As a validation-only check, this fv drift operator equals the FD generator transpose \$A^\top\$ to \$\sim 3\times 10^{-10}\$ (the FD solver enters only as the reference, so scrambling it leaves training bit-identical).

**KFE, form `"strong"`**: pure mesh-free, the KFE enforced by exact autodiff at the collocation points, with the policy pair \$(s,\partial_a s)\$ detached (`Zygote.dropgrad`). Here *mesh-free* means precisely that the PDE is enforced by exact autodiff at sampled points; the integral identities may still be evaluated by quadrature of net *values*. The pointwise residual alone is not well posed, so exact identities close the gaps, each killing a failure mode diagnosed in the JAX runs:

1. **Pointwise total-flux identity** \$\sum_j s_j g_j = 0\$ (`flux_loss`). Summing the two KFE equations cancels the jump terms, so the strong residual only forces \$\partial_a(\text{total flux})=0\$; total flux = any nonzero constant has zero residual, the family \$g\sim 1/|s|\$ (a JAX run found exactly this: \$K\$ drifted to ~11, \$r<0\$, near-zero KFE loss). Forcing the flux to \$0\$ pointwise excludes it. In fv form the boundary faces are hard-zero, so this is impossible.
2. **Labor-mass balance** \$\lambda_1 M_1=\lambda_2 M_2\$ (`mass_balance_loss`). An MSE-imperfect pointwise residual does not pin the labor split (a JAX run settled at \$L\approx 0.3\$, a pseudo-equilibrium). In fv form the balance telescopes exactly inside the operator.

Plus **zero-flux endpoint penalties** `boundary_loss` (the BCs the fv operator gets for free) and, in strong mode, an extra **top collocation band** over the employed's \$s\approx 0\$ weak-transport zone, where the transport term \$s\,\partial_a g\$ barely transmits information so the region needs explicit sampling density.

**Aggregate-saving identity** (both forms, ramped) \$L_{\text{agg}}=(\sum_{\text{nodes},j} w_q\,s_j\,g_j)^2\$: the first moment of the stationary KFE (\$dK/dt=0\$). The pointwise KFE MSE plateaus at a floor below which a tail overweight — worth a bias on \$K\$ through the long lever arm at large \$a\$ — hides; the identity is exactly the \$K\$-relevant functional the pointwise residual barely sees, and its gradient \$\propto s_i\$ moves mass from dissaving regions toward savers. It is **ramped** (weight \$0\$ early, linear to full — the Python \$[0.2, 0.5]\cdot\$`n_iter` window, `agg_ramp` here) because switched on from step 0 it acts on the garbage pretrain policy and crushes all mass to the bottom (\$K\$ locks at ~1.2–1.4). By ~20% of training the policy is sane and the identity is purely stabilizing.

\$\$ \text{total} = L_{\text{hjb}} + L_{\text{kfe}} + L_{\text{flux}} + \text{agg\_w}\cdot L_{\text{agg}} + L_{\text{mass}} + L_{\text{bc}} + L_{\text{shape}}, \$\$
with all weights 1 (the ramp factor is the only schedule) and, in fv mode, \$L_{\text{flux}}=L_{\text{mass}}=L_{\text{bc}}=0\$ identically.
"""

# ╔═╡ 63156af4-33b0-d4e4-6521-ad340d1a1429
md"""
## 6) Gradient Gating: the Continuous Gauss–Seidel

There is **one joint loss, one backward pass, one Adam step over both nets** simultaneously — no alternation. The Gauss–Seidel character comes from exactly two detachments, both `Zygote.dropgrad` here (the Julia counterpart of PyTorch `.detach()`):

- \$r,w\$ are **detached** right after computation (inside `detached_prices_from_density`), so \$\partial L_{\text{hjb}}/\partial(\text{g-net})=0\$. The g-net cannot bend aggregates/prices to flatter the HJB residual; the W-net solves "the HJB at the current prices."
- the policy values (\$s_q\$ on the grid; \$s,\partial_a s\$ at collocation in strong mode) are **detached** in every g-side term, so \$\partial(L_{\text{kfe}}+L_{\text{flux}}+L_{\text{agg}}+L_{\text{bc}})/\partial(\text{W-net})=0\$. The W-net cannot bend the policy to flatter the KFE; the g-net solves "the KFE under the current policy."

Each net descends only its own equation given the other's current values — a continuous Gauss–Seidel iteration on the Aiyagari fixed point, which converges because that fixed point is economically stable. The HJB's own policy stays *live*: that is the net's optimality condition, not a coupling channel. (An audit test in the full Python notebook confirmed both cross-gradients are exactly \$0.0\$ at the trained point — the two `dropgrad` detachments zero them by construction.)
"""

# ╔═╡ 55555555-1308-4555-8555-555555555555
begin
    aiyagari_features(a, j, params) = begin
        an = 2 * (a - params.a_min) / (params.a_max - params.a_min) - 1
        j == 1 ? [an, one(an), zero(an)] : [an, zero(an), one(an)]
    end

    function both_raw_derivative(ps_mlp, a_vec; params)
        scale = 2 / (params.a_max - params.a_min)
        raw = [tanh_mlp_scalar_derivatives(ps_mlp, aiyagari_features(a, j, params))[1] for a in a_vec, j in 1:2]
        d_raw = [tanh_mlp_scalar_derivatives(ps_mlp, aiyagari_features(a, j, params))[2][1] * scale for a in a_vec, j in 1:2]
        return raw, d_raw
    end

    function both_marginal_value(ps_w, a_vec; params)
        raw, d_raw = both_raw_derivative(ps_w, a_vec; params)
        W = NNlib.softplus.(raw) .+ params.eps_safe
        dW = NNlib.sigmoid.(raw) .* d_raw
        return W, dW
    end

    function both_log_density(ps_g, a_vec; params)
        raw, d_raw = both_raw_derivative(ps_g.mlp, a_vec; params)
        beta = NNlib.softplus.(ps_g.beta)
        tail = reshape(collect(a_vec) .- params.a_min, :, 1) .* reshape(beta, 1, :)
        return raw .- tail, d_raw .- reshape(beta, 1, :)
    end

    function marginal_value_policy(W, a, r, w; params)
        c = max.(W, params.eps_safe) .^ (-1 / params.gamma)
        s = w .* reshape(params.labor, 1, 2) .+ r .* reshape(a, :, 1) .- c
        return (consumption = c, savings = s)
    end

    function detached_prices_from_density(ps_g, quad; params)
        log_g_q, _ = both_log_density(ps_g, quad.nodes; params)
        normalized = ct_aiyagari_normalized_density(log_g_q, quad.weights)
        aggregates = ct_aiyagari_distribution_aggregates(normalized.density, quad.nodes, quad.weights; params)
        live_prices = ct_aiyagari_prices(aggregates.K, aggregates.L; params)
        prices = (r = Zygote.dropgrad(live_prices.r), w = Zygote.dropgrad(live_prices.w))
        return normalized, aggregates, prices
    end

    function aiyagari_pinn_loss(models, ps, st, a_col; params, kfe_form = :fv, agg_weight = 1.0)
        form = Symbol(kfe_form)
        form in (:fv, :strong) || throw(ArgumentError("kfe_form must be :fv or :strong"))
        a = vec(a_col)
        quad = Zygote.ignore() do
            ct_aiyagari_trapezoid_rule(params)
        end
        normalized, aggregates, prices = detached_prices_from_density(ps.g, quad; params)

        W, dW = both_marginal_value(ps.w, a; params)
        policy = marginal_value_policy(W, a, prices.r, prices.w; params)
        lam = reshape(params.lambda, 1, 2)
        hjb = (params.rho - prices.r .+ lam) .* W .- policy.savings .* dW .-
            lam .* reverse(W; dims = 2) .-
            reshape([ct_aiyagari_soft_penalty_grad(ai; params) for ai in a], :, 1)
        hjb_loss = mean(abs2, hjb)
        shape_loss = mean(abs2, max.(dW, zero(eltype(dW))))

        W_q, _ = both_marginal_value(ps.w, quad.nodes; params)
        policy_q = marginal_value_policy(W_q, quad.nodes, prices.r, prices.w; params)
        s_q = Zygote.dropgrad(policy_q.savings)
        agg_loss = sum(reshape(quad.weights, :, 1) .* s_q .* normalized.density)^2

        if form == :fv
            mu = ct_aiyagari_kfe_drift(normalized.density .* quad.da, s_q, quad.da; params)
            kfe = mu ./ quad.da
            kfe_loss = mean(abs2, kfe)
            flux_loss = zero(kfe_loss)
        else
            log_g, dlog_g = both_log_density(ps.g, a; params)
            g = exp.(log_g .- normalized.logZ)
            s_c = Zygote.dropgrad(policy.savings)
            dsa = Zygote.dropgrad(prices.r .+ (1 / params.gamma) .* (policy.consumption ./ W) .* dW)
            kfe = -(s_c .* g .* dlog_g .+ g .* dsa) .+
                reverse(lam; dims = 2) .* reverse(g; dims = 2) .- lam .* g
            kfe_loss = mean(abs2, kfe)
            flux_loss = mean(abs2, sum(s_c .* g; dims = 2))
        end

        mass_balance_loss = zero(hjb_loss)
        boundary_loss = zero(hjb_loss)
        if form == :strong
            labor_mass = vec(sum(normalized.mass; dims = 1))
            mass_balance_loss = (params.lambda[1] * labor_mass[1] - params.lambda[2] * labor_mass[2])^2
            endpoints = [params.a_min, params.a_max]
            W_end, _ = both_marginal_value(ps.w, endpoints; params)
            policy_end = marginal_value_policy(W_end, endpoints, prices.r, prices.w; params)
            log_g_end, _ = both_log_density(ps.g, endpoints; params)
            density_end = exp.(log_g_end .- normalized.logZ)
            boundary_loss = sum(abs2, Zygote.dropgrad(policy_end.savings) .* density_end)
        end

        total = hjb_loss + kfe_loss + flux_loss + agg_weight * agg_loss +
            mass_balance_loss + boundary_loss + shape_loss
        return (
            loss = total,
            hjb_loss = hjb_loss,
            kfe_loss = kfe_loss,
            flux_loss = flux_loss,
            agg_loss = agg_loss,
            mass_balance_loss = mass_balance_loss,
            boundary_loss = boundary_loss,
            shape_loss = shape_loss,
            hjb = hjb,
            K = aggregates.K,
            L = aggregates.L,
            r = prices.r,
            w = prices.w,
            mass = aggregates.mass,
        ), st
    end
end

# ╔═╡ 9c925b08-b44c-85bb-f5e4-e0f7b6465f6f
md"""
## 7) Training Pipeline

The cell below builds the two nets (`make_mlp(3, hp.hidden, 1; activation = NNlib.tanh)`, `setup_model` with `Float64` parameters and seeded RNGs), initializes the trainable tail slopes `ps_g.beta`, and defines the training helpers.

**Pre-training** (model-implied targets, never FD): \$f\to -a/\bar a\$ (a generic decreasing exponential with prior capital \$K_0\$) and \$W\to u'(w_0 l + r_0 a)\$ (the marginal value of the *zero-saving* policy at the prices implied by the pretrain density's own \$(K_0, L_0)\$). The level matters: the HJB near \$a_{\min}\$ must cancel \$\psi'(a_{\min})\approx \kappa a_{lb}=3\$ against \$(\rho - r + \lambda)W\approx 0.3\,W\$, which needs \$W(a_{\min},l_1)\sim u'(w_0 l_1)\approx 10\$ (so the \$\approx 0.3\times 10\$ level roughly balances the \$\approx 3\$ penalty gradient; the prior capital is \$K_0\approx 4.63\$). Both nets are warm-started to equal quality so neither overfits the other's noise. Pretraining (`pretrain_loss`) is deliberately crude; residual training does all the work. (Its reduction here is a mean over collocation points and both labor states; the Python original sums over the two labor states then means over points — a constant \$2\times\$ larger, immaterial at the zero-loss optimum but a factor on the transient pretrain gradient.)

**Collocation sampling** (`draw_collocation`, fresh every step): a uniform fraction on \$[a_{\min},a_{\max}]\$, a focus fraction on \$[a_{\min},a_{\text{focus}}]\$ (over the constraint and density peak), and in strong mode an extra band near \$a_{\max}\$. The quadrature grid is fixed; sampling never moves it.

**Optimization**: a single Adam (`Optimisers.Adam`) over both nets, learning rate linearly decayed \$lr_0\to lr_1\$ (`learning_rate_for_step`), the agg-identity ramp over the `agg_ramp` window (`agg_weight_for_step`). The Python original decays \$3\times10^{-4}\to 10^{-6}\$ over the full 40k-step production run; the smoke budget here uses tiny steps and grids, so it exercises the pipeline's structure, not its accuracy. The full Python notebook instantiates a *fresh* Adam after pretraining, resetting the moment estimates before residual descent; this preview reuses one optimizer state across both loops, so the pretrain moments carry into the first residual steps.
"""

# ╔═╡ 66666666-1308-4666-8666-666666666666
begin
    w_model = make_mlp(3, hp.hidden, 1; activation = NNlib.tanh)
    g_model = make_mlp(3, hp.hidden, 1; activation = NNlib.tanh)
    ps_w, st_w = setup_model(rng_from_seed(SEED; offset = 1), w_model; parameter_type = Float64)
    ps_g_mlp, st_g_mlp = setup_model(rng_from_seed(SEED; offset = 2), g_model; parameter_type = Float64)
    slope0 = Float64(log(expm1(1 / hp.pretrain_abar)))
    ps_g = (mlp = ps_g_mlp, beta = fill(slope0, 2))
    st_g = (mlp = st_g_mlp,)
    models = (w = w_model, g = g_model)
    train_state = setup_training(models, (w = ps_w, g = ps_g), (w = st_w, g = st_g), Optimisers.Adam(hp.lr0))

    function draw_collocation(rng, n, params, hp; kfe_form = :fv)
        n_focus = min(n, floor(Int, n * hp.frac_focus))
        n_top = Symbol(kfe_form) == :strong ? min(n - n_focus, floor(Int, n * hp.frac_bnd)) : 0
        n_uniform = n - n_focus - n_top
        upper_focus = min(params.a_max, hp.a_focus)
        upper_top = max(params.a_min, params.a_max - hp.a_bnd_top)
        draws(count, lo, hi) = collect(lo .+ (hi - lo) .* rand(rng, count))
        parts = Vector{Vector{Float64}}()
        n_uniform > 0 && push!(parts, draws(n_uniform, params.a_min, params.a_max))
        n_focus > 0 && push!(parts, draws(n_focus, params.a_min, upper_focus))
        n_top > 0 && push!(parts, draws(n_top, upper_top, params.a_max))
        return vcat(parts...)
    end

    function pretrain_context(params, hp)
        q = ct_aiyagari_trapezoid_rule(params)
        dens = exp.(-q.nodes ./ hp.pretrain_abar)
        dens ./= sum(dens .* q.weights)
        K0 = sum(q.nodes .* dens .* q.weights)
        L0 = mean(params.labor)
        p0 = ct_aiyagari_prices(K0, L0; params)
        return (r0 = p0.r, w0 = p0.w, abar = hp.pretrain_abar)
    end

    pretrain = pretrain_context(params, hp)
    function pretrain_loss(models, ps, st, batch)
        a = vec(batch)
        W, _ = both_marginal_value(ps.w, a; params)
        f, _ = both_log_density(ps.g, a; params)
        cash = pretrain.w0 .* reshape(params.labor, 1, 2) .+ pretrain.r0 .* reshape(a, :, 1)
        W_target = max.(cash, params.eps_safe) .^ (-params.gamma)
        f_target = .-reshape(a ./ pretrain.abar, :, 1)
        return mean(abs2, W .- W_target) + mean(abs2, f .- f_target), st
    end

    agg_weight_for_step(step, hp) = begin
        ramp0, ramp1 = hp.agg_ramp
        frac = (step - 1) / max(hp.steps, 1)
        min(1.0, max(0.0, (frac - ramp0) / max(ramp1 - ramp0, 1e-9)))
    end

    learning_rate_for_step(step, hp) = hp.lr0 + (hp.lr1 - hp.lr0) * (step - 1) / max(hp.steps, 1)
    pinn_loss(models, ps, st, batch, agg_weight = 1.0) = begin
        pieces, st_new = aiyagari_pinn_loss(models, ps, st, batch; params, kfe_form = Symbol(KFE_FORM), agg_weight)
        return pieces.loss, st_new
    end
end

# ╔═╡ c6162f57-62e5-8ba7-d1a5-be5ee1b16dcc
md"""
## Running the training, and the Saving \$L^\infty\$ gate

The cell below runs the pipeline: `hp.pretrain_steps` of functional-form pretraining, then `hp.steps` of joint residual descent (adjusting the learning rate and the agg-identity weight each step via `Optimisers.adjust!` and `train_step!`), and finally evaluates the loss pieces and the saving gate.

### 9) The Saving \$L^\infty\$ Gate

The gate is \$\lVert s_{\text{NN}} - s_{\text{FD}}\rVert_\infty\$ on the FD `n_a = n_quad` grid at matched FD-equilibrium prices, with \$s_{\text{NN}} = w_{\text{fd}}\,l + r_{\text{fd}}\,a - c_{\text{NN}}\$ (`ct_aiyagari_saving_linf`). Since \$c_{\text{NN}} = W^{-1/\gamma}\$ does not depend on prices, passing requires *both* the right policy shape and the right equilibrium: if \$r_{\text{nn}}\neq r_{\text{fd}}\$ the W-net solved a different problem and the comparison blows up. The pass threshold is \$\epsilon=10^{-2}\$ (`params.eps_gate`).

A full `production` run clears the gate (the JAX recipe reports `fv` ~3.9e-3, `strong` ~8.2e-3, with \$K_{\text{nn}}\approx 5.07\$ against \$K_{\text{fd}}\approx 5.10\$). The **`smoke` run reports FAIL by design**: the aggregate-saving ramp engages while the policy is still unconverged, so it does not reach the threshold. (If the FD solve itself has not converged in the smoke budget, `saving_gate_status` is reported as a diagnostic rather than pass/fail — the returned \$r_{\text{fd}}, w_{\text{fd}}\$ are the supply-implied prices, which equal the bisection rate only once the market has cleared.)

Resolution matching matters: the FD equilibrium \$K\$ is grid-sensitive (5.27 at 93 points, 5.10 at 400, 5.07 at 800), so the reference deliberately uses `n_a = n_quad`. *The full Python notebook* builds the gate reference by re-solving the fixed-price FD HJB on a separate dense grid (`n_dense = 1000`) at the FD-equilibrium prices, so \$s_{\text{NN}}\$ and \$s_{\text{FD}}\$ share identical prices on a mesh fine enough to resolve the policy kink; this preview compares against the coarse `n_a`-grid FD equilibrium saving directly. Notably, the trained \$K_{\text{nn}}\approx 5.0706\$ lands almost exactly on the dense FD(800) value \$\approx 5.0730\$ — closer to the continuum than its own 400-point reference — because the mesh-free HJB side pulls the equilibrium toward the continuum solution.
"""

# ╔═╡ 77777777-1308-4777-8777-777777777777
begin
    initial_batch = draw_collocation(rng, hp.n_col, params, hp; kfe_form = Symbol(KFE_FORM))
    initial_loss = loss_value(train_state, (m, ps, st, b) -> pinn_loss(m, ps, st, b, 0.0), initial_batch)
    pretrain_history = NamedTuple[]
    for step in 1:hp.pretrain_steps
        local batch = draw_collocation(rng, hp.n_col, params, hp; kfe_form = Symbol(KFE_FORM))
        metrics = train_step!(train_state, pretrain_loss, batch; max_grad_norm = 10.0)
        append_metric!(pretrain_history; step, loss = metrics.loss)
    end
    history = NamedTuple[]
    for step in 1:hp.steps
        Optimisers.adjust!(train_state.opt_state, learning_rate_for_step(step, hp))
        local batch = draw_collocation(rng, hp.n_col, params, hp; kfe_form = Symbol(KFE_FORM))
        local agg_weight = agg_weight_for_step(step, hp)
        metrics = train_step!(train_state, (m, ps, st, b) -> pinn_loss(m, ps, st, b, agg_weight), batch; max_grad_norm = 10.0)
        append_metric!(history; step, loss = metrics.loss, agg_weight = agg_weight)
    end
    eval_batch = collect(range(params.a_min, params.a_max; length = hp.n_col))
    final_pieces, _ = aiyagari_pinn_loss(train_state.model, train_state.ps, train_state.st, eval_batch; params, kfe_form = Symbol(KFE_FORM), agg_weight = 1.0)
    saving_error = ct_aiyagari_saving_linf(train_state.ps.w, fd.a, fd.s, fd.r, fd.w; params)
    saving_gate_status = fd.converged ? (saving_error < params.eps_gate ? :pass : :fail) : :diagnostic_fd_not_converged
end

# ╔═╡ f34615bb-9d94-0be2-29c9-9cf6b62a85c5
md"""
## 8) Diagnostics: FD vs PINN

The full Python notebook renders two dashboards here: it overlays the PINN solution on the FD reference for the marginal value \$W\$, consumption \$c\$, saving \$s\$ (at the matched FD-equilibrium prices, the gate convention), and the stationary density \$g\$, and it plots the joint loss, the loss-component breakdown, the running \$L^\infty\$ saving gate and the \$K\$ trajectory. The PINN is never trained on these FD arrays; they are ground truth for comparison only.

This compact Julia/Lux preview does not draw those Matplotlib figures. Instead it returns the same objects **numerically** in the machine-checkable summary at the very bottom — the FD-vs-PINN aggregates (\$K\$, \$r\$, \$w\$), each loss component, the density mass check, and the saving \$L^\infty\$ gate — so the notebook reads and validates standalone.
"""

# ╔═╡ 94b6147c-b3ae-f2a0-8bf5-87fe7e8f34e2
md"""
### Reference Results (CPU, seed 0 unless noted)

These are the numbers a full `production` run reproduces in the ground-truth solver; the `smoke` run here does not reach them.

| run | gate \$\lVert s-s_{FD}\rVert_\infty\$ | \$K_{\text{nn}}\$ | \$r_{\text{nn}}\$ | reference |
|---|---|---|---|---|
| fv (default) | **3.9e-3 PASS** | 5.0706 | 0.01295 | \$K_{fd}=5.0994,\ r_{fd}=0.01251\$ |
| strong | **8.2e-3 PASS** | 5.0384 | 0.01342 | noisier late phase (see Failure Modes) |
| fv, prior \$\bar a=2.5\$ | 4.9e-3 PASS | 5.1115 | 0.01253 | prior-independence check |
| fv, seed 1 | 6.6e-3 PASS | 5.0523 | 0.01314 | matches the JAX seed spread |

The strong form's gate oscillates 4e-3 ↔ 1.2e-2 over the last ~15k steps (briefly failing at intermediate checkpoints, final 8.2e-3), the price of losing the FV operator's built-in conservation.
"""

# ╔═╡ f1bf2934-677d-c487-d503-18de3e684dcb
md"""
## Failure Modes (all observed, all understood)

- **K-crush by the agg identity** (both forms): engage the identity on a garbage policy and it drains all mass to the bottom, \$K\to\$ ~1.2–1.4, and the run never recovers. Cure: the ramp. Any smoke run short enough that the ramp hits an unconverged policy will fail the gate, by design.
- **Const-flux cheat family** (strong only): \$g\sim 1/|s|\$, zero pointwise residual, wrong answer. Cure: the pointwise total-flux identity. Impossible in fv.
- **Labor-split drift** (strong only): pointwise MSE tolerates a wrong \$M_1/M_2\$; the equilibrium drifts through \$L\$. Cure: the mass-balance identity. Impossible in fv.
- **Tail overweight / K bias** (both): hides below the KFE MSE floor. Cure: the agg identity plus the trainable tail slope.
- **\$W\to 0\$ attractor**: approached (never reached, softplus) if the W-net's level near the constraint is initialized far too low. Cure: the pretrain level plus the shape penalty.
- **Strong-form late-phase noise**: the gate metric oscillates over the last ~15k steps, the price of losing the FV operator's built-in conservation.
"""

# ╔═╡ ed664345-4eff-c428-36cf-d4c83483c6fa
md"""
## What Is and Isn't Ported; Scope

**Ported** (behavior-faithful to the JAX `ss/` solver — same calibration, recipe, ramp, targets, and identities): the soft-penalty **baseline** (fv finite-volume KFE) and the **variant** (strong mesh-free KFE).

**Not ported** from `ss/`: the hard-constraint variant (atoms of mass at \$a_{\min}\$, the boundary-singularity exponent, the weak/CDF-form KFE) and the EMA/Polyak stabilizers. The deterministic steady state collapses the EMINN master equation (the \$z\$-derivative and distribution-coupling terms all vanish), which is exactly what makes this self-contained; aggregate shocks reintroduce them and call for the surrogate-density approach of §8.7 (EMINNs).
"""

# ╔═╡ 63ced688-d500-4182-5a35-935362583be5
md"""
## Economic Discussion (for Teaching)

### What should students see?

1. **The FD benchmark is the trusted reference.** Standard upwind discretization with outer bisection on \$r\$; the HJB and KFE residuals are tiny and the GE fixed point converges quickly.
2. **The PINN recovers the same economic objects without an outer price loop.** Market clearing holds by construction because prices are the firm's marginal products at the density net's own aggregates. The distribution shifting *is* the price adjustment.
3. **Structure is what makes the mesh-free PINN work.** Functional-form pretraining (level at the constraint, zero-saving start), the trainable tail slope, normalization by construction, and the ramped aggregate-saving identity together pin the parts of the solution a raw pointwise residual cannot see.

### Economic interpretation

- The soft borrowing penalty makes households endogenously avoid low wealth, so mass concentrates above \$a_{lb}\$ without a hard constraint ever binding.
- The employed state \$l_2\$ saves more and populates the right tail more heavily; the tail carries a long lever arm into aggregate \$K\$.
- The equilibrium interest rate reflects the intersection of precautionary-savings supply and firm capital demand, recovered here by the joint net rather than by bisection.
"""

# ╔═╡ 314c17f1-d9ab-8ac4-60e2-d031fd126e5d
md"""
## Suggested Classroom Use

1. Run the notebook once at `RUN_MODE = "smoke"` to see the full pipeline quickly (the gate will FAIL by design), then at `"teaching"` or `"production"` to approach the reference numbers.
2. Flip `KFE_FORM` from `"fv"` to `"strong"` and watch which loss terms switch on (`flux_loss`, `mass_balance_loss`, `boundary_loss`) and how the late-phase gate gets noisier.
3. Change one parameter — the switching rate \$\lambda\$, the soft-penalty \$\kappa\$, or the agg-identity ramp window `agg_ramp` — and compare how convergence, policies, and the density tail move.
"""

# ╔═╡ 80dec142-560a-a231-0952-8358f69fd75f
md"""
## Takeaway

On the deterministic steady-state Aiyagari calibration, two small nets — the marginal value \$W=\partial_a V\$ and a normalized log-density \$g\$ — reproduce the FD reference for the policy, the distribution, and the equilibrium aggregates, with market clearing built in by construction and **no FD solve anywhere in training**. The continuous Gauss–Seidel gating (two detachments) decouples the policy and distribution problems, and a handful of exact integral identities pin the \$K\$-relevant content the pointwise residuals miss. For aggregate-shock extensions, the same recipe carries over by replacing \$g\$ with a finite-dimensional surrogate (§8.7, EMINNs; Gu et al., 2024).

The cell below returns the machine-checkable diagnostics summary for this notebook's smoke run.
"""

# ╔═╡ 88888888-1308-4888-8888-888888888888
(
    RUN_MODE = RUN_MODE,
    KFE_FORM = KFE_FORM,
    detach_semantics = "prices detached for HJB; W-policy detached inside KFE and aggregate identity",
    initial_loss = initial_loss,
    pretrain_final_loss = pretrain_history[end].loss,
    final_loss = history[end].loss,
    eval_loss = final_pieces.loss,
    hjb_loss = final_pieces.hjb_loss,
    kfe_loss = final_pieces.kfe_loss,
    flux_loss = final_pieces.flux_loss,
    agg_loss = final_pieces.agg_loss,
    mass_balance_loss = final_pieces.mass_balance_loss,
    boundary_loss = final_pieces.boundary_loss,
    shape_loss = final_pieces.shape_loss,
    mass = final_pieces.mass,
    K_pinn = final_pieces.K,
    K_fd = fd.K,
    fd_market_gap = fd.market_gap,
    fd_converged = fd.converged,
    r_pinn = final_pieces.r,
    r_fd = fd.r,
    w_pinn = final_pieces.w,
    w_fd = fd.w,
    flat_density_mass = flat_aggregates.mass,
    saving_linf_vs_fd = saving_error,
    saving_gate_status = saving_gate_status,
    finite_loss_check = isfinite(final_pieces.loss),
    density_normalized_check = abs(final_pieces.mass - 1) < 1e-8,
)

# ╔═╡ Cell order:
# ╟─11111111-1308-4111-8111-111111111111
# ╟─e1c08b21-dda6-c03d-55b8-39193a93ee28
# ╟─74a05e29-39b8-7748-3ad0-b9d1f794eb50
# ╟─9c9b90ea-2a52-8757-507d-0cc69ef9440f
# ╠═22222222-1308-4222-8222-222222222222
# ╟─8f70b8bc-0f5a-1e57-536c-a04379078f08
# ╟─9a6b9280-8d27-135e-f6cb-4b28f9aa3cbc
# ╠═33333333-1308-4333-8333-333333333333
# ╟─411f6609-f432-9b68-aa9d-9745e56d6313
# ╟─c2738022-aabf-e860-7bf2-2617d6d2dc03
# ╠═44444444-1308-4444-8444-444444444444
# ╟─013c31a4-1b14-0563-ee2a-646ea5f015e5
# ╟─afd0684e-661f-d63b-9bd6-03e3534448b8
# ╟─63156af4-33b0-d4e4-6521-ad340d1a1429
# ╠═55555555-1308-4555-8555-555555555555
# ╟─9c925b08-b44c-85bb-f5e4-e0f7b6465f6f
# ╠═66666666-1308-4666-8666-666666666666
# ╟─f34615bb-9d94-0be2-29c9-9cf6b62a85c5
# ╟─c6162f57-62e5-8ba7-d1a5-be5ee1b16dcc
# ╠═77777777-1308-4777-8777-777777777777
# ╟─94b6147c-b3ae-f2a0-8bf5-87fe7e8f34e2
# ╟─f1bf2934-677d-c487-d503-18de3e684dcb
# ╟─ed664345-4eff-c428-36cf-d4c83483c6fa
# ╟─63ced688-d500-4182-5a35-935362583be5
# ╟─314c17f1-d9ab-8ac4-60e2-d031fd126e5d
# ╟─80dec142-560a-a231-0952-8358f69fd75f
# ╠═88888888-1308-4888-8888-888888888888
