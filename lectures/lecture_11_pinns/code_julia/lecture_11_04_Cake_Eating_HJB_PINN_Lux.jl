### A Pluto.jl notebook ###
# v0.20.0

using Markdown
using InteractiveUtils

# ╔═╡ 11111111-1104-4111-8111-111111111111
md"""
# Lecture 11, Notebook 04: Cake-Eating HJB PINN in Lux

The trial value function satisfies endpoint values exactly. The HJB residual is
computed from a Lux tanh MLP with explicit first input derivatives so parameter
gradients remain available to `Zygote`.


This Julia smoke translation uses Adam only. The deterministic L-BFGS polish from the Python notebook is deferred until the Julia track adopts a narrow L-BFGS dependency.
"""

# ╔═╡ 4a225698-bccd-7149-3e30-2b6b82f045e0
md"""
## Lecture 11, Notebook 04: The cake-eating HJB equation with a hard-BC PINN

**Course:** Deep Learning for Solving and Estimating Dynamic Models in Economics and Finance
**Script reference:** §7.6 (the HJB equation and the cake-eating problem)
**Notebook role:** core
**Author:** Simon Scheidegger

*Julia/Lux/Pluto preview of* `lectures/lecture_11_pinns/code/lecture_11_04_Cake_Eating_HJB_PINN.ipynb`.

> **Run mode.** The checked-in run uses `RUN_MODE = "smoke"` for fast execution; the accuracy figures quoted in the slides and the companion script use the longer `teaching` / `production` budgets. Set `RUN_MODE` in the next cell accordingly to reproduce them.

> **In-class notebook** (Day 6, Block 1 — PINNs Foundations & Economic Applications, 75 min)
"""

# ╔═╡ c9ca1422-0415-4e6f-80b6-0cbca6014cd4
md"""
## Solving the Cake-Eating HJB Equation with PINNs

In this notebook we solve the **continuous-time consumption-savings (cake-eating) problem** using a Physics-Informed Neural Network (PINN) built as a **hard-boundary trial solution** with a compact MLP.

The cake-eating problem is one of the simplest dynamic optimization problems in economics: an agent holds a stock of wealth (the "cake") that earns interest at rate \$r\$ and must choose how much to consume at every instant in time so as to maximise lifetime discounted utility. In continuous time the optimality condition takes the form of a **Hamilton-Jacobi-Bellman (HJB)** ordinary differential equation for the value function \$V(a)\$.

This exercise connects directly to the **Deep Equilibrium Nets (DEQNs)** approach introduced earlier in the course (Chapter 2, DEQNs): there we parameterised policy and value functions with neural networks and trained them to satisfy equilibrium conditions. Here we do the same thing, but the equilibrium condition is a PDE (or, in this stationary case, an ODE) that we enforce through the PINN residual loss.

We will:
1. State the HJB equation and derive its analytical solution.
2. Build a scaled trial-solution network to approximate \$V(a)\$ while satisfying endpoint values exactly.
3. Train the network by minimising the HJB residual on collocation points.
4. Compare the learned value and policy functions against the closed-form solution.
"""

# ╔═╡ 22222222-1104-4222-8222-222222222222
begin
    import Pkg
    Pkg.activate(joinpath(@__DIR__, "..", "..", "..", "julia"))

    using DLEFJulia
    using Lux
    using NNlib
    using Optimisers
    using Statistics
end

# ╔═╡ b89a51b2-4a3e-43de-358c-8ba82e41d78f
md"""
### The Economic Model

An infinitely-lived agent maximises
\$\$\max_{\{c_t\}} \int_0^\infty e^{-\rho t}\,u(c_t)\,dt, \qquad u(c)=\frac{c^{1-\gamma}}{1-\gamma},\$\$
subject to the wealth ("cake") evolution
\$\$\dot a_t = r\,a_t - c_t, \qquad a_0 \text{ given}.\$\$

**HJB equation.** In the stationary formulation the value function \$V(a)\$ satisfies
\$\$\rho\,V(a) = \max_{c}\left\{\frac{c^{1-\gamma}}{1-\gamma} + V'(a)\,(r\,a - c)\right\}.\$\$

**First-order condition.** Differentiating the right-hand side with respect to \$c\$ gives
\$\$c^{-\gamma} = V'(a) \quad\Longrightarrow\quad c^* = \bigl(V'(a)\bigr)^{-1/\gamma}.\$\$

**Analytical solution.** With CRRA utility one can verify by substitution that
\$\$V^*(a) = \frac{\kappa^{-\gamma}}{1-\gamma}\,a^{1-\gamma}, \qquad c^*(a)=\kappa\,a,\$\$
where the marginal propensity to consume is
\$\$\kappa = \frac{\rho - (1-\gamma)\,r}{\gamma}.\$\$

**Parameters.** We set \$\gamma=2\$, \$\rho=0.05\$, \$r=0.03\$, giving \$\kappa=0.04\$. In the Julia preview these constants live in `CakeEatingParams()`, dispatched alongside the run-mode budget.
"""

# ╔═╡ 33333333-1104-4333-8333-333333333333
begin
    RUN_MODE = "smoke"
    SEED = 0
    budgets = (
        smoke = (steps = 3, batch_size = 6, lr = 0.002),
        teaching = (steps = 800, batch_size = 128, lr = 0.002),
        production = (steps = 8_000, batch_size = 512, lr = 0.001),
    )
    hp = run_mode_budget(RUN_MODE; budgets)
    rng = rng_from_seed(SEED)
    params = CakeEatingParams()
end

# ╔═╡ 9fe2dde2-25d1-467a-3ab4-f7f6737fcd92
md"""
### Scaled Trial Solution with Hard Boundary Conditions

For this one-dimensional HJB, a full DGM architecture is more machinery than we need. Here we use a smaller, scaled MLP inside a **hard-boundary trial solution**:

\$\$
\hat V(a) = V^*(a_{\min}) + x(a)\{V^*(a_{\max})-V^*(a_{\min})\} + x(a)(1-x(a))S f_\theta(x(a)),
\$\$
where \$x(a)=(a-a_{\min})/(a_{\max}-a_{\min})\$ and \$S\$ is a value-function scale. The endpoints are exact by construction, so the optimizer can focus on the HJB residual. Note that the anchor uses the *closed-form* value \$V^*\$ at \$a_{\min}\$ and \$a_{\max}\$: this is a known-answer benchmark, and the point is to isolate the PINN residual mechanics. In a model without a closed-form solution the endpoint anchors must instead come from asymptotic conditions, state constraints, or a coarse numerical solve on a few boundary points.

In Lux the free network \$f_\theta\$ is `make_mlp(1, (16, 16), 1; activation = tanh)`, and `DLEFJulia`'s `cake_eating_trial_value_derivative` evaluates the trial solution \$\hat V(a)\$ together with its derivative \$\hat V'(a)\$ — the input derivative comes from `ForwardDiff`. The closed-form references \$V^*\$ and \$c^*\$ are `cake_eating_value_exact` and `cake_eating_consumption_exact`. The Python notebook warm-starts with Adam and then applies a deterministic double-precision L-BFGS polish; this Julia smoke preview trains with Adam only (via `Optimisers.jl`) and defers the L-BFGS step.
"""

# ╔═╡ 1c52370f-d7f0-ec78-12b5-50c2189cbd72
md"""
### PDE Residual

We enforce the HJB equation by penalising its residual at randomly sampled collocation points. After substituting the FOC \$c^* = (V'(a))^{-1/\gamma}\$ back into the HJB we obtain the residual
\$\$\mathcal{R}(a) = \rho\,V(a) - \left[\frac{(c^*)^{1-\gamma}}{1-\gamma} + V'(a)\,(r\,a - c^*)\right].\$\$

A practical issue is that \$V'(a)\$ must be **strictly positive** to evaluate \$c^* = (V'(a))^{-1/\gamma}\$. Early in training the network output is essentially random and \$V'(a)\$ can easily become negative. We apply a **softplus safeguard**:
\$\$\widetilde{V'} = \mathrm{softplus}(V'(a)) + \varepsilon\$\$
to guarantee positivity without destroying gradients. In Lux this residual and safeguard live in `cake_eating_hjb_loss`; parameter gradients of the mean-squared residual flow through `Zygote`, while `ForwardDiff` supplies the input derivative \$V'(a)\$. Double precision (`Float64`) is used throughout because the FOC inversion is sensitive to numerical error.
"""

# ╔═╡ 44444444-1104-4444-8444-444444444444
begin
    sample_assets(rng, n) = reshape(params.a_min .+ (params.a_max - params.a_min) .* rand(rng, n), 1, :)
    model = make_mlp(1, (16, 16), 1; activation = NNlib.tanh)
    train_state = setup_training(rng_from_seed(SEED; offset = 1), model, Optimisers.Adam(hp.lr); parameter_type = Float64)

    hjb_loss(model, ps, st, batch) = begin
        pieces, st_new = cake_eating_hjb_loss(model, ps, st, batch; params)
        return pieces.loss, st_new
    end
end

# ╔═╡ ff2292b9-4a17-2a09-f936-69c93318640a
md"""
### Training

Because the trial solution satisfies the two value boundaries exactly, the training loss is just the interior HJB residual. Each step resamples collocation points with `sample_assets` and takes an Adam step through `train_step!`, recording the loss history. The Python notebook finishes with an L-BFGS polish evaluated on a *fixed* grid, so the quasi-Newton line search sees a deterministic double-precision objective; the Julia smoke run trains with Adam only and defers that polish.
"""

# ╔═╡ 55555555-1104-4555-8555-555555555555
begin
    initial_batch = sample_assets(rng, hp.batch_size)
    initial_loss = loss_value(train_state, hjb_loss, initial_batch)
    history = NamedTuple[]
    for step in 1:hp.steps
        local batch = sample_assets(rng, hp.batch_size)
        metrics = train_step!(train_state, hjb_loss, batch; max_grad_norm = 50.0)
        append_metric!(history; step, loss = metrics.loss)
    end
end

# ╔═╡ d531b595-a3ec-818a-9400-8f09ba197c3b
md"""
### Results: value function and consumption policy

We evaluate the trained trial solution on a fine asset grid and compare it with the closed-form benchmark on two fronts. First the value function \$\hat V(a)\$ against \$V^*(a)\$; then the consumption policy recovered from the FOC, \$\hat c(a) = \bigl(\widetilde{V'}(a)\bigr)^{-1/\gamma}\$, against \$c^*(a)=\kappa\,a\$. The Python notebook plots these in two separate result sections; the Julia preview computes both in a single cell and reports their relative \$L_2\$ errors via `relative_l2_error`.
"""

# ╔═╡ 66666666-1104-4666-8666-666666666666
begin
    a_eval = collect(range(params.a_min, params.a_max; length = 60))
    V_hat = [cake_eating_trial_value_derivative(train_state.ps, a; params)[1] for a in a_eval]
    V_exact = [cake_eating_value_exact(a; params) for a in a_eval]
    c_hat = [NNlib.softplus(cake_eating_trial_value_derivative(train_state.ps, a; params)[2])^(-1 / params.gamma) for a in a_eval]
    c_exact = [cake_eating_consumption_exact(a; params) for a in a_eval]
    final_pieces, _ = cake_eating_hjb_loss(train_state.model, train_state.ps, train_state.st, reshape(a_eval, 1, :); params)
end

# ╔═╡ 0f5da72a-9138-c2e1-3834-8c8f35e0a25c
md"""
### Takeaway and Discussion

The hard-boundary trial-solution PINN accurately recovers both the value function \$V(a)\$ and the consumption policy \$c(a)\$ for the cake-eating problem. Key observations:

* **The PDE residual serves as an unsupervised loss.** We only supplied boundary values; the network learned the interior solution by satisfying the HJB equation.
* **The softplus safeguard is essential.** Without it, early iterates produce negative \$V'(a)\$ and the FOC inversion \$c=(V')^{-1/\gamma}\$ yields NaN. Softplus is smooth and gradient-friendly, making it preferable to hard clipping.
* **DGM is useful, but not needed here.** For this one-dimensional benchmark, the compact trial-solution MLP is easier to train because the endpoints are exact and no soft boundary penalty competes with the interior HJB residual. DGM remains useful as an optional heavier architecture for higher-dimensional or sharper PDEs.

#### Extensions

The same methodology extends naturally to richer economic models:

* **Income risk (Merton problem):** Add a diffusion term \$\sigma\,a\,dW_t\$ to the wealth dynamics. The HJB becomes a second-order PDE and requires an additional \$V''(a)\$ term in the residual.
* **Aiyagari / Huggett models:** Solve the stationary distribution jointly with the HJB by adding a Kolmogorov Forward Equation (KFE) loss.
* **General equilibrium:** Prices (e.g. the interest rate \$r\$) become endogenous and must clear markets. A PINN can enforce both the HJB and the market-clearing condition simultaneously, scaling to problems where traditional grid methods are infeasible.

The cell below returns the machine-checkable diagnostics summary for this notebook's smoke run.
"""

# ╔═╡ 77777777-1104-4777-8777-777777777777
(
    initial_loss = initial_loss,
    final_loss = history[end].loss,
    eval_hjb_loss = final_pieces.loss,
    value_relative_l2 = relative_l2_error(V_hat, V_exact),
    consumption_relative_l2 = relative_l2_error(c_hat, c_exact),
    boundary_low = cake_eating_trial_value_derivative(train_state.ps, params.a_min; params)[1],
    boundary_high = cake_eating_trial_value_derivative(train_state.ps, params.a_max; params)[1],
)

# ╔═╡ Cell order:
# ╟─11111111-1104-4111-8111-111111111111
# ╟─4a225698-bccd-7149-3e30-2b6b82f045e0
# ╟─c9ca1422-0415-4e6f-80b6-0cbca6014cd4
# ╠═22222222-1104-4222-8222-222222222222
# ╟─b89a51b2-4a3e-43de-358c-8ba82e41d78f
# ╠═33333333-1104-4333-8333-333333333333
# ╟─9fe2dde2-25d1-467a-3ab4-f7f6737fcd92
# ╟─1c52370f-d7f0-ec78-12b5-50c2189cbd72
# ╠═44444444-1104-4444-8444-444444444444
# ╟─ff2292b9-4a17-2a09-f936-69c93318640a
# ╠═55555555-1104-4555-8555-555555555555
# ╟─d531b595-a3ec-818a-9400-8f09ba197c3b
# ╠═66666666-1104-4666-8666-666666666666
# ╟─0f5da72a-9138-c2e1-3834-8c8f35e0a25c
# ╠═77777777-1104-4777-8777-777777777777
