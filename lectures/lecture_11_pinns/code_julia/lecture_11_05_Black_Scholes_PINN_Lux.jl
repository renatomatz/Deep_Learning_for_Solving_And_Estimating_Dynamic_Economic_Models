### A Pluto.jl notebook ###
# v0.20.0

using Markdown
using InteractiveUtils

# ╔═╡ 11111111-1105-4111-8111-111111111111
md"""
# Lecture 11, Notebook 05: Black-Scholes PINN in Lux

A Lux tanh MLP maps normalized `(S,t)` features to a European call value. The
PINN loss combines the Black-Scholes PDE, the zero-spot boundary, the terminal
payoff, and the high-spot asymptotic boundary.


This Julia smoke translation uses Adam only. The deterministic L-BFGS polish from the Python notebook is deferred until the Julia track adopts a narrow L-BFGS dependency.
"""

# ╔═╡ 34e41429-1c58-f8bc-a966-0e0a6a84361c
md"""
## Lecture 11, Notebook 05: The Black-Scholes PDE with a PINN

**Course:** Deep Learning for Solving and Estimating Dynamic Models in Economics and Finance
**Script reference:** §7.8 (the Black-Scholes PDE)
**Notebook role:** core
**Author:** Simon Scheidegger

*Julia/Lux/Pluto preview of* `lectures/lecture_11_pinns/code/lecture_11_05_Black_Scholes_PINN.ipynb`.

> **Run mode.** The checked-in run uses `RUN_MODE = "smoke"` for fast execution; the accuracy figures quoted in the slides and the companion script use the longer `teaching` / `production` budgets. Set `RUN_MODE` in the next cell accordingly to reproduce them.

> **Self-study notebook** — This notebook complements the in-class PINNs session (Day 6, Block 1). Work through it at your own pace.

> **Backend note.** This Julia preview is an Adam-only smoke translation: input derivatives (\$V_t\$, \$V_S\$, \$V_{SS}\$) come from `ForwardDiff`, parameter gradients from `Zygote` via `Optimisers.jl`, and the deterministic L-BFGS polish from the Python notebook is deferred until the Julia track adopts a narrow L-BFGS dependency.
"""

# ╔═╡ 443f02a1-9615-199b-4859-460ffe5a06c5
md"""
## Solving the Black-Scholes PDE with Physics-Informed Neural Networks

This notebook demonstrates how to solve the **Black-Scholes partial differential equation** for European call option pricing using a **Physics-Informed Neural Network (PINN)**.

### Relevance to Central Banking

Option pricing models are a cornerstone of modern financial risk management. Central banks routinely monitor option-implied measures — such as implied volatilities, risk-neutral densities, and the Greeks — to assess market expectations and gauge financial stability. The Black-Scholes model, while stylized, provides the canonical framework on which more realistic models are built.

PINNs offer a mesh-free, differentiable alternative to traditional finite-difference solvers for pricing PDEs. Because PINNs leverage automatic differentiation, they yield the option price **and** its sensitivities (the Greeks) simultaneously, without any additional numerical effort. This makes them attractive for stress-testing and scenario analysis in supervisory contexts.

### The Black-Scholes PDE

For a European call option with value \$V(S,t)\$, the PDE reads:

\$\$\frac{\partial V}{\partial t} + \frac{1}{2}\sigma^2 S^2 \frac{\partial^2 V}{\partial S^2} + r S \frac{\partial V}{\partial S} - r V = 0,\$\$

subject to:
- **Terminal condition:** \$V(S, T) = \max(S - K, 0)\$
- **Boundary at \$S=0\$:** \$V(0, t) = 0\$
- **Boundary at \$S=S_{\max}\$:** \$V(S_{\max}, t) = S_{\max} - K e^{-r(T-t)}\$

The Julia preview collects these parameters in `BlackScholesParams()` (volatility \$\sigma\$, rate \$r\$, strike \$K\$, maturity \$T\$, and the truncation \$S_{\max}\$) and threads \$(S,t)\$ features through a Lux MLP.
"""

# ╔═╡ 22222222-1105-4222-8222-222222222222
begin
    import Pkg
    Pkg.activate(joinpath(@__DIR__, "..", "..", "..", "julia"))

    using DLEFJulia
    using Lux
    using NNlib
    using Optimisers
    using Statistics
end

# ╔═╡ 33333333-1105-4333-8333-333333333333
begin
    RUN_MODE = "smoke"
    SEED = 0
    budgets = (
        smoke = (steps = 2, n_interior = 6, n_bc = 3, n_terminal = 5, lr = 0.001),
        teaching = (steps = 800, n_interior = 256, n_bc = 64, n_terminal = 128, lr = 0.001),
        production = (steps = 8_000, n_interior = 1_024, n_bc = 256, n_terminal = 512, lr = 0.001),
    )
    hp = run_mode_budget(RUN_MODE; budgets)
    rng = rng_from_seed(SEED)
    params = BlackScholesParams()
end

# ╔═╡ d036b363-f424-ba8c-226e-970d419a1005
md"""
### PINN Architecture

The network takes a two-dimensional input \$(S, t)\$ and outputs the option value \$V(S, t)\$. The Python notebook uses a PyTorch `nn.Module` with three hidden layers of 50 neurons each and \$\tanh\$ activations; the Julia preview builds the analogous model with `make_mlp(2, (16, 16), 1; activation = NNlib.tanh)` — a smaller network for this smoke-sized preview — and evaluates it with the explicit `y, st = model(x, ps, st)` Lux pattern. Smooth \$\tanh\$ activations matter because the PDE loss requires **second-order** derivatives of the network output.
"""

# ╔═╡ 23da76e6-05ca-5b42-8bfa-a0834d553a6a
md"""
### PDE Residual via Automatic Differentiation

The key idea behind PINNs is to enforce the PDE in its strong form at a set of collocation points. We need the partial derivatives \$V_t\$, \$V_S\$, and \$V_{SS}\$ of the network output. Where the Python notebook calls `torch.autograd.grad(..., create_graph=True)`, the Julia preview differentiates the network with respect to its \$(S,t)\$ inputs using `ForwardDiff` (inside `black_scholes_value_derivatives`), while the parameter gradients used for training flow through `Zygote`. The PDE residual

\$\$V_t + \tfrac{1}{2}\sigma^2 S^2 V_{SS} + r S V_S - r V\$\$

is driven toward zero during training.
"""

# ╔═╡ 8f0212f4-b975-d52b-9884-d433b33a35ab
md"""
### Sampling Collocation Points

We draw random collocation points in the interior of the domain \$[0, S_{\max}] \times [0, T]\$ as well as on each boundary (the terminal condition, the \$S=0\$ boundary, and the \$S=S_{\max}\$ boundary). Points are resampled every step so the network does not overfit to a fixed grid. In the Julia preview `sample_bs_batch` returns a NamedTuple holding the interior points and the three boundary sets, and `black_scholes_loss` combines the scaled PDE residual with the boundary and terminal penalties. Architecture, residual, and sampling all live in the single setup cell below.
"""

# ╔═╡ 44444444-1105-4444-8444-444444444444
begin
    function sample_bs_batch(rng, hp, params)
        return (
            S_int = params.s_max .* rand(rng, hp.n_interior),
            t_int = params.maturity .* rand(rng, hp.n_interior),
            S_bc0 = zeros(hp.n_bc),
            t_bc0 = params.maturity .* rand(rng, hp.n_bc),
            S_term = params.s_max .* rand(rng, hp.n_terminal),
            t_term = fill(params.maturity, hp.n_terminal),
            S_bcmax = fill(params.s_max, hp.n_bc),
            t_bcmax = params.maturity .* rand(rng, hp.n_bc),
        )
    end

    model = make_mlp(2, (16, 16), 1; activation = NNlib.tanh)
    train_state = setup_training(rng_from_seed(SEED; offset = 1), model, Optimisers.Adam(hp.lr); parameter_type = Float64)
    bs_loss(model, ps, st, batch) = begin
        pieces, st_new = black_scholes_loss(model, ps, st, batch; params, terminal_weight = 10.0)
        return pieces.loss, st_new
    end
end

# ╔═╡ df6e7571-f457-64bb-e3a8-f99138dadd0c
md"""
### Training the PINN

The loss terms are divided by the strike so that a boundary error of one currency unit does not overwhelm the scaled PDE residual, and `terminal_weight = 10.0` up-weights the terminal payoff. The Python notebook uses Adam to reach the right basin and then runs **L-BFGS** on a fixed double-precision batch — FP64 matters there because the L-BFGS line-search and stopping criteria are sensitive to small loss changes. This Julia smoke preview keeps the FP64 parameters (`parameter_type = Float64`) and the Adam stage via `train_step!`, but **defers the deterministic L-BFGS polish**; smoke mode is a finite-execution check, not an accuracy guarantee.
"""

# ╔═╡ 55555555-1105-4555-8555-555555555555
begin
    initial_batch = sample_bs_batch(rng, hp, params)
    initial_loss = loss_value(train_state, bs_loss, initial_batch)
    history = NamedTuple[]
    for step in 1:hp.steps
        local batch = sample_bs_batch(rng, hp, params)
        metrics = train_step!(train_state, bs_loss, batch; max_grad_norm = 50.0)
        append_metric!(history; step, loss = metrics.loss)
    end
end

# ╔═╡ a3534327-ab8b-193a-382f-bc6f3797240e
md"""
### Analytical Black-Scholes Formula

For validation we compare the PINN solution against the closed-form Black-Scholes formula for a European call:

\$\$C(S, t) = S\,\Phi(d_1) - K e^{-r(T-t)}\Phi(d_2),\$\$

where \$d_1 = \frac{\ln(S/K) + (r + \sigma^2/2)(T-t)}{\sigma\sqrt{T-t}}\$ and \$d_2 = d_1 - \sigma\sqrt{T-t}\$, and \$\Phi\$ is the standard normal CDF. The Julia preview evaluates the price with `black_scholes_call_price` and its delta with `black_scholes_delta`.

### Comparison: PINN vs. Analytical Solution

We evaluate both the trained PINN and the analytical formula at \$t = 0\$ (i.e. time to maturity \$= T\$) across the full range of spot prices \$S \in [0, S_{\max}]\$.

### Error Analysis

We quantify the point-wise absolute error \$|V_{\text{PINN}}(S,0) - V_{\text{exact}}(S,0)|\$ across the spot-price domain; the final diagnostics cell reports its maximum alongside the relative \$L^2\$ errors of the price and of the delta.

### Discussion: The Greeks

A powerful advantage of the PINN approach is that the **Greeks** — the price sensitivities with respect to the underlying parameters — are available essentially for free via automatic differentiation:

| Greek   | Definition          | Interpretation                       |
|---------|---------------------|--------------------------------------|
| Delta   | \$\Delta = V_S\$     | Sensitivity to spot price            |
| Gamma   | \$\Gamma = V_{SS}\$  | Convexity with respect to spot price |
| Theta   | \$\Theta = V_t\$     | Sensitivity to passage of time       |

There is no need for finite-difference bumping or re-solving the PDE. The cell below computes **Delta** at \$t = 0\$ from the same `ForwardDiff` derivatives (`black_scholes_value_derivatives(...).dS`) that enter the residual, and compares it against `black_scholes_delta`.
"""

# ╔═╡ 66666666-1105-4666-8666-666666666666
begin
    S_eval = collect(range(0.0, params.s_max; length = 60))
    V_hat = [black_scholes_value_derivatives(train_state.ps, S, 0.0; params).value for S in S_eval]
    V_exact = [black_scholes_call_price(S, params.maturity; params) for S in S_eval]
    delta_hat = [black_scholes_value_derivatives(train_state.ps, S, 0.0; params).dS for S in S_eval]
    delta_exact = [black_scholes_delta(S, params.maturity; params) for S in S_eval]
    final_pieces, _ = black_scholes_loss(train_state.model, train_state.ps, train_state.st, initial_batch; params, terminal_weight = 10.0)
end

# ╔═╡ a0f97e2a-8ba8-3171-18cc-8473ea386d4e
md"""
### Takeaway

- The Black–Scholes PDE has a **closed-form solution**, so this notebook is a *known-answer benchmark*: we verify the PINN recipe (smooth activations, soft BCs with a tuned terminal weight, autodiff for \$V_S\$ and \$V_{SS}\$, FP64 parameters — Adam here, Adam-then-L-BFGS in the full Python run) before applying it to PDEs without closed forms (American options, jump diffusions, multi-asset pricing).
- The Greeks (\$\Delta = \partial V/\partial S\$ and beyond) come for free via `ForwardDiff` (the Julia counterpart of `torch.autograd.grad`), one of the practical advantages of PINN-based pricing over many traditional numerical schemes that need separate finite-difference evaluations.
- Quality of the fit at \$t = 0\$ should be checked against the analytical formula: if the PINN cannot recover Black–Scholes to plotting accuracy here, no further trust should be placed in its output on harder problems.

The cell below returns the machine-checkable diagnostics summary for this notebook's smoke run.
"""

# ╔═╡ 77777777-1105-4777-8777-777777777777
(
    initial_loss = initial_loss,
    final_loss = history[end].loss,
    eval_loss = final_pieces.loss,
    price_max_abs_error = max_abs_error(V_hat, V_exact),
    price_relative_l2 = relative_l2_error(V_hat, V_exact),
    delta_relative_l2 = relative_l2_error(delta_hat, delta_exact),
)

# ╔═╡ Cell order:
# ╟─11111111-1105-4111-8111-111111111111
# ╟─34e41429-1c58-f8bc-a966-0e0a6a84361c
# ╟─443f02a1-9615-199b-4859-460ffe5a06c5
# ╠═22222222-1105-4222-8222-222222222222
# ╠═33333333-1105-4333-8333-333333333333
# ╟─d036b363-f424-ba8c-226e-970d419a1005
# ╟─23da76e6-05ca-5b42-8bfa-a0834d553a6a
# ╟─8f0212f4-b975-d52b-9884-d433b33a35ab
# ╠═44444444-1105-4444-8444-444444444444
# ╟─df6e7571-f457-64bb-e3a8-f99138dadd0c
# ╠═55555555-1105-4555-8555-555555555555
# ╟─a3534327-ab8b-193a-382f-bc6f3797240e
# ╠═66666666-1105-4666-8666-666666666666
# ╟─a0f97e2a-8ba8-3171-18cc-8473ea386d4e
# ╠═77777777-1105-4777-8777-777777777777
