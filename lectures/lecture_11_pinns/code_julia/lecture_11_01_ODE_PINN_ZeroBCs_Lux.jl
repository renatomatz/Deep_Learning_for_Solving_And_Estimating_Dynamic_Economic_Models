### A Pluto.jl notebook ###
# v0.20.0

using Markdown
using InteractiveUtils

# ╔═╡ 11111111-1101-4111-8111-111111111111
md"""
# Lecture 11, Notebook 01: A 1D ODE PINN with Lux

The problem is `y''(x) + 1 = 0` on `[0, 1]` with zero boundary conditions.
The analytical solution is `x * (1 - x) / 2`.
"""

# ╔═╡ e054cc6c-ec1b-b2e5-6673-509e0456a643
md"""
## Lecture 11, Notebook 01: A 1D ODE PINN with soft zero boundary conditions

**Course:** Deep Learning for Solving and Estimating Dynamic Models in Economics and Finance
**Script reference:** §7.2 (the PINN loss and automatic differentiation; the 1D ODE example)
**Notebook role:** core
**Author:** Simon Scheidegger

*Julia/Lux/Pluto preview of* `lectures/lecture_11_pinns/code/lecture_11_01_ODE_PINN_ZeroBCs.ipynb`.

> **Run mode.** The checked-in run uses `RUN_MODE = "smoke"` for fast execution; the accuracy figures quoted in the slides and the companion script use the longer `teaching` / `production` budgets. Set `RUN_MODE` in the next cell accordingly to reproduce them.

> **Self-study notebook** — This notebook complements the in-class PINNs session (Day 6, Block 1). Work through it at your own pace.
"""

# ╔═╡ 0ce72f43-c79c-941d-d95e-860f03bca0cf
md"""
## Physics-Informed Neural Networks: 1D ODE with Zero Boundary Conditions

**Physics-Informed Neural Networks (PINNs)** embed the governing equations of a physical system directly into the loss function of a neural network. Instead of learning purely from data, the network is trained to satisfy:

1. The **differential equation** (the "physics") at a set of collocation points in the domain.
2. The **boundary conditions** (and/or initial conditions) at the domain boundaries.

Because modern deep-learning frameworks provide automatic differentiation, we can compute exact derivatives of the network output with respect to its inputs and penalize any violation of the ODE/PDE.

### Problem statement

We consider the simplest possible second-order ODE with homogeneous (zero) Dirichlet boundary conditions:

\$\$
y''(x) = -1, \quad x \in (0, 1), \qquad y(0) = 0, \quad y(1) = 0.
\$\$

The **analytical solution** is

\$\$
y(x) = \frac{x(1 - x)}{2}.
\$\$

We train a small fully connected `Lux` network to approximate this solution, using `ForwardDiff` for the input derivatives and `Zygote` for the parameter gradients.
"""

# ╔═╡ 22222222-1101-4222-8222-222222222222
begin
    import Pkg
    Pkg.activate(joinpath(@__DIR__, "..", "..", "..", "julia"))

    using CairoMakie
    using DLEFJulia
    using Lux
    using NNlib
    using Optimisers
end

# ╔═╡ 33333333-1101-4333-8333-333333333333
begin
    RUN_MODE = "smoke"
    SEED = 0
    budgets = (
        smoke = (steps = 12, n_interior = 8, lr = 0.01),
        teaching = (steps = 500, n_interior = 32, lr = 0.003),
        production = (steps = 3_000, n_interior = 64, lr = 0.001),
    )
    hp = run_mode_budget(RUN_MODE; budgets)
    rng = rng_from_seed(SEED)
end

# ╔═╡ 21ca0b3d-96fb-ad28-f849-528fc8d47a86
md"""
### Network Architecture

We use a simple **multi-layer perceptron (MLP)** with two hidden layers (here 16 units each, via `make_mlp(1, (16, 16), 1; activation = tanh)`). The activation function is `tanh`, which is infinitely differentiable (\$C^{\infty}\$). This smoothness is important because PINNs require computing higher-order derivatives of the network output with respect to its inputs via automatic differentiation. Using a non-smooth activation such as ReLU would produce zero (or undefined) second derivatives almost everywhere, making it unsuitable for our second-order ODE.
"""

# ╔═╡ 89300835-1915-9778-e6f6-d1d02ad9fcd5
md"""
### Computing the PDE Residual via Automatic Differentiation

The key ingredient of a PINN is the **residual** of the differential equation, evaluated using the network's current predictions. For our ODE \$y''(x) + 1 = 0\$ the residual is

\$\$
r(x) = y''_{\mathrm{NN}}(x) + 1.
\$\$

In PyTorch this residual is assembled with `torch.autograd.grad(..., create_graph=True)`, differentiating the network twice with respect to \$x\$. The Lux preview instead takes the second derivative \$y''_{\mathrm{NN}}(x)\$ with **`ForwardDiff`** (nested forward-mode AD through the network input), while the parameter gradients used for training flow through **`Zygote`**. The residual itself is assembled inside the `zero_bc_tanh_mlp_loss` helper.
"""

# ╔═╡ 297aa083-cb6a-f3c0-7018-64144356857c
md"""
### Training the PINN

The total loss is the sum of two terms:

1. **PDE loss** — mean squared residual over a set of interior collocation points:
   \$\$\mathcal{L}_{\text{PDE}} = \frac{1}{N}\sum_{i=1}^{N} r(x_i)^2.\$\$

2. **Boundary condition (BC) loss** — squared error at the two endpoints:
   \$\$\mathcal{L}_{\text{BC}} = y_{\mathrm{NN}}(0)^2 + y_{\mathrm{NN}}(1)^2.\$\$

We minimize \$\mathcal{L} = \mathcal{L}_{\text{PDE}} + \mathcal{L}_{\text{BC}}\$ with the Adam optimizer. In this Lux preview the next cell builds the model, wraps `zero_bc_tanh_mlp_loss` as the training objective, and runs the `Optimisers.Adam` loop via `train_step!` — architecture, residual, and training are condensed into a single cell rather than the separate steps of the Python notebook. Interior collocation points are resampled each step.
"""

# ╔═╡ 44444444-1101-4444-8444-444444444444
begin
    model = make_mlp(1, (16, 16), 1; activation = NNlib.tanh)
    train_state = setup_training(rng_from_seed(SEED; offset = 1), model, Optimisers.Adam(hp.lr); parameter_type = Float64)

    sample_x(rng, n) = reshape(rand(rng, n), 1, :)
    pinn_loss(model, ps, st, x_batch) = begin
        pieces, st_new = zero_bc_tanh_mlp_loss(model, ps, st, x_batch)
        return pieces.loss, st_new
    end

    initial_batch = sample_x(rng, hp.n_interior)
    initial_loss = loss_value(train_state, pinn_loss, initial_batch)
    history = NamedTuple[]
    for _ in 1:hp.steps
        local batch = sample_x(rng, hp.n_interior)
        metrics = train_step!(train_state, pinn_loss, batch; max_grad_norm = 10.0)
        append_metric!(history; step = metrics.step, loss = metrics.loss)
    end
end

# ╔═╡ ffe7299e-6b83-d002-ee7e-ab3abf39ace8
md"""
### Evaluation and Visualization

We compare the trained network's prediction with the analytical solution \$y(x) = x(1 - x)/2\$ on a fine grid. This Lux preview plots the solution against the analytic curve; the absolute pointwise error is reported numerically instead (its maximum appears in the diagnostics summary below).

> The full Python notebook also renders two extra panels: the absolute pointwise error on a log scale, and the training-loss convergence curve (a `semilogy` of the loss history versus step). In this preview the per-step loss is recorded in `history` and surfaced through the diagnostics summary rather than plotted.
"""

# ╔═╡ 55555555-1101-4555-8555-555555555555
begin
    x_eval = reshape(collect(range(0.0, 1.0; length = 100)), 1, :)
    y_pred, _ = train_state.model(x_eval, train_state.ps, train_state.st)
    y_exact = analytic_zero_bc_solution.(x_eval)
    abs_error = abs.(y_pred .- y_exact)
    final_pieces, _ = zero_bc_tanh_mlp_loss(train_state.model, train_state.ps, train_state.st, x_eval)
end

# ╔═╡ 66666666-1101-4666-8666-666666666666
begin
    fig = Figure(size = figure_size(RUN_MODE))
    ax = Axis(fig[1, 1], xlabel = "x", ylabel = "y(x)")
    lines!(ax, vec(x_eval), vec(y_exact); color = :black, linewidth = 3, label = "analytic")
    lines!(ax, vec(x_eval), vec(y_pred); color = :dodgerblue3, linewidth = 3, label = "PINN")
    axislegend(ax; position = :ct)
    fig
end

# ╔═╡ 4bf0ed90-4fe7-d410-2dce-daae4993e6d6
md"""
### Takeaway

- The PINN converged on a 1D Poisson-type ODE with **soft zero BCs**: a single penalty term \$y_{\mathrm{NN}}(0)^2 + y_{\mathrm{NN}}(1)^2\$ at the two endpoints is enough at this scale.
- A 2-layer tanh MLP on a few dozen collocation points reproduces the analytical solution \$y(x) = x(1-x)/2\$ to mean-absolute error well below `5e-3` in the *teaching* run and below `1e-3` in *production*; the checked-in `smoke` run is a finite-execution check rather than an accuracy guarantee.
- The next notebook (`lecture_11_02_ODE_PINN_SoftVsHardBCs`) compares this soft-BC pipeline with a hard-BC trial solution on a problem with non-zero Dirichlet data.

The cell below returns the machine-checkable diagnostics summary for this notebook's smoke run.
"""

# ╔═╡ 77777777-1101-4777-8777-777777777777
(
    initial_loss = initial_loss,
    final_loss = history[end].loss,
    eval_loss = final_pieces.loss,
    max_abs_error = maximum(abs_error),
    boundary = final_pieces.boundary,
)

# ╔═╡ Cell order:
# ╟─11111111-1101-4111-8111-111111111111
# ╟─e054cc6c-ec1b-b2e5-6673-509e0456a643
# ╟─0ce72f43-c79c-941d-d95e-860f03bca0cf
# ╠═22222222-1101-4222-8222-222222222222
# ╠═33333333-1101-4333-8333-333333333333
# ╟─21ca0b3d-96fb-ad28-f849-528fc8d47a86
# ╟─89300835-1915-9778-e6f6-d1d02ad9fcd5
# ╟─297aa083-cb6a-f3c0-7018-64144356857c
# ╠═44444444-1101-4444-8444-444444444444
# ╟─ffe7299e-6b83-d002-ee7e-ab3abf39ace8
# ╠═55555555-1101-4555-8555-555555555555
# ╠═66666666-1101-4666-8666-666666666666
# ╟─4bf0ed90-4fe7-d410-2dce-daae4993e6d6
# ╠═77777777-1101-4777-8777-777777777777
