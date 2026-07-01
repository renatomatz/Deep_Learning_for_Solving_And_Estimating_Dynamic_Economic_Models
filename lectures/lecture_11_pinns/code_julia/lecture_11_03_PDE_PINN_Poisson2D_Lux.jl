### A Pluto.jl notebook ###
# v0.20.0

using Markdown
using InteractiveUtils

# ╔═╡ 11111111-1103-4111-8111-111111111111
md"""
# Lecture 11, Notebook 03: Poisson PINNs in Lux

This notebook mirrors the 2D manufactured Poisson example with soft and hard
Dirichlet boundary conditions. The hard version uses the boundary lifting plus a
bubble correction, so the boundary values are exact by construction.
"""

# ╔═╡ b96ceba9-90ce-78ac-4215-6741a7956318
md"""
## Lecture 11, Notebook 03: A 2D Poisson PINN with transfinite-interpolation hard BCs

**Course:** Deep Learning for Solving and Estimating Dynamic Models in Economics and Finance
**Script reference:** §7.3 (the 2D Poisson benchmark; transfinite interpolation)
**Notebook role:** core
**Author:** Simon Scheidegger

*Julia/Lux/Pluto preview of* `lectures/lecture_11_pinns/code/lecture_11_03_PDE_PINN_Poisson2D.ipynb`.

> **Run mode.** The checked-in run uses `RUN_MODE = "smoke"` for fast execution; the accuracy figures quoted in the slides and the companion script use the longer `teaching` / `production` budgets. Set `RUN_MODE` in the next cell accordingly to reproduce them.

> **Self-study notebook** — This notebook complements the in-class PINNs session (Day 6, Block 1). Work through it at your own pace.
"""

# ╔═╡ 8ea117b9-6e39-6385-f802-d19cd1ead176
md"""
## 2D Poisson Equation with PINNs: Soft vs. Hard Boundary Conditions

In this notebook we solve the **2D Poisson equation**

\$\$\nabla^2 u(x,y) = f(x,y), \quad (x,y) \in (0,1)^2,\$\$

subject to Dirichlet boundary conditions, using **Physics-Informed Neural Networks (PINNs)**.

We compare two strategies for enforcing boundary conditions:

1. **Soft boundary conditions** — the boundary mismatch is penalised in the loss function.
2. **Hard boundary conditions** — the network ansatz is constructed so that boundary conditions are satisfied *exactly* via transfinite interpolation.

**Manufactured solution:**
\$u^*(x,y) = x^2 + y + \sin(\pi x)\sin(\pi y)\$
which gives the forcing term
\$f(x,y) = 2 - 2\pi^2 \sin(\pi x)\sin(\pi y).\$
"""

# ╔═╡ 22222222-1103-4222-8222-222222222222
begin
    import Pkg
    Pkg.activate(joinpath(@__DIR__, "..", "..", "..", "julia"))

    using DLEFJulia
    using Lux
    using NNlib
    using Optimisers
    using Statistics
end

# ╔═╡ 33333333-1103-4333-8333-333333333333
begin
    RUN_MODE = "smoke"
    SEED = 0
    budgets = (
        smoke = (steps = 2, n_interior = 6, n_boundary = 3, lr = 0.001),
        teaching = (steps = 600, n_interior = 256, n_boundary = 48, lr = 0.001),
        production = (steps = 4_000, n_interior = 1_024, n_boundary = 128, lr = 0.001),
    )
    hp = run_mode_budget(RUN_MODE; budgets)
    rng = rng_from_seed(SEED)
end

# ╔═╡ 02b0ce16-d0eb-7480-2462-5ae968d72102
md"""
### Analytical solution, network, differential operators, and sampling

The Python notebook develops these in four short sections. The Julia preview pulls the manufactured solution, forcing term, network evaluation, and Laplacian from the shared `DLEFJulia` helpers, and only builds the models and samples the collocation points in the cell below.

**Analytical solution and forcing term.** The manufactured solution \$u^*(x,y) = x^2 + y + \sin(\pi x)\sin(\pi y)\$ and its forcing \$f(x,y) = 2 - 2\pi^2 \sin(\pi x)\sin(\pi y)\$ are supplied by `poisson2d_exact`; the residual helpers reuse them internally.

**Network architecture.** Both models are small tanh MLPs, `make_mlp(2, (16, 16), 1; activation = NNlib.tanh)`, mapping \$(x,y) \mapsto u\$ — the Lux analogue of the Python `MLP2D` (2 → 64 → 64 → 64 → 64 → 1). Parameters are Float64 so the second derivatives stay accurate.

**Differential operators.** The Laplacian \$\nabla^2 u = u_{xx} + u_{yy}\$ is taken with `ForwardDiff` on the network inputs (replacing PyTorch's `torch.autograd.grad(..., create_graph=True)`) inside the `poisson2d_*_loss` helpers, while `Zygote` supplies the parameter gradients during training.

**Sampling functions.** `sample_interior` draws points uniformly in \$(0,1)^2\$ and `sample_boundary` draws points on the four edges of \$\partial\Omega\$ for the soft-BC penalty. Both return feature-by-batch arrays (features in rows) as expected at the Lux boundary.
"""

# ╔═╡ 44444444-1103-4444-8444-444444444444
begin
    sample_interior(rng, n) = rand(rng, 2, n)

    function sample_boundary(rng, n)
        t = reshape(rand(rng, n), 1, :)
        left = vcat(zeros(1, n), t)
        right = vcat(ones(1, n), t)
        bottom = vcat(t, zeros(1, n))
        top = vcat(t, ones(1, n))
        return hcat(left, right, bottom, top)
    end

    soft_model = make_mlp(2, (16, 16), 1; activation = NNlib.tanh)
    hard_model = make_mlp(2, (16, 16), 1; activation = NNlib.tanh)
    soft_state = setup_training(rng_from_seed(SEED; offset = 1), soft_model, Optimisers.Adam(hp.lr); parameter_type = Float64)
    hard_state = setup_training(rng_from_seed(SEED; offset = 2), hard_model, Optimisers.Adam(hp.lr); parameter_type = Float64)
end

# ╔═╡ e2c8fa5f-c705-7c88-3cd5-7e5df7589ee9
md"""
### Training: soft vs. hard boundary conditions

**Soft boundary conditions.** The soft loss adds a boundary penalty to the interior PDE residual,

\$\$\mathcal{L}_{\text{soft}} = \frac{1}{N_r}\sum_i \bigl(\nabla^2 u(x_i,y_i) - f(x_i,y_i)\bigr)^2 + \lambda\,\frac{1}{N_b}\sum_j \bigl(u(x_j,y_j) - u^*(x_j,y_j)\bigr)^2,\$\$

with penalty weight \$\lambda = 10\$ (`bc_weight = 10.0`); the boundary conditions are only satisfied approximately.

**Hard boundary conditions via transfinite interpolation.** The hard model never sees a boundary penalty. It uses the ansatz

\$\$\hat{u}(x,y) = A(x,y) + B(x,y)\,N(x,y;\theta),\$\$

where the anchor \$A(x,y)\$ reproduces the Dirichlet data and the bubble mask \$B(x,y) = x(1-x)\,y(1-y)\$ vanishes on all four edges, so \$\hat{u} = A + B\cdot N\$ satisfies the boundary conditions by construction and the loss reduces to the interior residual alone. `poisson2d_hard_loss` builds this ansatz internally.

The Julia preview defines both losses and trains the two models **side by side** in one Adam loop, rather than in the two separate passes of the Python notebook. In this preview the shared Lux training loop also applies gradient-norm clipping (`max_grad_norm = 50.0`); the Python notebook uses plain Adam steps with no clipping. The clip is a loose harness default that leaves the short smoke run unchanged.
"""

# ╔═╡ 55555555-1103-4555-8555-555555555555
begin
    soft_loss(model, ps, st, batch) = begin
        pieces, st_new = poisson2d_soft_loss(model, ps, st, batch.interior, batch.boundary; bc_weight = 10.0)
        return pieces.loss, st_new
    end
    hard_loss(model, ps, st, batch) = begin
        pieces, st_new = poisson2d_hard_loss(model, ps, st, batch.interior)
        return pieces.loss, st_new
    end

    initial_batch = (interior = sample_interior(rng, hp.n_interior), boundary = sample_boundary(rng, hp.n_boundary))
    initial_soft_loss = loss_value(soft_state, soft_loss, initial_batch)
    initial_hard_loss = loss_value(hard_state, hard_loss, initial_batch)
    history = NamedTuple[]
    for step in 1:hp.steps
        local batch = (interior = sample_interior(rng, hp.n_interior), boundary = sample_boundary(rng, hp.n_boundary))
        soft_metrics = train_step!(soft_state, soft_loss, batch; max_grad_norm = 50.0)
        hard_metrics = train_step!(hard_state, hard_loss, batch; max_grad_norm = 50.0)
        append_metric!(history; step, soft_loss = soft_metrics.loss, hard_loss = hard_metrics.loss)
    end
end

# ╔═╡ 05be557a-47a9-9474-ac59-3bee67be703e
md"""
### Evaluation and Comparison

We evaluate the trained hard-BC model against the manufactured solution and summarise both models' interior residuals. The Julia preview reports the **relative \$L_2\$ error** of the hard-BC solution and the residual RMSE for each model on a coarse grid; the Python notebook additionally draws matplotlib contour plots of \$u_{\text{soft}}\$, \$u_{\text{hard}}\$, the true field, and the pointwise errors.
"""

# ╔═╡ 66666666-1103-4666-8666-666666666666
begin
    eval_xy = hcat([[x, y] for x in range(0.0, 1.0; length = 5) for y in range(0.0, 1.0; length = 5)]...)
    hard_values = [poisson2d_hard_value_derivatives(hard_state.ps, eval_xy[:, i])[1] for i in axes(eval_xy, 2)]
    exact_values = [poisson2d_exact(eval_xy[1, i], eval_xy[2, i]) for i in axes(eval_xy, 2)]
    final_soft, _ = poisson2d_soft_loss(soft_state.model, soft_state.ps, soft_state.st, initial_batch.interior, initial_batch.boundary)
    final_hard, _ = poisson2d_hard_loss(hard_state.model, hard_state.ps, hard_state.st, initial_batch.interior)
end

# ╔═╡ 93d070c9-aed4-7480-f03a-6bd90fe3c5f3
md"""
### Takeaway

- The **manufactured solution** `u*(x,y) = x² + y + sin(πx)sin(πy)` has non-zero Dirichlet data on three of the four edges, so this benchmark genuinely exercises the transfinite-interpolation anchor `A(x,y)` rather than collapsing to the trivial zero-BC case.
- Because `sin(πx)sin(πy)` already vanishes on `∂Ω`, the polynomial part `x² + y` serves as the anchor and the bubble mask `B(x,y) = x(1-x)y(1-y)` vanishes on every edge, so `û = A + B·N` satisfies the BCs by construction.
- **Hard BCs** typically beat soft BCs by ~1 order of magnitude on the relative L2 error here, with no penalty-weight tuning required.

The cell below returns the machine-checkable diagnostics summary for this notebook's smoke run.
"""

# ╔═╡ 77777777-1103-4777-8777-777777777777
(
    initial_soft_loss = initial_soft_loss,
    initial_hard_loss = initial_hard_loss,
    final_soft_loss = history[end].soft_loss,
    final_hard_loss = history[end].hard_loss,
    soft_residual_rmse = residual_summary(final_soft.residual).rmse,
    hard_residual_rmse = residual_summary(final_hard.residual).rmse,
    hard_eval_relative_l2 = relative_l2_error(hard_values, exact_values),
)

# ╔═╡ Cell order:
# ╟─11111111-1103-4111-8111-111111111111
# ╟─b96ceba9-90ce-78ac-4215-6741a7956318
# ╟─8ea117b9-6e39-6385-f802-d19cd1ead176
# ╠═22222222-1103-4222-8222-222222222222
# ╠═33333333-1103-4333-8333-333333333333
# ╟─02b0ce16-d0eb-7480-2462-5ae968d72102
# ╠═44444444-1103-4444-8444-444444444444
# ╟─e2c8fa5f-c705-7c88-3cd5-7e5df7589ee9
# ╠═55555555-1103-4555-8555-555555555555
# ╟─05be557a-47a9-9474-ac59-3bee67be703e
# ╠═66666666-1103-4666-8666-666666666666
# ╟─93d070c9-aed4-7480-f03a-6bd90fe3c5f3
# ╠═77777777-1103-4777-8777-777777777777
