### A Pluto.jl notebook ###
# v0.20.0

using Markdown
using InteractiveUtils

# ╔═╡ 11111111-1102-4111-8111-111111111111
md"""
# Lecture 11, Notebook 02: Soft versus hard ODE boundary conditions in Lux

This notebook compares soft boundary penalties with a hard trial solution for
the manufactured ODE `y''(x) + 1 = 0` on `[0, 1]`, with `y(0)=1` and `y(1)=2`.
The exact solution is `-x^2 / 2 + 3x / 2 + 1`.
"""

# ╔═╡ 350c75c3-a72a-9215-89d1-c4a1f3cb4e53
md"""
## Lecture 11, Notebook 02: Soft versus hard boundary conditions in PINNs

**Course:** Deep Learning for Solving and Estimating Dynamic Models in Economics and Finance
**Script reference:** §7.3 (boundary conditions: soft vs hard enforcement)
**Notebook role:** core
**Author:** Simon Scheidegger

*Julia/Lux/Pluto preview of* `lectures/lecture_11_pinns/code/lecture_11_02_ODE_PINN_SoftVsHardBCs.ipynb`.

> **Run mode.** The checked-in run uses `RUN_MODE = "smoke"` for fast execution; the accuracy figures quoted in the slides and the companion script use the longer `teaching` / `production` budgets. Set `RUN_MODE` in the next cell accordingly to reproduce them.

> **In-class notebook** (Day 6, Block 1 — PINNs Foundations & Economic Applications, 75 min)
"""

# ╔═╡ 380a52bc-697d-9e78-de9e-bd3814958af4
md"""
## Soft vs. Hard Boundary Conditions in PINNs

Physics-Informed Neural Networks (PINNs) embed differential equations into the training loss so that the network learns a solution that satisfies the governing PDE. A key design choice is **how boundary conditions are enforced**.

This notebook compares two strategies on a simple second-order ODE:

\$\$y''(x) = -1, \quad x \in (0,1), \qquad y(0)=1,\; y(1)=2.\$\$

The analytical solution is \$y(x) = -\tfrac{1}{2}x^2 + \tfrac{3}{2}x + 1\$.

| Approach | Idea | Pros | Cons |
|----------|------|------|------|
| **Soft (penalty)** | Add BC residuals to the loss with a penalty weight | Simple to implement; works for any BC type | BCs only approximately satisfied; weight tuning needed |
| **Hard (trial solution)** | Construct an ansatz \$\hat{y}(x)=A(x)+B(x)\,N(x;\theta)\$ that satisfies BCs exactly by design | BCs satisfied exactly; loss has fewer terms | Requires problem-specific construction of \$A\$ and \$B\$ |
"""

# ╔═╡ 22222222-1102-4222-8222-222222222222
begin
    import Pkg
    Pkg.activate(joinpath(@__DIR__, "..", "..", "..", "julia"))

    using CairoMakie
    using DLEFJulia
    using Lux
    using NNlib
    using Optimisers
    using Statistics
end

# ╔═╡ 33333333-1102-4333-8333-333333333333
begin
    RUN_MODE = "smoke"
    SEED = 0
    budgets = (
        smoke = (steps = 12, n_colloc = 12, lr = 0.001, bc_weight = 1.0, hard_resample_every = 100),
        teaching = (steps = 2_000, n_colloc = 30, lr = 0.001, bc_weight = 1.0, hard_resample_every = 100),
        production = (steps = 10_000, n_colloc = 50, lr = 0.001, bc_weight = 1.0, hard_resample_every = 100),
    )
    hp = run_mode_budget(RUN_MODE; budgets)
    rng = rng_from_seed(SEED)
end

# ╔═╡ 78e7e0a1-6cb8-2297-11d6-0dbf8a9bab6e
md"""
### Part 1: Soft Boundary Conditions

In the **soft** (penalty) approach we train a raw neural network \$N(x;\theta)\$ and define the loss as

\$\$\mathcal{L} = \underbrace{\frac{1}{N_r}\sum_{i=1}^{N_r}\bigl(y''(x_i)+1\bigr)^2}_{\text{PDE residual}}
  + \underbrace{\bigl(y(0)-1\bigr)^2 + \bigl(y(1)-2\bigr)^2}_{\text{BC penalty}}.\$\$

The boundary conditions are only satisfied **approximately**, to the extent that the optimiser can drive the penalty terms to zero alongside the PDE residual.

The Julia preview defines the soft and hard loss functions together in the next cell and trains both models side by side, rather than in two separate passes as in the Python notebook. Second derivatives \$y''\$ come from an analytic tanh-MLP derivative propagation (the shared `tanh_mlp_value_second_derivative` helper, built on `tanh_mlp_scalar_derivatives`, which walks the Dense layers propagating the input Jacobian and Hessian in closed form). `ForwardDiff` is used only for the exact-solution residual check, not to differentiate the network in the PINN residual.
"""

# ╔═╡ 37486fce-5d0d-8a5c-8d72-790d179b4662
md"""
### Part 2: Hard Boundary Conditions

In the **hard** (trial-solution) approach we never ask the optimiser to enforce boundary conditions. Instead we construct an ansatz (trial solution) that satisfies them **by construction**:

\$\$\hat{y}(x) = A(x) + B(x)\,N(x;\theta),\$\$

where

* \$A(x) = 1 + x\$ satisfies both BCs: \$A(0)=1\$, \$A(1)=2\$,
* \$B(x) = x(1-x)\$ vanishes at the boundaries: \$B(0)=B(1)=0\$,
* \$N(x;\theta)\$ is a free neural network.

No matter what \$N\$ outputs, \$\hat{y}\$ always satisfies \$\hat{y}(0)=1\$ and \$\hat{y}(1)=2\$. The loss therefore contains **only** the PDE residual. In Lux these pieces are `ode_boundary_lift` (\$A\$), `ode_bubble` (\$B\$), and the network raw output.
"""

# ╔═╡ 44444444-1102-4444-8444-444444444444
begin
    ode_exact_solution(x) = -0.5 * x^2 + 1.5 * x + 1.0
    ode_boundary_lift(x) = 1.0 + x
    ode_bubble(x) = x * (1.0 - x)

    function hard_trial_value_second_derivative(ps, x::Real)
        n, dn, d2n = tanh_mlp_scalar_derivatives(ps, [x])
        bubble = ode_bubble(x)
        dbubble = 1.0 - 2.0 * x
        d2bubble = -2.0
        value = ode_boundary_lift(x) + bubble * n
        d2value = d2bubble * n + 2.0 * dbubble * dn[1] + bubble * d2n[1, 1]
        return value, d2value
    end

    function soft_bc_ode_loss(model, ps, st, x_points; bc_weight::Real = 1.0)
        x_batch = assert_feature_batch(x_points, 1)
        residual = reshape([tanh_mlp_value_second_derivative(ps, x)[2] + one(x) for x in vec(x_batch)], 1, :)
        pde_loss = mean(abs2, residual)
        y0, _ = tanh_mlp_value_second_derivative(ps, 0.0)
        y1, _ = tanh_mlp_value_second_derivative(ps, 1.0)
        bc0 = abs2(y0 - 1.0)
        bc1 = abs2(y1 - 2.0)
        bc_loss = bc0 + bc1
        return (
            loss = pde_loss + bc_weight * bc_loss,
            pde_loss = pde_loss,
            bc_loss = bc_loss,
            bc0 = bc0,
            bc1 = bc1,
            residual = residual,
            boundary = (y0 = y0, y1 = y1),
        ), st
    end

    function hard_bc_ode_loss(model, ps, st, x_points)
        x_batch = assert_feature_batch(x_points, 1)
        residual = reshape([hard_trial_value_second_derivative(ps, x)[2] + one(x) for x in vec(x_batch)], 1, :)
        pde_loss = mean(abs2, residual)
        y0, _ = hard_trial_value_second_derivative(ps, 0.0)
        y1, _ = hard_trial_value_second_derivative(ps, 1.0)
        bc_loss = abs2(y0 - 1.0) + abs2(y1 - 2.0)
        return (
            loss = pde_loss,
            pde_loss = pde_loss,
            bc_loss = bc_loss,
            residual = residual,
            boundary = (y0 = y0, y1 = y1),
        ), st
    end

    function hard_trial_values(model, ps, st, x_points)
        x_batch = assert_feature_batch(x_points, 1)
        raw, st_new = model(x_batch, ps, st)
        return ode_boundary_lift.(x_batch) .+ ode_bubble.(x_batch) .* raw, st_new
    end
end

# ╔═╡ 55555555-1102-4555-8555-555555555555
begin
    x_check = reshape(collect(range(0.0, 1.0; length = 9)), 1, :)
    exact_residual = reshape([second_derivative(ode_exact_solution, x) + 1.0 for x in vec(x_check)], 1, :)
    exact_boundary_error = max(abs(ode_exact_solution(0.0) - 1.0), abs(ode_exact_solution(1.0) - 2.0))
    exact_checks = (
        max_abs_residual = maximum(abs.(exact_residual)),
        boundary_error = exact_boundary_error,
    )
    @assert exact_checks.max_abs_residual < 1e-10
    @assert exact_checks.boundary_error < 1e-12
    exact_checks
end

# ╔═╡ 66666666-1102-4666-8666-666666666666
begin
    fixed_interior = reshape(collect(range(0.01, 0.99; length = hp.n_colloc)), 1, :)
    sample_hard_collocation(rng, n) = reshape(rand(rng, n), 1, :)

    soft_model = make_mlp(1, (20, 20), 1; activation = NNlib.tanh)
    hard_model = make_mlp(1, (20, 20), 1; activation = NNlib.tanh)
    soft_state = setup_training(rng_from_seed(SEED; offset = 1), soft_model, Optimisers.Adam(hp.lr); parameter_type = Float64)
    hard_state = setup_training(rng_from_seed(SEED; offset = 1), hard_model, Optimisers.Adam(hp.lr); parameter_type = Float64)

    soft_loss(model, ps, st, batch) = begin
        pieces, st_new = soft_bc_ode_loss(model, ps, st, batch; bc_weight = hp.bc_weight)
        return pieces.loss, st_new
    end
    hard_loss(model, ps, st, batch) = begin
        pieces, st_new = hard_bc_ode_loss(model, ps, st, batch)
        return pieces.loss, st_new
    end

    hard_batch_ref = Ref(sample_hard_collocation(rng, hp.n_colloc))
    initial_soft_loss = loss_value(soft_state, soft_loss, fixed_interior)
    initial_hard_loss = loss_value(hard_state, hard_loss, hard_batch_ref[])
    history = NamedTuple[]
    for step in 1:hp.steps
        if step % hp.hard_resample_every == 1
            hard_batch_ref[] = sample_hard_collocation(rng, hp.n_colloc)
        end
        soft_metrics = train_step!(soft_state, soft_loss, fixed_interior; max_grad_norm = 50.0)
        hard_metrics = train_step!(hard_state, hard_loss, hard_batch_ref[]; max_grad_norm = 50.0)
        append_metric!(history; step, soft_loss = soft_metrics.loss, hard_loss = hard_metrics.loss)
    end
end

# ╔═╡ 77777777-1102-4777-8777-777777777777
begin
    x_eval = reshape(collect(range(0.0, 1.0; length = 100)), 1, :)
    y_soft, _ = soft_state.model(x_eval, soft_state.ps, soft_state.st)
    y_hard, _ = hard_trial_values(hard_state.model, hard_state.ps, hard_state.st, x_eval)
    y_exact = ode_exact_solution.(x_eval)

    soft_error = abs.(y_soft .- y_exact)
    hard_error = abs.(y_hard .- y_exact)
    final_soft, _ = soft_bc_ode_loss(soft_state.model, soft_state.ps, soft_state.st, x_eval; bc_weight = hp.bc_weight)
    final_hard, _ = hard_bc_ode_loss(hard_state.model, hard_state.ps, hard_state.st, x_eval)

    loss_diagnostics = (
        all_losses_finite = finite_loss((
            initial_soft_loss,
            initial_hard_loss,
            history[end].soft_loss,
            history[end].hard_loss,
            final_soft.loss,
            final_hard.loss,
        )),
        soft_residual = residual_summary(final_soft.residual),
        hard_residual = residual_summary(final_hard.residual),
    )
    @assert loss_diagnostics.all_losses_finite
end

# ╔═╡ 3d871737-28ff-0904-88c1-f4f99a070048
md"""
### Comparison

We now plot both PINN solutions against the analytical reference and compare their absolute errors across the domain.
"""

# ╔═╡ 88888888-1102-4888-8888-888888888888
begin
    fig = Figure(size = (900, 420))
    ax_solution = Axis(fig[1, 1], xlabel = "x", ylabel = "y(x)", title = "PINN solutions")
    lines!(ax_solution, vec(x_eval), vec(y_exact); color = :black, linewidth = 3, label = "analytic")
    lines!(ax_solution, vec(x_eval), vec(y_soft); color = :dodgerblue3, linestyle = :dash, linewidth = 3, label = "soft BC")
    lines!(ax_solution, vec(x_eval), vec(y_hard); color = :firebrick3, linestyle = :dashdot, linewidth = 3, label = "hard BC")
    axislegend(ax_solution; position = :lt)

    ax_error = Axis(fig[1, 2], xlabel = "x", ylabel = "|error|", title = "Absolute error", yscale = log10)
    lines!(ax_error, vec(x_eval), max.(vec(soft_error), 1e-12); color = :dodgerblue3, linewidth = 3, label = "soft BC")
    lines!(ax_error, vec(x_eval), max.(vec(hard_error), 1e-12); color = :firebrick3, linewidth = 3, label = "hard BC")
    axislegend(ax_error; position = :rt)
    fig
end

# ╔═╡ ea0e9bf9-d18b-cce1-d9ff-c34c9cc13f4b
md"""
### Takeaway

- **Soft BCs** (penalty term in the loss) require tuning the BC weight: a small weight produces visible boundary violations; a large weight crowds out the PDE residual. Even after balancing, the soft-BC error is typically 1–2 orders of magnitude larger than the hard-BC error on this 1D problem.
- **Hard BCs** via the trial solution `ŷ(x) = A(x) + B(x)·N(x)` satisfy the Dirichlet conditions exactly by construction and remove the BC term entirely; the optimizer only sees the interior PDE residual.
- The cost of hard BCs is reduced expressivity at the boundary (the mask `B` damps gradients near `∂Ω`), which becomes relevant when the true solution has a sharp boundary layer (e.g., HJB at the borrowing constraint, see notebook 04).

The cell below returns the machine-checkable diagnostics summary for this notebook's smoke run.
"""

# ╔═╡ 99999999-1102-4999-8999-999999999999
(
    initial_soft_loss = initial_soft_loss,
    initial_hard_loss = initial_hard_loss,
    final_soft_training_loss = history[end].soft_loss,
    final_hard_training_loss = history[end].hard_loss,
    soft_eval_loss = final_soft.loss,
    hard_eval_loss = final_hard.loss,
    soft_max_abs_error = max_abs_error(y_soft, y_exact),
    hard_max_abs_error = max_abs_error(y_hard, y_exact),
    soft_boundary_error = max(abs(final_soft.boundary.y0 - 1.0), abs(final_soft.boundary.y1 - 2.0)),
    hard_boundary_error = max(abs(final_hard.boundary.y0 - 1.0), abs(final_hard.boundary.y1 - 2.0)),
    exact_max_abs_residual = exact_checks.max_abs_residual,
    losses_finite = loss_diagnostics.all_losses_finite,
)

# ╔═╡ Cell order:
# ╟─11111111-1102-4111-8111-111111111111
# ╟─350c75c3-a72a-9215-89d1-c4a1f3cb4e53
# ╟─380a52bc-697d-9e78-de9e-bd3814958af4
# ╠═22222222-1102-4222-8222-222222222222
# ╠═33333333-1102-4333-8333-333333333333
# ╟─78e7e0a1-6cb8-2297-11d6-0dbf8a9bab6e
# ╟─37486fce-5d0d-8a5c-8d72-790d179b4662
# ╠═44444444-1102-4444-8444-444444444444
# ╠═55555555-1102-4555-8555-555555555555
# ╠═66666666-1102-4666-8666-666666666666
# ╠═77777777-1102-4777-8777-777777777777
# ╟─3d871737-28ff-0904-88c1-f4f99a070048
# ╠═88888888-1102-4888-8888-888888888888
# ╟─ea0e9bf9-d18b-cce1-d9ff-c34c9cc13f4b
# ╠═99999999-1102-4999-8999-999999999999
