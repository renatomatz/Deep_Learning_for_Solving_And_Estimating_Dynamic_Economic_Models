### A Pluto.jl notebook ###
# v0.20.0

using Markdown
using InteractiveUtils

# ╔═╡ 11111111-0206-4111-8111-111111111111
md"""
# Lecture 02, Notebook 06: Lux Training Fundamentals

The Python notebook introduces PyTorch modules, tensors, gradients, and
optimizers. Here the same concepts are shown with Lux's explicit
`model(x, ps, st)` interface and the shared training helper.

Read this after `lecture_02_00_Lux_Pluto_orientation.jl` if you want the
mechanics behind later notebooks. The goal is to see exactly where model
parameters `ps`, threaded layer state `st`, gradients, optimizer state, and
metrics live.
"""

# ╔═╡ 45c8537e-5d0f-0087-d4a6-4435fac61aa2
md"""
## Lecture 02, Notebook 06: PyTorch Fundamentals

**Course:** Deep Learning for Solving and Estimating Dynamic Models in Economics and Finance
**Script reference:** §1.5–1.8 (Optimization and training: PyTorch fundamentals)
**Notebook role:** core
**Author:** Simon Scheidegger

*Julia/Lux/Pluto preview of* `lectures/lecture_02_intro_deep_learning/code/lecture_02_06_PyTorch_intro.ipynb`.

> **Run mode.** The checked-in run uses `RUN_MODE = "smoke"` for fast execution; set `RUN_MODE` to `"teaching"` or `"production"` in the setup cell below for the longer training budgets.
"""

# ╔═╡ 95ff596b-4274-750a-82b1-5f12ac232fbb
md"""
## From PyTorch modules to Lux training fundamentals

The Python ground-truth notebook is a PyTorch tour: it builds `nn.Module` networks and trains them with `torch.optim` on two tasks — a **regression** example (MSE loss) and a binary **classification** example (BCE loss). PyTorch bundles parameters, layer state, and gradients inside the module object.

This Julia preview teaches the **same training fundamentals in Lux**, where those pieces stay explicit instead of hidden inside an object:

- the **model** is a pure description of the architecture (`make_mlp`), separate from
- its **parameters** `ps` and layer **state** `st`, and
- **gradients** come from `Zygote`, while **optimizer state** lives in an `Optimisers.jl` rule.

We walk the full loop on a supervised **regression** target: define the model, inspect a single gradient with `Zygote.pullback`, run the shared `train_step!` helper, and visualise the fit. The Python notebook's classification example (BCE loss, sigmoid output, decision-boundary plot) is summarised near the end — the training mechanics are identical; only the loss and the data differ.
"""

# ╔═╡ 21212121-0206-4212-8212-212121212121
md"""
Lux separates the model architecture from its parameters and state. A model call
returns both predictions and updated state:

```julia
prediction, st_new = model(x, ps, st)
```

That is the pattern used throughout the Julia translations, including the DEQN
and PINN residual notebooks.
"""

# ╔═╡ 22222222-0206-4222-8222-222222222222
begin
    import Pkg
    Pkg.activate(joinpath(@__DIR__, "..", "..", "..", "julia"))

    using CairoMakie
    using DLEFJulia
    using Lux
    using NNlib
    using Optimisers
    using Statistics
    using Zygote
end

# ╔═╡ 33333333-0206-4333-8333-333333333333
begin
    RUN_MODE = "smoke"
    SEED = 0
    budgets = (
        smoke = (epochs = 30, batch_size = 32),
        teaching = (epochs = 200, batch_size = 64),
        production = (epochs = 1_000, batch_size = 128),
    )
    hp = run_mode_budget(RUN_MODE; budgets)
    rng = rng_from_seed(SEED)
end

# ╔═╡ 34343434-0206-4343-8343-343434343434
md"""
---
### 1. Regression example with a neural network

The Python notebook fits a small MLP to noisy data from \$y = 2x + 1 + \text{noise}\$. Here we use a smooth nonlinear target so the training mechanics are visible on a slightly harder fit; the Lux boundary is the point, not the exact curve.

We build the model with `make_mlp` (input dimension 1, two hidden layers, `tanh` activation, output dimension 1) and pair it with an `Optimisers.Adam` rule via `setup_training`. The inputs are already in Lux's **feature-by-batch** layout — one feature row, many observation columns — so the explicit call is `prediction, st_new = model(batch.x, ps, st)`.
"""

# ╔═╡ 44444444-0206-4444-8444-444444444444
begin
    x = reshape(collect(range(-2.0, 2.0; length = 160)), 1, :)
    y = @. 0.5 * x^3 - x + 0.15 * sin(8x)
    model = make_mlp(1, (20, 20), 1; activation = NNlib.tanh)
    state = setup_training(rng_from_seed(SEED; offset = 1), model, Optimisers.Adam(0.01); parameter_type = Float64)

    supervised_loss(model, ps, st, batch) = begin
        prediction, st_new = model(batch.x, ps, st)
        return mse_loss(prediction, batch.y), st_new
    end
    full_batch = (x = x, y = y)
end

# ╔═╡ 45454545-0206-4455-8455-454545454545
md"""
#### Inspecting a gradient

PyTorch computes gradients implicitly when you call `loss.backward()`. In Lux the gradient is an explicit object: `Zygote.pullback` evaluates the scalar loss and returns a function that backpropagates through the parameters `ps`, leaving Lux state handling visible. This is exactly what the shared `train_step!` helper automates in the next cell.
"""

# ╔═╡ 55555555-0206-4555-8555-555555555555
begin
    initial_loss = loss_value(state, supervised_loss, full_batch)
    (loss_before, st_candidate), back = Zygote.pullback(state.ps) do ps
        supervised_loss(state.model, ps, state.st, full_batch)
    end
    manual_grads = only(back((one(loss_before), nothing)))
    manual_grad_norm = sqrt(tree_sum_abs2(manual_grads))
end

# ╔═╡ 56565656-0206-4566-8566-565656565656
md"""
#### Training the regression model

The Python notebook trains with `nn.MSELoss()` and `optim.Adam`. The Lux equivalent is the shared `train_step!` helper, which repeats the same pieces on each step: compute the loss and updated state, take parameter gradients with `Zygote`, clip the gradient norm, update the `Optimisers.Adam` state, and record finite metrics. Later economic notebooks plug residual losses into this same shape.
"""

# ╔═╡ 66666666-0206-4666-8666-666666666666
begin
    history = NamedTuple[]
    for _ in 1:hp.epochs
        metrics = train_step!(state, supervised_loss, full_batch; max_grad_norm = 10.0)
        append_metric!(history; step = metrics.step, loss = metrics.loss, grad_norm = metrics.grad_norm)
    end
    final_loss = loss_value(state, supervised_loss, full_batch)
end

# ╔═╡ 11820547-a755-e21f-3730-bf0291cb7ac5
md"""
#### Visualizing the regression results

We compare the trained Lux MLP's predictions to the true target across the input range.
"""

# ╔═╡ 77777777-0206-4777-8777-777777777777
begin
    prediction, _ = state.model(x, state.ps, state.st)
    fig = Figure(size = figure_size(RUN_MODE))
    ax = Axis(fig[1, 1], xlabel = "x", ylabel = "y")
    lines!(ax, vec(x), vec(y); label = "target", color = :black, linewidth = 3)
    lines!(ax, vec(x), vec(prediction); label = "Lux MLP", color = :dodgerblue3, linewidth = 3)
    axislegend(ax; position = :lt)
    fig
end

# ╔═╡ 0e33b01e-0fe7-db66-7569-c3c31b5340e6
md"""
---
### The full Python notebook also covers: classification

The PyTorch ground truth continues with a second task — binary **classification**:

- **Data.** Two clusters drawn from distinct Gaussian distributions, giving a 2-feature, 2-class problem.
- **Model.** A small MLP (input dimension 2, one hidden layer, sigmoid output) returning class probabilities.
- **Loss.** Binary cross-entropy, \$J = -[y\log\hat{y} + (1-y)\log(1-\hat{y})]\$, minimised with Adam — the same optimizer as the regression task.
- **Output.** The decision boundary is visualised by predicting class probabilities over a grid of the feature space.

In Lux this is the identical training loop shown above: only the final activation (sigmoid), the loss kernel (BCE instead of MSE), and the data change. The `model, ps, st` / `Zygote` / `Optimisers` machinery is unchanged.
"""

# ╔═╡ 0d8da4e1-653c-18b6-9586-f971aa702b74
md"""
---
### Conclusion

This notebook re-cast the PyTorch introduction as **Lux training fundamentals**. Using a supervised regression example we saw exactly where each piece lives:

- the **model** (`make_mlp`) is separate from its **parameters** `ps` and **state** `st`,
- **gradients** come from `Zygote.pullback`,
- **optimizer state** lives in an `Optimisers.Adam` rule driven by `train_step!`.

The Python ground truth reuses this same loop for a binary-classification MLP (BCE loss, sigmoid output). Together, regression and classification are the foundation for the deeper architectures and economic residual losses that follow.

**Next steps.**

- `lecture_02_07_Genz_Approximation_and_Loss_Functions_Lux.jl` shows the shared loss kernels used later in stochastic residuals.
- `../../lecture_03_deep_equilibrium_nets/code_julia/lecture_03_01_Brock_Mirman_1972_DEQN_Lux.jl` is the first notebook where this training pattern drives an economic equilibrium residual.

The cell below returns a machine-checkable summary of this notebook's run.
"""

# ╔═╡ 88888888-0206-4888-8888-888888888888
(
    initial_loss = initial_loss,
    final_loss = final_loss,
    manual_pullback_loss = loss_before,
    manual_grad_norm = manual_grad_norm,
    candidate_state_type = typeof(st_candidate),
    steps = length(history),
)

# ╔═╡ Cell order:
# ╟─11111111-0206-4111-8111-111111111111
# ╟─45c8537e-5d0f-0087-d4a6-4435fac61aa2
# ╟─95ff596b-4274-750a-82b1-5f12ac232fbb
# ╟─21212121-0206-4212-8212-212121212121
# ╠═22222222-0206-4222-8222-222222222222
# ╠═33333333-0206-4333-8333-333333333333
# ╟─34343434-0206-4343-8343-343434343434
# ╠═44444444-0206-4444-8444-444444444444
# ╟─45454545-0206-4455-8455-454545454545
# ╠═55555555-0206-4555-8555-555555555555
# ╟─56565656-0206-4566-8566-565656565656
# ╠═66666666-0206-4666-8666-666666666666
# ╟─11820547-a755-e21f-3730-bf0291cb7ac5
# ╠═77777777-0206-4777-8777-777777777777
# ╟─0e33b01e-0fe7-db66-7569-c3c31b5340e6
# ╟─0d8da4e1-653c-18b6-9586-f971aa702b74
# ╠═88888888-0206-4888-8888-888888888888
