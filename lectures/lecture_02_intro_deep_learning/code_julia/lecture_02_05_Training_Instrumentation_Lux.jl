### A Pluto.jl notebook ###
# v0.20.0

using Markdown
using InteractiveUtils

# ╔═╡ 11111111-0205-4111-8111-111111111111
md"""
# Lecture 02, Notebook 05: Training Instrumentation in Lux

The original notebook uses TensorBoard callbacks. This translation keeps the
same lesson without adding a logging dependency: training, validation, gradient
norms, and best-checkpoint metadata are recorded as plain Julia data.
"""

# ╔═╡ d9a6b4ec-7d09-cf74-0694-0280171c287f
md"""
## Lecture 02, Notebook 05: TensorBoard Instrumentation

**Course:** Deep Learning for Solving and Estimating Dynamic Models in Economics and Finance
**Script reference:** §1.5–1.8 (Training and regularization: TensorBoard instrumentation)
**Notebook role:** extension
**Author:** Simon Scheidegger

*Julia/Lux/Pluto preview of* `lectures/lecture_02_intro_deep_learning/code/lecture_02_05_Tensorboard.ipynb`.
"""

# ╔═╡ 94a192dd-50df-b891-9031-0ca2c693a6ea
md"""
## Monitoring training

When you train a network, the loss curves are the primary signal that it is converging to something sensible — indispensable once we start training networks to solve dynamic stochastic models (Day 2 onward), where the loss decomposes into Euler residuals, market clearing, and other components.

In TensorFlow this monitoring is done with **TensorBoard**, an interactive dashboard fed by callbacks during `model.fit`: scalar metrics (loss, MAE), weight and gradient histograms, the computational graph, and embeddings. The Julia analog would be `TensorBoardLogger.jl`, but this preview keeps zero logging dependencies and instead records the same signals as **plain Julia data**: a per-step metrics `DataFrame` (train loss, validation loss, gradient norm), best-checkpoint metadata, and an early-stopping counter.

In this notebook we:
1. Generate a synthetic 2-D regression problem.
2. Train a small Lux MLP, recording per-step train/validation loss and gradient norms.
3. Track the best validation checkpoint and stop early when it stops improving (`patience`).
4. Inspect the recorded metrics with a loss-curve plot — no dashboard required.
"""

# ╔═╡ 22222222-0205-4222-8222-222222222222
begin
    import Pkg
    Pkg.activate(joinpath(@__DIR__, "..", "..", "..", "julia"))

    using CairoMakie
    using DataFrames
    using DLEFJulia
    using Lux
    using NNlib
    using Optimisers
    using Statistics
end

# ╔═╡ 33333333-0205-4333-8333-333333333333
begin
    RUN_MODE = "smoke"
    SEED = 0
    budgets = (
        smoke = (steps = 45, n_train = 160, n_val = 80, patience = 12),
        teaching = (steps = 300, n_train = 800, n_val = 200, patience = 30),
        production = (steps = 1_500, n_train = 2_000, n_val = 500, patience = 60),
    )
    hp = run_mode_budget(RUN_MODE; budgets)
    rng = rng_from_seed(SEED)
end

# ╔═╡ 6494506f-23de-9d1f-cacf-cff146579f2b
md"""
### 1. Synthetic data, model, and "callbacks"

The Julia target is \$f(x) = \sin(3x_1) + \cos(2x_2)\$ on \$[-1,1]^2\$ (the Python ground truth uses \$\sin(x)\,e^{y}\$ on \$[-2,2]^2\$); the training targets get a little Gaussian noise while the validation set is clean.

The model is a two-hidden-layer MLP (`make_mlp(2, (24, 24), 1)`, `tanh`) trained with `Optimisers.Adam` under an `mse_loss`. Where Keras attaches a `ModelCheckpoint` callback (save weights each epoch) and a `TensorBoard` callback (stream scalar/histogram summaries), the Julia training loop below plays both roles explicitly.
"""

# ╔═╡ 44444444-0205-4444-8444-444444444444
begin
    signal(x) = sin.(3 .* x[1:1, :]) .+ cos.(2 .* x[2:2, :])
    x_train = 2 .* rand(rng, 2, hp.n_train) .- 1
    y_train = signal(x_train) .+ 0.08 .* randn(rng, 1, hp.n_train)
    x_val = 2 .* rand(rng, 2, hp.n_val) .- 1
    y_val = signal(x_val)

    model = make_mlp(2, (24, 24), 1; activation = NNlib.tanh)
    state = setup_training(rng_from_seed(SEED; offset = 1), model, Optimisers.Adam(0.006); parameter_type = Float64)
    loss_fn(model, ps, st, batch) = begin
        prediction, st_new = model(batch.x, ps, st)
        return mse_loss(prediction, batch.y), st_new
    end
end

# ╔═╡ 27c02613-1a63-be9b-6052-5c82fe4f0fb0
md"""
### 2. Training with metric logging and early stopping

We run up to `hp.steps` gradient steps. Each step records the training loss, a fresh **validation** loss, and the **gradient norm** into `history` (assembled into a `metrics_table` `DataFrame` — the plain-data stand-in for TensorBoard's scalar logs). We also keep a `best_checkpoint` (the lowest validation loss seen so far) and an early-stopping counter: if the validation loss fails to improve for `patience` steps, training stops early. This is the same information TensorBoard would surface — train vs. validation curves that should track each other (a widening gap signals over-fitting), plus gradient norms to catch exploding or vanishing updates.
"""

# ╔═╡ 55555555-0205-4555-8555-555555555555
begin
    train_batch = (x = x_train, y = y_train)
    val_batch = (x = x_val, y = y_val)
    history = NamedTuple[]
    best_checkpoint = Ref((step = 0, validation_loss = Inf))
    stale_steps = Ref(0)

    for _ in 1:hp.steps
        metrics = train_step!(state, loss_fn, train_batch; max_grad_norm = 10.0)
        val_loss = loss_value(state, loss_fn, val_batch)
        append_metric!(history; step = metrics.step, train_loss = metrics.loss, validation_loss = val_loss, grad_norm = metrics.grad_norm)
        if val_loss < best_checkpoint[].validation_loss - 1e-8
            best_checkpoint[] = (step = metrics.step, validation_loss = val_loss)
            stale_steps[] = 0
        else
            stale_steps[] += 1
        end
        stale_steps[] >= hp.patience && break
    end

    metrics_table = DataFrame(history)
    best = best_checkpoint[]
end

# ╔═╡ 1d6a6446-b105-9b19-236e-1bfee1407e16
md"""
### 3. Inspecting the run

We plot the recorded train and validation MSE (log scale) against the training step, marking the best-validation step with a dashed line. The two curves should track each other; a wide, growing gap signals over-fitting.

The full Python notebook also pairs this loss curve with a second diagnostic — an actual-vs-predicted scatter of the trained network's outputs against the ground-truth signal on a fresh 2000-point test grid, read against the `y = x` reference line to confirm fit quality. This preview keeps only the loss-curve panel.

> **In the Python notebook** the same metrics are inspected through the TensorBoard dashboard — inline via the `%tensorboard` magic (the recommended path on Nuvolos), or from a terminal with `tensorboard --logdir=./logs --port 6006` — offering **Scalars** (epoch loss/MAE, train vs. validation), **Histograms** (per-layer weight/bias distributions, to spot dead or exploding neurons), and **Graphs** (the static computation graph). Here the `metrics_table` `DataFrame` holds the same scalar series, so we read them straight off a Makie plot.
"""

# ╔═╡ 66666666-0205-4666-8666-666666666666
begin
    fig = Figure(size = figure_size(RUN_MODE))
    ax = Axis(fig[1, 1], xlabel = "step", ylabel = "MSE", yscale = log10)
    lines!(ax, metrics_table.step, metrics_table.train_loss; label = "train", color = :dodgerblue3, linewidth = 2)
    lines!(ax, metrics_table.step, metrics_table.validation_loss; label = "validation", color = :darkorange, linewidth = 2)
    vlines!(ax, [best.step]; color = :gray45, linestyle = :dash, label = "best")
    axislegend(ax; position = :rt)
    fig
end

# ╔═╡ b362d78e-74dd-e4ec-92f1-31ca07bde78f
md"""
### Recap

- Instrumenting a training loop needs only a place to record per-step metrics and a rule for keeping the best checkpoint. Keras spells this with two callbacks (`TensorBoard`, `ModelCheckpoint`); the Lux preview does it with a metrics `DataFrame`, best-checkpoint tracking, and patience-based early stopping — no logging dependency, though `TensorBoardLogger.jl` could stream the same scalars to a TensorBoard dashboard.
- This monitoring becomes invaluable from Day 2 on, where we train networks to solve dynamic stochastic models: watching the loss components (Euler residuals, market clearing) live is the only way to catch a stuck training run early.

The cell below returns a machine-checkable summary of the logged run.
"""

# ╔═╡ 77777777-0205-4777-8777-777777777777
(
    rows_logged = nrow(metrics_table),
    initial_train_loss = metrics_table.train_loss[1],
    final_train_loss = metrics_table.train_loss[end],
    best_validation = best,
    stopped_early = length(history) < hp.steps,
    finite_losses = finite_loss(metrics_table.validation_loss),
)

# ╔═╡ Cell order:
# ╟─11111111-0205-4111-8111-111111111111
# ╟─d9a6b4ec-7d09-cf74-0694-0280171c287f
# ╟─94a192dd-50df-b891-9031-0ca2c693a6ea
# ╠═22222222-0205-4222-8222-222222222222
# ╠═33333333-0205-4333-8333-333333333333
# ╟─6494506f-23de-9d1f-cacf-cff146579f2b
# ╠═44444444-0205-4444-8444-444444444444
# ╟─27c02613-1a63-be9b-6052-5c82fe4f0fb0
# ╠═55555555-0205-4555-8555-555555555555
# ╟─1d6a6446-b105-9b19-236e-1bfee1407e16
# ╠═66666666-0205-4666-8666-666666666666
# ╟─b362d78e-74dd-e4ec-92f1-31ca07bde78f
# ╠═77777777-0205-4777-8777-777777777777
