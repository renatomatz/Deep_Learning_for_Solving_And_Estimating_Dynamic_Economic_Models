### A Pluto.jl notebook ###
# v0.20.0

using Markdown
using InteractiveUtils

# ╔═╡ 11111111-0204-4111-8111-111111111111
md"""
# Lecture 02, Notebook 04: Gentle Deep Networks in Lux

The Python notebook teaches Keras `Sequential` models. The Julia version uses
`Lux.Chain` directly and keeps the parameter/state call explicit for both a
regression task and a small synthetic classification task.
"""

# ╔═╡ 28656216-d9ec-966f-1ec3-e919c1a0a87f
md"""
## Lecture 02, Notebook 04: Deep Feedforward Networks and Backpropagation

**Course:** Deep Learning for Solving and Estimating Dynamic Models in Economics and Finance
**Script reference:** §1.4–1.8 (Deep feedforward networks, backpropagation, initialization, regularization)
**Notebook role:** core
**Author:** Simon Scheidegger

*Julia/Lux/Pluto preview of* `lectures/lecture_02_intro_deep_learning/code/lecture_02_04_Gentle_DNN.ipynb`.
"""

# ╔═╡ d4f179e4-967b-5ec0-f416-4813766d2288
md"""
## Approximating functions with deep neural networks

Two gentle supervised-learning examples with fully connected deep neural networks:

1. **Regression** — train a deep network to learn an analytical 2-D function.
2. **Classification** — train a network with a softmax cross-entropy head.

The Python ground truth builds these with TensorFlow/Keras `Sequential` models and tours the surrounding API (pre-implemented losses such as MSE, MAE, and cross-entropy; weight initialization; early stopping; dropout; batch normalization). This Julia preview builds the networks directly with `Lux.Chain` via `make_mlp`, keeping the parameter/state call explicit — `prediction, st_new = model(x, ps, st)` — and trains with `Optimisers.jl`.
"""

# ╔═╡ 22222222-0204-4222-8222-222222222222
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

# ╔═╡ 33333333-0204-4333-8333-333333333333
begin
    RUN_MODE = "smoke"
    SEED = 0
    budgets = (
        smoke = (steps = 35, n_train = 160, n_test = 80),
        teaching = (steps = 250, n_train = 600, n_test = 240),
        production = (steps = 1_000, n_train = 2_000, n_test = 500),
    )
    hp = run_mode_budget(RUN_MODE; budgets)
    rng = rng_from_seed(SEED)
end

# ╔═╡ b0d452bf-fc5e-9da7-b9d3-28ddad115e66
md"""
---
### 1. A simple regression example

As a first example we approximate a 2-D analytical function on the unit cube \$[-1,1]^2\$. The Julia target surface is \$f(x) = \sin(2x_1) + \tfrac{1}{2}x_2^2\$, sampled at random points with a little Gaussian noise added to the training targets (the test targets are clean).

Where the Python notebook stacks Keras `Dense` layers into a `Sequential` model and `compile`s it with an optimizer, loss, and metrics, the Julia preview builds the network directly with `make_mlp(2, (24, 24), 1; activation = relu)` and trains it with `Optimisers.Adam` through the explicit `model(x, ps, st)` loss pattern. The loss is `mse_loss`; we track train and test loss across `hp.steps` gradient steps.

> **The full Python notebook also tours the Keras `Sequential` API** on this regression model: `model.summary()` and `.layers` introspection, the incremental `.add()` builder, `model.evaluate` test accuracy, per-example `.predict`, saving/restoring weights to a TensorFlow checkpoint, **early stopping** callbacks (monitor the loss, stop once it plateaus within `min_delta`/`patience`), **dropout** layers (drop 20–50% of units to regularise), and **batch normalization** (keep per-batch activations near mean 0, std 1). These are framework-specific conveniences; the Lux preview keeps one explicit training loop and leaves the API tour to the Python ground truth.
"""

# ╔═╡ 44444444-0204-4444-8444-444444444444
begin
    target_surface(x) = sin.(2 .* x[1:1, :]) .+ 0.5 .* x[2:2, :] .^ 2
    x_train = 2 .* rand(rng, 2, hp.n_train) .- 1
    y_train = target_surface(x_train) .+ 0.05 .* randn(rng, 1, hp.n_train)
    x_test = 2 .* rand(rng, 2, hp.n_test) .- 1
    y_test = target_surface(x_test)

    regression_model = make_mlp(2, (24, 24), 1; activation = NNlib.relu)
    regression_state = setup_training(rng_from_seed(SEED; offset = 1), regression_model, Optimisers.Adam(0.01); parameter_type = Float64)

    regression_loss(model, ps, st, batch) = begin
        prediction, st_new = model(batch.x, ps, st)
        return mse_loss(prediction, batch.y), st_new
    end

    regression_batch = (x = x_train, y = y_train)
    regression_initial = loss_value(regression_state, regression_loss, regression_batch)
    regression_history = NamedTuple[]
    for _ in 1:hp.steps
        metrics = train_step!(regression_state, regression_loss, regression_batch; max_grad_norm = 10.0)
        append_metric!(regression_history; step = metrics.step, loss = metrics.loss)
    end
    regression_test = loss_value(regression_state, regression_loss, (x = x_test, y = y_test))
end

# ╔═╡ 779f7491-f2a9-875d-b3b4-71a55e34b56a
md"""
---
### 2. Classification example

The Python notebook trains a classifier on the **Fashion-MNIST** dataset — 60,000 28×28 grayscale images across 10 clothing classes, with pixel values rescaled from \$[0,255]\$ to \$[0,1]\$ — builds a `Sequential` model ending in a softmax, and inspects per-image prediction confidences (for a given test image, the model's top class is an ankle boot).

The Julia preview keeps the *idea* — softmax classification trained with **cross-entropy** — on a small synthetic 2-D binary problem instead of Fashion-MNIST, so it runs instantly and stays self-contained. `onehot` builds the target matrix, and `cross_entropy_loss` implements the numerically stabilised softmax cross-entropy

\$\$J = -\frac{1}{N}\sum_{n}\sum_{k} y_{k}\,\log \hat{p}_{k}\$\$

using the max-subtraction log-sum-exp trick for stability. The class label is a nonlinear function of the 2-D input, so the `tanh` MLP must bend a curved decision boundary; we report the classification accuracy after training.
"""

# ╔═╡ 55555555-0204-4555-8555-555555555555
begin
    function onehot(classes, n_classes)
        y = zeros(Float64, n_classes, length(classes))
        for (i, cls) in pairs(classes)
            y[cls, i] = 1.0
        end
        return y
    end

    function cross_entropy_loss(logits, target)
        shifted = logits .- maximum(logits; dims = 1)
        log_probs = shifted .- log.(sum(exp.(shifted); dims = 1))
        return -mean(sum(target .* log_probs; dims = 1))
    end

    x_class = 2 .* rand(rng, 2, hp.n_train) .- 1
    class_ids = ifelse.(vec(sum(abs2, x_class; dims = 1) .+ 0.35 .* x_class[1:1, :] .> 0.65), 2, 1)
    y_class = onehot(class_ids, 2)

    class_model = make_mlp(2, (18, 18), 2; activation = NNlib.tanh)
    class_state = setup_training(rng_from_seed(SEED; offset = 2), class_model, Optimisers.Adam(0.01); parameter_type = Float64)
    class_loss(model, ps, st, batch) = begin
        logits, st_new = model(batch.x, ps, st)
        return cross_entropy_loss(logits, batch.y), st_new
    end

    class_batch = (x = x_class, y = y_class)
    class_initial = loss_value(class_state, class_loss, class_batch)
    for _ in 1:hp.steps
        train_step!(class_state, class_loss, class_batch; max_grad_norm = 10.0)
    end
    logits, _ = class_state.model(x_class, class_state.ps, class_state.st)
    predicted_class = vec(map(x -> x[1] > x[2] ? 1 : 2, eachcol(logits)))
    class_accuracy = mean(predicted_class .== class_ids)
end

# ╔═╡ 8b2bfd88-53ba-8a58-3c00-35308298bab5
md"""
---
### Visualising the regression fit

We visualise the trained regression network: the noisy training targets against the MLP's prediction surface evaluated on a dense grid over \$[-1,1]^2\$ (plotted against the first input coordinate \$x_1\$).
"""

# ╔═╡ 66666666-0204-4666-8666-666666666666
begin
    grid_x = collect(range(-1, 1; length = 80))
    grid = reduce(hcat, ([a, b] for a in grid_x for b in grid_x))
    pred_grid, _ = regression_state.model(grid, regression_state.ps, regression_state.st)
    fig = Figure(size = figure_size(RUN_MODE))
    ax = Axis(fig[1, 1], xlabel = "x1", ylabel = "predicted y")
    scatter!(ax, vec(x_train[1:1, :]), vec(y_train); color = (:gray35, 0.45), label = "train")
    scatter!(ax, vec(grid[1:1, :]), vec(pred_grid); color = (:dodgerblue3, 0.25), markersize = 5, label = "Lux MLP")
    axislegend(ax; position = :lt)
    fig
end

# ╔═╡ 7e46e90e-4278-5417-be29-22d5c0bd820c
md"""
### Takeaway

- **Regression:** a small Lux MLP (`make_mlp` + `Optimisers.Adam`) learns a smooth 2-D analytical surface end-to-end, trained through the explicit `model(x, ps, st)` loss pattern rather than a Keras `compile`/`fit` wrapper.
- **Classification:** the same machinery — swapping `mse_loss` for a stabilised softmax **cross-entropy** — learns a curved decision boundary on a synthetic 2-D problem, the pedagogical stand-in for the Python notebook's Fashion-MNIST classifier.
- Keras conveniences (dropout, batch normalization, early stopping, checkpointing) are surveyed in the Python ground truth; in Lux they would be composed explicitly into the `Chain` and the training loop.

The cell below returns a machine-checkable summary of both tasks for this notebook's run.
"""

# ╔═╡ 77777777-0204-4777-8777-777777777777
(
    regression_initial_loss = regression_initial,
    regression_test_loss = regression_test,
    regression_steps = length(regression_history),
    classification_initial_loss = class_initial,
    classification_accuracy = class_accuracy,
)

# ╔═╡ Cell order:
# ╟─11111111-0204-4111-8111-111111111111
# ╟─28656216-d9ec-966f-1ec3-e919c1a0a87f
# ╟─d4f179e4-967b-5ec0-f416-4813766d2288
# ╠═22222222-0204-4222-8222-222222222222
# ╠═33333333-0204-4333-8333-333333333333
# ╟─b0d452bf-fc5e-9da7-b9d3-28ddad115e66
# ╠═44444444-0204-4444-8444-444444444444
# ╟─779f7491-f2a9-875d-b3b4-71a55e34b56a
# ╠═55555555-0204-4555-8555-555555555555
# ╟─8b2bfd88-53ba-8a58-3c00-35308298bab5
# ╠═66666666-0204-4666-8666-666666666666
# ╟─7e46e90e-4278-5417-be29-22d5c0bd820c
# ╠═77777777-0204-4777-8777-777777777777
