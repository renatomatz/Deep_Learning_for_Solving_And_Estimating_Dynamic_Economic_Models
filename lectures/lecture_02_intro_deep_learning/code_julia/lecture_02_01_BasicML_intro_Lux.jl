### A Pluto.jl notebook ###
# v0.20.0

using Markdown
using InteractiveUtils

# ╔═╡ 11111111-0201-4111-8111-111111111111
md"""
# Lecture 02, Notebook 01: Lux-Native Foundations

This Pluto translation keeps the classical supervised-learning idea from the
Python notebook, then crosses into Lux with feature-by-batch arrays and an
explicit `model(x, ps, st)` call.
"""

# ╔═╡ 9bece5b6-8158-f5e0-5175-2f004ac7f0b8
md"""
## Lecture 02, Notebook 01: ML Foundations and Supervised Learning

**Course:** Deep Learning for Solving and Estimating Dynamic Models in Economics and Finance
**Script reference:** §1.1–1.3 (ML foundations: supervised learning, loss functions, clustering)
**Notebook role:** core
**Author:** Simon Scheidegger

*Julia/Lux/Pluto preview of* `lectures/lecture_02_intro_deep_learning/code/lecture_02_01_BasicML_intro.ipynb`.
"""

# ╔═╡ 19b9cb6a-0a71-1f99-23e7-2e7b2939c103
md"""
## Tutorial: Linear Regression, Classification, Unsupervised Learning, and Loss Functions

The Python ground-truth notebook illustrates four core machine-learning ideas:

1. **Linear Regression:** fit a linear model to synthetic data and measure prediction quality with MSE and MAE.
2. **Linear Classification:** train a logistic-regression classifier and visualise the decision boundary.
3. **Unsupervised Learning:** apply k-means clustering to discover natural groupings in unlabeled data.
4. **Loss Functions:** plot the standard losses for regression (squared error, absolute error) and classification (binary cross-entropy).

This compact Lux/Pluto preview focuses on the **linear-regression** example, contrasting an ordinary-least-squares baseline with a small Lux MLP, and reproduces the regression/classification **loss gallery** as prose. Classification and clustering remain in the Python ground truth above.
"""

# ╔═╡ 22222222-0201-4222-8222-222222222222
begin
    import Pkg
    Pkg.activate(joinpath(@__DIR__, "..", "..", "..", "julia"))

    using CairoMakie
    using DLEFJulia
    using LinearAlgebra
    using Lux
    using NNlib
    using Optimisers
    using Statistics
end

# ╔═╡ 33333333-0201-4333-8333-333333333333
begin
    RUN_MODE = "smoke"
    SEED = 0
    rng = rng_from_seed(SEED)
    budget = run_mode_budget(RUN_MODE)
end

# ╔═╡ 44444444-0201-4444-8444-444444444444
md"""
---
### 1. Linear Regression

**Goal:** predict a continuous target \$y \in \mathbb{R}\$ from a single feature \$x\$.

We generate synthetic data from the true model \$y = a + b\,x + \varepsilon\$, \$\varepsilon \sim \mathcal{N}(0,\sigma^2)\$, fit an ordinary-least-squares baseline, and compare it with a small Lux MLP trained by gradient descent.

Rows are observations here; we convert to feature-by-batch arrays only at the Lux boundary, then call the model with the explicit `y, st = model(x, ps, st)` pattern.
"""

# ╔═╡ 55555555-0201-4555-8555-555555555555
begin
    n = 80
    x_scalar = collect(range(-2.0, 2.0; length = n))
    noise = 0.15 .* randn(rng, n)
    y_scalar = 1.0 .+ 2.5 .* x_scalar .+ noise

    design = hcat(ones(n), x_scalar)
    ols_coef = design \ y_scalar
    y_ols = design * ols_coef
    ols_metrics = (
        intercept = ols_coef[1],
        slope = ols_coef[2],
        mse = mse_loss(y_ols, y_scalar),
        mae = mae_loss(y_ols, y_scalar),
    )
end

# ╔═╡ 66666666-0201-4666-8666-666666666666
begin
    features = to_feature_batch(reshape(x_scalar, :, 1))
    targets = reshape(y_scalar, 1, :)
    assert_matching_batch(features, targets)

    model = make_mlp(1, (16, 16), 1; activation = NNlib.tanh)
    ps, st = setup_model(rng_from_seed(SEED; offset = 1), model; parameter_type = Float64)
    train_state = setup_training(model, ps, st, Optimisers.Adam(0.02))

    supervised_loss(model, ps, st, batch) = begin
        prediction, st_new = model(batch.x, ps, st)
        return mse_loss(prediction, batch.y), st_new
    end

    batch = (x = features, y = targets)
    initial_loss = loss_value(train_state, supervised_loss, batch)
    history = NamedTuple[]
    for _ in 1:budget.steps
        metrics = train_step!(train_state, supervised_loss, batch; max_grad_norm = 10.0)
        append_metric!(history; step = metrics.step, loss = metrics.loss)
    end
    final_loss = loss_value(train_state, supervised_loss, batch)
end

# ╔═╡ 77777777-0201-4777-8777-777777777777
begin
    y_lux, _ = train_state.model(features, train_state.ps, train_state.st)
    fig = Figure(size = figure_size(RUN_MODE))
    ax = Axis(fig[1, 1], xlabel = "x", ylabel = "y")
    scatter!(ax, x_scalar, y_scalar; color = (:gray35, 0.55), label = "data")
    lines!(ax, x_scalar, y_ols; color = :dodgerblue3, linewidth = 3, label = "OLS")
    lines!(ax, x_scalar, vec(y_lux); color = :darkorange, linewidth = 3, label = "Lux MLP")
    axislegend(ax; position = :lt)
    fig
end

# ╔═╡ d950cd1b-c81e-3e0f-f712-20cdfae3316c
md"""
---
### Beyond regression: classification, clustering, and the loss gallery

The full Python notebook continues past regression:

- **Linear classification.** A logistic-regression model assigns each input \$\mathbf{x}\$ to one of \$K\$ classes, and its decision boundary is visualised.
- **Unsupervised learning.** K-means partitions \$n\$ observations into \$k\$ clusters by minimising the within-cluster sum of squared distances to the centroid — no labels required.

#### Loss functions

Every supervised-learning algorithm follows the same recipe:

1. Choose a model \$\hat{y} = h(\mathbf{x};\boldsymbol{\theta})\$
2. Define a loss \$J(\boldsymbol{\theta})\$
3. Optimise \$\boldsymbol{\theta}^{*} = \arg\min J(\boldsymbol{\theta})\$

Regression losses (used above via `mse_loss` and `mae_loss`):

| Loss | Per-sample formula |
|---|---|
| Squared error (MSE) | \$(y - \hat{y})^{2}\$ |
| Absolute error (MAE) | \$\lvert y - \hat{y}\rvert\$ |

Classification loss — binary cross-entropy:

\$\$
J = -\Bigl[y\,\log\hat{y} \;+\; (1-y)\,\log(1-\hat{y})\Bigr],
\qquad y\in\{0,1\},\;\hat{y}\in(0,1)
\$\$

This is the standard loss used in logistic regression and neural-network classifiers.
"""

# ╔═╡ 3722944d-0541-3b09-ff09-2b005798149f
md"""
---
### Conclusion

In this Lux/Pluto preview we:

- **Linear Regression:** fitted an OLS baseline and a small Lux MLP to synthetic data, measuring quality with MSE and MAE.
- **Classification & Clustering:** reviewed the logistic-regression and k-means examples carried in full by the Python ground truth.
- **Loss Functions:** summarised the squared-error, absolute-error, and binary cross-entropy losses.

These building blocks — models, losses, and optimisation — reappear throughout the course as we move to deep neural networks. The cell below returns a machine-checkable summary of this notebook's run.
"""

# ╔═╡ 88888888-0201-4888-8888-888888888888
(
    ols = ols_metrics,
    lux_initial_loss = initial_loss,
    lux_final_loss = final_loss,
    steps = length(history),
)

# ╔═╡ Cell order:
# ╟─11111111-0201-4111-8111-111111111111
# ╟─9bece5b6-8158-f5e0-5175-2f004ac7f0b8
# ╟─19b9cb6a-0a71-1f99-23e7-2e7b2939c103
# ╠═22222222-0201-4222-8222-222222222222
# ╠═33333333-0201-4333-8333-333333333333
# ╟─44444444-0201-4444-8444-444444444444
# ╠═55555555-0201-4555-8555-555555555555
# ╠═66666666-0201-4666-8666-666666666666
# ╠═77777777-0201-4777-8777-777777777777
# ╟─d950cd1b-c81e-3e0f-f712-20cdfae3316c
# ╟─3722944d-0541-3b09-ff09-2b005798149f
# ╠═88888888-0201-4888-8888-888888888888
