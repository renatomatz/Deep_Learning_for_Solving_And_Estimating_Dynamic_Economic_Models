### A Pluto.jl notebook ###
# v0.20.0

using Markdown
using InteractiveUtils

# ╔═╡ 11111111-0203-4111-8111-111111111111
md"""
# Lecture 02, Notebook 03: Double Descent in Julia

This Pluto translation keeps the statistical experiment from the Python
notebook: random Fourier features expand a fixed training set through the
interpolation threshold, and the minimum-norm least-squares fit produces the
classic double-descent shape.
"""

# ╔═╡ 6c49df37-79ab-d482-3873-6efc259fd818
md"""
## Lecture 02, Notebook 03: Double Descent and Generalization

**Course:** Deep Learning for Solving and Estimating Dynamic Models in Economics and Finance
**Script reference:** §1.9 (Generalization: double descent, overparameterization, random Fourier features)
**Notebook role:** core
**Author:** Simon Scheidegger

*Julia/Lux/Pluto preview of* `lectures/lecture_02_intro_deep_learning/code/lecture_02_03_Double_Descent.ipynb`.
"""

# ╔═╡ b248be9b-532b-002e-bbfb-1957a20912e7
md"""
## The Double-Descent Phenomenon

> *"Bigger models are better models — but only after they get worse first."*

Classical statistics predicts a clean **U-shaped** test-error curve as model complexity grows: first the error falls (less bias), then it rises again (more variance) — the textbook **bias-variance trade-off**.

Recent work reveals a striking extension. When the number of parameters \$p\$ exceeds the number of training samples \$n\$, test error can *decrease again* — a phenomenon called **double descent** (Belkin et al., 2019; Nakkiran et al., 2019).

This notebook demonstrates double descent with a transparent, minimal example: random Fourier features expand a fixed training set through the interpolation threshold, and the minimum-norm least-squares fit reproduces the classic double-descent shape.
"""

# ╔═╡ 22222222-0203-4222-8222-222222222222
begin
    import Pkg
    Pkg.activate(joinpath(@__DIR__, "..", "..", "..", "julia"))

    using CairoMakie
    using DLEFJulia
    using LinearAlgebra
    using Statistics
end

# ╔═╡ 33333333-0203-4333-8333-333333333333
begin
    RUN_MODE = "smoke"
    SEED = 0
    budgets = (
        smoke = (n_train = 35, n_test = 180, max_p = 95),
        teaching = (n_train = 50, n_test = 500, max_p = 300),
        production = (n_train = 50, n_test = 1_000, max_p = 500),
    )
    hp = run_mode_budget(RUN_MODE; budgets)
    rng = rng_from_seed(SEED)
end

# ╔═╡ b273fbc5-d03d-c3c7-0db2-0f4d90cb5f30
md"""
---
### 1. Data generation and random Fourier features

We create a regression task with \$d = 5\$ input features whose target depends nonlinearly on only a few coordinates, plus a little Gaussian noise:

\$\$y = \sin(x_1) + \tfrac{1}{2}\cos(\tfrac{3}{2}x_2) + \tfrac{1}{4}x_3 x_4 + \varepsilon.\$\$

(The Python ground truth uses the two-coordinate target \$\sin(x_1) + \tfrac{1}{2}\cos(x_2) + \varepsilon\$; the Julia preview adds a mild interaction term.) The training/test sizes and the maximum feature count come from `run_mode_budget`: the smoke run uses \$n = 35\$ train points and sweeps up to \$p = 95\$ features, while the teaching/production budgets restore the Python-scale \$n = 50\$ with \$p\$ up to \$300\$–\$500\$.

To sweep model complexity continuously we use **Random Fourier Features** (Rahimi & Recht, 2007). Each input \$\mathbf{x} \in \mathbb{R}^d\$ is mapped to a \$p\$-dimensional feature vector

\$\$\phi_j(\mathbf{x}) = \sqrt{\tfrac{2}{p}}\,\cos\!\bigl(\boldsymbol{\omega}_j^\top \mathbf{x} + b_j\bigr), \qquad j = 1,\dots,p,\$\$

with \$\boldsymbol{\omega}_j \sim \mathcal{N}(\mathbf{0}, \mathbf{I}_d)\$ and \$b_j \sim \mathrm{Uniform}(0, 2\pi)\$. The \$\sqrt{2/p}\$ prefactor keeps the feature scale constant as \$p\$ grows. `rff_matrix(X, p)` builds the feature matrix \$\Phi\$ for a given \$p\$.
"""

# ╔═╡ 44444444-0203-4444-8444-444444444444
begin
    d = 5
    noise_sigma = 0.15
    true_fn(X) = sin.(X[:, 1]) .+ 0.5 .* cos.(1.5 .* X[:, 2]) .+ 0.25 .* X[:, 3] .* X[:, 4]

    X_train = randn(rng, hp.n_train, d)
    X_test = randn(rng, hp.n_test, d)
    y_train = true_fn(X_train) .+ noise_sigma .* randn(rng, hp.n_train)
    y_test = true_fn(X_test)

    omega = randn(rng, d, hp.max_p)
    phase = 2pi .* rand(rng, hp.max_p)

    function rff_matrix(X, p)
        raw = X * omega[:, 1:p] .+ reshape(phase[1:p], 1, p)
        return sqrt(2 / p) .* cos.(raw)
    end
end

# ╔═╡ 37c739d5-e598-82fd-7697-3839042b1afa
md"""
---
### 2. Sweeping model complexity and the minimum-norm solution

For every \$p = 1, 2, \dots\$ we build \$\Phi\$, solve the linear system \$\hat{y} = \Phi\,\boldsymbol{\theta}\$, and record training MSE, test MSE, and the weight norm \$\lVert\boldsymbol{\theta}\rVert_2\$. Julia's `\` (like NumPy's `lstsq`) returns:

| Regime | What `\` computes |
|:---|:---|
| \$p < n\$ | ordinary least squares (unique) |
| \$p \ge n\$ | **minimum-norm** solution among all interpolating \$\boldsymbol{\theta}\$ |

Near the interpolation threshold \$p \approx n\$ the model is forced to use enormous weights to fit the noisy data exactly; once \$p \gg n\$ the minimum-norm constraint selects a solution with much smaller weights — a smoother interpolant that generalises better. The cell also records the interpolation-threshold index and the best-test-MSE index.
"""

# ╔═╡ 55555555-0203-4555-8555-555555555555
begin
    p_values = collect(1:hp.max_p)
    train_mse = similar(float.(p_values))
    test_mse = similar(float.(p_values))
    weight_norm = similar(float.(p_values))

    for (i, p) in enumerate(p_values)
        Phi_train = rff_matrix(X_train, p)
        Phi_test = rff_matrix(X_test, p)
        theta = Phi_train \ y_train
        train_mse[i] = mean(abs2, Phi_train * theta .- y_train)
        test_mse[i] = mean(abs2, Phi_test * theta .- y_test)
        weight_norm[i] = norm(theta)
    end

    threshold_index = argmin(abs.(p_values .- hp.n_train))
    overfit_index = argmin(test_mse)
end

# ╔═╡ 1ed0260f-ddcd-6cfc-e4fb-f7c90c5ca4d0
md"""
---
### 3. The double-descent curve

We plot training and test MSE (log scale) against the number of random features \$p\$, with a dashed line at the interpolation threshold \$p = n\$. The classical U-shape appears for \$p < n\$; test error peaks at \$p \approx n\$, then *descends again* for \$p \gg n\$.
"""

# ╔═╡ 66666666-0203-4666-8666-666666666666
begin
    fig = Figure(size = figure_size(RUN_MODE))
    ax = Axis(fig[1, 1], xlabel = "number of random features p", ylabel = "MSE", yscale = log10)
    lines!(ax, p_values, train_mse; label = "train", linewidth = 2, color = :dodgerblue3)
    lines!(ax, p_values, test_mse; label = "test", linewidth = 2, color = :darkorange)
    vlines!(ax, [hp.n_train]; color = :gray45, linestyle = :dash, label = "p = n")
    axislegend(ax; position = :rt)
    fig
end

# ╔═╡ 8b6352a4-e26f-1e07-4876-3e1843642c9b
md"""
---
### Discussion: the three regimes

| Regime | Description |
|:---|:---|
| \$p \ll n\$ | **Under-parameterised.** The model is too simple; high bias, low variance. |
| \$p \approx n\$ | **Interpolation threshold.** The model can *just barely* fit the noisy training data. The solution is extremely sensitive to the noise — maximum variance, peak test error, and enormous weight norm. |
| \$p \gg n\$ | **Over-parameterised.** Many weight vectors interpolate the data; the minimum-norm solution (\$\min\lVert\boldsymbol{\theta}\rVert\$) is the *smoothest* interpolant. Test error and weight norm both decrease. |

**Why does this matter for deep learning?** Modern neural networks are massively over-parameterised (\$p \gg n\$) yet generalise remarkably well. Double descent helps explain why:

- **Overparameterisation is not the enemy.** Once past the interpolation threshold, adding more parameters *helps*.
- **Implicit regularisation** from SGD, weight decay, and early stopping selects small-norm solutions — analogous to the minimum-norm solution computed here.
- The dangerous zone is \$p \approx n\$. In practice, practitioners either stay well below it (classical regime) or well above it (modern regime).

**Key references**

- M. Belkin, D. Hsu, S. Ma, S. Mandal (2019). *Reconciling modern machine-learning practice and the classical bias-variance trade-off.* PNAS 116(32), 15849-15854.
- P. Nakkiran, G. Kaplun, Y. Bansal, T. Yang, B. Barak, I. Sutskever (2019). *Deep double descent: where bigger models and more data can hurt.* arXiv:1912.02292.
- T. Hastie, A. Montanari, S. Rosset, R. J. Tibshirani (2022). *Surprises in high-dimensional ridgeless least squares interpolation.* Annals of Statistics 50(2), 949-986.

The cell below returns machine-checkable diagnostics for this notebook's run.
"""

# ╔═╡ 77777777-0203-4777-8777-777777777777
(
    interpolation_threshold = hp.n_train,
    test_mse_at_threshold = test_mse[threshold_index],
    best_p = p_values[overfit_index],
    best_test_mse = test_mse[overfit_index],
    final_test_mse = test_mse[end],
    finite_share = finite_share(test_mse),
)

# ╔═╡ Cell order:
# ╟─11111111-0203-4111-8111-111111111111
# ╟─6c49df37-79ab-d482-3873-6efc259fd818
# ╟─b248be9b-532b-002e-bbfb-1957a20912e7
# ╠═22222222-0203-4222-8222-222222222222
# ╠═33333333-0203-4333-8333-333333333333
# ╟─b273fbc5-d03d-c3c7-0db2-0f4d90cb5f30
# ╠═44444444-0203-4444-8444-444444444444
# ╟─37c739d5-e598-82fd-7697-3839042b1afa
# ╠═55555555-0203-4555-8555-555555555555
# ╟─1ed0260f-ddcd-6cfc-e4fb-f7c90c5ca4d0
# ╠═66666666-0203-4666-8666-666666666666
# ╟─8b6352a4-e26f-1e07-4876-3e1843642c9b
# ╠═77777777-0203-4777-8777-777777777777
