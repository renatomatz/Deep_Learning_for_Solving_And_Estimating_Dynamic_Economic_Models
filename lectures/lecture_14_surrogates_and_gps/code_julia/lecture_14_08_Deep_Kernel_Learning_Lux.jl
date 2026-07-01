### A Pluto.jl notebook ###
# v0.20.0

using Markdown
using InteractiveUtils

# ╔═╡ 11111111-1408-4111-8111-111111111111
md"""
# Lecture 14, Notebook 08: Deep-Kernel Learning Preview in Lux

This smoke translation compares a raw-input GP with a GP fit on a learned-style
one-dimensional feature. Later production work can train the feature map jointly;
the notebook boundary and scoring logic are already in Julia.
"""

# ╔═╡ a7c6bbb4-8836-4e57-9066-bb07cf669a26
md"""
## Lecture 14, Notebook 08: Deep kernel learning — a neural feature extractor with a GP head

**Course:** Deep Learning for Solving and Estimating Dynamic Models in Economics and Finance
**Script reference:** §9.7 (Deep kernel learning — neural feature extractor + GP head on a hidden-regime problem)
**Notebook role:** core
**Author:** Simon Scheidegger

*Julia/Lux/Pluto preview of* `lectures/lecture_14_surrogates_and_gps/code/lecture_14_08_Deep_Kernel_Learning.ipynb`.
"""

# ╔═╡ 95f023c6-e6b2-23cb-4417-05f718d5e2bc
md"""
### Overview

The organizing idea of a **deep kernel** is simple:

1. A plain Gaussian process (GP) assumes that **distance in the raw input space** is already meaningful.
2. A deep kernel first passes the input through a feature map \$\phi_\theta(x)\$.
3. The GP is then applied to that representation instead of to the raw input:

\$\$k_{\text{DKL}}(x, x') = k_{\text{base}}\!\bigl(\phi_\theta(x),\,\phi_\theta(x')\bigr).\$\$

Once the representation is good, even a simple RBF kernel performs well — the gain comes from **changing the input geometry the GP sees**, not from a fancier GP.

> **Scope of this Julia preview.** The in-house Cholesky GP (`fit_cholesky_gp` / `gp_predict`) is compared on **raw inputs** versus a **known (teacher) feature coordinate**, isolating the deep-kernel geometry gain. It does *not* learn the feature map jointly through the GP marginal likelihood (true DKL). The full Python notebook walks the complete pedagogical arc — a plain-GP warm-up, a hidden-regime problem, RBF/Matérn raw baselines, a hand-crafted teacher sigmoid feature map, and a small neural network trained to approximate it — before drawing the same conclusion.
"""

# ╔═╡ 22222222-1408-4222-8222-222222222222
begin
    import Pkg
    Pkg.activate(joinpath(@__DIR__, "..", "..", "..", "julia"))

    using DLEFJulia
    using LinearAlgebra
end

# ╔═╡ 190c1214-1006-c7ce-7e1d-35a6c3a5d197
md"""
### The test problem and its latent feature

This preview uses a known **teacher feature map** — a linear latent coordinate

\$\$\phi(x) = x_1 - 0.5\,x_2 + 0.25\,x_3,\$\$

and a target that is smooth in that coordinate:

\$\$f(x) = \sin\!\bigl(4\,\phi(x)\bigr) + 0.1\,x_4^2.\$\$

The dominant structure lives on the 1D coordinate \$\phi(x)\$ (with a small residual quadratic in \$x_4\$). A GP given \$\phi(x)\$ therefore sees an easy 1D problem, while a GP given the raw 4D input \$x\$ must recover the same structure through a single stationary metric in the ambient space — exactly the situation a deep kernel is meant to repair.
"""

# ╔═╡ 33333333-1408-4333-8333-333333333333
begin
    RUN_MODE = "smoke"
    SEED = 0
    budgets = (
        smoke = (n = 35, n_test = 90),
        teaching = (n = 300, n_test = 1_000),
        production = (n = 1_500, n_test = 4_000),
    )
    hp = run_mode_budget(RUN_MODE; budgets)
    rng = rng_from_seed(SEED)
    feature(x) = x[1] - 0.5x[2] + 0.25x[3]
    target(x) = sin(4feature(x)) + 0.1x[4]^2
end

# ╔═╡ 4ca64bd4-36f2-0fd9-1093-9337a1cf17f7
md"""
### Raw-input GP vs. feature-space GP

We fit two Gaussian processes with the same in-house Cholesky solver:

- a **raw-input GP** on the full 4D input \$x\$ (`fit_cholesky_gp(x, y; lengthscale = 1.0, …)`), and
- a **feature-space GP** on the 1D teacher coordinate \$z = \phi(x)\$ (`fit_cholesky_gp(z, y; lengthscale = 0.35, …)`).

The feature-space GP is the cleanest form of the deep-kernel idea: the GP itself is unchanged; only the representation it sees has been improved.
"""

# ╔═╡ 44444444-1408-4444-8444-444444444444
begin
    x = 2 .* rand(rng, 4, hp.n) .- 1
    y = reshape([target(x[:, j]) for j in axes(x, 2)], 1, :)
    z = reshape([feature(x[:, j]) for j in axes(x, 2)], 1, :)
    raw_gp = fit_cholesky_gp(x, y; lengthscale = 1.0, noise = 1e-5)
    feature_gp = fit_cholesky_gp(z, y; lengthscale = 0.35, noise = 1e-5)
end

# ╔═╡ 3617d59a-5156-90b5-a0f3-aa0287cd7b7f
md"""
### Evaluation

We draw a held-out test set, map it through the same feature map, and compare the root-mean-squared errors of the raw-input and feature-space GPs.

*The full Python notebook goes further:* it adds a **Matérn** raw baseline, replaces the teacher feature map with a **small neural network** trained to approximate it, reports observation-interval calibration (`ObsNLPD`, `ObsCoverage95`), and draws per-regime **slice plots**. Those steps sharpen the same story this preview makes with two GP fits.
"""

# ╔═╡ 55555555-1408-4555-8555-555555555555
begin
    x_test = 2 .* rand(rng_from_seed(SEED; offset = 1), 4, hp.n_test) .- 1
    y_test = reshape([target(x_test[:, j]) for j in axes(x_test, 2)], 1, :)
    z_test = reshape([feature(x_test[:, j]) for j in axes(x_test, 2)], 1, :)
    raw_rmse = gp_rmse(raw_gp, x_test, y_test)
    feature_rmse = gp_rmse(feature_gp, z_test, y_test)
end

# ╔═╡ 81a8ede2-d5ce-b0a0-c974-f868a7a72a0d
md"""
### What this shows

The main lesson: a deep kernel helps when the target is smooth in a **latent coordinate system** but not in the raw coordinates seen by a standard stationary GP.

- The **raw-input GP** is the weaker model — it must use a single stationary geometry in the full input space.
- The **feature-space GP** gives the lower RMSE because the teacher coordinate \$\phi(x)\$ makes the target nearly one-dimensional and smooth.

In real applied work the unknown part is the feature map \$\phi_\theta(x)\$ itself: deep kernel learning replaces the hand-crafted or teacher map with a neural network trained from data — jointly with the GP marginal likelihood in the full DKL form — so the GP once again sees a representation where a simple kernel is appropriate.

The cell below returns the machine-checkable summary: the raw-input GP RMSE, the feature-space GP RMSE, and the feature-space dimension.
"""

# ╔═╡ 66666666-1408-4666-8666-666666666666
(
    raw_gp_rmse = raw_rmse,
    feature_gp_rmse = feature_rmse,
    feature_space_dim = size(z, 1),
)

# ╔═╡ Cell order:
# ╟─11111111-1408-4111-8111-111111111111
# ╟─a7c6bbb4-8836-4e57-9066-bb07cf669a26
# ╟─95f023c6-e6b2-23cb-4417-05f718d5e2bc
# ╠═22222222-1408-4222-8222-222222222222
# ╟─190c1214-1006-c7ce-7e1d-35a6c3a5d197
# ╠═33333333-1408-4333-8333-333333333333
# ╟─4ca64bd4-36f2-0fd9-1093-9337a1cf17f7
# ╠═44444444-1408-4444-8444-444444444444
# ╟─3617d59a-5156-90b5-a0f3-aa0287cd7b7f
# ╠═55555555-1408-4555-8555-555555555555
# ╟─81a8ede2-d5ce-b0a0-c974-f868a7a72a0d
# ╠═66666666-1408-4666-8666-666666666666
