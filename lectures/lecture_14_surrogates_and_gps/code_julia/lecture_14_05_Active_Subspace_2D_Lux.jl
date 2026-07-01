### A Pluto.jl notebook ###
# v0.20.0

using Markdown
using InteractiveUtils

# ╔═╡ 11111111-1405-4111-8111-111111111111
md"""
# Lecture 14, Notebook 05: Active Subspace in 2D

We compute the gradient covariance for a two-dimensional test function, project
onto the dominant active direction, and fit a polynomial link.
"""

# ╔═╡ db22258e-9f4a-b59e-b4de-0f90de11dcca
md"""
## Lecture 14, Notebook 05: Active subspaces — a 2D illustration

**Course:** Deep Learning for Solving and Estimating Dynamic Models in Economics and Finance
**Script reference:** §9.5 (Active subspaces — 2D illustration)
**Notebook role:** core
**Author:** Simon Scheidegger

*Julia/Lux/Pluto preview of* `lectures/lecture_14_surrogates_and_gps/code/lecture_14_05_Active_Subspace_2D.ipynb`.

> **Run mode.** The checked-in run uses `RUN_MODE = "smoke"` with small sample budgets; the `teaching` / `production` budgets in the next cell enlarge the training and test designs.
"""

# ╔═╡ 283cfd1f-21b2-9553-235e-264a27d68885
md"""
## Dimensionality Reduction for Gaussian Process Surrogates

**Reference:** Scheidegger & Bilionis (2019), *"Machine Learning for High-Dimensional Dynamic Stochastic Economies"*, J. Computational Science 33, 68–82. **Figure 3.**

### Motivation

When using Gaussian processes to approximate value or policy functions in dynamic programming, the **curse of dimensionality** quickly makes standard GPR infeasible (for \$D \gtrsim 10\$ the Euclidean distance becomes uninformative).

**Active subspaces (AS)** solve this by discovering a low-dimensional linear projection \$\mathbf{W}\$ that captures most of the function's variation. The surrogate then operates on the projected, low-dimensional inputs instead of the full state space.

### What this notebook demonstrates

1. A simple 2D test function where one direction matters more than the other.
2. How to construct the active subspace from gradient information.
3. The eigenvalue spectrum and the projection direction \$\mathbf{W}\$.
4. A low-dimensional surrogate on the active subspace versus the full-input regression.
5. How the active-subspace surrogate wins at small sample sizes — the regime that matters for expensive economic models.
"""

# ╔═╡ 22222222-1405-4222-8222-222222222222
begin
    import Pkg
    Pkg.activate(joinpath(@__DIR__, "..", "..", "..", "julia"))

    using DLEFJulia
    using LinearAlgebra
    using Statistics
end

# ╔═╡ 2cc51392-622d-5430-de53-026c9c2cc356
md"""
## 1. The Test Function

The Julia preview uses a single-index exponential

\$\$f(\mathbf{x}) = \exp\!\left(0.8\,\mathbf{w}^\top \mathbf{x}\right), \qquad \mathbf{w} = \frac{(1, 2)}{\sqrt{5}}, \quad \mathbf{x} \in [-1, 1]^2,\$\$

with the closed-form gradient

\$\$\nabla f(\mathbf{x}) = 0.8\, f(\mathbf{x})\, \mathbf{w}.\$\$

Because \$f\$ varies only along the fixed direction \$\mathbf{w}\$, it has an exact **one-dimensional** active subspace, and the active-subspace method should recover \$\mathbf{w}\$ without being told the coefficients.

> **Divergence from the Python ground truth.** The Python notebook uses \$f = \exp(0.3 x_1 + 0.7 x_2)\$, i.e. the un-normalised direction \$(0.3, 0.7)\$. The Julia code instead fixes the *unit* direction \$\mathbf{w} = (1, 2)/\sqrt{5}\$ scaled by \$0.8\$; the math (single-index model, rank-one gradient covariance) is identical, only the numerical direction differs.
"""

# ╔═╡ 33333333-1405-4333-8333-333333333333
begin
    RUN_MODE = "smoke"
    SEED = 0
    budgets = (
        smoke = (n = 40, n_test = 80),
        teaching = (n = 400, n_test = 1_000),
        production = (n = 2_000, n_test = 4_000),
    )
    hp = run_mode_budget(RUN_MODE; budgets)
    rng = rng_from_seed(SEED)
    direction = [1.0, 2.0] ./ sqrt(5.0)
    fvec(x) = exp(0.8 * dot(direction, x))
    grad(x) = 0.8 * fvec(x) .* direction
end

# ╔═╡ 4bd8a3f5-92c8-b3a8-ad98-7ec20103a974
md"""
## 2. Constructing the Active Subspace

The algorithm has three steps:

1. **Collect gradients** at \$N\$ sample points: \$\mathbf{g}^{(i)} = \nabla f(\mathbf{x}^{(i)})\$.
2. **Form the gradient outer-product matrix**
\$\$\mathbf{C}_N = \frac{1}{N} \sum_{i=1}^{N} \mathbf{g}^{(i)} \bigl(\mathbf{g}^{(i)}\bigr)^{\!\top}.\$\$
3. **Eigendecompose** \$\mathbf{C}_N\$: a **sharp drop** in the eigenvalues reveals the active-subspace dimension \$d\$, and the top \$d\$ eigenvectors form \$\mathbf{W}\$.

`active_subspace_matrix` assembles \$\mathbf{C}_N\$ from the sampled gradients and `active_subspace` returns its eigenvalues/eigenvectors. For this single-index \$f\$ we expect one large eigenvalue (aligned with \$\mathbf{w}\$) and one negligible one. The cell then fits a one-dimensional link surrogate on the projected coordinate \$y = \mathbf{W}^\top \mathbf{x}\$ with `fit_active_subspace_surrogate`.

> **The full Python notebook also covers** the three diagnostic plots of §3 (eigenvalue spectrum, the projection direction over a contour plot, and the *sufficient summary plot* \$f\$ vs. \$y = \mathbf{W}^\top\mathbf{x}\$). This preview computes the same eigenvalues and projection alignment but draws no figures.
"""

# ╔═╡ 44444444-1405-4444-8444-444444444444
begin
    x = 2 .* rand(rng, 2, hp.n) .- 1
    y = reshape([fvec(x[:, j]) for j in axes(x, 2)], 1, :)
    gradients = hcat([grad(x[:, j]) for j in axes(x, 2)]...)
    as = active_subspace(active_subspace_matrix(gradients))
    fit = fit_active_subspace_surrogate(x, y, as.vectors; dims = 1, degree = 3, lambda = 1e-8)
end

# ╔═╡ 6a0cc528-5b1c-b7df-45fa-52df8bb31fad
md"""
## 4. Active-subspace surrogate vs. full-input regression

The Python notebook runs a head-to-head between an **ASGP** (a GP on the 1D projected inputs \$y_i = \mathbf{W}^\top\mathbf{x}_i\$) and a **standard GP** on the full 2D inputs, sweeping the training-set size \$N \in \{4, 8, 16, 32\}\$ and measuring the **relative RMSE**

\$\$\text{Rel. RMSE} = \sqrt{\frac{1}{N_{\text{test}}} \sum_{i=1}^{N_{\text{test}}} \left(\frac{f(\mathbf{x}_i) - \hat{f}(\mathbf{x}_i)}{f(\mathbf{x}_i)}\right)^{\!2}}\$\$

on held-out points. This Julia preview evaluates one active-subspace surrogate — a **degree-3 polynomial ridge** link on the projected coordinate (`predict_active_subspace_surrogate`) — on a fresh test design and reports its relative L2 error and max absolute error.

> **The full Python notebook also covers** the \$N\$-sweep convergence curve (§4) and the detailed \$N = 8\$ comparison of §5 (1D ASGP prediction with a 95% band, and 2D error heatmaps for both methods). The preview keeps the single-fit accuracy check.
"""

# ╔═╡ 55555555-1405-4555-8555-555555555555
begin
    x_test = 2 .* rand(rng_from_seed(SEED; offset = 1), 2, hp.n_test) .- 1
    y_test = reshape([fvec(x_test[:, j]) for j in axes(x_test, 2)], 1, :)
    y_hat = predict_active_subspace_surrogate(fit, x_test)
    err = residual_summary(y_hat .- y_test)
end

# ╔═╡ 956ca1bf-c815-6c79-2b66-3633af837383
md"""
## Summary

### Key takeaways

1. **Active subspaces automatically discover the dominant direction of variation** — the leading eigenvector \$\mathbf{W}\$ of the gradient covariance \$\mathbf{C}_N\$ aligns with the direction in which \$f\$ changes most (here \$\mathbf{w} = (1, 2)/\sqrt{5}\$).
2. **The eigenvalue spectrum reveals the effective dimensionality** — a sharp drop from \$\lambda_1\$ to \$\lambda_2\$ (ratio \$\gg 1\$) confirms that a 1D projection captures nearly all the variation.
3. **The low-dimensional surrogate wins at small sample sizes** — operating in 1D instead of 2D, it needs far fewer training points, an advantage that grows with the dimensionality of the original problem.
4. **The sufficient summary plot is the key diagnostic** — if \$f(\mathbf{x})\$ collapses to a clean 1D curve against \$y = \mathbf{W}^\top\mathbf{x}\$, the active subspace is working.

### Connection to dynamic programming

In the ASGP-VFI framework (see NB 04 and Part IV of the slides), the value function \$V(\mathbf{k})\$ of a stochastic growth model often lives on a low-dimensional active subspace even when the state space has \$D = 500\$ sectors — this is what lets GP-based dynamic programming scale to hundreds of dimensions. Notebooks 06 (10D) and 07 (nonlinear, \$d = 2\$–\$3\$) push the same method further.

The cell below returns the machine-checkable diagnostics summary: the two eigenvalues, the projection alignment \$|\mathbf{W}_{:,1}^\top\mathbf{w}|\$, and the surrogate's relative RMSE and max absolute error.
"""

# ╔═╡ 66666666-1405-4666-8666-666666666666
(
    leading_eigenvalue = as.values[1],
    second_eigenvalue = as.values[2],
    projection_alignment = abs(dot(as.vectors[:, 1], direction)),
    relative_rmse = relative_l2_error(y_hat, y_test),
    max_abs_error = err.max_abs,
)

# ╔═╡ Cell order:
# ╟─11111111-1405-4111-8111-111111111111
# ╟─db22258e-9f4a-b59e-b4de-0f90de11dcca
# ╟─283cfd1f-21b2-9553-235e-264a27d68885
# ╠═22222222-1405-4222-8222-222222222222
# ╟─2cc51392-622d-5430-de53-026c9c2cc356
# ╠═33333333-1405-4333-8333-333333333333
# ╟─4bd8a3f5-92c8-b3a8-ad98-7ec20103a974
# ╠═44444444-1405-4444-8444-444444444444
# ╟─6a0cc528-5b1c-b7df-45fa-52df8bb31fad
# ╠═55555555-1405-4555-8555-555555555555
# ╟─956ca1bf-c815-6c79-2b66-3633af837383
# ╠═66666666-1405-4666-8666-666666666666
