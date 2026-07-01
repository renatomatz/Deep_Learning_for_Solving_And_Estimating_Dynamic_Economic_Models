### A Pluto.jl notebook ###
# v0.20.0

using Markdown
using InteractiveUtils

# ╔═╡ 11111111-1406-4111-8111-111111111111
md"""
# Lecture 14, Notebook 06: Active Subspace in 10D

A high-dimensional exponential ridge is reduced to one active coordinate before
fitting a low-order polynomial link.
"""

# ╔═╡ ea587554-0f55-e1aa-f85a-fc0e3d307e62
md"""
## Lecture 14, Notebook 06: Active subspaces — a 10D example with a near-1D effective subspace

**Course:** Deep Learning for Solving and Estimating Dynamic Models in Economics and Finance
**Script reference:** §9.5 (Active subspaces — 10D illustration with a near-1D effective subspace)
**Notebook role:** core
**Author:** Simon Scheidegger

*Julia/Lux/Pluto preview of* `lectures/lecture_14_surrogates_and_gps/code/lecture_14_06_Active_Subspace_10D.ipynb`.
"""

# ╔═╡ f31b0b99-b5cf-3056-76c1-ef7ce5aaa353
md"""
This notebook demonstrates the active subspace method on a **10-dimensional** test function. Despite the high-dimensional input, the eigenvalue spectrum of the gradient covariance matrix \$C_N\$ shows a **sharp drop after the first eigenvalue**, so the function effectively lives on a **1D active subspace**, and the leading eigenvector recovers the dominant input direction.

**Reference:** Scheidegger & Bilionis (2019), *Machine Learning for High-Dimensional Dynamic Stochastic Economies*, Journal of Computational Science 33, 68–82.

**Corresponds to:** Figure 4 in the paper.
"""

# ╔═╡ 22222222-1406-4222-8222-222222222222
begin
    import Pkg
    Pkg.activate(joinpath(@__DIR__, "..", "..", "..", "julia"))

    using DLEFJulia
    using LinearAlgebra
end

# ╔═╡ a5d76b1a-b6ac-46b7-eef5-148865d6a509
md"""
### 1. The 10D exponential ridge

This Julia preview uses a clean single-direction *ridge* in \$\mathbb{R}^{10}\$:

\$\$f(x) = \exp\!\bigl(0.7\,\langle w, x\rangle\bigr), \qquad w = \frac{(1.0,\,0.9,\,0.8,\,\ldots,\,0.1)}{\lVert(1.0,\,0.9,\,\ldots,\,0.1)\rVert}.\$\$

Because \$f\$ depends on \$x\$ only through the scalar projection \$\langle w, x\rangle\$, its gradient is always parallel to \$w\$: \$\nabla f(x) = 0.7\,f(x)\,w\$. The function therefore lives *exactly* on a 1D active subspace spanned by \$w\$, and the leading eigenvector recovered below should align with \$w\$.

(The Python ground truth uses the closely related ridge \$f(x) = \exp(0.01\,x_1 + 0.7\,x_2 + \cdots + 0.1\,x_{10})\$, whose dominant coefficient \$0.7\$ sits on dimension 2; both targets share the same near-1D structure.)
"""

# ╔═╡ 33333333-1406-4333-8333-333333333333
begin
    RUN_MODE = "smoke"
    SEED = 0
    budgets = (
        smoke = (n = 60, n_test = 100),
        teaching = (n = 600, n_test = 2_000),
        production = (n = 3_000, n_test = 8_000),
    )
    hp = run_mode_budget(RUN_MODE; budgets)
    rng = rng_from_seed(SEED)
    w = collect(range(1.0, 0.1; length = 10))
    w ./= sqrt(sum(abs2, w))
    fvec(x) = exp(0.7 * dot(w, x))
    grad(x) = 0.7 * fvec(x) .* w
end

# ╔═╡ d4fe5102-88da-6b49-c49a-320d57083b33
md"""
### 2. Gradient covariance \$C_N\$ and its eigendecomposition

We sample \$N\$ points in \$[-1,1]^{10}\$, evaluate \$f\$ and \$\nabla f\$ at each, and form the **gradient covariance matrix**

\$\$C_N = \frac{1}{N}\sum_{i=1}^{N}\nabla f(x_i)\,\nabla f(x_i)^\top \;\in\;\mathbb{R}^{10\times 10}.\$\$

This symmetric positive semi-definite matrix (built by `active_subspace_matrix`) encodes how strongly \$f\$ varies along each input direction. Its eigendecomposition (`active_subspace`) returns eigenvalues \$\lambda_1 \ge \lambda_2 \ge \cdots\$ and eigenvectors; a **sharp drop** in the spectrum signals that the variability is concentrated on a low-dimensional subspace, and the leading eigenvector spans the 1D active subspace. Since every gradient is parallel to \$w\$, \$C_N\$ is rank one and a single eigenvalue dominates.

The same cell then projects the inputs onto the leading direction and fits a degree-3 polynomial-ridge surrogate on the 1D coordinate via `fit_active_subspace_surrogate`.
"""

# ╔═╡ 44444444-1406-4444-8444-444444444444
begin
    x = 2 .* rand(rng, 10, hp.n) .- 1
    y = reshape([fvec(x[:, j]) for j in axes(x, 2)], 1, :)
    gradients = hcat([grad(x[:, j]) for j in axes(x, 2)]...)
    as = active_subspace(active_subspace_matrix(gradients))
    fit = fit_active_subspace_surrogate(x, y, as.vectors; dims = 1, degree = 3, lambda = 1e-7)
end

# ╔═╡ 2e7d9541-71aa-4f49-238c-ee8175d547ef
md"""
### 3. Evaluating the active-subspace surrogate

We draw a fresh held-out test set, predict with the fitted 1D active-subspace surrogate, and report the relative \$L^2\$ error.

*The full Python notebook also covers* the eigenvalue-spectrum and active-direction bar plots and a training-set-size sweep \$N \in \{4, 8, 16, 32, 64\}\$ that pits the 1D **ASGP** (a GP fit on the projected coordinate) against a **full 10D GP**. This compact preview keeps a single polynomial-ridge ASGP at \$d = 1\$; the mechanism it illustrates — reduce dimension first, then fit — is the point.
"""

# ╔═╡ 55555555-1406-4555-8555-555555555555
begin
    x_test = 2 .* rand(rng_from_seed(SEED; offset = 1), 10, hp.n_test) .- 1
    y_test = reshape([fvec(x_test[:, j]) for j in axes(x_test, 2)], 1, :)
    y_hat = predict_active_subspace_surrogate(fit, x_test)
end

# ╔═╡ 158a714e-78ae-da3d-1779-1f2b29cca23d
md"""
### Read the curves & takeaway

The 1D active-subspace surrogate matches the true function with only a handful of points: it is solving a 1D regression problem, and a low-order polynomial (or an RBF-kernel GP) picks up the smooth 1D profile almost immediately. A full-dimensional GP, in contrast, must fit a 10D function and needs substantially more data to reach comparable accuracy.

This is the mechanism that lets GPs scale to \$D = 100\$ or more in Scheidegger & Bilionis (2019): identify the active subspace first, then fit the surrogate there. Notebook `07` is the follow-up on a target where \$d = 1\$ no longer suffices; notebooks `09` and `10` push further by replacing the *linear* projection with a learned nonlinear encoder.

The cell below returns the machine-checkable diagnostics: the leading eigenvalue's share of the spectrum, the alignment of the recovered direction with \$w\$, and the surrogate's relative \$L^2\$ error.
"""

# ╔═╡ 66666666-1406-4666-8666-666666666666
(
    leading_share = as.values[1] / sum(as.values),
    direction_alignment = abs(dot(as.vectors[:, 1], w)),
    relative_rmse = relative_l2_error(y_hat, y_test),
)

# ╔═╡ Cell order:
# ╟─11111111-1406-4111-8111-111111111111
# ╟─ea587554-0f55-e1aa-f85a-fc0e3d307e62
# ╟─f31b0b99-b5cf-3056-76c1-ef7ce5aaa353
# ╠═22222222-1406-4222-8222-222222222222
# ╟─a5d76b1a-b6ac-46b7-eef5-148865d6a509
# ╠═33333333-1406-4333-8333-333333333333
# ╟─d4fe5102-88da-6b49-c49a-320d57083b33
# ╠═44444444-1406-4444-8444-444444444444
# ╟─2e7d9541-71aa-4f49-238c-ee8175d547ef
# ╠═55555555-1406-4555-8555-555555555555
# ╟─158a714e-78ae-da3d-1779-1f2b29cca23d
# ╠═66666666-1406-4666-8666-666666666666
