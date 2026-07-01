### A Pluto.jl notebook ###
# v0.20.0

using Markdown
using InteractiveUtils

# ╔═╡ 11111111-1407-4111-8111-111111111111
md"""
# Lecture 14, Notebook 07: Nonlinear Active-Subspace Diagnostics

The test function includes a product interaction, so the eigenspectrum is a
diagnostic rather than a perfect one-dimensional collapse.
"""

# ╔═╡ 9e36fe3d-c46e-63bb-ced4-00b8abddaee9
md"""
## Lecture 14, Notebook 07: Active subspaces — a nonlinear 10D function

**Course:** Deep Learning for Solving and Estimating Dynamic Models in Economics and Finance
**Script reference:** §9.5 (Active subspaces — nonlinear 10D extension where linear AS needs \$d \geq 2\$)
**Notebook role:** extension
**Author:** Simon Scheidegger

*Julia/Lux/Pluto preview of* `lectures/lecture_14_surrogates_and_gps/code/lecture_14_07_Active_Subspace_Nonlinear.ipynb`.
"""

# ╔═╡ 5bcceed1-1b01-eff2-55d6-b958c6b93094
md"""
The previous notebooks showed functions where a single active dimension (\$d = 1\$) suffices. Here we introduce a **nonlinear** 10D test function with a **product term** \$x_2 x_3\$, which makes the gradient depend on the input location. As a result a 1D active subspace is no longer enough — we need \$d = 2\$ or \$d = 3\$ active dimensions for good surrogate accuracy. This notebook fits ASGP surrogates with \$d \in \{1, 2, 3\}\$ active dimensions.

**Reference:** Scheidegger & Bilionis (2019), *Machine Learning for High-Dimensional Dynamic Stochastic Economies*, Journal of Computational Science 33, 68–82.

**Extends:** the examples in the paper with a nonlinear variant that requires multiple active dimensions.
"""

# ╔═╡ 22222222-1407-4222-8222-222222222222
begin
    import Pkg
    Pkg.activate(joinpath(@__DIR__, "..", "..", "..", "julia"))

    using DLEFJulia
    using Statistics
end

# ╔═╡ 1f190247-7b41-56fd-a962-1821c0d2214e
md"""
### 1. A nonlinear test function

The previous notebook had a single active dimension (\$d = 1\$). Here the Julia preview uses a **nonlinear** target with a product interaction:

\$\$f(x) = \sin\!\bigl(x_1 + 0.5\,x_2 x_3\bigr) + 0.1\sum_{i=4}^{10} x_i.\$\$

The cross-term \$x_2 x_3\$ sits *inside* the nonlinearity, so \$\partial f/\partial x_2\$ depends on \$x_3\$ and vice versa: the gradient direction is no longer constant across input space, and a single linear projection cannot capture all the variation. We therefore expect a linear active subspace to need \$d = 2\$ (or more), not \$d = 1\$.

(The Python ground truth uses the analogous target \$f(x) = x_2 x_3 \exp(0.01\,x_1 + 0.7\,x_2 + \cdots + 0.1\,x_{10})\$; both share the same lesson — a product interaction forces more than one active direction.)
"""

# ╔═╡ 33333333-1407-4333-8333-333333333333
begin
    RUN_MODE = "smoke"
    SEED = 0
    budgets = (
        smoke = (n = 60, n_test = 100),
        teaching = (n = 800, n_test = 2_000),
        production = (n = 4_000, n_test = 8_000),
    )
    hp = run_mode_budget(RUN_MODE; budgets)
    rng = rng_from_seed(SEED)
    fvec(x) = sin(x[1] + 0.5x[2] * x[3]) + 0.1sum(x[4:end])
end

# ╔═╡ 8a4f0412-7124-701a-8a26-4b75c40887c7
md"""
### 2. Active-subspace construction

We sample \$N\$ points in \$[-1,1]^{10}\$, evaluate \$f\$, and estimate the gradients by **finite differences** (`finite_difference_gradients`, since the analytic gradient is not used here). We build \$C_N = \frac{1}{N} G^\top G\$, eigen-decompose it (`active_subspace`), and for each active dimension \$d \in \{1, 2, 3\}\$ project onto the top-\$d\$ eigenvectors \$W_d \in \mathbb{R}^{10\times d}\$ and fit a degree-3 polynomial-ridge ASGP on the \$d\$-dimensional projected inputs.

*The full Python notebook also* sweeps several training sizes \$N \in \{10, 30, 100, 250, 500, 1000\}\$ and fits a full 10D GP for comparison; this compact preview fits the three ASGP dimensions once at the smoke budget. (Python tunes GP length-scales with restarts of the marginal likelihood; the closed-form polynomial-ridge fit used here needs no such restarts.)
"""

# ╔═╡ 44444444-1407-4444-8444-444444444444
begin
    x = 2 .* rand(rng, 10, hp.n) .- 1
    y = reshape([fvec(x[:, j]) for j in axes(x, 2)], 1, :)
    gradients = finite_difference_gradients(fvec, x; h = 1e-4)
    as = active_subspace(active_subspace_matrix(gradients))
    fits = [fit_active_subspace_surrogate(x, y, as.vectors; dims = d, degree = 3, lambda = 1e-6) for d in 1:3]
end

# ╔═╡ c917e981-ae46-ee89-f4ae-952f40644151
md"""
### 3. Errors for d = 1, 2, 3

We evaluate each fitted ASGP on a fresh held-out test set and report the relative \$L^2\$ error as a function of the number of active dimensions.

*In the Python notebook* the accompanying plots show (left) the eigenvalue spectrum decaying more gradually than in Notebook 06 — the top eigenvalue still dominates but the second and third are non-negligible, reflecting the \$x_2 x_3\$ interaction — and (right) the error curves: \$d = 1\$ plateaus because it cannot capture the interaction, \$d = 2\$ improves substantially, and \$d = 3\$ improves a little more, approaching the full 10D GP while working in a far lower-dimensional space.
"""

# ╔═╡ 55555555-1407-4555-8555-555555555555
begin
    x_test = 2 .* rand(rng_from_seed(SEED; offset = 1), 10, hp.n_test) .- 1
    y_test = reshape([fvec(x_test[:, j]) for j in axes(x_test, 2)], 1, :)
    rel_errors = [relative_l2_error(predict_active_subspace_surrogate(fit, x_test), y_test) for fit in fits]
end

# ╔═╡ c8008e87-9100-d4f7-abb9-7366ba877969
md"""
### Take-aways

- The cross-term \$x_2 x_3\$ **mixes** two input directions so the gradient direction varies across input space; the gradient outer product \$C_N\$ therefore picks up **more than one** nonzero eigendirection.
- Adding a second active direction (\$d = 2\$) yields a substantial drop in surrogate error; a third buys a little more. The elbow of the eigenvalue spectrum aligns with the elbow of the error curve — the spectrum is a *predictive* diagnostic for how many active dimensions are worth keeping.
- When even \$d = 2\$ is too many, the active manifold may be **curved** rather than linear: two linear features combined through a nonlinear aggregator. Notebook `09` shows a constructed target where linear AS needs \$d = 2\$ but a deep encoder collapses the same problem to \$d = 1\$; notebook `10` runs the deep-vs-linear comparison on the borehole benchmark.

The cell below returns the machine-checkable diagnostics: the top and third eigenvalues, the best relative \$L^2\$ error across \$d\$, and the errors at \$d = 1\$ and \$d = 3\$.
"""

# ╔═╡ 66666666-1407-4666-8666-666666666666
(
    top_eigenvalue = as.values[1],
    third_eigenvalue = as.values[3],
    best_relative_rmse = minimum(rel_errors),
    rel_error_d1 = rel_errors[1],
    rel_error_d3 = rel_errors[3],
)

# ╔═╡ Cell order:
# ╟─11111111-1407-4111-8111-111111111111
# ╟─9e36fe3d-c46e-63bb-ced4-00b8abddaee9
# ╟─5bcceed1-1b01-eff2-55d6-b958c6b93094
# ╠═22222222-1407-4222-8222-222222222222
# ╟─1f190247-7b41-56fd-a962-1821c0d2214e
# ╠═33333333-1407-4333-8333-333333333333
# ╟─8a4f0412-7124-701a-8a26-4b75c40887c7
# ╠═44444444-1407-4444-8444-444444444444
# ╟─c917e981-ae46-ee89-f4ae-952f40644151
# ╠═55555555-1407-4555-8555-555555555555
# ╟─c8008e87-9100-d4f7-abb9-7366ba877969
# ╠═66666666-1407-4666-8666-666666666666
