### A Pluto.jl notebook ###
# v0.20.0

using Markdown
using InteractiveUtils

# ╔═╡ 11111111-1402-4111-8111-111111111111
md"""
# Lecture 14, Notebook 02: Gaussian Processes and BAL in Julia

This notebook uses the in-house Cholesky GP helper and selects the next design
point by posterior variance, the Bayesian active-learning signal used in the
Python version.
"""

# ╔═╡ bf619f6c-1100-8701-1a94-cd6d912a13a7
md"""
## Lecture 14, Notebook 02: Gaussian-process regression and Bayesian active learning

**Course:** Deep Learning for Solving and Estimating Dynamic Models in Economics and Finance
**Script reference:** §9.1-9.3 (Gaussian-process regression, kernels, and Bayesian active learning)
**Notebook role:** core
**Author:** Simon Scheidegger

*Julia/Lux/Pluto preview of* `lectures/lecture_14_surrogates_and_gps/code/lecture_14_02_GP_and_BAL.ipynb`.

> **Run mode.** The checked-in run uses `RUN_MODE = "smoke"` with small point budgets; the `teaching` / `production` budgets in the next cell enlarge the initial design, candidate grid, and test grid.
"""

# ╔═╡ f5b60c74-db1f-61a4-6ab1-309ac79256f7
md"""
## Overview

This notebook covers:

1. **GP regression from scratch** — the squared-exponential kernel and posterior inference through a Cholesky factorisation.
2. **GP regression with automatic hyperparameters** — in the Python ground truth via `scikit-learn`'s marginal-likelihood optimisation.
3. **Bayesian active learning (BAL)** — using the GP posterior variance to select training points.

The Julia track replaces `scikit-learn`/NumPy with the in-house Cholesky GP helpers `fit_cholesky_gp`, `gp_predict`, and the BAL scorer `bal_next_index`.

**References:**
- Rasmussen & Williams (2006), *Gaussian Processes for Machine Learning*, MIT Press.
- Renner & Scheidegger (2018), *Machine Learning for Dynamic Incentive Problems*.
- Kübler, Scheidegger & Surbek (2026, forthcoming), *J. Political Economy: Macroeconomics*.
"""

# ╔═╡ 22222222-1402-4222-8222-222222222222
begin
    import Pkg
    Pkg.activate(joinpath(@__DIR__, "..", "..", "..", "julia"))

    using DLEFJulia
    using Statistics
end

# ╔═╡ 33333333-1402-4333-8333-333333333333
begin
    RUN_MODE = "smoke"
    SEED = 0
    budgets = (
        smoke = (n_initial = 6, n_candidates = 60, n_test = 80),
        teaching = (n_initial = 12, n_candidates = 400, n_test = 500),
        production = (n_initial = 24, n_candidates = 2_000, n_test = 2_000),
    )
    hp = run_mode_budget(RUN_MODE; budgets)
    f(x) = sin(6x) + 0.5cos(11x)
end

# ╔═╡ e73a68e7-45ec-7958-21bf-da91fa2499bf
md"""
---
## Part 1: GP Regression with a Cholesky Kernel

### The kernel function

The **squared-exponential (SE / RBF)** kernel is

\$\$k_{SE}(x, x') = \sigma_f^2 \exp\!\left(-\frac{(x - x')^2}{2\ell^2}\right),\$\$

with **length scale** \$\ell\$ (horizontal correlation range) and **signal variance** \$\sigma_f^2\$ (vertical scale). The Julia track uses `fit_cholesky_gp`, which builds \$K\$ from this kernel and factorises \$K + \sigma_n^2 I = LL^\top\$ for numerical stability — the same Cholesky route the Python notebook writes out by hand.

### The GP posterior

Given training data \$\mathcal{D} = \{(x_i, f_i)\}_{i=1}^N\$ and test points \$X_*\$, the posterior is

\$\$\mu_* = K_*^\top K^{-1}\mathbf{f}, \qquad \Sigma_* = K_{**} - K_*^\top K^{-1} K_*,\$\$

where \$K = k(X, X)\$, \$K_* = k(X, X_*)\$, and \$K_{**} = k(X_*, X_*)\$. `gp_predict` returns exactly this posterior mean and variance.

> **The full Python notebook also covers** sampling functions from the GP *prior* via \$\mathbf{f} = L\mathbf{u}\$, \$\mathbf{u} \sim \mathcal{N}(0, I)\$; the effect of the length scale (small \$\ell\$ overfits, large \$\ell\$ underfits); noiseless vs. noisy regression; and **Part 2**, `scikit-learn`'s `GaussianProcessRegressor`, which optimises \$(\ell, \sigma_f)\$ by maximising the log marginal likelihood \$-\tfrac{1}{2}\mathbf{y}^\top K_y^{-1}\mathbf{y} - \tfrac{1}{2}\log|K_y| - \tfrac{N}{2}\log 2\pi\$ (RBF vs. Matérn-3/2). This preview fixes \$\ell = 0.18\$ and a small noise floor, so no hyperparameter search runs here.
"""

# ╔═╡ 468a64fa-302f-fdba-c032-36855f6c4019
md"""
---
## Part 3: Bayesian Active Learning (BAL)

When each model evaluation is expensive, we choose training points **wisely** — uniform sampling wastes evaluations in smooth regions. BAL scores a candidate \$x\$ by

\$\$U(x) = w_{\mathrm{exp}}\,\mu(x) + \frac{\beta}{2}\log \sigma^2(x),\$\$

and for surrogate building we use **pure exploration** (\$w_{\mathrm{exp}} = 0\$): pick the point of highest posterior variance. `bal_next_index` implements this score and returns the arg-max candidate. The cell below fits the initial GP to the target \$f(x) = \sin 6x + \tfrac{1}{2}\cos 11x\$, selects one BAL point, and refits on the augmented design.

> **The full Python notebook also covers** the multi-step BAL *loop* on the kinked target \$|\sin(\pi x/2)|\$, a same-budget **uniform-sampling baseline**, and MAE-versus-budget convergence curves. This Julia preview takes a single BAL step and compares held-out RMSE against the initial design.
"""

# ╔═╡ 44444444-1402-4444-8444-444444444444
begin
    x_train = reshape(collect(range(0.05, 0.95; length = hp.n_initial)), 1, :)
    y_train = reshape(f.(vec(x_train)), 1, :)
    gp = fit_cholesky_gp(x_train, y_train; lengthscale = 0.18, noise = 1e-6)
    candidates = reshape(collect(range(0.0, 1.0; length = hp.n_candidates)), 1, :)
    next_idx = bal_next_index(gp, candidates)
    x_next = candidates[:, next_idx:next_idx]
    y_next = reshape(f.(vec(x_next)), 1, :)
    gp_bal = fit_cholesky_gp(hcat(x_train, x_next), hcat(y_train, y_next); lengthscale = 0.18, noise = 1e-6)
end

# ╔═╡ a2f162ef-c63f-f469-6e23-46916f962cfa
md"""
### Posterior prediction and RMSE comparison

We evaluate both GPs on a dense test grid with `gp_predict` and compare held-out RMSE (`gp_rmse`) for the initial design versus the BAL-augmented design. `posterior.variance` is the SE-kernel posterior variance \$\Sigma_*\$ from above.
"""

# ╔═╡ 55555555-1402-4555-8555-555555555555
begin
    x_test = reshape(collect(range(0.0, 1.0; length = hp.n_test)), 1, :)
    y_test = reshape(f.(vec(x_test)), 1, :)
    base_rmse = gp_rmse(gp, x_test, y_test)
    bal_rmse = gp_rmse(gp_bal, x_test, y_test)
    posterior = gp_predict(gp_bal, x_test)
end

# ╔═╡ 17828942-c68e-6c2e-a927-6c87d6b98f67
md"""
---
## Summary

### Key takeaways

1. **Gaussian processes** give a principled nonparametric Bayesian approach to regression with **built-in uncertainty quantification**.
2. The **kernel** encodes prior beliefs about smoothness; in production its hyperparameters are learned by maximising the marginal likelihood (here \$\ell\$ is fixed for the preview).
3. **Bayesian active learning** leverages the GP posterior variance to select training points, converging faster than uniform sampling — especially for functions with kinks or local features.
4. **Limitation:** GPs scale as \$O(N^3)\$, so they best suit moderate dimensions (\$d \leq 10\$) and modest training sets (\$N \leq 10^4\$).

On the target here, one BAL step is directed at the maximum-variance candidate; the headline economic payoff of active learning appears in §9.4 and the GP-VFI loop of NB 04.

### References

- Rasmussen & Williams (2006), *Gaussian Processes for Machine Learning*, MIT Press.
- Renner & Scheidegger (2018), *Machine Learning for Dynamic Incentive Problems*.
- Kübler, Scheidegger & Surbek (2026), *Globally Optimal Policies*, J. Political Economy: Macroeconomics.

The cell below returns the machine-checkable diagnostics summary for this notebook's smoke run.
"""

# ╔═╡ 66666666-1402-4666-8666-666666666666
(
    initial_points = hp.n_initial,
    selected_candidate = x_next[1],
    base_rmse = base_rmse,
    bal_rmse = bal_rmse,
    max_posterior_variance = maximum(posterior.variance),
)

# ╔═╡ Cell order:
# ╟─11111111-1402-4111-8111-111111111111
# ╟─bf619f6c-1100-8701-1a94-cd6d912a13a7
# ╟─f5b60c74-db1f-61a4-6ab1-309ac79256f7
# ╠═22222222-1402-4222-8222-222222222222
# ╠═33333333-1402-4333-8333-333333333333
# ╟─e73a68e7-45ec-7958-21bf-da91fa2499bf
# ╟─468a64fa-302f-fdba-c032-36855f6c4019
# ╠═44444444-1402-4444-8444-444444444444
# ╟─a2f162ef-c63f-f469-6e23-46916f962cfa
# ╠═55555555-1402-4555-8555-555555555555
# ╟─17828942-c68e-6c2e-a927-6c87d6b98f67
# ╠═66666666-1402-4666-8666-666666666666
