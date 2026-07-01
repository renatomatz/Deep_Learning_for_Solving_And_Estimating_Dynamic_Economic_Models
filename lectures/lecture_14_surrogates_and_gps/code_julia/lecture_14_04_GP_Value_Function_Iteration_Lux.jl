### A Pluto.jl notebook ###
# v0.20.0

using Markdown
using InteractiveUtils

# ╔═╡ 11111111-1404-4111-8111-111111111111
md"""
# Lecture 14, Notebook 04: GP Value-Function Iteration Preview

The full Python notebook adaptively enriches a GP inside value-function
iteration. This Julia smoke notebook fits a GP to the closed-form
full-depreciation Brock-Mirman value and checks the induced consumption policy.
"""

# ╔═╡ bc361485-4252-4d1f-8aa7-8535ed04d634
md"""
## Lecture 14, Notebook 04: GP-based value-function iteration

**Course:** Deep Learning for Solving and Estimating Dynamic Models in Economics and Finance
**Script reference:** Chapter 9, §9.6
**Purpose:** value-function iteration for the one-state stochastic growth model using a Gaussian-process surrogate.

*Julia/Lux/Pluto preview of* `lectures/lecture_14_surrogates_and_gps/code/lecture_14_04_GP_Value_Function_Iteration.ipynb`.
"""

# ╔═╡ 6576b66b-023e-676e-c0dc-da420b141e24
md"""
### About this preview

The Python ground truth keeps the notebook focused on the dynamic-programming object and produces three figure families for the script:

1. 1D GP-VFI convergence and value-function uncertainty.
2. Same-budget 1D active enrichment versus a fixed Latin-hypercube design.
3. A deterministic full-depreciation Brock–Mirman verification against the closed-form solution.

> **Compact Julia preview.** Rather than re-running the adaptive GP-VFI loop, this Julia notebook fits the in-house Cholesky GP directly to the **closed-form deterministic full-depreciation Brock–Mirman value function** in \$\log k\$ space and checks the induced consumption policy against the analytical benchmark. The full Python notebook additionally builds the **Bellman oracle** (§3, one scalar nonlinear program per label), runs the **baseline GP-VFI loop** (§4), contrasts **few versus more Bellman design points** (§5), performs **active enrichment inside the loop** (§6, pure-exploration posterior-variance sampling), and argues (§9) why a separable 2D toy does not belong in a VFI section. None of that machinery runs here — the preview isolates the surrogate-accuracy and closed-form-verification steps.
"""

# ╔═╡ 22222222-1404-4222-8222-222222222222
begin
    import Pkg
    Pkg.activate(joinpath(@__DIR__, "..", "..", "..", "julia"))

    using DLEFJulia
    using Statistics
end

# ╔═╡ 87a40778-1114-c5a8-d8ca-59d6d30d1571
md"""
## 1. Model and numerical settings

The baseline model is the one-state stochastic growth problem

\$\$
V(k)=\max_{0<c<\bar c(k)}\left\{\log c+\beta \mathbb{E}_z[V(k')]\right\},
\qquad
k'=(1-\delta)k+z A k^\alpha-c.
\$\$

The expectation is approximated by Gauss–Hermite quadrature. Since the state is positive and the value function is close to logarithmic, the GP is fitted on the transformed input \$x = \log k\$, which makes the surrogate much more stationary and removes most of the boundary behaviour near low capital.

This preview specialises to the **deterministic full-depreciation** case (\$\delta = 1\$, \$z = 1\$; `BrockMirmanParams(delta = 1.0, beta = 0.96)`), whose value function has the closed form used in §8 below — `value(k)` implements it, and the savings rate is \$\alpha\beta\$.
"""

# ╔═╡ 33333333-1404-4333-8333-333333333333
begin
    RUN_MODE = "smoke"
    SEED = 0
    budgets = (
        smoke = (n_train = 12, n_eval = 50),
        teaching = (n_train = 40, n_eval = 200),
        production = (n_train = 120, n_eval = 800),
    )
    hp = run_mode_budget(RUN_MODE; budgets)
    bm = BrockMirmanParams(delta = 1.0, beta = 0.96)
    savings_rate = bm.alpha * bm.beta
    value(k) = log(1 - savings_rate) / (1 - bm.beta) +
        savings_rate * log(savings_rate) / ((1 - bm.beta) * (1 - savings_rate)) +
        (bm.alpha / (1 - savings_rate)) * log(k)
end

# ╔═╡ 75fbc7c0-f30b-452c-caf0-d7343f976477
md"""
## 2. Gaussian-process surrogate and diagnostics

The GP is the interpolation layer. The Python notebook uses a **Matérn-\$5/2\$** kernel in \$\log k\$ space, optimises the hyperparameters during the first few VFI iterations and then freezes them, and reads a nearly-free **leave-one-out RMSE** off the Cholesky factor. This Julia preview calls `fit_cholesky_gp` with a fixed length scale (\$0.55\$) and a small noise floor, then evaluates the posterior mean with `gp_predict` on a dense capital grid and reports the error against the exact closed-form value via `residual_summary`.
"""

# ╔═╡ 44444444-1404-4444-8444-444444444444
begin
    k_train = collect(range(0.3, 8.0; length = hp.n_train))
    x_train = reshape(log.(k_train), 1, :)
    y_train = reshape(value.(k_train), 1, :)
    gp = fit_cholesky_gp(x_train, y_train; lengthscale = 0.55, noise = 1e-6)
    k_eval = collect(range(0.35, 7.5; length = hp.n_eval))
    x_eval = reshape(log.(k_eval), 1, :)
    v_hat = gp_predict(gp, x_eval).mean
    v_exact = reshape(value.(k_eval), 1, :)
    gp_error = residual_summary(v_hat .- v_exact)
end

# ╔═╡ 24abf288-db80-bd93-777c-037f4ff02622
md"""
## 7. Policy recovery and 8. Closed-form verification

After the GP has converged, the Python notebook recovers the consumption and expected-shock savings policies by re-solving the scalar Bellman maximisation at validation states. Here we verify against the analytical **deterministic full-depreciation** Brock–Mirman solution, valid for \$\delta = 1\$, \$z = 1\$:

\$\$
c^*(k)=(1-\alpha\beta)A k^\alpha,\qquad
k'(k)=\alpha\beta A k^\alpha,
\$\$

\$\$
V^*(k)=\frac{\log(A(1-\alpha\beta))}{1-\beta}
+\frac{\alpha\beta\log(A\alpha\beta)}{(1-\beta)(1-\alpha\beta)}
+\frac{\alpha}{1-\alpha\beta}\log k.
\$\$

`bm_full_depreciation_policy` returns the savings \$k'(k)\$, so consumption is \$c(k) = k^\alpha - k'(k)\$; the cell compares it with \$c^*(k)\$ through `residual_summary`.
"""

# ╔═╡ 55555555-1404-4555-8555-555555555555
begin
    c_exact = (1 - savings_rate) .* k_eval .^ bm.alpha
    c_policy = (k_eval .^ bm.alpha .- bm_full_depreciation_policy(k_eval, bm))
    consumption_error = residual_summary(c_policy .- c_exact)
end

# ╔═╡ 0edc8a22-8d55-b615-121b-ad6970246c2b
md"""
## Summary

The preview supports the script's coherent story, isolating the surrogate and verification steps of the full GP-VFI notebook:

1. **GP-VFI is a Bellman solver in 1D.** In the full notebook every training label is generated by a Bellman maximisation; here the labels are the closed-form values.
2. **GP uncertainty is interpolation uncertainty.** It is a statement about the surrogate, not about policy uncertainty, unless propagated through the arg-max.
3. **LOO RMSE separates surrogate health from Bellman convergence.** A good surrogate can still be far from the fixed point.
4. **Active enrichment belongs inside the VFI loop.** The acquisition rule is pure exploration, because the goal is uniform value-function interpolation, not objective maximisation.
5. **The closed-form check is deterministic full-depreciation Brock–Mirman**, made explicit here by fixing \$\delta = 1\$ (and \$z = 1\$) for the verification run.

The cell below returns the machine-checkable diagnostics summary: the GP value-function RMSE and max absolute error, and the consumption-policy RMSE against the closed form.
"""

# ╔═╡ 66666666-1404-4666-8666-666666666666
(
    gp_value_rmse = gp_error.rmse,
    gp_value_max_abs = gp_error.max_abs,
    consumption_rmse = consumption_error.rmse,
    savings_rate = savings_rate,
)

# ╔═╡ Cell order:
# ╟─11111111-1404-4111-8111-111111111111
# ╟─bc361485-4252-4d1f-8aa7-8535ed04d634
# ╟─6576b66b-023e-676e-c0dc-da420b141e24
# ╠═22222222-1404-4222-8222-222222222222
# ╟─87a40778-1114-c5a8-d8ca-59d6d30d1571
# ╠═33333333-1404-4333-8333-333333333333
# ╟─75fbc7c0-f30b-452c-caf0-d7343f976477
# ╠═44444444-1404-4444-8444-444444444444
# ╟─24abf288-db80-bd93-777c-037f4ff02622
# ╠═55555555-1404-4555-8555-555555555555
# ╟─0edc8a22-8d55-b615-121b-ad6970246c2b
# ╠═66666666-1404-4666-8666-666666666666
