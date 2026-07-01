### A Pluto.jl notebook ###
# v0.20.0

using Markdown
using InteractiveUtils

# ╔═╡ 11111111-0701-4111-8111-111111111111
md"""
# Lecture 07, Notebook 01: Automatic Differentiation Examples in Julia

The Python notebook used TensorFlow tapes. This Pluto version uses
`ForwardDiff` for small input derivatives, which is the convention used by the
Julia translation track for analytical checks and PINN inputs.
"""

# ╔═╡ ee7e728f-67e2-6062-da84-0c4f9c95c2ae
md"""
## Lecture 07, Notebook 01: Automatic Differentiation — Analytical Examples

**Course:** Deep Learning for Solving and Estimating Dynamic Models in Economics and Finance
**Script reference:** §2.7.1–2.7.2 (three ways to differentiate; computational graph; forward/reverse modes), Appendix B (matrix calculus)
**Notebook role:** core
**Author:** Simon Scheidegger

*Julia/Lux/Pluto preview of* `lectures/lecture_07_autodiff_for_deqns/code/lecture_07_01_AutoDiff_Analytical_Examples.ipynb`.

> **Run mode.** The checked-in run uses `RUN_MODE = "smoke"` and `SEED = 0`. Autodiff here uses `ForwardDiff` for the small input and second derivatives, in place of the Python notebook's `tf.GradientTape`.
"""

# ╔═╡ 5d193da1-1948-28c5-8046-d11fd342823a
md"""
## Automatic Differentiation: Analytical Examples

### Notebook 1 (Day 4): companion to the autodiff lecture (deck `05b_AutoDiff_for_DEQN`)

This notebook accompanies the Day 4 morning lecture on automatic differentiation. It walks through six small, self-contained examples that economists will recognise:

1. **Warm-up.** A single-line autodiff demo on \$y = x^2 + \sin(x)\$.
2. **Finite differences in practice.** Recreate the U-curve plot from the slides; compare against autodiff.
3. **CRRA utility.** Compute \$u'(c)\$ and \$u''(c)\$ via nested differentiation.
4. **Cobb-Douglas production.** A genuinely *multi-variate* example: marginal products \$f_K, f_L\$ as a 2D gradient field.
5. **Capital adjustment cost.** A second 2D example, with messy hand derivation vs one-line autodiff.
6. **The Hessian.** Second-order autodiff via `ForwardDiff.hessian`.

The two notebooks `02_Brock_Mirman_AutoDiff_DEQN` and `03_Brock_Mirman_Uncertainty_AutoDiff_DEQN` then apply exactly this same machinery to a full dynamic stochastic model.

In this Julia/Lux preview the autodiff calls are `ForwardDiff.derivative`, `ForwardDiff.gradient`, and `ForwardDiff.hessian` rather than TensorFlow tapes; every numerical cross-check against the hand-derived formula is preserved.
"""

# ╔═╡ 22222222-0701-4222-8222-222222222222
begin
    import Pkg
    Pkg.activate(joinpath(@__DIR__, "..", "..", "..", "julia"))

    using CairoMakie
    using DLEFJulia
    using ForwardDiff
    using LinearAlgebra
end

# ╔═╡ 33333333-0701-4333-8333-333333333333
begin
    RUN_MODE = "smoke"
    SEED = 0
end

# ╔═╡ a3dddabd-5e6b-2ed2-90d5-9430069c9a40
md"""
### 1. Warm-up: \$y = x^2 + \sin(x)\$

Take a function whose derivative we can compute by hand:
\$\$f(x) = x^2 + \sin(x), \qquad f'(x) = 2x + \cos(x).\$\$
At \$x_0 = 2\$ the analytical answer is \$f'(2) = 4 + \cos(2) \approx 3.5839\$.

The computer takes one line per derivative — here `ForwardDiff.derivative(f1, x0)`.

**Three things to notice.** (i) Forward-mode autodiff propagates a dual number through the forward computation. (ii) `ForwardDiff.derivative` returns the exact derivative of the expression it evaluated. (iii) The answer agrees with the closed-form derivative to machine precision — there is no \$h\$, no truncation, no roundoff.
"""

# ╔═╡ 44444444-0701-4444-8444-444444444444
begin
    f1(x) = x^2 + sin(x)
    x0 = 2.0
    ad_f1 = ForwardDiff.derivative(f1, x0)
    hand_f1 = 2x0 + cos(x0)
    warmup_error = abs(ad_f1 - hand_f1)
end

# ╔═╡ 1f52b99e-1f39-9780-ee8d-763e8b50702e
md"""
### 2. Finite differences in practice

The lecture showed the *theoretical* U-curve. Here we plot the *actual* error of central finite differences applied to \$f(x) = e^x\$ at \$x = 1\$, where \$f'(1) = e \approx 2.7182818\$.

Central FD: \$\widehat{f'}(x_0) = \dfrac{f(x_0 + h) - f(x_0 - h)}{2h}\$, with leading error \$\tfrac{1}{6} f'''(x_0)\, h^2\$ and a roundoff floor of order \$\epsilon / h\$ (for double precision \$\epsilon \approx 10^{-16}\$).

This cell computes the finite-difference error curve `fd_errors` and the single-point `ForwardDiff` reference error; the U-curve figure itself is rendered near the end of the notebook, because the Julia preview computes all numerical cross-checks first and then draws its one figure.

> **Float precision.** In double precision (`Float64`, Julia's default here) the finite-difference U-curve reveals the rounding regime. In `Float32` the rounding floor sits near `1e-6` and the U collapses; the DEQN training loops in notebooks 02–04 train in single precision for speed, so do not expect this U-curve there.
"""

# ╔═╡ 55555555-0701-4555-8555-555555555555
begin
    fd_central(f, x, h) = (f(x + h) - f(x - h)) / (2h)
    hs = 10.0 .^ range(-16, -1; length = 80)
    fd_errors = [abs(fd_central(exp, 1.0, h) - exp(1.0)) for h in hs]
    ad_exp_error = abs(ForwardDiff.derivative(exp, 1.0) - exp(1.0))
    best_fd = minimum(fd_errors)
end

# ╔═╡ 4d96a8cb-a47f-ae44-e5f8-c9ba83ecc4d7
md"""
### 3. CRRA utility: \$u(c) = c^{1-\gamma} / (1-\gamma)\$

The standard isoelastic (CRRA) utility:
\$\$u(c) = \frac{c^{1-\gamma}}{1-\gamma}, \qquad u'(c) = c^{-\gamma}, \qquad u''(c) = -\gamma\, c^{-\gamma-1}.\$\$

We compute *both* derivatives — \$u'\$ via `ForwardDiff.derivative` and \$u''\$ via the `second_derivative` helper (nested forward-mode) — and confirm they match the closed-form expressions. The full Python notebook also plots \$u\$, \$u'\$, and \$u''\$ together over a 200-point consumption grid; the Julia preview computes the same derivatives on a coarser 100-point grid and keeps only the numerical cross-check `crra_errors` in place of the plot.

**Take-away.** Change `gamma` to `4.0` — the code does not move. The autodiff answer follows automatically. The same cell can produce the marginal utility for *any* utility function we plug into `utility`.
"""

# ╔═╡ 66666666-0701-4666-8666-666666666666
begin
    gamma = 2.0
    utility(c) = c^(1 - gamma) / (1 - gamma)
    c_grid = collect(range(0.2, 5.0; length = 100))
    up_ad = [ForwardDiff.derivative(utility, c) for c in c_grid]
    upp_ad = [second_derivative(utility, c) for c in c_grid]
    up_hand = c_grid .^ (-gamma)
    upp_hand = -gamma .* c_grid .^ (-gamma - 1)
    crra_errors = (
        marginal = maximum(abs.(up_ad .- up_hand)),
        second = maximum(abs.(upp_ad .- upp_hand)),
    )
end

# ╔═╡ 04171c15-a38a-6b9f-5401-4f4661bc05fb
md"""
### 4. Cobb-Douglas production: a genuinely 2D example

\$\$f(K, L) = K^{\alpha} L^{1-\alpha}, \qquad
f_K = \alpha K^{\alpha - 1} L^{1-\alpha}, \qquad
f_L = (1-\alpha) K^{\alpha} L^{-\alpha}.\$\$

Here one `ForwardDiff.gradient` call returns both partials \$(f_K, f_L)\$ at a point, and we cross-check against the closed-form expressions.

**What this example makes vivid.** Each gradient vector points in the direction of steepest increase of \$f\$; the marginal products are large near the axes (one input scarce) and smaller in the interior (decreasing returns). All of that behaviour falls out of one gradient call — we never wrote a partial derivative. The full Python notebook visualises this as a `quiver` gradient field over a 2D grid; the Julia preview keeps the numerical cross-check `cobb_douglas_errors` in place of the plot.
"""

# ╔═╡ e3e9da40-6baa-e36b-d14a-d0274afc9cec
md"""
### 6. Bonus: the Hessian via `ForwardDiff.hessian`

> **Section order.** The Hessian is shown here — right after Cobb-Douglas (section 4) and before the capital adjustment cost (section 5) — because it reuses the very same Cobb-Douglas primitive: one code cell computes both the gradient and the Hessian. The intro list above keeps the slide order (5 before 6); only the presentation is folded together.

The Hessian of Cobb-Douglas at a point \$(K, L)\$ has three independent entries:
\$\$H = \begin{pmatrix} f_{KK} & f_{KL} \\ f_{KL} & f_{LL} \end{pmatrix}.\$\$

The same Cobb-Douglas cell computes the whole matrix with one `ForwardDiff.hessian` call (nested forward-mode) and cross-checks it against the hand-derived entries.

- The Hessian is symmetric — both autodiff and the hand formula recover this exactly. (Autodiff does *not* assume symmetry; it gets it because it correctly computes second derivatives.)
- The same nested pattern generalises to *any* Hessian. Trying this with finite differences would lose roughly 10 digits of precision; here we lose zero.
- The same idea, applied to a neural network's loss, is what produces curvature information for second-order optimisers (K-FAC, Newton-CG, etc.).
"""

# ╔═╡ 77777777-0701-4777-8777-777777777777
begin
    alpha = 0.36
    cobb_douglas(v) = v[1]^alpha * v[2]^(1 - alpha)
    point = [3.0, 2.0]
    grad_ad = ForwardDiff.gradient(cobb_douglas, point)
    hess_ad = ForwardDiff.hessian(cobb_douglas, point)
    grad_hand = [
        alpha * point[1]^(alpha - 1) * point[2]^(1 - alpha),
        (1 - alpha) * point[1]^alpha * point[2]^(-alpha),
    ]
    hess_hand = [
        alpha * (alpha - 1) * point[1]^(alpha - 2) * point[2]^(1 - alpha) alpha * (1 - alpha) * point[1]^(alpha - 1) * point[2]^(-alpha);
        alpha * (1 - alpha) * point[1]^(alpha - 1) * point[2]^(-alpha) (1 - alpha) * (-alpha) * point[1]^alpha * point[2]^(-alpha - 1)
    ]
    cobb_douglas_errors = (
        gradient = norm(grad_ad - grad_hand, Inf),
        hessian = norm(hess_ad - hess_hand, Inf),
    )
end

# ╔═╡ 11126ca8-d866-6014-e901-6d73f18b6b25
md"""
### 5. Capital adjustment cost: hand vs autodiff

The convex adjustment cost from the Day 3 IRBC model:
\$\$\Gamma(K, K') = \frac{\kappa}{2}\, K \left(\frac{K'}{K} - 1\right)^2.\$\$

By hand:
\$\$\frac{\partial \Gamma}{\partial K'} = \kappa \left(\frac{K'}{K} - 1\right), \qquad
\frac{\partial \Gamma}{\partial K} = \frac{\kappa}{2}\left(\frac{K'}{K} - 1\right)^2
- \kappa\,\frac{K'}{K}\left(\frac{K'}{K} - 1\right).\$\$

The first partial is easy. The second needs the quotient and chain rules and is exactly the kind of expression where typos hide. Autodiff: one `ForwardDiff.gradient` line.

**What the cross-check shows.** On the diagonal \$K' = K\$ the cost is zero and \$\nabla \Gamma = 0\$; off the diagonal the gradient points back towards the diagonal. Autodiff agrees with the hand-derived expression to ~\$10^{-15}\$ (`adjustment_error`). We did *not* have to differentiate by hand to trust it — we ran both and compared. The Python notebook also draws the contour-plus-quiver field; the Julia preview keeps the numerical comparison.
"""

# ╔═╡ 88888888-0701-4888-8888-888888888888
begin
    kappa = 0.3
    adj_cost(v) = 0.5 * kappa * v[1] * (v[2] / v[1] - 1)^2
    adj_point = [2.5, 3.0]
    adj_grad_ad = ForwardDiff.gradient(adj_cost, adj_point)
    ratio = adj_point[2] / adj_point[1]
    adj_grad_hand = [
        0.5 * kappa * (ratio - 1)^2 - kappa * ratio * (ratio - 1),
        kappa * (ratio - 1),
    ]
    adjustment_error = norm(adj_grad_ad - adj_grad_hand, Inf)
end

# ╔═╡ 94bf7058-1254-3841-4007-0d7a0b4fa179
md"""
**Reading the finite-difference plot.**

This figure plots the finite-difference error curve `fd_errors` computed above against the `ForwardDiff` reference.

- The finite-difference errors form the classic U-curve: error first falls as \$h^2\$ (truncation), then climbs as \$\epsilon/h\$ (catastrophic cancellation).
- The minimum sits around \$h^\star \approx \sqrt{\epsilon} \approx 10^{-5.3}\$, with best-case error \$\sim 10^{-11}\$ — we have lost ~5 digits.
- Autodiff hits machine precision at *zero* \$h\$ tuning.

The full Python figure also overlays the two theoretical asymptote lines — truncation \$\sim h^2\$ and roundoff \$\sim \epsilon/h\$ — that bracket the U from below and above; this preview draws only the measured `fd_errors` curve and the `ForwardDiff` reference, with those two regimes described in the bullets above.

For Hessians and higher derivatives, finite differences lose many more digits and are essentially unusable for moderate-precision work.
"""

# ╔═╡ 99999999-0701-4999-8999-999999999999
begin
    fig = Figure(size = figure_size(RUN_MODE))
    ax = Axis(fig[1, 1], xscale = log10, yscale = log10, xlabel = "h", ylabel = "absolute error")
    lines!(ax, hs, fd_errors; color = :dodgerblue3, label = "central FD")
    hlines!(ax, [max(ad_exp_error, eps())]; color = :darkorange, linestyle = :dash, label = "ForwardDiff")
    axislegend(ax; position = :rb)
    fig
end

# ╔═╡ 9aaa5f53-103b-43d5-338c-dd80c3d4757d
md"""
### 7. Summary

You have just used `ForwardDiff` to:
- compute first and second derivatives of one-variable functions (warm-up, CRRA),
- compute gradients on 2D examples (Cobb-Douglas, capital adjustment cost),
- regenerate the textbook finite-difference U-curve and confirm autodiff sits at machine precision,
- assemble a full Hessian with one `ForwardDiff.hessian` call.

In every example the user writes only the **model primitive** — the utility function, the production function, the cost function. All derivatives are computed by autodiff, exactly to machine precision, with no hand algebra and no finite-difference step-size tuning.

#### Where to go next

The same pattern — write only the *period payoff* \$\Pi(K_t, K_{t+1}, z_t)\$, then differentiate — is what powers the Brock-Mirman DEQN solvers in:

- `02_Brock_Mirman_AutoDiff_DEQN` — deterministic case; cross-checks against the analytical \$K_{t+1} = \alpha\beta K_t^\alpha\$.
- `03_Brock_Mirman_Uncertainty_AutoDiff_DEQN` — AR(1) productivity with Gauss-Hermite; cross-checks against the closed-form \$K_{t+1} = \alpha\beta z_t K_t^\alpha\$ in the full-depreciation side-experiment.

Both notebooks include numerical cross-checks of the autodiff loss against the hand-derived Euler residual at machine precision (~\$10^{-6}\$ in float32). The cell below returns this notebook's machine-checkable error summary.
"""

# ╔═╡ aaaaaaaa-0701-4aaa-8aaa-aaaaaaaaaaaa
(
    warmup_error = warmup_error,
    best_fd_error = best_fd,
    autodiff_exp_error = ad_exp_error,
    crra = crra_errors,
    cobb_douglas = cobb_douglas_errors,
    adjustment_cost = adjustment_error,
)

# ╔═╡ Cell order:
# ╟─11111111-0701-4111-8111-111111111111
# ╟─ee7e728f-67e2-6062-da84-0c4f9c95c2ae
# ╟─5d193da1-1948-28c5-8046-d11fd342823a
# ╠═22222222-0701-4222-8222-222222222222
# ╠═33333333-0701-4333-8333-333333333333
# ╟─a3dddabd-5e6b-2ed2-90d5-9430069c9a40
# ╠═44444444-0701-4444-8444-444444444444
# ╟─1f52b99e-1f39-9780-ee8d-763e8b50702e
# ╠═55555555-0701-4555-8555-555555555555
# ╟─4d96a8cb-a47f-ae44-e5f8-c9ba83ecc4d7
# ╠═66666666-0701-4666-8666-666666666666
# ╟─04171c15-a38a-6b9f-5401-4f4661bc05fb
# ╟─e3e9da40-6baa-e36b-d14a-d0274afc9cec
# ╠═77777777-0701-4777-8777-777777777777
# ╟─11126ca8-d866-6014-e901-6d73f18b6b25
# ╠═88888888-0701-4888-8888-888888888888
# ╟─94bf7058-1254-3841-4007-0d7a0b4fa179
# ╠═99999999-0701-4999-8999-999999999999
# ╟─9aaa5f53-103b-43d5-338c-dd80c3d4757d
# ╠═aaaaaaaa-0701-4aaa-8aaa-aaaaaaaaaaaa
