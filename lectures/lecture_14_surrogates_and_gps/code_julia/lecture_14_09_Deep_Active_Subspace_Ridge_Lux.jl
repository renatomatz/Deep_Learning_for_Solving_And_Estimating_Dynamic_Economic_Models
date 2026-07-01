### A Pluto.jl notebook ###
# v0.20.0

using Markdown
using InteractiveUtils

# ╔═╡ 11111111-1409-4111-8111-111111111111
md"""
# Lecture 14, Notebook 09: Deep Active-Subspace Ridge in Lux

The radial ridge has two linear directions but one nonlinear aggregator. This
translation builds the Lux encoder/link model and compares it with the linear
active-subspace spectrum in smoke mode.
"""

# ╔═╡ 568c7d99-9132-cade-b7a0-ecee61a60e55
md"""
## Lecture 14, Notebook 09: Deep active subspaces — Tripathy & Bilionis on a 20D radial ridge

**Course:** Deep Learning for Solving and Estimating Dynamic Models in Economics and Finance
**Script reference:** §9.5 (Deep active subspaces — Tripathy & Bilionis 2018 on a 20D radial-ridge function)
**Notebook role:** extension
**Author:** Simon Scheidegger

*Julia/Lux/Pluto preview of* `lectures/lecture_14_surrogates_and_gps/code/lecture_14_09_Deep_Active_Subspace_Ridge.ipynb`.
"""

# ╔═╡ 9445c969-e3b3-cc8f-376b-bc16fae2254a
md"""
### Overview

The *linear* active-subspace (AS) method approximates a scalar map \$f : \mathbb{R}^D \to \mathbb{R}\$ as

\$\$f(\xi) \approx g\bigl(U_m^\top \xi\bigr), \qquad U_m \in \mathbb{R}^{D \times d},\$\$

a function of \$d \ll D\$ *linear* projections of the inputs, where \$U_m\$ spans the top eigenvectors of the gradient outer-product matrix \$C = \mathbb{E}[\nabla f\,\nabla f^\top]\$.

**Failure mode.** If the function varies along a *curved* manifold, linear projections cannot capture the low intrinsic dimension, and one needs many components.

**Fix (Tripathy & Bilionis, 2018).** Replace the linear encoder with a deep neural network,

\$\$\hat f(\xi) = g\bigl(h(\xi)\bigr), \qquad h : \mathbb{R}^D \to \mathbb{R}^d,\quad g : \mathbb{R}^d \to \mathbb{R},\$\$

and train \$h\$ and \$g\$ *jointly* by minimising the data loss plus an elastic-net penalty, choosing \$d\$ from the validation-MSE elbow. No gradient samples or Stiefel-manifold constraint are required.

**What this notebook does.** It defines a radial-ridge target \$y(\xi) = \exp(-[(w_1^\top\xi)^2 + (w_2^\top\xi)^2])\$ in \$D = 20\$ dimensions, shows that *linear* AS needs \$d_{\text{lin}} = 2\$ directions, and trains a *deep* AS surrogate to confirm the true nonlinear intrinsic dimension \$d_{\text{nl}} = 1\$. (This Julia preview builds the encoder/link model in Lux and trains it at \$d = 1\$; the full Python notebook sweeps \$d \in \{1, 2, 3, 4\}\$, reads the intrinsic dimension off the validation-MSE elbow, and plots the learned latent coordinate against the true radius.)
"""

# ╔═╡ 22222222-1409-4222-8222-222222222222
begin
    import Pkg
    Pkg.activate(joinpath(@__DIR__, "..", "..", "..", "julia"))

    using DLEFJulia
    using LinearAlgebra
    using Lux
    using Optimisers
    using Statistics
end

# ╔═╡ 07c4dd5e-7dd8-af90-8280-6cccf8b27c23
md"""
### 1. The radial-ridge target

We pick two fixed orthogonal unit vectors \$w_1, w_2 \in \mathbb{R}^{20}\$ (here the first two columns of a QR factorisation) and define

\$\$y(\xi) = \exp\!\Bigl(-\bigl[(w_1^\top \xi)^2 + (w_2^\top \xi)^2\bigr]\Bigr), \qquad \xi \sim \mathcal{N}(0, I_{20}).\$\$

The aggregator \$r^2 = (w_1^\top\xi)^2 + (w_2^\top\xi)^2\$ is a *nonlinear* function of two linear features, so

- the **linear** AS sees two relevant directions (\$d_{\text{lin}} = 2\$);
- a **nonlinear** encoder can collapse them into a single scalar (\$d_{\text{nl}} = 1\$).
"""

# ╔═╡ 33333333-1409-4333-8333-333333333333
begin
    RUN_MODE = "smoke"
    SEED = 0
    budgets = (
        smoke = (steps = 3, n = 64, width = 12),
        teaching = (steps = 600, n = 4_000, width = 32),
        production = (steps = 2_000, n = 16_000, width = 64),
    )
    hp = run_mode_budget(RUN_MODE; budgets)
    rng = rng_from_seed(SEED)
    q, _ = qr(rand(rng, 20, 2))
    directions = Matrix(q[:, 1:2])
end

# ╔═╡ 6bc218c2-1d15-367a-9953-8e31ecf8d58f
md"""
### 2. Linear active-subspace baseline

We form the gradient outer-product matrix

\$\$C = \frac{1}{N} \sum_{i=1}^N \nabla y(\xi_i)\,\nabla y(\xi_i)^\top\$\$

(`active_subspace_matrix`) and inspect its spectrum (`active_subspace`). By construction only \$w_1\$ and \$w_2\$ carry information, so \$C\$ has rank \$2\$: the spectrum drops sharply after index \$2\$, confirming that a *linear* surrogate needs \$d_{\text{lin}} = 2\$. Next we see that a *deep* surrogate needs only one.
"""

# ╔═╡ 2b39bf20-4cac-523f-f242-fcd46d8d5fbd
md"""
### 3. Deep active-subspace architecture

Following Tripathy & Bilionis (2018) we parametrise \$\hat y(\xi) = g\bigl(h(\xi)\bigr)\$ with

- **encoder** \$h : \mathbb{R}^D \to \mathbb{R}^d\$ — an MLP whose layer widths decay exponentially from \$D\$ to \$d\$ (Eq. 20): \$d_k = \lceil D \exp(\rho k) \rceil\$, \$\rho = L^{-1}\log(d/D)\$ (the `encoder_widths` helper);
- **link** \$g : \mathbb{R}^d \to \mathbb{R}\$ — a small MLP;
- **activation** the Swish \$\sigma(z) = z / (1 + \exp(-z))\$;
- **training loss** data MSE plus an elastic-net penalty \$\lambda_1\lVert W\rVert_1 + \lambda_2\lVert W\rVert_2^2\$ on all weights.

`make_deep_active_subspace(20, 1; …)` builds this encoder/link model in Lux at \$d = 1\$. Crucially, no gradient samples and no orthogonality constraint are needed.
"""

# ╔═╡ 44444444-1409-4444-8444-444444444444
begin
    x = randn(rng, 20, hp.n)
    y = radial_ridge_target(x, directions)
    gradients = radial_ridge_gradients(x, directions)
    as = active_subspace(active_subspace_matrix(gradients))
    deep_as = make_deep_active_subspace(20, 1; link_hidden = hp.width)
    state = setup_training(rng_from_seed(SEED; offset = 1), deep_as, Optimisers.Adam(0.01); parameter_type = Float64)
    deep_loss(model, ps, st, batch) = begin
        yhat, st_new = model(batch.x, ps, st)
        return mse_loss(yhat, batch.y), st_new
    end
end

# ╔═╡ f60cab5f-1414-1c8e-3785-f094046c8a76
md"""
### 4. Train and score the deep surrogate

We train the \$d = 1\$ deep AS with Adam on the data MSE, record the loss history, then compute the coefficient of determination \$R^2 = 1 - \operatorname{MSE}/\operatorname{Var}(y)\$ on the training points. A *constant* predictor (the sample mean) would score \$R^2 = 0\$; the honest criterion for the intrinsic dimension is *where the validation curve flattens*.

*The full Python notebook* trains one model per \$d \in \{1, 2, 3, 4\}\$ and reads the intrinsic dimension off the elbow of that curve: at \$d = 1\$ the deep encoder already captures essentially all of the response variance (\$R^2 > 99.8\%\$ at the production budget), and \$d \ge 2\$ adds no new structure — because \$r^2\$ is a nonlinear aggregator of two linear features, the deep encoder learns it in a single dimension.
"""

# ╔═╡ 55555555-1409-4555-8555-555555555555
begin
    initial_loss = loss_value(state, deep_loss, (x = x, y = y))
    history = NamedTuple[]
    for step in 1:hp.steps
        metrics = train_step!(state, deep_loss, (x = x, y = y); max_grad_norm = 25.0)
        append_metric!(history; step, loss = metrics.loss)
    end
    yhat, _ = state.model(x, state.ps, state.st)
    r2 = 1 - mean(abs2, yhat .- y) / mean(abs2, y .- mean(y))
end

# ╔═╡ 9f1778e6-5098-1b86-29bd-90e3744c9389
md"""
### Summary

| method    | intrinsic dimension needed | reason |
|-----------|:--:|--------|
| linear AS | \$d_{\text{lin}} = 2\$ | two linear directions \$w_1, w_2\$ both carry signal |
| deep AS   | \$d_{\text{nl}} = 1\$  | encoder learns the nonlinear aggregator \$r^2\$ in one dimension |

Take-aways:

- Deep AS strictly generalises linear AS: setting \$h(\xi) = U_m^\top\xi\$ recovers the linear case.
- Linear AS requires gradient samples of \$f\$; deep AS is gradient-free and learns the low-dimensional structure directly from \$(\xi, y)\$ pairs.
- The right \$d\$ comes from the validation-MSE elbow rather than the spectral gap of \$C\$.
- The encoder-width formula \$d_k = \lceil D \exp(\rho k)\rceil\$ smoothly interpolates between the input and latent dimensions and avoids a brittle hyperparameter choice.

**Reference.** R. Tripathy and I. Bilionis. *Deep UQ: learning deep neural network surrogate models for high-dimensional uncertainty quantification*. Journal of Computational Physics 375 (2018), 565–588.

The cell below returns the machine-checkable diagnostics: the linear-AS rank-2 spectral gap, the encoder widths, the initial and final training loss, and the smoke-run \$R^2\$ at \$d = 1\$.
"""

# ╔═╡ 66666666-1409-4666-8666-666666666666
(
    linear_rank2_gap = as.values[2] / max(as.values[3], eps(Float64)),
    encoder_widths = Tuple(encoder_widths(20, 1, 3)),
    initial_loss = initial_loss,
    final_loss = history[end].loss,
    smoke_r2 = r2,
)

# ╔═╡ Cell order:
# ╟─11111111-1409-4111-8111-111111111111
# ╟─568c7d99-9132-cade-b7a0-ecee61a60e55
# ╟─9445c969-e3b3-cc8f-376b-bc16fae2254a
# ╠═22222222-1409-4222-8222-222222222222
# ╟─07c4dd5e-7dd8-af90-8280-6cccf8b27c23
# ╠═33333333-1409-4333-8333-333333333333
# ╟─6bc218c2-1d15-367a-9953-8e31ecf8d58f
# ╟─2b39bf20-4cac-523f-f242-fcd46d8d5fbd
# ╠═44444444-1409-4444-8444-444444444444
# ╟─f60cab5f-1414-1c8e-3785-f094046c8a76
# ╠═55555555-1409-4555-8555-555555555555
# ╟─9f1778e6-5098-1b86-29bd-90e3744c9389
# ╠═66666666-1409-4666-8666-666666666666
