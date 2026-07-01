### A Pluto.jl notebook ###
# v0.20.0

using Markdown
using InteractiveUtils

# ╔═╡ 11111111-1410-4111-8111-111111111111
md"""
# Lecture 14, Notebook 10: Deep AS vs Linear AS on the Borehole Benchmark

The borehole benchmark is sampled in unit coordinates, mapped to the physical
Harper-Gupta marginals, and evaluated with the canonical water-flow formula.
This Lux preview compares the same two surrogates as the Python notebook:

1. a finite-difference linear active subspace with a cubic polynomial link;
2. a Tripathy-Bilionis-style deep active subspace with a Lux encoder/link model.

Smoke mode keeps the deep training budget small, so the result is a parity and
diagnostic check rather than a production convergence claim.
"""

# ╔═╡ 6f6b533f-79eb-96f5-913c-fd1249028a10
md"""
## Lecture 14, Notebook 10: Deep AS vs linear AS on the 8D Borehole benchmark

**Course:** Deep Learning for Solving and Estimating Dynamic Models in Economics and Finance
**Script reference:** §9.5 (Deep AS vs linear AS on the 8D Borehole UQ benchmark)
**Notebook role:** extension
**Author:** Simon Scheidegger

*Julia/Lux/Pluto preview of* `lectures/lecture_14_surrogates_and_gps/code/lecture_14_10_Deep_AS_vs_Linear_AS_Borehole.ipynb`.
"""

# ╔═╡ 3262f1fb-5445-67e4-0b57-00ae2a59c0c5
md"""
### Overview

The **borehole function** is the canonical \$D = 8\$ benchmark for uncertainty-quantification surrogate methods. It models the water flow \$y\$ through a borehole given eight physical inputs and has a known, *mostly-linear* active-subspace structure: one direction dominates, a second carries a smaller amount of information, and the remaining six are nearly inert.

In this notebook we

1. implement the borehole function and sample a training / validation set;
2. fit a **linear** active-subspace surrogate (gradient outer-product + cubic-polynomial link) for \$d \in \{1, \dots, 4\}\$;
3. fit a **deep** active-subspace surrogate (Tripathy & Bilionis, 2018) for the same \$d\$;
4. compare held-out RMSE.

The value of the comparison is diagnostic. As we will see, the two curves **cross each other**: deep AS is the stronger tool at \$d = 1\$ (it can curve the active direction), while a cubic polynomial on top of the top two linear directions is the more data-efficient choice at \$d \ge 2\$ for this nearly-ridge function.

> **Smoke mode.** The checked-in run uses a small deep-training budget, so the result is a parity and diagnostic check rather than a production-convergence claim. Set `RUN_MODE` in the next cell to `teaching`/`production` for the larger budgets.
"""

# ╔═╡ 22222222-1410-4222-8222-222222222222
begin
    import Pkg
    Pkg.activate(joinpath(@__DIR__, "..", "..", "..", "julia"))

    using DLEFJulia
    using Lux
    using Optimisers
    using Statistics
end

# ╔═╡ fccd787b-3851-8773-4a78-bb71ede5e890
md"""
### 1. The borehole function

\$\$f(r_w, r, T_u, H_u, T_l, H_l, L, K_w) = \frac{2\pi\, T_u\, (H_u - H_l)}{\ln(r/r_w)\,\bigl(1 + \tfrac{2 L T_u}{\ln(r/r_w)\, r_w^2\, K_w} + \tfrac{T_u}{T_l}\bigr)}.\$\$

Each input has a prescribed marginal distribution (Harper & Gupta, 1983). We sample in unit coordinates, map each sample to the physical marginals via `borehole_physical_from_unit`, and evaluate the water-flow formula with `borehole_function`; the composition `borehole_xi` therefore works directly with standardised inputs \$\xi \in [0,1]^8\$. The next cell draws the training / validation split and standardises the target (a constant-predictor RMSE is recorded as the baseline every surrogate must beat).
"""

# ╔═╡ 33333333-1410-4333-8333-333333333333
begin
    RUN_MODE = "smoke"
    SEED = 0
    budgets = (
        smoke = (n_train = 160, n_test = 320, deep_steps = 120, width = 24, lr = 0.005),
        teaching = (n_train = 500, n_test = 1_000, deep_steps = 400, width = 32, lr = 0.005),
        production = (n_train = 10_000, n_test = 20_000, deep_steps = 2_000, width = 32, lr = 0.005),
    )
    hp = run_mode_budget(RUN_MODE; budgets)
    rng = rng_from_seed(SEED)
    dims = 1:4

    borehole_xi(xi) = borehole_function(borehole_physical_from_unit(xi))
    rmse(pred, truth) = sqrt(mean(abs2, pred .- truth))
    population_std(y) = sqrt(mean(abs2, y .- mean(y)))
end

# ╔═╡ 44444444-1410-4444-8444-444444444444
begin
    xi_train = rand(rng, 8, hp.n_train)
    xi_test = rand(rng, 8, hp.n_test)
    xphys_train = borehole_physical_from_unit(xi_train)
    xphys_test = borehole_physical_from_unit(xi_test)
    y_train = borehole_function(xphys_train)
    y_test = borehole_function(xphys_test)

    y_mean = mean(y_train)
    y_std = population_std(y_train)
    y_train_s = (y_train .- y_mean) ./ y_std
    split_diagnostics = (
        n_train = hp.n_train,
        n_test = hp.n_test,
        y_min = minimum(y_train),
        y_max = maximum(y_train),
        y_mean = y_mean,
        y_std = y_std,
        constant_rmse = population_std(y_test),
    )
end

# ╔═╡ 07bf66dd-56b6-ea11-794a-1589dc325083
md"""
### 2. Linear active-subspace surrogate

We estimate the gradient outer-product matrix from **finite differences** on \$\xi \in [0,1]^8\$ (clipped at the box edges), eigendecompose it (`active_subspace`), and for each \$d \in \{1, \dots, 4\}\$ fit a degree-3 polynomial in the top \$d\$ active coordinates \$U_d^\top \xi\$ (`fit_active_subspace_surrogate`). Predictions are de-standardised before the held-out RMSE is computed.
"""

# ╔═╡ 55555555-1410-4555-8555-555555555555
begin
    function clipped_finite_difference_gradients(f, x::AbstractMatrix; h::Real = 1e-4)
        gradients = similar(float.(x))
        for i in axes(x, 1)
            xp = copy(x)
            xm = copy(x)
            xp[i, :] .= clamp.(@view(x[i, :]) .+ h, 0.0, 1.0)
            xm[i, :] .= clamp.(@view(x[i, :]) .- h, 0.0, 1.0)
            gradients[i, :] .= vec((f(xp) .- f(xm)) ./ (2h))
        end
        return gradients
    end

    gradients = clipped_finite_difference_gradients(borehole_xi, xi_train; h = 1e-4)
    as = active_subspace(active_subspace_matrix(gradients))

    function linear_as_metrics(d::Integer)
        fit = fit_active_subspace_surrogate(xi_train, y_train_s, as.vectors;
            dims = d, degree = 3, lambda = 1e-3)
        pred_s = predict_active_subspace_surrogate(fit, xi_test)
        pred = y_mean .+ y_std .* pred_s
        return (
            d = d,
            rmse = rmse(pred, y_test),
            relative_error = relative_l2_error(pred, y_test),
        )
    end

    linear_results = [linear_as_metrics(d) for d in dims]
end

# ╔═╡ 21a0c7c7-8537-d831-70c5-a65045c3082b
md"""
### 3. Deep active-subspace surrogate

We reuse the Tripathy & Bilionis (2018) architecture from the previous notebook — exponentially-decaying encoder widths, Swish activation, and an elastic-net penalty on the weights, with no orthogonality constraint (`make_deep_active_subspace(8, d; …)`). One model is trained per latent dimension \$d \in \{1, \dots, 4\}\$ with Adam on the standardised MSE plus the elastic-net penalty, and evaluated on the same held-out test set.
"""

# ╔═╡ 66666666-1410-4666-8666-666666666666
begin
    function matrix_elastic_net(ps; lam1::Real = 1e-5, lam2::Real = 1e-4)
        return _matrix_elastic_net(ps, lam1, lam2)
    end

    _matrix_elastic_net(x::AbstractArray, lam1, lam2) =
        ndims(x) >= 2 ? lam1 * sum(abs, x) + lam2 * sum(abs2, x) : zero(eltype(x))
    _matrix_elastic_net(x::NamedTuple, lam1, lam2) = _matrix_elastic_net(values(x), lam1, lam2)
    _matrix_elastic_net(x::Tuple{}, lam1, lam2) = 0.0
    _matrix_elastic_net(x::Tuple, lam1, lam2) =
        _matrix_elastic_net(first(x), lam1, lam2) + _matrix_elastic_net(Base.tail(x), lam1, lam2)
    _matrix_elastic_net(x, lam1, lam2) = 0.0

    function deep_as_metrics(d::Integer)
        model = make_deep_active_subspace(8, d; link_hidden = hp.width)
        state = setup_training(rng_from_seed(SEED; offset = d), model,
            Optimisers.Adam(hp.lr); parameter_type = Float64)

        deep_loss(model, ps, st, batch) = begin
            pred_s, st_new = model(batch.x, ps, st)
            loss = mse_loss(pred_s, batch.y) + matrix_elastic_net(ps; lam1 = 1e-5, lam2 = 1e-4)
            return loss, st_new
        end

        initial_loss = loss_value(state, deep_loss, (x = xi_train, y = y_train_s))
        final_loss = initial_loss
        for _ in 1:hp.deep_steps
            metrics = train_step!(state, deep_loss, (x = xi_train, y = y_train_s); max_grad_norm = 25.0)
            final_loss = metrics.loss
        end

        pred_s, _ = state.model(xi_test, state.ps, state.st)
        pred = y_mean .+ y_std .* pred_s
        return (
            d = d,
            initial_loss = initial_loss,
            final_loss = final_loss,
            rmse = rmse(pred, y_test),
            relative_error = relative_l2_error(pred, y_test),
        )
    end

    deep_results = [deep_as_metrics(d) for d in dims]
end

# ╔═╡ e0472d87-d3a2-ad05-8ebb-2e8b00980d3c
md"""
### What the curves actually show

The two methods **cross each other**:

| \$d\$ | linear AS + cubic link | deep AS (Tripathy–Bilionis) |
|:---:|:---:|:---:|
| 1  | large RMSE (one linear direction is insufficient) | small RMSE (encoder learns a nonlinear 1D aggregator) |
| 2+ | small RMSE (polynomial in two linear features fits well) | slightly larger RMSE (limited training budget, elastic-net bias) |

**Read this as a diagnostic, not a contest.**

- When you are *forced* to \$d = 1\$ (visualisation, policy-space exploration, or a downstream GP with very few training points), deep AS is stronger: it can curve the active direction.
- When the function is close to a polynomial in a few linear features — exactly the borehole at \$d \ge 2\$ — the top eigenvectors already identify a good basis and a cubic link on top of them is very data-efficient.
- The deep-AS pay-off grows with the **curvature of the active manifold** (notebook 09, radial ridge) and with **input dimension** (\$D \gg 10\$).

**Rule of thumb.** Always fit linear AS first and inspect the spectrum. Escalate to deep AS if (i) no gradient samples are available, (ii) the spectral gap is ambiguous, or (iii) the physics suggests a curved low-dimensional manifold.

**Reference.** R. Tripathy and I. Bilionis. *Deep UQ: learning deep neural network surrogate models for high-dimensional uncertainty quantification*. Journal of Computational Physics 375 (2018), 565–588.

The final cell collects both RMSE curves, the leading eigenvalues, the best dimension for each method, and the machine-checkable diagnostics (each surrogate beats the constant predictor; deep AS beats linear AS at \$d = 1\$).
"""

# ╔═╡ 77777777-1410-4777-8777-777777777777
begin
    linear_rmse = [r.rmse for r in linear_results]
    deep_rmse = [r.rmse for r in deep_results]
    linear_relative = [r.relative_error for r in linear_results]
    deep_relative = [r.relative_error for r in deep_results]

    comparison = (
        split = split_diagnostics,
        dims = Tuple(dims),
        leading_eigenvalues = Tuple(as.values[1:4]),
        linear_rmse = Tuple(linear_rmse),
        deep_rmse = Tuple(deep_rmse),
        linear_relative_error = Tuple(linear_relative),
        deep_relative_error = Tuple(deep_relative),
        deep_loss_pairs = Tuple((r.initial_loss, r.final_loss) for r in deep_results),
        best_linear_dim = dims[argmin(linear_rmse)],
        best_deep_dim = dims[argmin(deep_rmse)],
        linear_beats_constant = minimum(linear_rmse) < split_diagnostics.constant_rmse,
        deep_beats_constant = minimum(deep_rmse) < split_diagnostics.constant_rmse,
        deep_d1_beats_linear_d1 = deep_rmse[1] < linear_rmse[1],
    )

    if RUN_MODE == "smoke"
        @assert comparison.linear_beats_constant "Linear-AS surrogate did not beat the constant predictor in smoke mode"
    end

    comparison
end

# ╔═╡ Cell order:
# ╟─11111111-1410-4111-8111-111111111111
# ╟─6f6b533f-79eb-96f5-913c-fd1249028a10
# ╟─3262f1fb-5445-67e4-0b57-00ae2a59c0c5
# ╠═22222222-1410-4222-8222-222222222222
# ╟─fccd787b-3851-8773-4a78-bb71ede5e890
# ╠═33333333-1410-4333-8333-333333333333
# ╠═44444444-1410-4444-8444-444444444444
# ╟─07bf66dd-56b6-ea11-794a-1589dc325083
# ╠═55555555-1410-4555-8555-555555555555
# ╟─21a0c7c7-8537-d831-70c5-a65045c3082b
# ╠═66666666-1410-4666-8666-666666666666
# ╟─e0472d87-d3a2-ad05-8ebb-2e8b00980d3c
# ╠═77777777-1410-4777-8777-777777777777
