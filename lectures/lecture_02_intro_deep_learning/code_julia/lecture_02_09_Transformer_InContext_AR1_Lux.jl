### A Pluto.jl notebook ###
# v0.20.0

using Markdown
using InteractiveUtils

# ╔═╡ 11111111-0209-4111-8111-111111111111
md"""
# Lecture 02, Notebook 09: In-Context AR(1) Forecasting

The Python notebook trains a tiny Transformer. This Phase-0 Julia translation
keeps the econometric point in a smaller Lux forecaster: each example supplies a
prompt-specific OLS slope, and the model learns how to turn that context into a
next-period forecast.
"""

# ╔═╡ a708968e-09ac-faa4-7a00-40beb9773530
md"""
## Lecture 02, Notebook 09: Transformer In-Context Learning for AR(1) Forecasting

**Course:** Deep Learning for Solving and Estimating Dynamic Models in Economics and Finance
**Script reference:** §1.11 (In-context learning: Transformer for AR(1) forecasting)
**Notebook role:** extension
**Author:** Simon Scheidegger

*Julia/Lux/Pluto preview of* `lectures/lecture_02_intro_deep_learning/code/lecture_02_09_Transformer_InContext_AR1.ipynb`.

> **Run mode.** The checked-in run uses `RUN_MODE = "smoke"` for fast execution; set `RUN_MODE` to `"teaching"` or `"production"` in the setup cell for larger batches and longer sequences.
"""

# ╔═╡ 7e9b0687-b52e-f429-3539-0d7f86e7872b
md"""
## In-Context Learning of an AR(1) Process

This is the **advanced / optional** day-1 transformer notebook. The core lecture message is simpler — self-attention lets each token search the whole sequence directly, in parallel. This notebook pushes further into the econometric interpretation: self-attention can behave like a **learned regression rule**.

The smallest setting that makes this visible is **AR(1) forecasting**. A model is trained on many AR(1) paths, each with its own persistence \$\rho\$, then evaluated on new paths with unseen \$\rho\$ and no weight updates.

**Phase-0 Julia preview.** Instead of a tiny Transformer that reads the raw sequence and *implicitly* learns to regress, this compact Lux forecaster is handed explicit **context features** of each prompt — the last value, the OLS-on-the-prompt slope \$\hat{\rho}_{\text{OLS}}\$, and the path volatility — and learns to combine them into a next-period forecast. This makes the "learned regression rule" idea concrete: we can watch the model track the OLS-on-the-prompt predictor directly, and compare both against the naive last-value baseline.

### Reading guide

1. Generate many AR(1) sequences, each with its own \$\rho \sim \mathcal{U}(-0.9, 0.9)\$.
2. Train the compact Lux forecaster on their context features to predict the next value.
3. At inference, feed it **new** \$\rho\$'s and compare against the OLS-on-the-prompt estimator \$\hat{\rho}_{\text{OLS}}\, x_t\$ and the last-value baseline.
4. Interpret the result as the model learning **how to regress**, not memorising any one \$\rho\$.

Keep the hyperparameters classroom-sized; one CPU run is a few seconds at smoke size.
"""

# ╔═╡ 22222222-0209-4222-8222-222222222222
begin
    import Pkg
    Pkg.activate(joinpath(@__DIR__, "..", "..", "..", "julia"))

    using CairoMakie
    using DLEFJulia
    using Lux
    using NNlib
    using Optimisers
    using Statistics
end

# ╔═╡ 33333333-0209-4333-8333-333333333333
begin
    RUN_MODE = "smoke"
    SEED = 0
    budgets = (
        smoke = (steps = 45, batch = 180, T = 18),
        teaching = (steps = 350, batch = 800, T = 24),
        production = (steps = 1_500, batch = 2_000, T = 28),
    )
    hp = run_mode_budget(RUN_MODE; budgets)
    rng = rng_from_seed(SEED)
end

# ╔═╡ 61cfb9ee-d34a-9512-42f2-0ba6bcf2cc1f
md"""
### 1. AR(1) data generator

Each example is a sequence of length \$T\$ drawn from
\$\$x_t = \rho\, x_{t-1} + \sigma \varepsilon_t, \qquad \varepsilon_t \sim \mathcal{N}(0, 1),\$\$
with \$\rho\$ **drawn once per example** from \$\mathcal{U}(-0.9, 0.9)\$. The model never sees \$\rho\$; it only sees the realizations. The next cell samples these prompts and reduces each to its context features — the last value, the OLS-on-the-prompt slope \$\hat{\rho}_{\text{OLS}} = \frac{\sum_i x_i x_{i-1}}{\sum_i x_{i-1}^2}\$, and the volatility — alongside the OLS and naive (last-value) benchmark predictions.
"""

# ╔═╡ 44444444-0209-4444-8444-444444444444
begin
    function sample_ar1_prompts(rng, n, T; sigma = 0.3)
        rho = -0.9 .+ 1.8 .* rand(rng, n)
        paths = zeros(Float64, T + 1, n)
        paths[1, :] .= sigma .* randn(rng, n)
        for t in 2:(T + 1)
            paths[t, :] .= rho .* paths[t - 1, :] .+ sigma .* randn(rng, n)
        end
        return paths, rho
    end

    function ols_rho(paths)
        lag = paths[1:(end - 2), :]
        lead = paths[2:(end - 1), :]
        return vec(sum(lag .* lead; dims = 1) ./ (sum(abs2, lag; dims = 1) .+ 1e-8))
    end

    function context_features(paths)
        T = size(paths, 1) - 1
        slope = ols_rho(paths)
        last_value = paths[T, :]
        volatility = vec(sqrt.(mean(abs2, diff(paths[1:T, :]; dims = 1); dims = 1)))
        return vcat(reshape(last_value, 1, :), reshape(slope, 1, :), reshape(volatility, 1, :))
    end

    paths, rho = sample_ar1_prompts(rng, hp.batch, hp.T)
    x = context_features(paths)
    y = reshape(paths[end, :], 1, :)
    ols_prediction = reshape(ols_rho(paths) .* paths[end - 1, :], 1, :)
    naive_prediction = reshape(paths[end - 1, :], 1, :)
end

# ╔═╡ 6c360d83-dc89-610a-8ac6-4df666b52123
md"""
### 2. The forecaster

The Python ground truth builds a *tiny* Transformer encoder: a scalar-to-\$d\$ input projection (\$d_{\text{model}}=32\$), learned positional encoding, 2 encoder layers with 2 attention heads, and a causal mask so position \$t\$ attends only to positions \$\le t\$.

This preview replaces it with a compact Lux MLP (`make_mlp`, input dimension 3, two hidden layers, `tanh`) acting on the three context features. Where the Transformer would *discover* the regression rule from the raw sequence, this model is handed the sufficient statistics and learns how to weight them — the same econometric endpoint, reached explicitly.
"""

# ╔═╡ 776c96af-9e91-b47c-e917-2b60d9efad06
md"""
### 3. Training

We draw a batch of AR(1) prompts (each with its own \$\rho\$) and train the forecaster with `Optimisers.Adam` under MSE loss to predict \$x_{t+1}\$. Because a fresh \$\rho\$ is drawn per example, the model cannot memorise a single persistence; it must learn a rule that works across the whole prior.
"""

# ╔═╡ 55555555-0209-4555-8555-555555555555
begin
    model = make_mlp(3, (24, 24), 1; activation = NNlib.tanh)
    state = setup_training(rng_from_seed(SEED; offset = 1), model, Optimisers.Adam(0.008); parameter_type = Float64)
    loss_fn(model, ps, st, batch) = begin
        prediction, st_new = model(batch.x, ps, st)
        return mse_loss(prediction, batch.y), st_new
    end

    batch = (x = x, y = y)
    initial_loss = loss_value(state, loss_fn, batch)
    history = NamedTuple[]
    for _ in 1:hp.steps
        metrics = train_step!(state, loss_fn, batch; max_grad_norm = 10.0)
        append_metric!(history; step = metrics.step, loss = metrics.loss)
    end
    lux_prediction, _ = state.model(x, state.ps, state.st)
end

# ╔═╡ c70cb66c-8440-b50b-ba5e-02efaf45586a
md"""
### 4. In-context evaluation: the regression view

For fresh test series with unseen \$\rho\$, we compare three predictors of \$x_{t+1}\$ given \$x_1,\dots,x_t\$:

- **Oracle:** \$\rho\, x_t\$ (uses the true \$\rho\$; unattainable in practice).
- **OLS-on-the-prompt:** \$\hat{\rho}_{\text{OLS}}\, x_t\$, with \$\hat{\rho}_{\text{OLS}}\$ estimated from the prompt itself.
- **Lux context model:** the trained forecaster's prediction.

The next cell sweeps \$\rho\$ over a grid and records the forecast RMSE of the OLS-on-the-prompt predictor, the Lux model, and the naive last-value baseline. If the regression interpretation holds, the Lux model should track OLS closely across \$\rho\$.
"""

# ╔═╡ 66666666-0209-4666-8666-666666666666
begin
    rhos_test = collect(range(-0.85, 0.85; length = 17))
    ols_rmse = Float64[]
    lux_rmse = Float64[]
    naive_rmse = Float64[]
    for rho_value in rhos_test
        local paths_test = zeros(Float64, hp.T + 1, 40)
        paths_test[1, :] .= 0.2 .* randn(rng, 40)
        for t in 2:(hp.T + 1)
            paths_test[t, :] .= rho_value .* paths_test[t - 1, :] .+ 0.3 .* randn(rng, 40)
        end
        truth = reshape(paths_test[end, :], 1, :)
        features = context_features(paths_test)
        ols_pred = reshape(ols_rho(paths_test) .* paths_test[end - 1, :], 1, :)
        lux_pred, _ = state.model(features, state.ps, state.st)
        naive_pred = reshape(paths_test[end - 1, :], 1, :)
        push!(ols_rmse, sqrt(mse_loss(ols_pred, truth)))
        push!(lux_rmse, sqrt(mse_loss(lux_pred, truth)))
        push!(naive_rmse, sqrt(mse_loss(naive_pred, truth)))
    end
end

# ╔═╡ 3cdb5349-2ac8-5c13-29e7-351848ef23d5
md"""
### Visualising forecast RMSE across ρ

We plot forecast RMSE against the true \$\rho\$ for the last-value baseline, the OLS-on-the-prompt predictor, and the Lux context model.
"""

# ╔═╡ 77777777-0209-4777-8777-777777777777
begin
    fig = Figure(size = figure_size(RUN_MODE))
    ax = Axis(fig[1, 1], xlabel = "true rho", ylabel = "forecast RMSE")
    lines!(ax, rhos_test, naive_rmse; label = "last value", color = :gray45)
    lines!(ax, rhos_test, ols_rmse; label = "OLS prompt", color = :black)
    lines!(ax, rhos_test, lux_rmse; label = "Lux context model", color = :dodgerblue3)
    axislegend(ax; position = :ct)
    fig
end

# ╔═╡ f3e958ac-0500-1d3a-7022-7f67517f319c
md"""
### The full Python notebook also covers

**§5 — What does the Transformer effectively do?** For a single test sequence, the Python notebook backs out an *implicit* \$\hat{\rho}\$ from the Transformer by regressing its prediction on \$x_{t-1}\$, and shows this implicit slope tracks the OLS slope computed on the prompt — direct evidence that self-attention is performing in-context regression. That diagnostic needs the raw-sequence Transformer; this feature-based preview instead makes the regression rule explicit by construction.
"""

# ╔═╡ e04fdc88-a9a3-8907-71e5-166306eb34d9
md"""
### Take-away

Within training noise, the in-context predictor behaves like a **shrunken OLS estimate** — shrinkage toward 0 near the boundary, because the training prior on \$\rho\$ is uniform on \$[-0.9, 0.9]\$ and the optimizer regularizes. In the interior the predictors agree:

- **Oracle** achieves irreducible noise \$\sigma\$.
- **OLS on prompt** comes within a factor of \$\sim 1 + O(1/T)\$ of the oracle.
- The **learned model** matches OLS across \$\rho\$ values it never saw explicitly.

That is the econometric point: the model learned **how to regress**, not any specific regression, and applies that rule to a new series at inference. In the Python notebook this rule *emerges* inside a Transformer via self-attention; here it is made explicit through context features.

*Production-scale note.* LLMs use \$d_{\text{model}} \sim 10^3\$, \$L \sim 10^2\$ layers, and train on \$\sim 10^{12}\$ tokens. The mechanism is the same; only the scale differs.

The cell below returns a machine-checkable summary of the RMSE comparison.
"""

# ╔═╡ 88888888-0209-4888-8888-888888888888
(
    initial_loss = initial_loss,
    final_loss = history[end].loss,
    in_sample_rmse = sqrt(mse_loss(lux_prediction, y)),
    ols_prompt_rmse = sqrt(mse_loss(ols_prediction, y)),
    naive_rmse = sqrt(mse_loss(naive_prediction, y)),
    mean_lux_test_rmse = mean(lux_rmse),
    mean_ols_test_rmse = mean(ols_rmse),
)

# ╔═╡ Cell order:
# ╟─11111111-0209-4111-8111-111111111111
# ╟─a708968e-09ac-faa4-7a00-40beb9773530
# ╟─7e9b0687-b52e-f429-3539-0d7f86e7872b
# ╠═22222222-0209-4222-8222-222222222222
# ╠═33333333-0209-4333-8333-333333333333
# ╟─61cfb9ee-d34a-9512-42f2-0ba6bcf2cc1f
# ╠═44444444-0209-4444-8444-444444444444
# ╟─6c360d83-dc89-610a-8ac6-4df666b52123
# ╟─776c96af-9e91-b47c-e917-2b60d9efad06
# ╠═55555555-0209-4555-8555-555555555555
# ╟─c70cb66c-8440-b50b-ba5e-02efaf45586a
# ╠═66666666-0209-4666-8666-666666666666
# ╟─3cdb5349-2ac8-5c13-29e7-351848ef23d5
# ╠═77777777-0209-4777-8777-777777777777
# ╟─f3e958ac-0500-1d3a-7022-7f67517f319c
# ╟─e04fdc88-a9a3-8907-71e5-166306eb34d9
# ╠═88888888-0209-4888-8888-888888888888
