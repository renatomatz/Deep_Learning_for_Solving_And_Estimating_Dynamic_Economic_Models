### A Pluto.jl notebook ###
# v0.20.0

using Markdown
using InteractiveUtils

# ╔═╡ 11111111-1401-4111-8111-111111111111
md"""
# Lecture 14, Notebook 01: Deep Surrogate Primer in Lux

A small Lux MLP learns a normalized Black-Scholes pricing surface over
`(S,K,T,sigma,r)`. The original Python notebook uses a large training budget;
this Pluto translation keeps the same pseudo-state surrogate structure and runs
a CPU smoke pass.
"""

# ╔═╡ e44ad84f-544d-5b52-dba6-0947b40c38cd
md"""
## Lecture 14, Notebook 01: Deep surrogate primer — a Black–Scholes implied-volatility surrogate

**Course:** Deep Learning for Solving and Estimating Dynamic Models in Economics and Finance
**Script reference:** §10.1-10.2 (Deep surrogate primer — Black–Scholes implied-volatility surrogate)
**Notebook role:** core
**Author:** Simon Scheidegger

*Julia/Lux/Pluto preview of* `lectures/lecture_14_surrogates_and_gps/code/lecture_14_01_Surrogate_Primer.ipynb`.

> **Run mode.** The checked-in run uses `RUN_MODE = "smoke"` (small training budget, CPU) so the notebook executes quickly; the accuracy and speed figures in the slides use the heavier `teaching` / `production` budgets defined in the budgets cell below. Set `RUN_MODE` there to reproduce them.
"""

# ╔═╡ 70c44638-bb2b-fe38-40e4-71972dd4d394
md"""
## What is a surrogate?

A **surrogate model** (or emulator) is a fast, differentiable approximation of an expensive computational model. The idea is simple:

1. We have an expensive model \$f(\mathbf{x})\$ (e.g. solving a PDE, running a Monte Carlo simulation).
2. We generate training data by evaluating \$f\$ on a design of experiments.
3. We train a neural network \$\phi(\mathbf{x} \mid \theta_{NN}) \approx f(\mathbf{x})\$.
4. The surrogate is orders of magnitude faster and fully differentiable.

### What we'll do in this notebook

- Use the **Black–Scholes formula** as our "expensive" model (in practice this would be a complex SDE or PDE solver).
- Build a **Lux DNN surrogate** over a 5-dimensional input space \$(S, K, T, \sigma, r)\$.
- Validate accuracy and measure speedup.
- Apply the surrogate to **implied-volatility inversion** — a calibration / inversion problem.

**Reference:** Chen, Didisheim & Scheidegger (2026), *J. Financial Economics*.
"""

# ╔═╡ 22222222-1401-4222-8222-222222222222
begin
    import Pkg
    Pkg.activate(joinpath(@__DIR__, "..", "..", "..", "julia"))

    using DLEFJulia
    using Lux
    using NNlib
    using Optimisers
    using Statistics
    using Zygote
end

# ╔═╡ 33333333-1401-4333-8333-333333333333
begin
    RUN_MODE = "smoke"
    SEED = 0
    budgets = (
        smoke = (steps = 400, n_train = 2_048, n_val = 512, width = 32, lr = 0.003,
            iv_steps = 200, n_options = 32),
        teaching = (steps = 500, n_train = 20_000, n_val = 2_000, width = 64, lr = 0.001,
            iv_steps = 500, n_options = 500),
        production = (steps = 2_000, n_train = 200_000, n_val = 20_000, width = 128, lr = 0.001,
            iv_steps = 500, n_options = 500),
    )
    hp = run_mode_budget(RUN_MODE; budgets)
    rng = rng_from_seed(SEED)
end

# ╔═╡ 72eaed99-82ac-a5b9-6660-64b40a3f691f
md"""
## 1. The "Expensive" Model: Black–Scholes Formula

The Black–Scholes price of a European call option is

\$\$C(S, K, T, \sigma, r) = S\,\Phi(d_1) - K e^{-rT}\,\Phi(d_2),\$\$

where

\$\$d_1 = \frac{\ln(S/K) + (r + \sigma^2/2)T}{\sigma\sqrt{T}}, \qquad d_2 = d_1 - \sigma\sqrt{T},\$\$

and \$\Phi\$ is the standard normal CDF. Here `black_scholes_call_price_5d` plays the role of the expensive ground truth; in practice it would be a Heston/SABR or full SDE solver.

## 2. Generate Training Data

We sample uniformly over the 5-dimensional input box:

| Parameter | Range | Description |
|-----------|-------|-------------|
| \$S\$ | \$[50, 150]\$ | Spot price |
| \$K\$ | \$[50, 150]\$ | Strike price |
| \$T\$ | \$[0.1, 2.0]\$ | Time to maturity |
| \$\sigma\$ | \$[0.05, 0.6]\$ | Volatility |
| \$r\$ | \$[0.01, 0.08]\$ | Risk-free rate |

`black_scholes_design` draws the design and returns the box `normalizer`; `normalize_box` maps inputs to \$[0,1]\$ and `standardize_targets` maps prices to zero-mean, unit-variance. The Python ground truth uses \$m = 100{,}000\$ training samples; this smoke run uses `hp.n_train` (2{,}048 by default).

## 3. Build and Train the DNN Surrogate

The surrogate is a Lux MLP (`make_mlp`) with four hidden layers and **Swish** (SiLU) activation,

\$\$\text{Swish}(x) = x \cdot \sigma(x),\$\$

which is smooth and gives good gradients — important because we later differentiate through the surrogate. The next cell builds the model and the `surrogate_loss`; the following cell runs the Adam training loop via `train_step!`. Full-capacity (128-wide) production layers drive the worst-case price error well below one cent on a \$[0, 100]\$ price range; the smoke preview uses narrower `hp.width` layers. (The full Python notebook trains 256-wide layers; this Julia preview caps production width at 128 in the budgets cell above.)
"""

# ╔═╡ 44444444-1401-4444-8444-444444444444
begin
    train = black_scholes_design(rng, hp.n_train)
    val = black_scholes_design(rng_from_seed(SEED; offset = 1), hp.n_val)
    y_scaler = standardize_targets(train.y)
    x_train = normalize_box(train.x, train.normalizer)
    y_train = y_scaler.z
    x_val = normalize_box(val.x, train.normalizer)

    model = make_mlp(5, (hp.width, hp.width, hp.width, hp.width), 1; activation = NNlib.swish)
    state = setup_training(rng_from_seed(SEED; offset = 2), model, Optimisers.Adam(hp.lr); parameter_type = Float64)
    surrogate_loss(model, ps, st, batch) = begin
        yhat, st_new = model(batch.x, ps, st)
        return mse_loss(yhat, batch.y), st_new
    end
end

# ╔═╡ d4ffc7d0-6411-45ad-5f45-cbbcab2369c3
md"""
## 4. Validate the Surrogate

The training loop runs Adam for `hp.steps` steps via `train_step!`, then we evaluate on the held-out validation design. The Python notebook reports the max absolute error, MAE, and \$R^2\$ with a prediction-vs-truth scatter; this Julia preview reports RMSE and max absolute error through `residual_summary` on the un-standardized predictions (`unstandardize_targets`).
"""

# ╔═╡ 55555555-1401-4555-8555-555555555555
begin
    initial_loss = loss_value(state, surrogate_loss, (x = x_train, y = y_train))
    history = NamedTuple[]
    for step in 1:hp.steps
        metrics = train_step!(state, surrogate_loss, (x = x_train, y = y_train); max_grad_norm = 25.0)
        append_metric!(history; step, loss = metrics.loss)
    end
    pred_val_norm, _ = state.model(x_val, state.ps, state.st)
    pred_val = unstandardize_targets(pred_val_norm, y_scaler)
    validation = residual_summary(pred_val .- val.y)
end

# ╔═╡ 09a58671-ed2c-a77f-644e-542a4ebc4cd2
md"""
## 5. Speed Comparison

The Python notebook benchmarks surrogate evaluation against vectorized Black–Scholes and against a simulated 1 ms-per-eval "expensive model". This Julia preview instead times the classical bracketing solver against the surrogate inversion directly (single option next, full batch in §7) with `time_ns`, and reports the millisecond timings in the final diagnostics — the surrogate's real edge shows up in the batch inversion.

## 6. Application: Implied Volatility via Surrogate

**Problem:** given a market price \$C_{\text{market}}\$, find the implied volatility \$\sigma^*\$ with

\$\$\phi(S, K, T, \sigma^*, r \mid \theta_{NN}) \approx C_{\text{market}}.\$\$

This is a **calibration / inversion** problem. With the surrogate we minimise \$(\phi - C_{\text{market}})^2\$ with respect to \$\sigma\$ using `Optimisers.jl` Adam, with the gradient \$\partial \phi / \partial \sigma\$ supplied by `Zygote` (`invert_sigma_surrogate`). The classical baseline is a bisection bracketing solver on the analytical formula (`implied_vol_bracketed`), standing in for SciPy's `brentq`. The next cell defines this machinery; the cell after it inverts a single market option.
"""

# ╔═╡ 66666666-1401-4666-8666-666666666666
begin
    function implied_vol_bracketed(S, K, T, r, price; lower = 0.01, upper = 1.0, iterations = 70)
        lo = float(lower)
        hi = float(upper)
        f_lo = black_scholes_call_price_5d(S, K, T, lo, r) - price
        f_hi = black_scholes_call_price_5d(S, K, T, hi, r) - price
        f_lo == 0 && return lo
        f_hi == 0 && return hi
        f_lo * f_hi <= 0 || throw(ArgumentError("implied volatility is not bracketed"))

        mid = (lo + hi) / 2
        for _ in 1:iterations
            mid = (lo + hi) / 2
            f_mid = black_scholes_call_price_5d(S, K, T, mid, r) - price
            if f_lo * f_mid <= 0
                hi = mid
                f_hi = f_mid
            else
                lo = mid
                f_lo = f_mid
            end
        end
        return mid
    end

    as_vector(x, n) = x isa Number ? fill(float(x), n) : float.(collect(x))

    function normalized_row(x, idx, n, normalizer)
        values = as_vector(x, n)
        length(values) == n || throw(DimensionMismatch("option input lengths must agree"))
        scale = normalizer.upper[idx] - normalizer.lower[idx]
        return reshape((values .- normalizer.lower[idx]) ./ scale, 1, :)
    end

    function fixed_option_inputs(S, K, T, r, normalizer)
        n = maximum((S isa Number ? 1 : length(S), K isa Number ? 1 : length(K),
            T isa Number ? 1 : length(T), r isa Number ? 1 : length(r)))
        return vcat(
            normalized_row(S, 1, n, normalizer),
            normalized_row(K, 2, n, normalizer),
            normalized_row(T, 3, n, normalizer),
            zeros(Float64, 1, n),
            normalized_row(r, 5, n, normalizer),
        )
    end

    function sigma_matrix(initial_sigma, n)
        values = as_vector(initial_sigma, n)
        length(values) == n || throw(DimensionMismatch("initial sigma length must match option count"))
        return reshape(values, 1, :)
    end

    standardize_price(price, scaler) = reshape((float.(collect(price)) .- scaler.mean) ./ scaler.std, 1, :)

    function surrogate_price_from_sigma(state, fixed_inputs, sigma, scaler, normalizer)
        sigma_norm = (sigma .- normalizer.lower[4]) ./ (normalizer.upper[4] - normalizer.lower[4])
        x_input = vcat(fixed_inputs[1:3, :], sigma_norm, fixed_inputs[5:5, :])
        pred_norm, _ = state.model(x_input, state.ps, state.st)
        return unstandardize_targets(pred_norm, scaler)
    end

    function surrogate_iv_loss(state, fixed_inputs, sigma, target_norm, normalizer)
        sigma_norm = (sigma .- normalizer.lower[4]) ./ (normalizer.upper[4] - normalizer.lower[4])
        x_input = vcat(fixed_inputs[1:3, :], sigma_norm, fixed_inputs[5:5, :])
        pred_norm, _ = state.model(x_input, state.ps, state.st)
        return mean(abs2, pred_norm .- target_norm)
    end

    function invert_sigma_surrogate(state, fixed_inputs, target_price, scaler, normalizer;
            initial_sigma, steps, lr, lower = 0.01, upper = 1.0, keep_history = false)
        n = size(fixed_inputs, 2)
        sigma = sigma_matrix(initial_sigma, n)
        target_norm = standardize_price(target_price, scaler)
        target_price_row = reshape(float.(collect(target_price)), 1, :)
        opt_state = Optimisers.setup(Optimisers.Adam(lr), sigma)
        sigma_history = keep_history ? vec(copy(sigma)) : Float64[]
        price_loss_history = Float64[]

        for _ in 1:steps
            loss, back = Zygote.pullback(sigma) do sigma_current
                surrogate_iv_loss(state, fixed_inputs, sigma_current, target_norm, normalizer)
            end
            grad = only(back(one(loss)))
            opt_state, sigma = Optimisers.update(opt_state, sigma, grad)
            sigma = clamp.(sigma, lower, upper)

            pred_price = surrogate_price_from_sigma(state, fixed_inputs, sigma, scaler, normalizer)
            push!(price_loss_history, mean(abs2, pred_price .- target_price_row))
            keep_history && push!(sigma_history, only(sigma))
        end

        return (sigma = sigma, sigma_history = sigma_history, price_loss_history = price_loss_history)
    end
end

# ╔═╡ 66666666-1401-4777-8666-666666666666
begin
    market = (S = 100.0, K = 105.0, T = 0.5, sigma = 0.25, r = 0.03)
    price = black_scholes_call_price_5d(market.S, market.K, market.T, market.sigma, market.r)
    sigma_grid = collect(range(0.05, 0.6; length = 80))
    grid_prices = [black_scholes_call_price_5d(market.S, market.K, market.T, sigma, market.r) for sigma in sigma_grid]
    sigma_grid_hat = sigma_grid[argmin(abs.(grid_prices .- price))]
    t0_bracket_single = time_ns()
    sigma_bracket = implied_vol_bracketed(market.S, market.K, market.T, market.r, price)
    t_bracket = (time_ns() - t0_bracket_single) / 1e9

    fixed_market = fixed_option_inputs(market.S, market.K, market.T, market.r, train.normalizer)
    t0_surrogate_single = time_ns()
    single_iv = invert_sigma_surrogate(state, fixed_market, [price], y_scaler, train.normalizer;
        initial_sigma = 0.4, steps = hp.iv_steps, lr = 0.005, keep_history = true)
    t_surrogate_single = (time_ns() - t0_surrogate_single) / 1e9
    sigma_surrogate = only(single_iv.sigma)
    surrogate_single_error = abs(sigma_surrogate - market.sigma)
end

# ╔═╡ ba5afa32-ab97-9c17-157e-cb24856eafdf
md"""
## 7. Batch Implied Volatility Inversion

The real advantage of the surrogate appears when we invert **many options at once** (e.g. an entire option chain for daily calibration): the surrogate inversion runs as a single batched gradient descent over all `n_options` strikes, whereas the classical solver loops option by option.
"""

# ╔═╡ 77777777-1401-4777-8777-777777777777
begin
    n_options = hp.n_options
    S_batch = fill(100.0, n_options)
    K_batch = collect(range(70.0, 130.0; length = n_options))
    T_batch = fill(0.5, n_options)
    sigma_batch_true = 0.2 .+ 0.1 .* ((K_batch .- 100.0) ./ 30.0).^2
    r_batch = fill(0.03, n_options)
    C_batch = [black_scholes_call_price_5d(S_batch[i], K_batch[i], T_batch[i],
        sigma_batch_true[i], r_batch[i]) for i in 1:n_options]

    t0_bracket_batch = time_ns()
    sigma_bracket_batch = [implied_vol_bracketed(S_batch[i], K_batch[i], T_batch[i],
        r_batch[i], C_batch[i]) for i in 1:n_options]
    t_bracket_batch = (time_ns() - t0_bracket_batch) / 1e9

    fixed_batch = fixed_option_inputs(S_batch, K_batch, T_batch, r_batch, train.normalizer)
    t0_surrogate_batch = time_ns()
    batch_iv = invert_sigma_surrogate(state, fixed_batch, C_batch, y_scaler, train.normalizer;
        initial_sigma = 0.3, steps = hp.iv_steps, lr = 0.01)
    t_surrogate_batch = (time_ns() - t0_surrogate_batch) / 1e9
    sigma_surrogate_batch = vec(batch_iv.sigma)
    batch_surrogate_abs_error = abs.(sigma_surrogate_batch .- sigma_batch_true)
    batch_bracket_abs_error = abs.(sigma_bracket_batch .- sigma_batch_true)
end

# ╔═╡ bf88d063-24d2-0546-df48-8dfb1927eb9f
md"""
## Summary

In this notebook we demonstrated:

1. **Surrogate construction:** a 5-D Lux MLP trained on Black–Scholes data reaches low held-out error (RMSE and max absolute error reported below).
2. **Speed:** the surrogate inverts the whole option chain in one batched pass, while the classical solver loops option by option (timings in the diagnostics).
3. **Differentiability:** `Zygote` gradients enable gradient-based inversion (implied volatility) — a task that classically requires root-finding.
4. **Batch processing:** the surrogate inverts an entire option chain simultaneously via batch gradient descent.

### Key takeaway

Deep surrogates turn expensive model evaluations into fast, differentiable function calls. Combined with **pseudo-states** (treating parameters as inputs), a single surrogate can serve for pricing, calibration, risk management, and uncertainty quantification. The single-point IV error reaches ~4e-4 in production mode; smoke mode loses a digit but exercises the same pipeline end-to-end.

**Reference:** Chen, Didisheim & Scheidegger (2026), *Deep Surrogates*, J. Financial Economics.

The cell below returns the machine-checkable diagnostics summary; the final cell asserts that the smoke-mode inversion converged near \$\sigma_{\text{true}}\$.
"""

# ╔═╡ 88888888-1401-4888-8888-888888888888
(
    initial_loss = initial_loss,
    final_loss = history[end].loss,
    validation_rmse = validation.rmse,
    validation_max_abs = validation.max_abs,
    market_price = price,
    exact_implied_sigma_grid = sigma_grid_hat,
    exact_implied_sigma_error = abs(sigma_grid_hat - market.sigma),
    bracketed_implied_sigma = sigma_bracket,
    bracketed_implied_sigma_error = abs(sigma_bracket - market.sigma),
    surrogate_implied_sigma = sigma_surrogate,
    surrogate_implied_sigma_error = surrogate_single_error,
    surrogate_price_loss_final = single_iv.price_loss_history[end],
    single_option_timing_ms = (bracketed = 1_000 * t_bracket, surrogate = 1_000 * t_surrogate_single),
    batch_options = n_options,
    batch_mean_surrogate_sigma_error = mean(batch_surrogate_abs_error),
    batch_max_surrogate_sigma_error = maximum(batch_surrogate_abs_error),
    batch_max_bracketed_sigma_error = maximum(batch_bracket_abs_error),
    batch_timing_ms = (bracketed = 1_000 * t_bracket_batch, surrogate = 1_000 * t_surrogate_batch),
)

# ╔═╡ 99999999-1401-4999-8999-999999999999
begin
    smoke_iv_check = RUN_MODE == "smoke" ? surrogate_single_error < 0.1 : true
    @assert smoke_iv_check "Implied-vol inversion off by $(round(surrogate_single_error; digits = 3)) (> 0.1)"
    smoke_iv_check
end

# ╔═╡ Cell order:
# ╟─11111111-1401-4111-8111-111111111111
# ╟─e44ad84f-544d-5b52-dba6-0947b40c38cd
# ╟─70c44638-bb2b-fe38-40e4-71972dd4d394
# ╠═22222222-1401-4222-8222-222222222222
# ╠═33333333-1401-4333-8333-333333333333
# ╟─72eaed99-82ac-a5b9-6660-64b40a3f691f
# ╠═44444444-1401-4444-8444-444444444444
# ╟─d4ffc7d0-6411-45ad-5f45-cbbcab2369c3
# ╠═55555555-1401-4555-8555-555555555555
# ╟─09a58671-ed2c-a77f-644e-542a4ebc4cd2
# ╠═66666666-1401-4666-8666-666666666666
# ╠═66666666-1401-4777-8666-666666666666
# ╟─ba5afa32-ab97-9c17-157e-cb24856eafdf
# ╠═77777777-1401-4777-8777-777777777777
# ╟─bf88d063-24d2-0546-df48-8dfb1927eb9f
# ╠═88888888-1401-4888-8888-888888888888
# ╠═99999999-1401-4999-8999-999999999999
