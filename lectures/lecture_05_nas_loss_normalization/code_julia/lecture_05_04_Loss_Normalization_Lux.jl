### A Pluto.jl notebook ###
# v0.20.0

using Markdown
using InteractiveUtils

# ╔═╡ 11111111-0504-4111-8111-111111111111
md"""
# Lecture 05, Notebook 04: Multi-Component Loss Balancing in Julia

This Lux translation mirrors the Python loss-normalization notebook at smoke
scale. A single shared neural trunk learns three Genz targets whose natural
scales differ by orders of magnitude, and the training loop compares static
equal weights, inverse-loss weights, SoftAdapt, and the deterministic classroom
ReLoBRaLo update.
"""

# ╔═╡ 2414ce06-b73e-4e36-1bcc-9ea487da8364
md"""
## Lecture 05, Notebook 04: Multi-Component Loss Balancing (ReLoBRaLo)

**Course:** Deep Learning for Solving and Estimating Dynamic Models in Economics and Finance
**Script reference:** §4.8 (multi-component losses: the scale problem; inverse-loss weighting; ReLoBRaLo Components 1–3; sensitivity to the temperature \$T\$)
**Notebook role:** core
**Author:** Simon Scheidegger

*Julia/Lux/Pluto preview of* `lectures/lecture_05_nas_loss_normalization/code/lecture_05_04_Loss_Normalization.ipynb`.

> **Run mode.** `RUN_MODE = "smoke"` trains for a few epochs on a small sample; `teaching` and `production` use the full schedules and the wider temperature sweep. This preview writes none of the `loss_norm_*` figure pairs the Python notebook saves.

> **Beyond the Python three.** The Python ground truth compares equal, inverse-loss, and ReLoBRaLo weighting. This Lux preview adds **SoftAdapt** (Heydari et al., 2019) as a fourth scheme via the shared `softadapt_weights` helper.
"""

# ╔═╡ 9903b80d-5ff9-9be6-91a1-27c1feb60512
md"""
This notebook accompanies **Lecture 05 — Multi-Component Loss Balancing**.

**Goals:**

1. Understand why multi-component losses at different scales cause problems.
2. Train a multi-output NN with equal weights and observe failure.
3. Implement inverse-loss weighting and see partial improvement.
4. Implement a deterministic classroom version of ReLoBRaLo (Bischof & Kraus, 2025) and achieve balanced learning.
5. Study sensitivity to the temperature parameter \$T\$.

**Outline**

| Section | Topic |
|:--------|:------|
| 1 | Target functions at different scales |
| 2 | Multi-output network |
| 3 | Baseline: equal weighting |
| 4 | Inverse-loss weighting |
| 5 | ReLoBRaLo (and SoftAdapt in this preview) |
| 6 | Comparison |
| 7 | Sensitivity to \$T\$ |
| 8 | Discussion |
"""

# ╔═╡ 22222222-0504-4222-8222-222222222222
begin
    import Pkg
    Pkg.activate(joinpath(@__DIR__, "..", "..", "..", "julia"))

    using CairoMakie
    using DLEFJulia
    using Lux
    using NNlib
    using Optimisers
    using Random: randperm
    using Statistics
end

# ╔═╡ 33333333-0504-4333-8333-333333333333
begin
    RUN_MODE = "smoke"
    SEED = 0
    budgets = (
        smoke = (
            epochs = 12,
            n_train = 128,
            n_test = 192,
            batch_size = 64,
            hidden_dims = (32, 32),
            grid_n = 21,
            lr = 1e-3,
            temperature_sweep = (0.1, 1.0),
        ),
        teaching = (
            epochs = 200,
            n_train = 1_000,
            n_test = 2_000,
            batch_size = 64,
            hidden_dims = (64, 64, 64),
            grid_n = 60,
            lr = 1e-3,
            temperature_sweep = (0.01, 0.1, 1.0, 10.0),
        ),
        production = (
            epochs = 1_500,
            n_train = 1_000,
            n_test = 2_000,
            batch_size = 64,
            hidden_dims = (64, 64, 64),
            grid_n = 60,
            lr = 1e-3,
            temperature_sweep = (0.01, 0.1, 1.0, 10.0),
        ),
    )
    hp = run_mode_budget(RUN_MODE; budgets)
end

# ╔═╡ 749f3cd9-caa7-2f00-5f43-221550e79872
md"""
### 1. The multi-component loss problem

We approximate **three functions simultaneously** with a shared-trunk network. The functions live at vastly different scales:

| Function | Formula | Scale |
|:---------|:--------|:------|
| \$f_A(\mathbf{x})\$ | \$\text{genz\_gaussian}(\mathbf{x})\$ | \$[0, 1]\$ |
| \$f_B(\mathbf{x})\$ | \$100 \times \text{genz\_oscillatory}(\mathbf{x})\$ | \$[-100, 100]\$ |
| \$f_C(\mathbf{x})\$ | \$10{,}000 \times \text{genz\_continuous}(\mathbf{x})\$ | \$[0, 10{,}000]\$ |

The composite loss is

\$\$\mathcal{L} = w_A \cdot \text{MSE}_A + w_B \cdot \text{MSE}_B + w_C \cdot \text{MSE}_C.\$\$

With equal weights (\$w_A = w_B = w_C = 1\$), \$\text{MSE}_C \approx 2 \times 10^7\$ dominates while \$\text{MSE}_A \approx 0.4\$ is essentially ignored. The Genz shape constants `c_param` / `w_param` are the exact `NumPy RandomState(7)` values from the ground-truth notebook, so the targets match.
"""

# ╔═╡ 44444444-0504-4444-8444-444444444444
begin
    const D = 2
    const TARGET_SCALES = (A = 1.0, B = 100.0, C = 10_000.0)

    # Python RandomState(7) constants from the ground-truth notebook.
    const c_param = [1.1526165787479143, 2.5598375844802292]
    const w_param = [0.4630455388645361, 0.6340791066985647]
    const c_col = reshape(c_param, :, 1)
    const w_col = reshape(w_param, :, 1)

    function genz_gaussian(x)
        return exp.(-sum((c_col .^ 2) .* (x .- w_col) .^ 2; dims = 1))
    end

    function genz_oscillatory(x)
        return cos.(2.0 * pi * w_param[1] .+ sum(c_col .* x; dims = 1))
    end

    function genz_continuous(x)
        return exp.(-sum(c_col .* abs.(x .- w_col); dims = 1))
    end

    function multi_scale_targets(x)
        return vcat(
            TARGET_SCALES.A .* genz_gaussian(x),
            TARGET_SCALES.B .* genz_oscillatory(x),
            TARGET_SCALES.C .* genz_continuous(x),
        )
    end
end

# ╔═╡ 2633f98e-0b88-c139-fb52-68aa97209a3b
md"""
### Generating the data

We draw train and test inputs uniformly on \$[0,1]^2\$ and evaluate the three targets, storing everything feature-by-batch. `train_mse_scales` records the mean squared magnitude of each component on the training set — this is the imbalance the weighting schemes have to correct. A dense grid supplies the target-range diagnostics.
"""

# ╔═╡ 55555555-0504-4555-8555-555555555555
begin
    data_rng = rng_from_seed(SEED; offset = 1)
    x_train = rand(data_rng, D, hp.n_train)
    x_test = rand(data_rng, D, hp.n_test)
    y_train = multi_scale_targets(x_train)
    y_test = multi_scale_targets(x_test)

    train_mse_scales = (
        A = mean(abs2, @view y_train[1, :]),
        B = mean(abs2, @view y_train[2, :]),
        C = mean(abs2, @view y_train[3, :]),
    )

    grid_axis = range(0.0, 1.0; length = hp.grid_n)
    x_grid = reduce(hcat, ([x1, x2] for x2 in grid_axis for x1 in grid_axis))
    y_grid = multi_scale_targets(x_grid)
    target_ranges = (
        A = (minimum(y_grid[1, :]), maximum(y_grid[1, :])),
        B = (minimum(y_grid[2, :]), maximum(y_grid[2, :])),
        C = (minimum(y_grid[3, :]), maximum(y_grid[3, :])),
    )
end

# ╔═╡ 031a1795-172c-6702-dac2-422259c53cad
md"""
### 2. Multi-output network

A single shared-trunk MLP with three linear outputs (`make_mlp(D, hidden_dims, 3)`) approximates all three targets at once — the Lux counterpart of the Python multi-output Keras model. The helpers here compute the **per-component MSE** (`component_mse`), evaluate losses on a batch, and summarise per-component mean/max test error plus a scale-normalised `total_relative_error`.
"""

# ╔═╡ 66666666-0504-4666-8666-666666666666
begin
    function build_multi_output_model(hidden_dims = hp.hidden_dims)
        return make_mlp(D, hidden_dims, 3; activation = NNlib.relu)
    end

    function component_mse(prediction, target)
        return [
            mse_loss(prediction[1:1, :], target[1:1, :]),
            mse_loss(prediction[2:2, :], target[2:2, :]),
            mse_loss(prediction[3:3, :], target[3:3, :]),
        ]
    end

    function evaluate_losses(state, batch)
        prediction, _ = state.model(batch.x, state.ps, state.st)
        return component_mse(prediction, batch.y)
    end

    function test_error_summary(state)
        prediction, _ = state.model(x_test, state.ps, state.st)
        err = abs.(prediction .- y_test)
        return (
            mean_A = mean(@view err[1, :]),
            mean_B = mean(@view err[2, :]),
            mean_C = mean(@view err[3, :]),
            max_A = maximum(@view err[1, :]),
            max_B = maximum(@view err[2, :]),
            max_C = maximum(@view err[3, :]),
        )
    end

    total_relative_error(err) =
        err.mean_A / TARGET_SCALES.A + err.mean_B / TARGET_SCALES.B + err.mean_C / TARGET_SCALES.C
end

# ╔═╡ 1eb7e994-6f83-53c6-a8a5-f2842dc380ee
md"""
### 4 & 5. Weighting schemes

Every scheme returns a weight vector \$w = (w_A, w_B, w_C)\$, count-normalised so \$\sum_i w_i = K = 3\$. The training loop below calls one `schedule(losses, prev_losses, init_losses, prev_weights, epoch)` per epoch.

**Inverse-loss weighting** (§4) sets each weight inversely proportional to a smoothed component loss:

\$\$w_i^{(t)} = \frac{1}{\bar{\mathcal{L}}_i^{(t)} + \epsilon}, \qquad \bar{\mathcal{L}}_i^{(t)} = \beta\,\bar{\mathcal{L}}_i^{(t-1)} + (1-\beta)\,\mathcal{L}_i^{(t)},\$\$

built from the `inverse_loss_weights` helper applied to the EMA-smoothed losses. Simple, but unstable as a loss approaches zero.

**SoftAdapt** (Heydari et al., 2019 — added in this Lux preview) weights each component by a softmax over its recent relative loss *slope*:

\$\$s_i^{(t)} = \frac{\mathcal{L}_i^{(t)} - \mathcal{L}_i^{(t-1)}}{\lvert\mathcal{L}_i^{(t-1)}\rvert}, \qquad w_i^{(t)} = K\,\operatorname{softmax}_i\!\left(\frac{s_i^{(t)}}{T}\right),\$\$

so a component whose loss is falling slowly gets more weight (`softadapt_weights`).

**ReLoBRaLo** — Relative Loss Balancing with Random Lookback (deterministic classroom version), Bischof & Kraus (2025). At each epoch \$t\$:

1. **Step-wise weights:** \$\hat{w}_{i,\mathrm{step}}^{(t)} = K\,\operatorname{softmax}_i\!\left(\frac{\mathcal{L}_i^{(t)}}{T \cdot \mathcal{L}_i^{(t-1)}}\right)\$
2. **Baseline weights:** \$\hat{w}_{i,\mathrm{base}}^{(t)} = K\,\operatorname{softmax}_i\!\left(\frac{\mathcal{L}_i^{(t)}}{T \cdot \mathcal{L}_i^{(0)}}\right)\$
3. **Combine:** \$w_i^{(t)} = \alpha\!\left[\rho\, w_i^{(t-1)} + (1-\rho)\, \hat{w}_{i,\mathrm{base}}^{(t)}\right] + (1-\alpha)\, \hat{w}_{i,\mathrm{step}}^{(t)}\$

with \$T = 1.0\$ (temperature), \$\alpha = 0.999\$ (smoothing), and \$\rho = 0.999\$ (baseline mix). The count-softmax of loss ratios matches the `relobralo_weights` helper.
"""

# ╔═╡ 77777777-0504-4777-8777-777777777777
begin
    function make_inverse_loss_fn(; beta = 0.99, eps = 1e-8)
        ema = Ref{Union{Nothing,Vector{Float64}}}(nothing)
        return function inverse_loss_schedule(losses, prev_losses, init_losses, prev_weights, epoch)
            current = Float64.(losses)
            ema[] = ema[] === nothing ? current : beta .* ema[] .+ (1 - beta) .* current
            return inverse_loss_weights(ema[]; eps)
        end
    end

    function make_softadapt_fn(; temperature = 1.0, eps = 1e-8)
        return function softadapt_schedule(losses, prev_losses, init_losses, prev_weights, epoch)
            return softadapt_weights(losses, prev_losses; temperature, normalize = :count, eps)
        end
    end

    function _count_softmax(logits)
        raw = exp.(logits .- maximum(logits))
        return length(raw) .* raw ./ sum(raw)
    end

    function make_relobralo_fn(; temperature = 1.0, alpha = 0.999, rho = 0.999, eps = 1e-12)
        return function relobralo_schedule(losses, prev_losses, init_losses, prev_weights, epoch)
            current = Float64.(losses)
            previous = Float64.(prev_losses)
            initial = Float64.(init_losses)
            K = length(current)

            step_hat = _count_softmax(current ./ (temperature .* previous .+ eps))
            base_hat = _count_softmax(current ./ (temperature .* initial .+ eps))

            return [
                rho * alpha * prev_weights[i] +
                (1 - rho) * alpha * base_hat[i] +
                (1 - alpha) * step_hat[i]
                for i in 1:K
            ]
        end
    end

    function weight_schedule(method; temperature = 1.0)
        method == :equal && return (losses, prev_losses, init_losses, prev_weights, epoch) -> equal_loss_weights(losses)
        method == :inverse_loss && return make_inverse_loss_fn()
        method == :softadapt && return make_softadapt_fn(; temperature)
        method == :relobralo && return make_relobralo_fn(; temperature)
        error("unknown weighting method: $method")
    end
end

# ╔═╡ 3a0b6f9f-588b-c791-4c2d-516a8e7e9e2b
md"""
### 3. Baseline (equal weighting) and the shared training loop

Equal weighting is the baseline: \$\mathcal{L} = \text{MSE}_A + \text{MSE}_B + \text{MSE}_C\$ (all \$w_i = 1\$). Rather than a separate pass per scheme, `train_multi_output(method; temperature)` runs one custom mini-batch loop that logs per-component losses, the current weights, and per-component test error each epoch, then applies the chosen `schedule`. We run it for all four methods (`:equal`, `:inverse_loss`, `:softadapt`, `:relobralo`) so their histories can be compared directly.
"""

# ╔═╡ 88888888-0504-4888-8888-888888888888
begin
    function train_multi_output(method; temperature = 1.0)
        model = build_multi_output_model()
        state = setup_training(rng_from_seed(SEED; offset = 100), model, Optimisers.Adam(hp.lr); parameter_type = Float64)
        shuffle_rng = rng_from_seed(SEED; offset = 200)
        schedule = weight_schedule(method; temperature)

        weights = [1.0, 1.0, 1.0]
        init_losses = nothing
        prev_losses = nothing
        history = NamedTuple[]
        n_batches = max(1, hp.n_train ÷ hp.batch_size)

        for epoch in 1:hp.epochs
            perm = randperm(shuffle_rng, hp.n_train)
            epoch_losses = zeros(3)

            for batch_id in 1:n_batches
                first = (batch_id - 1) * hp.batch_size + 1
                last = batch_id * hp.batch_size
                idx = @view perm[first:last]
                batch = (x = x_train[:, idx], y = y_train[:, idx])
                epoch_losses .+= Float64.(evaluate_losses(state, batch)) ./ n_batches

                loss_fn(model, ps, st, local_batch) = begin
                    prediction, st_new = model(local_batch.x, ps, st)
                    losses = component_mse(prediction, local_batch.y)
                    return sum(weights .* losses), st_new
                end
                train_step!(state, loss_fn, batch)
            end

            init_losses === nothing && (init_losses = copy(epoch_losses))
            prev_losses === nothing && (prev_losses = copy(epoch_losses))
            weights = schedule(epoch_losses, prev_losses, init_losses, weights, epoch)
            prev_losses = copy(epoch_losses)

            err = test_error_summary(state)
            append_metric!(
                history;
                epoch = epoch,
                loss_A = epoch_losses[1],
                loss_B = epoch_losses[2],
                loss_C = epoch_losses[3],
                weighted_loss = sum(weights .* epoch_losses),
                w_A = weights[1],
                w_B = weights[2],
                w_C = weights[3],
                test_A = err.mean_A,
                test_B = err.mean_B,
                test_C = err.mean_C,
            )
        end

        return (state = state, history = history, final_errors = test_error_summary(state))
    end

    methods = (:equal, :inverse_loss, :softadapt, :relobralo)
    method_labels = Dict(
        :equal => "Equal",
        :inverse_loss => "Inverse-loss",
        :softadapt => "SoftAdapt",
        :relobralo => "ReLoBRaLo",
    )
    runs = Dict(method => train_multi_output(method; temperature = 1.0) for method in methods)
end

# ╔═╡ d5ff548b-ff65-f83c-eebb-a73680aab54e
md"""
### 6. Comparison

`comparison_rows` collects, for each method, the final per-component training losses, the final weights (and their sum), per-component mean/max test error, and the scale-normalised `total_relative_error` — the single number that says which scheme balanced the three tasks best.
"""

# ╔═╡ 99999999-0504-4999-8999-999999999999
begin
    comparison_rows = [begin
        result = runs[method]
        err = result.final_errors
        final = result.history[end]
        (
            method = method_labels[method],
            loss_A = final.loss_A,
            loss_B = final.loss_B,
            loss_C = final.loss_C,
            w_A = final.w_A,
            w_B = final.w_B,
            w_C = final.w_C,
            weight_sum = final.w_A + final.w_B + final.w_C,
            mean_A = err.mean_A,
            mean_B = err.mean_B,
            mean_C = err.mean_C,
            max_A = err.max_A,
            max_B = err.max_B,
            max_C = err.max_C,
            total_relative_error = total_relative_error(err),
        )
    end for method in methods]
end

# ╔═╡ 2dacb4c3-469b-3091-1df1-66974c3e49b9
md"""
#### Weighted-loss curves

The weighted training loss (log scale) per epoch for each method. Equal weighting stalls on the dominant large-scale component; the adaptive schemes drive the composite loss down more evenly.
"""

# ╔═╡ aaaaaaaa-0504-4aaa-8aaa-aaaaaaaaaaaa
begin
    colors = Dict(
        :equal => :gray45,
        :inverse_loss => :goldenrod2,
        :softadapt => :seagreen3,
        :relobralo => :steelblue3,
    )

    fig = Figure(size = figure_size(RUN_MODE))
    ax = Axis(fig[1, 1], xlabel = "epoch", ylabel = "weighted training loss", yscale = log10)
    for method in methods
        hist = runs[method].history
        lines!(
            ax,
            [row.epoch for row in hist],
            max.([row.weighted_loss for row in hist], eps()),
            label = method_labels[method],
            color = colors[method],
            linewidth = 2,
        )
    end
    axislegend(ax; position = :rt)
    fig
end

# ╔═╡ b49ddb1d-713a-46de-c446-f96d450dab29
md"""
### 7. Sensitivity to the temperature \$T\$

We re-run ReLoBRaLo across a temperature sweep and report `total_relative_error` for each. The smoke sweep is \$\{0.1, 1.0\}\$; `teaching`/`production` use \$\{0.01, 0.1, 1.0, 10.0\}\$. Intuitively \$T \ll 1\$ is winner-take-all (one component at a time), \$T \gg 1\$ reverts to near-equal weights (the scale problem returns), and \$T \approx 1\$ is the robust default.
"""

# ╔═╡ bbbbbbbb-0504-4bbb-8bbb-bbbbbbbbbbbb
begin
    sensitivity_rows = [begin
        result = T == 1.0 ? runs[:relobralo] : train_multi_output(:relobralo; temperature = T)
        err = result.final_errors
        (
            temperature = T,
            total_relative_error = total_relative_error(err),
            mean_A = err.mean_A,
            mean_B = err.mean_B,
            mean_C = err.mean_C,
        )
    end for T in hp.temperature_sweep]
end

# ╔═╡ d5822a9f-3613-7709-4dd6-3afb4e2e6f48
md"""
### What the full Python notebook also plots

The Python ground truth adds two dedicated figures this preview folds into the logged history instead: the **equal-weighting result panels** (per-component fit under the failing baseline) and the **ReLoBRaLo weight-evolution plot** (\$w_A, w_B, w_C\$ over epochs). Here those quantities live in each run's `history` (`w_A`/`w_B`/`w_C`, `loss_*`, `test_*`) and in the summary below.
"""

# ╔═╡ 3ae3c742-4932-be0a-1430-a4f21371f165
md"""
### 8. Discussion

#### Key observations

1. **Equal weighting fails** when loss components differ by more than \$\sim 10\times\$ in scale: the large-scale component dominates the gradient and the small-scale one is essentially ignored.
2. **Inverse-loss weighting** is a simple fix but can be unstable, especially when a loss approaches zero.
3. **ReLoBRaLo** gives robust, adaptive balancing by using *relative* loss changes and a deterministic baseline/history blend that prevents weight drift in this notebook version. **SoftAdapt** (this preview's fourth scheme) similarly rebalances by relative loss slope.
4. The **temperature** \$T\$ controls the sharpness of rebalancing: \$T \ll 1\$ is winner-take-all, \$T \gg 1\$ reverts to the scale problem, and \$T \approx 1\$ is a robust default.

#### When to use loss balancing in economics

- **PINNs:** PDE residual + boundary + initial conditions at different scales.
- **DSGE / DEQNs:** Euler equations, budget constraints, and market-clearing conditions with different units and magnitudes.
- **Multi-task learning:** predicting several economic variables from shared features (e.g., GDP, inflation, unemployment).

#### References

- Bischof & Kraus (2025). *Multi-objective loss balancing for physics-informed deep learning.* arXiv:2110.09813; Computer Methods in Applied Mechanics and Engineering 439:117914.
- Chen, Badrinarayanan, Lee & Rabinovich (2018). *GradNorm: Gradient normalization for adaptive loss balancing in deep multitask networks.* ICML.
- Heydari, Thompson & Mehmood (2019). *SoftAdapt: Techniques for adaptive loss weighting of neural networks with multi-part loss functions.* arXiv:1912.12355.

The cell below returns a machine-checkable summary of this notebook's run (per-method comparison, the ReLoBRaLo temperature sweep, and finiteness checks).
"""

# ╔═╡ cccccccc-0504-4ccc-8ccc-cccccccccccc
begin
    finite_histories = all(methods) do method
        all(runs[method].history) do row
            all(isfinite, (
                row.loss_A, row.loss_B, row.loss_C,
                row.weighted_loss, row.w_A, row.w_B, row.w_C,
                row.test_A, row.test_B, row.test_C,
            ))
        end
    end

    (
        run_mode = RUN_MODE,
        seed = SEED,
        epochs = hp.epochs,
        train_size = hp.n_train,
        test_size = hp.n_test,
        target_ranges = target_ranges,
        train_mse_scales = train_mse_scales,
        comparison_rows = comparison_rows,
        relobralo_temperature_sweep = sensitivity_rows,
        all_weight_sums_count_normalized = all(row -> isapprox(row.weight_sum, 3.0; atol = 1e-8), comparison_rows),
        finite_histories = finite_histories,
    )
end

# ╔═╡ Cell order:
# ╟─11111111-0504-4111-8111-111111111111
# ╟─2414ce06-b73e-4e36-1bcc-9ea487da8364
# ╟─9903b80d-5ff9-9be6-91a1-27c1feb60512
# ╠═22222222-0504-4222-8222-222222222222
# ╠═33333333-0504-4333-8333-333333333333
# ╟─749f3cd9-caa7-2f00-5f43-221550e79872
# ╠═44444444-0504-4444-8444-444444444444
# ╟─2633f98e-0b88-c139-fb52-68aa97209a3b
# ╠═55555555-0504-4555-8555-555555555555
# ╟─031a1795-172c-6702-dac2-422259c53cad
# ╠═66666666-0504-4666-8666-666666666666
# ╟─1eb7e994-6f83-53c6-a8a5-f2842dc380ee
# ╠═77777777-0504-4777-8777-777777777777
# ╟─3a0b6f9f-588b-c791-4c2d-516a8e7e9e2b
# ╠═88888888-0504-4888-8888-888888888888
# ╟─d5ff548b-ff65-f83c-eebb-a73680aab54e
# ╠═99999999-0504-4999-8999-999999999999
# ╟─2dacb4c3-469b-3091-1df1-66974c3e49b9
# ╠═aaaaaaaa-0504-4aaa-8aaa-aaaaaaaaaaaa
# ╟─b49ddb1d-713a-46de-c446-f96d450dab29
# ╠═bbbbbbbb-0504-4bbb-8bbb-bbbbbbbbbbbb
# ╟─d5822a9f-3613-7709-4dd6-3afb4e2e6f48
# ╟─3ae3c742-4932-be0a-1430-a4f21371f165
# ╠═cccccccc-0504-4ccc-8ccc-cccccccccccc
