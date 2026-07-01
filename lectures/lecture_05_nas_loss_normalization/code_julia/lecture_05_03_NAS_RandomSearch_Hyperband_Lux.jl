### A Pluto.jl notebook ###
# v0.20.0

using Markdown
using InteractiveUtils

# ╔═╡ 11111111-0503-4111-8111-111111111111
md"""
# Lecture 05, Notebook 03: Random Search and Successive Halving

This Pluto translation follows the Python ground-truth notebook at smoke
scale: the target is the 2-D Genz Gaussian on ``[0, 1]^2``, the search space is
depth/width/activation/learning rate, lower validation loss is better, and the
SHA loop promotes the top ``1 / eta`` configurations after each rung.

Trial records stay in memory; the Julia preview does not read or rewrite the
Python `nas_results/search_records.pkl` cache.
"""

# ╔═╡ 470d7b38-5a88-d3d5-e8f5-4c4c3f89d3bc
md"""
## Lecture 05, Notebook 03: Random Search and Successive Halving from Scratch

**Course:** Deep Learning for Solving and Estimating Dynamic Models in Economics and Finance
**Script reference:** §4.3 (random search), §4.5 (Hyperband and successive halving), §4.7 (implementing the search in practice)
**Notebook role:** core
**Author:** Simon Scheidegger

*Julia/Lux/Pluto preview of* `lectures/lecture_05_nas_loss_normalization/code/lecture_05_03_NAS_RandomSearch_Hyperband.ipynb`.

> **Run mode.** `RUN_MODE = "smoke"` runs a small CPU bracket (few trials, short rungs); `teaching` and `production` mirror the Python 30-trial random search and the 27 → 9 → 3 successive-halving bracket. Trial records stay in memory — this preview does **not** read or rewrite the Python `nas_results/search_records.pkl` cache, and it does not save the `nas_search_results.*` / `nas_best_surface.*` figures.
"""

# ╔═╡ 3f0c5cda-5b5b-1fba-9d38-d349562e8e8c
md"""
Accompanies **Lecture 05 — Neural Architecture Search**.

We implement the two key automated-search algorithms from §4 of the lecture script directly, without relying on a hyperparameter-tuning library. The search space and target task are the same as in the slides (Genz Gaussian on \$[0,1]^2\$, search over depth/width/activation/learning rate), which lets you read off the algorithm logic with no extra abstraction.

> **Production tooling.** Real projects rarely hand-roll the search loop. Established libraries wrap (and parallelize) the same algorithms: KerasTuner, Optuna, Ray Tune, Hyperopt, Ax/BoTorch, NNI, AutoKeras. We mention them here for completeness; the algorithms below are what they all implement underneath.

This is the **second of two hands-on NAS examples** for Lecture 05:

| Notebook | Method | Testbed |
|:--|:--|:--|
| `02_NAS_Random_Search_10D` | Random Search, no external library | 10-D analytical regression |
| `03_NAS_RandomSearch_Hyperband` *(this notebook)* | Random Search + Successive Halving, from scratch | Genz Gaussian on \$[0,1]^2\$ |

**Goals**

1. Understand why architecture choice matters for function approximation.
2. Implement Random Search as a transparent Julia loop.
3. Implement Successive Halving (the building block of Hyperband) as a transparent Julia loop.
4. Compare both against a fixed baseline on the Genz Gaussian test function.

**Outline**

| Section | Topic |
|:--------|:------|
| 1 | Setup and data generation |
| 2 | Baseline: fixed architecture |
| 3 | Search space and trial helper |
| 4 | Random Search from scratch |
| 5 | Successive Halving from scratch |
| 6 | Comparison and best-model surface |
| 7 | Discussion |
"""

# ╔═╡ 22222222-0503-4222-8222-222222222222
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

# ╔═╡ 33333333-0503-4333-8333-333333333333
begin
    RUN_MODE = "smoke"
    SEED = 0

    # Smoke keeps the Python algorithms but uses fewer Lux trials/epochs so
    # notebook smoke includes stay CPU-fast; teaching/production mirror the
    # Python 30-trial random search and 27 -> 9 -> 3 SHA bracket.
    budgets = (
        smoke = (
            n_train = 64,
            n_test = 128,
            baseline_epochs = 3,
            random_trials = 4,
            random_epochs = 3,
            eta = 3,
            sha_n0 = 6,
            sha_r0 = 1,
            sha_rounds = 2,
            final_epochs = 6,
            patience = 3,
            final_patience = 4,
            grid_n = 24,
        ),
        teaching = (
            n_train = 1_000,
            n_test = 2_000,
            baseline_epochs = 100,
            random_trials = 30,
            random_epochs = 50,
            eta = 3,
            sha_n0 = 27,
            sha_r0 = 8,
            sha_rounds = 3,
            final_epochs = 300,
            patience = 10,
            final_patience = 30,
            grid_n = 80,
        ),
        production = (
            n_train = 1_000,
            n_test = 2_000,
            baseline_epochs = 100,
            random_trials = 30,
            random_epochs = 50,
            eta = 3,
            sha_n0 = 27,
            sha_r0 = 8,
            sha_rounds = 3,
            final_epochs = 300,
            patience = 10,
            final_patience = 30,
            grid_n = 80,
        ),
    )
    hp = run_mode_budget(RUN_MODE; budgets)
end

# ╔═╡ bd9f5eb1-a0e6-955e-ac39-261eef4f42ec
md"""
### 1. The NAS problem and data

**Neural Architecture Search (NAS)** aims to find the best network architecture for a given task *automatically*.

The **search space** covers the number of hidden layers, the number of units per layer, the activation function, and the learning rate.

We use the **Genz Gaussian** function on \$[0,1]^2\$ as our test case:

\$\$f(\mathbf{x}) = \exp\!\left(-\sum_{i=1}^{d} c_i^2 (x_i - w_i)^2\right).\$\$

The shape constants `c_param` and `w_param` are the exact `NumPy RandomState(7)` values printed by the Python notebook, so the target surface matches. Inputs are drawn uniformly on \$[0,1]^2\$ and split into train and test sets, stored feature-by-batch (\$d \times n\$) as Lux expects.
"""

# ╔═╡ 44444444-0503-4444-8444-444444444444
begin
    d = 2

    # Exact NumPy RandomState(7) constants printed by the Python notebook.
    c_param = [1.15261658, 2.55983758]
    w_param = [0.46304554, 0.63407911]

    function genz_gaussian(x, c, w)
        c_col = reshape(c, :, 1)
        w_col = reshape(w, :, 1)
        return reshape(exp.(-sum((c_col .^ 2) .* (x .- w_col) .^ 2; dims = 1)), 1, :)
    end

    rng_data = rng_from_seed(SEED)
    x_train = rand(rng_data, d, hp.n_train)
    y_train = genz_gaussian(x_train, c_param, w_param)
    x_test = rand(rng_data, d, hp.n_test)
    y_test = genz_gaussian(x_test, c_param, w_param)

    y_range = (minimum(y_train), maximum(y_train))
end

# ╔═╡ 77ba4c2c-c395-cf7a-2fc9-a1fc778d208b
md"""
### 3. Search space and the `train_trial` helper

Both Random Search and Successive Halving draw architectures from the same search space:

| Hyperparameter | Range |
|:---|:---|
| `layers` | \$\{1, 2, 3, 4, 5\}\$ |
| `units` (per layer) | \$\{32, 64, 96, \ldots, 256\}\$ |
| `activation` | \$\{\mathrm{relu}, \mathrm{tanh}, \mathrm{swish}\}\$ |
| `lr` | log-uniform on \$[10^{-4}, 10^{-2}]\$ |

`sample_config(rng)` draws one configuration; `build_model` turns it into a `Lux.Chain` via `make_mlp` (the Julia counterpart of the Python `keras.Sequential` builder). The trial helper `train_trial(config, epochs; …)` trains the model for the requested number of epochs with **early stopping** on validation loss (patience-based, keeping the best parameters) and returns a record with the final validation loss, MAE, wall-clock time, epochs run, and parameter count. This is the only primitive the search algorithms below call.
"""

# ╔═╡ 55555555-0503-4555-8555-555555555555
begin
    LAYER_OPTIONS = collect(1:5)
    UNIT_OPTIONS = collect(32:32:256)
    ACT_OPTIONS = ("relu", "tanh", "swish")
    LR_LOG_LO, LR_LOG_HI = -4.0, -2.0

    activation_fn(name) =
        name == "relu" ? NNlib.relu :
        name == "tanh" ? NNlib.tanh :
        name == "swish" ? NNlib.swish :
        throw(ArgumentError("unknown activation: $name"))

    function sample_config(rng)
        n_layers = rand(rng, LAYER_OPTIONS)
        units = Tuple(rand(rng, UNIT_OPTIONS) for _ in 1:n_layers)
        activation = rand(rng, ACT_OPTIONS)
        lr = 10.0 ^ (LR_LOG_LO + (LR_LOG_HI - LR_LOG_LO) * rand(rng))
        return (layers = n_layers, units = units, activation = activation, lr = lr)
    end

    function build_model(config)
        return make_mlp(d, config.units, 1; activation = activation_fn(config.activation))
    end

    param_count(::Nothing) = 0
    param_count(x::Number) = 1
    param_count(x::AbstractArray) = length(x)
    param_count(x::NamedTuple) = sum(param_count(v) for v in values(x); init = 0)
    param_count(x::Tuple) = sum(param_count(v) for v in x; init = 0)

    function train_trial(config, epochs; seed_offset, patience)
        model = build_model(config)
        state = setup_training(
            rng_from_seed(SEED; offset = seed_offset),
            model,
            Optimisers.Adam(config.lr);
            parameter_type = Float64,
        )
        train_batch = (x = x_train, y = y_train)
        val_batch = (x = x_test, y = y_test)

        loss_fn(model, ps, st, batch) = begin
            prediction, st_new = model(batch.x, ps, st)
            return mse_loss(prediction, batch.y), st_new
        end

        t0 = time()
        best_val = Inf
        best_ps = deepcopy(state.ps)
        best_st = deepcopy(state.st)
        stale_epochs = 0
        epochs_ran = 0

        for epoch in 1:epochs
            train_step!(state, loss_fn, train_batch; max_grad_norm = 10.0)
            val_loss = loss_value(state, loss_fn, val_batch)
            epochs_ran = epoch
            if val_loss < best_val
                best_val = val_loss
                best_ps = deepcopy(state.ps)
                best_st = deepcopy(state.st)
                stale_epochs = 0
            else
                stale_epochs += 1
            end
            stale_epochs >= patience && break
        end

        pred, _ = model(x_test, best_ps, best_st)
        mae = mean(abs.(pred .- y_test))
        wall = time() - t0
        record = (
            config = config,
            val_loss = Float64(best_val),
            mae = Float64(mae),
            time = Float64(wall),
            epochs = epochs_ran,
            params = param_count(best_ps),
        )
        return record, (model = model, ps = best_ps, st = best_st)
    end

    fixed_config = (layers = 3, units = (64, 64, 64), activation = "relu", lr = 1e-3)
    demo_config = sample_config(rng_from_seed(SEED))
end

# ╔═╡ 4f1773c6-d933-9040-c0bd-825ca8e2a543
md"""
### 2 & 4. Fixed baseline and Random Search from scratch

**Baseline.** As a reference point we train a fixed, intuition-driven architecture — a \$3 \times 64\$ ReLU MLP — with the same `train_trial` helper. Automated search has to beat this.

**Random Search.** The loop is six lines of real logic: sample \$N\$ architectures independently from the search space, train each for a small fixed budget with early stopping, and remember the best.

```text
for i = 1..N:
    cfg     <- sample_config(rng)
    record  <- train_trial(cfg, epochs = R)
    log(record)
return argmin val_loss over records
```

Random Search ignores the structure of the response surface, but it is embarrassingly parallel and gives an honest baseline that more sophisticated methods need to beat (Bergstra & Bengio, 2012).
"""

# ╔═╡ 66666666-0503-4666-8666-666666666666
begin
    baseline_record, _ = train_trial(
        fixed_config,
        hp.baseline_epochs;
        seed_offset = 10,
        patience = hp.patience,
    )

    rng_rs = rng_from_seed(SEED)
    random_records = NamedTuple[]
    for i in 1:hp.random_trials
        cfg = sample_config(rng_rs)
        rec, _ = train_trial(
            cfg,
            hp.random_epochs;
            seed_offset = 1_000 + i,
            patience = hp.patience,
        )
        push!(random_records, (trial = i, rec...))
    end

    best_record(records) = records[argmin([row.val_loss for row in records])]
    best_random = best_record(random_records)
end

# ╔═╡ 38105851-7a2e-8880-69c0-09394c9f5483
md"""
### 5. Successive Halving from scratch

Successive Halving (SHA) is the inner loop of Hyperband (Li et al., 2018). The idea: cheap configurations should be killed early, expensive ones promoted.

With \$n_0 = 27\$ initial random configs and a halving rate \$\eta = 3\$, SHA runs three rounds and returns a 1-of-27 winner:

| Round | Configs alive | Epoch budget per config |
|:--|:--:|:--:|
| 0 | 27 | \$r_0 = 8\$ |
| 1 | 9 | \$r_1 = 24\$ |
| 2 | 3 | \$r_2 = 72\$ |
| 3 | 1 | (final winner) |

Each round trains every surviving config for \$r_k = r_0\,\eta^{k}\$ epochs, sorts by validation loss, and keeps the top \$1/\eta\$ (`n_keep = length ÷ η`). The total epoch budget is roughly matched to Random Search (\$27 \cdot 8 + 9 \cdot 24 + 3 \cdot 72 \approx 648\$ epoch-trials, vs \$30 \cdot 50 = 1500\$), but SHA spends most of its compute on the promising tail. The smoke run uses a smaller bracket (the budget above shrinks \$n_0\$, \$r_0\$, and the number of rounds) so the algorithm stays identical while the notebook remains CPU-fast.
"""

# ╔═╡ 77777777-0503-4777-8777-777777777777
begin
    rng_sha = rng_from_seed(SEED + 1)
    survivors_ref = Ref([sample_config(rng_sha) for _ in 1:hp.sha_n0])
    sha_records = NamedTuple[]
    rung_summaries = NamedTuple[]

    for k in 0:(hp.sha_rounds - 1)
        epochs_k = hp.sha_r0 * (hp.eta ^ k)
        round_records = NamedTuple[]
        for (j, cfg) in enumerate(survivors_ref[])
            rec, _ = train_trial(
                cfg,
                epochs_k;
                seed_offset = 2_000 + 100 * k + j,
                patience = hp.patience,
            )
            row = (round = k, rung_epochs = epochs_k, rec...)
            push!(round_records, row)
            push!(sha_records, row)
        end

        sort!(round_records; by = row -> row.val_loss)
        n_keep = max(1, length(round_records) ÷ hp.eta)
        push!(
            rung_summaries,
            (
                round = k,
                evaluated = length(round_records),
                epochs = epochs_k,
                best_val_loss = round_records[1].val_loss,
                promoted = n_keep,
            ),
        )
        survivors_ref[] = [row.config for row in round_records[1:n_keep]]
    end

    best_sha = best_record(sha_records)
    random_wins = best_random.val_loss <= best_sha.val_loss
    best_overall = random_wins ? best_random : best_sha
    best_method = random_wins ? "Random Search" : "Successive Halving"
end

# ╔═╡ 9aebf874-a57c-81ca-b6cf-415f7de83995
md"""
### 6. Comparison

We plot the **best-so-far** validation loss against trial index for each algorithm; the fixed baseline appears as a constant horizontal line, and a bar chart compares the three final losses. In the Python notebook this figure is saved as `nas_search_results.pdf` for the slides; here it is only displayed.
"""

# ╔═╡ 88888888-0503-4888-8888-888888888888
begin
    function best_so_far(records)
        out = Float64[]
        current = Inf
        for row in records
            current = min(current, row.val_loss)
            push!(out, current)
        end
        return out
    end

    rs_curve = best_so_far(random_records)
    sha_curve = best_so_far(sha_records)

    fig_search = Figure(size = figure_size(RUN_MODE))
    ax1 = Axis(fig_search[1, 1], xlabel = "trial index", ylabel = "best validation loss so far", yscale = log10)
    lines!(ax1, 1:length(rs_curve), rs_curve; color = :tomato, label = "Random Search")
    scatter!(ax1, 1:length(rs_curve), rs_curve; color = :tomato)
    lines!(ax1, 1:length(sha_curve), sha_curve; color = :teal, label = "Successive Halving")
    scatter!(ax1, 1:length(sha_curve), sha_curve; color = :teal, marker = :rect)
    hlines!(ax1, [baseline_record.val_loss]; color = :gray45, linestyle = :dash, label = "Baseline")
    axislegend(ax1; position = :rt)

    ax2 = Axis(fig_search[1, 2], ylabel = "best validation loss", yscale = log10)
    vals = [baseline_record.val_loss, rs_curve[end], sha_curve[end]]
    barplot!(ax2, 1:3, vals; color = [:gray60, :tomato, :teal])
    ax2.xticks = (1:3, ["Baseline", "Random", "SHA"])

    fig_search
end

# ╔═╡ 35a11364-84f6-ac5d-d503-f107c4cb5c1f
md"""
#### Best-model surface

We refit the overall winning configuration (`best_overall`) for a longer budget, then plot the true Genz surface, the network prediction \$\hat{f}\$, and the absolute error over a grid on \$[0,1]^2\$. In the Python notebook this is saved as `nas_best_surface.pdf`.
"""

# ╔═╡ 99999999-0503-4999-8999-999999999999
begin
    final_record, final_fit = train_trial(
        best_overall.config,
        hp.final_epochs;
        seed_offset = 3_000,
        patience = hp.final_patience,
    )

    grid = collect(range(0.0, 1.0; length = hp.grid_n))
    x1 = repeat(reshape(grid, 1, :), hp.grid_n, 1)
    x2 = repeat(reshape(grid, :, 1), 1, hp.grid_n)
    x_grid = permutedims(hcat(vec(x1), vec(x2)))

    z_true = reshape(vec(genz_gaussian(x_grid, c_param, w_param)), hp.grid_n, hp.grid_n)
    z_pred_raw, _ = final_fit.model(x_grid, final_fit.ps, final_fit.st)
    z_pred = reshape(vec(z_pred_raw), hp.grid_n, hp.grid_n)
    z_err = abs.(z_true .- z_pred)

    fig_surface = Figure(size = (900, 280))
    ax_true = Axis3(fig_surface[1, 1], xlabel = "x1", ylabel = "x2", zlabel = "f")
    ax_pred = Axis3(fig_surface[1, 2], xlabel = "x1", ylabel = "x2", zlabel = "fhat")
    ax_err = Axis3(fig_surface[1, 3], xlabel = "x1", ylabel = "x2", zlabel = "abs error")

    surface!(ax_true, grid, grid, z_true; colormap = :viridis)
    surface!(ax_pred, grid, grid, z_pred; colormap = :viridis)
    surface!(ax_err, grid, grid, z_err; colormap = :reds)

    fig_surface
end

# ╔═╡ 92d1239d-1122-17a3-7034-cbc0fe2513da
md"""
### What the full Python notebook also does

Between the search loops the Python ground truth **caches** trial records to `nas_results/search_records.pkl`, short-circuiting re-runs. This Julia preview keeps every record in memory and reruns from scratch each time (`cache_used = false` in the summary below).
"""

# ╔═╡ 6ef47b98-98fc-1012-9fba-8568463a0e2d
md"""
### 7. Discussion

#### Key observations

1. **Both automated methods beat the fixed baseline.** Even with a modest number of trials, Random Search finds an architecture better than the intuition-driven \$3 \times 64\$ ReLU choice (Bergstra & Bengio, 2012).
2. **Random Search is a surprisingly strong baseline.** Sampling independently across all hyperparameters covers the important dimensions well, especially when only a few axes matter (here, learning rate and width).
3. **Successive Halving spends compute where it pays off.** By killing poorly-initialized configs after \$r_0\$ epochs, SHA reallocates the budget to the long tail of survivors. Hyperband (Li et al., 2018) wraps SHA in an outer loop over different initial \$(n_0, r_0)\$ pairs to hedge against the early-stopping bias.
4. **Bayesian Optimisation** (not implemented here) typically matches or slightly beats Random Search on smooth, low-dimensional response surfaces, at the cost of a sequential bottleneck. We cover the GP + Expected-Improvement machinery in **Day 7** (Surrogates, GPs, and Bayesian Estimation).

#### When does NAS matter in economics?

- **Surrogate models.** When replacing expensive simulations (climate models, agent-based models) with neural networks, architecture choice directly determines approximation quality.
- **Policy-function approximation.** In deep equilibrium nets (Lecture 03), the network approximates policy/value functions; NAS can automate the architecture selection usually done by hand.
- **Diminishing returns.** For simple functions (\$d \leq 5\$), a few dozen random trials is usually enough. For high-dimensional problems the budget should scale with input dimension, and SHA / Hyperband become essential to keep wall-clock time tractable.

#### Production tooling

The from-scratch loops here are pedagogical. In practice you would reach for **KerasTuner** (TF/Keras workflows), **Optuna** (framework-agnostic, define-by-run), **Ray Tune** (distributed trials), **Hyperopt** (TPE), **Ax / BoTorch** (Bayesian optimization on PyTorch), or **NNI** / **AutoKeras** (broader AutoML). Every one of them implements the algorithms above under the hood.

#### References

- Bergstra, J., & Bengio, Y. (2012). *Random search for hyper-parameter optimization.* JMLR 13, 281–305.
- Li, L., Jamieson, K., DeSalvo, G., Rostamizadeh, A., & Talwalkar, A. (2018). *Hyperband: A novel bandit-based approach to hyperparameter optimization.* JMLR 18(185), 1–52.
- Garnett, R. (2023). *Bayesian optimization.* Cambridge University Press.
- Snoek, J., Larochelle, H., & Adams, R. P. (2012). *Practical Bayesian optimization of machine learning algorithms.* NeurIPS.

The cell below returns a machine-checkable summary of this notebook's run (best configs per method, rung summaries, and finiteness checks).
"""

# ╔═╡ aaaaaaaa-0503-4aaa-8aaa-aaaaaaaaaaaa
(
    run_mode = RUN_MODE,
    seed = SEED,
    d = d,
    c_param = c_param,
    w_param = w_param,
    y_train_range = y_range,
    demo_config = demo_config,
    baseline = baseline_record,
    random_trials = length(random_records),
    best_random = best_random,
    sha_trials = length(sha_records),
    sha_rungs = rung_summaries,
    final_survivors = length(survivors_ref[]),
    best_sha = best_sha,
    best_method = best_method,
    final_refit = final_record,
    final_surface_mae = mean(z_err),
    finite_losses = all(row -> isfinite(row.val_loss), vcat(random_records, sha_records)) &&
        isfinite(baseline_record.val_loss) &&
        isfinite(final_record.val_loss),
    cache_used = false,
)

# ╔═╡ Cell order:
# ╟─11111111-0503-4111-8111-111111111111
# ╟─470d7b38-5a88-d3d5-e8f5-4c4c3f89d3bc
# ╟─3f0c5cda-5b5b-1fba-9d38-d349562e8e8c
# ╠═22222222-0503-4222-8222-222222222222
# ╠═33333333-0503-4333-8333-333333333333
# ╟─bd9f5eb1-a0e6-955e-ac39-261eef4f42ec
# ╠═44444444-0503-4444-8444-444444444444
# ╟─77ba4c2c-c395-cf7a-2fc9-a1fc778d208b
# ╠═55555555-0503-4555-8555-555555555555
# ╟─4f1773c6-d933-9040-c0bd-825ca8e2a543
# ╠═66666666-0503-4666-8666-666666666666
# ╟─38105851-7a2e-8880-69c0-09394c9f5483
# ╠═77777777-0503-4777-8777-777777777777
# ╟─9aebf874-a57c-81ca-b6cf-415f7de83995
# ╠═88888888-0503-4888-8888-888888888888
# ╟─35a11364-84f6-ac5d-d503-f107c4cb5c1f
# ╠═99999999-0503-4999-8999-999999999999
# ╟─92d1239d-1122-17a3-7034-cbc0fe2513da
# ╟─6ef47b98-98fc-1012-9fba-8568463a0e2d
# ╠═aaaaaaaa-0503-4aaa-8aaa-aaaaaaaaaaaa
