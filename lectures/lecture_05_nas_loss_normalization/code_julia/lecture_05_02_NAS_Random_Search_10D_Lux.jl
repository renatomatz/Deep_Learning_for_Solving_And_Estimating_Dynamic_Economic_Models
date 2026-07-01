### A Pluto.jl notebook ###
# v0.20.0

using Markdown
using InteractiveUtils

# ╔═╡ 11111111-0502-4111-8111-111111111111
md"""
# Lecture 05, Notebook 02: Random Search in Lux

This Julia notebook keeps the random-search idea from the 10-D NAS notebook. It
runs a small CPU smoke sweep over Lux MLP width, depth, activation, and learning
rate without writing `nas_outputs/` or figure artifacts.
"""

# ╔═╡ c5ce5877-bb54-e746-d6a7-234de7f44e80
md"""
## Lecture 05, Notebook 02: Random Search NAS on a 10-D Regression

**Course:** Deep Learning for Solving and Estimating Dynamic Models in Economics and Finance
**Script reference:** §4.1 (the hyperparameter space), §4.3 (random search; Bergstra & Bengio projection argument)
**Notebook role:** core
**Author:** Simon Scheidegger

*Julia/Lux/Pluto preview of* `lectures/lecture_05_nas_loss_normalization/code/lecture_05_02_NAS_Random_Search_10D.ipynb`.

> **Run mode.** The checked-in run uses `RUN_MODE = "smoke"` for a fast CPU sweep (few trials, few steps); the `teaching` and `production` budgets in the next cell widen the search. This Lux preview does not write the `nas_outputs/` CSVs or figure artifacts the Python notebook saves.
"""

# ╔═╡ 964695fa-cb40-2545-9efe-a96aabc708d9
md"""
Accompanies **Lecture 05 — Neural Architecture Search**.

This is the **first of two hands-on NAS examples** for Lecture 05:

| Notebook | Method | Testbed |
|:--|:--|:--|
| `02_NAS_Random_Search_10D` *(this notebook)* | Random Search, no external library | 10-D analytical regression |
| `03_NAS_RandomSearch_Hyperband` | Random Search + Successive Halving, from scratch | Genz Gaussian on \$[0,1]^2\$ |

The Python ground truth demonstrates an **easy, transparent** form of Neural Architecture Search (NAS) without extra tuning libraries — just Keras and a bit of Python. It:

1. Defines a **10-dimensional analytical regression task** (synthetic data with noise).
2. Specifies a **large search space** (depth, width, activations, optimizers, learning rates, batch sizes, normalization, dropout).
3. Uses **Random Search** to sample candidate architectures and hyperparameters.
4. Trains each candidate briefly and evaluates on a validation set.
5. **Selects the top 5** architectures by validation performance.
6. Retrains the top models longer on train+val and evaluates on a held-out test set.

This Julia/Lux preview keeps steps 1–4 with `Lux.Chain` MLPs, `Optimisers.Adam`, and `train_step!`, then reports the single best trial rather than retraining a top-5 cohort.
"""

# ╔═╡ e0ac80be-bace-3cdd-11c5-89b6af49497f
md"""
### Why Random Search?

Random Search is a surprisingly strong baseline for hyperparameter/architecture tuning:

- Scales trivially with search-space size.
- Parallelizable and easy to reason about.
- Often outperforms naive grid search given the same budget (many hyperparameters are low-sensitivity — the Bergstra & Bengio projection argument).

Other simple NAS strategies (not implemented here):

- **Successive Halving / Hyperband:** early-stop poor performers aggressively (see notebook 03).
- **Bayesian Optimization:** model the response surface to guide sampling.
- **Evolutionary Strategies:** mutate and select architectures over generations.

Here we focus on Random Search for **clarity** and **reproducibility**.
"""

# ╔═╡ f59d92b2-0dd4-5503-af0b-26c445a6ec4b
md"""
### 0. Setup

We activate the shared `DLEFJulia` project and load Lux, Optimisers, and the plotting/seed helpers. There are no `pip install` steps: seeding is explicit through `rng_from_seed(SEED)` and run-mode budgets come from `run_mode_budget`.
"""

# ╔═╡ 22222222-0502-4222-8222-222222222222
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

# ╔═╡ 33333333-0502-4333-8333-333333333333
begin
    RUN_MODE = "smoke"
    SEED = 0
    budgets = (
        smoke = (trials = 5, steps = 18, n_train = 160, n_test = 80),
        teaching = (trials = 20, steps = 80, n_train = 800, n_test = 240),
        production = (trials = 80, steps = 300, n_train = 2_000, n_test = 500),
    )
    hp = run_mode_budget(RUN_MODE; budgets)
    rng = rng_from_seed(SEED)
end

# ╔═╡ 5ddff1e4-f055-e82b-4322-75aaa8a59445
md"""
### 1. A 10-D analytical target and the search space

**Data.** We draw inputs \$x \in [-1,1]^{10}\$ uniformly and evaluate a fixed nonlinear target

\$\$y = \sum_{i=1}^{3}\sin(x_i) + 0.25\sum_{j=4}^{10} x_j^2.\$\$

Rows are features and columns are observations here (the feature-by-batch layout Lux expects), so `target_10d` returns a \$1 \times n\$ row. The Python ground truth instead draws inputs from a standard normal (`np.random.randn`, i.e. \$N(0,1)\$ with unbounded support), uses a richer target with pairwise interactions, an exponential term, and additive Gaussian noise, and standardizes inputs/targets with a `StandardScaler` (fit on train, applied to val/test). This compact preview instead samples inputs uniformly on \$[-1,1]^{10}\$ and drops the noise and the standardization, using a single train/test split.

**Search space.** We treat depth, width, learning rate, and activation as tunable:

- **width** ∈ {12, 20, 32} units per hidden layer
- **depth** ∈ {1, 2, 3} hidden layers
- **learning rate** ∈ {0.003, 0.006, 0.01}
- **activation** ∈ {`relu`, `tanh`}

The Python space is larger still (depth 1–6, width 32–512, `relu`/`gelu`/`tanh`/`elu`, batch-norm on/off, dropout 0–0.5, `adam`/`rmsprop`/`sgd`, log-uniform learning rate, batch size ∈ {64, 128, 256}). The random-search machinery is identical; only the grid differs.
"""

# ╔═╡ 44444444-0502-4444-8444-444444444444
begin
    target_10d(x) = reshape(sum(sin.(x[1:3, :]); dims = 1) .+ 0.25 .* sum(x[4:10, :] .^ 2; dims = 1), 1, :)
    x_train = 2 .* rand(rng, 10, hp.n_train) .- 1
    y_train = target_10d(x_train)
    x_test = 2 .* rand(rng, 10, hp.n_test) .- 1
    y_test = target_10d(x_test)

    activation_pool = [NNlib.relu, NNlib.tanh]
    widths = [12, 20, 32]
    depths = [1, 2, 3]
    rates = [0.003, 0.006, 0.01]
end

# ╔═╡ d7033361-0c6d-e259-ad95-b6d022bab7e3
md"""
### 2. Model builder and the random-search loop

`train_candidate` maps a sampled configuration to a `Lux.Chain` MLP via `make_mlp(10, hidden, 1; activation)` — the Julia counterpart of the Python `build_model` (a Keras functional `Model` — `keras.Input` piped through `keras.Model(inputs, outputs)`, not `keras.Sequential` — with a linear regression head). Each candidate is trained with `Optimisers.Adam(config.lr)` through `setup_training` / `train_step!` (gradient-clipped at norm 10), scoring the explicit `prediction, st = model(x, ps, st)` call with `mse_loss`.

The random search then:

- samples `hp.trials` configurations uniformly from the pools above,
- trains each for `hp.steps` gradient steps,
- records the held-out test MSE,

and keeps the single configuration with the lowest test loss as `best_trial`. The Python notebook additionally uses per-trial early stopping and validation-based ranking; the smoke budget here is small enough that each candidate is trained to completion.
"""

# ╔═╡ 55555555-0502-4555-8555-555555555555
begin
    function train_candidate(config, seed_offset)
        hidden = ntuple(_ -> config.width, config.depth)
        model = make_mlp(10, hidden, 1; activation = config.activation)
        state = setup_training(rng_from_seed(SEED; offset = seed_offset), model, Optimisers.Adam(config.lr); parameter_type = Float64)
        loss_fn(model, ps, st, batch) = begin
            prediction, st_new = model(batch.x, ps, st)
            return mse_loss(prediction, batch.y), st_new
        end
        batch = (x = x_train, y = y_train)
        initial = loss_value(state, loss_fn, batch)
        for _ in 1:hp.steps
            train_step!(state, loss_fn, batch; max_grad_norm = 10.0)
        end
        test_loss = loss_value(state, loss_fn, (x = x_test, y = y_test))
        return (config..., initial_loss = initial, test_loss = test_loss)
    end

    configs = [(
        width = widths[rand(rng, 1:length(widths))],
        depth = depths[rand(rng, 1:length(depths))],
        lr = rates[rand(rng, 1:length(rates))],
        activation = activation_pool[rand(rng, 1:length(activation_pool))],
    ) for _ in 1:hp.trials]

    trial_results = [train_candidate(config, i) for (i, config) in enumerate(configs)]
    best_trial = trial_results[argmin([row.test_loss for row in trial_results])]
end

# ╔═╡ 82c6199f-e248-99e2-e1f1-7f3a99bc5437
md"""
### 3. Visualize the search results

A quick scatter of test MSE (log scale) across trials shows how sensitive the objective is to the sampled architecture and learning rate.
"""

# ╔═╡ 66666666-0502-4666-8666-666666666666
begin
    fig = Figure(size = figure_size(RUN_MODE))
    ax = Axis(fig[1, 1], xlabel = "trial", ylabel = "test MSE", yscale = log10)
    scatter!(ax, 1:length(trial_results), [row.test_loss for row in trial_results]; color = :dodgerblue3)
    fig
end

# ╔═╡ 92174c6b-aca2-bacf-9808-11d33b4c5387
md"""
### What the full Python notebook also covers

Beyond the single-best pick reported below, the Python ground truth continues with a top-5 workflow:

- **Select the top 5** configurations by validation MSE.
- **Retrain the top 5** on the concatenated train+val set for more epochs and evaluate on the held-out **test** split, mapping predictions back to the original target scale (RMSE in original units).
- **Compare** the top-5 test RMSE as a bar chart.
- **Introspect** which hyperparameter values (depth, width, activation, batch-norm, dropout, optimizer, batch size) recur among the winners — descriptive, not causal.
- **Save artifacts:** the full trial table and the top-5 leaderboard as CSVs under `nas_outputs/`.

This Lux preview stops at the single lowest-test-loss trial and writes no artifacts.
"""

# ╔═╡ df4455aa-e99c-694e-420e-9ed9bde1604b
md"""
### Teaching notes and takeaways

- **Budget vs. space:** raise `hp.trials` or `hp.steps` (via `RUN_MODE`) for better results; lower them for speed.
- **Alternative strategies:** Successive Halving / Hyperband (notebook 03), Bayesian optimization with a surrogate, or evolutionary search over a population of architectures.
- **Search-space design:** residual connections, per-layer activations, weight decay, and spectral norm all extend the grid.
- **Metrics:** we report MSE; for noisy targets, RMSE in original units is easier to interpret.
- **Reproducibility:** fix seeds (`SEED = 0`), record each trial's config, and log the run mode.
- **Caveat:** Random Search is strong but not magic — with a huge space and a tiny budget you may under-sample the good regions.

The cell below returns a machine-checkable summary of this notebook's smoke run.
"""

# ╔═╡ 77777777-0502-4777-8777-777777777777
(
    trials = length(trial_results),
    best_width = best_trial.width,
    best_depth = best_trial.depth,
    best_lr = best_trial.lr,
    best_test_loss = best_trial.test_loss,
    finite_losses = all(row -> isfinite(row.test_loss), trial_results),
)

# ╔═╡ Cell order:
# ╟─11111111-0502-4111-8111-111111111111
# ╟─c5ce5877-bb54-e746-d6a7-234de7f44e80
# ╟─964695fa-cb40-2545-9efe-a96aabc708d9
# ╟─e0ac80be-bace-3cdd-11c5-89b6af49497f
# ╟─f59d92b2-0dd4-5503-af0b-26c445a6ec4b
# ╠═22222222-0502-4222-8222-222222222222
# ╠═33333333-0502-4333-8333-333333333333
# ╟─5ddff1e4-f055-e82b-4322-75aaa8a59445
# ╠═44444444-0502-4444-8444-444444444444
# ╟─d7033361-0c6d-e259-ad95-b6d022bab7e3
# ╠═55555555-0502-4555-8555-555555555555
# ╟─82c6199f-e248-99e2-e1f1-7f3a99bc5437
# ╠═66666666-0502-4666-8666-666666666666
# ╟─92174c6b-aca2-bacf-9808-11d33b4c5387
# ╟─df4455aa-e99c-694e-420e-9ed9bde1604b
# ╠═77777777-0502-4777-8777-777777777777
