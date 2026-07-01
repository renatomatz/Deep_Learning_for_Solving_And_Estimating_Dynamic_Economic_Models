### A Pluto.jl notebook ###
# v0.20.0

using Markdown
using InteractiveUtils

# ╔═╡ 11111111-0207-4111-8111-111111111111
md"""
# Lecture 02, Notebook 07: Genz Approximation and Loss Functions

This notebook keeps the Genz-function approximation example and uses the shared
loss-kernel helper to compare MSE, MAE, Huber, quantile, CVaR, and log-cosh on
one residual vector.
"""

# ╔═╡ 4511dab1-df93-0437-8430-867543fec1ca
md"""
## Lecture 02, Notebook 07: Loss Functions and Genz Function Approximation

**Course:** Deep Learning for Solving and Estimating Dynamic Models in Economics and Finance
**Script reference:** §1.3–1.4 (Loss functions and function approximation: Genz test functions)
**Notebook role:** core
**Author:** Simon Scheidegger

*Julia/Lux/Pluto preview of* `lectures/lecture_02_intro_deep_learning/code/lecture_02_07_Genz_Approximation_and_Loss_Functions.ipynb`.

> **Run mode.** The checked-in run uses `RUN_MODE = "smoke"` for fast execution; set `RUN_MODE` to `"teaching"` or `"production"` in the setup cell for the larger training sets and longer schedules quoted in the slides.
"""

# ╔═╡ 0074b36f-0b50-8f9f-ac4a-8cca2e2198c1
md"""
## Genz Test Functions: Neural Network Approximation and Loss Functions

This notebook accompanies **Lecture 1 — Introduction to ML & Deep Learning**.

**Goals:**
1. Implement Genz (1987) test functions.
2. Train a neural network to approximate one and visualise the result.
3. Study how the choice of **loss function** (MSE, MAE, Huber, quantile, CVaR, log-cosh) affects robustness, especially with outliers.
4. Note the curse of dimensionality when scaling from \$d=2\$ to \$d=5\$.

This compact Lux/Pluto preview implements two Genz functions — \$f_1\$ (oscillatory) and \$f_3\$ (corner peak) — trains one Lux MLP on the corner-peak target with MSE, then evaluates the shared `LOSS_KERNELS` gallery on a single residual vector so the loss families can be compared on identical errors. The larger contamination, multi-quantile, and \$d=5\$ experiments remain in the Python ground truth and are summarised at the end.

**Outline**

| Section | Topic |
|:--------|:------|
| 1 | The Genz test functions |
| 2 | Visualisation (\$d{=}2\$) |
| 3 | NN approximation with MSE loss |
| 4 | Convergence with training-set size |
| 5 | Beyond MSE — robust and asymmetric losses |
| 6 | Comparing losses on contaminated data |
| 7 | Quantile loss and asymmetric behaviour |
| 8 | Scaling to \$d{=}5\$: the curse of dimensionality |
| 9 | Discussion and takeaways |
"""

# ╔═╡ 22222222-0207-4222-8222-222222222222
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

# ╔═╡ 33333333-0207-4333-8333-333333333333
begin
    RUN_MODE = "smoke"
    SEED = 0
    budgets = (
        smoke = (steps = 40, n_train = 192, n_test = 96),
        teaching = (steps = 300, n_train = 1_000, n_test = 400),
        production = (steps = 1_500, n_train = 4_000, n_test = 1_000),
    )
    hp = run_mode_budget(RUN_MODE; budgets)
    rng = rng_from_seed(SEED)
end

# ╔═╡ ec224516-9e24-eb39-894d-735dbd90baec
md"""
---
### 1. The Genz (1987) Test Functions

Genz (1987) introduced six families of integrands on \$[0,1]^d\$ widely used as benchmarks for numerical integration and function approximation. Each family is parameterised by vectors \$\mathbf{c}\$ (difficulty) and \$\mathbf{w}\$ (shift).

| # | Name | Formula |
|---|------|:--------|
| \$f_1\$ | **Oscillatory** | \$\cos\!\bigl(2\pi w_1 + \sum_i c_i x_i\bigr)\$ |
| \$f_2\$ | Product peak | \$\prod_i (c_i^{-2} + (x_i - w_i)^2)^{-1}\$ |
| \$f_3\$ | **Corner peak** | \$(1 + \sum_i c_i x_i)^{-(d+1)}\$ |
| \$f_4\$ | Gaussian | \$\exp\!\bigl(-\sum_i c_i^2 (x_i - w_i)^2\bigr)\$ |
| \$f_5\$ | Continuous | \$\exp\!\bigl(-\sum_i c_i \lvert x_i - w_i\rvert\bigr)\$ |
| \$f_6\$ | Discontinuous | \$0\$ if \$x_1>w_1\$ or \$x_2>w_2\$; else \$\exp(\sum_i c_i x_i)\$ |

The next cell implements \$f_1\$ (**oscillatory** — smooth and periodic) and \$f_3\$ (**corner peak** — a smooth ridge that steepens toward a corner) on \$[0,1]^2\$, and builds train/test samples for the corner-peak target. The Python ground truth instead showcases \$f_1\$, \$f_4\$ (Gaussian), and \$f_5\$ (continuous, with kinks); the smooth-vs.-peaked contrast survives here (both \$f_1\$ and \$f_3\$ are smooth), but the non-smooth (kink) case — Python's \$f_5\$ — is deferred to the Python ground truth rather than reproduced in this preview.
"""

# ╔═╡ 44444444-0207-4444-8444-444444444444
begin
    a = [1.5, 0.75]
    u = [0.2, 0.8]
    genz_oscillatory(x) = cos.(2pi * u[1] .+ a' * x)
    genz_corner_peak(x) = (1 .+ a' * x) .^ -3

    x_train = rand(rng, 2, hp.n_train)
    y_train = genz_corner_peak(x_train)
    x_test = rand(rng, 2, hp.n_test)
    y_test = genz_corner_peak(x_test)
end

# ╔═╡ a1579626-f6eb-16bd-99e6-ebe287ef2806
md"""
---
### 3. Build and train a neural network (MSE loss)

We approximate the corner-peak target with a Lux MLP: input dimension 2, two hidden layers of 32 units, `tanh` activation, trained with `Optimisers.Adam` under MSE loss. (The Python ground truth uses a deeper 3×64 ReLU network trained for 200 epochs; this smoke-sized preview keeps the network small so it trains in seconds.)
"""

# ╔═╡ d917f5e5-d43c-20c0-b4cb-70f15f8a3066
md"""
---
### 5. Beyond MSE: robust and asymmetric losses

MSE is the default regression loss, but it has weaknesses:
- **Sensitive to outliers:** a single large residual dominates the gradient.
- **Symmetric:** over- and under-predictions are penalised equally.

In economics and finance we often care about **tail risk** — getting the worst-case predictions right matters more than small errors in the bulk.

| Loss | Formula | When it helps |
|:-----|:--------|:--------------|
| **MSE** | \$\frac{1}{n}\sum(y - \hat{y})^2\$ | Baseline; optimal under Gaussian noise |
| **MAE** (L1) | \$\frac{1}{n}\sum\lvert y - \hat{y}\rvert\$ | Robust to outliers; median regression |
| **Huber(\$\delta\$)** | Quadratic for \$\lvert e\rvert<\delta\$, linear otherwise | Best of both: smooth + robust |
| **Quantile(\$\tau\$)** | \$\rho_\tau(e) = e\,(\tau - \mathbf{1}_{e<0})\$ | Asymmetric risk; penalises under-prediction at quantile \$\tau\$ |

The **quantile (pinball)** loss is the bridge to risk management: training at \$\tau = 0.05\$ produces a Value-at-Risk (VaR) estimator directly, since it targets the conditional quantile exceeded only \$\alpha\%\$ of the time.

The same code cell that trains the MSE model also evaluates the shared `LOSS_KERNELS` gallery — MSE, MAE, Huber, quantile, CVaR, and log-cosh — on one common residual vector (`kernel_baseline`), so the loss families can be compared on identical errors. These are the same kernels reused later for stochastic economic residuals.
"""

# ╔═╡ 55555555-0207-4555-8555-555555555555
begin
    model = make_mlp(2, (32, 32), 1; activation = NNlib.tanh)
    state = setup_training(rng_from_seed(SEED; offset = 1), model, Optimisers.Adam(0.006); parameter_type = Float64)
    loss_fn(model, ps, st, batch) = begin
        prediction, st_new = model(batch.x, ps, st)
        return mse_loss(prediction, batch.y), st_new
    end

    batch = (x = x_train, y = y_train)
    initial_prediction, _ = state.model(x_test, state.ps, state.st)
    initial_residual = initial_prediction .- y_test
    kernel_baseline = [(kernel = k, value = loss_kernel_value(k, initial_residual; delta = 0.1, quantile = 0.9, alpha = 0.9)) for k in LOSS_KERNELS]

    initial_loss = loss_value(state, loss_fn, batch)
    history = NamedTuple[]
    for _ in 1:hp.steps
        metrics = train_step!(state, loss_fn, batch; max_grad_norm = 10.0)
        append_metric!(history; step = metrics.step, loss = metrics.loss)
    end
    test_loss = loss_value(state, loss_fn, (x = x_test, y = y_test))
end

# ╔═╡ 2065eb0d-f4aa-5928-d046-39dc5791e6e5
md"""
---
### 2. Visualising the approximation

We plot the trained Lux MLP against the true corner-peak function along a slice through the unit square (\$x_2 = 0.5\$).

*(The section numbers follow the outline's Python ordering. In this preview the visualisation is placed after §3/§5 because it reuses the network trained there, so the notebook's cell order runs 1, 3, 5, 2, 9 rather than strictly ascending.)*
"""

# ╔═╡ 66666666-0207-4666-8666-666666666666
begin
    grid = reduce(hcat, ([x1, 0.5] for x1 in range(0, 1; length = 120)))
    fit, _ = state.model(grid, state.ps, state.st)
    fig = Figure(size = figure_size(RUN_MODE))
    ax = Axis(fig[1, 1], xlabel = "x1 at x2 = 0.5", ylabel = "Genz corner peak")
    lines!(ax, vec(grid[1:1, :]), vec(genz_corner_peak(grid)); label = "truth", color = :black, linewidth = 3)
    lines!(ax, vec(grid[1:1, :]), vec(fit); label = "Lux MLP", color = :dodgerblue3, linewidth = 3)
    axislegend(ax; position = :rt)
    fig
end

# ╔═╡ 8fbf82e7-7ff8-ce15-c86d-20a07a0c6072
md"""
---
### The full Python notebook also covers

- **§4 Convergence with training-set size** — approximation error falls as the number of training points grows.
- **§6 Comparing losses on contaminated data** — the MSE, MAE, Huber, and quantile models are retrained on outlier-contaminated data and their error heatmaps compared, making the robustness differences visible.
- **§7 Quantile loss** — training at \$\tau = 0.1, 0.5, 0.9\$ shifts predictions to target different conditional quantiles.
- **§8 Scaling to \$d=5\$** — the curse of dimensionality: for a fixed training budget the error grows sharply with input dimension, motivating structured models (PINNs, deep equilibrium nets) that exploit problem structure.
"""

# ╔═╡ 024ed589-36e9-8485-766f-5b91e0d42428
md"""
---
### 9. Discussion and Takeaways

1. **Neural networks are effective function approximators** — even a modest MLP can capture oscillations, peaks, and kinks in the Genz test functions.
2. **MSE is not always the best loss.** With outliers, MSE is distorted by quadratic penalisation of large residuals; **MAE** and **Huber** are more robust, and Huber combines smoothness (easier optimisation) with robustness (linear tails).
3. **Quantile loss enables asymmetric risk management** — choosing \$\tau < 0.5\$ or \$\tau > 0.5\$ targets a specific conditional quantile, directly useful for **Value-at-Risk (VaR)** estimation.
4. **The curse of dimensionality is real** — approximation error grows with \$d\$ for a fixed number of training points, motivating structured models that exploit problem structure.

#### References

- A. Genz (1987). *Testing multidimensional integration routines.* In: Tools, Methods and Languages for Scientific and Engineering Computation, Elsevier, pp. 81–94.
- P. Huber (1964). *Robust estimation of a location parameter.* Annals of Mathematical Statistics 35(1), 73–101.
- R. Koenker & G. Bassett (1978). *Regression quantiles.* Econometrica 46(1), 33–50.

The cell below returns a machine-checkable summary of this notebook's run, including the loss-kernel gallery values on the shared residual.
"""

# ╔═╡ 77777777-0207-4777-8777-777777777777
(
    oscillatory_example_mean = mean(genz_oscillatory(x_test)),
    initial_loss = initial_loss,
    final_train_loss = history[end].loss,
    test_loss = test_loss,
    kernel_baseline = kernel_baseline,
    finite_kernel_values = all(row -> isfinite(row.value), kernel_baseline),
)

# ╔═╡ Cell order:
# ╟─11111111-0207-4111-8111-111111111111
# ╟─4511dab1-df93-0437-8430-867543fec1ca
# ╟─0074b36f-0b50-8f9f-ac4a-8cca2e2198c1
# ╠═22222222-0207-4222-8222-222222222222
# ╠═33333333-0207-4333-8333-333333333333
# ╟─ec224516-9e24-eb39-894d-735dbd90baec
# ╠═44444444-0207-4444-8444-444444444444
# ╟─a1579626-f6eb-16bd-99e6-ebe287ef2806
# ╟─d917f5e5-d43c-20c0-b4cb-70f15f8a3066
# ╠═55555555-0207-4555-8555-555555555555
# ╟─2065eb0d-f4aa-5928-d046-39dc5791e6e5
# ╠═66666666-0207-4666-8666-666666666666
# ╟─8fbf82e7-7ff8-ce15-c86d-20a07a0c6072
# ╟─024ed589-36e9-8485-766f-5b91e0d42428
# ╠═77777777-0207-4777-8777-777777777777
