### A Pluto.jl notebook ###
# v0.20.0

using Markdown
using InteractiveUtils

# ╔═╡ 11111111-0202-4111-8111-111111111111
md"""
# Lecture 02, Notebook 02: Gradient Descent and SGD

This notebook translates the optimization mechanics into small Julia functions.
It keeps the cricket chirp data from the Python notebook and uses explicit
linear-model gradients before returning to neural-network training later.
"""

# ╔═╡ 24c14f94-3513-cbb5-0484-f1ddb6b9ef6c
md"""
## Lecture 02, Notebook 02: Gradient Descent and Stochastic Gradient Descent

**Course:** Deep Learning for Solving and Estimating Dynamic Models in Economics and Finance
**Script reference:** §1.5 (Optimization: gradient descent, SGD, mini-batch, learning rate)
**Notebook role:** core
**Author:** Simon Scheidegger

*Julia/Lux/Pluto preview of* `lectures/lecture_02_intro_deep_learning/code/lecture_02_02_GradientDescent_and_StochasticGradientDescent.ipynb`.
"""

# ╔═╡ ed05871d-6ff0-888c-1976-f86e0d2503f5
md"""
## Gradient and Stochastic Gradient Descent

This notebook is adjusted from https://nbviewer.jupyter.org/github/dtnewman/gradient_descent/blob/master/stochastic_gradient_descent.ipynb

The Julia preview keeps the optimization mechanics as small explicit functions: a one-dimensional gradient-descent walk on an analytical objective, a batch gradient-descent fit on the cricket-chirp data, and a stochastic gradient-descent (SGD) fit on the same data — all compared against the closed-form least-squares solution.
"""

# ╔═╡ 22222222-0202-4222-8222-222222222222
begin
    import Pkg
    Pkg.activate(joinpath(@__DIR__, "..", "..", "..", "julia"))

    using CSV
    using CairoMakie
    using DLEFJulia
    using DataFrames
    using LinearAlgebra
    using Random: randperm
    using Statistics
end

# ╔═╡ 33333333-0202-4333-8333-333333333333
begin
    RUN_MODE = "smoke"
    SEED = 0
    rng = rng_from_seed(SEED)
end

# ╔═╡ 3efe68cc-ae96-b8bd-4ba3-fcf8c732e632
md"""
---
### Gradient Descent

**Gradient descent** (also called **steepest descent**) finds a local minimum of a function by repeatedly stepping in the direction of the negative gradient:

1. Choose an initial guess \$x_0\$.
2. For \$k = 0, 1, 2, \dots\$: set the step direction \$s_k = -\nabla f(x_k)\$, choose a step size \$\alpha_k\$, and update \$x_{k+1} = x_k + \alpha_k s_k\$.

(**Gradient ascent** is identical but steps along \$+\nabla f\$ to find maxima instead.)

As an analytical example we minimise \$f(x) = x^3 - 2x^2 + 2\$, whose local minimum sits near \$x \approx 1.4\$. The Julia cell below defines `f`, its gradient `grad_f`, and a fixed-step `gradient_descent_1d` walk, reporting the start, finish, and objective value.

> **The full Python notebook also covers** two refinements to the constant step size (learning rate) used here. First, choosing \$\alpha_k\$ adaptively at each step by line search (via SciPy's `fmin`), which reaches the minimum in about 4 iterations instead of 17. Second, a decreasing schedule \$\eta(t+1) = \eta(t) / (1 + t\,d)\$ with a decay constant \$d\$. The Julia preview keeps the constant-step walk for simplicity.
"""

# ╔═╡ 44444444-0202-4444-8444-444444444444
begin
    f(x) = x^3 - 2x^2 + 2
    grad_f(x) = 3x^2 - 4x

    function gradient_descent_1d(x0; eta = 0.05, steps = 40)
        xs = Vector{Float64}(undef, steps + 1)
        xs[1] = x0
        for t in 1:steps
            xs[t + 1] = xs[t] - eta * grad_f(xs[t])
        end
        return xs
    end

    path = gradient_descent_1d(2.0)
    (start = path[1], finish = path[end], objective = f(path[end]))
end

# ╔═╡ d72511d2-ddb5-1732-fd7a-eaf7f84fda62
md"""
---
### A more complicated example: cricket chirp rate vs temperature

Consider a simple linear regression: how does temperature affect the chirp rate of crickets? `SGD_data.txt` is a two-column CSV with no header (15 data rows, plus a trailing blank line for 16 lines in total) — column 1 is chirps per second (striped ground cricket, *Gryllus rubens*), column 2 is ambient temperature in degrees Fahrenheit. This is a standard introductory dataset (originally from G. W. Pierce, *The Songs of Insects*, Harvard University Press, 1948).

Our goal is to fit the straight line \$h_\theta(x) = \theta_0 + \theta_1 x\$ by minimising the mean-squared cost

\$\$J(\theta_0,\theta_1) = \frac{1}{m}\sum_{i=1}^m (h_\theta(x_i)-y_i)^2,\$\$

whose two-dimensional gradient is

\$\$\frac{\partial J}{\partial \theta_0} = \frac{1}{m}\sum_{i=1}^m (h_\theta(x_i)-y_i), \qquad \frac{\partial J}{\partial \theta_1} = \frac{1}{m}\sum_{i=1}^m (h_\theta(x_i)-y_i)\,x_i.\$\$

The Julia cell below loads the data into `x` and `y` and defines `predict`, `linear_cost`, and `linear_grad`. (`linear_cost` uses the \$\tfrac{1}{2}\$-scaled convention, which is why the factor of 2 drops out of the gradient.)
"""

# ╔═╡ 55555555-0202-4555-8555-555555555555
begin
    data_path = joinpath(@__DIR__, "..", "code", "SGD_data.txt")
    cricket = DataFrame(CSV.File(data_path; header = [:chirps, :temperature]))
    x = Float64.(cricket.chirps)
    y = Float64.(cricket.temperature)
    m = length(y)

    predict(theta, x) = theta[1] .+ theta[2] .* x
    linear_cost(theta, x, y) = mean(abs2, predict(theta, x) .- y) / 2
    linear_grad(theta, x, y) = [
        mean(predict(theta, x) .- y),
        mean((predict(theta, x) .- y) .* x),
    ]
end

# ╔═╡ ce39c8a8-671f-ceec-ba35-fe499e2a0927
md"""
Now we run **batch gradient descent** — recomputing the full-data gradient at every step — and compare it against the closed-form least-squares solution `hcat(ones(m), x) \ y` (Julia's analog of SciPy's `linregress`). The two should land close to each other, though gradient descent is much slower to get there.

> **In this preview** batch gradient descent is capped at a fixed step budget (`max_steps`), and the loose tolerance is never met within it, so the run stops well short of convergence: it settles on a slower near-centroid fit (intercept ≈ 5, loss ≈ 8) rather than the closed-form parameters (intercept ≈ 25, loss ≈ 6). Reaching the closed-form fit by plain gradient descent here would take on the order of hundreds of thousands of steps; the cell illustrates the *mechanics*, not a converged fit.
"""

# ╔═╡ 66666666-0202-4666-8666-666666666666
begin
    function batch_gradient_descent(theta0, x, y; eta = 0.001, tolerance = 1e-3, max_steps = 20_000)
        theta = copy(theta0)
        history = NamedTuple[]
        for step in 1:max_steps
            grad = linear_grad(theta, x, y)
            theta .-= eta .* grad
            step % 100 == 0 && push!(history, (step = step, loss = linear_cost(theta, x, y)))
            norm(grad) < tolerance && return theta, history
        end
        return theta, history
    end

    theta_batch, batch_history = batch_gradient_descent([1.0, 1.0], x, y)
    theta_closed_form = hcat(ones(m), x) \ y
    (
        batch_gradient_descent = theta_batch,
        closed_form = theta_closed_form,
        loss = linear_cost(theta_batch, x, y),
    )
end

# ╔═╡ 0b60101b-b79f-4dbd-dbd1-6e043cecfd4e
md"""
### Stochastic gradient descent

Batch gradient descent recomputes the gradient over the *entire* dataset at every step. With only 15 cricket observations that is cheap, but for very large datasets it is wasteful. **Stochastic gradient descent (SGD)** instead updates the parameters after looking at *each individual* example, so it makes progress right away:

\$\$\theta_0 \leftarrow \theta_0 - \alpha\,(h_\theta(x_i)-y_i), \qquad \theta_1 \leftarrow \theta_1 - \alpha\,x_i\,(h_\theta(x_i)-y_i).\$\$

Typically one shuffles the data and runs several passes (epochs) over it. Unlike batch descent, SGD tends to oscillate *near* the minimum rather than settling exactly on it; a decreasing step size can damp this, but a fixed \$\alpha\$ is more common. The Julia cell below first standardizes the chirp counts (so the intercept and slope directions share a scale, which is what lets a single learning rate converge), reshuffles with `randperm(rng, ...)` each pass, applies the per-example update, and finally maps the fitted coefficients back to the original chirp scale so they line up with the batch-GD and closed-form fits.

> **The full Python notebook also covers** an SGD demo on 500,000 synthetic points drawn around \$y = 4x + 10 + \varepsilon\$, tracking the running cost every 10,000 steps to show it fall quickly and then level off. The Julia preview keeps SGD on the small cricket dataset so it can be read directly against the batch-GD and closed-form fits.
"""

# ╔═╡ 77777777-0202-4777-8777-777777777777
begin
    function stochastic_gradient_descent(theta0, x, y; eta = 0.05, passes = 100, rng)
        # Standardize the chirp counts before descending. The raw values sit
        # near 15-20, so the slope direction has curvature ~x^2 while the
        # intercept direction has curvature ~1; a single learning rate then
        # leaves the intercept crawling. On z = (x - mu) / sigma both
        # directions share a scale, so SGD reaches the least-squares fit.
        mu = mean(x)
        sigma = std(x)
        z = (x .- mu) ./ sigma
        to_original(beta) = [beta[1] - beta[2] * mu / sigma, beta[2] / sigma]

        theta = copy(theta0)
        history = NamedTuple[]
        for pass in 1:passes
            for i in randperm(rng, length(y))
                err = theta[1] + theta[2] * z[i] - y[i]
                theta .-= eta .* [err, err * z[i]]
            end
            push!(history, (pass = pass, loss = linear_cost(to_original(theta), x, y)))
        end
        return to_original(theta), history
    end

    theta_sgd, sgd_history = stochastic_gradient_descent([1.0, 1.0], x, y; rng)
end

# ╔═╡ 88888888-0202-4888-8888-888888888888
begin
    xx = collect(range(minimum(x), maximum(x); length = 100))
    fig = Figure(size = figure_size(RUN_MODE))
    ax = Axis(fig[1, 1], xlabel = "chirps/sec", ylabel = "temperature")
    scatter!(ax, x, y; color = (:gray30, 0.7), label = "data")
    lines!(ax, xx, predict(theta_closed_form, xx); color = :black, linewidth = 3, label = "closed form")
    lines!(ax, xx, predict(theta_batch, xx); color = :dodgerblue3, linewidth = 2, label = "batch GD")
    lines!(ax, xx, predict(theta_sgd, xx); color = :darkorange, linewidth = 2, label = "SGD")
    axislegend(ax; position = :lt)
    fig
end

# ╔═╡ 08532a14-cd85-b36d-5db7-f515854af791
md"""
### Takeaway

- **Gradient descent** steps downhill along \$-\nabla f\$; the constant step size (learning rate) trades convergence speed against stability, and adaptive or decaying schedules can help.
- On the cricket-chirp regression, **stochastic gradient descent** (on standardized inputs) recovers parameters close to the **closed-form** least-squares solution via cheap per-example updates. **Batch gradient descent** — recomputing the full-data gradient at every step — heads toward the same fit but is much slower, and within this preview's fixed step budget it stops short of the closed-form parameters.

The cell below returns a machine-checkable summary comparing the batch-GD, SGD, and closed-form losses for this notebook's run.
"""

# ╔═╡ 99999999-0202-4999-8999-999999999999
(
    batch_loss = linear_cost(theta_batch, x, y),
    sgd_loss = linear_cost(theta_sgd, x, y),
    closed_form_loss = linear_cost(theta_closed_form, x, y),
    batch_steps_recorded = length(batch_history),
    sgd_passes = length(sgd_history),
)

# ╔═╡ Cell order:
# ╟─11111111-0202-4111-8111-111111111111
# ╟─24c14f94-3513-cbb5-0484-f1ddb6b9ef6c
# ╟─ed05871d-6ff0-888c-1976-f86e0d2503f5
# ╠═22222222-0202-4222-8222-222222222222
# ╠═33333333-0202-4333-8333-333333333333
# ╟─3efe68cc-ae96-b8bd-4ba3-fcf8c732e632
# ╠═44444444-0202-4444-8444-444444444444
# ╟─d72511d2-ddb5-1732-fd7a-eaf7f84fda62
# ╠═55555555-0202-4555-8555-555555555555
# ╟─ce39c8a8-671f-ceec-ba35-fe499e2a0927
# ╠═66666666-0202-4666-8666-666666666666
# ╟─0b60101b-b79f-4dbd-dbd1-6e043cecfd4e
# ╠═77777777-0202-4777-8777-777777777777
# ╠═88888888-0202-4888-8888-888888888888
# ╟─08532a14-cd85-b36d-5db7-f515854af791
# ╠═99999999-0202-4999-8999-999999999999
