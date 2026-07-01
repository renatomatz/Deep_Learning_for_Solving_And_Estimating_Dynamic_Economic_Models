### A Pluto.jl notebook ###
# v0.20.0

using Markdown
using InteractiveUtils

# ╔═╡ 11111111-0301-4111-8111-111111111111
md"""
# Lecture 03, Notebook 01: Deterministic Brock-Mirman DEQN in Lux

The network predicts a savings rate. The economic residual stays separate from
the optimizer loop so the Lux call pattern remains explicit.
"""

# ╔═╡ d3ebdcdf-d94d-6a3e-83d7-8c28b18c39a9
md"""
## Lecture 03, Notebook 01: Deterministic Brock–Mirman DEQN

**Course:** Deep Learning for Solving and Estimating Dynamic Models in Economics and Finance
**Script reference:** §2.4 (deterministic Brock–Mirman benchmark), §2.5 (hard/soft constraint split)
**Notebook role:** core
**Author:** Simon Scheidegger

*Julia/Lux/Pluto preview of* `lectures/lecture_03_deep_equilibrium_nets/code/lecture_03_01_Brock_Mirman_1972_DEQN.ipynb`.
"""

# ╔═╡ 3fac62e4-5722-ae97-f245-99bc8c708f94
md"""
## Simple Introduction to Deep Equilibrium Nets

### Notebook 1: no uncertainty and exogenous sampling of states

#### Purpose of the notebook and economic model

This notebook is a simple introduction to **Deep Equilibrium Nets** (DEQNs), the deep-learning method of [Azinovic et al. (2022)](https://onlinelibrary.wiley.com/doi/full/10.1111/iere.12575). To focus on the method we solve a simple one-agent optimal growth model, a simplified [Brock and Mirman (1972)](https://assoeconomiepolitique.org/wp-content/uploads/Brock-et-Mirman-1972.pdf).

The planner maximizes time-separable lifetime utility subject to the budget constraint:

\$\$
\begin{aligned}
&\max_{\{C_t\}_{t=0}^{\infty}} \sum_{t=0}^{\infty} \beta^{t}\, \ln(C_t)\\
\text{s.t.}\quad & K_{t+1} + C_t = Y_t + (1-\delta)K_t,
\end{aligned}
\$\$

where \$Y_t = K_t^{\alpha}\$. Under full depreciation (\$\delta = 1\$) this problem has a closed-form solution:

\$\$K_{t+1} = \beta\alpha K_t^{\alpha}.\$\$

Instead of value-function iteration or time iteration, we approximate the recursive-equilibrium policy directly with a neural network, following Azinovic et al. (2022).
"""

# ╔═╡ a295a0d3-905c-f741-9934-bb30d305af6e
md"""
The DEQN idea is to write the equilibrium conditions as a set of equations that must hold in equilibrium and characterize the optimal policy. Given a candidate policy, the extent to which those equations are violated measures its accuracy.

The recursive form has Bellman equation

\$\$V(K_t) = \max_{K_{t+1}} \ln(C_t) + \beta V(K_{t+1}), \qquad C_t = Y_t + (1-\delta)K_t - K_{t+1}.\$\$

\$K_t\$ is the **state** and \$K_{t+1} = f(K_t)\$ is the **policy**; we approximate \$f(\cdot)\$ with a network \$\mathcal{N}(\cdot)\$ so that \$\mathcal{N}(K_t) \approx K_{t+1}\$. The first-order condition together with the [envelope theorem](https://en.wikipedia.org/wiki/Envelope_theorem) give the Euler equation

\$\$0 = -\frac{1}{C_t} + \beta\frac{1}{C_{t+1}}\left(1 - \delta + r_{t+1}\right), \qquad r_{t+1} = \alpha K_{t+1}^{\alpha-1}.\$\$

To interpret residuals as **relative consumption errors** we rearrange into

\$\$0 = \frac{C_{t+1}}{C_t\,\beta\left(1 - \delta + r_{t+1}\right)} - 1,\$\$

which we encode as the **loss** and drive to zero over the training states. Two remarks:

1. The residual is only well defined when \$C_t > 0\$ and \$K_{t+1} > 0\$. We therefore have the network output a **savings rate** \$s_t\$ with \$K_{t+1} = (1-\delta)K_t + Y_t\, s_t\$ and squash \$s_t \in (0,1)\$ with a sigmoid, guaranteeing feasibility at every step of training (a **hard** constraint; see below).
2. We must choose the states on which the residual should hold. Here we sample \$K_t\$ exogenously from an interval \$\mathcal{K} = [\underline{K}, \overline{K}]\$ known to be forward-invariant. In high dimensions one samples instead from the simulated ergodic distribution (see [Maliar et al. (2011)](https://www.sciencedirect.com/science/article/pii/S0165188910002186)); the next notebook does exactly that.
"""

# ╔═╡ 647ba83c-30ac-724d-a150-77f98fc14b9b
md"""
### Implementing the loss function

The Julia preview builds on the shared `DLEFJulia` helpers. Where the Python notebook imports NumPy and TensorFlow/Keras, we load `Lux` (explicit `model(x, ps, st)` networks), `Optimisers` (Adam), `NNlib` (activations), and `CairoMakie` (plots).
"""

# ╔═╡ 22222222-0301-4222-8222-222222222222
begin
    import Pkg
    Pkg.activate(joinpath(@__DIR__, "..", "..", "..", "julia"))

    using CairoMakie
    using DLEFJulia
    using Lux
    using NNlib
    using Optimisers
end

# ╔═╡ 27c441ad-2c9e-f991-97e8-8233e9df64a3
md"""
`RUN_MODE` selects a training budget: `smoke` for a fast sanity check, `teaching` and `production` for the longer runs behind the slide figures. `SEED = 0` fixes the RNG. As required, this preview keeps `RUN_MODE = "smoke"` and `SEED = 0`.
"""

# ╔═╡ 33333333-0301-4333-8333-333333333333
begin
    RUN_MODE = "smoke"
    SEED = 0
    budgets = (
        smoke = (steps = 40, batch_size = 64),
        teaching = (steps = 600, batch_size = 128),
        production = (steps = 4_000, batch_size = 256),
    )
    hp = run_mode_budget(RUN_MODE; budgets)
    rng = rng_from_seed(SEED)
end

# ╔═╡ 98fe6cc0-9fac-b0ad-5463-0334f4133cdf
md"""
### Economic parameters and the closed-form check

We fix the constant parameters \$\alpha = 0.36\$ (capital share), \$\beta = 0.99\$ (discount factor), and \$\delta = 1.0\$ (full depreciation). Because \$\delta = 1\$, the model admits the analytical policy \$K_{t+1} = \beta\alpha K_t^{\alpha}\$ and steady state \$K^\star\$, which we later use to check the network's solution. Here `BrockMirmanParams`, `bm_steady_state`, and `bm_full_depreciation_policy` package these.
"""

# ╔═╡ 44444444-0301-4444-8444-444444444444
begin
    params = BrockMirmanParams(alpha = 0.36, beta = 0.99, delta = 1.0)
    k_star = bm_steady_state(params)
    analytic_policy(k) = bm_full_depreciation_policy(k, params)
end

# ╔═╡ fc88e5e1-538d-ba0b-1a1f-27f84f22f6a1
md"""
#### Deep neural network

The network approximates the **savings rate** \$s_t\$, so that \$K_{t+1} = Y_t\, s_t \approx Y_t\, \mathcal{N}(K_t)\$. The input is the 1-dimensional state \$K_t\$ and the output is the 1-dimensional savings rate. The Python notebook uses two hidden ReLU layers with a sigmoid output head; this Lux preview uses `make_mlp(1, (24, 24), 1; activation = tanh)` and applies the sigmoid separately through `savings_transform`, so the savings share still lands in \$(0,1)\$.

**The batch dimension.** Networks are evaluated on many states at once. Lux uses **feature-by-batch** arrays: a batch of \$N\$ capital levels is a \$1 \times N\$ matrix, and the network returns a \$1 \times N\$ matrix of savings rates. (This is the transpose of the Python/Keras samples-on-rows convention.)
"""

# ╔═╡ 521a9d12-e316-b8fb-84fd-cf4f246e4ca8
md"""
##### Hard vs. soft constraints — the central design choice in DEQNs

Two kinds of equilibrium conditions appear in any dynamic stochastic model:

- **Inequality / feasibility constraints** — e.g. \$C_t > 0\$, \$K_{t+1} > 0\$, the resource constraint \$C_t + K_{t+1} = Y_t\$. These must hold *exactly*.
- **Optimality conditions** — e.g. the Euler equation. These hold in equilibrium but not at every intermediate guess of the policy.

Azinovic, Gaegauf & Scheidegger (2022, §4.2.2; lecture script Fig. 2.3) treat the two very differently:

| | Hard constraint (architecture) | Soft constraint (loss) |
|--|--|--|
| **What** | Built into the network output | Penalised in the cost function |
| **How** | Activation choice + algebraic identities | Squared residuals in \$\mathcal{L}\$ |
| **Cost** | Always satisfied — even at random init | Only satisfied at convergence |

**Why it matters here.** The next cell parameterises the savings *share* \$s_t \in (0,1)\$ via a **sigmoid** (`savings_transform`). With \$K_{t+1} = s_t Y_t\$ and \$C_t = (1-s_t) Y_t\$, this **guarantees \$C_t > 0\$ and \$K_{t+1} > 0\$ simultaneously**, at every training iteration — we never penalise infeasibility, the architecture rules it out. The Euler equation, by contrast, is enforced softly through the loss. This split removes a class of bad local minima and is one reason DEQNs converge where naive penalty methods do not.
"""

# ╔═╡ d33f128f-4995-aff1-c739-ce4a7566d48b
md"""
#### Cost function, gradients, optimizer, sampling, and training

This single Lux cell does the work spread across several Python cells:

- **Cost function.** `deterministic_bm_residual` evaluates the relative-consumption residual \$\frac{C_{t+1}}{C_t\,\beta(1-\delta+r_{t+1})} - 1\$ on a batch of states and returns its mean-squared value as the loss. It stays *outside* the optimizer as a pure function threading `ps`/`st`.
- **Gradients.** `train_step!` differentiates the loss with respect to the network parameters using `Zygote` (the reverse-mode analogue of TensorFlow's `GradientTape`), with gradient-norm clipping at 10.
- **Optimizer.** `Optimisers.Adam(0.01)` updates the parameters — an improved SGD, matching the Python Adam optimizer.
- **Sampling.** `sample_k` draws capital uniformly from the exogenous interval \$[0.1, 1.0]\$, fresh each step.
- **Training.** We iterate: sample a batch, take an Adam step, and record the loss. `RUN_MODE` sets the number of steps and the batch size.
"""

# ╔═╡ 55555555-0301-4555-8555-555555555555
begin
    model = make_mlp(1, (24, 24), 1; activation = NNlib.tanh)
    train_state = setup_training(rng_from_seed(SEED; offset = 1), model, Optimisers.Adam(0.01); parameter_type = Float64)
    savings_transform = NNlib.sigmoid

    sample_k(rng, n) = reshape(0.1 .+ 0.9 .* rand(rng, n), 1, :)
    bm_loss(model, ps, st, k_batch) = begin
        pieces, st_new = deterministic_bm_residual(model, ps, st, k_batch; params, transform = savings_transform)
        return pieces.loss, st_new
    end

    initial_batch = sample_k(rng, hp.batch_size)
    initial_loss = loss_value(train_state, bm_loss, initial_batch)
    history = NamedTuple[]
    for _ in 1:hp.steps
        local batch = sample_k(rng, hp.batch_size)
        metrics = train_step!(train_state, bm_loss, batch; max_grad_norm = 10.0)
        append_metric!(history; step = metrics.step, loss = metrics.loss)
    end
end

# ╔═╡ 898415b0-ff14-be9e-e0fd-95a2b6e3f922
md"""
### Comparison to the closed-form policy

After training we evaluate the learned policy on a fixed capital grid, measure its relative \$L^2\$ distance from the analytical full-depreciation policy \$K_{t+1} = \beta\alpha K_t^{\alpha}\$, and summarise the residual. The figure below overlays the DEQN policy, the analytic policy, and the 45-degree line.
"""

# ╔═╡ 66666666-0301-4666-8666-666666666666
begin
    k_grid = reshape(collect(range(0.1, 1.0; length = 100)), 1, :)
    diagnostics, _ = deterministic_bm_residual(train_state.model, train_state.ps, train_state.st, k_grid; params, transform = savings_transform)
    policy_error = relative_l2_error(diagnostics.next_capital, analytic_policy(k_grid))
    residual_stats = residual_summary(diagnostics.residual)
end

# ╔═╡ 77777777-0301-4777-8777-777777777777
begin
    fig = Figure(size = figure_size(RUN_MODE))
    ax = Axis(fig[1, 1], xlabel = "K_t", ylabel = "K_{t+1}")
    lines!(ax, vec(k_grid), vec(k_grid); color = :gray55, linestyle = :dash, label = "45 degree")
    lines!(ax, vec(k_grid), vec(analytic_policy(k_grid)); color = :black, linewidth = 3, label = "analytic")
    lines!(ax, vec(k_grid), vec(diagnostics.next_capital); color = :dodgerblue3, linewidth = 3, label = "DEQN")
    axislegend(ax; position = :lt)
    fig
end

# ╔═╡ 476c2cd1-45b9-5b7f-6250-757f0b493a3b
md"""
### Conclusion

This notebook introduced Deep Equilibrium Nets on the deterministic Brock–Mirman growth model: we parameterised the savings rate with a Lux network, encoded feasibility as a **hard** (sigmoid) constraint and the Euler equation as a **soft** (residual) loss, trained on exogenously sampled capital, and checked the result against the closed-form policy available under full depreciation. The next notebook adds productivity uncertainty and quadrature for the conditional expectation. The cell below returns a machine-checkable summary of this run.
"""

# ╔═╡ 88888888-0301-4888-8888-888888888888
(
    steady_state = k_star,
    initial_loss = initial_loss,
    final_loss = history[end].loss,
    policy_relative_l2 = policy_error,
    max_abs_residual = residual_stats.max_abs,
)

# ╔═╡ Cell order:
# ╟─11111111-0301-4111-8111-111111111111
# ╟─d3ebdcdf-d94d-6a3e-83d7-8c28b18c39a9
# ╟─3fac62e4-5722-ae97-f245-99bc8c708f94
# ╟─a295a0d3-905c-f741-9934-bb30d305af6e
# ╟─647ba83c-30ac-724d-a150-77f98fc14b9b
# ╠═22222222-0301-4222-8222-222222222222
# ╟─27c441ad-2c9e-f991-97e8-8233e9df64a3
# ╠═33333333-0301-4333-8333-333333333333
# ╟─98fe6cc0-9fac-b0ad-5463-0334f4133cdf
# ╠═44444444-0301-4444-8444-444444444444
# ╟─fc88e5e1-538d-ba0b-1a1f-27f84f22f6a1
# ╟─521a9d12-e316-b8fb-84fd-cf4f246e4ca8
# ╟─d33f128f-4995-aff1-c739-ce4a7566d48b
# ╠═55555555-0301-4555-8555-555555555555
# ╟─898415b0-ff14-be9e-e0fd-95a2b6e3f922
# ╠═66666666-0301-4666-8666-666666666666
# ╠═77777777-0301-4777-8777-777777777777
# ╟─476c2cd1-45b9-5b7f-6250-757f0b493a3b
# ╠═88888888-0301-4888-8888-888888888888
