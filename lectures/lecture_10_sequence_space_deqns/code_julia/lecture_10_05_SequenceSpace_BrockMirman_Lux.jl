### A Pluto.jl notebook ###
# v0.20.0

using Markdown
using InteractiveUtils

# ╔═╡ 11111111-1005-4111-8111-111111111111
md"""
# Lecture 10, Notebook 05: Sequence-Space Brock-Mirman in Lux

The policy reads a shock-history vector, not the current endogenous state. The
shared helper keeps histories as `(features_per_lag, history_length, batch)` and
flattens only at the Lux boundary.
"""

# ╔═╡ 9f0bd55d-0137-5b1a-4e4a-6b044cdaff9f
md"""
## Lecture 10: Sequence-space DEQNs

**Course:** Deep Learning for Solving and Estimating Dynamic Models in Economics and Finance
**Script reference:** §6.7 (Sequence-space DEQNs) — Brock-Mirman warm-up
**Notebook role:** core
**Author:** Simon Scheidegger

*Julia/Lux/Pluto preview of* `lectures/lecture_10_sequence_space_deqns/code/lecture_10_05_SequenceSpace_BrockMirman.ipynb`.
"""

# ╔═╡ 588b6cac-ebeb-1754-ce66-47740cea9993
md"""
## Sequence-Space DEQN: Brock–Mirman Warm-Up

> **Status: in-class walkthrough — Day 4, sequence-space slot.** This notebook is the simplest setting in which the sequence-space idea (network input = recent history of exogenous shocks instead of current endogenous aggregate state) can be made visible. The companion notebooks `06_SequenceSpace_KrusellSmith.ipynb` and `KrusellSmith_Tutorial_CPU.ipynb` apply the same template to the heterogeneous-agent benchmark.

### Learning goals

By the end of this notebook you will be able to:

1. **State the sequence-space formulation** of a recursive equilibrium and explain the ergodicity-and-truncation argument that makes the truncated history a valid input for the policy network.
2. **Recognise the single conceptual change** versus a state-space DEQN: the network's *input* changes (a shock-history window replaces the current endogenous aggregate state); the loss, the equilibrium conditions, and the gradient step are unchanged.
3. **Read the four ingredients** of the sequence-space cloud method as implemented here: a savings-rate network with sigmoid output, a Gauss–Hermite quadrature for the conditional expectation, a forward simulator that rolls the shock history one step at a time, and a per-episode SGD update on the squared Euler residual.
4. **Validate the trained policy** against the Brock–Mirman closed form \$K_{t+1} = \alpha\beta\, A_t K_t^{\alpha}\$ (available under \$\delta = 1\$, log utility), and read the diagnostics: loss curve, ergodic state cloud, and policy / Euler-error histograms.

### What this notebook is *not*

- It is **not** a heterogeneous-agent solver. There is one representative household and one aggregate state. Sequence space here is a *teaching device*, not a dimensionality reduction. The payoff appears in the HA notebooks.
- It does **not** use shape-preserving I-spline outputs. The sigmoid-savings-rate parameterisation is enough for the BM warm-up; I-splines enter in `06_…KrusellSmith`.
- The loss is the **direct residual minimisation** algorithm of Azinovic-Yang & Žemlička (2025). The alternative *time iteration with EGM* algorithm is not used here.

### References
- Azinovic-Yang, M., & Žemlička, J. (2025). *Deep learning in the sequence space.* arXiv:2509.13623.
- Brock, W. A., & Mirman, L. J. (1972). *Optimal economic growth and uncertainty: the discounted case.* Journal of Economic Theory 4(3), 479–513.
- Companion JAX repository (as linked by the Python notebook): [`azinoma/DeepLearningInTheSequenceSpace`](https://github.com/azinoma/DeepLearningInTheSequenceSpace).
"""

# ╔═╡ 9b777215-e09f-ce5c-72d6-4ac152ed0421
md"""
### 1. The Brock–Mirman model

A representative household maximises
\$\$\mathbb{E}_0 \sum_{t=0}^\infty \beta^t\, u(c_t), \qquad u(c) = \log c \;\;(\gamma = 1),\$\$
subject to the resource constraint
\$\$c_t + k_{t+1} \;=\; A_t\, k_t^{\alpha} + (1-\delta)\, k_t \;\equiv\; y_t,\$\$
with full depreciation \$\delta = 1\$ and an exogenous AR(1) for log productivity
\$\$\log A_{t+1} \;=\; z_{t+1}, \qquad z_{t+1} \;=\; \rho_z\, z_t + \sigma_z\, \varepsilon_{t+1}, \qquad \varepsilon_{t+1} \sim \mathcal{N}(0, 1).\$\$

#### Key analytic fact (used as a benchmark)
With \$\gamma = 1\$ and \$\delta = 1\$ the planner's policy is closed-form:
\$\$k_{t+1} \;=\; s\cdot y_t \quad\text{with savings rate}\quad s \;=\; \alpha\beta.\$\$
We use this as a ground truth in the diagnostic checks. The notebook also runs in regimes where the closed form does not apply — those skip the policy comparison.
"""

# ╔═╡ 1075955b-d191-abf2-75e1-557ad09292ba
md"""
### 2. State-space vs sequence-space — the same equilibrium, two domains

The classical *state-space* representation of the recursive equilibrium reads:

| | Equation | Domain |
|---|---|---|
| Recursive policy | \$y_t = f(x_t)\$ | state \$x_t = (k_t, z_t)\$ |
| Transition | \$x_{t+1} = H(x_t,\, y_t,\, \varepsilon_{t+1})\$ | known function |
| Functional equation | \$G(f, x) = 0 \quad \forall x\$ | enforced on the state space |

The *sequence-space* representation expresses the same equilibrium policy as a function of the **history of exogenous shocks** instead of the current endogenous state:

| | Equation | Domain |
|---|---|---|
| Sequence-space policy | \$y_t = \Psi\bigl(\varepsilon_t, \varepsilon_{t-1}, \ldots \mid x_0\bigr)\$ | shock history |
| Iterated law of motion | \$x_t = \mathcal{H}\bigl(\mathcal{E}_t, x_0 \mid \Psi\bigr)\$ | derived from \$\Psi\$ |
| Functional equation | \$G(\Psi, \mathcal{E}, x_0) = 0 \quad \forall \mathcal{E},\,\forall x_0\$ | enforced on the shock-history space |

> **Same equilibrium, different domain of approximation.** The two formulations describe the same set of equilibrium policies. They differ in *what the network takes as input*. State-space inputs may be high-dimensional once a distribution \$\mu_t\$ enters; sequence-space inputs live on a (potentially long but) **purely exogenous** domain.
"""

# ╔═╡ 6ffcfdbc-4026-4b48-c535-bcc1f8bbd4cc
md"""
### 3. Ergodicity and truncation — why a finite history is enough

For a large class of economies the equilibrium policy satisfies the *ergodicity* property
\$\$\lim_{\tau \to \infty}\, \frac{\partial \Psi}{\partial \varepsilon_{t-\tau}} \;=\; 0,\$\$
i.e. the influence of long-past shocks decays. Truncating the history after \$T\$ lags gives a **truncated sequence-space solution**
\$\$y_t \;\approx\; \widehat\Psi\bigl(\varepsilon_t, \varepsilon_{t-1}, \ldots, \varepsilon_{t-T+1}\bigr).\$\$

#### Brock–Mirman makes this concrete
Iterating the BM transition (with \$\delta = 1\$, log utility, \$s = \alpha\beta\$) gives
\$\$\log K_t \;=\; \mathrm{const} + \log A_{t-1} + \alpha\,\log A_{t-2} + \alpha^2\,\log A_{t-3} + \cdots,\$\$
so the lag-\$j\$ shock \$A_{t-j}\$ enters with weight \$\alpha^{j-1}\$. With \$\alpha = 1/3\$, the lag-25 weight is \$\alpha^{24} \approx 3.5 \times 10^{-12}\$ — truncating at \$T = 25\$ leaves negligible error. **This is the entire justification for using a length-25 shock-history window as the network input.**
"""

# ╔═╡ f2966166-ceb8-a200-74bf-f818e831c84a
md"""
### 4. Algorithm sketch — direct residual minimisation with a cloud method

The training loop has three pieces. We will see each one in code below.

1. **Initialisation.** Sample \$N_{\text{cloud}}\$ initial state-history pairs \$(x_j^{(0)}, \mathcal{E}_j^{(0)})\$. We initialise all of them at the deterministic steady state with empty histories.
2. **Per-episode training (mini-batch SGD).**
   - Draw a mini-batch of cloud entries.
   - Forward-pass the network \$\mathcal{N}_\theta(\mathcal{E})\$ to get a savings rate \$s\$, then \$c\$, \$\mu = u'(c)\$, and the implied \$k_{t+1}\$.
   - For each Gauss–Hermite quadrature node, build the *next-period* shock history and run the network *again* to get \$\mathbb{E}_t[u'(c_{t+1})\,\mathrm{MPK}_{t+1}]\$.
   - Compute the squared Euler residual; backprop and Adam step.
3. **Cloud refresh.** After every episode, draw fresh shocks for the entire cloud and roll one period forward (forward-simulator step, no gradients). The cloud now lives one period later — the *ergodic* distribution under the *current* policy emerges over many episodes.

The cloud-method sampling concentrates training data on the ergodic set: states the model actually visits under the policy being trained, rather than uniformly drawn states which may include economically irrelevant regions.
"""

# ╔═╡ 94717842-f1c2-e788-925f-8a25bb9bcf9b
md"""
### 5. Imports and setup

The Julia preview uses **Lux** for the network, **Optimisers.jl** for Adam, **Zygote** for parameter gradients, and the shared `DLEFJulia` helpers for the sequence-space plumbing — the canonical shock-history layout, the Gauss–Hermite quadrature, and the Brock–Mirman residual and forward step. All later cells assume these imports are in scope. (The Python ground truth uses TensorFlow 2 / Keras with a `USE_GPU` switch; here the model runs on CPU with explicit `model(x, ps, st)` state threading.)
"""

# ╔═╡ 22222222-1005-4222-8222-222222222222
begin
    import Pkg
    Pkg.activate(joinpath(@__DIR__, "..", "..", "..", "julia"))

    using DLEFJulia
    using Lux
    using NNlib
    using Optimisers
    using Random
end

# ╔═╡ 4ddfbc84-dfeb-b97e-af8e-ef80cfe9ea34
md"""
### 7. Algorithm hyperparameters and run mode

The Python notebook fixes its schedule directly:

| Symbol | Value | Role |
|---|---|---|
| `N_neur_sequence` | 64 | width of each hidden layer |
| `gh_order` | 8 | Gauss–Hermite quadrature nodes per (one-dim) shock |
| `n_cloud` | \$2^{12} = 4096\$ | size of the simulated cloud |
| `n_minib` | \$2^{8} = 256\$ | mini-batch size for SGD |
| `n_history` | 25 | length of the shock-history window |
| `lr_adam` | \$10^{-5}\$ | Adam learning rate |
| `cloud_steps` | \$256\times 8 = 2048\$ | total training episodes (fixed, not scaled) |

This Lux preview instead scales the schedule from `RUN_MODE`: the checked-in `smoke` budget uses 5 episodes, batch 16, and a length-8 shock history for fast CI; `teaching` and `production` recover the longer length-25 / length-80 histories. Set `RUN_MODE` below to move between budgets. (In the Python notebook `cloud_steps = 256 × 8` is a fixed constant, not derived from a run-mode switch.)

Two further preview simplifications are *not* `RUN_MODE`-scaled and differ from the tabulated Python values above: the conditional-expectation integral uses a 3-node Gauss–Hermite rule (`gauss_hermite_rule(3)`) rather than the tabulated `gh_order = 8`, and Adam runs at learning rate `0.002` rather than the tabulated `lr_adam = 1e-5`. Both are chosen to match the far shorter preview episode budget; at \$\sigma_z = 0.01\$ the 3-node quadrature integrates the (nearly linear) expectation to well below the loss floor, and the larger step size lets the 5-episode smoke run make visible progress.
"""

# ╔═╡ 33333333-1005-4333-8333-333333333333
begin
    RUN_MODE = "smoke"
    SEED = 0
    budgets = (
        smoke = (steps = 5, batch_size = 16, history_length = 8),
        teaching = (steps = 200, batch_size = 256, history_length = 25),
        production = (steps = 2_000, batch_size = 512, history_length = 80),
    )
    hp = run_mode_budget(RUN_MODE; budgets)
    rng = rng_from_seed(SEED)
end

# ╔═╡ 526b7856-3314-03d2-52cb-65793af247f2
md"""
### 6. Calibration, steady state, and quadrature

| Symbol | Value | Comment |
|---|---|---|
| \$\alpha\$ | \$1/3\$ | capital share |
| \$\beta\$  | \$0.97\$ | discount factor |
| \$\gamma\$ | \$1.0\$ | log utility (BM closed form available) |
| \$\delta\$ | \$1.0\$ | full depreciation (BM closed form available) |
| \$\rho_z\$ | \$0.98\$ | TFP persistence |
| \$\sigma_z\$ | \$0.01\$ | TFP innovation std |

The choice \$\gamma = \delta = 1\$ is what unlocks the closed-form benchmark \$s = \alpha\beta\$. Here `SequenceBrockMirmanParams()` carries the calibration and `gauss_hermite_rule` builds the quadrature (nodes scaled by \$\sqrt{2}\$, weights by \$1/\sqrt{\pi}\$) so that \$\mathbb{E}_{\varepsilon \sim \mathcal{N}(0,1)}[f(\varepsilon)] \approx \sum_q w_q\, f(\xi_q)\$.

**State and policy conventions.** The state vector is **5-wide**: alongside \$(k_t, z_t)\$ the helper precomputes \$(Z_t = e^{z_t},\, y_t,\, \mathrm{MPK}_t)\$ so the residual can read them off directly. The policy vector is **3-wide**: savings \$s\$, consumption \$c\$, marginal utility \$\mu = u'(c)\$.

**Deterministic steady state and initial cloud.** At \$z = 0\$ (so \$A = 1\$) the steady state solves \$\tfrac{1}{\beta} = \alpha k_{\text{ss}}^{\alpha-1} + (1-\delta)\$; with \$\delta = 1\$ this gives \$k_{\text{ss}} = (\alpha\beta)^{1/(1-\alpha)}\$. `sequence_bm_initial_state` starts every track of the cloud there with an all-zero shock history; the policy-driven simulator then disperses the cloud to its ergodic distribution.
"""

# ╔═╡ 946e2200-a680-5c3f-d403-af1c75353ef0
md"""
### 9–11. Economic primitives, `ExpandState`, and the sequence-space network

The `DLEFJulia` helpers carry the Brock–Mirman primitives: a positivity-safe marginal utility \$u'(c) = (1-\beta)\, c^{-\gamma}\$ and its inverse (used to turn \$\beta\,\mathbb{E}_t[u'(c_{t+1})\,\mathrm{MPK}_{t+1}]\$ into implied consumption), the analytic savings policy \$s_{\text{BM}} = \alpha\beta\, y\$ under \$\gamma = \delta = 1\$, and an `ExpandState`-style routine that precomputes the derived quantities \$(Z, y, \mathrm{MPK})\$ at every quadrature node.

The **sequence-space network** is intentionally minimal. Its input is the length-\$T\$ shock history \$\mathcal{E}_t = (\varepsilon_t, \varepsilon_{t-1}, \ldots, \varepsilon_{t-T+1})\$ — histories are stored as `(features_per_lag, history_length, batch)` and **flattened only at the Lux boundary** — and a scalar output is mapped to a savings rate \$s \in (0,1)\$ so feasibility (\$0 < s < 1\$, hence \$c > 0\$ and \$k_{t+1} > 0\$) is a *hard constraint* built into the architecture, never a loss term. Here `make_mlp(history_length, (24, 24), 1)` builds a compact `tanh` MLP (the Python notebook uses three width-64 `gelu` layers); `setup_training` pairs it with `Optimisers.Adam`.

> Why predict the savings rate instead of \$k_{t+1}\$ directly? (1) it is bounded and smoother, hence easier to learn than the wide-range level \$k_{t+1}\$; (2) cumulating \$s\, y\$ recovers \$k_{t+1}\$ exactly without an extra layer.
"""

# ╔═╡ 44444444-1005-4444-8444-444444444444
begin
    params = SequenceBrockMirmanParams()
    rule = gauss_hermite_rule(3)
    states = sequence_bm_initial_state(params; batch = hp.batch_size)
    histories = zeros(Float64, 1, hp.history_length, hp.batch_size)
    model = make_mlp(hp.history_length, (24, 24), 1; activation = NNlib.tanh)
    state = setup_training(rng_from_seed(SEED; offset = 1), model, Optimisers.Adam(0.002); parameter_type = Float64)
end

# ╔═╡ 5c119fb6-881c-7c3e-b205-704f3806f425
md"""
### 12–13. From network output to the Euler residual — the loss core

`sequence_bm_residual` turns the network output into the full 3-wide policy \$(s,\ c,\ \mu) = (s_{\text{rate}}\cdot y,\ (1 - s_{\text{rate}})\cdot y,\ u'(c))\$ and evaluates the stochastic Euler equation. With full depreciation \$\delta = 1\$,
\$\$u'(c_t) \;=\; \beta\,\mathbb{E}_t\bigl[u'(c_{t+1})\,\mathrm{MPK}_{t+1}\bigr].\$\$
Writing \$\mu_{t+1} = u'(c_{t+1})\$ and integrating the conditional expectation by Gauss–Hermite,
\$\$\mathbb{E}_t[\mu'\,\mathrm{MPK}'] \;\approx\; \sum_{q=1}^{Q} w_q\, \mu(z_t', \xi_q)\, \mathrm{MPK}(z_t', \xi_q), \qquad z_{t+1} = \rho_z z_t + \sigma_z\, \xi_q.\$\$
The residual minimised is the **relative-error** form
\$\$\mathrm{err\_eul}(x_t) \;=\; \frac{u^{-1}\!\bigl(\beta\,\mathbb{E}_t[\mu'\,\mathrm{MPK}']\bigr)}{c_t} - 1,\$\$
i.e. *implied consumption today / actual consumption today − 1*. Squaring and averaging over the cloud gives `pieces.loss`.

For each of \$Q\$ quadrature nodes the helper builds the next-period shock \$z_{t+1}\$, the next-period expanded state at \$(k_{t+1} = s_t,\, z_{t+1})\$, and the next-period shock history (prepend \$\xi_q\$, drop the oldest lag). The network is therefore called **twice** — once at \$\mathcal{E}_t\$ and once at every \$(\mathcal{E}_{t+1}, \xi_q)\$ — with the *same* weights \$\theta\$; Zygote backpropagates through both. Batched quadrature histories keep the canonical `(features_per_lag, history_length, batch)` layout and are flattened via `flatten_quadrature_histories` at the Lux boundary.
"""

# ╔═╡ 04e4f94c-ecc5-c5da-457e-e08c42a2a04d
md"""
### 14–16. Forward simulator, the SGD step, and the cloud-method loop

`sequence_bm_forward_step` is the **cloud-refresh** step (no gradients): for each cloud entry it draws one innovation \$\varepsilon_{t+1}\$, sets \$z_{t+1} = \rho_z z_t + \sigma_z \varepsilon_{t+1}\$ and \$k_{t+1} = s_t\$, rebuilds the expanded state, and rolls the history forward (prepend \$\varepsilon_{t+1}\$, drop the oldest lag).

Each episode of the outer loop then: (1) evaluates `sequence_loss` on the current cloud; (2) takes one Adam step via `train_step!` (a Zygote gradient of the mean-squared Euler residual, gradient-norm clipped at 10); and (3) draws fresh shocks with `randn(rng, …)` and calls the forward step to move the whole cloud one period later. Over many episodes the cloud converges to the ergodic distribution under the *current* policy — the states the model actually visits — which is where the residual is minimised. In the Python notebook the cloud refresh uses `tf.random.normal` and a `tf.data.Dataset` mini-batch loop; the SGD step wraps the loss in a `tf.GradientTape`.

> **Budget note.** With the Python defaults the loop runs \$2048 \times 16 = 32{,}768\$ SGD steps (CPU: 20–40 min; GPU: a few minutes). This preview runs the `smoke` schedule (5 episodes) so it finishes in seconds; raise `RUN_MODE` for the teaching/production schedule.
"""

# ╔═╡ 55555555-1005-4555-8555-555555555555
begin
    sequence_loss(model, ps, st, batch) = begin
        pieces, st_new = sequence_bm_residual(model, ps, st, batch.states, batch.histories, rule; params)
        return pieces.loss, st_new
    end

    train_result = let states_local = states, histories_local = histories
        initial_loss_local = loss_value(state, sequence_loss, (states = states_local, histories = histories_local))
        history_log_local = NamedTuple[]
        for step in 1:hp.steps
            metrics = train_step!(state, sequence_loss, (states = states_local, histories = histories_local); max_grad_norm = 10.0)
            append_metric!(history_log_local; step, loss = metrics.loss)
            shocks = reshape(randn(rng, hp.batch_size), 1, hp.batch_size)
            states_local, histories_local, _ = sequence_bm_forward_step(state.model, state.ps, state.st, states_local, histories_local, shocks; params)
        end
        (initial_loss = initial_loss_local, history_log = history_log_local, states = states_local, histories = histories_local)
    end
    initial_loss = train_result.initial_loss
    history_log = train_result.history_log
    states_final = train_result.states
    histories_final = train_result.histories
end

# ╔═╡ 33711e02-0d25-a89b-ac39-9593cdc25762
md"""
### 17. Diagnostics

The Python notebook produces three plots: the \$\log_{10}\$ loss curve (should fall to \$\sim 10^{-6}\$ on BM), the final \$(k, z)\$ state cloud (a tight blob around the stochastic steady state, spread set by the AR(1) dynamics), and — only when \$\gamma = \delta = 1\$ — a policy overlay of \$s_{\text{NN}}\$ against the analytic \$\alpha\beta\, y\$ plus histograms of the policy error \$|s_{\text{NN}} - s_{\text{analytic}}|/|s_{\text{analytic}}|\$ and the relative Euler error \$|\mathrm{err\_eul}|\$.

This preview instead recomputes `sequence_bm_residual` on the trained cloud and reports machine-checkable diagnostics: the residual RMSE, the mean savings rate, and the shock-history shapes before and after flattening (`flatten_history` / `flatten_quadrature_histories`) — confirming the `(features_per_lag, history_length, batch)` layout collapses correctly at the Lux boundary.
"""

# ╔═╡ 66666666-1005-4666-8666-666666666666
begin
    diagnostics, _ = sequence_bm_residual(state.model, state.ps, state.st, states_final, histories_final, rule; params)
    flat_history = flatten_history(histories_final)
    quadrature_flat = flatten_quadrature_histories(quadrature_histories(histories_final, rule.nodes))
end

# ╔═╡ cf4ad3c1-e110-6ffd-1556-f20b5a185b6c
md"""
### 18. Take-away

**The single conceptual change** versus the state-space DEQNs of Day 4:

| State-space DEQN | This notebook (sequence-space) |
|---|---|
| Network input: current state \$(k_t, z_t)\$ — *2 floats*. | Network input: shock history \$(\varepsilon_t, \ldots, \varepsilon_{t-T+1})\$. |
| Network output: savings rate (or \$K_{t+1}\$). | Network output: savings rate. |
| Loss: squared Euler residual. | Loss: squared Euler residual. *(unchanged)* |
| Sampling: cloud method. | Sampling: cloud method. *(unchanged)* |

In the BM model the input swap is **larger**, not smaller — sequence space is a teaching device here, not a dimensionality reduction. The payoff appears in the heterogeneous-agent setting (`06_SequenceSpace_KrusellSmith.ipynb` and `KrusellSmith_Tutorial_CPU.ipynb`), where the alternative input would be a 100-bin wealth histogram; there the shock history is genuinely smaller *and* exogenous (it does not move with the policy update), which is the source of the stability gains documented in Azinovic-Yang & Žemlička (2025).

**Suggested follow-ons.** (1) Change the history length and re-run — the error bottoms out once \$\alpha^{T}\$ falls below the loss floor. (2) Set \$\delta < 1\$: the policy is no longer \$\alpha\beta\, y\$ but the algorithm still works (only the analytic comparison is skipped). (3) Continue to the Krusell-Smith sequence-space notebooks for the heterogeneous-agent extension.

The cell below returns the machine-checkable diagnostics summary for this notebook's smoke run.
"""

# ╔═╡ 77777777-1005-4777-8777-777777777777
(
    initial_loss = initial_loss,
    final_loss = history_log[end].loss,
    residual_rmse = residual_summary(diagnostics.euler).rmse,
    history_shape = size(histories_final),
    lux_history_shape = size(flat_history),
    quadrature_history_shape = size(quadrature_flat),
    mean_savings_rate = sum(diagnostics.savings_rate) / length(diagnostics.savings_rate),
)

# ╔═╡ Cell order:
# ╟─11111111-1005-4111-8111-111111111111
# ╟─9f0bd55d-0137-5b1a-4e4a-6b044cdaff9f
# ╟─588b6cac-ebeb-1754-ce66-47740cea9993
# ╟─9b777215-e09f-ce5c-72d6-4ac152ed0421
# ╟─1075955b-d191-abf2-75e1-557ad09292ba
# ╟─6ffcfdbc-4026-4b48-c535-bcc1f8bbd4cc
# ╟─f2966166-ceb8-a200-74bf-f818e831c84a
# ╟─94717842-f1c2-e788-925f-8a25bb9bcf9b
# ╠═22222222-1005-4222-8222-222222222222
# ╟─4ddfbc84-dfeb-b97e-af8e-ef80cfe9ea34
# ╠═33333333-1005-4333-8333-333333333333
# ╟─526b7856-3314-03d2-52cb-65793af247f2
# ╟─946e2200-a680-5c3f-d403-af1c75353ef0
# ╠═44444444-1005-4444-8444-444444444444
# ╟─5c119fb6-881c-7c3e-b205-704f3806f425
# ╟─04e4f94c-ecc5-c5da-457e-e08c42a2a04d
# ╠═55555555-1005-4555-8555-555555555555
# ╟─33711e02-0d25-a89b-ac39-9593cdc25762
# ╠═66666666-1005-4666-8666-666666666666
# ╟─cf4ad3c1-e110-6ffd-1556-f20b5a185b6c
# ╠═77777777-1005-4777-8777-777777777777
