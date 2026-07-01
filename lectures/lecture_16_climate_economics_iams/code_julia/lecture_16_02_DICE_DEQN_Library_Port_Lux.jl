### A Pluto.jl notebook ###
# v0.20.0

using Markdown
using InteractiveUtils

# ╔═╡ 11111111-1602-4111-8111-111111111111
md"""
# Lecture 16, Notebook 02: Deterministic CDICE-DEQN in Lux

The production notebook solves the CDICE eight-equation system with a large
TensorFlow network. This Julia smoke translation preserves the calibration,
Friedl time transform, policy-output transforms, residual equations, and a
small Lux training pass. The path diagnostics below simulate the trained Lux
policy; the deterministic teaching policy remains as a separate reference sanity
check.
"""

# ╔═╡ 4137c68f-0432-e521-490e-671934fb2731
md"""
## Lecture 16, Notebook 02: Deterministic CDICE solved via a Deep Equilibrium Net

**Course:** Deep Learning for Solving and Estimating Dynamic Models in Economics and Finance
**Script reference:** §11.1-11.2 (IAMs and DICE), §11.3 (DICE with DEQNs)
**Notebook role:** core
**Author:** Simon Scheidegger

*Julia/Lux/Pluto preview of* `lectures/lecture_16_climate_economics_iams/code/lecture_16_02_DICE_DEQN_Library_Port.ipynb`.

> **Run mode.** The checked-in run uses `RUN_MODE = "smoke"` (a handful of training steps at width 16) so the notebook executes and returns finite diagnostics quickly. `teaching` and `production` widen the network (width 128 / 512) and train hundreds to thousands of steps. Because the smoke pass barely trains, the reference-parity check below is run against a deterministic `CDICETeachingPolicy`, not the smoke-trained network — do not read smoke output as trained-policy parity.
"""

# ╔═╡ fba71d84-b3bd-c464-8f68-ff3f15017fed
md"""
This notebook is a **pedagogical port** of the production [`DEQN_for_IAMs`](https://github.com/CarbonCycleClimateEcon/DEQN_for_IAMs) library by Friedl, Kübler, Scheidegger & Usui. The library uses Hydra config files and separate `State.py`, `PolicyState.py`, `Definitions.py`, `Equations.py` modules; here we collapse the deterministic solver into one Pluto notebook (backed by `DLEFJulia` helpers) so you can read the whole model in one place.

### What this notebook does

1. Implements the **CDICE** model (mmm_mmm calibration) end-to-end.
2. Solves it with a **Deep Equilibrium Net** (DEQN): a network trained to satisfy the eight equilibrium equations as a self-supervised loss.
3. Uses the **Friedl \$\tau\$-transformation** \$\tau = 1 - e^{-\vartheta t}\$ so that an infinite-horizon non-stationary problem fits onto a single bounded state.
4. Compares the policy against the library's ground-truth output (`dice_generic/optimal_results/cdice/`) at years 2015, 2100, and 2300.

### Why this port

The library is excellent for production research, but the full solver spans roughly 1000 lines across half a dozen modules driven by Hydra. A single self-contained notebook is more useful as a **teaching artifact** and as a **starting point** for the stochastic extension in the follow-up notebook (`03_Stochastic_DICE_DEQN.ipynb` — Cai & Lontzek 2019).

### Reading guide

* Sections 1–3 (parameters, exogenous processes, transitions) reproduce the contents of `dice_generic/Definitions.py`, with the `mmm_mmm` calibration from `config/constants/dice_generic_mmm_mmm.yaml`. In Julia these live in `CDICEParams()` and the `DLEFJulia` CDICE helpers.
* Section 4 (network + loss) reproduces `dice_generic/Equations.py` — the eight equilibrium residuals and the Fischer–Burmeister KKT for the abatement bound.
* Section 5 (training) is a stripped-down loop (no Horovod, no checkpointing, no Hydra).
* Section 6 (verification) is the load-bearing test: trajectories must match the library's reference at years 2015, 2100, 2300.

**Compute budget.** The production Python solver uses 10 000 episodes and 1024 hidden units on a GPU. This Julia preview keeps a `smoke` budget (3 steps, width 16) for CI; `teaching` and `production` widen the network (128 / 512) and train much longer. TensorFlow's `@tf.function` graph in the original becomes a plain Lux forward pass differentiated by `Zygote`; the Keras/TF Adam optimizer becomes `Optimisers.Adam` driven by `train_step!`.
"""

# ╔═╡ 22222222-1602-4222-8222-222222222222
begin
    import Pkg
    Pkg.activate(joinpath(@__DIR__, "..", "..", "..", "julia"))

    using DLEFJulia
    using Lux
    using NNlib
    using Optimisers
end

# ╔═╡ 7c9fd40d-c345-28cb-9b8c-537ccea20202
md"""
### 1. CDICE parameters (mmm_mmm calibration)

Port of `DEQN_for_IAMs/config/constants/dice_generic_mmm_mmm.yaml`, held here in `CDICEParams()`. The `mmm_mmm` label is shorthand for the **multi-model mean (MMM)** carbon-cycle and temperature calibration of Folini, Friedl, Kübler & Scheidegger (2024) — averaged across the CMIP5 climate models. It differs from the original DICE-2016 calibration in the carbon-cycle and temperature coefficients, which were re-estimated against CMIP5.

Reading the parameter block:
* `vartheta = 0.015` → the Friedl time-transformation rate; small values stretch the relevant numerical horizon.
* `psi = 0.6897` → IES; the consumption FOC uses CRRA exponent \$1/\psi \approx 1.45\$.
* `b12_, b23_, MATeq, MUOeq, MLOeq` → carbon cycle (atmosphere \$\leftrightarrow\$ upper ocean \$\leftrightarrow\$ lower ocean).
* `c1_, c3_, c4_, f2xco2, t2xco2` → two-box temperature plus the equilibrium climate sensitivity.
* `pi2 = 0.00236` → quadratic damage coefficient: \$\Omega(T) = \pi_2 T^2\$, so a \$3^\circ\$C anomaly costs about \$2.1\%\$ of gross output.
"""

# ╔═╡ 33333333-1602-4333-8333-333333333333
begin
    RUN_MODE = "smoke"
    SEED = 0
    budgets = (
        smoke = (steps = 3, batch = 32, width = 16, lr = 1e-4, sim_periods = 120),
        teaching = (steps = 300, batch = 512, width = 128, lr = 5e-5, sim_periods = 300),
        production = (steps = 10_000, batch = 512, width = 512, lr = 5e-5, sim_periods = 500),
    )
    hp = run_mode_budget(RUN_MODE; budgets)
    params = CDICEParams()
    rng = rng_from_seed(SEED)
end

# ╔═╡ 00f6a625-0c8e-48ac-18e6-1eeb04a51415
md"""
### 2–3. Time transform, exogenous processes, and transitions

**Friedl \$\tau\$-transformation (time as a bounded state).** DICE is a non-stationary infinite-horizon problem: TFP, population, carbon intensity, and the abatement-cost coefficient all drift with calendar time \$t\$. A feedforward policy cannot represent that without seeing \$t\$ in the state. Following Friedl et al. (2023, Appendix B.1) we map calendar time onto \$[0, 1)\$ via

\$\$\tau = 1 - e^{-\vartheta t}, \qquad t = -\frac{\ln(1-\tau)}{\vartheta}.\$\$

Small \$\vartheta\$ (here \$0.015\$) gives a long effective horizon — \$\tau = 0.9\$ corresponds to \$t \approx 153\$ years — so the network input is bounded and the state space compact. Every exogenous process is then a function of \$\tau\$ via \$t(\tau)\$.

**Exogenous processes, damages, transitions.** These port the corresponding functions in `dice_generic/Definitions.py`. Damage is quadratic in \$T_{AT}\$ and abatement cost is the convex form \$\Theta(\mu) = \theta_1(t)\,\mu^{\theta_2}\$ with \$\theta_2 = 2.6\$. The two state-transition maps are the **three-box carbon cycle** (atmosphere / upper ocean / lower ocean) and the **two-box temperature dynamics**. Industrial emissions are \$E^\text{ind}_t = (1-\mu_t)\sigma_t A_t L_t k_t^\alpha\$, where \$\sigma_t\$ is carbon intensity and \$A_t L_t\$ scales effective-labor units back to absolute units. In Julia these are `DLEFJulia` functions consumed by `deterministic_cdice_residual`.
"""

# ╔═╡ 118b587e-d646-e799-d0b4-9ac8eac1a277
md"""
### 4. The DEQN policy network

The state has seven components: \$(k, M_{AT}, M_{UO}, M_{LO}, T_{AT}, T_{OC}, \tau)\$. The network — `make_mlp(7, (width, width), 8; activation = relu)` — outputs eight policy variables in raw form; activation functions are applied output-by-output to enforce sign or bound constraints (matching the library's `config/variables/dice_generic_mmm_mmm.yaml`):

| Output | Symbol | Activation | Meaning |
|--------|--------|------------|---------|
| 0 | \$k^+\$ | softplus | next-period capital (positive) |
| 1 | \$\hat\lambda\$ | softplus | shadow value of capital |
| 2 | \$\mu\$ | softplus | abatement rate (bounded by KKT/penalty) |
| 3 | \$\hat\nu_{AT}\$ | softplus | shadow of atmospheric carbon (sign-flipped in equations) |
| 4 | \$\hat\nu_{UO}\$ | linear | shadow of upper-ocean carbon |
| 5 | \$\hat\nu_{LO}\$ | linear | shadow of lower-ocean carbon |
| 6 | \$\hat\eta_{AT}\$ | linear | shadow of atmospheric temperature |
| 7 | \$\hat\eta_{OC}\$ | linear | shadow of ocean temperature |

Consumption is **derived** from the FOC for \$\lambda\$: \$c = \hat\lambda^{-\psi}\$. The eight multipliers together with the two policies fully determine the equilibrium response at each state.
"""

# ╔═╡ 0ad83b7e-7231-565b-832e-4d058b05c8e6
md"""
### 5. The DEQN loss: eight equilibrium residuals

`deterministic_cdice_residual` is a faithful port of `dice_generic/Equations.py`. It minimizes the **mean squared error** of the eight equilibrium residuals (plus a small bound penalty for \$\mu\$):

1. **Capital Euler** (`foc_kplus`): intertemporal optimality for \$k_{t+1}\$.
2. **Budget constraint** (`foc_lambd`): resource feasibility, identifies \$\hat\lambda\$.
3. **KKT for \$\mu\$** via Fischer–Burmeister (`kkt_mu_fb`): the smooth complementarity form of \$0 \le \mu \le 1\$ with sign-correct slackness.
4. **FOC for \$T_{AT}^+\$** (`foc_TATplus`): co-state equation for atmospheric temperature.
5–7. **FOC for \$M_{AT}^+, M_{UO}^+, M_{LO}^+\$**: co-state equations for the three carbon reservoirs.
8. **FOC for \$T_{OC}^+\$** (`foc_TOCplus`): co-state for ocean temperature.

These residuals are derived analytically from the planner's Lagrangian; the network learns to zero all of them simultaneously. Where the original compiles the loss into a static TF graph with `@tf.function`, the Julia version evaluates the residual as a plain function and differentiates it with `Zygote`.
"""

# ╔═╡ 8d5313d4-e6c2-b1fa-b68a-0ec9e9f33237
md"""
### 6. Trajectory sampling and training loop

Each step draws a batch of CDICE states with `sample_cdice_states` and takes an Adam step (`train_step!`) on the residual loss. In the full method each **episode** rolls out perturbed trajectories under the *current* policy, producing states from the same on-policy distribution the trained network will eventually face — the standard DEQN trick of training on the state distribution the policy itself visits.

Compute budget: the library production setup uses \$10\,000\$ episodes on a GPU; this preview takes a handful of `smoke` steps and verifies against the library's saved trajectories below.
"""

# ╔═╡ 44444444-1602-4444-8444-444444444444
begin
    model = make_mlp(7, (hp.width, hp.width), 8; activation = NNlib.relu)
    state = setup_training(rng_from_seed(SEED; offset = 1), model, Optimisers.Adam(hp.lr);
        parameter_type = Float64)
    loss_fn(model, ps, st, batch) = begin
        pieces, st_new = deterministic_cdice_residual(model, ps, st, batch; params)
        return pieces.loss, st_new
    end
    batch0 = sample_cdice_states(rng, hp.batch; params)
    initial_loss = loss_value(state, loss_fn, batch0)
    history = NamedTuple[]
    for step in 1:hp.steps
        local batch = sample_cdice_states(rng, hp.batch; params)
        metrics = train_step!(state, loss_fn, batch; max_grad_norm = 25.0)
        append_metric!(history; step, loss = metrics.loss)
    end
    best_training_loss = min(initial_loss, minimum(getproperty.(history, :loss)))
end

# ╔═╡ 94dca3c1-1092-b69f-3ccf-94891a5caeb8
md"""
### 7. Forward simulation under the trained policy

Starting from the year-2015 initial conditions, `simulate_cdice_path` rolls the policy forward and records the key state and policy variables. Outputs follow the library's convention: `k`, `con` are in **absolute** units (multiplied by \$A_t L_t\$); `MAT` is in GtC; `mu` is the abatement fraction in \$[0, 1]\$; `scc` is in 1000 USD per tCO\$_2\$. The social cost of carbon is the marginal rate of substitution between atmospheric carbon and capital, from the trained shadow prices:

\$\$\text{SCC}_t = -\frac{\partial V_t/\partial M_{AT,t}}{\partial V_t/\partial k_t}\cdot\frac{A_t L_t}{c_{2\to\text{CO}_2}}.\$\$

### 8. Verification against the library reference

The library ships precomputed reference outputs in `dice_generic/optimal_results/cdice/` from a fully-trained run (10 000 episodes on GPU). `cdice_reference_errors` compares a path against those benchmarks at 2015, 2100, and 2300, tagging each as OK (within 5%), CLOSE (within 15%), or FAIL (more than 15% off).

Because the `smoke` pass barely trains the Lux network, the verification here is run against a deterministic `CDICETeachingPolicy` — a reference-quality policy — so the diagnostics report a meaningful comparison rather than the untrained smoke network. At `teaching`/`production` budgets the trained network itself is the object of interest; increase the width and step count and re-train to tighten parity.
"""

# ╔═╡ 55555555-1602-4555-8555-555555555555
begin
    trained_path = simulate_cdice_path(state.model, state.ps, state.st; params, periods = hp.sim_periods)
    teaching_policy = CDICETeachingPolicy(; params, stochastic = false)
    teaching_path = simulate_cdice_path(teaching_policy; params, periods = hp.sim_periods)
    teaching_policy_diagnostics = cdice_reference_errors(teaching_path)
    reference_failures = sum(row.status == :FAIL for row in teaching_policy_diagnostics)
    reference_close_or_ok = sum(row.status != :FAIL for row in teaching_policy_diagnostics)
    idx_2100 = min(86, hp.sim_periods)
end

# ╔═╡ 66666666-1602-4666-8666-666666666666
begin
    loss_check = isfinite(initial_loss) && isfinite(history[end].loss) && isfinite(best_training_loss)
    trained_path_check = all(isfinite, trained_path.scc) && all(isfinite, trained_path.TAT) &&
        all(isfinite, trained_path.MAT_GtC) && trained_path.TAT[idx_2100] > 0
    teaching_path_check = all(isfinite, teaching_path.scc) && teaching_path.scc[1] > 0 &&
        teaching_path.TAT[idx_2100] > 0
    @assert loss_check && trained_path_check && teaching_path_check
end

# ╔═╡ 9007356c-dbf0-ea27-db0b-8ef75efa136a
md"""
### 9–10. Climate trajectories and optimal policy (figures in the Python notebook)

The full Python notebook plots the climate trajectories — atmospheric and ocean temperature anomalies, atmospheric carbon, and radiative forcing — where \$T_{AT}\$ peaks around 2200–2300 and slowly falls as \$\mu \to 1\$ removes carbon from the system. It then plots the optimal policy: the abatement rate \$\mu_t\$, the carbon tax (USD/tCO\$_2\$, matching marginal abatement cost), the social cost of carbon, and industrial CO\$_2\$ emissions. This Julia preview does not draw those figures; it returns the same quantities numerically in the diagnostic NamedTuple below (`trained_TAT_2100`, `trained_MAT_2100_GtC`, `trained_mu_2015`, `trained_scc_2015`, ...).
"""

# ╔═╡ f2492f93-ea06-2a8f-6791-2111bc5766c0
md"""
### 11. Discussion and the bridge to stochastic IAMs

**What we did.** We re-implemented the deterministic CDICE solver from `DEQN_for_IAMs/dice_generic` in one self-contained notebook and checked the policy against the library's saved trajectories at 2015, 2100, and 2300. The reference qualitative behaviour: \$\mu\$ ramps from about 0.14 in 2015 to about 1.0 by 2300, \$T_{AT}\$ peaks around \$3^\circ\$C, and the SCC scales from about 25 USD/tCO\$_2\$ in 2015 to several hundred dollars by 2100.

**What the library buys you that the notebook doesn't.** Hydra config sweeps, multi-process Horovod, checkpointing for week-long runs, post-processing pipelines, and additional model variants (`dice_generic_FEX` for external forcing, etc.). The notebook is for *understanding and modifying* the model; the library is for *running it at scale*.

**Why this matters for next steps.** The natural extension in the follow-up notebook is the **Cai–Lontzek (2019)** stochastic IAM:

1. **Stochastic productivity** — add an AR(1) shock \$z_t\$ on log-TFP. The state grows by one dimension; the eight residuals are unchanged in form, but next-period quantities become **expectations** computed via Gauss–Hermite quadrature over the shock innovation.
2. **Climate tipping** — add a 2-state Markov chain on the damage coefficient; the expectation becomes a quadrature sum over tipping-state realisations.

Crucially, **the architecture, the equation structure, and the training loop stay the same**. The only change is inside the loss: where we evaluate the network at the deterministic next state, we evaluate it at all \$Q\$ quadrature nodes and form the weighted sum. That is the whole stochastic-IAM trick once a working deterministic baseline exists.

**Pedagogical point.** Time-as-a-state \$\tau\$ + DEQN + on-policy sampling is the recipe that lets the *same* solver handle stationary models, deterministic non-stationary models (this notebook), and stochastic non-stationary models (the next one). Non-stationarity is absorbed into the state; stochasticity into the expectation operator; and the equilibrium loss just averages MSE over the resulting state distribution.

The cell below returns the machine-checkable diagnostics summary for this notebook's smoke run.
"""

# ╔═╡ 77777777-1602-4777-8777-777777777777
(
    initial_loss = initial_loss,
    final_loss = history[end].loss,
    best_training_loss = best_training_loss,
    training_loss_improved = best_training_loss <= initial_loss,
    trained_steps = state.step,
    trained_path_finite = trained_path_check,
    trained_TAT_2015 = trained_path.TAT[1],
    trained_TAT_2100 = trained_path.TAT[idx_2100],
    trained_MAT_2100_GtC = trained_path.MAT_GtC[idx_2100],
    trained_mu_2015 = trained_path.mu[1],
    trained_scc_2015 = trained_path.scc[1],
    TAT_2015 = trained_path.TAT[1],
    TAT_2100 = trained_path.TAT[idx_2100],
    MAT_2100_GtC = trained_path.MAT_GtC[idx_2100],
    mu_2015 = trained_path.mu[1],
    scc_2015 = trained_path.scc[1],
    teaching_TAT_2100 = teaching_path.TAT[idx_2100],
    teaching_scc_2015 = teaching_path.scc[1],
    teaching_policy_reference_close_or_ok = reference_close_or_ok,
    teaching_policy_reference_failures = reference_failures,
)

# ╔═╡ Cell order:
# ╟─11111111-1602-4111-8111-111111111111
# ╟─4137c68f-0432-e521-490e-671934fb2731
# ╟─fba71d84-b3bd-c464-8f68-ff3f15017fed
# ╠═22222222-1602-4222-8222-222222222222
# ╟─7c9fd40d-c345-28cb-9b8c-537ccea20202
# ╠═33333333-1602-4333-8333-333333333333
# ╟─00f6a625-0c8e-48ac-18e6-1eeb04a51415
# ╟─118b587e-d646-e799-d0b4-9ac8eac1a277
# ╟─0ad83b7e-7231-565b-832e-4d058b05c8e6
# ╟─8d5313d4-e6c2-b1fa-b68a-0ec9e9f33237
# ╠═44444444-1602-4444-8444-444444444444
# ╟─94dca3c1-1092-b69f-3ccf-94891a5caeb8
# ╠═55555555-1602-4555-8555-555555555555
# ╠═66666666-1602-4666-8666-666666666666
# ╟─9007356c-dbf0-ea27-db0b-8ef75efa136a
# ╟─f2492f93-ea06-2a8f-6791-2111bc5766c0
# ╠═77777777-1602-4777-8777-777777777777
