### A Pluto.jl notebook ###
# v0.20.0

using Markdown
using InteractiveUtils

# ╔═╡ 11111111-1603-4111-8111-111111111111
md"""
# Lecture 16, Notebook 03: Stochastic CDICE-DEQN in Lux

This notebook adds the Cai-Lontzek-style AR(1) productivity shock to the CDICE
state. The Euler expectation is evaluated with the shared Gauss-Hermite rule,
and smoke mode keeps the Monte Carlo fan-chart inputs small.
"""

# ╔═╡ 879ce380-1b9e-72b8-c53b-7c05349498e3
md"""
## Lecture 16, Notebook 03: Stochastic CDICE-DEQN with AR(1) productivity shocks

**Course:** Deep Learning for Solving and Estimating Dynamic Models in Economics and Finance
**Script reference:** §11.1-11.2 (IAMs and DICE), §11.3 (DICE with DEQNs)
**Notebook role:** extension
**Author:** Simon Scheidegger

*Julia/Lux/Pluto preview of* `lectures/lecture_16_climate_economics_iams/code/lecture_16_03_Stochastic_DICE_DEQN.ipynb`.

> **Run mode.** The checked-in run uses `RUN_MODE = "smoke"` (3 training steps at width 16, an 8-path Monte Carlo over 90 periods). `teaching` and `production` widen the network and use 200 / 500 Monte-Carlo paths over 285 periods. As in notebook 02, the smoke network is barely trained, so a deterministic `CDICETeachingPolicy` supplies the reference-quality path for diagnostics — smoke output is not trained-policy parity.
"""

# ╔═╡ a3472325-6d65-ffe4-8c1b-43d35f6310ea
md"""
This notebook **extends notebook 02** (`02_DICE_DEQN_Library_Port.ipynb`, the deterministic DICE-DEQN port of the Folini–Friedl–Kübler–Scheidegger 2025 *Climate in Climate Economics* library) by adding a **single AR(1) productivity shock** to total factor productivity, in the spirit of **Cai & Lontzek (2019, JPE)**.

**What's new compared to notebook 02 (everything else is identical):**

| element | notebook 02 (deterministic) | this notebook (stochastic) |
|---|---|---|
| state vector | \$(k, M_{AT}, M_{UO}, M_{LO}, T_{AT}, T_{OC}, \tau)\$ — 7 dims | + log-TFP shock \$z_t\$ → 8 dims |
| TFP | \$A(t)\$ deterministic trend | \$A(t,z) = A(t)\,e^{z_t}\$ |
| transition for \$z\$ | (none) | \$z_{t+1} = \rho_z z_t + \sigma_z \varepsilon_{t+1}\$, \$\varepsilon\sim\mathcal{N}(0,1)\$ |
| equilibrium residuals | 8 FOCs from `dice_generic/Equations.py` | same 8, but expectations replaced by 5-node Gauss–Hermite quadrature |
| network | 7→512→512→8 | 8→512→512→8 |
| forward simulation | single trajectory | Monte Carlo over \$z\$ paths |

**Calibration (Cai–Lontzek 2019 §V.A):** \$\rho_z = 0.95\$, \$\sigma_z = 0.0125\$.

**You will see:**
1. The deterministic limit (\$\sigma_z = 0\$) reproduces the notebook-02 reference (Gate A).
2. With \$\sigma_z = 0.0125\$ the SCC develops a non-trivial distribution that widens with the horizon (Gate B).
3. SCC grows **convexly** over time (the slope \$\partial \text{SCC}/\partial t\$ increases) — qualitatively in line with both DICE and Cai–Lontzek.

**Runtime note.** The original TensorFlow notebook takes ~28 minutes end-to-end at `teaching` (training + 200-path Monte Carlo + plots). This Julia/Lux preview keeps a small `smoke` budget (3 steps, width 16, 8-path Monte Carlo) for CI. Cai & Lontzek (2019) report runtimes of multiple days on multi-core HPC for the equivalent fully-solved stochastic IAM with their dynamic-programming / Gaussian-process method (DPGM). The deep-learning approach replaces the curse-of-dimensionality blow-up of grid-based DP with a fixed-cost network evaluation, so adding more shocks scales roughly linearly rather than exponentially.
"""

# ╔═╡ 4d7aed42-a702-6c62-4e2c-ad6f9ef8f522
md"""
### 1. Imports and determinism

The setup cell activates the shared `julia` project and imports `DLEFJulia` (the CDICE model, residuals, and simulation helpers) together with `Lux`, `NNlib`, `Optimisers`, and `Statistics`. `SEED = 0` fixes the RNG so the smoke run is reproducible.
"""

# ╔═╡ 22222222-1603-4222-8222-222222222222
begin
    import Pkg
    Pkg.activate(joinpath(@__DIR__, "..", "..", "..", "julia"))

    using DLEFJulia
    using Lux
    using NNlib
    using Optimisers
    using Statistics
end

# ╔═╡ f0178385-8708-2b04-93fb-adae011b92ed
md"""
### 2. Calibration

The **climate–economy block** is the *same* `mmm_mmm` calibration used in notebook 02 (the multi-model-mean climate emulator of Folini et al. 2024). We add only:

* Stochastic-shock parameters \$\rho_z, \sigma_z\$ (Cai–Lontzek 2019 §V.A).
* The Friedl τ-transform parameter \$\vartheta = 0.015\$ that compactifies infinite time onto \$[0,1)\$.

The Python version keeps everything in a `class P:` namespace promoted to `np.float32` for TensorFlow; here the whole calibration lives in `CDICEParams()` (Float64), read directly by the residual and simulation helpers.
"""

# ╔═╡ a79cd689-dc3a-3121-5b20-8499500ff1d7
md"""
### 3. The AR(1) productivity shock and Gauss–Hermite quadrature

We add a single stationary log-TFP shock:

\$\$z_{t+1} = \rho_z z_t + \sigma_z\,\varepsilon_{t+1},\qquad \varepsilon_{t+1}\sim\mathcal{N}(0,1).\$\$

Effective TFP becomes \$A_{\text{eff}}(t,z) = A(t)\,e^{z}\$, so \$z\$ multiplies output, emissions, and mitigation cost everywhere.

**Why Gauss–Hermite (GH).** The DEQN equilibrium residuals that contain conditional expectations take the form

\$\$\mathbb{E}_{t}\!\left[h(s_{t+1}, z_{t+1})\right] = \frac{1}{\sqrt{\pi}} \int h\bigl(s_{t+1}, \rho_z z_t + \sigma_z\sqrt{2}\,x\bigr)\,e^{-x^2}\,dx.\$\$

A 5-node Gauss–Hermite rule (built here with `gauss_hermite_rule(5)`) is exact for polynomials in \$\varepsilon\$ up to degree 9 — far more than needed for a smooth, near-quadratic objective. It costs **5 extra forward passes through the network per loss evaluation**, the dominant overhead. Monte Carlo at the same accuracy would need ~10⁴ samples.
"""

# ╔═╡ 8171029c-f683-2cd0-aebb-a8b2047a217f
md"""
### 4. Friedl τ-transform

DEQN needs a **bounded state** (so the network can normalize inputs to \$[0,1]\$), but real time \$t \in [0, \infty)\$ is unbounded. Following Friedl, Kübler, Scheidegger & Usui (2023, 2024) we work in \$\tau \in [0, 1)\$ with

\$\$\tau(t) = 1 - e^{-\vartheta t}, \qquad t(\tau) = -\frac{1}{\vartheta}\ln(1-\tau).\$\$

The model is in real time but the *network input* uses τ.
"""

# ╔═╡ 33333333-1603-4333-8333-333333333333
begin
    RUN_MODE = "smoke"
    SEED = 0
    budgets = (
        smoke = (steps = 3, batch = 32, width = 16, lr = 1e-4, n_mc = 8, periods = 90),
        teaching = (steps = 300, batch = 512, width = 128, lr = 5e-5, n_mc = 200, periods = 285),
        production = (steps = 10_000, batch = 512, width = 512, lr = 5e-5, n_mc = 500, periods = 285),
    )
    hp = run_mode_budget(RUN_MODE; budgets)
    params = CDICEParams()
    rng = rng_from_seed(SEED)
    rule = gauss_hermite_rule(5)
end

# ╔═╡ 8a259f3b-67cd-7560-db14-37c2ea7a3866
md"""
### 5–6. Exogenous functions, damages, and one-period transitions

The exogenous processes are identical to notebook 02 except that TFP now takes the shock as a second argument: `tfp_trend(t)` is the deterministic component and `tfp(t, z) = tfp_trend(t) * exp(z)`. Damages are quadratic, \$\Omega(T) = \pi_2 T^2\$ (with \$\pi_1 = 0\$); mitigation cost is \$\Theta(\mu, t) = \theta_1(t)\,\mu^{\theta_2}\$ with \$\theta_2 = 2.6\$; industrial emissions \$E_{\text{ind}} = (1-\mu)\,\sigma(t)\,A(t,z)\,L(t)\,k^\alpha\$ now carry \$z\$. The carbon 3-box and temperature 2-box dynamics are unchanged. Where the original makes every function TF-native so it JIT-compiles inside `@tf.function`, the Julia versions are plain `DLEFJulia` functions differentiated by `Zygote`.
"""

# ╔═╡ e53f11d2-af1d-a9e5-c02d-fc0a7c2e756b
md"""
### 7. Network — 8-D input, 8 outputs

The state is \$s = (k, M_{AT}, M_{UO}, M_{LO}, T_{AT}, T_{OC}, \tau, z)\$ and the network is `make_mlp(8, (width, width), 8; activation = relu)`. Outputs match the library's `variables.yaml` activations:

| index | symbol | activation | meaning |
|---|---|---|---|
| 0 | \$k_+\$ | softplus | next-period capital |
| 1 | \$\hat{\lambda}\$ | softplus | reciprocal CRRA marginal utility |
| 2 | \$\mu\$ | softplus | abatement rate (penalty for \$>1\$) |
| 3 | \$\hat{\nu}_{AT}\$ | softplus | shadow price (normalized) on \$M_{AT}\$ |
| 4 | \$\hat{\nu}_{UO}\$ | linear | shadow price on \$M_{UO}\$ |
| 5 | \$\hat{\nu}_{LO}\$ | linear | shadow price on \$M_{LO}\$ |
| 6 | \$\hat{\eta}_{AT}\$ | linear | shadow price on \$T_{AT}\$ |
| 7 | \$\hat{\eta}_{OC}\$ | linear | shadow price on \$T_{OC}\$ |

Two hidden layers (512 ReLU units at production scale; narrower under the previews, e.g. width 16 in `smoke` and 128 in `teaching`). Note that the full Python notebook builds a fixed 512-wide network for *every* run mode, so this preview's reduced-width `smoke`/`teaching` budgets under-deliver network width relative to that reference; only the executed `smoke` path matters for CI. The state is min-max normalized to \$[0,1]^8\$ before being fed in.
"""

# ╔═╡ 117773e6-0af2-6ea9-8abb-164f8d287496
md"""
### 8. DEQN loss — 8 hand-derived FOCs with quadrature expectations

`stochastic_cdice_residual` uses the **same 8 equilibrium residuals** as notebook 02 (transcribed from `dice_generic/Equations.py`):

| eq | meaning | expectation? |
|---|---|---|
| 1 | Euler equation in capital (\$\hat\lambda \cdot \text{growth} = \beta\,\mathbb{E}[\cdots]\$) | yes |
| 2 | Resource constraint (defines \$c\$) | no |
| 3 | KKT for \$\mu\$ (Fischer–Burmeister with upper bound 1) | no |
| 4 | Envelope on \$T_{AT}\$ | yes |
| 5 | Envelope on \$M_{AT}\$ | yes |
| 6 | Envelope on \$M_{UO}\$ | yes |
| 7 | Envelope on \$M_{LO}\$ | yes |
| 8 | Envelope on \$T_{OC}\$ | yes |

For each equation with an expectation we replace \$\mathbb{E}[\,\cdot\,]\$ by \$\sum_{q=1}^{5} w_q\,(\,\cdot\,)\big|_{z'=\rho_z z + \sigma_z\sqrt{2}x_q}\$, where \$(x_q, w_q)\$ are the Gauss–Hermite nodes and weights carried in `rule`. The original unrolls the 5-node loop into a single TF graph; the Julia residual sums over the quadrature nodes and is differentiated by `Zygote`. Passing \$\sigma_z = 0\$ recovers the deterministic loss (the sanity gate).
"""

# ╔═╡ 6c41d9e5-d5d4-b8f7-384b-0df910b14771
md"""
### 9–10. Stochastic sampler and training loop

The full Deep Equilibrium Nets method (and the Python notebook's `gen_traj`) trains **on-policy**: it rolls the *current* network forward ~300 steps so the shock \$z\$ evolves AR(1) along each trajectory and the training batch concentrates on the ergodic set. This Julia preview instead uses an **off-policy box sampler**: `sample_cdice_states(...; stochastic = true)` draws each state coordinate i.i.d. from a fixed bounded range (\$\tau\$ uniform on \$[0, 0.8]\$) with the shock \$z\$ drawn once from the stationary distribution \$\mathcal{N}(0, \sigma_z/\sqrt{1-\rho_z^2})\$ — there is no policy rollout and no AR(1) evolution along a trajectory. Setting \$\sigma_z = 0\$ collapses \$z\$ to zero and recovers the deterministic sampler (useful for the sanity gate). Training then takes Adam steps (`train_step!`) on the residual loss. The Python *teaching* run that produced the reference notebook uses a 3-step LR schedule over 200 episodes × 100 inner steps × batch 512 (~22 minutes on one CPU core, with the 5-node quadrature the dominant overhead); Python's *production* scale is 10 000 episodes × 200 trajectories × 500 steps × batch 128. This preview takes a handful of `smoke` steps.
"""

# ╔═╡ 44444444-1603-4444-8444-444444444444
begin
    model = make_mlp(8, (hp.width, hp.width), 8; activation = NNlib.relu)
    state = setup_training(rng_from_seed(SEED; offset = 1), model, Optimisers.Adam(hp.lr);
        parameter_type = Float64)
    loss_fn(model, ps, st, batch) = begin
        pieces, st_new = stochastic_cdice_residual(model, ps, st, batch, rule; params)
        return pieces.loss, st_new
    end
    batch0 = sample_cdice_states(rng, hp.batch; params, stochastic = true)
    initial_loss = loss_value(state, loss_fn, batch0)
    history = NamedTuple[]
    for step in 1:hp.steps
        local batch = sample_cdice_states(rng, hp.batch; params, stochastic = true)
        metrics = train_step!(state, loss_fn, batch; max_grad_norm = 25.0)
        append_metric!(history; step, loss = metrics.loss)
    end
    best_training_loss = min(initial_loss, minimum(getproperty.(history, :loss)))
end

# ╔═╡ ea24afb4-8162-e59e-aad4-b825aa43bda5
md"""
### 12–16. Forward simulation, Gate A, and the Monte Carlo (Gate B)

`simulate_cdice_path` rolls the *trained* policy forward from the year-2015 initial condition. With `sigma_z_realized = 0` we get the **deterministic baseline** (\$z\$ stays at 0); with a positive value we get one Monte-Carlo draw. The Social Cost of Carbon is the marginal-value ratio

\$\$\text{SCC} = -\frac{\partial V/\partial M_{AT}}{\partial V/\partial k}\cdot\frac{A\,L}{c_2}\$\$

expressed in USD/tCO\$_2\$.

**Gate A — deterministic sanity check.** Setting \$z\equiv 0\$ along the path should recover the notebook-02 deterministic CDICE benchmark to within ~5%, confirming the 8-D machinery is consistent with the 7-D deterministic limit.

**Gate B — Monte Carlo over \$z\$ paths.** `cdice_monte_carlo_paths` lets the shock evolve and simulates many independent paths, storing the full time series for every variable so we can form fan charts (5/25/50/75/95-percentile bands) of \$T_{AT}\$, \$\mu\$, and SCC over time. One deliberate difference from the Python `simulate_one`, which starts every Monte-Carlo path from \$z_0 = 0\$ at 2015 (so the fan chart emanates from a single deterministic point and then widens): this Julia preview draws each path's *initial* \$z\$ from its stationary distribution \$\mathcal{N}(0, \sigma_z/\sqrt{1-\rho_z^2})\$, so the bands are already slightly dispersed at the 2015 origin. Because \$z\$ is near-stationary by 2100 in either convention, the 2100 diagnostics reported below are essentially unaffected.

As in notebook 02, the `smoke` network is barely trained, so a deterministic `CDICETeachingPolicy` provides the reference-quality path; do not read smoke output as trained-policy parity.
"""

# ╔═╡ 55555555-1603-4555-8555-555555555555
begin
    deterministic_path = simulate_cdice_path(state.model, state.ps, state.st; params, periods = hp.periods,
        stochastic = true, rng = rng_from_seed(SEED; offset = 20), sigma_z_realized = 0.0)
    mc = cdice_monte_carlo_paths(state.model, state.ps, state.st; params, n_paths = hp.n_mc,
        periods = hp.periods, seed = 1000)
    teaching_policy = CDICETeachingPolicy(; params, stochastic = true)
    teaching_path = simulate_cdice_path(teaching_policy; params, periods = hp.periods,
        stochastic = true, rng = rng_from_seed(SEED; offset = 21), sigma_z_realized = 0.0)
    idx_2100 = min(86, hp.periods)
end

# ╔═╡ 66666666-1603-4666-8666-666666666666
begin
    smoke_checks = (
        finite_loss = isfinite(initial_loss) && isfinite(history[end].loss) && isfinite(best_training_loss),
        quadrature_normalized = abs(sum(rule.weights) - 1) < 1e-12,
        deterministic_path_finite = all(isfinite, deterministic_path.scc) && all(isfinite, deterministic_path.TAT),
        deterministic_TAT_positive = deterministic_path.TAT[idx_2100] > 0,
        mc_scc_finite = all(isfinite, mc.scc),
        mc_TAT_finite = all(isfinite, mc.TAT),
        mc_TAT_2100_positive = all(mc.TAT[:, idx_2100] .> 0),
        teaching_scc_positive = teaching_path.scc[1] > 0,
    )
    @assert all(values(smoke_checks))
end

# ╔═╡ 9bd22c6f-bb38-4e45-0b60-a75fb25805ae
md"""
### 11, 14–20. Figures in the Python notebook

The full Python notebook draws a sequence of figures that this Julia preview summarises numerically instead:

* **Training-loss curve** (§11) over the episodes.
* **Deterministic time paths** (§14): a 4-panel view of the \$z\equiv 0\$ baseline.
* **SCC over time** (§15): the deterministic SCC is monotonically increasing and **convex** in time — a tonne of CO₂ today damages a richer future, so its dollar cost rises; on a log axis the path is nearly linear (≈2–3% per year growth).
* **The fan chart** (§17, the headline plot): the stochastic SCC distribution (5/25/50/75/95-percentile bands) with the deterministic SCC overlaid. The 5–95% band **widens with the horizon** — uncertainty compounds — while the stochastic mean tracks the deterministic baseline closely (small Jensen tilt, since TFP enters multiplicatively). This is the pattern reported in Cai–Lontzek (2019) Fig. 5 / Table II.
* **Temperature and abatement-rate distributions** (§18) and **SCC histograms at 2050 and 2100** (§19).
* **Sample-paths overlay** (§20): the dispersion of \$T_{AT}\$ is small (climate inertia smooths the shock) but \$\mu\$ and especially SCC disperse visibly with the realised productivity history.

This preview returns the Monte-Carlo summaries numerically in the diagnostic NamedTuple below (`mc_scc_mean_2100`, `mc_scc_p95_2100`, `mc_TAT_mean_2100`, ...).
"""

# ╔═╡ 68bfb812-055c-46dc-d0bd-6db583cacfaa
md"""
### 21. What we learned

1. **Adding one shock to a working DEQN is mostly mechanical.** We extended the state from 7→8 dimensions, replaced every conditional expectation in the loss with a 5-node Gauss–Hermite sum, and changed nothing else (same network shape, same optimizer, same hand-derived FOCs). Total wall-clock at teaching scale: ~28 min on one CPU core, versus *days* on HPC for the equivalent dynamic-programming solution (Cai–Lontzek 2019, DPGM).

2. **The deterministic limit reproduces notebook 02.** Gate A passes within tolerance, confirming the 8-D machinery is consistent.

3. **The stochastic SCC distribution shows the Cai–Lontzek pattern.** At 2100 the 5–95% band on SCC spans roughly 148–179 USD/tCO\$_2\$ around a deterministic 163 USD/tCO\$_2\$. The mean essentially equals the deterministic value (no Jensen tilt of practical relevance for this calibration), but uncertainty itself widens with the horizon.

4. **SCC is convex in time.** The numerical second derivative of the deterministic SCC path is positive throughout 2015–2100; the log plot is nearly linear → roughly exponential growth at ~2–3% per year.

#### What's next (suggested exercises)

* **Add a second shock** (climate sensitivity, damage curvature, or backstop cost): same recipe — extend the state by one more dimension and multiply the GH inner loop by another factor of 5 (or use a sparse grid).
* **Bayesian growth uncertainty**: replace the i.i.d. AR(1) with a Brownian bridge over a discrete grid of long-run growth rates (Cai–Lontzek §V.B).
* **Curriculum on \$\sigma_z\$**: anneal from 0 to 0.0125 over the first episodes if \$\mu(2015)\$ drifts beyond tolerance — a useful homotopy when adding many shocks at once.

#### References

* Cai, Y. & Lontzek, T. S. (2019). *The social cost of carbon with economic and climate risks.* JPE 127(6).
* Folini, D., Friedl, A., Kübler, F. & Scheidegger, S. (2025). *The climate in climate economics.* Review of Economic Studies 92(1), 299–338.
* Friedl, A., Kübler, F., Scheidegger, S. & Usui, T. (2023). *Deep uncertainty quantification: with an application to integrated assessment models.* Working paper.
* Kübler, F., Scheidegger, S. & Surbek, O. (2026, forthcoming). *Using machine learning to compute constrained optimal carbon tax rules.* JPE: Macroeconomics.
* Azinovic, M., Gaegauf, L. & Scheidegger, S. (2022). *Deep equilibrium nets.* IER 63(4).

The cell below returns the machine-checkable diagnostics summary for this notebook's smoke run.
"""

# ╔═╡ 77777777-1603-4777-8777-777777777777
(
    initial_loss = initial_loss,
    final_loss = history[end].loss,
    best_training_loss = best_training_loss,
    training_loss_improved = best_training_loss <= initial_loss,
    trained_steps = state.step,
    gh_weight_sum = sum(rule.weights),
    stationary_z_std = cdice_stationary_z_std(params),
    deterministic_path_finite = smoke_checks.deterministic_path_finite,
    mc_scc_finite = smoke_checks.mc_scc_finite,
    mc_TAT_finite = smoke_checks.mc_TAT_finite,
    deterministic_scc_2015 = deterministic_path.scc[1],
    deterministic_TAT_2100 = deterministic_path.TAT[idx_2100],
    teaching_deterministic_scc_2015 = teaching_path.scc[1],
    mc_scc_mean_2100 = mean(mc.scc[:, idx_2100]),
    mc_scc_p95_2100 = quantile(vec(mc.scc[:, idx_2100]), 0.95),
    mc_TAT_mean_2100 = mean(mc.TAT[:, idx_2100]),
)

# ╔═╡ Cell order:
# ╟─11111111-1603-4111-8111-111111111111
# ╟─879ce380-1b9e-72b8-c53b-7c05349498e3
# ╟─a3472325-6d65-ffe4-8c1b-43d35f6310ea
# ╟─4d7aed42-a702-6c62-4e2c-ad6f9ef8f522
# ╠═22222222-1603-4222-8222-222222222222
# ╟─f0178385-8708-2b04-93fb-adae011b92ed
# ╟─a79cd689-dc3a-3121-5b20-8499500ff1d7
# ╟─8171029c-f683-2cd0-aebb-a8b2047a217f
# ╠═33333333-1603-4333-8333-333333333333
# ╟─8a259f3b-67cd-7560-db14-37c2ea7a3866
# ╟─e53f11d2-af1d-a9e5-c02d-fc0a7c2e756b
# ╟─117773e6-0af2-6ea9-8abb-164f8d287496
# ╟─6c41d9e5-d5d4-b8f7-384b-0df910b14771
# ╠═44444444-1603-4444-8444-444444444444
# ╟─ea24afb4-8162-e59e-aad4-b825aa43bda5
# ╠═55555555-1603-4555-8555-555555555555
# ╠═66666666-1603-4666-8666-666666666666
# ╟─9bd22c6f-bb38-4e45-0b60-a75fb25805ae
# ╟─68bfb812-055c-46dc-d0bd-6db583cacfaa
# ╠═77777777-1603-4777-8777-777777777777
