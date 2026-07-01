### A Pluto.jl notebook ###
# v0.20.0

using Markdown
using InteractiveUtils

# ╔═╡ 11111111-1601-4111-8111-111111111111
md"""
# Lecture 16, Notebook 01: DICE Climate Exercise in Julia

This Pluto translation keeps the original NumPy warm-up as a pure Julia
calculation: a three-box carbon cycle, a one-layer temperature response, and
quadratic damages under BAU and 50 percent mitigation emissions paths.
"""

# ╔═╡ d2643d55-9831-27b3-2a69-d9238d11b088
md"""
## Lecture 16, Notebook 01: DICE carbon cycle and climate damages — a Julia warm-up

**Course:** Deep Learning for Solving and Estimating Dynamic Models in Economics and Finance
**Script reference:** §11.1-11.2 (IAMs and DICE), §11.3 (DICE with DEQNs)
**Notebook role:** exercise
**Author:** Simon Scheidegger

*Julia/Lux/Pluto preview of* `lectures/lecture_16_climate_economics_iams/code/lecture_16_01_Climate_Exercise.ipynb`.

> **Run mode.** This is a pure deterministic calculation — no neural network — so `RUN_MODE = "smoke"` and `SEED = 0` only gate the sanity checks; the physics is identical across run modes.
"""

# ╔═╡ 5a35af53-7d59-a2c0-5566-521a5e154bf1
md"""
In this exercise we simulate the **DICE carbon cycle** and compute **climate damages** under different emission scenarios.

**Goals (approx. 30 min):**
1. Simulate the 3-box carbon cycle model forward in time.
2. Compute the temperature response and economic damages.
3. Compare business-as-usual against a mitigation policy.

**No neural networks needed** — this is a pure Julia calculation that builds intuition for the climate module of DICE before we study the full CDICE/DEQN solution in notebook `02_DICE_DEQN_Library_Port.ipynb`.
"""

# ╔═╡ 22222222-1601-4222-8222-222222222222
begin
    import Pkg
    Pkg.activate(joinpath(@__DIR__, "..", "..", "..", "julia"))

    using DLEFJulia
end

# ╔═╡ a87ff5a6-1cc6-5293-6c5a-a22d7f82863a
md"""
### Background

The **DICE model** (Nordhaus, 2017) tracks CO\$_2\$ concentrations in three reservoirs:

| Reservoir | Description |
|-----------|-------------|
| \$M_{AT}\$ | Atmosphere |
| \$M_{UO}\$ | Upper ocean / biosphere |
| \$M_{LO}\$ | Lower (deep) ocean |

The carbon cycle evolves according to a **linear transition**:

\$\$\begin{pmatrix} M_{AT}(t+1) \\ M_{UO}(t+1) \\ M_{LO}(t+1) \end{pmatrix} = \Phi \begin{pmatrix} M_{AT}(t) \\ M_{UO}(t) \\ M_{LO}(t) \end{pmatrix} + \begin{pmatrix} E(t) \\ 0 \\ 0 \end{pmatrix}\$\$

where \$\Phi\$ is the transfer matrix and \$E(t)\$ are anthropogenic emissions entering the atmosphere. Temperature responds to radiative forcing from atmospheric CO\$_2\$, and economic **damages** are a quadratic function of temperature.

Here the calibration lives in `DICEClimateParams()`: the transfer matrix \$\Phi\$, the radiative-forcing and temperature coefficients, and the quadratic damage coefficient \$\pi_2\$.
"""

# ╔═╡ 33333333-1601-4333-8333-333333333333
begin
    RUN_MODE = "smoke"
    SEED = 0
    params = DICEClimateParams()
end

# ╔═╡ 48f689ed-fdcc-bcb8-efd4-02d5d9ef51e9
md"""
### The DICE climate exercise

The Python notebook works through three timed tasks; the Julia preview runs all three in a single call to `simulate_dice_climate_exercise`, which iterates the carbon cycle and temperature forward over 30 five-year steps (150 years) under both a business-as-usual and a mitigation emissions path.

**Task 1 — carbon cycle.** Given the transition matrix \$\Phi\$ and an emissions path \$E(t)\$, iterate the three-box carbon cycle forward and track **atmospheric CO\$_2\$** (\$M_{AT}\$).

**Task 2 — temperature and damages.** Link atmospheric CO\$_2\$ to temperature and economic damages. The simplified single-layer temperature model is

\$\$T(t+1) = T(t) + \xi \left[ F(t) - \lambda \, T(t) \right] \cdot \Delta t,\$\$

where \$F(t) = \eta \cdot \log_2\!\left(\frac{M_{AT}(t)}{M_{AT,1750}}\right)\$ is the radiative forcing (W/m\$^2\$). The DICE quadratic damage function

\$\$\Omega(T) = \pi_2 \cdot T^2\$\$

gives the fraction of output lost to climate damages.

**Task 3 — policy comparison.** Compare business-as-usual (BAU) emissions with an aggressive mitigation scenario where emissions are halved, \$E_{\text{mitigate}}(t) = 0.5 \cdot E_{\text{BAU}}(t)\$, and read off the **avoided warming** and **avoided damages** at year 2100.

The returned `exercise` bundles both scenarios' carbon and temperature paths together with the avoided-warming and avoided-damage summaries used below (`mitigation_fraction = 0.5` sets the halving).
"""

# ╔═╡ 44444444-1601-4444-8444-444444444444
begin
    exercise = simulate_dice_climate_exercise(; params, mitigation_fraction = 0.5)
    idx_2100 = findfirst(==(2100.0), exercise.years)
end

# ╔═╡ 74e8866d-336a-8414-edc0-222620c3ed22
md"""
### Smoke-mode validation

The checks below confirm the run is physically sane: mitigation avoids some warming and some damages (both positive), and every carbon and temperature path is finite. These are the Julia counterpart of the Python notebook's smoke-mode assertions.
"""

# ╔═╡ 55555555-1601-4555-8555-555555555555
begin
    smoke_checks = (
        avoided_warming_positive = exercise.avoided_warming > 0,
        avoided_damages_positive = exercise.avoided_damages > 0,
        finite_temperature = all(isfinite, exercise.temperature_bau) &&
            all(isfinite, exercise.temperature_mitigation),
        finite_carbon = all(isfinite, exercise.carbon_bau) &&
            all(isfinite, exercise.carbon_mitigation),
    )
    @assert all(values(smoke_checks))
end

# ╔═╡ a99622cd-5216-bf6e-a53f-902c85b23d50
md"""
### Summary

**Key takeaways from this exercise:**

1. **The carbon cycle is linear in emissions:** atmospheric CO\$_2\$ accumulates proportionally to cumulative emissions, with slow diffusion into the ocean reservoirs.

2. **Temperature responds with a lag:** even after emissions slow, temperature keeps rising because the climate system has thermal inertia (\$\xi\$ is small).

3. **Damages are quadratic in temperature** (\$\Omega = \pi_2 T^2\$): this nonlinearity is crucial — doubling the temperature more than doubles the damages. It is this convexity that makes the cost of inaction grow rapidly.

4. **Mitigation has large long-run benefits:** halving emissions produces moderate near-term CO\$_2\$ reductions but significant long-run temperature and damage reductions, precisely because of the quadratic damage function.

5. **Connection to CDICE/DEQN:** in notebook `02_DICE_DEQN_Library_Port.ipynb` the social planner *optimally* chooses the mitigation rate \$\mu(t)\$ to balance abatement costs against avoided damages — solving the full intertemporal problem with neural networks.

The cell below returns a machine-checkable summary of this notebook's run.
"""

# ╔═╡ 66666666-1601-4666-8666-666666666666
(
    year_2100 = exercise.years[idx_2100],
    atmospheric_carbon_2015 = exercise.carbon_bau[1, 1],
    atmospheric_carbon_2100_bau = exercise.carbon_bau[1, idx_2100],
    atmospheric_carbon_2100_mitigation = exercise.carbon_mitigation[1, idx_2100],
    temperature_2100_bau = exercise.temperature_bau[idx_2100],
    temperature_2100_mitigation = exercise.temperature_mitigation[idx_2100],
    avoided_warming_2100 = exercise.avoided_warming,
    avoided_damages_2100 = exercise.avoided_damages,
    mean_atmospheric_carbon_gap = exercise.mean_atmospheric_carbon_gap,
)

# ╔═╡ Cell order:
# ╟─11111111-1601-4111-8111-111111111111
# ╟─d2643d55-9831-27b3-2a69-d9238d11b088
# ╟─5a35af53-7d59-a2c0-5566-521a5e154bf1
# ╠═22222222-1601-4222-8222-222222222222
# ╟─a87ff5a6-1cc6-5293-6c5a-a22d7f82863a
# ╠═33333333-1601-4333-8333-333333333333
# ╟─48f689ed-fdcc-bcb8-efd4-02d5d9ef51e9
# ╠═44444444-1601-4444-8444-444444444444
# ╟─74e8866d-336a-8414-edc0-222620c3ed22
# ╠═55555555-1601-4555-8555-555555555555
# ╟─a99622cd-5216-bf6e-a53f-902c85b23d50
# ╠═66666666-1601-4666-8666-666666666666
