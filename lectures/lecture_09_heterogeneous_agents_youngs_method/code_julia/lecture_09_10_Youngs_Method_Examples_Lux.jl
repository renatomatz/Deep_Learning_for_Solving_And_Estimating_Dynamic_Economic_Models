### A Pluto.jl notebook ###
# v0.20.0

using Markdown
using InteractiveUtils

# ╔═╡ 11111111-0910-4111-8111-111111111111
md"""
# Lecture 09, Notebook 10: Young's Method Examples in Julia

This notebook isolates the deterministic histogram operator. The key checks are
mass conservation, mean preservation inside the grid, and explicit clipping at
boundaries.
"""

# ╔═╡ 4f3b0641-3f20-799c-93ba-9e82863fbb78
md"""
## Lecture 09, Notebook 10: Young's non-stochastic simulation by example

**Course:** Deep Learning for Solving and Estimating Dynamic Models in Economics and Finance
**Script reference:** §6.3 (Young's non-stochastic simulation)
**Notebook role:** core
**Author:** Simon Scheidegger

*Julia/Lux/Pluto preview of* `lectures/lecture_09_heterogeneous_agents_youngs_method/code/lecture_09_10_Youngs_Method_Examples.ipynb`.
"""

# ╔═╡ 805ad444-deb7-1528-3a01-0c629bb07b7e
md"""
## Young's (2010) Method to Simulate a Cross-Section

**Based on [Julien Pascal's notebook](https://julienpascal.github.io/post/young_2010/).** This Julia preview reuses the histogram operators from `DLEFJulia` instead of defining them inline.

**Reference:** Young, E.R. (2010). "Solving the incomplete markets model with aggregate uncertainty using the Krusell–Smith algorithm and non-stochastic simulations." *Journal of Economic Dynamics and Control*, 34(1), 36–41.

### Overview

Solving economic models with heterogeneous agents involves two key steps:
1. Finding the **optimal response of individuals** (policy functions)
2. **Simulating the model** to track how the cross-sectional distribution evolves

Young's (2010) method addresses step 2: instead of simulating a panel of agents (Monte Carlo), we simulate the **distribution directly** on a fixed grid — deterministically, with no sampling noise.

#### Pedagogical role in Day 4
This notebook isolates the **Young update operator itself**. The companion notebook `11_Continuum_of_Agents_DEQN.ipynb` then uses the **same histogram logic** inside the larger Appendix A.5 DEQN teaching model. So the move from this notebook (10) to notebook 11 is a change in the surrounding equilibrium architecture, **not** a change in the distribution update method.

#### This notebook demonstrates:
1. The core **mass redistribution** algorithm
2. Approximating a **single mass point**
3. Approximating a **multi-point distribution** (truncated normal)
4. **Mean preservation** property
5. Comparison with **panel-based simulation** (Monte Carlo)
6. **Convergence analysis**: how many agents does the panel approach need?
7. The conceptual bridge to the DEQN implementation in notebook 11
8. Reproducing the script's hand-worked 4-point Young update (Chapter 6, §6.3)

> **Run mode.** This preview uses small, fixed example sizes by design: the kit-mandated `RUN_MODE` constant is present only for consistency across the kit, and `SEED` is fixed at 0. The toy grids below are deterministic and do not depend on either constant.
"""

# ╔═╡ 22222222-0910-4222-8222-222222222222
begin
    import Pkg
    Pkg.activate(joinpath(@__DIR__, "..", "..", "..", "julia"))

    using DLEFJulia
    using LinearAlgebra
end

# ╔═╡ d1dd6ef5-f2ad-ef54-4080-db7e68fba451
md"""
### 1. The Method

#### Core idea

Choose a fixed grid \$[w_1, w_2, \ldots, w_N]\$ and represent the distribution as a **histogram** — a vector of masses at each grid point.

When a value \$w\$ falls between grid points \$w_n\$ and \$w_{n+1}\$, split its mass using **linear interpolation**:

\$\$p = 1 - \frac{w - w_n}{w_{n+1} - w_n}\$\$

- Mass assigned to \$w_n\$: \$m \times p\$
- Mass assigned to \$w_{n+1}\$: \$m \times (1 - p)\$

**Key property:** This preserves the mean exactly (when the grid is wide enough).

**Helper functions.** Where the Python notebook defines these operators inline ("2. Helper Functions"), the Julia preview imports them from `DLEFJulia`: `redistribute_mass` (split one off-grid mass point across its bracketing nodes), `redistribute_distribution` (project a whole source distribution onto the grid), `young_step` (one forward push of a histogram through a policy, optionally with a Markov shock transition), and the `young_mass` / `young_mean` diagnostics.
"""

# ╔═╡ 50314b61-0b06-a400-3dbf-3988358db397
md"""
### 3. Example 1: Single Mass Point

Consider a single mass point at \$w = 2.5\$ with mass \$m = 1.0\$. The grid has 10 points equally spaced on \$[0, 4]\$.

Since \$w = 2.5\$ is not a grid point, Young's method splits the mass between the two nearest grid points. The next cell also fixes `RUN_MODE` and `SEED`, builds the grid, and calls `redistribute_mass(grid, 2.5, 1.0)`, checking both total mass and mean via `young_mass` / `young_mean`.

#### What happened?

The true value \$w = 2.5\$ falls between grid points \$w_5 \approx 2.22\$ and \$w_6 \approx 2.67\$. The mass is split proportionally:

\$\$p = 1 - \frac{2.5 - 2.22}{2.67 - 2.22} \approx 0.375\$\$

So ~37.5% of the mass goes to \$w_5\$ and ~62.5% goes to \$w_6\$. The **mean is preserved exactly**.
"""

# ╔═╡ 33333333-0910-4333-8333-333333333333
begin
    RUN_MODE = "smoke"
    SEED = 0
    grid = collect(range(0.0, 4.0; length = 10))
    single = redistribute_mass(grid, 2.5, 1.0)
    single_check = (mass = young_mass(single), mean = young_mean(grid, single))
end

# ╔═╡ ef5b2596-f43c-569a-76ce-6fc4efc0e139
md"""
### 4. Example 2: Multi-Point Distribution (Truncated Normal)

Now consider a more realistic case: the true distribution is a truncated normal on \$[1, 3]\$ with mean 2. We evaluate it at 10 equally spaced points, then project it onto a **different** grid (10 points on \$[0, 4]\$) with `redistribute_distribution`.

In place of SciPy's `truncnorm`, the Julia preview evaluates the unnormalised Gaussian density \$\exp(-\tfrac{1}{2}(w-2)^2)\$ (mean 2, \$\sigma = 1\$) at the 10 source points and renormalises — a discrete truncated normal — then checks that the projection preserves both total mass and the mean.
"""

# ╔═╡ 44444444-0910-4444-8444-444444444444
begin
    sample_values = collect(range(1.0, 3.0; length = 10))
    masses = exp.(-0.5 .* (sample_values .- 2.0).^2)
    masses ./= sum(masses)
    approx = redistribute_distribution(grid, sample_values, masses)
    distribution_check = (
        mass = young_mass(approx),
        true_mean = sum(sample_values .* masses),
        young_mean = young_mean(grid, approx),
    )
end

# ╔═╡ 1c0e3471-513f-29eb-d312-62b3d291feb1
md"""
### The full Python notebook also covers

Between the projection examples and the dynamic application, the Python ground truth runs a set of experiments the Julia preview summarises rather than re-executes:

- **5. Grid refinement.** The mean is always preserved; higher moments (variance, median) improve as the grid is refined.
- **6.–8. Panel (Monte Carlo) simulation.** An alternative to the histogram is to draw a panel of \$N\$ agents from the distribution. Sampling error in the mean decays only as \$O(1/\sqrt{N})\$, so tens of thousands of agents are needed for accurate moments — whereas Young's method has **zero sampling error** for the mean.
- **9.–10. Side-by-side comparison.** A direct Young-vs-panel comparison and a summary table make the noise-free advantage explicit.

The economics is unchanged; only the deterministic histogram path is carried in Julia below.
"""

# ╔═╡ ae3f3574-03f9-3c52-b959-01362218b093
md"""
### 11. Dynamic Application: Forward Simulation with a Policy Function

In a heterogeneous agent model, we use Young's method to propagate the distribution **forward in time** using the agents' policy functions. Here we demonstrate with a simple savings policy.

Consider agents with wealth \$k\$ who save according to:
\$\$k' = g(k) = s \cdot k + (1 - s) \cdot \bar{k}\$\$

This is a linear policy that pushes wealth toward the mean \$\bar{k}\$ at rate \$(1-s)\$. The Julia cell uses \$s = 0.7\$, \$\bar{k} = 5\$ on a 50-point grid over \$[0, 10]\$, starting from a uniform block on \$[2, 8]\$, and advances the histogram 10 steps with `young_step`, recording the mean at each step.
"""

# ╔═╡ 55555555-0910-4555-8555-555555555555
begin
    dynamic_grid = collect(range(0.0, 10.0; length = 50))
    hist0 = zeros(length(dynamic_grid))
    hist0[findall(k -> 2.0 <= k <= 8.0, dynamic_grid)] .= 1.0
    hist0 ./= sum(hist0)
    toy_policy(k) = 0.7 * k + 0.3 * 5.0
    hists = [hist0]
    for _ in 1:10
        push!(hists, young_step(dynamic_grid, hists[end], toy_policy))
    end
    dynamic_means = [young_mean(dynamic_grid, h) for h in hists]
end

# ╔═╡ b5a7cdde-2b6e-9e86-b21a-f5923d87ccd5
md"""
### 12. With Idiosyncratic Shocks: Two Employment States

In a Krusell-Smith economy, agents face idiosyncratic employment shocks \$\varepsilon \in \{e, u\}\$. The histogram is now 2D: mass at each \$(k, \varepsilon)\$ pair. Young's method handles this by combining the policy-based mass redistribution with the Markov transition probabilities — exactly the `transition` keyword of `young_step`.
"""

# ╔═╡ e7d0aec9-9a40-d009-f7ef-02cdf036cc15
md"""
### 12b. Reproducing the Script's Worked 4-Point Example

Chapter 6, §6.3 walks through a hand-worked Young update on a tiny grid. We reproduce it here with a 2D `young_step` and check the resulting histogram and mean against the script.

**Setup.** Grid \$\{1,2,3,4\}\$, two idiosyncratic states \$\{\text{low},\text{high}\}\$, savings rule \$k' = 0.4\,k + 0.5\,y(\varepsilon)\$ with \$y_\text{low}=1\$, \$y_\text{high}=3\$, and an identity shock transition. The initial histogram \$G_0\$ has masses \$(0.10,0.20,0.10,0.05)\$ in the low row and \$(0.05,0.15,0.20,0.15)\$ in the high row, so \$\bar{k}_0 = 2.55\$. The script reports \$\bar{k}_1 = 2.08\$, with the unclipped policy-implied mean equal to \$2.07\$ (boundary clipping at \$k'=0.9\$ adds \$\approx 0.01\$).
"""

# ╔═╡ c07930cf-8aa7-4183-4ebf-98867d6e8850
md"""
### Also in the Python notebook: the cascading fork

Section 13 of the ground truth makes the two-stage cascade behind the 2D step explicit on a single source bin — the **capital lottery** (split an off-grid \$k'\$ between \$k_J\$ and \$k_{J+1}\$) composed with the **shock fork** (transition from \$\varepsilon\$ to \$\varepsilon'\$ via \$\pi_{\varepsilon\varepsilon'}\$), giving four destination leaves whose weights are the products of the two stages and sum to the source mass. This is the `fig:young_cascade` figure of §6.3 (Fig. 1 of Young 2010); the same composition is precisely what the 2D `young_step` above performs.
"""

# ╔═╡ 66666666-0910-4666-8666-666666666666
begin
    g4 = [1.0, 2.0, 3.0, 4.0]
    G0 = [0.10 0.20 0.10 0.05; 0.05 0.15 0.20 0.15]
    policy4 = [0.9 1.3 1.7 2.1; 1.9 2.3 2.7 3.1]
    G1 = young_step(g4, G0, policy4; transition = Matrix{Float64}(I, 2, 2))
    script_check = (mass = young_mass(G1), mean = young_mean(g4, G1), row_masses = vec(sum(G1; dims = 2)))
end

# ╔═╡ 8dc82294-37c3-ef68-0765-7a15a749439e
md"""
## Conclusion

This notebook demonstrated Young's (2010) non-stochastic simulation method:

1. **Core algorithm**: Linear interpolation to redistribute mass on a fixed grid
2. **Mean preservation**: The method preserves the mean of the distribution exactly
3. **Higher moments**: Variance and median are approximated; finer grids → better accuracy
4. **No sampling noise**: Unlike panel simulation, which requires tens of thousands of agents
5. **Forward simulation**: Can propagate distributions forward in time using policy functions
6. **Idiosyncratic shocks**: Naturally extends to multiple employment/income states

### Bridge to `11_Continuum_of_Agents_DEQN`
- The toy object `h(k)` becomes a stacked histogram over \$(\eta, b)\$ blocks
- The hand-set policy function becomes a policy network output \$b'(\eta, b, x_t^{agg})\$
- The weighted mean calculation becomes exact market clearing via the histogram dot product with network-implied savings

### Next steps
- Combine with value function iteration (or neural networks) to solve full heterogeneous agent models
- Use as the distribution propagation step inside DEQN training loops
- See the companion slides: `lecture_09_heterogeneous_agents_youngs.pdf`

### Reference
Young, E.R. (2010). "Solving the incomplete markets model with aggregate uncertainty using the Krusell–Smith algorithm and non-stochastic simulations." *Journal of Economic Dynamics and Control*, 34(1), 36–41.

The cell below returns the machine-checkable summary of this notebook's run.
"""

# ╔═╡ 77777777-0910-4777-8777-777777777777
(
    single = single_check,
    distribution = distribution_check,
    dynamic_initial_mean = first(dynamic_means),
    dynamic_final_mean = last(dynamic_means),
    script_four_point = script_check,
)

# ╔═╡ Cell order:
# ╟─11111111-0910-4111-8111-111111111111
# ╟─4f3b0641-3f20-799c-93ba-9e82863fbb78
# ╟─805ad444-deb7-1528-3a01-0c629bb07b7e
# ╠═22222222-0910-4222-8222-222222222222
# ╟─d1dd6ef5-f2ad-ef54-4080-db7e68fba451
# ╟─50314b61-0b06-a400-3dbf-3988358db397
# ╠═33333333-0910-4333-8333-333333333333
# ╟─ef5b2596-f43c-569a-76ce-6fc4efc0e139
# ╠═44444444-0910-4444-8444-444444444444
# ╟─1c0e3471-513f-29eb-d312-62b3d291feb1
# ╟─ae3f3574-03f9-3c52-b959-01362218b093
# ╠═55555555-0910-4555-8555-555555555555
# ╟─b5a7cdde-2b6e-9e86-b21a-f5923d87ccd5
# ╟─e7d0aec9-9a40-d009-f7ef-02cdf036cc15
# ╟─c07930cf-8aa7-4183-4ebf-98867d6e8850
# ╠═66666666-0910-4666-8666-666666666666
# ╟─8dc82294-37c3-ef68-0765-7a15a749439e
# ╠═77777777-0910-4777-8777-777777777777
