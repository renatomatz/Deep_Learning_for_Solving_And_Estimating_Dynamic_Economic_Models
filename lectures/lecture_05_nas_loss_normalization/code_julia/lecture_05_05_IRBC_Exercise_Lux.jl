### A Pluto.jl notebook ###
# v0.20.0

using Markdown
using InteractiveUtils

# ╔═╡ 11111111-0505-4111-8111-111111111111
md"""
# Lecture 05, Notebook 05: IRBC Loss-Balancing Exercise

This exercise notebook intentionally keeps TODO prompts. Students use the shared
IRBC residuals and the loss-balancing helpers from Notebook 04 to build a
balanced multi-residual DEQN training objective.
"""

# ╔═╡ 5c0c2127-33e4-7ca4-50b7-1cd5dccdf3b6
md"""
## Lecture 05, Notebook 05: IRBC Exercise — Comparative Statics and Loss Weighting

**Course:** Deep Learning for Solving and Estimating Dynamic Models in Economics and Finance
**Script reference:** Chapter 3, Exercises 3.6 (steady-state comparative statics) and 3.7 (inverse-loss weighting); see also §4.8 (loss normalization)
**Notebook role:** exercise
**Author:** Simon Scheidegger

*Julia/Lux/Pluto preview of* `lectures/lecture_05_nas_loss_normalization/code/lecture_05_05_IRBC_Exercise.ipynb`.

> **Exercise notebook.** The code cells below intentionally keep **TODO** prompts and a `blanks_are_intentional = true` flag — this is a student exercise, not a worked solution. Fill them in using the shared IRBC residuals and the loss-balancing helpers from Notebook 04.
"""

# ╔═╡ e1f53714-b8cd-16f1-5579-4ddda758415c
md"""
In this exercise you work with the **2-country International Real Business Cycle (IRBC)** model from Lecture 04 (`lecture_04_01_IRBC_DEQN_smooth`). Rather than re-training the full model, the Python exercise runs three short tasks:

1. **Task 1 (comparative statics):** change one parameter at a time (\$\beta\$, \$\delta\$, \$\zeta\$) and observe how steady-state capital and consumption respond.
2. **Task 2 (inverse-loss weighting):** implement inverse-loss weighting for a multi-component loss (Euler equations, resource constraint, Fischer-Burmeister complementarity).
3. **Task 3 (training comparison):** compare synthetic equal-weight vs. inverse-weight training histories and quantify the convergence speedup.

This Julia/Lux preview is a compact version of **Task 2**: it builds the same multi-residual loss-balancing objective directly on the shared `irbc_smooth_residual` and the `DLEFJulia` weighting helpers (`equal_loss_weights`, `simplex_inverse_loss_weights`, `relobralo_weights`, `softadapt_weights`). Loss-normalization theory is in Notebook 04.
"""

# ╔═╡ 22222222-0505-4222-8222-222222222222
begin
    import Pkg
    Pkg.activate(joinpath(@__DIR__, "..", "..", "..", "julia"))

    using DLEFJulia
end

# ╔═╡ 33333333-0505-4333-8333-333333333333
begin
    RUN_MODE = "smoke"
    SEED = 0
    hp = run_mode_budget(RUN_MODE; budgets = (
        smoke = (batch_size = 16,),
        teaching = (batch_size = 64,),
        production = (batch_size = 128,),
    ))
    rng = rng_from_seed(SEED)
end

# ╔═╡ 44444444-0505-4444-8444-444444444444
md"""
## Exercise (Task 2 — loss balancing on IRBC residuals)

When training the IRBC model with Deep Equilibrium Nets, the total loss is a sum of several components:

\$\$\mathcal{L} = w_1 L_{\text{Euler}_1} + w_2 L_{\text{Euler}_2} + w_3 L_{\text{ARC}} + w_4 L_{\text{FB}_1} + w_5 L_{\text{FB}_2}.\$\$

These can differ by **orders of magnitude** (e.g. \$L_{\text{ARC}} \sim 5\$ while \$L_{\text{FB}} \sim 10^{-4}\$). With equal weights (\$w_i = 1\$) the optimizer chases the largest loss and ignores small but important components. **Inverse-loss weighting** (a simplified ReLoBRaLo idea) restores balance:

\$\$w_i = \frac{1/L_i}{\sum_j 1/L_j}.\$\$

**Your turn.** Fill in the TODOs in the next cell:

1. Evaluate `irbc_smooth_residual` on a batch of states.
2. Split its components into Euler and resource residual losses.
3. Replace equal weights with `simplex_inverse_loss_weights`, `relobralo_weights`, or `softadapt_weights`.
4. Compare whether the balanced objective lowers the largest component.

The next cell already computes example weights on the toy component losses `[1.0, 100.0, 0.01]` so you can see each scheme rebalancing toward the large-scale component; the `balanced_irbc_loss` function itself is left as a TODO.
"""

# ╔═╡ 55555555-0505-4555-8555-555555555555
begin
    params = IRBCParams(countries = 2)
    states = irbc_sample_states(rng, params, hp.batch_size)
    example_component_losses = [1.0, 100.0, 0.01]
    equal_weights = equal_loss_weights(example_component_losses)
    inverse_weights = length(example_component_losses) .* simplex_inverse_loss_weights(example_component_losses)
    relobralo_example = relobralo_weights([0.8, 80.0, 0.02], example_component_losses; temperature = 0.5)

    # TODO: define balanced_irbc_loss(model, ps, st, states, weights).
    balanced_irbc_loss_todo = "TODO: call irbc_smooth_residual, compute component losses, apply weights, return scalar loss and st_new"
end

# ╔═╡ a9e7af53-41c3-69de-59d0-d6c5bee77f87
md"""
### The full Python exercise also covers

The Python ground truth wraps this loss-balancing task in two more, each with a *Your Turn* blank and a hidden *Solution*:

**Task 1 — comparative statics.** At the deterministic steady state (\$z^j = 0\$) the Euler equation gives \$\text{MPK} = 1/\beta\$, so

\$\$k_{ss} = \left(\frac{1/\beta - 1 + \delta}{\zeta\,A_{\text{tfp}}}\right)^{1/(\zeta - 1)}, \qquad c_{ss} = A_{\text{tfp}}\,k_{ss}^{\zeta} - \delta\,k_{ss}.\$\$

Students vary \$\beta\$, \$\delta\$, \$\zeta\$ one at a time and predict the sign of the change in \$k_{ss}\$ and \$c_{ss}\$ before computing it. (The exercise uses a standalone \$A_{\text{tfp}} = 1\$ calibration for readable numbers; Lecture 04 instead pins \$A_{\text{tfp}} = (1-\beta(1-\delta))/(\zeta\beta) \approx 0.0559\$ so that \$k_{ss} = 1\$.)

**Task 3 — training comparison.** Given synthetic equal-weight (A) vs. inverse-weight (B) loss histories, plot both on a semilog axis and compute the **speedup ratio**: the epoch at which A first reaches B's final loss.
"""

# ╔═╡ 3f03aa57-fbfd-2ab6-bf0b-cec0047bc40f
md"""
### Summary

1. **Parameter sensitivity builds economic intuition** — the steady-state formulas show how patience (\$\beta\$), depreciation (\$\delta\$), and capital share (\$\zeta\$) set long-run capital and consumption.
2. **Loss weighting is critical for multi-component objectives** — the IRBC Euler, resource-constraint, and Fischer-Burmeister residuals differ by orders of magnitude, and equal weighting lets the largest dominate the gradient.
3. **Inverse-loss weighting typically speeds up convergence by 2–3×** by giving every equilibrium condition adequate attention — a simplified version of ReLoBRaLo (Bischof & Kraus, 2025; full implementation in Notebook 04).

The cell below returns the exercise's current state, including `blanks_are_intentional = true` to flag that the TODOs are meant to be completed by the student.
"""

# ╔═╡ 66666666-0505-4666-8666-666666666666
(
    state_shape = size(states),
    equal_weights = equal_weights,
    inverse_weights = inverse_weights,
    relobralo_example = relobralo_example,
    weight_sums = (equal = sum(equal_weights), inverse = sum(inverse_weights), relobralo = sum(relobralo_example)),
    todo = balanced_irbc_loss_todo,
    blanks_are_intentional = true,
)

# ╔═╡ Cell order:
# ╟─11111111-0505-4111-8111-111111111111
# ╟─5c0c2127-33e4-7ca4-50b7-1cd5dccdf3b6
# ╟─e1f53714-b8cd-16f1-5579-4ddda758415c
# ╠═22222222-0505-4222-8222-222222222222
# ╠═33333333-0505-4333-8333-333333333333
# ╟─44444444-0505-4444-8444-444444444444
# ╠═55555555-0505-4555-8555-555555555555
# ╟─a9e7af53-41c3-69de-59d0-d6c5bee77f87
# ╟─3f03aa57-fbfd-2ab6-bf0b-cec0047bc40f
# ╠═66666666-0505-4666-8666-666666666666
