# Lecture 04: IRBC with DEQNs

The first large-scale nonlinear DSGE application of DEQNs.

`gpu-recommended` · `long` · builds on [Lecture 03](../lecture_03_deep_equilibrium_nets/README.md)

> 📑 **Slides:** [lecture_04_irbc.pdf](slides/lecture_04_irbc.pdf)  
> 📓 **Notebooks:** [smooth benchmark](code/lecture_04_01_IRBC_DEQN_smooth.ipynb) · [irreversible investment](code/lecture_04_02_IRBC_DEQN_irreversible.ipynb)  
> 📚 **Further reading:** [curated list](../../readings/links_by_lecture/lecture_04.md)  
> 📖 **Script:** §Chapter 3 (International real business cycle)

## What this lecture covers

- **The IRBC model.** N symmetric countries with capital, country-specific productivity shocks, and risk-sharing through a complete bond market; equilibrium is N Euler equations plus a world resource constraint.
- **Why DEQNs scale here.** The state space is 2N-dimensional; classical methods scale poorly with N, DEQNs do not.
- **Solution and validation.** Train the DEQN, recover the symmetric steady state, and validate the policy via Euler-equation residuals along a simulated path.
- **Comparative statics.** Read off the effect of a parameter change (e.g. doubling depreciation) directly from the trained policy.

## Learning objectives

After this lecture you can:

- Set up the IRBC residual loss on a simulated state distribution.
- Train an N-country IRBC DEQN and recover the symmetric steady state.
- Run a comparative-statics exercise and read the result from the trained policy.
- Report Euler-equation residuals as a diagnostic across the simulated state distribution.

## Slides

- [`slides/lecture_04_irbc.pdf`](slides/lecture_04_irbc.pdf)
- [`slides/lecture_04_irbc.tex`](slides/lecture_04_irbc.tex)

## Code

### Julia/Lux/Pluto preview

- [`code_julia/lecture_04_01_IRBC_DEQN_smooth_Lux.jl`](code_julia/lecture_04_01_IRBC_DEQN_smooth_Lux.jl) translates the smooth IRBC residual and zero-shock diagnostic with a smoke-size Lux policy.
- [`code_julia/lecture_04_02_IRBC_DEQN_irreversible_Lux.jl`](code_julia/lecture_04_02_IRBC_DEQN_irreversible_Lux.jl) translates the irreversible-investment complementarity structure with Fischer-Burmeister residuals.

### Python notebooks

- [`code/lecture_04_01_IRBC_DEQN_smooth.ipynb`](code/lecture_04_01_IRBC_DEQN_smooth.ipynb) — smooth benchmark IRBC: persistent-simulation training, time-invariance and zero-shock steady-state diagnostics.
- [`code/lecture_04_02_IRBC_DEQN_irreversible.ipynb`](code/lecture_04_02_IRBC_DEQN_irreversible.ipynb) — irreversible-investment extension with KKT multipliers and a Fischer–Burmeister complementarity loss.

## Figures

- [`figures/irbc_4approach_loss.pdf`](figures/irbc_4approach_loss.pdf)
- [`figures/irbc_4approach_loss.png`](figures/irbc_4approach_loss.png)

## In the lecture script

§Chapter 3 (International real business cycle). The full chapter map is in [`script_to_lectures.md`](../../lecture_script/script_to_lectures.md).

## Readings

Curated bibliography for this lecture: [`lecture_04.md`](../../readings/links_by_lecture/lecture_04.md). The full BibTeX is in [`readings/bibliography.bib`](../../readings/bibliography.bib).

---

| ← Previous | Next → |
|---|---|
| [**Lecture 03: Deep Equilibrium Nets**](../lecture_03_deep_equilibrium_nets/README.md)<br><sub>Brock-Mirman (deterministic, stochastic), Fischer-Burmeister constraints, six loss kernels</sub> | [**Lecture 05: Architecture search and loss balancing**](../lecture_05_nas_loss_normalization/README.md)<br><sub>Random search, Hyperband, ReLoBRaLo, SoftAdapt, GradNorm</sub> |

[↑ Course map](../../COURSE_MAP.md)
