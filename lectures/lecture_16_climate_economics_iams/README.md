# Lecture 16: Climate economics and integrated assessment models

Integrated assessment models (IAMs), the canonical climate-economy framework: DICE and CDICE.

`gpu-recommended` · `long` · builds on [Lecture 07](../lecture_07_autodiff_for_deqns/README.md)

> 📑 **Slides:** [lecture_16_climate_economics_iams.pdf](slides/lecture_16_climate_economics_iams.pdf)  
> 📓 **Notebooks:** [start here](code/lecture_16_01_Climate_Exercise.ipynb) (3 in [`code/`](code/))  
> 📚 **Further reading:** [curated list](../../readings/links_by_lecture/lecture_16.md)  
> 📖 **Script:** §11.1-11.2 (IAMs and DICE), §11.3-11.8 (CDICE-DEQN, deterministic and stochastic)

## What this lecture covers

- **The IAM building blocks.** A macro-growth block, a carbon cycle, temperature dynamics, and a damage function that ties climate back to output.
- **DICE and CDICE.** The Nordhaus DICE benchmark and the calibrated CDICE extension (Folini et al. 2025, *Review of Economic Studies*).
- **Carbon-cycle simulation.** Business-as-usual and a mitigation scenario; reading off the social cost of carbon.
- **Deterministic CDICE-DEQN.** Solve CDICE with a DEQN and verify against the production-code reference.
- **Stochastic CDICE-DEQN.** Add AR(1) productivity shocks and use Gauss-Hermite quadrature for the conditional expectations.

## Learning objectives

After this lecture you can:

- Simulate the DICE carbon cycle and temperature dynamics under business-as-usual and a mitigation scenario.
- Read off the social cost of carbon and connect IAM building blocks to climate science.
- Solve deterministic CDICE with a DEQN and verify against the reference.
- Extend to stochastic CDICE with AR(1) productivity shocks.

## Slides

- [`slides/lecture_16_climate_economics_iams.pdf`](slides/lecture_16_climate_economics_iams.pdf)
- [`slides/lecture_16_climate_economics_iams.tex`](slides/lecture_16_climate_economics_iams.tex)

## Code

- [`code/lecture_16_01_Climate_Exercise.ipynb`](code/lecture_16_01_Climate_Exercise.ipynb)
- [`code/lecture_16_02_DICE_DEQN_Library_Port.ipynb`](code/lecture_16_02_DICE_DEQN_Library_Port.ipynb)
- [`code/lecture_16_03_Stochastic_DICE_DEQN.ipynb`](code/lecture_16_03_Stochastic_DICE_DEQN.ipynb)


### Julia/Lux/Pluto preview

- [`code_julia/lecture_16_01_Climate_Exercise_Lux.jl`](code_julia/lecture_16_01_Climate_Exercise_Lux.jl) translates the DICE carbon-cycle, temperature, and mitigation warm-up into pure Julia.
- [`code_julia/lecture_16_02_DICE_DEQN_Library_Port_Lux.jl`](code_julia/lecture_16_02_DICE_DEQN_Library_Port_Lux.jl) preserves the deterministic CDICE calibration, policy transforms, residual equations, and teaching-policy SCC diagnostics with a Lux smoke pass.
- [`code_julia/lecture_16_03_Stochastic_DICE_DEQN_Lux.jl`](code_julia/lecture_16_03_Stochastic_DICE_DEQN_Lux.jl) adds the AR(1) productivity state, Gauss-Hermite expectation, and small Monte Carlo fan-chart inputs.

## In the lecture script

§11.1-11.2 (IAMs and DICE), §11.3-11.8 (CDICE-DEQN, deterministic and stochastic). The full chapter map is in [`script_to_lectures.md`](../../lecture_script/script_to_lectures.md).

## Readings

Curated bibliography for this lecture: [`lecture_16.md`](../../readings/links_by_lecture/lecture_16.md). The full BibTeX is in [`readings/bibliography.bib`](../../readings/bibliography.bib).

---

| ← Previous | Next → |
|---|---|
| [**Lecture 15: Structural estimation via SMM**](../lecture_15_structural_estimation_smm/README.md)<br><sub>Brock-Mirman SMM (single- and joint-parameter) on a deep surrogate</sub> | [**Lecture 17: Deep UQ and Pareto-improving carbon-tax design**](../lecture_17_deep_uq_pareto_improving_policy/README.md)<br><sub>GP surrogates, Bayesian active learning, Sobol / Shapley, constrained Pareto-improving rules</sub> |

[↑ Course map](../../COURSE_MAP.md)
