# Lecture 08: OLG models with DEQNs

Overlapping-generations (OLG) models with DEQNs, at two scales.

`gpu-recommended` · `long` · builds on [Lecture 07](../lecture_07_autodiff_for_deqns/README.md)

> 📑 **Slides:** [lecture_08_olg_models_deqns.pdf](slides/lecture_08_olg_models_deqns.pdf)  
> 📓 **Notebooks:** [start here](code/lecture_08_08_OLG_Analytic_DEQN_persistent.ipynb) (5 in [`code/`](code/))  
> 📚 **Further reading:** [curated list](../../readings/links_by_lecture/lecture_08.md)  
> 📖 **Script:** §5.1-5.5 (OLG with DEQNs), §5.6 (Large OLG benchmark)

## What this lecture covers

- **Cohort structure.** One Euler equation per cohort, stacked into a single Lagrangian primitive; the DEQN training principle does not change.
- **Analytic small OLG.** A closed-form lifecycle savings model used as a sanity check on the DEQN solution.
- **The 56-period benchmark.** The standard production-scale OLG model with borrowing constraints.
- **Borrowing constraints.** Product-form KKT complementarity used cohort-by-cohort to handle the inequalities (softplus heads for non-negativity, squared product residuals `(λ·k')²` in the loss).
- **Diagnostics.** Lifecycle profiles, aggregate dynamics, and equilibrium residuals across cohorts.

## Learning objectives

After this lecture you can:

- Write the cohort-stacked Lagrangian for an OLG DEQN.
- Train an analytic small-OLG DEQN and verify lifecycle savings against the closed form.
- Reproduce the 56-period OLG benchmark with borrowing and collateral constraints via product-form KKT residuals.
- Read off lifecycle profiles, aggregate dynamics, and equilibrium residuals across cohorts.

## Slides

- [`slides/lecture_08_olg_models_deqns.pdf`](slides/lecture_08_olg_models_deqns.pdf)
- [`slides/lecture_08_olg_models_deqns.tex`](slides/lecture_08_olg_models_deqns.tex)

## Code

### Julia/Lux preview

- [`code_julia/lecture_08_07_OLG_Analytic_DEQN_exogenous_Lux.ipynb`](code_julia/lecture_08_07_OLG_Analytic_DEQN_exogenous_Lux.ipynb) translates the analytic exogenous-cloud ablation with the shared OLG residual helper.
- [`code_julia/lecture_08_08_OLG_Analytic_DEQN_persistent_Lux.ipynb`](code_julia/lecture_08_08_OLG_Analytic_DEQN_persistent_Lux.ipynb) translates the analytic persistent-simulation variant and validates against closed-form savings rates.
- [`code_julia/lecture_08_09_OLG_Benchmark_DEQN_exogenous_Lux.ipynb`](code_julia/lecture_08_09_OLG_Benchmark_DEQN_exogenous_Lux.ipynb) translates the benchmark exogenous-cloud variant with smoke-mode cohort reduction.
- [`code_julia/lecture_08_10_OLG_Benchmark_DEQN_persistent_Lux.ipynb`](code_julia/lecture_08_10_OLG_Benchmark_DEQN_persistent_Lux.ipynb) translates the benchmark persistent-simulation variant with borrowing and collateral transforms.
- [`code_julia/lecture_08_11_OLG_Exercise_Lux.ipynb`](code_julia/lecture_08_11_OLG_Exercise_Lux.ipynb) translates the OLG exercise with closed-form savings and lifecycle diagnostics.

### Python notebooks

**Analytic 6-agent OLG** — closed-form validation target.
- [`code/lecture_08_08_OLG_Analytic_DEQN_persistent.ipynb`](code/lecture_08_08_OLG_Analytic_DEQN_persistent.ipynb) — primary classroom variant: persistent-simulation training, validation against the Krueger–Kübler closed-form savings rates.
- [`code/lecture_08_07_OLG_Analytic_DEQN_exogenous.ipynb`](code/lecture_08_07_OLG_Analytic_DEQN_exogenous.ipynb) — feedback-free ablation: training cloud drawn from broad exogenous boxes.

**Benchmark 56-agent OLG** — Azinovic–Gaegauf–Scheidegger (2022) production scale.
- [`code/lecture_08_10_OLG_Benchmark_DEQN_persistent.ipynb`](code/lecture_08_10_OLG_Benchmark_DEQN_persistent.ipynb) — primary classroom variant: persistent-simulation training on the model's ergodic set.
- [`code/lecture_08_09_OLG_Benchmark_DEQN_exogenous.ipynb`](code/lecture_08_09_OLG_Benchmark_DEQN_exogenous.ipynb) — feedback-free ablation: training cloud drawn from broad exogenous boxes.

**Student exercise.**
- [`code/lecture_08_11_OLG_Exercise.ipynb`](code/lecture_08_11_OLG_Exercise.ipynb)

## In the lecture script

§5.1-5.5 (OLG with DEQNs), §5.6 (Large OLG benchmark). The full chapter map is in [`script_to_lectures.md`](../../lecture_script/script_to_lectures.md).

## Readings

Curated bibliography for this lecture: [`lecture_08.md`](../../readings/links_by_lecture/lecture_08.md). The full BibTeX is in [`readings/bibliography.bib`](../../readings/bibliography.bib).

---

| ← Previous | Next → |
|---|---|
| [**Lecture 07: Automatic differentiation for DEQNs**](../lecture_07_autodiff_for_deqns/README.md)<br><sub>Lagrangian primitives, two-tape gradients, IRBC autodiff</sub> | [**Lecture 09: Heterogeneous agents and Young's method**](../lecture_09_heterogeneous_agents_youngs_method/README.md)<br><sub>Young's histogram, Krusell-Smith, continuum-of-agents DEQN</sub> |

[↑ Course map](../../COURSE_MAP.md)
