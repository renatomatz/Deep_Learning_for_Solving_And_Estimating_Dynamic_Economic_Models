# Lecture 07: Automatic differentiation for DEQNs

The automatic-differentiation machinery that DEQN training depends on, made explicit.

`cpu-standard` · `standard` · builds on [Lecture 03](../lecture_03_deep_equilibrium_nets/README.md)

> 📑 **Slides:** [lecture_07_autodiff_for_deqns.pdf](slides/lecture_07_autodiff_for_deqns.pdf)  
> 📓 **Notebooks:** [start here](code/lecture_07_01_AutoDiff_Analytical_Examples.ipynb) (4 in [`code/`](code/))  
> 📚 **Further reading:** [curated list](../../readings/links_by_lecture/lecture_07.md)  
> 📖 **Script:** §2.7 (Automatic differentiation), §Appendix B (Matrix calculus)

## What this lecture covers

- **Lagrangian primitives.** Deriving a single per-agent primitive Π whose partial derivatives give every Euler-equation residual.
- **Two-tape autodiff.** Recovering each gradient with two `tf.GradientTape` (or equivalent) calls per Euler equation.
- **Cross-checking.** A machine-precision comparison of autodiff residuals against hand-derived residuals on Brock-Mirman.
- **Lifting to IRBC.** Applying the same template to the multi-country setup of the previous lecture.
- **Common pitfalls.** Graph mode vs eager, dtype, in-place ops, and what to do when gradients silently disappear.

## Learning objectives

After this lecture you can:

- Derive a Lagrangian primitive analytically for a small recursive problem.
- Implement a two-tape autodiff Euler residual and verify it against the closed-form derivative.
- Apply the same template to deterministic and stochastic Brock-Mirman, and to multi-country IRBC.
- Diagnose autodiff numerical issues (graph mode vs eager, dtype, in-place ops).

## Slides

- [`slides/lecture_07_autodiff_for_deqns.pdf`](slides/lecture_07_autodiff_for_deqns.pdf)
- [`slides/lecture_07_autodiff_for_deqns.tex`](slides/lecture_07_autodiff_for_deqns.tex)

## Code

### Julia/Lux preview

- [`code_julia/lecture_07_01_AutoDiff_Analytical_Examples_Lux.ipynb`](code_julia/lecture_07_01_AutoDiff_Analytical_Examples_Lux.ipynb)
- [`code_julia/lecture_07_02_Brock_Mirman_AutoDiff_DEQN_Lux.ipynb`](code_julia/lecture_07_02_Brock_Mirman_AutoDiff_DEQN_Lux.ipynb)
- [`code_julia/lecture_07_03_Brock_Mirman_Uncertainty_AutoDiff_DEQN_Lux.ipynb`](code_julia/lecture_07_03_Brock_Mirman_Uncertainty_AutoDiff_DEQN_Lux.ipynb)
- [`code_julia/lecture_07_04_IRBC_AutoDiff_DEQN_Lux.ipynb`](code_julia/lecture_07_04_IRBC_AutoDiff_DEQN_Lux.ipynb)

### Python notebooks

- [`code/lecture_07_01_AutoDiff_Analytical_Examples.ipynb`](code/lecture_07_01_AutoDiff_Analytical_Examples.ipynb)
- [`code/lecture_07_02_Brock_Mirman_AutoDiff_DEQN.ipynb`](code/lecture_07_02_Brock_Mirman_AutoDiff_DEQN.ipynb)
- [`code/lecture_07_03_Brock_Mirman_Uncertainty_AutoDiff_DEQN.ipynb`](code/lecture_07_03_Brock_Mirman_Uncertainty_AutoDiff_DEQN.ipynb)
- [`code/lecture_07_04_IRBC_AutoDiff_DEQN.ipynb`](code/lecture_07_04_IRBC_AutoDiff_DEQN.ipynb)

## In the lecture script

§2.7 (Automatic differentiation), §Appendix B (Matrix calculus). The full chapter map is in [`script_to_lectures.md`](../../lecture_script/script_to_lectures.md).

## Readings

Curated bibliography for this lecture: [`lecture_07.md`](../../readings/links_by_lecture/lecture_07.md). The full BibTeX is in [`readings/bibliography.bib`](../../readings/bibliography.bib).

---

| ← Previous | Next → |
|---|---|
| [**Lecture 06: Agentic programming**](../lecture_06_agentic_programming/README.md)<br><sub>Claude Code workflow, prompts, project memory, custom skills, subagents, hooks, 12-exercise workshop</sub> | [**Lecture 08: OLG models with DEQNs**](../lecture_08_olg_models_deqns/README.md)<br><sub>Analytic OLG, 56-cohort benchmark, Fischer-Burmeister borrowing constraints</sub> |

[↑ Course map](../../COURSE_MAP.md)
