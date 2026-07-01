# Lecture 05: Architecture search and loss balancing

Two of the main hyperparameter-engineering tasks for DEQN training in practice: choosing an architecture, and balancing the multi-component residual loss.

`gpu-recommended` · `long` · builds on [Lecture 04](../lecture_04_irbc_with_deqns/README.md)

> 📑 **Slides:** [lecture_05_neural_architecture_search.pdf](slides/lecture_05_neural_architecture_search.pdf) and 1 more under [`slides/`](slides/)  
> 📓 **Notebooks:** [start here](code/lecture_05_02_NAS_Random_Search_10D.ipynb) (4 in [`code/`](code/))  
> 📚 **Further reading:** [curated list](../../readings/links_by_lecture/lecture_05.md)  
> 📖 **Script:** §Chapter 4 (Neural architecture search and loss normalization)

## What this lecture covers

- **Neural architecture search.** Random search and Hyperband (successive halving), implemented from scratch in pure Python.
- **A 10-D NAS problem.** Searching over depth, width, activation, and learning-rate decay on a DEQN benchmark.
- **Loss balancing.** Why different equilibrium equations on different scales kill training, and how to fix it.
- **Three balancing schemes.** ReLoBRaLo, SoftAdapt, and GradNorm compared head-to-head on the same multi-residual run.

## Learning objectives

After this lecture you can:

- Implement random search and Hyperband in pure Python.
- Run a 10-D NAS sweep on a DEQN problem and read off the winning architecture.
- Compute ReLoBRaLo loss weights by hand on a small example.
- Compare ReLoBRaLo, SoftAdapt, and GradNorm on a shared multi-residual training run.

## Slides

- [`slides/lecture_05_neural_architecture_search.pdf`](slides/lecture_05_neural_architecture_search.pdf)
- [`slides/lecture_05_neural_architecture_search.tex`](slides/lecture_05_neural_architecture_search.tex)
- [`slides/lecture_05_loss_normalization.pdf`](slides/lecture_05_loss_normalization.pdf)
- [`slides/lecture_05_loss_normalization.tex`](slides/lecture_05_loss_normalization.tex)

## Code

### Julia/Lux/Pluto preview

- [`code_julia/lecture_05_02_NAS_Random_Search_10D_Lux.jl`](code_julia/lecture_05_02_NAS_Random_Search_10D_Lux.jl) translates random search on a compact 10-D Lux approximation task.
- [`code_julia/lecture_05_03_NAS_RandomSearch_Hyperband_Lux.jl`](code_julia/lecture_05_03_NAS_RandomSearch_Hyperband_Lux.jl) translates successive halving without using or rewriting the Python pickle cache.
- [`code_julia/lecture_05_04_Loss_Normalization_Lux.jl`](code_julia/lecture_05_04_Loss_Normalization_Lux.jl) translates equal, inverse, ReLoBRaLo, and SoftAdapt-style loss weighting.
- [`code_julia/lecture_05_05_IRBC_Exercise_Lux.jl`](code_julia/lecture_05_05_IRBC_Exercise_Lux.jl) preserves the student IRBC loss-balancing TODO exercise.

### Python notebooks

- [`code/lecture_05_02_NAS_Random_Search_10D.ipynb`](code/lecture_05_02_NAS_Random_Search_10D.ipynb)
- [`code/lecture_05_03_NAS_RandomSearch_Hyperband.ipynb`](code/lecture_05_03_NAS_RandomSearch_Hyperband.ipynb)
- [`code/lecture_05_04_Loss_Normalization.ipynb`](code/lecture_05_04_Loss_Normalization.ipynb)
- [`code/lecture_05_05_IRBC_Exercise.ipynb`](code/lecture_05_05_IRBC_Exercise.ipynb)

## Figures

- [`figures/loss_norm_T_sensitivity.pdf`](figures/loss_norm_T_sensitivity.pdf)
- [`figures/loss_norm_T_sensitivity.png`](figures/loss_norm_T_sensitivity.png)
- [`figures/loss_norm_equal_errmap.pdf`](figures/loss_norm_equal_errmap.pdf)
- [`figures/loss_norm_equal_errmap.png`](figures/loss_norm_equal_errmap.png)
- [`figures/loss_norm_equal_weights.pdf`](figures/loss_norm_equal_weights.pdf)
- [`figures/loss_norm_equal_weights.png`](figures/loss_norm_equal_weights.png)
- [`figures/loss_norm_method_comparison.pdf`](figures/loss_norm_method_comparison.pdf)
- [`figures/loss_norm_method_comparison.png`](figures/loss_norm_method_comparison.png)
- [`figures/loss_norm_relobralo_errmap.pdf`](figures/loss_norm_relobralo_errmap.pdf)
- [`figures/loss_norm_relobralo_errmap.png`](figures/loss_norm_relobralo_errmap.png)
- [`figures/loss_norm_relobralo_weights.pdf`](figures/loss_norm_relobralo_weights.pdf)
- [`figures/loss_norm_relobralo_weights.png`](figures/loss_norm_relobralo_weights.png)
- [`figures/nas_best_surface.pdf`](figures/nas_best_surface.pdf)
- [`figures/nas_best_surface.png`](figures/nas_best_surface.png)
- [`figures/nas_random_search.pdf`](figures/nas_random_search.pdf)
- [`figures/nas_random_search.png`](figures/nas_random_search.png)
- [`figures/nas_search_results.pdf`](figures/nas_search_results.pdf)
- [`figures/nas_search_results.png`](figures/nas_search_results.png)

## In the lecture script

§Chapter 4 (Neural architecture search and loss normalization). The full chapter map is in [`script_to_lectures.md`](../../lecture_script/script_to_lectures.md).

## Readings

Curated bibliography for this lecture: [`lecture_05.md`](../../readings/links_by_lecture/lecture_05.md). The full BibTeX is in [`readings/bibliography.bib`](../../readings/bibliography.bib).

---

| ← Previous | Next → |
|---|---|
| [**Lecture 04: IRBC with DEQNs**](../lecture_04_irbc_with_deqns/README.md)<br><sub>International real business cycle with DEQNs at scale</sub> | [**Lecture 06: Agentic programming**](../lecture_06_agentic_programming/README.md)<br><sub>Claude Code workflow, prompts, project memory, custom skills, subagents, hooks, 12-exercise workshop</sub> |

[↑ Course map](../../COURSE_MAP.md)
