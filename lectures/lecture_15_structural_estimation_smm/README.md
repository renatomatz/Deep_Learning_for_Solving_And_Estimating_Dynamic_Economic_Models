# Lecture 15: Structural estimation via SMM

Structural estimation by simulated method of moments (SMM), made tractable by replacing the inner-loop model solve with a deep surrogate.

`cpu-standard` · `long` · builds on [Lecture 14](../lecture_14_surrogates_and_gps/README.md)

> 📑 **Slides:** [lecture_15_structural_estimation_smm.pdf](slides/lecture_15_structural_estimation_smm.pdf)  
> 📓 **Notebooks:** [start here](code/lecture_15_03_Structural_Estimation_BM.ipynb) (2 in [`code/`](code/))  
> 📚 **Further reading:** [curated list](../../readings/links_by_lecture/lecture_15.md)  
> 📖 **Script:** Chapter 10 (Structural estimation via SMM)

## What this lecture covers

- **SMM in one slide.** The moment-matching condition, the asymptotic distribution of the estimator, and the role of the weighting matrix.
- **Surrogate-based estimation.** Why the surrogate makes a brutal repeated re-solve into a cheap optimization.
- **Single-parameter Brock-Mirman.** Estimating the productivity persistence rho on a deep surrogate of the model.
- **Joint estimation.** Estimating (beta, rho) together; identification diagnostics, Jacobian rank, and the partial-identification ridge.
- **Stacking a GP over the moment map.** A second surrogate layer — one Gaussian process per moment, fit on a small (active-learning) design — so downstream objective evaluations need no simulation at all.

## Learning objectives

After this lecture you can:

- State the SMM moment condition and the asymptotic distribution of the estimator.
- Run a single-parameter SMM (rho) on a deep surrogate of Brock-Mirman.
- Run a joint (beta, rho) SMM and read off identification diagnostics.
- Stack a Gaussian-process layer over the moment map and reuse it for cheap objective evaluations.

## Slides

- [`slides/lecture_15_structural_estimation_smm.pdf`](slides/lecture_15_structural_estimation_smm.pdf)
- [`slides/lecture_15_structural_estimation_smm.tex`](slides/lecture_15_structural_estimation_smm.tex)

## Code

- [`code/lecture_15_03_Structural_Estimation_BM.ipynb`](code/lecture_15_03_Structural_Estimation_BM.ipynb)
- [`code/lecture_15_03b_Structural_Estimation_BM_Joint.ipynb`](code/lecture_15_03b_Structural_Estimation_BM_Joint.ipynb)

### Julia/Lux preview

- [`code_julia/lecture_15_03_Structural_Estimation_BM_Lux.ipynb`](code_julia/lecture_15_03_Structural_Estimation_BM_Lux.ipynb) translates scalar persistence SMM with a Lux pseudo-state policy surrogate and common random numbers.
- [`code_julia/lecture_15_03b_Structural_Estimation_BM_Joint_Lux.ipynb`](code_julia/lecture_15_03b_Structural_Estimation_BM_Joint_Lux.ipynb) translates joint `(beta, rho)` SMM with over-identified and weak-moment diagnostics.

## In the lecture script

Chapter 10 (Structural estimation via SMM). The full chapter map is in [`script_to_lectures.md`](../../lecture_script/script_to_lectures.md).

## Readings

Curated bibliography for this lecture: [`lecture_15.md`](../../readings/links_by_lecture/lecture_15.md). The full BibTeX is in [`readings/bibliography.bib`](../../readings/bibliography.bib).

---

| ← Previous | Next → |
|---|---|
| [**Lecture 14: Surrogates and Gaussian processes**](../lecture_14_surrogates_and_gps/README.md)<br><sub>Surrogate primer, GP regression, Bayesian active learning, active subspaces, deep kernel learning, GP-VFI</sub> | [**Lecture 16: Climate economics and IAMs**](../lecture_16_climate_economics_iams/README.md)<br><sub>DICE / CDICE simulation, deterministic and stochastic CDICE-DEQN</sub> |

[↑ Course map](../../COURSE_MAP.md)
