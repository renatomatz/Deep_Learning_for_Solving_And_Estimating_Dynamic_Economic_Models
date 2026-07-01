# Lecture 14: Surrogates and Gaussian processes

A toolkit of cheap, differentiable approximations for expensive simulators: deep surrogates, Gaussian processes, active subspaces, and GP value-function iteration.

`gpu-recommended` · `long` · builds on [Lecture 02](../lecture_02_intro_deep_learning/README.md)

> 📑 **Slides:** [lecture_14_surrogates_and_gps.pdf](slides/lecture_14_surrogates_and_gps.pdf) (figures under [`slides/fig/`](slides/fig/))  
> 📓 **Notebooks:** [start here](code/lecture_14_01_Surrogate_Primer.ipynb) (9 in [`code/`](code/))  
> 📚 **Further reading:** [curated list](../../readings/links_by_lecture/lecture_14.md)  
> 📖 **Script:** §10.1-10.2 (Deep surrogates, pseudo-states, Black-Scholes worked example), §9.1-9.3 (GP regression, kernels, Bayesian active learning), §9.4 (GPs vs.\ DNNs), §9.5 (Active subspaces), §9.6 (GP value-function iteration), §9.7 (Deep kernels), §9.8 (GPs among Bayesian cousins)

## What this lecture covers

- **Deep surrogate models.** A neural network trained on simulator input-output pairs; when the surrogate pays for itself over direct simulation.
- **Gaussian processes.** GP regression with built-in uncertainty quantification; the basis for Bayesian active learning.
- **Bayesian active learning (BAL).** Choose the next training point to maximize information gain rather than throwing samples at a hypercube.
- **Active subspaces.** Linear and nonlinear dimension reduction so GPs scale to higher input dimensions.
- **Deep kernel learning.** Combining a neural feature map with a GP kernel for the same scaling goal.
- **GP value-function iteration.** GPs inside the VFI loop as a competitor to DEQN-VFI.

## Learning objectives

After this lecture you can:

- Train a deep surrogate on a controlled test problem and validate it out-of-sample.
- Fit a GP regressor and run a Bayesian active-learning loop.
- Apply linear and nonlinear active subspaces to a 10-D test function.
- Run GP-VFI on a 2-D test economy and reach a stable value function.
- Pick a surrogate vs GP vs deep-kernel approach for a new problem.

## Slides

- [`slides/lecture_14_surrogates_and_gps.pdf`](slides/lecture_14_surrogates_and_gps.pdf)
- [`slides/lecture_14_surrogates_and_gps.tex`](slides/lecture_14_surrogates_and_gps.tex)
- [`slides/fig/gp_active_learning.pdf`](slides/fig/gp_active_learning.pdf)

## Code

- [`code/lecture_14_01_Surrogate_Primer.ipynb`](code/lecture_14_01_Surrogate_Primer.ipynb)
- [`code/lecture_14_02_GP_and_BAL.ipynb`](code/lecture_14_02_GP_and_BAL.ipynb)
- [`code/lecture_14_04_GP_Value_Function_Iteration.ipynb`](code/lecture_14_04_GP_Value_Function_Iteration.ipynb)
- [`code/lecture_14_05_Active_Subspace_2D.ipynb`](code/lecture_14_05_Active_Subspace_2D.ipynb)
- [`code/lecture_14_06_Active_Subspace_10D.ipynb`](code/lecture_14_06_Active_Subspace_10D.ipynb)
- [`code/lecture_14_07_Active_Subspace_Nonlinear.ipynb`](code/lecture_14_07_Active_Subspace_Nonlinear.ipynb)
- [`code/lecture_14_08_Deep_Kernel_Learning.ipynb`](code/lecture_14_08_Deep_Kernel_Learning.ipynb)
- [`code/lecture_14_09_Deep_Active_Subspace_Ridge.ipynb`](code/lecture_14_09_Deep_Active_Subspace_Ridge.ipynb)
- [`code/lecture_14_10_Deep_AS_vs_Linear_AS_Borehole.ipynb`](code/lecture_14_10_Deep_AS_vs_Linear_AS_Borehole.ipynb)

### Julia/Lux/Pluto preview

- [`code_julia/lecture_14_01_Surrogate_Primer_Lux.jl`](code_julia/lecture_14_01_Surrogate_Primer_Lux.jl) trains a smoke-mode Lux Black-Scholes surrogate and validates normalized pricing errors.
- [`code_julia/lecture_14_02_GP_and_BAL_Lux.jl`](code_julia/lecture_14_02_GP_and_BAL_Lux.jl) fits the in-house Cholesky GP and selects a BAL point by posterior variance.
- [`code_julia/lecture_14_04_GP_Value_Function_Iteration_Lux.jl`](code_julia/lecture_14_04_GP_Value_Function_Iteration_Lux.jl) previews GP-VFI against the closed-form Brock-Mirman benchmark.
- [`code_julia/lecture_14_05_Active_Subspace_2D_Lux.jl`](code_julia/lecture_14_05_Active_Subspace_2D_Lux.jl) builds a 2D active subspace and polynomial ridge surrogate.
- [`code_julia/lecture_14_06_Active_Subspace_10D_Lux.jl`](code_julia/lecture_14_06_Active_Subspace_10D_Lux.jl) repeats the active-subspace workflow on a 10D ridge.
- [`code_julia/lecture_14_07_Active_Subspace_Nonlinear_Lux.jl`](code_julia/lecture_14_07_Active_Subspace_Nonlinear_Lux.jl) uses finite-difference gradients for a nonlinear interaction example.
- [`code_julia/lecture_14_08_Deep_Kernel_Learning_Lux.jl`](code_julia/lecture_14_08_Deep_Kernel_Learning_Lux.jl) compares raw-input and feature-space GP surrogates.
- [`code_julia/lecture_14_09_Deep_Active_Subspace_Ridge_Lux.jl`](code_julia/lecture_14_09_Deep_Active_Subspace_Ridge_Lux.jl) builds the Lux deep active-subspace encoder/link model.
- [`code_julia/lecture_14_10_Deep_AS_vs_Linear_AS_Borehole_Lux.jl`](code_julia/lecture_14_10_Deep_AS_vs_Linear_AS_Borehole_Lux.jl) evaluates a linear active-subspace surrogate on the borehole benchmark.

## In the lecture script

§10.1-10.2 (Deep surrogates, pseudo-states, Black-Scholes worked example), §9.1-9.3 (GP regression, kernels, Bayesian active learning), §9.4 (GPs vs. DNNs), §9.5 (Active subspaces), §9.6 (GP value-function iteration), §9.7 (Deep kernels), §9.8 (GPs among Bayesian cousins). The full chapter map is in [`script_to_lectures.md`](../../lecture_script/script_to_lectures.md).

## Readings

Curated bibliography for this lecture: [`lecture_14.md`](../../readings/links_by_lecture/lecture_14.md). The full BibTeX is in [`readings/bibliography.bib`](../../readings/bibliography.bib).

---

| ← Previous | Next → |
|---|---|
| [**Lecture 13: Continuous-time HA, numerics**](../lecture_13_continuous_time_ha_numerics/README.md)<br><sub>Achdou-Han-Lasry-Lions-Moll finite-difference scheme, PINN for HJB-KFE, continuous-time Aiyagari</sub> | [**Lecture 15: Structural estimation via SMM**](../lecture_15_structural_estimation_smm/README.md)<br><sub>Brock-Mirman SMM (single- and joint-parameter) on a deep surrogate</sub> |

[↑ Course map](../../COURSE_MAP.md)
