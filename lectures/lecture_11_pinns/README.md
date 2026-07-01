# Lecture 11: Physics-informed neural networks

Physics-Informed Neural Networks (PINNs) solve differential equations by minimizing the equation residual on collocation points.

`cpu-standard` · `long` · builds on [Lecture 07](../lecture_07_autodiff_for_deqns/README.md)

> 📑 **Slides:** [lecture_11_pinns.pdf](slides/lecture_11_pinns.pdf)  
> 📓 **Notebooks:** [start here](code/lecture_11_01_ODE_PINN_ZeroBCs.ipynb) (5 in [`code/`](code/))  
> 📚 **Further reading:** [curated list](../../readings/links_by_lecture/lecture_11.md)  
> 📖 **Script:** §7.1-7.4 (PINN foundations), §7.5-7.9 (economic PDEs: HJB, Black-Scholes)

## What this lecture covers

- **The PINN loss.** Differentiate the network output with autodiff, plug into the ODE/PDE residual, sum the squared residuals on collocation points.
- **Boundary conditions.** Soft (penalty in the loss) versus hard (trial solution that satisfies BCs by construction); when to use each.
- **A worked PDE.** A 2-D Poisson PDE solved end-to-end with a PINN.
- **Two economic applications.** The cake-eating Hamilton-Jacobi-Bellman equation with hard BCs, and Black-Scholes option pricing.
- **Optimization tricks.** Adam-then-L-BFGS schedules and FP64 for sharp PINN solutions.

## Learning objectives

After this lecture you can:

- Write the PINN loss for a given ODE or PDE.
- Distinguish soft and hard boundary-condition parametrizations and choose between them.
- Solve a 2-D Poisson PDE with a PINN.
- Solve the cake-eating HJB with a hard-BC trial solution.
- Price a European call option with a Black-Scholes PINN.

## Slides

- [`slides/lecture_11_pinns.pdf`](slides/lecture_11_pinns.pdf)
- [`slides/lecture_11_pinns.tex`](slides/lecture_11_pinns.tex)

## Code

### Julia/Lux preview

- [`code_julia/lecture_11_01_ODE_PINN_ZeroBCs_Lux.ipynb`](code_julia/lecture_11_01_ODE_PINN_ZeroBCs_Lux.ipynb)
- [`code_julia/lecture_11_02_ODE_PINN_SoftVsHardBCs_Lux.ipynb`](code_julia/lecture_11_02_ODE_PINN_SoftVsHardBCs_Lux.ipynb)
- [`code_julia/lecture_11_03_PDE_PINN_Poisson2D_Lux.ipynb`](code_julia/lecture_11_03_PDE_PINN_Poisson2D_Lux.ipynb)
- [`code_julia/lecture_11_04_Cake_Eating_HJB_PINN_Lux.ipynb`](code_julia/lecture_11_04_Cake_Eating_HJB_PINN_Lux.ipynb)
- [`code_julia/lecture_11_05_Black_Scholes_PINN_Lux.ipynb`](code_julia/lecture_11_05_Black_Scholes_PINN_Lux.ipynb)

### Python notebooks

- [`code/lecture_11_01_ODE_PINN_ZeroBCs.ipynb`](code/lecture_11_01_ODE_PINN_ZeroBCs.ipynb)
- [`code/lecture_11_02_ODE_PINN_SoftVsHardBCs.ipynb`](code/lecture_11_02_ODE_PINN_SoftVsHardBCs.ipynb)
- [`code/lecture_11_03_PDE_PINN_Poisson2D.ipynb`](code/lecture_11_03_PDE_PINN_Poisson2D.ipynb)
- [`code/lecture_11_04_Cake_Eating_HJB_PINN.ipynb`](code/lecture_11_04_Cake_Eating_HJB_PINN.ipynb)
- [`code/lecture_11_05_Black_Scholes_PINN.ipynb`](code/lecture_11_05_Black_Scholes_PINN.ipynb)

## In the lecture script

§7.1-7.4 (PINN foundations), §7.5-7.9 (economic PDEs: HJB, Black-Scholes). The full chapter map is in [`script_to_lectures.md`](../../lecture_script/script_to_lectures.md).

## Readings

Curated bibliography for this lecture: [`lecture_11.md`](../../readings/links_by_lecture/lecture_11.md). The full BibTeX is in [`readings/bibliography.bib`](../../readings/bibliography.bib).

---

| ← Previous | Next → |
|---|---|
| [**Lecture 10: Sequence-space DEQNs**](../lecture_10_sequence_space_deqns/README.md)<br><sub>Brock-Mirman, IRBC, Krusell-Smith with shock-history inputs</sub> | [**Lecture 12: Continuous-time HA, theory**](../lecture_12_continuous_time_ha_theory/README.md)<br><sub>HJB, Kolmogorov-forward, master equation, Ito calculus</sub> |

[↑ Course map](../../COURSE_MAP.md)
