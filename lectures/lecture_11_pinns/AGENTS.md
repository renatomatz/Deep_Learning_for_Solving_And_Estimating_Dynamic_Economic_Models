# AGENTS.md

## Purpose

Lecture 11 introduces Physics-Informed Neural Networks with PyTorch: ODE PINNs,
soft versus hard boundary conditions, a 2D Poisson PDE, cake-eating HJB, and
Black-Scholes.

## Map

Start with `README.md`. Notebook order:

1. `code/lecture_11_01_ODE_PINN_ZeroBCs.ipynb`
2. `code/lecture_11_02_ODE_PINN_SoftVsHardBCs.ipynb`
3. `code/lecture_11_03_PDE_PINN_Poisson2D.ipynb`
4. `code/lecture_11_04_Cake_Eating_HJB_PINN.ipynb`
5. `code/lecture_11_05_Black_Scholes_PINN.ipynb`

Current Julia/Lux previews:

- `code_julia/lecture_11_01_ODE_PINN_ZeroBCs_Lux.ipynb`
- `code_julia/lecture_11_02_ODE_PINN_SoftVsHardBCs_Lux.ipynb`
- `code_julia/lecture_11_03_PDE_PINN_Poisson2D_Lux.ipynb`
- `code_julia/lecture_11_04_Cake_Eating_HJB_PINN_Lux.ipynb`
- `code_julia/lecture_11_05_Black_Scholes_PINN_Lux.ipynb`

Slide source is `slides/lecture_11_pinns.tex`. Pre-rendered slide figures live
under `slides/fig/ext/`; some are not directly included in the current deck but
are still nearby teaching outputs.

## Running And Editing

For Python notebooks, use the root Python environment. This lecture uses
PyTorch, NumPy, Matplotlib, and SciPy for Black-Scholes. Keep the existing
Python notebooks and checked-in outputs intact; do not clear, renumber, or
re-execute notebooks just to inspect them.

Lecture 11 notebooks default to `RUN_MODE = "smoke"`. Preserve PyTorch autograd
patterns such as `torch.autograd.grad(..., create_graph=True)`, smooth `tanh`
activations, and FP64/Adam/L-BFGS choices in the HJB and Black-Scholes notebooks
unless intentionally changing convergence behavior.

Run Julia previews from `code_julia/` with the shared `../../../julia` project.
They activate it with `Pkg.activate(joinpath(@__DIR__, "..", "..", "..",
"julia"))` and import `DLEFJulia`. Keep them as Jupyter `.ipynb` notebooks. Preserve `RUN_MODE = "smoke"` / `SEED = 0`,
`run_mode_budget`, `rng_from_seed`, Lux's explicit `model(x, ps, st)` state
threading, feature-by-batch arrays at Lux boundaries, and Float64 where
derivative accuracy matters. Notebook 01 is covered by
`julia/test/smoke/wave1_notebooks.jl`; notebooks 02-05 are covered by
`julia/test/smoke/wave4_notebooks.jl`.

The Julia HJB and Black-Scholes previews use tiny Adam-only smoke training and
defer the Python notebooks' deterministic L-BFGS polish. First and second
derivatives are fragile in these notebooks; smoke mode is a finite-execution
check, not an accuracy guarantee. The standalone soft-vs-hard boundary-condition
notebook now has a Julia counterpart, while Notebook 03 still demonstrates both
forms in the Poisson preview.

Do not let the Jupyter notebooks produce output churn as a side effect of inspection.
Only save notebook rewrites when the task intentionally changes that notebook.

Compile slides from `slides/` so relative image paths resolve.
