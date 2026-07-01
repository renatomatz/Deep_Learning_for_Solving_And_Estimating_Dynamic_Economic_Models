# AGENTS.md

## Purpose

Lecture 15 applies Lecture 14's surrogate idea to simulated method of moments in
Brock-Mirman. Current notebooks estimate scalar `rho` and joint `(beta, rho)`
using neural policy surrogates.

## Map

Start with `README.md`, then inspect `slides/lecture_15_structural_estimation_smm.tex`.
Notebook order:

1. `code/lecture_15_03_Structural_Estimation_BM.ipynb`
2. `code/lecture_15_03b_Structural_Estimation_BM_Joint.ipynb`

Matching Julia/Lux/Pluto previews live under `code_julia/`:

- `code_julia/lecture_15_03_Structural_Estimation_BM_Lux.jl`
- `code_julia/lecture_15_03b_Structural_Estimation_BM_Joint_Lux.jl`

The slide deck uses the `rho_*` and `joint_*` SMM figures under `slides/fig/`.
Files named `rho_gp_*` and `joint_gp_*` are optional or legacy assets not used by
the current deck.

## Running And Editing

Lecture 15 notebooks are CPU-only fixed-budget notebooks, not `RUN_MODE`
notebooks. They set PyTorch thread counts, use common random numbers, and use
fixed classroom budgets such as `N_TRAIN = 400` or `N_TRAIN = 200`. Preserve
seeds and common-random-number logic when changing estimation code.

Do not make this guidance Python-only. The Julia previews also preserve
fixed-budget style and common random numbers rather than adding `RUN_MODE`.
They activate the shared `../../../julia` project, import `DLEFJulia`, use
`SEED = 0`, and expose tiny smoke budgets such as `N_TRAIN = 4`. Preserve
Lux-native explicit parameter/state calls, feature-by-batch model inputs, and
the separation between simulation, moment computation, criterion evaluation,
and identification diagnostics.

The Julia budgets are intentionally tiny smoke previews, so do not treat them as
reproductions of the Python classroom-budget estimates. They are covered by
`julia/test/smoke/wave5_notebooks.jl`, which checks that the Pluto files include
and return finite small-run diagnostics; it is not an SMM accuracy or
identification-parity test.

`SAVE_FIGURES` defaults to `False`. If enabled, figures write to `../slides/fig`
relative to `code/`.

Gotcha: the README mentions a GP-over-moment-map layer, and GP assets exist, but
the current main notebooks and slides focus on neural-surrogate SMM. Notebook
`03b` says the BoTorch active-learning extension was removed from the main
notebook.
