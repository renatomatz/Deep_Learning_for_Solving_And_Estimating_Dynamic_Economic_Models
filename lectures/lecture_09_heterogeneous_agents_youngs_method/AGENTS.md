# AGENTS.md

## Purpose

Lecture 09 introduces heterogeneous-agent distribution tracking via Young's
deterministic histogram method, then embeds histograms in DEQN and
Krusell-Smith-style solvers.

## Map

Start with `README.md`. Notebook order:

1. `code/lecture_09_10_Youngs_Method_Examples.ipynb`
2. `code/lecture_09_11_Continuum_of_Agents_DEQN.ipynb`
3. `code/lecture_09_12_KrusellSmith_DeepLearning.ipynb`

Current Julia/Lux/Pluto previews:

- `code_julia/lecture_09_10_Youngs_Method_Examples_Lux.jl`
- `code_julia/lecture_09_11_Continuum_of_Agents_DEQN_Lux.jl`
- `code_julia/lecture_09_12_KrusellSmith_DeepLearning_Lux.jl`

The slide deck is encyclopedic and self-study oriented; use it as a map rather
than assuming every slide has a matching notebook implementation.

## Running And Editing

For Python notebooks, use the root Python environment and prefer
`RUN_MODE = "smoke"`. Production settings can involve tens of thousands of
episodes. Keep the existing Python notebooks and checked-in outputs intact; do
not clear, renumber, or re-execute notebooks just to inspect them.

Young's histogram update is the central pedagogical object. Preserve its
mass/mean behavior and the distinction between distribution propagation and
policy training.

Run Julia previews from `code_julia/` with the shared `../../../julia` project:
the notebooks activate it with `Pkg.activate(joinpath(@__DIR__, "..", "..",
"..", "julia"))` and import `DLEFJulia`. Keep them as Pluto `.jl` notebooks,
not Jupyter/IJulia replacements. Preserve `RUN_MODE = "smoke"` / `SEED = 0`,
`run_mode_budget`, `rng_from_seed`, Lux's explicit `model(x, ps, st)` state
threading, and feature-by-batch arrays at Lux boundaries.

The Julia previews are covered by `julia/test/smoke/wave3_notebooks.jl`. Treat
that harness as a loadability and finite-small-run check; it does not establish
production parity. The Julia continuum DEQN preview is a smoke-size
Bewley-style simplification with two idiosyncratic income states, while the
Krusell-Smith deep-learning preview preserves the Python Phase B running-panel
semantics at a compact smoke budget.

Do not let Pluto or Jupyter produce output churn as a side effect of inspection.
Only save notebook rewrites when the task intentionally changes that notebook.

The notebooks are self-contained in this folder; no local data files were found.
