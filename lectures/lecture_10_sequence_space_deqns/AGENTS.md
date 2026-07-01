# AGENTS.md

## Purpose

Lecture 10 switches from state-space inputs to sequence-space shock-history
inputs for DEQNs, with Brock-Mirman, IRBC, Krusell-Smith TensorFlow, and a
JAX/Optax CPU tutorial.

## Map

Start with `README.md`. Notebook order:

1. `code/lecture_10_05_SequenceSpace_BrockMirman.ipynb`
2. `code/lecture_10_05b_SequenceSpace_IRBC.ipynb`
3. `code/lecture_10_06_SequenceSpace_KrusellSmith.ipynb`
4. `code/lecture_10_KrusellSmith_Tutorial_CPU.ipynb`

Current Julia/Lux/Pluto previews:

- `code_julia/lecture_10_05_SequenceSpace_BrockMirman_Lux.jl`
- `code_julia/lecture_10_05b_SequenceSpace_IRBC_Lux.jl`
- `code_julia/lecture_10_06_SequenceSpace_KrusellSmith_Lux.jl`
- `code_julia/lecture_10_KrusellSmith_Tutorial_CPU_Lux.jl`

All four Python notebooks currently have Julia preview counterparts.

## Running And Editing

For Python notebooks, use the root Python environment. TensorFlow is used for
the main notebooks; JAX, JAXLIB, and Optax are used by the CPU tutorial. Keep
the existing Python notebooks and checked-in outputs intact; do not clear,
renumber, or re-execute notebooks just to inspect them.

Prefer `RUN_MODE = "smoke"`, but note that some notebooks define `RUN_MODE`
without scaling every training constant from it. In particular, the
Brock-Mirman sequence-space notebook uses fixed `cloud_steps = 256 * 8`, and the
IRBC sequence-space notebook has fixed history/pretraining/training defaults.

Preserve the sequence-space distinction: the key input is shock history, not the
current endogenous state. In Julia, standardize the history tensor layout in
lecture/shared helpers and flatten or orient only at the Lux boundary.

Run Julia previews from `code_julia/` with the shared `../../../julia` project:
the notebooks activate it with `Pkg.activate(joinpath(@__DIR__, "..", "..",
"..", "julia"))` and import `DLEFJulia`. Keep them as Pluto `.jl` notebooks,
not Jupyter/IJulia replacements. Preserve `RUN_MODE = "smoke"` / `SEED = 0`,
`run_mode_budget`, `rng_from_seed`, Lux's explicit `model(x, ps, st)` state
threading, and feature-by-batch arrays at Lux boundaries.

The Julia previews are covered by `julia/test/smoke/wave3_notebooks.jl`. Smoke
histories are much shorter than the Python lecture targets, so the smoke
harness checks structure and finite execution rather than sequence-length
parity. Do not let Pluto or Jupyter produce output churn as a side effect of
inspection; only save notebook rewrites when the task intentionally changes
that notebook.

Known Julia parity risks: the IRBC preview uses country-shock histories without
the Python notebook's separate aggregate shock channel, the Krusell-Smith shared
residual computes a complementarity diagnostic that is not included in the
reported loss, and monotonicity/concavity are checked after the fact rather than
guaranteed by construction.
