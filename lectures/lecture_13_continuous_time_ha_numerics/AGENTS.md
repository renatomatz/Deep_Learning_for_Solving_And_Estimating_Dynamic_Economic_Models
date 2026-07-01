# AGENTS.md

## Purpose

Lecture 13 is the numerical continuous-time heterogeneous-agent bridge. It pairs
finite-difference Aiyagari validation with a PINN re-solve of the coupled
HJB-KFE system.

## Map

- Start with `README.md`.
- `code/lecture_13_08_Aiyagari_Continuous_Time_FD_and_PINN_PyTorch.ipynb` is the
  live code companion.
- `code_julia/lecture_13_08_Aiyagari_Continuous_Time_FD_and_PINN_Lux.jl` is the
  current Julia/Lux/Pluto preview.
- `slides/lecture_13_continuous_time_ha_numerics.tex` is mostly TeX, TikZ, and
  PGFPlots; no external include-heavy figure tree was found.

## Running And Editing

This lecture is not Python-only. Keep the PyTorch notebook and the
Julia/Lux/Pluto preview aligned at the level of numerical objects and teaching
warnings: finite-difference benchmark, HJB residual, KFE residual, density
normalization, and diagnostics.

The notebook currently defaults to `RUN_MODE = "production"` and `DEV = "cpu"`.
Its own notes say production is about 9 minutes on CPU and smoke is about 30
seconds. Set `RUN_MODE = "smoke"` for structural checks unless the user asks for
full reproduction.

Smoke runs may intentionally fail the savings `L^inf` gate; that is documented
behavior, not automatically a bug. The finite-difference code is validation and
reference logic, not training targets for the PINN.

Preserve the distinction between `KFE_FORM = "fv"` and `"strong"`. Be careful
around dense NumPy linear solves, PyTorch autograd with `create_graph=True`,
log-density normalization, and the HJB/KFE residual definitions.

Run the Julia preview from `code_julia/` with the shared `../../../julia`
project activated from the notebook, and keep it importing `DLEFJulia` rather
than duplicating shared helpers locally. It defaults to `RUN_MODE = "smoke"`,
`SEED = 0`, and `KFE_FORM = "fv"`, with `teaching` and `production` budgets
defined in the file. It is covered by `julia/test/smoke/wave4_notebooks.jl`,
which is separate from the default Julia unit-test entry point.

Treat the Julia file as smoke-first validation scaffolding, not parity with the
Python notebook's production default. Its smoke configuration uses very small
grids and steps, preserves FD-as-validation and HJB/KFE residual structure, and
may still fail strict production-quality savings gates.

Preserve Lux-native conventions in the Pluto file: explicit parameter/state
calls, feature-by-batch arrays at Lux boundaries, `Float64` for delicate
HJB/KFE/PINN calculations, and seeded RNGs via shared helpers. Do not replace
the Pluto preview with Jupyter/IJulia or train the PINN on finite-difference
outputs; the finite-difference block remains validation and reference logic.
