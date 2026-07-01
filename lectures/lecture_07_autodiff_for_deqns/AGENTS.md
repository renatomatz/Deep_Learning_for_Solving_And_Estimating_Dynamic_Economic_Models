# AGENTS.md

## Purpose

Lecture 07 makes automatic differentiation explicit for DEQN Euler residuals,
moving from hand-derived first-order/envelope equations to
`tf.GradientTape` on a payoff or Lagrangian primitive.

## Map

Start with `README.md`, then use `slides/lecture_07_autodiff_for_deqns.pdf` for
the narrative and notebooks for runnable details:

1. `code/lecture_07_01_AutoDiff_Analytical_Examples.ipynb`
2. `code/lecture_07_02_Brock_Mirman_AutoDiff_DEQN.ipynb`
3. `code/lecture_07_03_Brock_Mirman_Uncertainty_AutoDiff_DEQN.ipynb`
4. `code/lecture_07_04_IRBC_AutoDiff_DEQN.ipynb`

Current Julia/Lux previews:

- `code_julia/lecture_07_01_AutoDiff_Analytical_Examples_Lux.ipynb`
- `code_julia/lecture_07_02_Brock_Mirman_AutoDiff_DEQN_Lux.ipynb`
- `code_julia/lecture_07_03_Brock_Mirman_Uncertainty_AutoDiff_DEQN_Lux.ipynb`
- `code_julia/lecture_07_04_IRBC_AutoDiff_DEQN_Lux.ipynb`

## Running And Editing

Use the repo-level environment. Prefer `RUN_MODE = "smoke"` for execution unless
the user asks for `teaching` or `production`.

This lecture is about derivative structure. Be careful around `Pi(...)`, nested
or separate `GradientTape` blocks, slot-specific derivatives, dtype choices, and
gradient persistence. Do not rewrite residuals casually.

Run the Julia previews from `code_julia/` with the shared `../../../julia`
project; the notebook activates it with `Pkg.activate(joinpath(@__DIR__, "..",
"..", "..", "julia"))` and imports `DLEFJulia`. Keep it as a Jupyter `.ipynb` notebook.

The Julia previews use `RUN_MODE = "smoke"` and `SEED = 0`, plus `DLEFJulia`
helpers, `ForwardDiff`, and Lux/Zygote training checks where the Python ground
truth requires parameter updates. They are covered
by `julia/test/smoke/wave1_notebooks.jl`; from the repository root this is:

```bash
cd julia
julia --project=test/smoke test/smoke/wave1_notebooks.jl
```

The DEQN Euler-residual notebooks now have Julia counterparts with hand-coded
residual validation; `lecture_07_04` also includes smoke-scale Approach A/B
training diagnostics. Keep these validation paths intact when editing.

For future Julia translations, isolate payoff/Lagrangian primitives and
slot-gradient helpers before training. Validate autodiff residuals against
hand-coded residuals, preserve explicit Lux parameter/state calls, and use
feature-by-batch arrays at Lux boundaries.

The checked-in Python notebooks and Julia Jupyter notebooks are teaching artifacts;
do not rerun, renumber, clear, or regenerate outputs just to understand them.
For Python content inspection, parse notebook source cells rather than raw
grepping output-heavy `.ipynb` files.
