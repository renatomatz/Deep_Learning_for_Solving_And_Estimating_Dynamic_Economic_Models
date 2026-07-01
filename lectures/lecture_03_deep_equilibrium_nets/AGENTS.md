# AGENTS.md

## Purpose

Lecture 03 is the central Deep Equilibrium Nets lecture: deterministic and
stochastic Brock-Mirman, Fischer-Burmeister constraints, and loss-kernel
comparison.

## Map

- Start with `README.md`.
- `code/lecture_03_01_Brock_Mirman_1972_DEQN.ipynb` is the first deterministic
  Brock-Mirman notebook.
- `code/lecture_03_02_Brock_Mirman_Uncertainty_DEQN.ipynb` adds uncertainty.
- `code/lecture_03_03_DEQN_Exercises_Blanks.ipynb` and
  `code/lecture_03_04_DEQN_Exercises_Solutions.ipynb` are a paired
  exercise/solution set; TODO-style blanks are intentional.
- `code/lecture_03_05_StochasticBM_LossComparison.ipynb` compares stochastic
  loss kernels.
- `code_julia/lecture_03_01_*_Lux.jl` through
  `code_julia/lecture_03_05_*_Lux.jl` are the current Julia/Lux/Pluto previews.
- Slide assets live under `slides/figures/`; there is no top-level `figures/`
  directory here.

## Julia/Lux/Pluto Preview Map

- `lecture_03_01_Brock_Mirman_1972_DEQN_Lux.jl` is the deterministic
  Brock-Mirman DEQN preview. It predicts a savings policy, keeps the residual
  outside the optimizer loop, and compares against the analytic full-depreciation
  policy.
- `lecture_03_02_Brock_Mirman_Uncertainty_DEQN_Lux.jl` adds productivity shocks,
  Gauss-Hermite expectations, uniform-state training, and a compact simulated
  training-state pass.
- `lecture_03_03_DEQN_Exercises_Blanks_Lux.jl` preserves TODO prompts for
  transforms, residuals, complementarity, and a small life-cycle exercise.
- `lecture_03_04_DEQN_Exercises_Solutions_Lux.jl` gives compact Lux-native
  solutions for stochastic Brock-Mirman, labor heads, Fischer-Burmeister
  complementarity, and a tiny life-cycle residual check.
- `lecture_03_05_StochasticBM_LossComparison_Lux.jl` compares MSE, MAE, Huber,
  quantile, CVaR, and log-cosh on shared stochastic Brock-Mirman residual
  batches while keeping slide PNG generation off outside production mode.

## Running And Editing

Use the root environment. Run notebooks from `code/` and prefer `RUN_MODE =
"smoke"` for checks. Preserve seed behavior; some notebooks reset `SEED = 0` in
later import cells, not only at the top.

`lecture_03_05_StochasticBM_LossComparison.ipynb` writes slide PNGs only when
`SAVE_FIGS = (RUN_MODE == "production")`. Do not regenerate or delete checked-in
slide figures casually.

For Julia previews, run Pluto notebooks from `code_julia/` with the shared
`../../../julia` project. They use `DLEFJulia`, Lux explicit
`model(x, ps, st)` calls, Gauss-Hermite quadrature helpers, and smoke budgets.
Smoke coverage is split between `julia/test/smoke/wave1_notebooks.jl` and
`julia/test/smoke/wave2_notebooks.jl`.

Keep feature-by-batch state arrays at Lux boundaries: deterministic states are
`1 x batch`; stochastic Brock-Mirman states are `2 x batch` with productivity
and capital rows. Preserve explicit `ps`/`st` threading through residual
functions, policy transforms, and diagnostics. Smoke runs check loadability and
finite residual behavior; they are not expected to reproduce production-quality
figures or exact TensorFlow/PyTorch/JAX trajectories.

The Julia blank exercise notebook intentionally preserves TODO-style placeholders
and should not be filled during cleanup. The Julia solution notebook is compact
and Lux-native; do not assume it reproduces every Python solution detail unless
you have checked the notebook contents.

Do not rewrite residual definitions, feasibility transforms, or
Fischer-Burmeister terms unless the task is explicitly mathematical. Do not
clear, renumber, or regenerate Python notebook outputs, Pluto cell order, or
checked-in slide figures just to inspect the lecture.
