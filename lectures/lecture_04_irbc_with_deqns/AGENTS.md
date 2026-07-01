# AGENTS.md

## Purpose

Lecture 04 is the first large-scale nonlinear DSGE application: international
real business cycle models solved with DEQNs, including smooth and
irreversible-investment variants.

## Map

- Start with `README.md`.
- `code/lecture_04_01_IRBC_DEQN_smooth.ipynb` is the smooth benchmark.
- `code/lecture_04_02_IRBC_DEQN_irreversible.ipynb` adds KKT multipliers and a
  Fischer-Burmeister complementarity loss.
- `code_julia/lecture_04_01_IRBC_DEQN_smooth_Lux.jl` and
  `code_julia/lecture_04_02_IRBC_DEQN_irreversible_Lux.jl` are the current
  Julia/Lux/Pluto previews.
- `slides/lecture_04_irbc.tex` is the slide source.
- `figures/` holds checked-in teaching figures used by the lecture.

## Julia/Lux/Pluto Preview Map

- `lecture_04_01_IRBC_DEQN_smooth_Lux.jl` translates the smooth IRBC mechanics
  with a two-country state, Stroud-3 normal shock checks, a Lux policy over
  `2N` state inputs, Euler residual diagnostics, and a zero-shock capital-drift
  check.
- `lecture_04_02_IRBC_DEQN_irreversible_Lux.jl` extends the smooth setup with a
  `3N`-output Lux policy, investment, KKT multipliers, and
  Fischer-Burmeister complementarity diagnostics for irreversible investment.

## Running And Editing

Run notebooks from `code/` and prefer smoke settings when present. Both notebooks
use a Stroud-3 monomial rule and a persistent-simulation training pipeline.

Run Julia previews from `code_julia/` with the shared `../../../julia` project.
They preserve Stroud/quadrature, zero-shock diagnostics, irreversible-investment
KKT multipliers, and Fischer-Burmeister residual structure, but use compact
sampled-state smoke training rather than full persistent-simulation parity.
Their smoke harness is `julia/test/smoke/wave2_notebooks.jl`.

Both Julia previews import `DLEFJulia`, use `RUN_MODE = "smoke"` and `SEED = 0`,
and should keep Lux calls explicit as `pieces, st_new = residual(model, ps, st,
states; params)`. Keep feature-by-batch state arrays at Lux boundaries: the
smooth model uses `2N x batch` state inputs and `2N` policy outputs; the
irreversible model uses `2N x batch` state inputs and `3N` outputs for policy
and multiplier components. Smoke mode checks finite small-run behavior, not
full GPU-scale convergence or exact Python persistent-simulation parity.

`figures/irbc_4approach_loss.*` is historical and not produced by the current
notebooks. `figures/irbc_euler_validation_country1/2.*` are not listed in the
README but are tracked artifacts; do not delete them casually.

Lecture 04 slides also point to
`../lecture_05_nas_loss_normalization/code/lecture_05_05_IRBC_Exercise.ipynb` as
the related hands-on exercise.

Do not clear, renumber, or regenerate Python notebook outputs, Pluto cell order,
lecture figures, or slide assets just to inspect this lecture.
