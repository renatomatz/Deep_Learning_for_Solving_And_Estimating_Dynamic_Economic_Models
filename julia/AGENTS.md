# AGENTS.md

## Purpose

`julia/` contains the shared Julia package and tests for the Lux/Pluto preview
track. The student-facing translated notebooks live in lecture-local
`code_julia/` directories and import this package.

## Source Of Truth

Read `../JULIA_LUX_PLUTO_TRANSLATION_REPORT.md` before changing Julia source,
dependencies, or translated notebooks. That report defines the Pluto notebook
choice, Lux-native API expectations, feature-by-batch convention, run modes, and
dependency policy.

## Package Conventions

- `Project.toml` is the shared Julia environment for all Pluto notebooks.
- `src/DLEFJulia.jl` should include every implemented shared module under
  `src/`.
- Keep Lux code explicit: call models as `y, st_new = model(x, ps, st)` and
  thread updated state through residual and training helpers.
- Use feature-by-batch arrays at Lux boundaries. Convert notebook data at the
  boundary instead of changing Lux helper conventions.
- Residual helpers should return structured diagnostics and finite losses when
  practical.
- Some PINN helpers are intentionally stateless Dense/tanh teaching helpers even
  though their signatures accept `(model, ps, st, ...)`; do not reuse them with
  stateful Lux layers without checking state propagation.

## Dependencies

Keep the stack narrow. The current project uses the Phase-0 core plus
`FastGaussQuadrature`, which is justified by stochastic DEQN/quadrature
notebooks. Do not add packages such as GPU backends, GP packages, spline
packages, `Optimization`, `QuadGK`, or `Enzyme` unless a concrete notebook or
shared helper requires them and a targeted unit or smoke test covers the use.

If dependencies change, update `Project.toml` and consider whether compat bounds
or the translation report need an explicit follow-up note.

## Tests

From this directory:

```bash
julia --project=. test/runtests.jl
```

This default test entry point covers shared helpers, shape conventions, finite
losses, analytic identities, quadrature, conservation checks, and representative
training steps. It does not include the notebook smoke files.

Run the Python/Julia notebook equivalence guard explicitly when changing
translated notebooks or shared helpers used by them:

```bash
julia --project=. test/python_julia_equivalence.jl
```

This integration suite checks the pair/gap inventory, semantic markers, and
smoke-scale cross-lecture mechanics against the Python course as ground truth.

Run smoke checks explicitly and selectively:

```bash
julia --project=. test/smoke/wave1_notebooks.jl
julia --project=. test/smoke/wave2_notebooks.jl
julia --project=. test/smoke/wave3_notebooks.jl
julia --project=. test/smoke/wave4_notebooks.jl
julia --project=. test/smoke/wave5_notebooks.jl
julia --project=. test/smoke/wave6_notebooks.jl
```

Full smoke coverage can be expensive. Prefer the wave that covers the lecture
you touched and report any skipped waves.
