# Julia/Lux/Pluto Translation Report

Status: planning source of truth  
Date: 2026-06-29  
Audience: coordinator and implementation subagents translating the Python
notebooks into Julia/Lux/Pluto material

## Purpose

This document is the standing source of truth for the Julia translation track.
All implementation agents should read this file before editing code, then read
the root `AGENTS.md`, `lectures/AGENTS.md`, the target lecture's local
`AGENTS.md`, and that lecture's `README.md`.

The goal is to create a Lux-native Julia/Pluto version of the course's runnable
notebook material, preserving the economic and numerical teaching intent while
using Julia idioms and Lux's explicit parameter/state model.

## Scope

Translate the local Python Jupyter notebooks under lecture `code/` directories
into Julia Pluto notebooks and shared Julia source.

Local notebook inventory:

| Lecture | Local notebooks | Translation status |
|---|---:|---|
| Lecture 01 Python primer | 12 | Mostly skipped; see scope decision below |
| Lecture 02 Intro deep learning | 9 | In scope, except Python API pedagogy should be redesigned |
| Lecture 03 Deep Equilibrium Nets | 5 | In scope |
| Lecture 04 IRBC with DEQNs | 2 | In scope |
| Lecture 05 NAS / loss normalization | 4 | In scope |
| Lecture 06 Agentic programming | 0 | No notebook translation |
| Lecture 07 Autodiff for DEQNs | 4 | In scope |
| Lecture 08 OLG models with DEQNs | 5 | In scope |
| Lecture 09 Heterogeneous agents / Young | 3 | In scope |
| Lecture 10 Sequence-space DEQNs | 4 | In scope |
| Lecture 11 PINNs | 5 | In scope |
| Lecture 12 CT-HA theory | 0 | No local code |
| Lecture 13 CT-HA numerics | 1 | In scope |
| Lecture 14 Surrogates and GPs | 9 | In scope |
| Lecture 15 Structural estimation SMM | 2 | In scope |
| Lecture 16 Climate economics IAMs | 3 | In scope |
| Lecture 17 Deep UQ / Pareto policy | 0 local | External-code boundary; do not invent local code |
| Lecture 18 Wrap-up | 0 | No local code |

Total local notebooks in scope for conversion planning: 68. The effective
translation target is smaller because Lecture 01 should not be line-by-line
ported as a Python primer.

## Scope Decisions

Use Pluto notebooks, not Jupyter/IJulia notebooks. Pluto notebooks are `.jl`
files and should live beside lecture materials under `code_julia/`.

Make the translated material Lux-native. Do not mimic Keras, PyTorch, or JAX
surface APIs when a clear Lux idiom exists. Teach Lux's explicit call pattern:

```julia
y, st_new = model(x, ps, st)
```

Skip a full Lecture 01 Julia primer. Instead, create a short Lux/Pluto
orientation notebook covering:

- Pluto workflow and the shared project environment
- `RUN_MODE` and `SEED`
- feature-by-batch arrays
- Lux `model, ps, st`
- `Optimisers.jl` update flow
- the shared training-loop conventions used in later notebooks

Keep existing Python notebooks intact. Do not clear, renumber, re-execute, or
rewrite checked-in Python notebook outputs just to inspect or translate them.

Preserve exercise blanks and solutions as separate teaching artifacts. Do not
fill intentional TODOs in blank notebooks.

Lecture 17 has no local code by design. Do not add local stubs or replacement
notebooks for Lecture 17 unless the owner explicitly changes the scope.

## Repository Layout

Use a shared Julia package plus per-lecture Pluto notebooks:

```text
julia/
  Project.toml
  Manifest.toml              # create after first successful instantiate
  src/
    DLEFJulia.jl
    run_modes.jl
    lux_helpers.jl
    training.jl
    plotting.jl
    quadrature.jl
    losses.jl
    diagnostics.jl
    deqn/
    pinn/
    youngs/
    sequence_space/
    surrogates/
    smm/
    climate/
  test/
    runtests.jl
    smoke/

lectures/
  lecture_XX_topic/
    code_julia/
      lecture_XX_nn_original_name_Lux.jl
```

`julia/src` holds reusable, tested mechanics. `code_julia/` notebooks hold the
student-facing narrative, small glue code, figures, and lecture-specific
experiments.

Run Pluto notebooks from their lecture `code_julia/` folder when local relative
paths matter. Shared source should be loaded from the root `julia` environment.

## Dependency Policy

The dependency stack must stay small. The broad ecosystem list from exploration
is a menu, not a mandate. Add packages by phase only when they solve a concrete
notebook need.

Before adding a non-core dependency:

1. Explain why the existing core stack is insufficient.
2. Prefer a mature Julia package with active maintenance.
3. Update `julia/Project.toml`.
4. Add or update a smoke test for the functionality.
5. Update this report if the dependency becomes a project convention.

Julia 1.11.3 is available locally. At planning time, the active Julia depot did
not have the needed packages installed yet.

### Phase-0 Core

Use these as the first shared project dependencies:

- Notebook: `Pluto`, `PlutoUI`
- Deep learning: `Lux`, `Optimisers`, `MLUtils`, `ComponentArrays`, `NNlib`
- AD: `Zygote`, `ForwardDiff`
- Reproducibility/base numerics: `StableRNGs`, plus stdlibs `Random`,
  `LinearAlgebra`, `Statistics`, `SparseArrays`
- Data: `CSV`, `DataFrames`
- Plotting: `CairoMakie`

`CairoMakie` is the default plotting backend for translated notebooks. Do not
also add `Plots`/`StatsPlots` unless a specific notebook has a strong reason.

### AD Policy

Use `Zygote` for parameter gradients through Lux models.

Use `ForwardDiff` for low-dimensional input derivatives in PINNs and analytical
autodiff demonstrations when it is clearer and more robust than reverse-mode
nested AD.

Use finite differences only for diagnostics and gradient checks, not as the
primary training derivative path.

`DifferentiationInterface` is optional. Add it only when a family of notebooks
benefits from a common derivative wrapper.

`Enzyme` is optional and should not be added in the initial scaffold. Consider it
only for hard nested-AD or performance cases after a simpler `Zygote` /
`ForwardDiff` implementation is understood and tested.

### Quadrature Policy

Use `FastGaussQuadrature` for fixed Gaussian rules, especially Gauss-Hermite
expectations in stochastic DEQN notebooks and quadrature-heavy benchmarks.

Do not include `QuadGK` in the core environment. Add `QuadGK` only if a notebook
requires adaptive one-dimensional integration. `FastGaussQuadrature` and
`QuadGK` are not mutually exclusive, but they solve different problems.

### Optimization Policy

Use `Optimisers.jl` for neural-network training, especially Adam and related
first-order optimizers.

Add `Optim` / `Optimization` / `OptimizationOptimJL` only when needed for
L-BFGS, SMM criteria, or deterministic nonlinear solves. These are likely in
PINN, SMM, and later surrogate phases, not in the initial DEQN scaffold.

### Phase-Specific Additions

Add these only when the relevant lecture family begins:

- `FastGaussQuadrature`: stochastic DEQN, IRBC, sequence-space, climate
- `Distributions`: stochastic simulation, Normal CDFs, truncated distributions
- `Roots` or `NonlinearSolve`: implied-vol inversion, steady states
- `Optim`, `Optimization`, `OptimizationOptimJL`: L-BFGS and SMM
- `JLD2` or stdlib `Serialization`: NAS caches and lightweight checkpoints
- `KernelFunctions`: GP kernels and deep-kernel experiments
- `AbstractGPs` or `GaussianProcesses`: only if in-house Cholesky GP utilities
  become too cumbersome
- `QuasiMonteCarlo`: Latin hypercube or Sobol-style designs
- `GlobalSensitivity`: sensitivity-analysis extensions when needed
- `BSplineKit` or `Interpolations`: sequence-space Krusell-Smith I-spline or
  interpolation needs
- `MLDatasets`: Fashion-MNIST in the deep-learning foundations lecture
- `TensorBoardLogger`: only for the TensorBoard instrumentation translation
- `CUDA`: optional GPU backend, never required for smoke tests

## Lux/Pluto Coding Conventions

Use feature-by-batch arrays at the Lux boundary. Python notebooks are mostly
batch-major `(batch, features)`, but Lux `Dense` layers expect features first.
Choose one orientation per notebook family and convert at the boundary.

Separate raw neural networks from economic output transforms. A typical pattern:

```julia
raw, st = model(features, ps, st)
policy = transform_policy(raw, params)
```

Keep residual equations separate from training loops. Residual functions should
return structured components and diagnostics whenever practical.

Use named tuples or small structs for:

- economic parameters
- run-mode budgets
- network dimensions
- training schedules
- diagnostic tolerances

Keep random number generation explicit. Pass RNGs or derive local RNGs from
`SEED` using `StableRNGs`.

Avoid hidden global mutable state in shared source. Pluto cells can be reactive;
shared helpers should be plain functions or simple immutable configs when
possible.

Use `Float64` where convergence, PINN derivatives, or economics diagnostics
need it. Do not silently switch delicate HJB/PDE notebooks to `Float32`.

## Run Modes And Reproducibility

Preserve the existing course convention:

```julia
RUN_MODE = "smoke"      # or "teaching", "production"
SEED = 0
```

Smoke mode should run on CPU and verify structure quickly. It does not have to
match production figures, and some original smoke modes intentionally fail
strict production-quality gates.

Lecture 15 is an exception in the Python course: it uses fixed CPU budgets
rather than `RUN_MODE`. A Julia translation may preserve that fixed-budget style
or introduce a minimal smoke wrapper, but must preserve common-random-number
logic.

## Shared Modules To Build First

`run_modes.jl`

- run-mode parsing
- budget lookup
- seed helpers

`lux_helpers.jl`

- MLP factory
- output-head splitting
- sigmoid/softplus/capped transforms
- parameter/state initialization
- feature-orientation helpers

`training.jl`

- `Optimisers` update loop
- gradient clipping
- mini-batching
- metric logging
- finite-loss/NaN guards

`quadrature.jl`

- dependency-free quadrature interface/scaffold in Wave 0
- Gauss-Hermite nodes/weights after `FastGaussQuadrature` is added
- Stroud-3 monomial rule for IRBC
- tensor-product helper

`losses.jl`

- MSE, MAE, Huber, LogCosh
- quantile/pinball
- CVaR-style loss
- Fischer-Burmeister complementarity
- inverse-loss and ReLoBRaLo weights

`diagnostics.jl`

- policy drift
- Euler residual summaries
- analytic-policy error checks
- steady-state checks
- conservation checks for histogram updates

`youngs/`

- one-dimensional and two-dimensional mass scatter
- mass/mean preservation tests

`pinn/`

- input derivative helpers
- hard-boundary wrappers
- soft-boundary loss terms
- Adam to L-BFGS handoff when later dependencies are available

`surrogates/`

- Black-Scholes analytic formulas
- normalization helpers
- Cholesky GP prediction
- BAL scoring

`smm/`

- common random numbers
- moment computation
- SMM criteria
- identification diagnostics

`climate/`

- DICE/CDICE calibration containers
- carbon and temperature transitions
- deterministic and stochastic DEQN residual scaffolds

## Translation Waves

Wave 0: scaffold

- Create `julia/Project.toml`
- Add shared source skeleton
- Add first tests for run modes, transforms, quadrature, losses
- Add Lux/Pluto orientation notebook

Wave 1: low-risk Lux foundations

- Lux/Pluto orientation notebook
- selected Lecture 02 Lux-native foundations notebooks
- Lecture 03 deterministic Brock-Mirman
- Lecture 03 stochastic Brock-Mirman
- Lecture 07 analytical autodiff examples
- Lecture 11 first ODE PINNs

Wave 2: reusable DEQN families

- remaining Lecture 02 sequence/loss/training notebooks
- Lecture 03 exercises blanks/solutions
- Lecture 03 loss-kernel comparison
- Lecture 04 smooth and irreversible IRBC
- Lecture 05 loss kernels and NAS basics

Wave 3: OLG, Young, and sequence-space

- Lecture 08 analytic OLG exogenous/persistent as parameterized variants
- Lecture 08 56-cohort benchmark variants
- Lecture 09 Young histogram and continuum-agent DEQN
- Lecture 10 sequence-space notebooks

Wave 4: PINNs and continuous time

- Lecture 11 Poisson, cake-eating HJB, Black-Scholes
- Lecture 13 Aiyagari FD + HJB/KFE PINN

Wave 5: surrogates, GPs, SMM

- Lecture 14 surrogate primer
- Lecture 14 GP/BAL and GP-VFI
- Lecture 14 active subspaces and deep kernels
- Lecture 15 scalar and joint SMM

Wave 6: climate

- Lecture 16 climate exercise
- Lecture 16 deterministic CDICE-DEQN
- Lecture 16 stochastic CDICE-DEQN

## Lecture-Specific Notes

### Lecture 01

Do not line-by-line port the Python primer. If needed, mine it for a short
Julia/Pluto orientation notebook. Preserve `temp_price.csv`, `example.wav`, and
`jupyter_intro.slides.html`.

### Lecture 02

Translate deep-learning foundations into Lux-native examples. Python API
teaching should become Julia/Lux teaching:

- Keras Sequential becomes `Lux.Chain`
- PyTorch `nn.Module` examples become explicit Lux models and training loops
- TensorBoard instrumentation can use `TensorBoardLogger` only in that notebook
- Fashion-MNIST should use `MLDatasets` if included
- LSTM/Transformer notebooks should use Lux recurrent/attention primitives where
  stable, otherwise compact explicit components

### Lecture 03

Core DEQN lecture. Preserve analytic Brock-Mirman checks, stochastic
Gauss-Hermite expectations, simulated training states, loss-kernel comparison,
and exercise blank/solution separation.

### Lecture 04

Preserve Stroud-3 monomial expectations, persistent-simulation training,
time-invariance diagnostics, zero-shock steady-state diagnostics, irreversible
investment KKT multipliers, and Fischer-Burmeister complementarity.

### Lecture 05

Preserve NAS artifacts and the known `top5`/`top10` cache naming mismatch until
the notebook and slides are reviewed together. Do not casually delete
`figures/`, `code/nas_outputs/`, or `code/nas_results/`.

### Lecture 07

This lecture is about derivative structure. Isolate payoff/Lagrangian primitives
and slot-gradient helpers. Validate autodiff residuals against hand-coded
residuals before training.

### Lecture 08

The analytic and benchmark OLG notebooks have exogenous and persistent variants.
Implement shared helpers parameterized by sampling mode, but keep separate
student-facing notebooks for the teaching variants. Preserve borrowing,
collateral, and Fischer-Burmeister terms.

### Lecture 09

Young's histogram update is the central object. Implement it once with explicit
mass and mean conservation tests. Keep distribution propagation conceptually
separate from policy training.

### Lecture 10

Preserve the sequence-space distinction: inputs are shock histories, not just
current endogenous states. Standardize history tensor layout and flatten at the
Lux boundary.

### Lecture 11

PINN notebooks require careful first and second derivatives. Preserve soft versus
hard boundary-condition comparisons, smooth activations, FP64 where used, and
Adam/L-BFGS behavior when translated.

### Lecture 13

Very high difficulty. Preserve the finite-difference benchmark as validation
logic, not training data. Preserve `KFE_FORM = "fv"` versus `"strong"`, log
density normalization, HJB/KFE residual definitions, and the note that smoke
runs may fail strict production gates.

### Lecture 14

Start with simple in-house GP utilities using Cholesky and kernels. Add a GP
package only if it materially simplifies the GP/BAL/VFI notebooks. Treat
`fig_gp_vfi/` as generated output, not slide assets.

### Lecture 15

Preserve common random numbers and fixed classroom CPU budgets. Keep simulation,
moment computation, criterion evaluation, and identification diagnostics
separate.

### Lecture 16

Very high difficulty. Preserve the distinction between DICE simulation,
deterministic CDICE-DEQN, and stochastic CDICE-DEQN. Do not silently replace
production-code references with simplified teaching approximations. Check
`assets/attributions.yml` before reusing or moving climate figures.

## Subagent Workflow

Every implementation subagent must:

1. Read this report.
2. Read all applicable `AGENTS.md` files.
3. Read the target lecture `README.md`.
4. Inspect notebook source cells with notebook-aware tooling.
5. Work in an assigned, disjoint write scope.
6. Avoid editing existing Python notebooks and checked-in outputs.
7. Add or update tests for shared source changes.
8. Run targeted smoke checks only; do not execute the whole course.
9. Report changed paths, tests run, and unresolved risks.

Workers are not alone in the codebase. They must not revert or overwrite changes
made by other agents. If a convention in this report becomes wrong, workers
should flag it to the coordinator instead of silently diverging.

## Validation Strategy

Do not validate by running all notebooks. Many notebooks are intentionally
expensive, dependency-sensitive, or saved with outputs.

Use focused checks:

- unit tests for shared losses, quadrature, transforms, and residual helpers
- conservation tests for Young histogram updates
- finite-loss and no-NaN checks for training steps
- shape tests at Lux model boundaries
- analytic policy comparisons for Brock-Mirman and OLG
- smoke-mode short training runs for representative notebooks
- tolerance-based diagnostics rather than exact parity with TensorFlow/PyTorch/JAX

## Current Known Risks

- Julia AD behavior will not exactly match TensorFlow/PyTorch/JAX.
- Lux feature-by-batch conventions can cause subtle shape bugs if wrappers are
  inconsistent.
- Pluto reactivity rewards pure helpers; hidden global state will be painful.
- GP/BAL/BoTorch workflows do not have perfect one-for-one Julia equivalents.
- Lecture 08 benchmark OLG, Lecture 13 Aiyagari, and Lecture 16 CDICE are the
  highest-risk conversions.
- Dependency bloat is a real maintenance risk; keep the stack narrow.
- Existing Python notebook outputs are teaching artifacts; accidental churn
  would create large, low-value diffs.

## Primary References

- Lux documentation: https://lux.csail.mit.edu/stable/
- Lux interface and explicit parameter/state calls:
  https://lux.csail.mit.edu/stable/manual/interface
- Lux beginner training tutorial:
  https://lux.csail.mit.edu/stable/tutorials/beginner/1_Basics
- Pluto package management:
  https://plutojl.org/en/docs/packages/
- Optimisers.jl documentation:
  https://fluxml.ai/Optimisers.jl/stable/
- DifferentiationInterface documentation:
  https://juliadiff.org/DifferentiationInterface.jl/
- FastGaussQuadrature.jl:
  https://github.com/JuliaApproximation/FastGaussQuadrature.jl
- QuadGK.jl:
  https://juliamath.github.io/QuadGK.jl/stable/
