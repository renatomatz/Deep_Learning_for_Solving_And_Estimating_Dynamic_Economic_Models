# AGENTS.md

## Purpose

Lecture 08 scales DEQNs to overlapping-generations models: a 6-age analytic
validation case and the 56-cohort AGS benchmark with borrowing and collateral
constraints.

## Map

Start with `README.md`. Notebook roles:

- `code/lecture_08_08_OLG_Analytic_DEQN_persistent.ipynb` is the primary
  analytic validation notebook.
- `code/lecture_08_07_OLG_Analytic_DEQN_exogenous.ipynb` is the exogenous-cloud
  ablation.
- `code/lecture_08_10_OLG_Benchmark_DEQN_persistent.ipynb` is the primary
  56-cohort benchmark notebook.
- `code/lecture_08_09_OLG_Benchmark_DEQN_exogenous.ipynb` is the benchmark
  exogenous-cloud ablation.
- `code/lecture_08_11_OLG_Exercise.ipynb` is the exercise.
- Current Julia/Lux/Pluto previews:
  - `code_julia/lecture_08_07_OLG_Analytic_DEQN_exogenous_Lux.jl`
  - `code_julia/lecture_08_08_OLG_Analytic_DEQN_persistent_Lux.jl`
  - `code_julia/lecture_08_09_OLG_Benchmark_DEQN_exogenous_Lux.jl`
  - `code_julia/lecture_08_10_OLG_Benchmark_DEQN_persistent_Lux.jl`
  - `code_julia/lecture_08_11_OLG_Exercise_Lux.jl`

## Running And Editing

Use `RUN_MODE = "smoke"` for checks. Production settings can be large, including
many simulation segments and wide networks.

Closed-form savings rates in the analytic notebooks are validation targets, not
training data. Preserve the distinction between persistent-simulation training
and exogenous-cloud ablations.

Run Julia previews from `code_julia/` with the shared `../../../julia` project.
The notebooks activate it with `Pkg.activate(joinpath(@__DIR__, "..", "..",
"..", "julia"))`, import `DLEFJulia`, preserve `RUN_MODE = "smoke"` and
`SEED = 0`, and use local `teaching` and `production` budgets.

The targeted smoke coverage is `julia/test/smoke/wave3_notebooks.jl`; from the
repository root this is:

```bash
cd julia
julia --project=. test/smoke/wave3_notebooks.jl
```

In smoke mode the benchmark notebooks reduce the 56-cohort problem to a small
cohort count for structural checks; this is not convergence or production
validation. The Julia previews are teaching previews, not full replacements for
the Python benchmark artifacts.

Preserve Lux-native explicit parameter/state calls, feature-by-batch arrays at
Lux boundaries, shared `DLEFJulia` OLG helpers, and Float64-sensitive
diagnostics where used. Keep the analytic and benchmark variants as separate
student-facing notebooks even when shared helper code is parameterized by
sampling mode.

Do not rewrite borrowing/collateral constraint transforms or
Fischer-Burmeister terms without a mathematical reason.

The checked-in notebooks and generated teaching outputs are artifacts; do not
rerun, clear, renumber, or regenerate them just to inspect the lecture.
