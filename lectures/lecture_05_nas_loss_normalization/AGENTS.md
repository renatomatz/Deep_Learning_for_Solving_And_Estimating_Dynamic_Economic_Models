# AGENTS.md

## Purpose

Lecture 05 covers DEQN hyperparameter engineering: neural architecture search,
random search, Hyperband/successive halving, and multi-component loss balancing
with ReLoBRaLo, SoftAdapt, and GradNorm.

## Map

- Start with `README.md`.
- Notebook numbering starts at `02`; this is intentional.
- `code/lecture_05_02_NAS_Random_Search_10D.ipynb` is the random-search
  notebook.
- `code/lecture_05_03_NAS_RandomSearch_Hyperband.ipynb` uses the
  `code/nas_results/search_records.pkl` cache.
- `code/lecture_05_04_Loss_Normalization.ipynb` generates loss-normalization
  comparisons.
- `code/lecture_05_05_IRBC_Exercise.ipynb` is a student exercise and
  intentionally contains TODOs.
- Current Julia/Lux/Pluto previews:
  - `code_julia/lecture_05_02_NAS_Random_Search_10D_Lux.jl`
  - `code_julia/lecture_05_03_NAS_RandomSearch_Hyperband_Lux.jl`
  - `code_julia/lecture_05_04_Loss_Normalization_Lux.jl`
  - `code_julia/lecture_05_05_IRBC_Exercise_Lux.jl`
- Two slide decks live under `slides/`.
- `figures/`, `code/nas_outputs/`, and `code/nas_results/` contain checked-in
  teaching artifacts.

## Generated Artifacts

Run notebooks from `code/` so relative writes resolve:

- Notebook 02 writes `../figures/nas_random_search.{pdf,png}` and CSVs under
  `nas_outputs/`.
- Notebook 03 writes `../figures/nas_search_results.*` and
  `../figures/nas_best_surface.*`.
- Notebook 04 writes `loss_norm_*` figure pairs.

One known mismatch: notebook 02 refers to `nas_outputs/top5_retrained_test_metrics.csv`,
while the current tree has `top10_retrained_test_metrics.csv` with five data
rows. Do not "clean this up" without checking the notebook and slides together.

`figures/nas_random_search_results.*` exists but is not referenced by the
inspected README, slide sources, or notebook save paths.

## Julia Preview

Run Julia previews from `code_julia/` with the shared `../../../julia` project;
the notebooks activate it with `Pkg.activate(joinpath(@__DIR__, "..", "..",
"..", "julia"))` and import `DLEFJulia`. Keep them as Pluto `.jl` notebooks,
not Jupyter/IJulia replacements.

The previews preserve the course's `RUN_MODE = "smoke"` and `SEED = 0`
convention, with `teaching` and `production` budgets defined locally. Use smoke
mode for documentation or loadability checks. The targeted smoke coverage is
`julia/test/smoke/wave2_notebooks.jl`; from the repository root this is:

```bash
cd julia
julia --project=. test/smoke/wave2_notebooks.jl
```

These are compact Lux-native translations, not full production parity claims.
The NAS previews search over small CPU budgets and synthetic Lux tasks; they do
not validate or regenerate the checked-in Python NAS cache, CSVs, or figures.
The Hyperband translation intentionally avoids using or rewriting the Python
pickle cache.

Preserve Lux-native explicit parameter/state calls and feature-by-batch arrays
at Lux boundaries. Reuse shared `DLEFJulia` helpers for run modes, seeding,
training setup, plotting sizes, loss weighting, and IRBC scaffolding rather than
duplicating local utilities.

The Julia NAS notebooks intentionally avoid rewriting `figures/`,
`code/nas_outputs/`, and `code/nas_results/`. `lecture_05_05_IRBC_Exercise_Lux.jl`
preserves TODOs by design and should not be turned into a completed solution
during documentation or smoke-test cleanup.
