# AGENTS.md

## Purpose

Lecture 01 is a beginner Python and Jupyter primer. Keep examples pedagogical,
small, and easy to inspect.

## Map

- `README.md` gives the notebook order.
- `code/*.ipynb` covers calculator use, variables, lists/tuples, strings,
  conditionals, loops, functions, classes, NumPy, pandas, plotting, and Jupyter.
- `code/temp_price.csv` is used by pandas/plotting notebooks.
- `code/example.wav` is used by `jupyter_intro.ipynb`.
- `code/jupyter_intro.slides.html` is generated slideshow output.

## Gotchas

Run notebooks from `code/` so local assets resolve.

Some notebooks intentionally demonstrate errors, kernel restarts, magics,
out-of-order execution hazards, and beginner mistakes. Do not "fix" pedagogical
errors unless the user asks for that.

Prefer notebook-aware editing. Preserve execution order and avoid noisy output
churn.

## Julia Boundary

Do not line-by-line port the 12 Python primer notebooks. Lecture 01 is a Python
and Jupyter prerequisite, not a second Julia basics course. The Julia translation
track uses Lecture 02's `code_julia/lecture_02_00_Lux_Pluto_orientation.jl` as
the shared Julia workflow entry point.

If the owner explicitly asks for Lecture 01 Julia material, keep it to a short
orientation boundary rather than a notebook-by-notebook port: shared
`../../../julia` activation, `DLEFJulia`, `RUN_MODE = "smoke"`, `SEED = 0`,
feature-by-batch arrays, Lux `model(x, ps, st)`, and Pluto workflow conventions.
Do not create Jupyter/IJulia replacements for the Julia track.

Preserve `code/temp_price.csv`, `code/example.wav`, and
`code/jupyter_intro.slides.html`. Do not clear, renumber, or regenerate checked-in
Python notebook outputs or the generated slideshow just to inspect them.
