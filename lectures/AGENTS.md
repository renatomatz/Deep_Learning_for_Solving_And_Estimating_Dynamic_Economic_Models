# AGENTS.md

## Purpose

`lectures/` contains the 18 lecture folders. This file gives the shared lecture
contract; individual lecture folders add local warnings and navigation.

## Standard Folder Contract

Lecture folders use `lecture_XX_snake_case_topic`. Start every lecture task by
reading that folder's `README.md`; it records the intended start point, notebook
order, slides, script mapping, readings, and previous/next course flow.

Common contents:

- `README.md`: authoritative local index.
- `slides/`: Beamer `.tex` source and built PDF, plus any slide-local assets.
- `code/`: notebooks, scripts, local data, or checked-in outputs when present.
- `code_julia/`: Jupyter `.ipynb` notebooks for the Julia/Lux preview track when
  present.
- `figures/`: lecture-level generated or teaching figures when present.

Slide assets may live in `slides/fig`, `slides/figures`, or `slides/images`.
Generated figures may be referenced by TeX with relative paths, so do not move
or delete image files casually.

## Known Exceptions

- Lecture 01 has notebooks and `code/jupyter_intro.slides.html`, but no normal
  slide deck source.
- Lecture 06 is a workshop with Markdown templates, example subagents, example
  skills, hooks, data, outputs, and Python scripts rather than notebooks.
- Lecture 12 and Lecture 18 intentionally have no local code.
- Lecture 17 keeps supporting code in an external research repository.
- Lecture 17 has no local Julia notebooks by design; do not invent
  `code_julia/` there without an explicit scope change.
- Lecture 04 and Lecture 05 have top-level `figures/` directories in addition to
  slide-local assets.

## Running And Editing

Use the root `requirements.txt` or `environment.yml`; there are no per-lecture
environment files. Run notebooks from their own `code/` directories so relative
paths resolve.

Use `RUN_MODE = "smoke"` for quick checks unless local guidance documents an
exception or the user asks for `teaching` or `production`. Treat production runs
as potentially long or GPU/HPC-scale.

Do not execute all notebooks as a validation strategy. Many are expensive,
dependency-sensitive, or intentionally saved with outputs. Prefer targeted runs
of the relevant notebook cells or scripts.

For Julia translations, use Jupyter `.ipynb` notebooks under `code_julia/`
(nbformat 4, `julia` kernel), committed output-free. Each translated notebook's
first code cell activates the shared project with
`Pkg.activate(joinpath(@__DIR__, "..", "..", "..", "julia"))` and imports
`DLEFJulia`; run it from its own `code_julia/` directory when relative paths
matter. Preserve `RUN_MODE = "smoke"` / `SEED = 0` conventions unless local
guidance documents an exception.

Keep Julia notebooks Lux-native instead of Python-framework-shaped. Use the
explicit parameter/state flow `y, st_new = model(x, ps, st)`, preserve
feature-by-batch arrays at Lux boundaries, and keep economic residuals,
diagnostics, and output transforms separate from optimizer loops. Reuse shared
helpers in `../../../julia/src` before adding lecture-local mechanics; do not add
Julia dependencies without a concrete notebook need, `julia/Project.toml` change,
and a targeted unit or smoke check.

Julia unit tests live under `julia/test/runtests.jl`; notebook smoke checks live
under `julia/test/smoke/wave*_notebooks.jl` and are intentionally separate from
the default unit-test entry point. Smoke checks establish loadability and finite
small-run behavior, not production parity or convergence quality. Use targeted
smoke checks only; do not validate a documentation or small notebook change by
running the whole course.

Checked-in Python notebook outputs, Julia notebook cell order, generated slide
figures, and lecture figures are teaching artifacts. Do not clear, renumber, or
regenerate outputs just to inspect material, and do not churn generated files
unless the task explicitly requires rebuilding them. The Julia `.ipynb` notebooks
are committed output-free; do not save executed cell outputs into them.

The previously documented Lecture 07, Lecture 08, Lecture 09, and Lecture 11
Julia coverage gaps now have Jupyter counterparts under `code_julia/` and smoke
coverage under `julia/test/smoke/`. Several previews remain smoke-first
redesigns rather than full artifact reproduction, so use the lecture-local
guidance and equivalence tests before claiming production parity.

Edit slide `.tex` sources, not PDFs. Rebuild PDFs only when requested or when the
task explicitly changes a deck.
