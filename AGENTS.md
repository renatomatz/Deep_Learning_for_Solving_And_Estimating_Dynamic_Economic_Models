# AGENTS.md

## Repository Purpose

This repository is an open-source, self-study course:
Deep Learning for Solving and Estimating Dynamic Models in Economics and
Finance.

It contains 18 lecture folders, runnable notebooks, Beamer slide decks, a
textbook-length companion script, reading guides, exercises, and supporting
assets. Use this root file for repository-wide orientation. Use lower-level
`AGENTS.md` files and each lecture `README.md` for local details.

## Start Here

- `README.md` is the public overview, setup guide, syllabus, citation block, and
  top-level map.
- `COURSE_MAP.md` records the lecture order, prerequisites, compute tiers,
  learning paths, and method-choice guide.
- `JULIA_LUX_JUPYTER_TRANSLATION_REPORT.md` is the source of truth for the
  Julia/Lux/Jupyter translation track and its dependency policy.
- `julia/AGENTS.md` gives package-local guidance for shared Julia source, unit
  tests, and smoke notebook checks.
- `lectures/lecture_XX_*/README.md` is the authoritative entry point for a
  lecture folder.
- `lecture_script/script_to_lectures.md` maps companion-script chapters and
  appendices to lecture folders.
- `readings/links_by_lecture/lecture_XX.md` gives curated reading links for
  each lecture.

Refer to lectures as `Lecture XX`. Lecture folders use
`lectures/lecture_XX_snake_case_topic/`.

## Environment

Python 3.10+ is required. Prefer one of the root setup paths:

```bash
pip install -r requirements.txt
```

or:

```bash
conda env create -f environment.yml
conda activate dlef
```

The main stacks are NumPy, SciPy, pandas, Matplotlib, scikit-learn,
TensorFlow/TensorFlow Probability, PyTorch, JAX/Optax/Flax, GPyTorch, BoTorch,
SALib, TensorBoard, requests, and PyYAML.

Some JAX GPU notebooks may require this shell setting before launching Jupyter:

```bash
export XLA_FLAGS="--xla_gpu_cuda_data_dir=/path/to/cuda"
```

The Julia translation track uses the shared project under `julia/`:

```bash
cd julia
julia --project=. test/runtests.jl
```

Run the notebook execute-smoke separately when checking Julia translations, for
example `julia --project=test/smoke test/smoke/wave1_notebooks.jl` (from the
`julia/` directory). These smoke files run under the `test/smoke` environment
(which carries `NBInclude`) and are not included by the default
`test/runtests.jl` entry point.

## Notebook Conventions

The executable units are mostly Jupyter notebooks under lecture `code/`
directories. Run a notebook from its own `code/` directory unless local guidance
says otherwise; several notebooks write or read relative paths such as
`../figures`, `../slides/fig`, `nas_outputs`, local CSV files, or runtime logs.

Most long-running notebooks expose:

- `RUN_MODE = "smoke" | "teaching" | "production"`
- `SEED = 0` or equivalent fixed seeds

Use `smoke` for quick checks. Use `teaching` or `production` only when the task
requires higher-fidelity reproduction and compute budget is available. Some
lecture-specific files document exceptions, including Lecture 13 defaulting to
`production` and Lecture 15 using fixed CPU budgets instead of `RUN_MODE`.

Checked-in notebook outputs are part of the teaching artifacts. Do not clear,
renumber, or regenerate outputs just to inspect a notebook. For content
inspection, prefer notebook-aware tooling or JSON source-cell extraction because
raw grep over `.ipynb` files can match embedded images and output blobs.

Julia translations live in lecture-local `code_julia/` directories as Jupyter
`.ipynb` notebooks (nbformat 4, `julia` kernel), committed output-free. Each
activates the root `julia` project in its first code cell and uses `DLEFJulia`
shared helpers. Preserve Lux-native explicit parameter/state calls such as
`y, st_new = model(x, ps, st)` and feature-by-batch arrays at Lux boundaries.
Use the same VSCode + Jupyter tooling as the Python `code/` track.

The Julia notebooks are previews of the Python course material, not replacements
for the checked-in Python notebooks. Some previews are intentionally compact or
smoke-first; local `AGENTS.md` files record known gaps and simplifications.

## Slides And Script

Slide sources are Beamer `.tex` files under lecture `slides/` directories. PDFs
are built artifacts but are intentionally checked in for readers.

Edit `.tex` sources, not PDFs. Compile from the directory that contains the
slide source so relative image paths resolve. Rebuild PDFs only when the task
requires it.

The companion script source is a monolithic LaTeX file under `lecture_script/`.
Use that directory's local guidance before changing the manuscript.

## Static Assets

Figures live in several places: `assets/`, `lecture_script/fig/`, lecture
`figures/` directories, and lecture `slides/fig`, `slides/figures`, or
`slides/images` directories. Some lecture-local figures are shared build inputs
for the companion script, so do not treat them as disposable local clutter.

For borrowed, adapted, or uncertain third-party media, check and update
`assets/attributions.yml`.

## Contribution Conventions

Keep changes focused and limited to one logical concern. Use American English in
student-facing files. Preserve the course's teaching intent: TODOs, blank
exercise notebooks, deliberate failure modes, and validation artifacts are often
intentional.

Keep the Julia stack narrow. Do not add Julia dependencies unless a concrete
notebook or shared helper requires them, `julia/Project.toml` is updated, and a
targeted unit or smoke check covers the new dependency. Follow the phase policy
in `JULIA_LUX_JUPYTER_TRANSLATION_REPORT.md`.

When reporting or fixing bugs, identify the relevant `Lecture XX`, notebook, or
slide deck. Include Python/library versions and a minimal reproducer when
applicable.

## Licensing And Citation

Source code is MIT licensed. Text, slides, figures, written notes,
`COURSE_MAP.md`, and per-lecture README files are CC0 1.0 Universal unless a
third-party asset says otherwise.

For research use, cite `CITATION.cff`; the preferred citation is the 2026 arXiv
preprint `arXiv:2605.14493`.
