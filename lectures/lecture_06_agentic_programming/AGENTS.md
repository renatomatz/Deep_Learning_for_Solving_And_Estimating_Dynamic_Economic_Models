# AGENTS.md

## Purpose

Lecture 06 is a self-contained workshop on using coding agents for research
workflows: prompting, project memory, custom skills, subagents, hooks,
autonomous loops, verification, and privacy.

The files here are teaching examples. They are not active instructions for every
future agent unless a user explicitly asks to adopt them. Do not add a Julia/Pluto
`code_julia/` translation here without an explicit scope change; Lecture 06 has
no notebook translation target in the Julia/Lux track.

## Map

- `README.md` is the canonical file index.
- `slides/` contains the main deck and exercise handout, with matching PDFs.
- `code/agentic_ai_lecture_syllabus.md` is the long-form syllabus.
- `code/exercise_prompts.md` and `code/exercise_solutions.md` are the workshop
  prompts and expected artifacts.
- `code/CLAUDE_md_template.md` is an example project-memory template.
- `code/generate_synthetic_data.py` creates `code/data/synthetic_panel.csv`.
- `code/mincer_demo.py` writes `code/outputs/mincer_table.tex` and
  `code/outputs/mincer_figure.pdf`.
- `code/skills/` and `code/subagents/` are example specs/templates.
- `code/hooks/settings.json` is a tool-specific hook example.

There is intentionally no `code_julia/` directory here. Keep this lecture
workshop-only unless the course owner explicitly changes the scope.

## Safety And Editing

Do not treat `CLAUDE_md_template.md`, example skills, example subagents, or hooks
as active policy unless they are intentionally copied into a project. Preserve
read-only, data-privacy, and verification examples.

Generated/reference artifacts are checked in:

- `code/data/synthetic_panel.csv`
- `code/outputs/mincer_table.tex`
- `code/outputs/mincer_figure.pdf`

Do not overwrite them casually. If changing exercises, update prompts,
solutions, slides, and README together.

Several exercise prompts reference `toolkit/...` paths; in this repository the
equivalent materials live under `lectures/lecture_06_agentic_programming/code/`.

## Verification

From repo root:

```bash
python lectures/lecture_06_agentic_programming/code/generate_synthetic_data.py
python lectures/lecture_06_agentic_programming/code/mincer_demo.py
```

From this lecture directory:

```bash
python code/generate_synthetic_data.py
python code/mincer_demo.py
```

`mincer_demo.py` requires optional extras not listed in the root environment:
`statsmodels` and `wooldridge`. Install them only when running that demo, e.g.
`pip install statsmodels wooldridge`. The expected education
coefficient is about `0.0841`, with a broad sanity assertion around
`0.07 < b1 < 0.10`.
