# AGENTS.md

## Purpose

Lecture 17 covers deep uncertainty quantification and Pareto-improving
climate-policy design: stochastic CDICE-DEQN, GP surrogates, Bayesian active
learning, Sobol/Shapley sensitivity analysis, and constrained carbon-tax search.

## Map

- Start with `README.md`.
- `slides/lecture_17_deep_uq_pareto_policy.tex` is the slide source.
- `slides/fig/` contains a large climate/UQ/Pareto-policy figure set.
- `slides/bib_econ.bib` is local to this deck.

## Code Boundary

This lecture intentionally has no local `code/` or `code_julia/` directory. The
supporting code is maintained in the external research repository linked from
`README.md`:
`sischei/JPE_Macro_Using_ML_to_compute_constrained_optimal_carbon_tax_rules`.

Do not invent local Python notebooks, Julia/Lux/Pluto notebooks, replacement
code, or stubs here unless the user explicitly changes the scope. If code
changes are requested, first clarify whether they belong in this teaching
repository or in the external research repository. The Julia translation track
does not extend Lecture 17 locally.

Gotcha: the TeX has a stale companion-note reference to "Notebook 09", but there
is no local Lecture 17 notebook. Trust the README's external-code boundary.

## Media And Attribution

Many figures are copied or related to Lecture 16 and Lecture 18 climate assets.
Check `../../assets/attributions.yml` before reusing, moving, or replacing
third-party-looking images.

Compile slides from `slides/`; the deck uses `\graphicspath{{fig/}}`, `natbib`,
and local `bib_econ.bib`, so a full PDF rebuild needs a BibTeX pass.
