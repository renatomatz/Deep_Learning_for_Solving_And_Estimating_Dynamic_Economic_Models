# AGENTS.md

## Purpose

This directory holds repository-level shared assets and the central attribution
registry. It is not the full figure library for the course.

## Map

- `hero/` contains shared cover and hero artwork.
- `hero/deep_learning_dynamic_models_hero.png` is used by the root `README.md`
  and by the companion-script cover.
- `attributions.yml` is the repo-relative registry for borrowed, adapted, or
  uncertain third-party media.

## Conventions

Keep attribution paths repo-relative. Course-owned figures are generally CC0
under `LICENSE`; third-party media keeps its upstream license.

For any added or replaced borrowed/adapted asset, update `attributions.yml` with
source, authors, year, license, redistribution status, and notes. Treat
`redistribution_allowed: review_required` as blocked for new reuse until
provenance is confirmed or the asset is replaced/redrawn.

Do not move or rename hero assets without updating both root `README.md` and the
companion-script LaTeX source.

## Related Asset Locations

Figures also live under lecture-local `slides/fig`, `slides/figures`,
`slides/images`, lecture `figures/`, and `lecture_script/fig`. Some of those are
shared build inputs for slides or the companion script.

## Known Issues

- `attributions.yml` references `COPYRIGHT_AUDIT.csv`, but that file is not
  present in this checkout.
- `attributions.yml` contains `damge.png` for Lecture 16, while Lecture 16 uses
  `damage.png`; Lecture 18 has `damge.png`.
- Several image sets are duplicated across lectures. Update all copies or
  document intentional divergence.
