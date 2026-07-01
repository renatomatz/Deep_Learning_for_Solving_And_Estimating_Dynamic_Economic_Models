# AGENTS.md

## Purpose

`lecture_script/` contains the textbook-length companion manuscript. The
canonical source is
`Deep_Learning_for_Solving_And_Estimating_Dynamic_Economic_Models.tex`; the PDF
is a built artifact that is checked in for readers.

## Navigation

- `Deep_Learning_for_Solving_And_Estimating_Dynamic_Economic_Models.tex` is one
  monolithic LaTeX source file. There are no included chapter `.tex` files.
- `script_to_lectures.md` maps chapters and appendices to lecture folders and
  notebooks.
- `glossary.md` is a grep-friendly mirror of Appendix A.
- `fig/` and `fig/chapter11/` hold manuscript figures.
- The cover image comes from `../assets/hero/`.
- The bibliography is external: `../readings/bibliography.bib`.

## Editing

Search by `\chapter{...}`, `\section{...}`, labels, or figure labels. Preserve
the existing LaTeX stack and style: `natbib`/`apalike`, TikZ/PGFPlots,
`tcolorbox`, `listings`, and the current color conventions.

When adding citations, add entries to `../readings/bibliography.bib` and keep the
reading guidance in repo-root `readings/` consistent if the citation is tied to a lecture.

Be careful with extensionless `\includegraphics` calls when both `.pdf` and
`.png` files exist; pdfLaTeX will usually prefer the PDF.

## Build

Preferred build from this directory:

```bash
latexmk -pdf -interaction=nonstopmode -halt-on-error Deep_Learning_for_Solving_And_Estimating_Dynamic_Economic_Models.tex
```

Manual fallback:

```bash
pdflatex -interaction=nonstopmode -halt-on-error Deep_Learning_for_Solving_And_Estimating_Dynamic_Economic_Models.tex
bibtex Deep_Learning_for_Solving_And_Estimating_Dynamic_Economic_Models
pdflatex -interaction=nonstopmode -halt-on-error Deep_Learning_for_Solving_And_Estimating_Dynamic_Economic_Models.tex
pdflatex -interaction=nonstopmode -halt-on-error Deep_Learning_for_Solving_And_Estimating_Dynamic_Economic_Models.tex
```

Do not treat generated `.aux`, `.bbl`, `.blg`, `.log`, `.out`, `.toc`, or similar
files as source.
