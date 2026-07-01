# AGENTS.md

## Purpose

`readings/` is the course reading index. It is intentionally link-only: do not
add or commit PDFs, scans, or redistributed copies of papers.

## Files

- `README.md` explains the directory purpose and no-PDF policy.
- `bibliography.bib` is the shared course BibTeX database for references cited in
  the script or slides.
- `links_by_lecture/lecture_NN.md` files provide curated reading links and short
  lecture-specific annotations.

## Conventions

When updating lecture reading files, keep the existing scaffold:

- `# Readings, Lecture NN: Topic`
- `Default policy: link only.`
- Topic-specific `##` sections with concise reading bullets
- `## Companion lecture script`
- `## Bibliography`

Use readable prose citations in lecture files rather than BibTeX keys. Prefer
source links such as DOI, arXiv, publisher pages, SSRN, author pages, or official
online editions. Keep annotations brief and explain why the item matters for the
lecture.

## Bibliography

Treat `bibliography.bib` as a shared course database. Preserve existing keys
unless a citation change requires otherwise. Expect mixed imported BibTeX style,
including varied entry types, capitalization, DOI formats, and arXiv fields. Do
not reformat the whole bibliography for a small reference update.

## Gotchas

Some lecture bullets intentionally have no inline URL; check `bibliography.bib`
before assuming a link is missing. Keep relative links from lecture files valid
from `readings/links_by_lecture/`.
