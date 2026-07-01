# AGENTS.md

## Purpose

This directory contains example skill specifications for Lecture 06. They are
teaching artifacts, not active skills unless installed or copied into an agent
environment.

This workshop folder is not part of the Julia/Lux/Pluto translation track. Do
not add `code_julia/` material or Lux-specific guidance inside these example
skill docs unless the Lecture 06 scope changes.

## Conventions

Each skill uses a `SKILL.md` spec with front matter, command name, workflow,
constraints, and expected outputs. Preserve those sections when editing.

- `example_skill/SKILL.md` defines `/data-diagnostics` and must not modify
  source CSVs.
- `strategic_revision/SKILL.md` defines `/strategic-revision`; it reads reports
  and manuscript context and writes only `notes/revision_plan.md`.

The directory name `strategic_revision` uses an underscore, while the command
name uses a hyphen. Keep that distinction clear.
