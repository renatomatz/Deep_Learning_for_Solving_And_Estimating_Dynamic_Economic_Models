# AGENTS.md

## Purpose

This directory contains example subagent templates for Lecture 06. They are
teaching artifacts with YAML front matter and role-specific workflows, not active
repository policy by themselves.

This workshop folder is outside the Julia/Lux translation track. Keep the
templates platform- and workflow-focused; do not add Lux or `code_julia/`
translation guidance here unless the Lecture 06 scope changes.

## Template Boundaries

Read-only templates:

- `verifier.md`
- `code_reviewer.md`
- `econometrics_reviewer.md`
- `backtest_validator.md`

Write-capable templates:

- `test_writer.md`
- `doc_generator.md`
- `monte_carlo_designer.md`

Preserve each template's role boundary, approval/constraint steps, evidence
requirements, and verdict format. Keep one responsibility per subagent.

When adapting these examples elsewhere, convert tool-specific front matter such
as `model` or `tools` to the target agent platform rather than assuming it is
portable.
