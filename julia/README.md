# DLEF Julia/Lux/Jupyter Track

This directory contains the shared Julia project for the Lux-native Jupyter
translation of the course notebooks.

## Setup

Instantiate the shared environment from the repository root:

```bash
julia --project=julia --startup-file=no -e 'using Pkg; Pkg.instantiate()'
```

Run the shared helper tests with:

```bash
julia --project=julia --startup-file=no -e 'using Pkg; Pkg.test()'
```

Run translated-notebook execute-smoke checks separately when touching lecture
`code_julia/` files. The smoke scripts run under their own environment
(`julia/test/smoke`), which carries `NBInclude` — the only extra dependency —
so the notebooks can be executed in-process without a Jupyter kernel:

```bash
julia --project=julia/test/smoke --startup-file=no julia/test/smoke/wave3_notebooks.jl
julia --project=julia/test/smoke --startup-file=no julia/test/smoke/wave4_notebooks.jl
julia --project=julia/test/smoke --startup-file=no julia/test/smoke/wave5_notebooks.jl
```

## Jupyter Notebooks

Student-facing translated notebooks live in lecture-local `code_julia/`
directories as Jupyter `.ipynb` files, using the same VSCode + Jupyter tooling
as the Python `code/` track. Each notebook's first code cell activates this
shared project explicitly, for example:

```julia
import Pkg
Pkg.activate(joinpath(@__DIR__, "..", "..", "..", "julia"))
```

Open a notebook from its lecture `code_julia/` directory so its relative
`Pkg.activate` path resolves. In VSCode the Julia extension runs `.ipynb` Julia
notebooks directly; for classic Jupyter, register a Julia kernel (`IJulia`) in
your own environment. The notebooks are committed **output-free** — avoid saving
executed cell outputs.

Automated validation does not need a Jupyter kernel: the smoke scripts above
execute each notebook in-process with `NBInclude` (which pulls only JSON +
SoftGlobalScope, not IJulia).
