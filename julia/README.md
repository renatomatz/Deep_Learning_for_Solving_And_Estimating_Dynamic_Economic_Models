# DLEF Julia/Lux/Pluto Track

This directory contains the shared Julia project for the Lux-native Pluto
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

Run translated-notebook smoke checks separately when touching lecture `code_julia/` files:

```bash
julia --project=julia --startup-file=no julia/test/smoke/wave3_notebooks.jl
julia --project=julia --startup-file=no julia/test/smoke/wave4_notebooks.jl
julia --project=julia --startup-file=no julia/test/smoke/wave5_notebooks.jl
```

## Pluto Notebooks

Student-facing translated notebooks live in lecture-local `code_julia/`
directories. They activate this shared project explicitly, for example:

```julia
import Pkg
Pkg.activate(joinpath(@__DIR__, "..", "..", "..", "julia"))
```

Launch Pluto from the shared project when editing or running these notebooks:

```bash
julia --project=julia --startup-file=no -e 'using Pluto; Pluto.run()'
```

Open the notebook from its lecture `code_julia/` directory so local relative
paths continue to resolve.
