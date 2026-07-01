# Shared execute-smoke driver for the converted Jupyter notebooks.
#
# Run each wave under the smoke environment so NBInclude is available, e.g.
#     julia --project=julia/test/smoke julia/test/smoke/wave1_notebooks.jl
#
# Each notebook is executed with NBInclude in a THROWAWAY module: the previews
# share global names and struct/const definitions, which collide in a shared
# session (a cross-notebook clash, not a defect), so one module per notebook
# isolates them. NBInclude runs the code cells in order and returns the value of
# the last code cell (the diagnostics NamedTuple). We only assert the notebook
# executes and returns something -- NOT `all_finite_numbers`, because several
# notebooks legitimately return String/Symbol fields (run_mode = "smoke",
# saving_gate_status = :pass, prose notes). The notebook's first code cell
# re-activates the main julia/ project for its own dependencies.

using Test
using NBInclude

const SMOKE_REPO_ROOT = normpath(joinpath(@__DIR__, "..", "..", ".."))

function smoke_run_notebook(rel_path::AbstractString)
    path = joinpath(SMOKE_REPO_ROOT, rel_path)
    modname = Symbol("Smoke_" * replace(basename(path), r"[^A-Za-z0-9]" => "_"))
    mod = Module(modname)
    # softscope=true matches how a Julia (IJulia) kernel runs a notebook's
    # top-level cells, so top-level loops that update globals behave as written.
    return NBInclude.nbinclude(mod, path; softscope = true)
end

function smoke_wave(name::AbstractString, notebooks)
    @testset "$name" begin
        for rel in notebooks
            @testset "$(basename(rel))" begin
                @test isfile(joinpath(SMOKE_REPO_ROOT, rel))
                result = smoke_run_notebook(rel)
                @test result !== nothing
            end
        end
    end
end
