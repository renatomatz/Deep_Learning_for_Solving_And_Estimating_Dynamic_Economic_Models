using Test

repo_root = normpath(joinpath(@__DIR__, "..", "..", ".."))
wave5_notebooks = [
    "lectures/lecture_14_surrogates_and_gps/code_julia/lecture_14_01_Surrogate_Primer_Lux.jl",
    "lectures/lecture_14_surrogates_and_gps/code_julia/lecture_14_02_GP_and_BAL_Lux.jl",
    "lectures/lecture_14_surrogates_and_gps/code_julia/lecture_14_04_GP_Value_Function_Iteration_Lux.jl",
    "lectures/lecture_14_surrogates_and_gps/code_julia/lecture_14_05_Active_Subspace_2D_Lux.jl",
    "lectures/lecture_14_surrogates_and_gps/code_julia/lecture_14_06_Active_Subspace_10D_Lux.jl",
    "lectures/lecture_14_surrogates_and_gps/code_julia/lecture_14_07_Active_Subspace_Nonlinear_Lux.jl",
    "lectures/lecture_14_surrogates_and_gps/code_julia/lecture_14_08_Deep_Kernel_Learning_Lux.jl",
    "lectures/lecture_14_surrogates_and_gps/code_julia/lecture_14_09_Deep_Active_Subspace_Ridge_Lux.jl",
    "lectures/lecture_14_surrogates_and_gps/code_julia/lecture_14_10_Deep_AS_vs_Linear_AS_Borehole_Lux.jl",
    "lectures/lecture_15_structural_estimation_smm/code_julia/lecture_15_03_Structural_Estimation_BM_Lux.jl",
    "lectures/lecture_15_structural_estimation_smm/code_julia/lecture_15_03b_Structural_Estimation_BM_Joint_Lux.jl",
]

all_finite_numbers(x::Number) = isfinite(x)
all_finite_numbers(x::AbstractArray{<:Number}) = all(isfinite, x)
all_finite_numbers(x::NamedTuple) = all(all_finite_numbers, values(x))
all_finite_numbers(x::Tuple) = all(all_finite_numbers, x)
all_finite_numbers(x) = true

@testset "Wave 5 Pluto notebook smoke includes" begin
    for relative_path in wave5_notebooks
        path = joinpath(repo_root, relative_path)
        @test isfile(path)
        source = read(path, String)
        @test startswith(source, "### A Pluto.jl notebook ###")
        @test occursin("# ╔═╡ Cell order:", source)
        cell_ids = [m.captures[1] for m in eachmatch(r"# [╔╟]═╡ ([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})", source)]
        @test length(cell_ids) == length(unique(cell_ids))
        result = include(path)
        @test result !== nothing
        @test all_finite_numbers(result)
    end
end
