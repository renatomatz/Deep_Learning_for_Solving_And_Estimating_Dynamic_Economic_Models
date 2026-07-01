using Test

repo_root = normpath(joinpath(@__DIR__, "..", "..", ".."))
wave3_notebooks = [
    "lectures/lecture_08_olg_models_deqns/code_julia/lecture_08_07_OLG_Analytic_DEQN_exogenous_Lux.jl",
    "lectures/lecture_08_olg_models_deqns/code_julia/lecture_08_08_OLG_Analytic_DEQN_persistent_Lux.jl",
    "lectures/lecture_08_olg_models_deqns/code_julia/lecture_08_09_OLG_Benchmark_DEQN_exogenous_Lux.jl",
    "lectures/lecture_08_olg_models_deqns/code_julia/lecture_08_10_OLG_Benchmark_DEQN_persistent_Lux.jl",
    "lectures/lecture_08_olg_models_deqns/code_julia/lecture_08_11_OLG_Exercise_Lux.jl",
    "lectures/lecture_09_heterogeneous_agents_youngs_method/code_julia/lecture_09_10_Youngs_Method_Examples_Lux.jl",
    "lectures/lecture_09_heterogeneous_agents_youngs_method/code_julia/lecture_09_11_Continuum_of_Agents_DEQN_Lux.jl",
    "lectures/lecture_09_heterogeneous_agents_youngs_method/code_julia/lecture_09_12_KrusellSmith_DeepLearning_Lux.jl",
    "lectures/lecture_10_sequence_space_deqns/code_julia/lecture_10_05_SequenceSpace_BrockMirman_Lux.jl",
    "lectures/lecture_10_sequence_space_deqns/code_julia/lecture_10_05b_SequenceSpace_IRBC_Lux.jl",
    "lectures/lecture_10_sequence_space_deqns/code_julia/lecture_10_06_SequenceSpace_KrusellSmith_Lux.jl",
    "lectures/lecture_10_sequence_space_deqns/code_julia/lecture_10_KrusellSmith_Tutorial_CPU_Lux.jl",
]

all_finite_numbers(x::Number) = isfinite(x)
all_finite_numbers(x::AbstractArray{<:Number}) = all(isfinite, x)
all_finite_numbers(x::NamedTuple) = all(all_finite_numbers, values(x))
all_finite_numbers(x::Tuple) = all(all_finite_numbers, x)
all_finite_numbers(x) = true

@testset "Wave 3 Pluto notebook smoke includes" begin
    for relative_path in wave3_notebooks
        path = joinpath(repo_root, relative_path)
        @test isfile(path)
        source = read(path, String)
        @test startswith(source, "### A Pluto.jl notebook ###")
        @test occursin("# ╔═╡ Cell order:", source)
        cell_ids = [m.captures[1] for m in eachmatch(r"# [╔╟]═╡ ([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})", source)]
        @test length(cell_ids) == length(unique(cell_ids))
        result = include(path)
        @test result !== nothing
        @test all_finite_numbers(result)
    end
end
