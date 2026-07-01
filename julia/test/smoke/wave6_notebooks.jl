using Test

repo_root = normpath(joinpath(@__DIR__, "..", "..", ".."))
wave6_notebooks = [
    "lectures/lecture_16_climate_economics_iams/code_julia/lecture_16_01_Climate_Exercise_Lux.jl",
    "lectures/lecture_16_climate_economics_iams/code_julia/lecture_16_02_DICE_DEQN_Library_Port_Lux.jl",
    "lectures/lecture_16_climate_economics_iams/code_julia/lecture_16_03_Stochastic_DICE_DEQN_Lux.jl",
]

all_finite_numbers(x::Number) = isfinite(x)
all_finite_numbers(x::AbstractArray{<:Number}) = all(isfinite, x)
all_finite_numbers(x::AbstractArray) = all(all_finite_numbers, x)
all_finite_numbers(x::NamedTuple) = all(all_finite_numbers, values(x))
all_finite_numbers(x::Tuple) = all(all_finite_numbers, x)
all_finite_numbers(x::Dict) = all(all_finite_numbers, values(x))
all_finite_numbers(x) = false

@testset "Wave 6 Pluto notebook smoke includes" begin
    for relative_path in wave6_notebooks
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
