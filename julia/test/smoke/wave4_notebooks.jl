using Test

repo_root = normpath(joinpath(@__DIR__, "..", "..", ".."))
wave4_notebooks = [
    "lectures/lecture_11_pinns/code_julia/lecture_11_02_ODE_PINN_SoftVsHardBCs_Lux.jl",
    "lectures/lecture_11_pinns/code_julia/lecture_11_03_PDE_PINN_Poisson2D_Lux.jl",
    "lectures/lecture_11_pinns/code_julia/lecture_11_04_Cake_Eating_HJB_PINN_Lux.jl",
    "lectures/lecture_11_pinns/code_julia/lecture_11_05_Black_Scholes_PINN_Lux.jl",
    "lectures/lecture_13_continuous_time_ha_numerics/code_julia/lecture_13_08_Aiyagari_Continuous_Time_FD_and_PINN_Lux.jl",
]

all_finite_numbers(x::Number) = isfinite(x)
all_finite_numbers(x::AbstractArray{<:Number}) = all(isfinite, x)
all_finite_numbers(x::NamedTuple) = all(all_finite_numbers, values(x))
all_finite_numbers(x::Tuple) = all(all_finite_numbers, x)
all_finite_numbers(x) = true

@testset "Wave 4 Pluto notebook smoke includes" begin
    for relative_path in wave4_notebooks
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
