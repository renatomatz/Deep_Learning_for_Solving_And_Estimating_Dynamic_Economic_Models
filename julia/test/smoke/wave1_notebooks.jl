using Test

repo_root = normpath(joinpath(@__DIR__, "..", "..", ".."))
wave1_notebooks = [
    "lectures/lecture_02_intro_deep_learning/code_julia/lecture_02_00_Lux_Pluto_orientation.jl",
    "lectures/lecture_02_intro_deep_learning/code_julia/lecture_02_01_BasicML_intro_Lux.jl",
    "lectures/lecture_02_intro_deep_learning/code_julia/lecture_02_02_GradientDescent_and_StochasticGradientDescent_Lux.jl",
    "lectures/lecture_03_deep_equilibrium_nets/code_julia/lecture_03_01_Brock_Mirman_1972_DEQN_Lux.jl",
    "lectures/lecture_03_deep_equilibrium_nets/code_julia/lecture_03_02_Brock_Mirman_Uncertainty_DEQN_Lux.jl",
    "lectures/lecture_07_autodiff_for_deqns/code_julia/lecture_07_01_AutoDiff_Analytical_Examples_Lux.jl",
    "lectures/lecture_07_autodiff_for_deqns/code_julia/lecture_07_02_Brock_Mirman_AutoDiff_DEQN_Lux.jl",
    "lectures/lecture_07_autodiff_for_deqns/code_julia/lecture_07_03_Brock_Mirman_Uncertainty_AutoDiff_DEQN_Lux.jl",
    "lectures/lecture_07_autodiff_for_deqns/code_julia/lecture_07_04_IRBC_AutoDiff_DEQN_Lux.jl",
    "lectures/lecture_11_pinns/code_julia/lecture_11_01_ODE_PINN_ZeroBCs_Lux.jl",
]

@testset "Wave 1 Pluto notebook smoke includes" begin
    for relative_path in wave1_notebooks
        path = joinpath(repo_root, relative_path)
        @test isfile(path)
        @test include(path) !== nothing
    end
end
