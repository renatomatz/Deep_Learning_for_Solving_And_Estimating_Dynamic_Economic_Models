using Test

repo_root = normpath(joinpath(@__DIR__, "..", "..", ".."))
wave2_notebooks = [
    "lectures/lecture_02_intro_deep_learning/code_julia/lecture_02_03_Double_Descent_Lux.jl",
    "lectures/lecture_02_intro_deep_learning/code_julia/lecture_02_04_Gentle_DNN_Lux.jl",
    "lectures/lecture_02_intro_deep_learning/code_julia/lecture_02_05_Training_Instrumentation_Lux.jl",
    "lectures/lecture_02_intro_deep_learning/code_julia/lecture_02_06_Lux_Training_Fundamentals.jl",
    "lectures/lecture_02_intro_deep_learning/code_julia/lecture_02_07_Genz_Approximation_and_Loss_Functions_Lux.jl",
    "lectures/lecture_02_intro_deep_learning/code_julia/lecture_02_08_MLP_LSTM_Transformer_Edgeworth_Cycles_Lux.jl",
    "lectures/lecture_02_intro_deep_learning/code_julia/lecture_02_09_Transformer_InContext_AR1_Lux.jl",
    "lectures/lecture_03_deep_equilibrium_nets/code_julia/lecture_03_03_DEQN_Exercises_Blanks_Lux.jl",
    "lectures/lecture_03_deep_equilibrium_nets/code_julia/lecture_03_04_DEQN_Exercises_Solutions_Lux.jl",
    "lectures/lecture_03_deep_equilibrium_nets/code_julia/lecture_03_05_StochasticBM_LossComparison_Lux.jl",
    "lectures/lecture_04_irbc_with_deqns/code_julia/lecture_04_01_IRBC_DEQN_smooth_Lux.jl",
    "lectures/lecture_04_irbc_with_deqns/code_julia/lecture_04_02_IRBC_DEQN_irreversible_Lux.jl",
    "lectures/lecture_05_nas_loss_normalization/code_julia/lecture_05_02_NAS_Random_Search_10D_Lux.jl",
    "lectures/lecture_05_nas_loss_normalization/code_julia/lecture_05_03_NAS_RandomSearch_Hyperband_Lux.jl",
    "lectures/lecture_05_nas_loss_normalization/code_julia/lecture_05_04_Loss_Normalization_Lux.jl",
    "lectures/lecture_05_nas_loss_normalization/code_julia/lecture_05_05_IRBC_Exercise_Lux.jl",
]

@testset "Wave 2 Pluto notebook smoke includes" begin
    for relative_path in wave2_notebooks
        path = joinpath(repo_root, relative_path)
        @test isfile(path)
        source = read(path, String)
        @test startswith(source, "### A Pluto.jl notebook ###")
        @test occursin("# ╔═╡ Cell order:", source)
        cell_ids = [m.captures[1] for m in eachmatch(r"# [╔╟]═╡ ([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})", source)]
        @test length(cell_ids) == length(unique(cell_ids))
        @test include(path) !== nothing
    end
end
