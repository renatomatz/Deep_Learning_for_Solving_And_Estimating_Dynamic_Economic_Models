# Execute-smoke for the converted Jupyter notebooks (Wave 2).
#
# Run under the smoke environment:
#     julia --project=julia/test/smoke julia/test/smoke/wave2_notebooks.jl

include("smoke_common.jl")

smoke_wave("Wave 2 notebook execute-smoke", [
    "lectures/lecture_02_intro_deep_learning/code_julia/lecture_02_03_Double_Descent_Lux.ipynb",
    "lectures/lecture_02_intro_deep_learning/code_julia/lecture_02_04_Gentle_DNN_Lux.ipynb",
    "lectures/lecture_02_intro_deep_learning/code_julia/lecture_02_05_Training_Instrumentation_Lux.ipynb",
    "lectures/lecture_02_intro_deep_learning/code_julia/lecture_02_06_Lux_Training_Fundamentals.ipynb",
    "lectures/lecture_02_intro_deep_learning/code_julia/lecture_02_07_Genz_Approximation_and_Loss_Functions_Lux.ipynb",
    "lectures/lecture_02_intro_deep_learning/code_julia/lecture_02_08_MLP_LSTM_Transformer_Edgeworth_Cycles_Lux.ipynb",
    "lectures/lecture_02_intro_deep_learning/code_julia/lecture_02_09_Transformer_InContext_AR1_Lux.ipynb",
    "lectures/lecture_03_deep_equilibrium_nets/code_julia/lecture_03_03_DEQN_Exercises_Blanks_Lux.ipynb",
    "lectures/lecture_03_deep_equilibrium_nets/code_julia/lecture_03_04_DEQN_Exercises_Solutions_Lux.ipynb",
    "lectures/lecture_03_deep_equilibrium_nets/code_julia/lecture_03_05_StochasticBM_LossComparison_Lux.ipynb",
    "lectures/lecture_04_irbc_with_deqns/code_julia/lecture_04_01_IRBC_DEQN_smooth_Lux.ipynb",
    "lectures/lecture_04_irbc_with_deqns/code_julia/lecture_04_02_IRBC_DEQN_irreversible_Lux.ipynb",
    "lectures/lecture_05_nas_loss_normalization/code_julia/lecture_05_02_NAS_Random_Search_10D_Lux.ipynb",
    "lectures/lecture_05_nas_loss_normalization/code_julia/lecture_05_03_NAS_RandomSearch_Hyperband_Lux.ipynb",
    "lectures/lecture_05_nas_loss_normalization/code_julia/lecture_05_04_Loss_Normalization_Lux.ipynb",
    "lectures/lecture_05_nas_loss_normalization/code_julia/lecture_05_05_IRBC_Exercise_Lux.ipynb",
])
