# Execute-smoke for the converted Jupyter notebooks (Wave 1).
#
# Run under the smoke environment:
#     julia --project=julia/test/smoke julia/test/smoke/wave1_notebooks.jl

include("smoke_common.jl")

smoke_wave("Wave 1 notebook execute-smoke", [
    "lectures/lecture_02_intro_deep_learning/code_julia/lecture_02_00_Lux_orientation.ipynb",
    "lectures/lecture_02_intro_deep_learning/code_julia/lecture_02_01_BasicML_intro_Lux.ipynb",
    "lectures/lecture_02_intro_deep_learning/code_julia/lecture_02_02_GradientDescent_and_StochasticGradientDescent_Lux.ipynb",
    "lectures/lecture_03_deep_equilibrium_nets/code_julia/lecture_03_01_Brock_Mirman_1972_DEQN_Lux.ipynb",
    "lectures/lecture_03_deep_equilibrium_nets/code_julia/lecture_03_02_Brock_Mirman_Uncertainty_DEQN_Lux.ipynb",
    "lectures/lecture_07_autodiff_for_deqns/code_julia/lecture_07_01_AutoDiff_Analytical_Examples_Lux.ipynb",
    "lectures/lecture_07_autodiff_for_deqns/code_julia/lecture_07_02_Brock_Mirman_AutoDiff_DEQN_Lux.ipynb",
    "lectures/lecture_07_autodiff_for_deqns/code_julia/lecture_07_03_Brock_Mirman_Uncertainty_AutoDiff_DEQN_Lux.ipynb",
    "lectures/lecture_07_autodiff_for_deqns/code_julia/lecture_07_04_IRBC_AutoDiff_DEQN_Lux.ipynb",
    "lectures/lecture_11_pinns/code_julia/lecture_11_01_ODE_PINN_ZeroBCs_Lux.ipynb",
])
