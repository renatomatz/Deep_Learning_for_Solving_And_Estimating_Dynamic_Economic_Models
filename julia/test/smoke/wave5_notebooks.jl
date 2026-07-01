# Execute-smoke for the converted Jupyter notebooks (Wave 5).
#
# Run under the smoke environment:
#     julia --project=julia/test/smoke julia/test/smoke/wave5_notebooks.jl

include("smoke_common.jl")

smoke_wave("Wave 5 notebook execute-smoke", [
    "lectures/lecture_14_surrogates_and_gps/code_julia/lecture_14_01_Surrogate_Primer_Lux.ipynb",
    "lectures/lecture_14_surrogates_and_gps/code_julia/lecture_14_02_GP_and_BAL_Lux.ipynb",
    "lectures/lecture_14_surrogates_and_gps/code_julia/lecture_14_04_GP_Value_Function_Iteration_Lux.ipynb",
    "lectures/lecture_14_surrogates_and_gps/code_julia/lecture_14_05_Active_Subspace_2D_Lux.ipynb",
    "lectures/lecture_14_surrogates_and_gps/code_julia/lecture_14_06_Active_Subspace_10D_Lux.ipynb",
    "lectures/lecture_14_surrogates_and_gps/code_julia/lecture_14_07_Active_Subspace_Nonlinear_Lux.ipynb",
    "lectures/lecture_14_surrogates_and_gps/code_julia/lecture_14_08_Deep_Kernel_Learning_Lux.ipynb",
    "lectures/lecture_14_surrogates_and_gps/code_julia/lecture_14_09_Deep_Active_Subspace_Ridge_Lux.ipynb",
    "lectures/lecture_14_surrogates_and_gps/code_julia/lecture_14_10_Deep_AS_vs_Linear_AS_Borehole_Lux.ipynb",
    "lectures/lecture_15_structural_estimation_smm/code_julia/lecture_15_03_Structural_Estimation_BM_Lux.ipynb",
    "lectures/lecture_15_structural_estimation_smm/code_julia/lecture_15_03b_Structural_Estimation_BM_Joint_Lux.ipynb",
])
