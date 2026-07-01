# Execute-smoke for the converted Jupyter notebooks (Wave 3).
#
# Run under the smoke environment:
#     julia --project=julia/test/smoke julia/test/smoke/wave3_notebooks.jl

include("smoke_common.jl")

smoke_wave("Wave 3 notebook execute-smoke", [
    "lectures/lecture_08_olg_models_deqns/code_julia/lecture_08_07_OLG_Analytic_DEQN_exogenous_Lux.ipynb",
    "lectures/lecture_08_olg_models_deqns/code_julia/lecture_08_08_OLG_Analytic_DEQN_persistent_Lux.ipynb",
    "lectures/lecture_08_olg_models_deqns/code_julia/lecture_08_09_OLG_Benchmark_DEQN_exogenous_Lux.ipynb",
    "lectures/lecture_08_olg_models_deqns/code_julia/lecture_08_10_OLG_Benchmark_DEQN_persistent_Lux.ipynb",
    "lectures/lecture_08_olg_models_deqns/code_julia/lecture_08_11_OLG_Exercise_Lux.ipynb",
    "lectures/lecture_09_heterogeneous_agents_youngs_method/code_julia/lecture_09_10_Youngs_Method_Examples_Lux.ipynb",
    "lectures/lecture_09_heterogeneous_agents_youngs_method/code_julia/lecture_09_11_Continuum_of_Agents_DEQN_Lux.ipynb",
    "lectures/lecture_09_heterogeneous_agents_youngs_method/code_julia/lecture_09_12_KrusellSmith_DeepLearning_Lux.ipynb",
    "lectures/lecture_10_sequence_space_deqns/code_julia/lecture_10_05_SequenceSpace_BrockMirman_Lux.ipynb",
    "lectures/lecture_10_sequence_space_deqns/code_julia/lecture_10_05b_SequenceSpace_IRBC_Lux.ipynb",
    "lectures/lecture_10_sequence_space_deqns/code_julia/lecture_10_06_SequenceSpace_KrusellSmith_Lux.ipynb",
    "lectures/lecture_10_sequence_space_deqns/code_julia/lecture_10_KrusellSmith_Tutorial_CPU_Lux.ipynb",
])
