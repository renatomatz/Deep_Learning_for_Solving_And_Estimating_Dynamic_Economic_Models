# Execute-smoke for the converted Jupyter notebooks (Wave 6).
#
# Run under the smoke environment:
#     julia --project=julia/test/smoke julia/test/smoke/wave6_notebooks.jl

include("smoke_common.jl")

smoke_wave("Wave 6 notebook execute-smoke", [
    "lectures/lecture_16_climate_economics_iams/code_julia/lecture_16_01_Climate_Exercise_Lux.ipynb",
    "lectures/lecture_16_climate_economics_iams/code_julia/lecture_16_02_DICE_DEQN_Library_Port_Lux.ipynb",
    "lectures/lecture_16_climate_economics_iams/code_julia/lecture_16_03_Stochastic_DICE_DEQN_Lux.ipynb",
])
