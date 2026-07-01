# Execute-smoke for the converted Jupyter notebooks (Wave 4).
#
# Run under the smoke environment:
#     julia --project=julia/test/smoke julia/test/smoke/wave4_notebooks.jl

include("smoke_common.jl")

smoke_wave("Wave 4 notebook execute-smoke", [
    "lectures/lecture_11_pinns/code_julia/lecture_11_02_ODE_PINN_SoftVsHardBCs_Lux.ipynb",
    "lectures/lecture_11_pinns/code_julia/lecture_11_03_PDE_PINN_Poisson2D_Lux.ipynb",
    "lectures/lecture_11_pinns/code_julia/lecture_11_04_Cake_Eating_HJB_PINN_Lux.ipynb",
    "lectures/lecture_11_pinns/code_julia/lecture_11_05_Black_Scholes_PINN_Lux.ipynb",
    "lectures/lecture_13_continuous_time_ha_numerics/code_julia/lecture_13_08_Aiyagari_Continuous_Time_FD_and_PINN_Lux.ipynb",
])
