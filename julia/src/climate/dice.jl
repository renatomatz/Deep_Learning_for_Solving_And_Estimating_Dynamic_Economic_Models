export DICEClimateParams,
    dice_bau_emissions,
    dice_years,
    simulate_dice_carbon,
    simulate_dice_temperature,
    dice_damage_percent,
    simulate_dice_climate_exercise

using Statistics: mean

Base.@kwdef struct DICEClimateParams
    Phi::Matrix{Float64} = [
        0.912 0.0444 0.0
        0.088 0.9055 0.0033
        0.0 0.0501 0.9967
    ]
    M0::Vector{Float64} = [851.0, 628.0, 1740.0]
    n_steps::Int = 30
    emissions0::Float64 = 50.0
    emissions_decline::Float64 = 0.01
    start_year::Int = 2015
    dt::Float64 = 5.0
    M_AT_1750::Float64 = 588.0
    eta::Float64 = 3.8 / log(2.0)
    feedback::Float64 = 3.8 / 3.0
    xi::Float64 = 0.015
    pi2::Float64 = 0.00236
    T0::Float64 = 1.1
end

function dice_bau_emissions(params::DICEClimateParams = DICEClimateParams())
    steps = collect(0:(params.n_steps - 1))
    return params.emissions0 .* exp.(-params.emissions_decline .* steps .* params.dt)
end

function dice_years(params::DICEClimateParams = DICEClimateParams())
    return params.start_year .+ params.dt .* collect(0:params.n_steps)
end

function simulate_dice_carbon(emissions; params::DICEClimateParams = DICEClimateParams())
    length(emissions) == params.n_steps ||
        throw(DimensionMismatch("emissions must have $(params.n_steps) entries"))
    size(params.Phi) == (3, 3) || throw(DimensionMismatch("Phi must be 3-by-3"))
    length(params.M0) == 3 || throw(DimensionMismatch("M0 must have three carbon boxes"))

    M = Matrix{Float64}(undef, 3, params.n_steps + 1)
    M[:, 1] .= params.M0
    for t in 1:params.n_steps
        M[:, t + 1] .= params.Phi * M[:, t]
        M[1, t + 1] += emissions[t]
    end
    return M
end

function simulate_dice_temperature(carbon_path::AbstractMatrix; params::DICEClimateParams = DICEClimateParams())
    size(carbon_path, 1) == 3 || throw(DimensionMismatch("carbon_path must be 3-by-time"))
    n = size(carbon_path, 2) - 1
    T = Vector{Float64}(undef, n + 1)
    T[1] = params.T0
    for t in 1:n
        forcing = params.eta * log2(max(carbon_path[1, t + 1], eps()) / params.M_AT_1750)
        T[t + 1] = T[t] + params.xi * (forcing - params.feedback * T[t]) * params.dt
    end
    return T
end

dice_damage_percent(temperature; params::DICEClimateParams = DICEClimateParams()) =
    params.pi2 .* temperature .^ 2 .* 100

function simulate_dice_climate_exercise(; params::DICEClimateParams = DICEClimateParams(),
        mitigation_fraction::Real = 0.5, comparison_year::Real = 2100)
    0 <= mitigation_fraction <= 1 ||
        throw(ArgumentError("mitigation_fraction must lie in [0, 1]"))
    emissions_bau = dice_bau_emissions(params)
    emissions_mitigation = mitigation_fraction .* emissions_bau

    M_bau = simulate_dice_carbon(emissions_bau; params)
    T_bau = simulate_dice_temperature(M_bau; params)
    damages_bau = dice_damage_percent(T_bau; params)

    M_mitigation = simulate_dice_carbon(emissions_mitigation; params)
    T_mitigation = simulate_dice_temperature(M_mitigation; params)
    damages_mitigation = dice_damage_percent(T_mitigation; params)

    years = dice_years(params)
    idx = argmin(abs.(years .- comparison_year))
    return (
        years = years,
        emissions_bau = emissions_bau,
        emissions_mitigation = emissions_mitigation,
        carbon_bau = M_bau,
        carbon_mitigation = M_mitigation,
        temperature_bau = T_bau,
        temperature_mitigation = T_mitigation,
        damages_bau = damages_bau,
        damages_mitigation = damages_mitigation,
        comparison_year = years[idx],
        avoided_warming = T_bau[idx] - T_mitigation[idx],
        avoided_damages = damages_bau[idx] - damages_mitigation[idx],
        mean_atmospheric_carbon_gap = mean(M_bau[1, :] .- M_mitigation[1, :]),
    )
end
