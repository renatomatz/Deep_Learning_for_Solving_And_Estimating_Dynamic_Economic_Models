export CDICEParams,
    CDICETeachingPolicy,
    cdice_state_dim,
    cdice_state_bounds,
    cdice_initial_state,
    cdice_normalize_states,
    cdice_denormalize_states,
    cdice_tau_to_time,
    cdice_time_to_tau,
    cdice_tau_next,
    cdice_population,
    cdice_tfp_trend,
    cdice_tfp,
    cdice_sigma,
    cdice_theta1,
    cdice_land_emissions,
    cdice_external_forcing,
    cdice_beta_hat,
    cdice_damage,
    cdice_damage_prime,
    cdice_carbon_next,
    cdice_temperature_next,
    cdice_policy_from_raw,
    cdice_teaching_policy_raw,
    sample_cdice_states,
    deterministic_cdice_residual,
    stochastic_cdice_residual,
    cdice_forward_step,
    simulate_cdice_path,
    cdice_reference_table,
    cdice_reference_errors,
    cdice_stationary_z_std,
    cdice_monte_carlo_paths

using NNlib
using Random: AbstractRNG, rand, randn
using Statistics: mean

Base.@kwdef struct CDICEParams
    Tstep::Float64 = 1.0
    vartheta::Float64 = 0.015
    rho::Float64 = 0.015
    psi::Float64 = 0.68965517
    alpha::Float64 = 0.30
    delta::Float64 = 0.10
    L0::Float64 = 7403.0
    Linfty::Float64 = 11500.0
    deltaL::Float64 = 0.0268
    A0hat::Float64 = 0.010295
    gA0hat::Float64 = 0.0217
    deltaA::Float64 = 0.005
    sigma0::Float64 = 0.0000955592
    gSigma0::Float64 = -0.0152
    deltaSigma::Float64 = 0.001
    theta2::Float64 = 2.6
    pback::Float64 = 0.55
    gback::Float64 = 0.005
    c2co2::Float64 = 3.666
    ELand0::Float64 = 0.00070922
    deltaLand::Float64 = 0.023
    fex0::Float64 = 0.5
    fex1::Float64 = 1.0
    Tyears::Float64 = 85.0
    pi1::Float64 = 0.0
    pi2::Float64 = 0.00236
    pow1::Float64 = 1.0
    pow2::Float64 = 2.0
    b12_::Float64 = 0.054
    b23_::Float64 = 0.0082
    MATeq::Float64 = 0.607
    MUOeq::Float64 = 0.489
    MLOeq::Float64 = 1.281
    c1_::Float64 = 0.137
    c3_::Float64 = 0.73
    c4_::Float64 = 0.00689
    f2xco2::Float64 = 3.45
    t2xco2::Float64 = 3.25
    MATbase::Float64 = 0.607
    k0::Float64 = 2.926
    MAT0::Float64 = 0.851
    MUO0::Float64 = 0.628
    MLO0::Float64 = 1.323
    TAT0::Float64 = 1.1
    TOC0::Float64 = 0.27
    tau0::Float64 = 0.0
    z0::Float64 = 0.0
    rho_z::Float64 = 0.95
    sigma_z::Float64 = 0.0125
    eps_safe::Float64 = 1e-8
end

struct CDICETeachingPolicy
    params::CDICEParams
    stochastic::Bool
end

CDICETeachingPolicy(; params::CDICEParams = CDICEParams(), stochastic::Bool = false) =
    CDICETeachingPolicy(params, stochastic)

function (policy::CDICETeachingPolicy)(x, ps, st)
    states = cdice_denormalize_states(x; params = policy.params, stochastic = policy.stochastic)
    return cdice_teaching_policy_raw(states; params = policy.params), st
end

_cdice_b12(p::CDICEParams) = p.Tstep * p.b12_
_cdice_b23(p::CDICEParams) = p.Tstep * p.b23_
_cdice_b21(p::CDICEParams) = p.MATeq / p.MUOeq * p.b12_ * p.Tstep
_cdice_b32(p::CDICEParams) = p.MUOeq / p.MLOeq * p.b23_ * p.Tstep
_cdice_c1(p::CDICEParams) = p.Tstep * p.c1_
_cdice_c1c3(p::CDICEParams) = p.Tstep * p.c1_ * p.c3_
_cdice_c1f(p::CDICEParams) = p.Tstep * p.c1_ * p.f2xco2 / p.t2xco2
_cdice_c4(p::CDICEParams) = p.Tstep * p.c4_
_cdice_delta_factor(p::CDICEParams) = (1 - p.delta)^p.Tstep

cdice_state_dim(; stochastic::Bool = false) = stochastic ? 8 : 7

function cdice_state_bounds(; params::CDICEParams = CDICEParams(), stochastic::Bool = false)
    if stochastic
        return (
            min = [0.5, 0.5, 0.2, 1.0, 0.0, 0.0, 0.0, -0.5],
            max = [60.0, 3.0, 3.0, 4.0, 10.0, 4.0, 0.99, 0.5],
        )
    end
    return (
        min = [0.5, 0.5, 0.2, 1.0, 0.0, 0.0, 0.0],
        max = [60.0, 3.0, 3.0, 4.0, 10.0, 4.0, 0.99],
    )
end

function cdice_initial_state(; params::CDICEParams = CDICEParams(), stochastic::Bool = false,
        batch::Integer = 1)
    batch > 0 || throw(ArgumentError("batch must be positive"))
    state = stochastic ?
        [params.k0, params.MAT0, params.MUO0, params.MLO0, params.TAT0, params.TOC0, params.tau0, params.z0] :
        [params.k0, params.MAT0, params.MUO0, params.MLO0, params.TAT0, params.TOC0, params.tau0]
    return repeat(reshape(state, :, 1), 1, batch)
end

function _cdice_stochastic_from_size(states, stochastic)
    stochastic !== nothing && return stochastic
    size(states, 1) == 8 && return true
    size(states, 1) == 7 && return false
    throw(DimensionMismatch("CDICE states must have 7 rows, or 8 rows with productivity shock z"))
end

function cdice_normalize_states(states::AbstractMatrix; params::CDICEParams = CDICEParams(),
        stochastic::Union{Nothing,Bool} = nothing)
    stoch = _cdice_stochastic_from_size(states, stochastic)
    bounds = cdice_state_bounds(; params, stochastic = stoch)
    lo = reshape(bounds.min, :, 1)
    hi = reshape(bounds.max, :, 1)
    size(states, 1) == length(bounds.min) ||
        throw(DimensionMismatch("state row count does not match stochastic=$(stoch) bounds"))
    return (states .- lo) ./ (hi .- lo .+ params.eps_safe)
end

function cdice_denormalize_states(states::AbstractMatrix; params::CDICEParams = CDICEParams(),
        stochastic::Union{Nothing,Bool} = nothing)
    stoch = _cdice_stochastic_from_size(states, stochastic)
    bounds = cdice_state_bounds(; params, stochastic = stoch)
    lo = reshape(bounds.min, :, 1)
    hi = reshape(bounds.max, :, 1)
    size(states, 1) == length(bounds.min) ||
        throw(DimensionMismatch("state row count does not match stochastic=$(stoch) bounds"))
    return lo .+ states .* (hi .- lo .+ params.eps_safe)
end

cdice_tau_to_time(tau; params::CDICEParams = CDICEParams()) =
    -log.(max.(1 .- tau, params.eps_safe)) ./ params.vartheta

cdice_time_to_tau(t; params::CDICEParams = CDICEParams()) =
    1 .- exp.(-params.vartheta .* t)

cdice_tau_next(tau; params::CDICEParams = CDICEParams()) =
    cdice_time_to_tau(cdice_tau_to_time(tau; params) .+ params.Tstep; params)

cdice_population(t; params::CDICEParams = CDICEParams()) =
    params.L0 .+ (params.Linfty - params.L0) .* (1 .- exp.(-params.Tstep .* params.deltaL .* t))

cdice_tfp_trend(t; params::CDICEParams = CDICEParams()) =
    params.A0hat .* exp.((params.Tstep * params.gA0hat) .*
        (1 .- exp.(-params.Tstep .* params.deltaA .* t)) ./ (params.Tstep * params.deltaA))

cdice_tfp(t, z = 0; params::CDICEParams = CDICEParams()) =
    cdice_tfp_trend(t; params) .* exp.(z)

_cdice_growth_tfp(t; params::CDICEParams = CDICEParams()) =
    params.gA0hat .* exp.(-params.Tstep .* params.deltaA .* t)

_cdice_growth_lab(t; params::CDICEParams = CDICEParams()) =
    params.deltaL ./ ((params.Linfty / (params.Linfty - params.L0)) .*
        exp.(params.Tstep .* params.deltaL .* t) .- 1)

function cdice_sigma(t; params::CDICEParams = CDICEParams())
    log_factor = log(1 + params.Tstep * params.deltaSigma)
    exponent = params.Tstep * params.gSigma0 / log_factor .* ((1 + params.Tstep * params.deltaSigma) .^ t .- 1)
    return params.sigma0 .* exp.(exponent)
end

cdice_theta1(t; params::CDICEParams = CDICEParams()) =
    params.pback .* (1000 * params.c2co2 .* cdice_sigma(t; params)) .*
    exp.(-params.Tstep .* params.gback .* t) ./ params.theta2

cdice_land_emissions(t; params::CDICEParams = CDICEParams()) =
    params.ELand0 .* exp.(-params.Tstep .* params.deltaLand .* t)

function cdice_external_forcing(t; params::CDICEParams = CDICEParams())
    years = params.Tyears / params.Tstep
    return params.fex0 .+ (params.fex1 - params.fex0) ./ years .* min.(t, years)
end

cdice_beta_hat(t; params::CDICEParams = CDICEParams()) =
    exp.((-params.rho .+ (1 - 1 / params.psi) .* _cdice_growth_tfp(t; params) .+
        _cdice_growth_lab(t; params)) .* params.Tstep)

cdice_damage(TAT; params::CDICEParams = CDICEParams()) =
    params.pi1 .* TAT .^ params.pow1 .+ params.pi2 .* TAT .^ params.pow2

cdice_damage_prime(TAT; params::CDICEParams = CDICEParams()) =
    params.pow1 * params.pi1 .* TAT .^ (params.pow1 - 1) .+
    params.pow2 * params.pi2 .* TAT .^ (params.pow2 - 1)

function cdice_carbon_next(MAT, MUO, MLO, mu, k, t, z = 0; params::CDICEParams = CDICEParams())
    A = cdice_tfp(t, z; params)
    L = cdice_population(t; params)
    sig = cdice_sigma(t; params)
    land = cdice_land_emissions(t; params)
    k_safe = max.(k, params.eps_safe)
    E_ind = (1 .- mu) .* sig .* A .* L .* k_safe .^ params.alpha
    MAT_next = (1 - _cdice_b12(params)) .* MAT .+ _cdice_b21(params) .* MUO .+
        params.Tstep .* E_ind .+ params.Tstep .* land
    MUO_next = _cdice_b12(params) .* MAT .+
        (1 - _cdice_b21(params) - _cdice_b23(params)) .* MUO .+
        _cdice_b32(params) .* MLO
    MLO_next = _cdice_b23(params) .* MUO .+ (1 - _cdice_b32(params)) .* MLO
    return MAT_next, MUO_next, MLO_next
end

function cdice_temperature_next(TAT, TOC, MAT, t; params::CDICEParams = CDICEParams())
    forcing = params.f2xco2 .* log.(max.(MAT, params.eps_safe) ./ params.MATbase) ./ log(2.0) .+
        cdice_external_forcing(t; params)
    TAT_next = (1 - _cdice_c1c3(params) - _cdice_c1f(params)) .* TAT .+
        _cdice_c1c3(params) .* TOC .+ _cdice_c1(params) .* forcing
    TOC_next = _cdice_c4(params) .* TAT .+ (1 - _cdice_c4(params)) .* TOC
    return TAT_next, TOC_next
end

function cdice_policy_from_raw(raw::AbstractMatrix)
    size(raw, 1) == 8 || throw(DimensionMismatch("CDICE policy raw output must be 8-by-batch"))
    return (
        k_next = NNlib.softplus.(raw[1:1, :]),
        lambda_hat = NNlib.softplus.(raw[2:2, :]),
        mu = NNlib.softplus.(raw[3:3, :]),
        nu_AT = NNlib.softplus.(raw[4:4, :]),
        nu_UO = raw[5:5, :],
        nu_LO = raw[6:6, :],
        eta_AT = raw[7:7, :],
        eta_OC = raw[8:8, :],
    )
end

_softplus_inverse(x; eps_safe = 1e-8) = log.(expm1.(max.(x, eps_safe)))

function _cdice_policy_raw_from_values(policy; params::CDICEParams = CDICEParams())
    return vcat(
        _softplus_inverse(policy.k_next; eps_safe = params.eps_safe),
        _softplus_inverse(policy.lambda_hat; eps_safe = params.eps_safe),
        _softplus_inverse(policy.mu; eps_safe = params.eps_safe),
        _softplus_inverse(policy.nu_AT; eps_safe = params.eps_safe),
        policy.nu_UO,
        policy.nu_LO,
        policy.eta_AT,
        policy.eta_OC,
    )
end

function cdice_teaching_policy_raw(states::AbstractMatrix; params::CDICEParams = CDICEParams())
    stoch = _cdice_stochastic_from_size(states, nothing)
    k = states[1:1, :]
    MAT = states[2:2, :]
    TAT = states[5:5, :]
    TOC = states[6:6, :]
    tau = states[7:7, :]
    z = stoch ? states[8:8, :] : zero(k)
    t = cdice_tau_to_time(tau; params)

    mu = 0.12 .+ 0.78 .* NNlib.sigmoid.(0.028 .* (t .- 80) .+ 0.30 .* (TAT .- 1.1) .+ 0.15 .* z)
    A = cdice_tfp(t, z; params)
    L = cdice_population(t; params)
    growth = exp.(params.Tstep .* (_cdice_growth_tfp(t; params) .+ _cdice_growth_lab(t; params)))
    net_output = max.(1 .- cdice_damage(TAT; params) .-
        cdice_theta1(t; params) .* mu .^ params.theta2, 0.05) .* max.(k, params.eps_safe) .^ params.alpha
    savings = 0.23 .- 0.04 .* mu
    consumption = max.((1 .- savings) .* net_output, params.eps_safe)
    k_next = max.((_cdice_delta_factor(params) .* k .+ savings .* net_output) ./ growth, 0.01)
    lambda_hat = consumption .^ (-1 / params.psi)

    nu_AT = max.(0.006 .+ 0.020 .* TAT .+ 0.002 .* max.(MAT .- params.MAT0, 0), params.eps_safe)
    policy = (
        k_next = k_next,
        lambda_hat = lambda_hat,
        mu = mu,
        nu_AT = nu_AT,
        nu_UO = -0.20 .* nu_AT,
        nu_LO = -0.05 .* nu_AT,
        eta_AT = 0.004 .* TAT,
        eta_OC = 0.001 .* TOC,
    )
    return _cdice_policy_raw_from_values(policy; params)
end

function sample_cdice_states(rng::AbstractRNG, n::Integer; params::CDICEParams = CDICEParams(),
        stochastic::Bool = false)
    n > 0 || throw(ArgumentError("n must be positive"))
    k = params.k0 .* (0.75 .+ 0.70 .* rand(rng, n))
    MAT = params.MAT0 .* (0.85 .+ 0.35 .* rand(rng, n))
    MUO = params.MUO0 .* (0.90 .+ 0.20 .* rand(rng, n))
    MLO = params.MLO0 .* (0.95 .+ 0.10 .* rand(rng, n))
    TAT = params.TAT0 .* (0.70 .+ 0.80 .* rand(rng, n))
    TOC = params.TOC0 .* (0.80 .+ 0.50 .* rand(rng, n))
    tau = 0.80 .* rand(rng, n)
    states = vcat(reshape(k, 1, :), reshape(MAT, 1, :), reshape(MUO, 1, :),
        reshape(MLO, 1, :), reshape(TAT, 1, :), reshape(TOC, 1, :), reshape(tau, 1, :))
    if stochastic
        z = cdice_stationary_z_std(params) .* randn(rng, n)
        states = vcat(states, reshape(z, 1, :))
    end
    return states
end

function _cdice_current_pieces(states::AbstractMatrix, policy; params::CDICEParams, stochastic::Bool)
    k = states[1:1, :]
    MAT = states[2:2, :]
    MUO = states[3:3, :]
    MLO = states[4:4, :]
    TAT = states[5:5, :]
    TOC = states[6:6, :]
    tau = states[7:7, :]
    z = stochastic ? states[8:8, :] : zero(k)
    t = cdice_tau_to_time(tau; params)

    A = cdice_tfp(t, z; params)
    L = cdice_population(t; params)
    sig = cdice_sigma(t; params)
    theta = cdice_theta1(t; params)
    beta = cdice_beta_hat(t; params)
    growth = exp.(params.Tstep .* (_cdice_growth_tfp(t; params) .+ _cdice_growth_lab(t; params)))
    damage = cdice_damage(TAT; params)
    abatement = theta .* policy.mu .^ params.theta2
    abatement_prime = theta .* params.theta2 .* policy.mu .^ (params.theta2 - 1)
    consumption = max.(policy.lambda_hat .^ (-params.psi), params.eps_safe)
    MAT_next, MUO_next, MLO_next = cdice_carbon_next(MAT, MUO, MLO, policy.mu, k, t, z; params)
    TAT_next, TOC_next = cdice_temperature_next(TAT, TOC, MAT, t; params)
    tau_next = cdice_tau_next(tau; params)
    return (
        k = k, MAT = MAT, MUO = MUO, MLO = MLO, TAT = TAT, TOC = TOC, tau = tau,
        z = z, t = t, A = A, L = L, sigma = sig, theta1 = theta, beta_hat = beta,
        growth = growth, damage = damage, abatement = abatement,
        abatement_prime = abatement_prime, consumption = consumption,
        MAT_next = MAT_next, MUO_next = MUO_next, MLO_next = MLO_next,
        TAT_next = TAT_next, TOC_next = TOC_next, tau_next = tau_next,
    )
end

function _cdice_residual_tuple(cur, policy, next_policy; params::CDICEParams)
    t_next = cdice_tau_to_time(cur.tau_next; params)
    A_next = cdice_tfp(t_next, cur.z; params)
    sigma_next = cdice_sigma(t_next; params)
    theta1_next = cdice_theta1(t_next; params)
    damage_next = cdice_damage(cur.TAT_next; params)
    damage_prime_next = cdice_damage_prime(cur.TAT_next; params)
    abatement_next = theta1_next .* next_policy.mu .^ params.theta2
    k_next_safe = max.(policy.k_next, params.eps_safe)
    MAT_next_safe = max.(cur.MAT_next, params.eps_safe)

    eq1 = cur.growth .* policy.lambda_hat .- cur.beta_hat .* (
        next_policy.lambda_hat .* (params.Tstep .* (1 .- abatement_next .- damage_next) .*
            params.alpha .* k_next_safe .^ (params.alpha - 1) .+ _cdice_delta_factor(params)) .+
        (-next_policy.nu_AT) .* (1 .- next_policy.mu) .* params.Tstep .* sigma_next .*
            A_next .* cdice_population(t_next; params) .* params.alpha .* k_next_safe .^ (params.alpha - 1)
    )
    eq2 = params.Tstep .* (1 .- cur.abatement .- cur.damage) .* max.(cur.k, params.eps_safe) .^ params.alpha .-
        params.Tstep .* cur.consumption .+ _cdice_delta_factor(params) .* cur.k .-
        cur.growth .* policy.k_next
    lambda_mu = -policy.lambda_hat .* params.Tstep .* cur.abatement_prime .*
        max.(cur.k, params.eps_safe) .^ params.alpha .-
        (-policy.nu_AT) .* params.Tstep .* cur.sigma .* cur.A .* cur.L .*
        max.(cur.k, params.eps_safe) .^ params.alpha
    eq3 = lambda_mu .+ (1 .- policy.mu) .-
        sqrt.(lambda_mu .^ 2 .+ (1 .- policy.mu) .^ 2 .+ 1e-12)
    eq4 = policy.eta_AT .- cur.beta_hat .* (
        next_policy.lambda_hat .* (-params.Tstep .* damage_prime_next) .* k_next_safe .^ params.alpha .+
        next_policy.eta_AT .* (1 - _cdice_c1c3(params) - _cdice_c1f(params)) .+
        next_policy.eta_OC .* _cdice_c4(params)
    )
    eq5 = (-policy.nu_AT) .- cur.beta_hat .* (
        (-next_policy.nu_AT) .* (1 - _cdice_b12(params)) .+
        next_policy.nu_UO .* _cdice_b12(params) .+
        next_policy.eta_AT .* _cdice_c1(params) .* params.f2xco2 ./ (log(2.0) .* MAT_next_safe)
    )
    eq6 = policy.nu_UO .- cur.beta_hat .* (
        (-next_policy.nu_AT) .* _cdice_b21(params) .+
        next_policy.nu_UO .* (1 - _cdice_b21(params) - _cdice_b23(params)) .+
        next_policy.nu_LO .* _cdice_b23(params)
    )
    eq7 = policy.nu_LO .- cur.beta_hat .* (
        next_policy.nu_UO .* _cdice_b32(params) .+
        next_policy.nu_LO .* (1 - _cdice_b32(params))
    )
    eq8 = policy.eta_OC .- cur.beta_hat .* (
        next_policy.eta_AT .* _cdice_c1c3(params) .+
        next_policy.eta_OC .* (1 - _cdice_c4(params))
    )
    return (eq1, eq2, eq3, eq4, eq5, eq6, eq7, eq8)
end

function deterministic_cdice_residual(model, ps, st, states::AbstractMatrix;
        params::CDICEParams = CDICEParams())
    size(states, 1) == 7 || throw(DimensionMismatch("deterministic CDICE states must be 7-by-batch"))
    # The CDICE residual evaluates current and next-period policies in one pullback;
    # use stateless Lux policies such as Dense-only MLPs for this teaching helper.
    raw, st_new = model(cdice_normalize_states(states; params, stochastic = false), ps, st)
    policy = cdice_policy_from_raw(raw)
    cur = _cdice_current_pieces(states, policy; params, stochastic = false)
    states_next = vcat(policy.k_next, cur.MAT_next, cur.MUO_next, cur.MLO_next,
        cur.TAT_next, cur.TOC_next, cur.tau_next)
    raw_next, _ = model(cdice_normalize_states(states_next; params, stochastic = false), ps, st_new)
    next_policy = cdice_policy_from_raw(raw_next)
    equations = vcat(_cdice_residual_tuple(cur, policy, next_policy; params)...)
    penalty = 1e-2 * mean(abs2, max.(policy.mu .- 1, 0))
    return (
        loss = mean(abs2, equations) + penalty,
        residuals = equations,
        residual_loss = mean(abs2, equations),
        penalty_mu_upper = penalty,
        policy = policy,
        next_state = states_next,
        consumption = cur.consumption,
        emissions = (1 .- policy.mu) .* cur.sigma .* cur.A .* cur.L .* max.(cur.k, params.eps_safe) .^ params.alpha,
    ), st_new
end

function stochastic_cdice_residual(model, ps, st, states::AbstractMatrix, rule::QuadratureRule;
        params::CDICEParams = CDICEParams(),
        sigma_z::Real = params.sigma_z)
    size(states, 1) == 8 || throw(DimensionMismatch("stochastic CDICE states must be 8-by-batch"))
    # The quadrature loop reuses the same Lux state across next-period evaluations;
    # use stateless policies such as Dense-only MLPs for this teaching helper.
    rule.nodes isa AbstractVector ||
        throw(ArgumentError("stochastic_cdice_residual expects a one-dimensional Gauss-Hermite rule"))
    raw, st_new = model(cdice_normalize_states(states; params, stochastic = true), ps, st)
    policy = cdice_policy_from_raw(raw)
    cur = _cdice_current_pieces(states, policy; params, stochastic = true)

    terms = [begin
        z_next = params.rho_z .* cur.z .+ sigma_z .* node
        states_next = vcat(policy.k_next, cur.MAT_next, cur.MUO_next, cur.MLO_next,
            cur.TAT_next, cur.TOC_next, cur.tau_next, z_next)
        raw_next, _ = model(cdice_normalize_states(states_next; params, stochastic = true), ps, st_new)
        next_policy = cdice_policy_from_raw(raw_next)
        cur_next_z = merge(cur, (z = z_next,))
        map(x -> weight .* x, _cdice_residual_tuple(cur_next_z, policy, next_policy; params))
    end for (node, weight) in zip(rule.nodes, rule.weights)]

    expected = ntuple(i -> reduce(+, (term[i] for term in terms)), 8)
    eq2 = expected[2]
    eq3 = expected[3]
    equations = vcat(expected[1], eq2, eq3, expected[4], expected[5], expected[6], expected[7], expected[8])
    penalty = 1e-2 * mean(abs2, max.(policy.mu .- 1, 0))
    z_mean_next = params.rho_z .* cur.z
    return (
        loss = mean(abs2, equations) + penalty,
        residuals = equations,
        residual_loss = mean(abs2, equations),
        penalty_mu_upper = penalty,
        policy = policy,
        z_mean_next = z_mean_next,
        consumption = cur.consumption,
    ), st_new
end

function cdice_forward_step(model, ps, st, states::AbstractMatrix;
        params::CDICEParams = CDICEParams(), stochastic::Bool = size(states, 1) == 8,
        shock = 0.0)
    expected_rows = cdice_state_dim(; stochastic)
    size(states, 1) == expected_rows ||
        throw(DimensionMismatch("expected $expected_rows rows for stochastic=$(stochastic)"))
    raw, st_new = model(cdice_normalize_states(states; params, stochastic), ps, st)
    policy = cdice_policy_from_raw(raw)
    cur = _cdice_current_pieces(states, policy; params, stochastic)
    if stochastic
        z_next = params.rho_z .* cur.z .+ params.sigma_z .* shock
        next_state = vcat(policy.k_next, cur.MAT_next, cur.MUO_next, cur.MLO_next,
            cur.TAT_next, cur.TOC_next, cur.tau_next, z_next)
    else
        next_state = vcat(policy.k_next, cur.MAT_next, cur.MUO_next, cur.MLO_next,
            cur.TAT_next, cur.TOC_next, cur.tau_next)
    end
    return next_state, policy, st_new
end

function _cdice_simulation_quantities(state::AbstractMatrix, policy; params::CDICEParams, stochastic::Bool)
    cur = _cdice_current_pieces(state, policy; params, stochastic)
    theta = cur.theta1 .* policy.mu .^ params.theta2
    damage = cur.damage
    k_safe = max.(cur.k, params.eps_safe)
    dvdk = policy.lambda_hat .* (params.Tstep .* (1 .- theta .- damage) .*
        params.alpha .* k_safe .^ (params.alpha - 1) .+ _cdice_delta_factor(params)) .+
        (-policy.nu_AT) .* (1 .- policy.mu) .* params.Tstep .* cur.sigma .* cur.A .* cur.L .*
        params.alpha .* k_safe .^ (params.alpha - 1)
    dvdMAT = (-policy.nu_AT) .* (1 - _cdice_b12(params)) .+
        policy.nu_UO .* _cdice_b12(params) .+
        policy.eta_AT .* _cdice_c1(params) .* params.f2xco2 ./ (log(2.0) .* max.(cur.MAT, params.eps_safe))
    scc = -dvdMAT ./ (dvdk .+ params.eps_safe) .* cur.A .* cur.L ./ params.c2co2
    carbon_tax = cur.theta1 .* params.theta2 .* policy.mu .^ (params.theta2 - 1) ./
        max.(cur.sigma, params.eps_safe) ./ params.c2co2
    E_ind = (1 .- policy.mu) .* cur.sigma .* cur.A .* cur.L .* k_safe .^ params.alpha
    return (
        k_abs = cur.k .* cur.A .* cur.L,
        k_eff = cur.k,
        MAT_GtC = cur.MAT .* 1000,
        TAT = cur.TAT,
        TOC = cur.TOC,
        mu = policy.mu,
        con_abs = cur.consumption .* cur.A .* cur.L,
        scc = scc,
        carbon_tax = carbon_tax,
        Eind_GtCO2 = E_ind .* 1000 .* params.c2co2,
        z = cur.z,
    )
end

function simulate_cdice_path(model, ps = nothing, st = NamedTuple();
        params::CDICEParams = CDICEParams(), periods::Integer = 300,
        stochastic::Bool = false, rng::Union{Nothing,AbstractRNG} = nothing,
        sigma_z_realized::Real = stochastic ? params.sigma_z : 0.0,
        draw_initial_z::Bool = stochastic && sigma_z_realized > 0)
    periods > 0 || throw(ArgumentError("periods must be positive"))
    stochastic && rng === nothing &&
        throw(ArgumentError("stochastic simulation requires an explicit rng"))
    state = cdice_initial_state(; params, stochastic, batch = 1)
    if stochastic && draw_initial_z
        state[8, 1] = sigma_z_realized / sqrt(max(1 - params.rho_z^2, params.eps_safe)) * randn(rng)
    end
    years = Vector{Float64}(undef, periods)
    k_abs = similar(years)
    k_eff = similar(years)
    MAT_GtC = similar(years)
    TAT = similar(years)
    TOC = similar(years)
    mu = similar(years)
    con_abs = similar(years)
    scc = similar(years)
    carbon_tax = similar(years)
    emissions = similar(years)
    z_path = similar(years)
    st_acc = st
    for i in 1:periods
        years[i] = 2015.0 + i - 1
        raw, st_acc = model(cdice_normalize_states(state; params, stochastic), ps, st_acc)
        policy = cdice_policy_from_raw(raw)
        q = _cdice_simulation_quantities(state, policy; params, stochastic)
        k_abs[i] = only(q.k_abs)
        k_eff[i] = only(q.k_eff)
        MAT_GtC[i] = only(q.MAT_GtC)
        TAT[i] = only(q.TAT)
        TOC[i] = only(q.TOC)
        mu[i] = only(q.mu)
        con_abs[i] = only(q.con_abs)
        scc[i] = only(q.scc)
        carbon_tax[i] = only(q.carbon_tax)
        emissions[i] = only(q.Eind_GtCO2)
        z_path[i] = only(q.z)
        shock = stochastic && params.sigma_z > 0 ? sigma_z_realized / params.sigma_z * randn(rng) : 0.0
        state, _, st_acc = cdice_forward_step(model, ps, st_acc, state; params, stochastic, shock)
    end
    return (
        year = years,
        k_abs = k_abs,
        k_eff = k_eff,
        MAT_GtC = MAT_GtC,
        MAT = MAT_GtC,
        TAT = TAT,
        TOC = TOC,
        mu = mu,
        con_abs = con_abs,
        con = con_abs,
        scc = scc,
        carbon_tax = carbon_tax,
        Eind_GtCO2 = emissions,
        z = z_path,
        st = st_acc,
    )
end

function cdice_reference_table()
    return Dict(
        (0, :TAT) => 1.10,
        (0, :MAT_GtC) => 851.0,
        (0, :mu) => 0.144,
        (0, :scc) => 24.82,
        (85, :TAT) => 2.92,
        (85, :MAT_GtC) => 1222.9,
        (85, :mu) => 0.673,
        (85, :scc) => 186.4,
        (285, :TAT) => 3.10,
        (285, :mu) => 0.999,
    )
end

function cdice_reference_errors(path; reference = cdice_reference_table())
    rows = NamedTuple[]
    for ((idx0, var), ref) in sort(collect(reference); by = x -> (x[1][1], String(x[1][2])))
        series = getproperty(path, var)
        idx = idx0 + 1
        idx <= length(series) || continue
        got = series[idx]
        err = abs(got - ref) / max(abs(ref), 1e-6) * 100
        status = err < 5 ? :OK : (err < 15 ? :CLOSE : :FAIL)
        push!(rows, (year = 2015 + idx0, variable = var, got = got, reference = ref,
            percent_error = err, status = status))
    end
    return rows
end

cdice_stationary_z_std(params::CDICEParams = CDICEParams()) =
    params.sigma_z / sqrt(max(1 - params.rho_z^2, params.eps_safe))

function cdice_monte_carlo_paths(model, ps = nothing, st = NamedTuple();
        params::CDICEParams = CDICEParams(), n_paths::Integer = 20, periods::Integer = 285,
        seed::Integer = 1000)
    n_paths > 0 || throw(ArgumentError("n_paths must be positive"))
    scc = Matrix{Float64}(undef, n_paths, periods)
    TAT = similar(scc)
    mu = similar(scc)
    z = similar(scc)
    for i in 1:n_paths
        path = simulate_cdice_path(model, ps, st; params, periods, stochastic = true,
            rng = rng_from_seed(seed; offset = i), sigma_z_realized = params.sigma_z)
        scc[i, :] .= path.scc
        TAT[i, :] .= path.TAT
        mu[i, :] .= path.mu
        z[i, :] .= path.z
    end
    return (scc = scc, TAT = TAT, mu = mu, z = z, years = 2015 .+ collect(0:(periods - 1)))
end
