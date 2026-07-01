export AnalyticOLGParams,
    BenchmarkOLGParams,
    analytic_olg_state_dim,
    analytic_olg_feature_dim,
    benchmark_olg_state_dim,
    benchmark_olg_feature_dim,
    analytic_olg_closed_form_savings_rates,
    analytic_olg_assemble_states,
    benchmark_olg_assemble_states,
    analytic_olg_features,
    benchmark_olg_features,
    sample_analytic_olg_states,
    sample_benchmark_olg_states,
    analytic_olg_policy_from_raw,
    benchmark_olg_policy_from_raw,
    analytic_olg_exact_policy,
    analytic_olg_policy_error,
    analytic_olg_residual,
    benchmark_olg_residual,
    analytic_olg_next_states,
    benchmark_olg_next_states

using LinearAlgebra: kron
using Statistics: mean

struct AnalyticOLGParams{T}
    n_ages::Int
    beta::T
    gamma::T
    alpha::T
    tfp::Vector{T}
    depreciation::Vector{T}
    transition::Matrix{T}
    labor::Vector{T}
    min_saving_fraction::T
    max_saving_fraction::T
    consumption_floor::T
end

function AnalyticOLGParams(;
        n_ages::Integer = 6,
        beta = 0.70,
        gamma = 1.0,
        alpha = 0.30,
        tfp = [0.95, 1.05, 0.95, 1.05],
        depreciation = [0.50, 0.50, 0.90, 0.90],
        transition = nothing,
        labor = nothing,
        min_saving_fraction = 1e-6,
        max_saving_fraction = 1 - 1e-5,
        consumption_floor = 1e-8)
    n_ages > 1 || throw(ArgumentError("n_ages must exceed one"))
    length(tfp) == length(depreciation) || throw(ArgumentError("tfp and depreciation must have the same length"))
    n_shocks = length(tfp)
    n_shocks > 0 || throw(ArgumentError("at least one aggregate shock is required"))
    transition === nothing && (transition = fill(1 / n_shocks, n_shocks, n_shocks))
    labor === nothing && (labor = [one(float(beta)); zeros(typeof(float(beta)), n_ages - 1)])

    T = promote_type(map(x -> eltype(float.(x)), (tfp, depreciation, transition, labor))..., typeof(float(beta)), typeof(float(gamma)), typeof(float(alpha)))
    tfp_T = T.(tfp)
    depreciation_T = T.(depreciation)
    transition_T = T.(transition)
    labor_T = T.(labor)
    _validate_olg_transition(transition_T, n_shocks)
    length(labor_T) == n_ages || throw(ArgumentError("labor must have one entry per age"))
    0 < min_saving_fraction < max_saving_fraction < 1 ||
        throw(ArgumentError("saving fraction bounds must lie inside (0, 1)"))
    consumption_floor > 0 || throw(ArgumentError("consumption_floor must be positive"))
    return AnalyticOLGParams(
        n_ages,
        T(beta),
        T(gamma),
        T(alpha),
        tfp_T,
        depreciation_T,
        transition_T,
        labor_T,
        T(min_saving_fraction),
        T(max_saving_fraction),
        T(consumption_floor),
    )
end

struct BenchmarkOLGParams{T}
    n_ages::Int
    beta::T
    gamma::T
    alpha::T
    zeta::T
    kappa::T
    tfp::Vector{T}
    depreciation::Vector{T}
    transition::Matrix{T}
    labor::Vector{T}
    consumption_floor::T
    capital_log_scale::T
    bond_log_scale::T
    price_log_scale::T
    multiplier_scale::T
    multiplier_bias::T
    min_capital_choice::T
    capital_base_min::T
    capital_base_max::T
end

function BenchmarkOLGParams(;
        n_ages::Integer = 56,
        beta = 0.95,
        gamma = 2.0,
        alpha = 0.30,
        zeta = 0.50,
        tfp = [0.978, 1.022, 0.978, 1.022],
        depreciation = [0.080, 0.080, 0.110, 0.110],
        transition = nothing,
        labor = nothing,
        kappa = nothing,
        consumption_floor = 1e-6,
        capital_log_scale = 1.50,
        bond_log_scale = 2.50,
        price_log_scale = 0.50,
        multiplier_scale = 0.10,
        multiplier_bias = -4.0,
        min_capital_choice = 1e-5,
        capital_base_min = 0.02,
        capital_base_max = 0.75)
    n_ages > 1 || throw(ArgumentError("n_ages must exceed one"))
    length(tfp) == length(depreciation) || throw(ArgumentError("tfp and depreciation must have the same length"))
    n_shocks = length(tfp)
    if transition === nothing
        pi_depr = [0.972 0.028; 0.300 0.700]
        pi_tfp = [0.905 0.095; 0.095 0.905]
        transition = kron(pi_depr, pi_tfp)
    end
    labor === nothing && (labor = _benchmark_labor_profile(n_ages))
    kappa === nothing && (kappa = inv(1 - maximum(depreciation)))

    T = promote_type(map(x -> eltype(float.(x)), (tfp, depreciation, transition, labor))..., typeof(float(beta)), typeof(float(gamma)), typeof(float(alpha)))
    tfp_T = T.(tfp)
    depreciation_T = T.(depreciation)
    transition_T = T.(transition)
    labor_T = T.(labor)
    _validate_olg_transition(transition_T, n_shocks)
    length(labor_T) == n_ages || throw(ArgumentError("labor must have one entry per age"))
    consumption_floor > 0 || throw(ArgumentError("consumption_floor must be positive"))
    min_capital_choice > 0 || throw(ArgumentError("min_capital_choice must be positive"))
    capital_base_min > 0 || throw(ArgumentError("capital_base_min must be positive"))
    capital_base_max > capital_base_min || throw(ArgumentError("capital_base_max must exceed capital_base_min"))
    return BenchmarkOLGParams(
        n_ages,
        T(beta),
        T(gamma),
        T(alpha),
        T(zeta),
        T(kappa),
        tfp_T,
        depreciation_T,
        transition_T,
        labor_T,
        T(consumption_floor),
        T(capital_log_scale),
        T(bond_log_scale),
        T(price_log_scale),
        T(multiplier_scale),
        T(multiplier_bias),
        T(min_capital_choice),
        T(capital_base_min),
        T(capital_base_max),
    )
end

_olg_python_feature_dim(n_ages::Integer, n_shocks::Integer) =
    1 + n_shocks + 2 + 5 + 4 * n_ages + n_shocks

analytic_olg_state_dim(params::AnalyticOLGParams) = 1 + params.n_ages
analytic_olg_feature_dim(params::AnalyticOLGParams) = _olg_python_feature_dim(params.n_ages, length(params.tfp))
benchmark_olg_state_dim(params::BenchmarkOLGParams) = 1 + 2 * params.n_ages
benchmark_olg_feature_dim(params::BenchmarkOLGParams) = _olg_python_feature_dim(params.n_ages, length(params.tfp))

function _benchmark_labor_profile(n_ages::Integer)
    labor = zeros(Float64, n_ages)
    a_is = floor(Int, 2 * n_ages / 3)
    a_decr = floor(Int, 5 * n_ages / 6)
    temp_l = 0.6 + (0.6 * 1.27 / (n_ages / 2)^2) * ((n_ages / 2)^2 - (a_is - n_ages / 2)^2)
    for a in 1:n_ages
        idx = a - 1
        if idx < a_is
            labor[a] = 0.6 + (0.6 * 1.27 / (n_ages / 2)^2) * ((n_ages / 2)^2 - (idx - n_ages / 2)^2)
        elseif idx < a_decr
            labor[a] = temp_l - 0.5 * temp_l * (idx - a_is) / max(a_decr - 1 - a_is, 1)
        else
            labor[a] = 0.5 * temp_l
        end
    end
    return labor
end

function _validate_olg_transition(transition, n_shocks::Integer)
    size(transition) == (n_shocks, n_shocks) ||
        throw(DimensionMismatch("transition matrix must be $n_shocks by $n_shocks"))
    all(transition .>= 0) || throw(ArgumentError("transition probabilities must be nonnegative"))
    all(abs.(sum(transition; dims = 2) .- 1) .< 1e-8) ||
        throw(ArgumentError("transition rows must sum to one"))
    return transition
end

function _shock_indices(row, n_shocks::Integer)
    return clamp.(round.(Int, vec(row)), 1, n_shocks)
end

function _onehot_shocks(indices, n_shocks::Integer, ::Type{T}) where {T}
    return T.(reshape(1:n_shocks, n_shocks, 1) .== reshape(indices, 1, length(indices)))
end

function _olg_transition_rows(z, params, ::Type{T}) where {T}
    return T.(permutedims(params.transition[z, :]))
end

function _olg_shock_features(z, params, ::Type{T}) where {T}
    n_shocks = length(params.tfp)
    z_scaled = reshape(T.(z .- 1) ./ T(max(1, n_shocks - 1)), 1, :)
    return z_scaled, _onehot_shocks(z, n_shocks, T), _olg_transition_rows(z, params, T)
end

function _olg_labor_feature(params, batch::Integer, ::Type{T}) where {T}
    labor_total = T(sum(params.labor))
    labor_scale = max(one(T), labor_total)
    return fill(labor_total / labor_scale, 1, batch)
end

function analytic_olg_closed_form_savings_rates(params::AnalyticOLGParams)
    params.gamma == 1 || throw(ArgumentError("closed-form savings rates require log utility, gamma = 1"))
    return [
        params.beta * (1 - params.beta^(params.n_ages - 1 - j)) /
            (1 - params.beta^(params.n_ages - j))
        for j in 0:(params.n_ages - 2)
    ]
end

function analytic_olg_assemble_states(z, k::AbstractMatrix)
    size(k, 1) > 1 || throw(ArgumentError("k must contain one row per age"))
    z_row = z isa Number ? fill(float(z), 1, size(k, 2)) : (length(z) == 1 ? fill(float(only(z)), 1, size(k, 2)) : reshape(float.(z), 1, :))
    size(z_row, 2) == size(k, 2) || throw(DimensionMismatch("z and k must have the same batch size"))
    return vcat(z_row, k)
end

function benchmark_olg_assemble_states(z, k::AbstractMatrix, b::AbstractMatrix)
    size(k) == size(b) || throw(DimensionMismatch("k and b must have the same shape"))
    z_row = z isa Number ? fill(float(z), 1, size(k, 2)) : (length(z) == 1 ? fill(float(only(z)), 1, size(k, 2)) : reshape(float.(z), 1, :))
    size(z_row, 2) == size(k, 2) || throw(DimensionMismatch("z and k must have the same batch size"))
    return vcat(z_row, k, b)
end

function _analytic_olg_blocks(states, params::AnalyticOLGParams)
    x = assert_feature_batch(states, analytic_olg_state_dim(params))
    z = Zygote.ignore() do
        _shock_indices(@view(x[1, :]), length(params.tfp))
    end
    k = @view x[2:(1 + params.n_ages), :]
    return z, k
end

function _benchmark_olg_blocks(states, params::BenchmarkOLGParams)
    x = assert_feature_batch(states, benchmark_olg_state_dim(params))
    z = Zygote.ignore() do
        _shock_indices(@view(x[1, :]), length(params.tfp))
    end
    k = @view x[2:(1 + params.n_ages), :]
    b = @view x[(2 + params.n_ages):(1 + 2 * params.n_ages), :]
    return z, k, b
end

function _firm_prices(K, z_idx, params)
    T = promote_type(eltype(K), eltype(params.tfp))
    eta = reshape(T.(params.tfp[z_idx]), 1, :)
    depr = reshape(T.(params.depreciation[z_idx]), 1, :)
    K_safe = max.(T(params.consumption_floor), T.(K))
    labor_total = T(sum(params.labor))
    r = params.alpha .* eta .* K_safe .^ (params.alpha - 1) .* labor_total^(1 - params.alpha) .+ (1 .- depr)
    w = (1 - params.alpha) .* eta .* K_safe .^ params.alpha .* labor_total^(-params.alpha)
    y = eta .* K_safe .^ params.alpha .* labor_total^(1 - params.alpha) .+ (1 .- depr) .* K_safe
    return r, w, y, eta, depr
end

function _analytic_olg_current_income(states, params::AnalyticOLGParams)
    z, k = _analytic_olg_blocks(states, params)
    K = sum(k; dims = 1)
    r, w, y, eta, depr = _firm_prices(K, z, params)
    labor_income = w .* reshape(params.labor, :, 1)
    financial_income = r .* k
    income = financial_income .+ labor_income
    return income, labor_income, financial_income, K, r, w, y, eta, depr
end

function _benchmark_olg_current_cash(states, params::BenchmarkOLGParams)
    z, k, b = _benchmark_olg_blocks(states, params)
    K = sum(k; dims = 1)
    r, w, y, eta, depr = _firm_prices(K, z, params)
    labor_income = w .* reshape(params.labor, :, 1)
    financial_income = r .* k .+ b
    cash = financial_income .+ labor_income
    return cash, labor_income, financial_income, K, r, w, y, eta, depr
end

function analytic_olg_features(states; params::AnalyticOLGParams = AnalyticOLGParams())
    z, k = _analytic_olg_blocks(states, params)
    income, labor_income, financial_income, K, r, w, y, eta, depr = _analytic_olg_current_income(states, params)
    T = promote_type(eltype(states), eltype(params.tfp))
    z_scaled, onehot, transition_rows = _olg_shock_features(z, params, T)
    aggregate_capital_scale = max(one(T), T(0.4) * T(params.n_ages))
    return vcat(
        z_scaled,
        onehot,
        T.(eta),
        T.(depr),
        T.(K) ./ aggregate_capital_scale,
        _olg_labor_feature(params, size(k, 2), T),
        T.(r),
        T.(w),
        T.(y),
        T.(k),
        T.(financial_income),
        T.(labor_income),
        T.(income),
        transition_rows,
    )
end

function benchmark_olg_features(states; params::BenchmarkOLGParams = BenchmarkOLGParams())
    z, k, b = _benchmark_olg_blocks(states, params)
    cash, labor_income, financial_income, K, r, w, y, eta, depr = _benchmark_olg_current_cash(states, params)
    T = promote_type(eltype(states), eltype(params.tfp))
    z_scaled, onehot, transition_rows = _olg_shock_features(z, params, T)
    aggregate_capital_scale = max(one(T), T(0.5) * T(params.n_ages))
    output_scale = max(one(T), T(sum(params.labor)))
    return vcat(
        z_scaled,
        onehot,
        T.(eta),
        T.(depr),
        T.(K) ./ aggregate_capital_scale,
        _olg_labor_feature(params, size(k, 2), T),
        T.(r),
        T.(w),
        T.(y) ./ output_scale,
        T.(k),
        T.(financial_income),
        T.(labor_income),
        T.(cash),
        transition_rows,
    )
end

function sample_analytic_olg_states(rng, params::AnalyticOLGParams, n::Integer;
        k_low = 0.005,
        k_high = 1.0)
    n > 0 || throw(ArgumentError("n must be positive"))
    0 < k_low < k_high || throw(ArgumentError("capital bounds must satisfy 0 < k_low < k_high"))
    z = rand(rng, 1:length(params.tfp), n)
    log_k = log(k_low) .+ (log(k_high) - log(k_low)) .* rand(rng, params.n_ages, n)
    k = exp.(log_k)
    k[1, :] .= 0
    return analytic_olg_assemble_states(z, k)
end

function sample_benchmark_olg_states(rng, params::BenchmarkOLGParams, n::Integer;
        k_low = 0.02,
        k_high = 1.25,
        bond_pos_max = 0.75,
        bond_neg_fraction = 0.90)
    n > 0 || throw(ArgumentError("n must be positive"))
    0 < k_low < k_high || throw(ArgumentError("capital bounds must satisfy 0 < k_low < k_high"))
    z = rand(rng, 1:length(params.tfp), n)
    log_k = log(k_low) .+ (log(k_high) - log(k_low)) .* rand(rng, params.n_ages, n)
    k = exp.(log_k)
    k[1, :] .= 0
    b_low = -bond_neg_fraction .* k ./ params.kappa
    b_high = bond_pos_max .* ones(size(k))
    b = b_low .+ (b_high .- b_low) .* rand(rng, params.n_ages, n)
    b[1, :] .= 0
    return benchmark_olg_assemble_states(z, k, b)
end

function analytic_olg_policy_from_raw(raw, states; params::AnalyticOLGParams = AnalyticOLGParams())
    size(raw, 1) == params.n_ages - 1 ||
        throw(DimensionMismatch("analytic OLG raw policy must have n_ages - 1 rows"))
    income, _, _, _, _, _, _, _, _ = _analytic_olg_current_income(states, params)
    choice_income = max.(@view(income[1:(params.n_ages - 1), :]), 0)
    fraction = sigmoid_bounds(raw, params.min_saving_fraction, params.max_saving_fraction)
    savings = fraction .* choice_income
    return (savings = savings, fraction = fraction, income = income)
end

function analytic_olg_exact_policy(states; params::AnalyticOLGParams = AnalyticOLGParams())
    income, _, _, _, _, _, _, _, _ = _analytic_olg_current_income(states, params)
    rates = reshape(analytic_olg_closed_form_savings_rates(params), :, 1)
    savings = rates .* @view(income[1:(params.n_ages - 1), :])
    return (savings = savings, fraction = repeat(rates, 1, size(states, 2)), income = income)
end

function analytic_olg_policy_error(savings, states; params::AnalyticOLGParams = AnalyticOLGParams())
    exact = analytic_olg_exact_policy(states; params).savings
    rel = (savings .- exact) ./ max.(abs.(exact), params.consumption_floor)
    return (relative = rel, summary = residual_summary(rel))
end

function _marginal_utility(c, gamma, floor)
    return max.(c, floor) .^ (-gamma)
end

function _inverse_marginal_utility(x, gamma, floor)
    return max.(x, floor) .^ (-inv(gamma))
end

function analytic_olg_residual(model, ps, st, states;
        params::AnalyticOLGParams = AnalyticOLGParams(),
        use_exact_policy::Bool = false)
    raw, st_new = use_exact_policy ? (zeros(eltype(states), params.n_ages - 1, size(states, 2)), st) :
        model(analytic_olg_features(states; params), ps, st)
    policy = use_exact_policy ? analytic_olg_exact_policy(states; params) :
        analytic_olg_policy_from_raw(raw, states; params)
    a = policy.savings
    income = policy.income
    a_all = vcat(a, zeros(eltype(a), 1, size(a, 2)))
    c_raw = income .- a_all
    c = max.(c_raw, params.consumption_floor)
    k_next = vcat(zeros(eltype(a), 1, size(a, 2)), a)

    z, _ = _analytic_olg_blocks(states, params)
    future = map(1:length(params.tfp)) do shock
        x_next = analytic_olg_assemble_states(fill(shock, size(states, 2)), k_next)
        next_policy = if use_exact_policy
            analytic_olg_exact_policy(x_next; params)
        else
            raw_next, _ = model(analytic_olg_features(x_next; params), ps, st_new)
            analytic_olg_policy_from_raw(raw_next, x_next; params)
        end
        a_next_all = vcat(next_policy.savings, zeros(eltype(a), 1, size(a, 2)))
        c_next_raw = next_policy.income .- a_next_all
        c_next = max.(c_next_raw, params.consumption_floor)
        _, _, _, _, r_next, _, _, _, _ = _analytic_olg_current_income(x_next, params)
        probs = reshape(params.transition[z, shock], 1, :)
        term = probs .* r_next .* _marginal_utility(@view(c_next[2:params.n_ages, :]), params.gamma, params.consumption_floor)
        (term = term, c_raw = c_next_raw)
    end
    expected = reduce(+, (item.term for item in future))

    euler = _inverse_marginal_utility(params.beta .* expected, params.gamma, params.consumption_floor) ./
        @view(c[1:(params.n_ages - 1), :]) .- 1
    all_future_c = mapreduce(item -> vec(item.c_raw), vcat, future)
    neg_c = max.(-vcat(vec(c_raw), all_future_c), 0)
    loss = mean(abs2, euler) + mean(abs2, neg_c ./ (1 .+ abs.(neg_c)))
    diagnostics = (
        loss = loss,
        euler = euler,
        savings = a,
        savings_fraction = policy.fraction,
        consumption = c,
        raw_consumption = c_raw,
        next_capital = k_next,
        policy_error = analytic_olg_policy_error(a, states; params).relative,
    )
    return diagnostics, st_new
end

function analytic_olg_next_states(model, ps, st, states, z_next;
        params::AnalyticOLGParams = AnalyticOLGParams(),
        use_exact_policy::Bool = false)
    policy = if use_exact_policy
        analytic_olg_exact_policy(states; params)
    else
        raw, _ = model(analytic_olg_features(states; params), ps, st)
        analytic_olg_policy_from_raw(raw, states; params)
    end
    k_next = vcat(zeros(eltype(policy.savings), 1, size(states, 2)), policy.savings)
    return analytic_olg_assemble_states(z_next, k_next)
end

function benchmark_olg_policy_from_raw(raw, states; params::BenchmarkOLGParams = BenchmarkOLGParams())
    n_choices = params.n_ages - 1
    size(raw, 1) == 4 * n_choices + 1 ||
        throw(DimensionMismatch("benchmark OLG raw policy must have 4(n_ages - 1) + 1 rows"))
    heads = split_output_heads(raw, (capital = n_choices, lambda = n_choices, bond = n_choices, mu = n_choices, price = 1))
    cash, _, _, _, _, _, _, _, _ = _benchmark_olg_current_cash(states, params)
    cash_choice = @view(cash[1:n_choices, :])
    a_base = clamp.(0.20 .* max.(cash_choice, 0) .+ params.capital_base_min, params.capital_base_min, params.capital_base_max)
    capital = max.(a_base .* exp.(params.capital_log_scale .* tanh.(heads.capital)), params.min_capital_choice)
    bond_positive = max.(capital ./ params.kappa, params.consumption_floor) .* exp.(params.bond_log_scale .* tanh.(heads.bond))
    bond = -capital ./ params.kappa .+ bond_positive
    collateral = capital .+ params.kappa .* bond
    lambda = params.multiplier_scale .* NNlib.softplus.(heads.lambda .+ params.multiplier_bias)
    mu = params.multiplier_scale .* NNlib.softplus.(heads.mu .+ params.multiplier_bias)
    price = params.beta .* exp.(params.price_log_scale .* tanh.(heads.price))
    return (capital = capital, lambda = lambda, bond = bond, mu = mu, price = price, collateral = collateral, cash = cash)
end

function benchmark_olg_residual(model, ps, st, states;
        params::BenchmarkOLGParams = BenchmarkOLGParams())
    raw, st_new = model(benchmark_olg_features(states; params), ps, st)
    policy = benchmark_olg_policy_from_raw(raw, states; params)
    a, lambda, d, mu, p, collateral = policy.capital, policy.lambda, policy.bond, policy.mu, policy.price, policy.collateral
    cash = policy.cash
    n_choices = params.n_ages - 1
    _, k, _ = _benchmark_olg_blocks(states, params)
    _, _, _, _, r, _, y, _, _ = _benchmark_olg_current_cash(states, params)

    a_all = vcat(a, zeros(eltype(a), 1, size(a, 2)))
    d_all = vcat(d, zeros(eltype(d), 1, size(d, 2)))
    adjustment = a_all .- r .* k
    adjustment_cost = 0.5 * params.zeta .* adjustment .^ 2
    c_raw = cash .- a_all .- p .* d_all .- adjustment_cost
    c = max.(c_raw, params.consumption_floor)
    k_next = vcat(zeros(eltype(a), 1, size(a, 2)), a)
    b_next = vcat(zeros(eltype(d), 1, size(d, 2)), d)

    z, _, _ = _benchmark_olg_blocks(states, params)
    future = map(1:length(params.tfp)) do shock
        x_next = benchmark_olg_assemble_states(fill(shock, size(states, 2)), k_next, b_next)
        raw_next, _ = model(benchmark_olg_features(x_next; params), ps, st_new)
        next_policy = benchmark_olg_policy_from_raw(raw_next, x_next; params)
        cash_next = next_policy.cash
        _, k_future, _ = _benchmark_olg_blocks(x_next, params)
        _, _, _, _, r_next, _, _, _, _ = _benchmark_olg_current_cash(x_next, params)
        a_next_all = vcat(next_policy.capital, zeros(eltype(a), 1, size(a, 2)))
        d_next_all = vcat(next_policy.bond, zeros(eltype(d), 1, size(d, 2)))
        adjustment_next = a_next_all .- r_next .* k_future
        c_next_raw = cash_next .- a_next_all .- next_policy.price .* d_next_all .- 0.5 * params.zeta .* adjustment_next .^ 2
        c_next = max.(c_next_raw, params.consumption_floor)
        probs = reshape(params.transition[z, shock], 1, :)
        adj_factor_next = 1 .+ params.zeta .* @view(adjustment_next[2:params.n_ages, :])
        mu_next = _marginal_utility(@view(c_next[2:params.n_ages, :]), params.gamma, params.consumption_floor)
        (
            capital = probs .* r_next .* adj_factor_next .* mu_next,
            bond = probs .* mu_next,
            c_raw = c_next_raw,
            adj = adj_factor_next,
        )
    end
    expected_capital = reduce(+, (item.capital for item in future))
    expected_bond = reduce(+, (item.bond for item in future))

    adj_now = 1 .+ params.zeta .* @view(adjustment[1:n_choices, :])
    euler_cap_arg = (params.beta .* expected_capital .+ lambda .+ mu) ./ max.(adj_now, params.consumption_floor)
    euler_capital = _inverse_marginal_utility(euler_cap_arg, params.gamma, params.consumption_floor) ./
        @view(c[1:n_choices, :]) .- 1
    euler_bond_arg = (params.beta .* expected_bond .+ params.kappa .* mu) ./ max.(p, params.consumption_floor)
    euler_bond = _inverse_marginal_utility(euler_bond_arg, params.gamma, params.consumption_floor) ./
        @view(c[1:n_choices, :]) .- 1
    kkt_capital = lambda .* a
    kkt_bond = mu .* collateral
    bond_market_raw = sum(d; dims = 1)
    bond_market_scaled = bond_market_raw ./ (1 .+ sum(abs.(d); dims = 1))
    all_future_c = mapreduce(item -> vec(item.c_raw), vcat, future)
    neg_c = max.(-vcat(vec(c_raw), all_future_c), 0)
    all_future_adj = mapreduce(item -> vec(item.adj), vcat, future)
    bad_adj = max.(-vcat(vec(adj_now), all_future_adj), 0)
    loss = mean(abs2, euler_capital) + mean(abs2, euler_bond) +
        mean(abs2, kkt_capital) + mean(abs2, kkt_bond) +
        mean(abs2, bond_market_scaled) +
        mean(abs2, neg_c ./ (1 .+ abs.(neg_c))) +
        mean(abs2, bad_adj ./ (1 .+ abs.(bad_adj)))
    diagnostics = (
        loss = loss,
        euler_capital = euler_capital,
        euler_bond = euler_bond,
        kkt_capital = kkt_capital,
        kkt_bond = kkt_bond,
        bond_market = bond_market_raw,
        capital = a,
        bond = d,
        collateral = collateral,
        lambda = lambda,
        mu = mu,
        price = p,
        consumption = c,
        raw_consumption = c_raw,
        next_capital = k_next,
        next_bond = b_next,
    )
    return diagnostics, st_new
end

function benchmark_olg_next_states(model, ps, st, states, z_next;
        params::BenchmarkOLGParams = BenchmarkOLGParams())
    raw, _ = model(benchmark_olg_features(states; params), ps, st)
    policy = benchmark_olg_policy_from_raw(raw, states; params)
    k_next = vcat(zeros(eltype(policy.capital), 1, size(states, 2)), policy.capital)
    b_next = vcat(zeros(eltype(policy.bond), 1, size(states, 2)), policy.bond)
    return benchmark_olg_assemble_states(z_next, k_next, b_next)
end
