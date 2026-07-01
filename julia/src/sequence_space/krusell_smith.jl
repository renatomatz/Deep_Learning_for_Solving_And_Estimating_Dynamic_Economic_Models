using Statistics: mean

export SequenceKSParams,
    sequence_ks_prices,
    sequence_ks_history,
    sequence_ks_current_z,
    sequence_ks_initial_distribution,
    sequence_ks_distribution_aggregates,
    sequence_ks_policy_grid,
    sequence_ks_residual,
    sequence_ks_distribution_step,
    sequence_ks_forward_step

struct SequenceKSParams{T}
    alpha::T
    beta::T
    delta::T
    gamma::T
    eps_safe::T
    capital_grid::Vector{T}
    idio_income::Vector{T}
    idio_transition::Matrix{T}
    aggregate_z::Vector{T}
    aggregate_transition::Matrix{T}
end

function SequenceKSParams(; alpha = 0.36, beta = 0.96, delta = 0.08, gamma = 2.0,
        eps_safe = 1e-6, capital_grid = collect(range(0.0, 25.0; length = 32)),
        idio_income = [0.55, 1.45], idio_transition = [0.92 0.08; 0.08 0.92],
        aggregate_z = [0.99, 1.01], aggregate_transition = [0.9 0.1; 0.1 0.9])
    eps_safe > 0 || throw(ArgumentError("eps_safe must be positive"))
    vals = promote(alpha, beta, delta, gamma, eps_safe)
    T = typeof(vals[1])
    grid = T.(collect(capital_grid))
    _check_grid(grid)
    incomes = T.(collect(idio_income))
    length(incomes) >= 1 || throw(ArgumentError("idio_income must be nonempty"))
    idio_trans = T.(idio_transition)
    size(idio_trans) == (length(incomes), length(incomes)) || throw(DimensionMismatch("idio_transition size must match idio_income"))
    validate_transition_matrix(idio_trans)
    z_vals = T.(collect(aggregate_z))
    length(z_vals) >= 1 || throw(ArgumentError("aggregate_z must be nonempty"))
    agg_trans = T.(aggregate_transition)
    size(agg_trans) == (length(z_vals), length(z_vals)) || throw(DimensionMismatch("aggregate_transition size must match aggregate_z"))
    validate_transition_matrix(agg_trans)
    return SequenceKSParams(vals..., grid, incomes, idio_trans, z_vals, agg_trans)
end

function sequence_ks_prices(K, L, Z; params::SequenceKSParams = SequenceKSParams())
    Ksafe = max.(K, params.eps_safe)
    Lsafe = max.(L, params.eps_safe)
    ratio = Ksafe ./ Lsafe
    R = 1 .+ params.alpha .* Z .* ratio .^ (params.alpha - 1) .- params.delta
    w = (1 - params.alpha) .* Z .* ratio .^ params.alpha
    return (R = R, w = w)
end

function sequence_ks_history(params::SequenceKSParams; history_length::Integer, batch::Integer = 1, z_index::Integer = 1, dtype::Type{<:AbstractFloat} = Float64)
    history_length > 0 || throw(ArgumentError("history_length must be positive"))
    batch > 0 || throw(ArgumentError("batch must be positive"))
    1 <= z_index <= length(params.aggregate_z) || throw(ArgumentError("z_index is out of range"))
    current = _sequence_ks_onehot(params, z_index, batch, dtype)
    return repeat(reshape(current, size(current, 1), 1, batch), 1, history_length, 1)
end

function sequence_ks_current_z(history, params::SequenceKSParams = SequenceKSParams())
    n_z = length(params.aggregate_z)
    if size(history, 1) == n_z
        weights = @view history[:, 1, :]
        return sum(reshape(params.aggregate_z, :, 1) .* weights; dims = 1)
    elseif size(history, 1) == n_z + 1
        return @view history[(n_z + 1):(n_z + 1), 1, :]
    end
    throw(DimensionMismatch("history feature count must be aggregate_z or aggregate_z plus shock value"))
end

function sequence_ks_initial_distribution(params::SequenceKSParams = SequenceKSParams(); K_target = mean(params.capital_grid), idio_masses = nothing)
    masses = idio_masses === nothing ? fill(inv(length(params.idio_income)), length(params.idio_income)) : collect(idio_masses)
    length(masses) == length(params.idio_income) || throw(DimensionMismatch("idio_masses length must match idio states"))
    hist = zeros(eltype(params.capital_grid), length(params.idio_income), length(params.capital_grid))
    for i in eachindex(masses)
        hist[i, :] .= redistribute_mass(params.capital_grid, K_target, masses[i])
    end
    return hist
end

function sequence_ks_distribution_aggregates(distribution, params::SequenceKSParams = SequenceKSParams())
    size(distribution) == (length(params.idio_income), length(params.capital_grid)) ||
        throw(DimensionMismatch("distribution must be idio states by capital grid"))
    mass_by_idio = sum(distribution; dims = 2)
    capital = sum(reshape(params.capital_grid, 1, :) .* distribution)
    labor = sum(reshape(params.idio_income, :, 1) .* mass_by_idio)
    return (capital = capital, labor = labor, mass = sum(distribution), mass_by_idio = mass_by_idio)
end

function _sequence_ks_onehot(params::SequenceKSParams, z_index::Integer, batch::Integer, dtype)
    n_z = length(params.aggregate_z)
    block = dtype.(reshape(1:n_z, :, 1) .== z_index)
    z_value = fill(dtype(params.aggregate_z[z_index]), 1, batch)
    return vcat(repeat(block, 1, batch), z_value)
end

function _sequence_ks_policy_from_raw(raw, history, distribution, params::SequenceKSParams)
    n_idio = length(params.idio_income)
    size(raw, 1) == n_idio || throw(DimensionMismatch("KS actor must output one MPC head per idiosyncratic state"))
    size(raw, 2) == size(history, 3) || throw(DimensionMismatch("actor output batch must match history batch"))
    size(raw, 2) == 1 || throw(DimensionMismatch("classroom KS helpers currently expect one aggregate history at a time"))
    agg = sequence_ks_distribution_aggregates(distribution, params)
    Z = only(sequence_ks_current_z(history, params))
    prices = sequence_ks_prices(agg.capital, agg.labor, Z; params)
    mpc = 0.05 .+ 0.90 .* NNlib.sigmoid.(raw)
    cash = prices.R .* reshape(params.capital_grid, 1, :) .+ prices.w .* reshape(params.idio_income, :, 1)
    consumption = clamp.(mpc .* cash, params.eps_safe, cash .- params.eps_safe)
    savings = max.(cash .- consumption, first(params.capital_grid))
    return (
        consumption = consumption,
        savings = savings,
        mpc = mpc,
        cash = cash,
        prices = prices,
        aggregate = agg,
        Z = Z,
    )
end

function sequence_ks_policy_grid(model, ps, st, history, distribution;
        params::SequenceKSParams = SequenceKSParams())
    raw, st_new = model(flatten_history(history), ps, st)
    policy = _sequence_ks_policy_from_raw(raw, history, distribution, params)
    return policy, st_new
end

function sequence_ks_residual(model, ps, st, history, distribution;
        params::SequenceKSParams = SequenceKSParams())
    policy, st_new = sequence_ks_policy_grid(model, ps, st, history, distribution; params)
    current_mu = policy.consumption .^ (-params.gamma)
    current_probs = @view history[1:length(params.aggregate_z), 1, 1]
    labor_next = policy.aggregate.labor
    capital_next = sum(policy.savings .* distribution)
    expected_terms = map(eachindex(params.aggregate_z)) do z_next_index
        z_block = _sequence_ks_onehot(params, z_next_index, 1, eltype(history))
        next_history = prepend_history(history, z_block)
        raw_next, _ = model(flatten_history(next_history), ps, st_new)
        probs_to_next = sum(current_probs .* params.aggregate_transition[:, z_next_index])
        prices_next = sequence_ks_prices(capital_next, labor_next, params.aggregate_z[z_next_index]; params)
        mpc_next = 0.05 .+ 0.90 .* NNlib.sigmoid.(raw_next)
        cash_next = prices_next.R .* reshape(params.capital_grid, 1, :) .+ prices_next.w .* reshape(params.idio_income, :, 1)
        consumption_next = clamp.(mpc_next .* cash_next, params.eps_safe, cash_next .- params.eps_safe)
        return probs_to_next .* prices_next.R .* (params.idio_transition * consumption_next .^ (-params.gamma))
    end
    expected_mu = reduce(+, expected_terms)
    euler_gap = current_mu .- params.beta .* expected_mu
    slack = policy.savings .- first(params.capital_grid)
    complementarity = fischer_burmeister(slack, euler_gap)
    weighted_euler = sum(distribution .* complementarity .^ 2) / max(policy.aggregate.mass, params.eps_safe)
    capital_market = capital_next - policy.aggregate.capital
    loss = weighted_euler + abs2(capital_market / (1 + abs(policy.aggregate.capital)))
    return (
        loss = loss,
        euler = complementarity,
        euler_gap = euler_gap,
        capital_market = capital_market,
        consumption = policy.consumption,
        savings = policy.savings,
        prices = policy.prices,
        aggregate = policy.aggregate,
    ), st_new
end

function sequence_ks_distribution_step(distribution, policy; params::SequenceKSParams = SequenceKSParams(), clip::Bool = true)
    return young_step(params.capital_grid, distribution, policy.savings; transition = params.idio_transition, clip)
end

function sequence_ks_forward_step(model, ps, st, history, distribution, z_next_index::Integer;
        params::SequenceKSParams = SequenceKSParams(), clip::Bool = true)
    1 <= z_next_index <= length(params.aggregate_z) || throw(ArgumentError("z_next_index is out of range"))
    policy, st_new = sequence_ks_policy_grid(model, ps, st, history, distribution; params)
    distribution_next = sequence_ks_distribution_step(distribution, policy; params, clip)
    history_next = prepend_history(history, _sequence_ks_onehot(params, z_next_index, size(history, 3), eltype(history)))
    return history_next, distribution_next, st_new
end
