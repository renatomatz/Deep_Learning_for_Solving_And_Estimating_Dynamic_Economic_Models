export IRBCParams,
    irbc_steady_state_capital,
    irbc_output,
    irbc_policy_from_raw,
    irbc_sample_states,
    irbc_stroud_rule,
    irbc_smooth_residual,
    irbc_irreversible_residual,
    irbc_zero_shock_step

using Statistics: mean

struct IRBCParams{T}
    countries::Int
    alpha::T
    beta::T
    delta::T
    shock_std::T
    shock_persistence::T
end

function IRBCParams(; countries::Integer = 2, alpha = 0.36, beta = 0.99, delta = 0.1, shock_std = 0.02, shock_persistence = 0.9)
    countries > 0 || throw(ArgumentError("countries must be positive"))
    0 <= shock_persistence <= 1 || throw(ArgumentError("shock_persistence must lie in [0, 1]"))
    vals = promote(alpha, beta, delta, shock_std, shock_persistence)
    return IRBCParams(countries, vals...)
end

function irbc_steady_state_capital(params::IRBCParams)
    return ((1 / params.beta - 1 + params.delta) / params.alpha)^(1 / (params.alpha - 1))
end

irbc_output(z, k, params::IRBCParams) = z .* k .^ params.alpha

function _irbc_state_blocks(states, params::IRBCParams)
    x = assert_feature_batch(states, 2 * params.countries)
    z = @view x[1:params.countries, :]
    k = @view x[(params.countries + 1):(2 * params.countries), :]
    return z, k
end

function irbc_policy_from_raw(raw, params::IRBCParams; irreversible::Bool = false)
    if irreversible
        if size(raw, 1) == 3 * params.countries
            heads = split_output_heads(raw, (consumption = params.countries, investment = params.countries, multiplier = params.countries))
            consumption = NNlib.sigmoid.(heads.consumption)
        elseif size(raw, 1) == 2 * params.countries
            heads = split_output_heads(raw, (investment = params.countries, multiplier = params.countries))
            consumption = nothing
        else
            throw(DimensionMismatch("irreversible IRBC policy must have two or three heads per country"))
        end
        investment = NNlib.sigmoid.(heads.investment)
        multiplier = positive_softplus(heads.multiplier; floor = 0.0, scale = 0.1)
        return (consumption = consumption, investment = investment, multiplier = multiplier)
    end

    if size(raw, 1) == 2 * params.countries
        heads = split_output_heads(raw, (consumption = params.countries, investment = params.countries))
        return (consumption = NNlib.sigmoid.(heads.consumption), investment = NNlib.sigmoid.(heads.investment))
    elseif size(raw, 1) == params.countries
        return (consumption = nothing, investment = NNlib.sigmoid.(raw))
    end
    throw(DimensionMismatch("smooth IRBC policy must have one or two heads per country"))
end

function irbc_sample_states(rng, params::IRBCParams, n::Integer; shock_radius = 0.06, capital_radius = 0.35)
    n > 0 || throw(ArgumentError("n must be positive"))
    kstar = irbc_steady_state_capital(params)
    z = exp.(shock_radius .* randn(rng, params.countries, n))
    k = kstar .* (1 .+ capital_radius .* (2 .* rand(rng, params.countries, n) .- 1))
    return vcat(z, max.(k, 0.05kstar))
end

irbc_stroud_rule(params::IRBCParams) = stroud3_normal_rule(params.countries)

function _irbc_current_quantities(raw, states, params::IRBCParams; irreversible::Bool = false)
    z, k = _irbc_state_blocks(states, params)
    policy = irbc_policy_from_raw(raw, params; irreversible)
    y = irbc_output(z, k, params)
    investment = policy.investment .* y
    consumption = isnothing(policy.consumption) ? y .- investment : policy.consumption .* y
    resource = sum(consumption .+ investment .- y; dims = 1)
    k_next = (1 - params.delta) .* k .+ investment
    return (z = z, k = k, y = y, policy = policy, investment = investment, consumption = consumption, resource = resource, next_capital = k_next)
end

function _irbc_next_shocks(z, node, params::IRBCParams)
    shock = reshape(node, params.countries, 1)
    return z .^ params.shock_persistence .* exp.(params.shock_std .* shock)
end

function _irbc_future_consumption(policy, y, investment)
    return isnothing(policy.consumption) ? y .- investment : policy.consumption .* y
end

function _irbc_expected_marginal_return(model, ps, st, z, k_next, params::IRBCParams; irreversible::Bool = false, rule = irbc_stroud_rule(params))
    terms = map(eachindex(rule.weights)) do i
        z_next = _irbc_next_shocks(z, @view(rule.nodes[:, i]), params)
        next_states = vcat(z_next, k_next)
        raw_next, _ = model(next_states, ps, st)
        policy_next = irbc_policy_from_raw(raw_next, params; irreversible)
        y_next = irbc_output(z_next, k_next, params)
        investment_next = policy_next.investment .* y_next
        consumption_next = _irbc_future_consumption(policy_next, y_next, investment_next)
        gross_return_next = 1 - params.delta .+ params.alpha .* z_next .* k_next .^ (params.alpha - 1)
        return rule.weights[i] .* gross_return_next ./ consumption_next
    end
    return foldl(+, terms)
end

function irbc_smooth_residual(model, ps, st, states; params::IRBCParams = IRBCParams(), rule = irbc_stroud_rule(params))
    raw, st_new = model(states, ps, st)
    q = _irbc_current_quantities(raw, states, params)
    expected_return = _irbc_expected_marginal_return(model, ps, st_new, q.z, q.next_capital, params; rule)
    euler = 1 .- params.beta .* q.consumption .* expected_return
    loss = mean(abs2, euler) + mean(abs2, q.resource)
    return (loss = loss, euler = euler, resource = q.resource, investment = q.investment, consumption = q.consumption, next_capital = q.next_capital), st_new
end

function irbc_irreversible_residual(model, ps, st, states; params::IRBCParams = IRBCParams(), rule = irbc_stroud_rule(params))
    raw, st_new = model(states, ps, st)
    q = _irbc_current_quantities(raw, states, params; irreversible = true)
    expected_return = _irbc_expected_marginal_return(model, ps, st_new, q.z, q.next_capital, params; irreversible = true, rule)
    euler_wedge = 1 .- params.beta .* q.consumption .* expected_return
    complementarity = fischer_burmeister(q.investment, q.policy.multiplier .- euler_wedge)
    loss = mean(abs2, euler_wedge) + mean(abs2, complementarity) + mean(abs2, q.resource)
    return (loss = loss, euler = euler_wedge, resource = q.resource, complementarity = complementarity, multiplier = q.policy.multiplier, investment = q.investment, consumption = q.consumption, next_capital = q.next_capital), st_new
end

function irbc_zero_shock_step(model, ps, st, states; params::IRBCParams = IRBCParams(), irreversible::Bool = false)
    pieces, st_new = irreversible ? irbc_irreversible_residual(model, ps, st, states; params) : irbc_smooth_residual(model, ps, st, states; params)
    z, _ = _irbc_state_blocks(states, params)
    z_next = z .^ params.shock_persistence
    return vcat(z_next, pieces.next_capital), st_new
end
