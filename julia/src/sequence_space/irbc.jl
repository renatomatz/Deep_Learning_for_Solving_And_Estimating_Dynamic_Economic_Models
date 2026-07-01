export sequence_irbc_history,
    sequence_irbc_policy,
    sequence_irbc_residual,
    sequence_irbc_forward_step

function sequence_irbc_history(params::IRBCParams; history_length::Integer, batch::Integer = 1, dtype::Type{<:AbstractFloat} = Float64)
    history_length > 0 || throw(ArgumentError("history_length must be positive"))
    batch > 0 || throw(ArgumentError("batch must be positive"))
    return zeros(dtype, params.countries, history_length, batch)
end

function _check_sequence_irbc_inputs(states, history, params::IRBCParams)
    assert_feature_batch(states, 2 * params.countries)
    size(history, 1) == params.countries || throw(DimensionMismatch("IRBC history must have one shock feature per country"))
    size(history, 3) == size(states, 2) || throw(DimensionMismatch("states and history must share a batch size"))
    return history
end

function _sequence_irbc_shock_block(shock, params::IRBCParams, batch::Integer)
    if shock isa Number
        return fill(shock, params.countries, batch)
    end
    block = reshape(shock, params.countries, :)
    size(block, 2) == batch && return block
    size(block, 2) == 1 && return repeat(block, 1, batch)
    throw(DimensionMismatch("shock must be a scalar, a countries-vector, or a countries-by-batch matrix"))
end

function sequence_irbc_policy(model, ps, st, states, history;
        params::IRBCParams = IRBCParams(), irreversible::Bool = false)
    _check_sequence_irbc_inputs(states, history, params)
    raw, st_new = model(flatten_history(history), ps, st)
    quantities = _irbc_current_quantities(raw, states, params; irreversible)
    return quantities, st_new
end

function sequence_irbc_residual(model, ps, st, states, history, rule::QuadratureRule;
        params::IRBCParams = IRBCParams(), irreversible::Bool = false)
    _check_sequence_irbc_inputs(states, history, params)
    rule.nodes isa AbstractMatrix || throw(ArgumentError("sequence_irbc_residual expects a matrix quadrature rule"))
    size(rule.nodes, 1) == params.countries || throw(DimensionMismatch("quadrature nodes must have one row per country"))

    q, st_new = sequence_irbc_policy(model, ps, st, states, history; params, irreversible)
    batch = size(states, 2)
    terms = map(eachindex(rule.weights)) do i
        node = @view rule.nodes[:, i]
        z_next = _irbc_next_shocks(q.z, node, params)
        next_states = vcat(z_next, q.next_capital)
        next_history = prepend_history(history, repeat(reshape(node, params.countries, 1), 1, batch))
        raw_next, _ = model(flatten_history(next_history), ps, st_new)
        policy_next = irbc_policy_from_raw(raw_next, params; irreversible)
        y_next = irbc_output(z_next, q.next_capital, params)
        investment_next = policy_next.investment .* y_next
        consumption_next = _irbc_future_consumption(policy_next, y_next, investment_next)
        gross_return_next = 1 - params.delta .+ params.alpha .* z_next .* q.next_capital .^ (params.alpha - 1)
        return rule.weights[i] .* gross_return_next ./ consumption_next
    end
    expected_return = reduce(+, terms)
    euler = 1 .- params.beta .* q.consumption .* expected_return
    if irreversible
        complementarity = fischer_burmeister(q.investment, q.policy.multiplier .- euler)
        loss = mean(abs2, euler) + mean(abs2, complementarity) + mean(abs2, q.resource)
        return (
            loss = loss,
            euler = euler,
            resource = q.resource,
            complementarity = complementarity,
            multiplier = q.policy.multiplier,
            investment = q.investment,
            consumption = q.consumption,
            next_capital = q.next_capital,
        ), st_new
    end
    loss = mean(abs2, euler) + mean(abs2, q.resource)
    return (
        loss = loss,
        euler = euler,
        resource = q.resource,
        investment = q.investment,
        consumption = q.consumption,
        next_capital = q.next_capital,
    ), st_new
end

function sequence_irbc_forward_step(model, ps, st, states, history, shock;
        params::IRBCParams = IRBCParams(), irreversible::Bool = false)
    q, st_new = sequence_irbc_policy(model, ps, st, states, history; params, irreversible)
    shock_block = _sequence_irbc_shock_block(shock, params, size(states, 2))
    z_next = q.z .^ params.shock_persistence .* exp.(params.shock_std .* shock_block)
    states_next = vcat(z_next, q.next_capital)
    history_next = prepend_history(history, shock_block)
    return states_next, history_next, st_new
end
