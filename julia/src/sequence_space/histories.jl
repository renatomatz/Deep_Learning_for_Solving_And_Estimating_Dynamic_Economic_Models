export flatten_history,
    unflatten_history,
    prepend_history,
    quadrature_histories,
    flatten_quadrature_histories,
    encode_markov_history,
    SequenceBrockMirmanParams,
    sequence_bm_steady_state,
    sequence_bm_initial_state,
    sequence_bm_expand_state,
    sequence_bm_policy,
    sequence_bm_residual,
    sequence_bm_forward_step

struct SequenceBrockMirmanParams{T}
    alpha::T
    beta::T
    gamma::T
    delta::T
    rho_z::T
    sigma_z::T
    eps_safe::T
end

function SequenceBrockMirmanParams(; alpha = 1 / 3, beta = 0.97, gamma = 1.0, delta = 1.0, rho_z = 0.98, sigma_z = 0.01, eps_safe = 1e-6)
    eps_safe > 0 || throw(ArgumentError("eps_safe must be positive"))
    vals = promote(alpha, beta, gamma, delta, rho_z, sigma_z, eps_safe)
    return SequenceBrockMirmanParams(vals...)
end

function flatten_history(history::AbstractArray{T,3}) where {T}
    features_per_lag, history_length, batch = size(history)
    return reshape(history, features_per_lag * history_length, batch)
end

function unflatten_history(flat::AbstractMatrix, features_per_lag::Integer, history_length::Integer)
    features_per_lag > 0 || throw(ArgumentError("features_per_lag must be positive"))
    history_length > 0 || throw(ArgumentError("history_length must be positive"))
    size(flat, 1) == features_per_lag * history_length ||
        throw(DimensionMismatch("flat history row count must equal features_per_lag * history_length"))
    return reshape(flat, features_per_lag, history_length, size(flat, 2))
end

function prepend_history(history::AbstractArray{T,3}, new_block::AbstractMatrix) where {T}
    features_per_lag, history_length, batch = size(history)
    size(new_block) == (features_per_lag, batch) ||
        throw(DimensionMismatch("new_block must be features_per_lag-by-batch"))
    newest = reshape(new_block, features_per_lag, 1, batch)
    history_length == 1 && return newest
    return cat(newest, history[:, 1:(history_length - 1), :]; dims = 2)
end

function quadrature_histories(history::AbstractArray{T,3}, nodes::AbstractVector) where {T}
    features_per_lag, history_length, batch = size(history)
    features_per_lag == 1 || throw(DimensionMismatch("vector nodes require one feature per lag"))
    out = Array{promote_type(T, eltype(nodes))}(undef, features_per_lag, history_length, batch, length(nodes))
    for (q, node) in pairs(nodes)
        out[:, :, :, q] = prepend_history(history, fill(node, 1, batch))
    end
    return out
end

function quadrature_histories(history::AbstractArray{T,3}, nodes::AbstractMatrix) where {T}
    features_per_lag, history_length, batch = size(history)
    size(nodes, 1) == features_per_lag || throw(DimensionMismatch("matrix nodes must be features_per_lag-by-nodes"))
    out = Array{promote_type(T, eltype(nodes))}(undef, features_per_lag, history_length, batch, size(nodes, 2))
    for q in axes(nodes, 2)
        out[:, :, :, q] = prepend_history(history, repeat(@view(nodes[:, q]), 1, batch))
    end
    return out
end

function flatten_quadrature_histories(histories::AbstractArray{T,4}) where {T}
    features_per_lag, history_length, batch, n_nodes = size(histories)
    return reshape(histories, features_per_lag * history_length, batch * n_nodes)
end

function encode_markov_history(z_hist, z_values; onehot::Bool = true)
    z = collect(z_hist)
    values = collect(z_values)
    if onehot
        encoded = zeros(Float64, length(values), length(z), 1)
        for (lag, value) in pairs(z)
            idx = findfirst(==(value), values)
            idx === nothing && throw(ArgumentError("history contains a value outside z_values"))
            encoded[idx, lag, 1] = 1.0
        end
        return encoded
    end
    return reshape(float.(z), 1, length(z), 1)
end

sequence_bm_steady_state(params::SequenceBrockMirmanParams) =
    (params.alpha / (1 / params.beta + params.delta - 1))^(1 / (1 - params.alpha))

function sequence_bm_expand_state(k, z; params::SequenceBrockMirmanParams = SequenceBrockMirmanParams())
    K = max.(k, params.eps_safe)
    Z = exp.(z)
    y = Z .* K .^ params.alpha .+ (1 - params.delta) .* K
    mpk = params.alpha .* Z .* K .^ (params.alpha - 1) .+ (1 - params.delta)
    return vcat(reshape(K, 1, :), reshape(z, 1, :), reshape(Z, 1, :), reshape(y, 1, :), reshape(mpk, 1, :))
end

function sequence_bm_initial_state(params::SequenceBrockMirmanParams = SequenceBrockMirmanParams(); batch::Integer = 1)
    batch > 0 || throw(ArgumentError("batch must be positive"))
    k = fill(sequence_bm_steady_state(params), batch)
    z = zeros(typeof(params.alpha), batch)
    return sequence_bm_expand_state(k, z; params)
end

_sequence_u_c(c, params::SequenceBrockMirmanParams) = (1 - params.beta) .* max.(c, params.eps_safe) .^ (-params.gamma)
_sequence_u_c_inv(u, params::SequenceBrockMirmanParams) = max.(u ./ (1 - params.beta), params.eps_safe) .^ (-inv(params.gamma))

function sequence_bm_policy(model, ps, st, states, history;
        params::SequenceBrockMirmanParams = SequenceBrockMirmanParams(),
        transform = NNlib.sigmoid)
    raw, st_new = model(flatten_history(history), ps, st)
    savings_rate = clamp.(transform(raw), params.eps_safe, 1 - params.eps_safe)
    y = @view states[4:4, :]
    savings = savings_rate .* y
    consumption = (1 .- savings_rate) .* y
    mu = _sequence_u_c(consumption, params)
    return (savings = savings, consumption = consumption, marginal_utility = mu, savings_rate = savings_rate), st_new
end

function sequence_bm_residual(model, ps, st, states, history, rule::QuadratureRule;
        params::SequenceBrockMirmanParams = SequenceBrockMirmanParams(),
        transform = NNlib.sigmoid)
    rule.nodes isa AbstractVector || throw(ArgumentError("sequence_bm_residual expects a one-dimensional quadrature rule"))
    assert_feature_batch(states, 5)
    size(history, 1) == 1 || throw(DimensionMismatch("Brock-Mirman histories must have one feature per lag"))
    size(history, 3) == size(states, 2) || throw(DimensionMismatch("states and history must share a batch size"))

    policy, st_new = sequence_bm_policy(model, ps, st, states, history; params, transform)
    k_next = policy.savings
    z = @view states[2:2, :]
    expectation_terms = map(zip(rule.nodes, rule.weights)) do (node, weight)
        z_next = params.rho_z .* z .+ params.sigma_z .* node
        next_states = sequence_bm_expand_state(k_next, z_next; params)
        next_history = prepend_history(history, fill(node, 1, size(states, 2)))
        next_policy, _ = sequence_bm_policy(model, ps, st_new, next_states, next_history; params, transform)
        mpk_next = @view next_states[5:5, :]
        weight .* next_policy.marginal_utility .* mpk_next
    end
    expectation = reduce(+, expectation_terms)
    euler = _sequence_u_c_inv(params.beta .* expectation, params) ./ policy.consumption .- 1
    diagnostics = (
        loss = mse_loss(euler, zero(euler)),
        euler = euler,
        savings = policy.savings,
        consumption = policy.consumption,
        savings_rate = policy.savings_rate,
    )
    return diagnostics, st_new
end

function sequence_bm_forward_step(model, ps, st, states, history, shock;
        params::SequenceBrockMirmanParams = SequenceBrockMirmanParams(),
        transform = NNlib.sigmoid)
    policy, st_new = sequence_bm_policy(model, ps, st, states, history; params, transform)
    z = @view states[2:2, :]
    shock_block = shock isa Number ? fill(shock, 1, size(states, 2)) : reshape(shock, 1, :)
    z_next = params.rho_z .* z .+ params.sigma_z .* shock_block
    states_next = sequence_bm_expand_state(policy.savings, z_next; params)
    history_next = prepend_history(history, shock_block)
    return states_next, history_next, st_new
end
