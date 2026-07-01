export BrockMirmanParams,
    bm_steady_state,
    bm_full_depreciation_policy,
    deterministic_bm_residual,
    stochastic_bm_residual

struct BrockMirmanParams{T}
    alpha::T
    beta::T
    delta::T
    rho_z::T
    sigma_z::T
end

function BrockMirmanParams(; alpha = 0.36, beta = 0.99, delta = 1.0, rho_z = 0.9, sigma_z = 0.04)
    return BrockMirmanParams(promote(alpha, beta, delta, rho_z, sigma_z)...)
end

function bm_steady_state(params::BrockMirmanParams)
    return ((1 / params.beta - 1 + params.delta) / params.alpha)^(1 / (params.alpha - 1))
end

bm_full_depreciation_policy(k, params::BrockMirmanParams) = params.alpha * params.beta .* k .^ params.alpha

function _policy_savings(model, ps, st, x; transform)
    raw, st_new = model(x, ps, st)
    return transform(raw), st_new
end

function deterministic_bm_residual(model, ps, st, k_feature_batch;
        params::BrockMirmanParams = BrockMirmanParams(),
        transform = identity)
    k = assert_feature_batch(k_feature_batch, 1)
    savings, st_new = _policy_savings(model, ps, st, k; transform)

    output = k .^ params.alpha
    k_next = (1 - params.delta) .* k .+ output .* savings
    consumption = output .* (1 .- savings)

    output_next = k_next .^ params.alpha
    savings_next, _ = _policy_savings(model, ps, st_new, k_next; transform)
    consumption_next = output_next .* (1 .- savings_next)
    return_next = params.alpha .* k_next .^ (params.alpha - 1)

    residual = 1 .- consumption_next ./ (params.beta .* consumption .* (return_next .+ 1 .- params.delta))
    diagnostics = (
        loss = mse_loss(residual, zero(residual)),
        residual = residual,
        savings = savings,
        consumption = consumption,
        next_capital = k_next,
        next_consumption = consumption_next,
        next_return = return_next,
    )
    return diagnostics, st_new
end

function stochastic_bm_residual(model, ps, st, state_feature_batch, rule::QuadratureRule;
        params::BrockMirmanParams = BrockMirmanParams(delta = 0.1),
        transform = identity)
    rule.nodes isa AbstractVector ||
        throw(ArgumentError("stochastic_bm_residual expects a one-dimensional quadrature rule"))

    x = assert_feature_batch(state_feature_batch, 2)
    z = @view x[1:1, :]
    k = @view x[2:2, :]

    output = z .* k .^ params.alpha
    savings, st_new = _policy_savings(model, ps, st, x; transform)
    k_next = (1 - params.delta) .* k .+ output .* savings
    consumption = output .* (1 .- savings)

    expectation_terms = [
        begin
            z_next = exp.(params.rho_z .* log.(z) .+ params.sigma_z .* node)
            x_next = vcat(z_next, k_next)
            output_next = z_next .* k_next .^ params.alpha
            return_next = params.alpha .* z_next .* k_next .^ (params.alpha - 1)
            savings_next, _ = _policy_savings(model, ps, st_new, x_next; transform)
            consumption_next = output_next .* (1 .- savings_next)
            weight .* (1 ./ consumption_next) .* (1 .- params.delta .+ return_next)
        end
        for (node, weight) in zip(rule.nodes, rule.weights)
    ]
    expectation = reduce(+, expectation_terms)

    residual = 1 .- 1 ./ (consumption .* params.beta .* expectation)
    diagnostics = (
        loss = mse_loss(residual, zero(residual)),
        residual = residual,
        savings = savings,
        consumption = consumption,
        next_capital = k_next,
        lhs = 1 ./ consumption,
        rhs = params.beta .* expectation,
    )
    return diagnostics, st_new
end
