export SMMBrockMirmanParams,
    smm_bm_steady_state,
    sample_smm_scalar_states,
    sample_smm_joint_states,
    smm_scalar_euler_loss,
    smm_joint_euler_loss,
    common_random_shocks,
    simulate_smm_scalar_rho,
    simulate_smm_joint_theta,
    autocorr_columns,
    smm_scalar_moments,
    smm_joint_moments,
    smm_criterion,
    smm_grid_estimate,
    smm_moment_sensitivity_1d,
    smm_moment_jacobian_2d,
    smm_identification_svd

using LinearAlgebra: svd
using NNlib
using Random: AbstractRNG, rand, randn
using Statistics: mean, std

struct SMMBrockMirmanParams{T}
    alpha::T
    delta::T
    beta::T
    sigma_z::T
    z_bounds::Tuple{T,T}
    k_bounds::Tuple{T,T}
    rho_bounds::Tuple{T,T}
    beta_bounds::Tuple{T,T}
    eps_safe::T
end

function SMMBrockMirmanParams(; alpha = 0.36, delta = 0.10, beta = 0.96, sigma_z = 0.04,
        z_bounds = (0.70, 1.30), k_bounds = (0.50, 12.0),
        rho_bounds = (0.50, 0.99), beta_bounds = (0.92, 0.99), eps_safe = 1e-10)
    alpha > 0 || throw(ArgumentError("alpha must be positive"))
    delta >= 0 || throw(ArgumentError("delta must be nonnegative"))
    beta > 0 || throw(ArgumentError("beta must be positive"))
    sigma_z >= 0 || throw(ArgumentError("sigma_z must be nonnegative"))
    eps_safe > 0 || throw(ArgumentError("eps_safe must be positive"))
    vals = promote(alpha, delta, beta, sigma_z, z_bounds..., k_bounds..., rho_bounds..., beta_bounds..., eps_safe)
    return SMMBrockMirmanParams(vals[1], vals[2], vals[3], vals[4],
        (vals[5], vals[6]), (vals[7], vals[8]), (vals[9], vals[10]), (vals[11], vals[12]), vals[13])
end

function smm_bm_steady_state(beta_value = SMMBrockMirmanParams().beta; params::SMMBrockMirmanParams = SMMBrockMirmanParams())
    K = (params.alpha / (1 / beta_value - 1 + params.delta))^(1 / (1 - params.alpha))
    s = params.delta * K / K^params.alpha
    return (capital = K, savings = s)
end

function _draw_uniform(rng::AbstractRNG, bounds, n::Integer)
    return bounds[1] .+ (bounds[2] - bounds[1]) .* rand(rng, n)
end

function sample_smm_scalar_states(rng::AbstractRNG, n::Integer; params::SMMBrockMirmanParams = SMMBrockMirmanParams())
    n > 0 || throw(ArgumentError("n must be positive"))
    z = _draw_uniform(rng, params.z_bounds, n)
    K = _draw_uniform(rng, params.k_bounds, n)
    rho = _draw_uniform(rng, params.rho_bounds, n)
    return vcat(reshape(z, 1, :), reshape(K, 1, :), reshape(rho, 1, :))
end

function sample_smm_joint_states(rng::AbstractRNG, n::Integer; params::SMMBrockMirmanParams = SMMBrockMirmanParams())
    n > 0 || throw(ArgumentError("n must be positive"))
    z = _draw_uniform(rng, params.z_bounds, n)
    K = _draw_uniform(rng, params.k_bounds, n)
    beta = _draw_uniform(rng, params.beta_bounds, n)
    rho = _draw_uniform(rng, params.rho_bounds, n)
    return vcat(reshape(z, 1, :), reshape(K, 1, :), reshape(beta, 1, :), reshape(rho, 1, :))
end

function _policy_savings_bounded(model, ps, st, features; transform)
    raw, st_new = model(features, ps, st)
    savings = transform === identity ? raw : transform.(raw)
    return clamp.(savings, eps(eltype(float.(savings))), 1 - eps(eltype(float.(savings)))), st_new
end

function _scalar_anchor_loss(model, ps, st; params::SMMBrockMirmanParams, transform, rho_anchor = 0.90)
    ss = smm_bm_steady_state(params.beta; params)
    x = reshape([one(ss.capital), ss.capital, rho_anchor], 3, 1)
    savings, _ = _policy_savings_bounded(model, ps, st, x; transform)
    return mean(abs2, savings .- ss.savings)
end

function _joint_anchor_loss(model, ps, st, betas; params::SMMBrockMirmanParams, transform, rho_anchor = 0.90)
    beta_vec = vec(betas)
    steady_states = [smm_bm_steady_state(beta_value; params) for beta_value in beta_vec]
    K = [ss.capital for ss in steady_states]
    s = [ss.savings for ss in steady_states]
    x = vcat(ones(1, length(beta_vec)), reshape(K, 1, :), reshape(beta_vec, 1, :), fill(rho_anchor, 1, length(beta_vec)))
    savings, _ = _policy_savings_bounded(model, ps, st, x; transform)
    return mean(abs2, savings .- reshape(s, 1, :))
end

function smm_scalar_euler_loss(model, ps, st, states::AbstractMatrix, rule::QuadratureRule;
        params::SMMBrockMirmanParams = SMMBrockMirmanParams(),
        anchor_weight::Real = 1e-3,
        transform = identity)
    size(states, 1) == 3 || throw(DimensionMismatch("scalar SMM states must be 3-by-batch: z, K, rho"))
    z = @view states[1:1, :]
    K = @view states[2:2, :]
    rho = @view states[3:3, :]
    savings, st_new = _policy_savings_bounded(model, ps, st, states; transform)
    Y = z .* K .^ params.alpha
    K_next = (1 - params.delta) .* K .+ savings .* Y
    C = max.((1 .- savings) .* Y, params.eps_safe)

    expectation_terms = [begin
        z_next = exp.(rho .* log.(z) .+ params.sigma_z .* node)
        features_next = vcat(z_next, K_next, rho)
        savings_next, _ = _policy_savings_bounded(model, ps, st_new, features_next; transform)
        Y_next = z_next .* K_next .^ params.alpha
        C_next = max.((1 .- savings_next) .* Y_next, params.eps_safe)
        R_next = 1 - params.delta .+ params.alpha .* z_next .* K_next .^ (params.alpha - 1)
        weight .* R_next ./ C_next
    end for (node, weight) in zip(rule.nodes, rule.weights)]
    expectation = reduce(+, expectation_terms)

    residual = 1 ./ (params.beta .* C .* expectation) .- 1
    euler = mean(abs2, residual)
    anchor = anchor_weight > 0 ? _scalar_anchor_loss(model, ps, st; params, transform) : zero(euler)
    return (
        loss = euler + anchor_weight * anchor,
        euler_loss = euler,
        anchor_loss = anchor,
        residual = residual,
        savings = savings,
        consumption = C,
        next_capital = K_next,
    ), st_new
end

function smm_joint_euler_loss(model, ps, st, states::AbstractMatrix, rule::QuadratureRule;
        params::SMMBrockMirmanParams = SMMBrockMirmanParams(),
        anchor_weight::Real = 1e-3,
        transform = identity)
    size(states, 1) == 4 || throw(DimensionMismatch("joint SMM states must be 4-by-batch: z, K, beta, rho"))
    z = @view states[1:1, :]
    K = @view states[2:2, :]
    beta = @view states[3:3, :]
    rho = @view states[4:4, :]
    savings, st_new = _policy_savings_bounded(model, ps, st, states; transform)
    Y = z .* K .^ params.alpha
    K_next = (1 - params.delta) .* K .+ savings .* Y
    C = max.((1 .- savings) .* Y, params.eps_safe)

    expectation_terms = [begin
        z_next = exp.(rho .* log.(z) .+ params.sigma_z .* node)
        features_next = vcat(z_next, K_next, beta, rho)
        savings_next, _ = _policy_savings_bounded(model, ps, st_new, features_next; transform)
        Y_next = z_next .* K_next .^ params.alpha
        C_next = max.((1 .- savings_next) .* Y_next, params.eps_safe)
        R_next = 1 - params.delta .+ params.alpha .* z_next .* K_next .^ (params.alpha - 1)
        weight .* R_next ./ C_next
    end for (node, weight) in zip(rule.nodes, rule.weights)]
    expectation = reduce(+, expectation_terms)

    residual = 1 ./ (beta .* C .* expectation) .- 1
    euler = mean(abs2, residual)
    anchor = anchor_weight > 0 ? _joint_anchor_loss(model, ps, st, beta; params, transform) : zero(euler)
    return (
        loss = euler + anchor_weight * anchor,
        euler_loss = euler,
        anchor_loss = anchor,
        residual = residual,
        savings = savings,
        consumption = C,
        next_capital = K_next,
    ), st_new
end

function common_random_shocks(rng::AbstractRNG, T::Integer)
    T > 0 || throw(ArgumentError("T must be positive"))
    return randn(rng, T)
end

function simulate_smm_scalar_rho(model, ps, st, rho_values, shocks;
        params::SMMBrockMirmanParams = SMMBrockMirmanParams(),
        T_burn::Integer = 100,
        T_sim::Integer = 600,
        K0::Real = 3.0,
        transform = identity)
    rho_vec = collect(float.(rho_values))
    T_burn >= 0 || throw(ArgumentError("T_burn must be nonnegative"))
    T_sim > 1 || throw(ArgumentError("T_sim must exceed one"))
    length(shocks) >= T_burn + T_sim || throw(ArgumentError("not enough shocks for burn-in plus simulation horizon"))
    n = length(rho_vec)
    rho = reshape(rho_vec, 1, :)
    z = ones(1, n)
    K = fill(float(K0), 1, n)
    C = Matrix{Float64}(undef, T_sim, n)
    I = similar(C)
    Y = similar(C)
    st_acc = st
    row = 1
    for t in 1:(T_burn + T_sim)
        features = vcat(z, K, rho)
        savings, st_acc = _policy_savings_bounded(model, ps, st_acc, features; transform)
        output = z .* K .^ params.alpha
        investment = savings .* output
        consumption = max.((1 .- savings) .* output, params.eps_safe)
        if t > T_burn
            C[row, :] = vec(consumption)
            I[row, :] = vec(investment)
            Y[row, :] = vec(output)
            row += 1
        end
        K = (1 - params.delta) .* K .+ investment
        z = exp.(rho .* log.(z) .+ params.sigma_z .* shocks[t])
    end
    return (C = C, I = I, Y = Y, st = st_acc)
end

function simulate_smm_joint_theta(model, ps, st, theta_values::AbstractMatrix, shocks;
        params::SMMBrockMirmanParams = SMMBrockMirmanParams(),
        T_burn::Integer = 100,
        T_sim::Integer = 500,
        K0::Real = 3.0,
        transform = identity)
    size(theta_values, 1) == 2 || throw(DimensionMismatch("theta_values must be 2-by-candidates: beta, rho"))
    T_burn >= 0 || throw(ArgumentError("T_burn must be nonnegative"))
    T_sim > 1 || throw(ArgumentError("T_sim must exceed one"))
    length(shocks) >= T_burn + T_sim || throw(ArgumentError("not enough shocks for burn-in plus simulation horizon"))
    beta = reshape(theta_values[1, :], 1, :)
    rho = reshape(theta_values[2, :], 1, :)
    n = size(theta_values, 2)
    z = ones(1, n)
    K = fill(float(K0), 1, n)
    C = Matrix{Float64}(undef, T_sim, n)
    I = similar(C)
    Y = similar(C)
    st_acc = st
    row = 1
    for t in 1:(T_burn + T_sim)
        features = vcat(z, K, beta, rho)
        savings, st_acc = _policy_savings_bounded(model, ps, st_acc, features; transform)
        output = z .* K .^ params.alpha
        investment = savings .* output
        consumption = max.((1 .- savings) .* output, params.eps_safe)
        if t > T_burn
            C[row, :] = vec(consumption)
            I[row, :] = vec(investment)
            Y[row, :] = vec(output)
            row += 1
        end
        K = (1 - params.delta) .* K .+ investment
        z = exp.(rho .* log.(z) .+ params.sigma_z .* shocks[t])
    end
    return (C = C, I = I, Y = Y, st = st_acc)
end

function autocorr_columns(X::AbstractMatrix; eps_safe::Real = 1e-12)
    size(X, 1) > 1 || throw(ArgumentError("autocorrelation requires at least two rows"))
    x0 = X[1:(end - 1), :] .- mean(X[1:(end - 1), :]; dims = 1)
    x1 = X[2:end, :] .- mean(X[2:end, :]; dims = 1)
    denom = sqrt.(sum(abs2, x0; dims = 1) .* sum(abs2, x1; dims = 1))
    return vec(sum(x0 .* x1; dims = 1) ./ max.(denom, eps_safe))
end

function smm_scalar_moments(C::AbstractMatrix, I::AbstractMatrix, Y::AbstractMatrix)
    dlog_C = diff(log.(C); dims = 1)
    log_Y = log.(Y)
    return hcat(
        vec(std(dlog_C; dims = 1, corrected = false)),
        autocorr_columns(dlog_C),
        autocorr_columns(log_Y),
        vec(mean(I ./ Y; dims = 1)),
    )
end

function smm_joint_moments(C::AbstractMatrix, I::AbstractMatrix, Y::AbstractMatrix)
    dlog_C = diff(log.(C); dims = 1)
    log_Y = log.(Y)
    return hcat(
        vec(mean(I ./ Y; dims = 1)),
        vec(std(dlog_C; dims = 1, corrected = false)),
        autocorr_columns(dlog_C),
        autocorr_columns(log_Y),
    )
end

function smm_criterion(moments::AbstractMatrix, target; mask = trues(size(moments, 2)), weights = nothing)
    mask_vec = collect(mask)
    length(mask_vec) == size(moments, 2) || throw(DimensionMismatch("mask length must match moment count"))
    diff = moments[:, mask_vec] .- reshape(collect(target)[mask_vec], 1, :)
    if weights === nothing
        return vec(sum(abs2, diff; dims = 2))
    end
    W = weights[mask_vec, mask_vec]
    return [dot(diff[i, :], W * diff[i, :]) for i in axes(diff, 1)]
end

function smm_grid_estimate(candidates, moments::AbstractMatrix, target; mask = trues(size(moments, 2)), weights = nothing)
    Q = smm_criterion(moments, target; mask, weights)
    idx = argmin(Q)
    theta = candidates isa AbstractMatrix ? candidates[:, idx] : candidates[idx]
    return (index = idx, theta = theta, criterion = Q, value = Q[idx], fitted_moments = moments[idx, :])
end

function smm_moment_sensitivity_1d(moments::AbstractMatrix, grid, idx::Integer)
    1 < idx < length(grid) || throw(ArgumentError("idx must have neighbors for central differences"))
    step = grid[idx + 1] - grid[idx - 1]
    return (moments[idx + 1, :] .- moments[idx - 1, :]) ./ step
end

function smm_moment_jacobian_2d(moments::Array{<:Real,3}, beta_grid, rho_grid, idx_beta::Integer, idx_rho::Integer)
    1 < idx_beta < length(beta_grid) || throw(ArgumentError("idx_beta must have neighbors for central differences"))
    1 < idx_rho < length(rho_grid) || throw(ArgumentError("idx_rho must have neighbors for central differences"))
    dbeta = beta_grid[idx_beta + 1] - beta_grid[idx_beta - 1]
    drho = rho_grid[idx_rho + 1] - rho_grid[idx_rho - 1]
    dm_dbeta = (moments[idx_rho, idx_beta + 1, :] .- moments[idx_rho, idx_beta - 1, :]) ./ dbeta
    dm_drho = (moments[idx_rho + 1, idx_beta, :] .- moments[idx_rho - 1, idx_beta, :]) ./ drho
    return hcat(vec(dm_dbeta), vec(dm_drho))
end

function smm_identification_svd(jacobian::AbstractMatrix; mask = trues(size(jacobian, 1)))
    sub = jacobian[collect(mask), :]
    decomp = svd(sub)
    return (singular_values = decomp.S, right_vectors = decomp.V, weak_direction = decomp.V[:, end])
end
