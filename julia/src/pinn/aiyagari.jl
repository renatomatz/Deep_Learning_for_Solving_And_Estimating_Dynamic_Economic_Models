export CTAiyagariParams,
    ct_aiyagari_trapezoid_rule,
    ct_aiyagari_prices,
    ct_aiyagari_soft_penalty_grad,
    ct_aiyagari_kfe_drift,
    ct_aiyagari_fd_inner,
    ct_aiyagari_fd_solve,
    ct_aiyagari_normalized_density,
    ct_aiyagari_distribution_aggregates,
    ct_aiyagari_marginal_value_policy,
    ct_aiyagari_pinn_loss,
    ct_aiyagari_saving_linf

using LinearAlgebra: I
using Statistics: mean

struct CTAiyagariParams{T}
    alpha::T
    delta::T
    gamma::T
    rho::T
    z_bar::T
    lambda::Vector{T}
    labor::Vector{T}
    a_min::T
    a_max::T
    a_lb::T
    kappa::T
    n_quad::Int
    n_a::Int
    eps_safe::T
    eps_gate::T
end

function CTAiyagariParams(; alpha = 1 / 3, delta = 0.1, gamma = 2.1, rho = 0.05,
        z_bar = 0.0, lambda = [0.4, 0.4], labor = [0.3, 1.7], a_min = 1e-6,
        a_max = 20.0, a_lb = 1.0, kappa = 3.0, n_quad::Integer = 80,
        n_a::Integer = 80, eps_safe = 1e-8, eps_gate = 1e-2)
    a_max > a_min || throw(ArgumentError("a_max must exceed a_min"))
    length(lambda) == 2 || throw(ArgumentError("lambda must have two entries"))
    length(labor) == 2 || throw(ArgumentError("labor must have two entries"))
    n_quad >= 3 || throw(ArgumentError("n_quad must be at least 3"))
    n_a >= 3 || throw(ArgumentError("n_a must be at least 3"))
    eps_safe > 0 || throw(ArgumentError("eps_safe must be positive"))
    vals = promote(alpha, delta, gamma, rho, z_bar, a_min, a_max, a_lb, kappa, eps_safe, eps_gate)
    T = typeof(vals[1])
    return CTAiyagariParams(vals[1], vals[2], vals[3], vals[4], vals[5], T.(lambda), T.(labor), vals[6], vals[7], vals[8], vals[9], Int(n_quad), Int(n_a), vals[10], vals[11])
end

function ct_aiyagari_trapezoid_rule(params::CTAiyagariParams = CTAiyagariParams(); n::Integer = params.n_quad)
    n >= 3 || throw(ArgumentError("n must be at least 3"))
    a = collect(range(params.a_min, params.a_max; length = n))
    da = a[2] - a[1]
    weights = [i == 1 || i == n ? da / 2 : da for i in 1:n]
    return (nodes = a, weights = weights, da = da)
end

function ct_aiyagari_prices(K, L; params::CTAiyagariParams = CTAiyagariParams())
    Ksafe = max(K, params.eps_safe)
    Lsafe = max(L, params.eps_safe)
    ratio = Ksafe / Lsafe
    z = exp(params.z_bar)
    r = params.alpha * z * ratio^(params.alpha - 1) - params.delta
    w = (1 - params.alpha) * z * ratio^params.alpha
    return (r = r, w = w)
end

ct_aiyagari_soft_penalty_grad(a; params::CTAiyagariParams = CTAiyagariParams()) =
    a <= params.a_lb ? -params.kappa * (a - params.a_lb) : zero(a)

function ct_aiyagari_kfe_drift(m, s, da; params::CTAiyagariParams = CTAiyagariParams())
    size(m, 2) == 2 || throw(DimensionMismatch("m must be grid-by-2 labor states"))
    size(s) == size(m) || throw(DimensionMismatch("s must match m"))
    sp = max.(s, zero(eltype(s)))
    sn = min.(s, zero(eltype(s)))
    zero_row = zeros(eltype(m), 1, 2)
    F = sp .* m .+ vcat((sn .* m)[2:end, :], zero_row)
    F = vcat(F[1:end-1, :], zero_row)
    Da = (F .- vcat(zero_row, F[1:end-1, :])) ./ da
    lam = reshape(params.lambda, 1, 2)
    return -Da .+ reverse(lam; dims = 2) .* reverse(m; dims = 2) .- lam .* m
end

function _ct_utility(c, params::CTAiyagariParams)
    params.gamma == 1 && return log(max(c, params.eps_safe))
    return (max(c, params.eps_safe)^(1 - params.gamma) - 1) / (1 - params.gamma)
end

function ct_aiyagari_fd_inner(params::CTAiyagariParams, a, w, r; max_iter::Integer = 200, tol::Real = 1e-8, dt::Real = 1000.0)
    n = length(a)
    daf = vcat(diff(a), a[end] - a[end - 1])
    dab = vcat(a[2] - a[1], diff(a))
    cash = w .* reshape(params.labor, 1, 2) .+ r .* reshape(a, :, 1)
    psi = [ai <= params.a_lb ? -0.5 * params.kappa * (ai - params.a_lb)^2 : zero(ai) for ai in a, _ in 1:2]
    V = _ct_utility.(max.(0.5 .* cash, params.eps_safe), Ref(params)) ./ params.rho
    A = zeros(eltype(a), 2n, 2n)
    c = copy(cash)
    s = zero(cash)

    for _ in 1:max_iter
        dV_F = vcat((V[2:end, :] .- V[1:end-1, :]) ./ reshape(daf[1:end-1], :, 1), zeros(eltype(V), 1, 2))
        dV_B = vcat(zeros(eltype(V), 1, 2), (V[2:end, :] .- V[1:end-1, :]) ./ reshape(dab[2:end], :, 1))
        dV_B[1, :] .= max.(cash[1, :], params.eps_safe) .^ (-params.gamma)
        dV_F[end, :] .= max.(cash[end, :], params.eps_safe) .^ (-params.gamma)
        c_F = max.(dV_F, params.eps_safe) .^ (-1 / params.gamma)
        c_B = max.(dV_B, params.eps_safe) .^ (-1 / params.gamma)
        s_F = cash .- c_F
        s_B = cash .- c_B
        c = ifelse.(s_F .> 0, c_F, ifelse.(s_B .< 0, c_B, cash))
        s = ifelse.(s_F .> 0, s_F, ifelse.(s_B .< 0, s_B, zero(cash)))
        sp = max.(s, zero(eltype(s)))
        sn = min.(s, zero(eltype(s)))
        fill!(A, zero(eltype(A)))
        for j in 1:2
            other = 3 - j
            for i in 1:n
                row = i + (j - 1) * n
                A[row, row] = -sp[i, j] / daf[i] + sn[i, j] / dab[i] - params.lambda[j]
                i < n && (A[row, row + 1] = sp[i, j] / daf[i])
                i > 1 && (A[row, row - 1] = -sn[i, j] / dab[i])
                A[row, i + (other - 1) * n] = params.lambda[j]
            end
        end
        rhs = vec(_ct_utility.(c, Ref(params)) .+ psi .+ V ./ dt)
        Vn = reshape(((params.rho + 1 / dt) .* Matrix{eltype(A)}(I, 2n, 2n) .- A) \ rhs, n, 2)
        maximum(abs.(Vn .- V)) < tol && (V = Vn; break)
        V = Vn
    end

    M = copy(transpose(A))
    M[end, :] .= one(eltype(M))
    b = zeros(eltype(M), 2n)
    b[end] = one(eltype(M))
    g = reshape(M \ b, n, 2)
    g = max.(g, zero(eltype(g)))
    g ./= sum(g)
    return (V = V, g = g, c = c, s = s, A = A)
end

function ct_aiyagari_fd_solve(params::CTAiyagariParams = CTAiyagariParams(); max_outer::Integer = 20, inner_max_iter::Integer = 120, tol::Real = 1e-4)
    a = collect(range(params.a_min, params.a_max; length = params.n_a))
    L = mean(params.labor)
    r_lo = -0.5 * params.delta
    r_hi = 0.999 * params.rho
    inner = nothing
    r = zero(eltype(a))
    K_demand = zero(eltype(a))
    K_supply = zero(eltype(a))
    market_gap = Inf
    converged = false
    outer_iterations = 0
    for outer in 1:max_outer
        outer_iterations = outer
        r = 0.5 * (r_lo + r_hi)
        K_demand = ((r + params.delta) / params.alpha)^(1 / (params.alpha - 1)) * L
        w = (1 - params.alpha) * K_demand^params.alpha * L^(-params.alpha)
        inner = ct_aiyagari_fd_inner(params, a, w, r; max_iter = inner_max_iter)
        K_supply = sum(a .* vec(sum(inner.g; dims = 2)))
        market_gap = K_supply - K_demand
        if abs(market_gap) < tol
            converged = true
            break
        end
        if K_supply > K_demand
            r_hi = r
        else
            r_lo = r
        end
    end
    prices = ct_aiyagari_prices(K_supply, L; params)
    return (
        a = a,
        V = inner.V,
        g = inner.g,
        c = inner.c,
        s = inner.s,
        K = K_supply,
        L = L,
        r = prices.r,
        w = prices.w,
        K_supply = K_supply,
        K_demand = K_demand,
        market_gap = market_gap,
        converged = converged,
        outer_iterations = outer_iterations,
    )
end

function _logsumexp(x)
    m = maximum(x)
    return m + log(sum(exp.(x .- m)))
end

function ct_aiyagari_normalized_density(log_density, weights)
    size(log_density, 1) == length(weights) || throw(DimensionMismatch("weights must match density grid"))
    weighted_log = log_density .+ log.(reshape(weights, :, 1))
    logZ = _logsumexp(vec(weighted_log))
    density = exp.(log_density .- logZ)
    mass = reshape(weights, :, 1) .* density
    return (density = density, mass = mass, logZ = logZ)
end

function ct_aiyagari_distribution_aggregates(density, a, weights; params::CTAiyagariParams = CTAiyagariParams())
    size(density) == (length(a), 2) || throw(DimensionMismatch("density must be grid-by-2"))
    mass = reshape(weights, :, 1) .* density
    K = sum(reshape(a, :, 1) .* mass)
    L = sum(reshape(params.labor, 1, 2) .* mass)
    return (K = K, L = L, mass = sum(mass), mass_by_labor = vec(sum(mass; dims = 1)))
end

function _ct_aiyagari_features(a, j, params::CTAiyagariParams)
    an = 2 * (a - params.a_min) / (params.a_max - params.a_min) - 1
    return j == 1 ? [an, one(an), zero(an)] : [an, zero(an), one(an)]
end

function _ct_aiyagari_both_raw_derivative(ps::NamedTuple, a_vec; params::CTAiyagariParams = CTAiyagariParams())
    scale = 2 / (params.a_max - params.a_min)
    raw = [tanh_mlp_scalar_derivatives(ps, _ct_aiyagari_features(a, j, params))[1] for a in a_vec, j in 1:2]
    d_raw = [tanh_mlp_scalar_derivatives(ps, _ct_aiyagari_features(a, j, params))[2][1] * scale for a in a_vec, j in 1:2]
    return raw, d_raw
end

function _ct_aiyagari_both_marginal_value(ps::NamedTuple, a_vec; params::CTAiyagariParams = CTAiyagariParams())
    raw, d_raw = _ct_aiyagari_both_raw_derivative(ps, a_vec; params)
    W = NNlib.softplus.(raw) .+ params.eps_safe
    dW = NNlib.sigmoid.(raw) .* d_raw
    return W, dW
end

function _ct_aiyagari_both_log_density(ps::NamedTuple, a_vec; params::CTAiyagariParams = CTAiyagariParams())
    return _ct_aiyagari_both_raw_derivative(ps, a_vec; params)
end

function ct_aiyagari_marginal_value_policy(W, a, r, w; params::CTAiyagariParams = CTAiyagariParams())
    c = max.(W, params.eps_safe) .^ (-1 / params.gamma)
    s = w .* reshape(params.labor, 1, 2) .+ r .* reshape(a, :, 1) .- c
    return (consumption = c, savings = s)
end

function ct_aiyagari_pinn_loss(models, ps, st, a_col; params::CTAiyagariParams = CTAiyagariParams(), kfe_form = :fv, agg_weight::Real = 1)
    _assert_tanh_mlp_parameters(ps.w, 3)
    _assert_tanh_mlp_parameters(ps.g, 3)
    form = Symbol(kfe_form)
    form in (:fv, :strong) || throw(ArgumentError("kfe_form must be :fv or :strong"))
    a = vec(a_col)
    quad = Zygote.ignore() do
        ct_aiyagari_trapezoid_rule(params)
    end
    log_g_q, _ = _ct_aiyagari_both_log_density(ps.g, quad.nodes; params)
    normalized = ct_aiyagari_normalized_density(log_g_q, quad.weights)
    aggregates = ct_aiyagari_distribution_aggregates(normalized.density, quad.nodes, quad.weights; params)
    live_prices = ct_aiyagari_prices(aggregates.K, aggregates.L; params)
    prices = (r = Zygote.dropgrad(live_prices.r), w = Zygote.dropgrad(live_prices.w))

    W, dW = _ct_aiyagari_both_marginal_value(ps.w, a; params)
    policy = ct_aiyagari_marginal_value_policy(W, a, prices.r, prices.w; params)
    lam = reshape(params.lambda, 1, 2)
    hjb = (params.rho - prices.r .+ lam) .* W .- policy.savings .* dW .- lam .* reverse(W; dims = 2) .-
        reshape([ct_aiyagari_soft_penalty_grad(ai; params) for ai in a], :, 1)
    hjb_loss = mean(abs2, hjb)
    shape_loss = mean(abs2, max.(dW, zero(eltype(dW))))

    W_q, dW_q = _ct_aiyagari_both_marginal_value(ps.w, quad.nodes; params)
    policy_q = ct_aiyagari_marginal_value_policy(W_q, quad.nodes, prices.r, prices.w; params)
    s_q = Zygote.dropgrad(policy_q.savings)
    agg_loss = sum(reshape(quad.weights, :, 1) .* s_q .* normalized.density)^2

    if form == :fv
        mu = ct_aiyagari_kfe_drift(normalized.density .* quad.da, s_q, quad.da; params)
        kfe = mu ./ quad.da
        kfe_loss = mean(abs2, kfe)
        flux_loss = zero(kfe_loss)
    else
        log_g, dlog_g = _ct_aiyagari_both_log_density(ps.g, a; params)
        g = exp.(log_g .- normalized.logZ)
        s_c = Zygote.dropgrad(policy.savings)
        dsa = Zygote.dropgrad(prices.r .+ (1 / params.gamma) .* (policy.consumption ./ W) .* dW)
        kfe = -(s_c .* g .* dlog_g .+ g .* dsa) .+ reverse(lam; dims = 2) .* reverse(g; dims = 2) .- lam .* g
        kfe_loss = mean(abs2, kfe)
        flux_loss = mean(abs2, sum(s_c .* g; dims = 2))
    end

    mass_balance_loss = zero(hjb_loss)
    boundary_loss = zero(hjb_loss)
    if form == :strong
        labor_mass = vec(sum(normalized.mass; dims = 1))
        mass_balance_loss = (params.lambda[1] * labor_mass[1] - params.lambda[2] * labor_mass[2])^2
        endpoints = [params.a_min, params.a_max]
        W_end, _ = _ct_aiyagari_both_marginal_value(ps.w, endpoints; params)
        policy_end = ct_aiyagari_marginal_value_policy(W_end, endpoints, prices.r, prices.w; params)
        log_g_end, _ = _ct_aiyagari_both_log_density(ps.g, endpoints; params)
        density_end = exp.(log_g_end .- normalized.logZ)
        boundary_loss = sum(abs2, Zygote.dropgrad(policy_end.savings) .* density_end)
    end

    total = hjb_loss + kfe_loss + flux_loss + agg_weight * agg_loss + mass_balance_loss + boundary_loss + shape_loss
    return (
        loss = total,
        hjb_loss = hjb_loss,
        kfe_loss = kfe_loss,
        flux_loss = flux_loss,
        agg_loss = agg_loss,
        mass_balance_loss = mass_balance_loss,
        boundary_loss = boundary_loss,
        shape_loss = shape_loss,
        hjb = hjb,
        K = aggregates.K,
        L = aggregates.L,
        r = prices.r,
        w = prices.w,
        mass = aggregates.mass,
    ), st
end

function ct_aiyagari_saving_linf(ps_w::NamedTuple, a, s_ref, r, w; params::CTAiyagariParams = CTAiyagariParams())
    W, _ = _ct_aiyagari_both_marginal_value(ps_w, a; params)
    policy = ct_aiyagari_marginal_value_policy(W, a, r, w; params)
    size(policy.savings) == size(s_ref) || throw(DimensionMismatch("reference savings must match policy grid"))
    return maximum(abs.(policy.savings .- s_ref))
end
