export BlackScholesParams,
    standard_normal_cdf,
    black_scholes_call_price,
    black_scholes_delta,
    black_scholes_value_derivatives,
    black_scholes_residual,
    black_scholes_loss

using Statistics: mean

struct BlackScholesParams{T}
    r::T
    sigma::T
    strike::T
    maturity::T
    s_max::T
    eps_safe::T
end

function BlackScholesParams(; r = 0.05, sigma = 0.2, strike = 50.0, maturity = 1.0, s_max = 100.0, eps_safe = 1e-12)
    sigma > 0 || throw(ArgumentError("sigma must be positive"))
    strike > 0 || throw(ArgumentError("strike must be positive"))
    maturity > 0 || throw(ArgumentError("maturity must be positive"))
    s_max > strike || throw(ArgumentError("s_max should exceed strike for the teaching domain"))
    eps_safe > 0 || throw(ArgumentError("eps_safe must be positive"))
    vals = promote(r, sigma, strike, maturity, s_max, eps_safe)
    return BlackScholesParams(vals...)
end

function standard_normal_cdf(x::Real)
    z = abs(x)
    t = inv(1 + 0.2316419 * z)
    poly = (((((1.330274429 * t - 1.821255978) * t) + 1.781477937) * t - 0.356563782) * t + 0.319381530) * t
    tail = inv(sqrt(2pi)) * exp(-0.5 * z^2) * poly
    cdf = 1 - tail
    return x >= 0 ? cdf : 1 - cdf
end

function black_scholes_call_price(S, tau; params::BlackScholesParams = BlackScholesParams())
    tau <= params.eps_safe && return max(S - params.strike, zero(S))
    S <= params.eps_safe && return zero(promote(S, tau)[1])
    d1 = (log(S / params.strike) + (params.r + 0.5 * params.sigma^2) * tau) / (params.sigma * sqrt(tau))
    d2 = d1 - params.sigma * sqrt(tau)
    return S * standard_normal_cdf(d1) - params.strike * exp(-params.r * tau) * standard_normal_cdf(d2)
end

function black_scholes_delta(S, tau; params::BlackScholesParams = BlackScholesParams())
    tau <= params.eps_safe && return S > params.strike ? one(S) : zero(S)
    S <= params.eps_safe && return zero(promote(S, tau)[1])
    d1 = (log(S / params.strike) + (params.r + 0.5 * params.sigma^2) * tau) / (params.sigma * sqrt(tau))
    return standard_normal_cdf(d1)
end

function black_scholes_value_derivatives(ps::NamedTuple, S::Real, t::Real; params::BlackScholesParams = BlackScholesParams())
    x = 2S / params.s_max - 1
    τ = 2t / params.maturity - 1
    raw, grad_raw, hess_raw = tanh_mlp_scalar_derivatives(ps, [x, τ])
    dS = 2 / params.s_max
    dt = 2 / params.maturity
    scale = params.strike
    return (
        value = scale * raw,
        dS = scale * grad_raw[1] * dS,
        dt = scale * grad_raw[2] * dt,
        dSS = scale * hess_raw[1, 1] * dS^2,
    )
end

function black_scholes_residual(ps::NamedTuple, S::Real, t::Real; params::BlackScholesParams = BlackScholesParams())
    d = black_scholes_value_derivatives(ps, S, t; params)
    return d.dt + 0.5 * params.sigma^2 * S^2 * d.dSS + params.r * S * d.dS - params.r * d.value
end

function black_scholes_loss(model, ps, st, batch; params::BlackScholesParams = BlackScholesParams(), terminal_weight::Real = 10)
    _assert_tanh_mlp_parameters(ps, 2)
    terminal_weight >= 0 || throw(ArgumentError("terminal_weight must be nonnegative"))
    S_int, t_int = vec(batch.S_int), vec(batch.t_int)
    S_bc0, t_bc0 = vec(batch.S_bc0), vec(batch.t_bc0)
    S_term, t_term = vec(batch.S_term), vec(batch.t_term)
    S_bcmax, t_bcmax = vec(batch.S_bcmax), vec(batch.t_bcmax)

    pde_residual = [black_scholes_residual(ps, S, t; params) / params.strike for (S, t) in zip(S_int, t_int)]
    bc0_residual = [black_scholes_value_derivatives(ps, S, t; params).value / params.strike for (S, t) in zip(S_bc0, t_bc0)]
    terminal_residual = [
        black_scholes_value_derivatives(ps, S, t; params).value / params.strike - max(S - params.strike, zero(S)) / params.strike
        for (S, t) in zip(S_term, t_term)
    ]
    smax_residual = [
        black_scholes_value_derivatives(ps, S, t; params).value / params.strike -
        (params.s_max - params.strike * exp(-params.r * (params.maturity - t))) / params.strike
        for (S, t) in zip(S_bcmax, t_bcmax)
    ]

    pde_loss = mean(abs2, pde_residual)
    bc0_loss = mean(abs2, bc0_residual)
    terminal_loss = mean(abs2, terminal_residual)
    smax_loss = mean(abs2, smax_residual)
    total = pde_loss + bc0_loss + terminal_weight * terminal_loss + smax_loss
    return (
        loss = total,
        pde_loss = pde_loss,
        bc0_loss = bc0_loss,
        terminal_loss = terminal_loss,
        smax_loss = smax_loss,
        residual = reshape(pde_residual, 1, :),
    ), st
end
