export CakeEatingParams,
    cake_eating_kappa,
    cake_eating_value_exact,
    cake_eating_consumption_exact,
    cake_eating_trial_value_derivative,
    cake_eating_hjb_residual,
    cake_eating_hjb_loss

using Statistics: mean

struct CakeEatingParams{T}
    gamma::T
    rho::T
    r::T
    a_min::T
    a_max::T
    eps_safe::T
end

function CakeEatingParams(; gamma = 2.0, rho = 0.05, r = 0.03, a_min = 0.1, a_max = 4.0, eps_safe = 1e-6)
    a_max > a_min || throw(ArgumentError("a_max must exceed a_min"))
    eps_safe > 0 || throw(ArgumentError("eps_safe must be positive"))
    vals = promote(gamma, rho, r, a_min, a_max, eps_safe)
    return CakeEatingParams(vals...)
end

cake_eating_kappa(params::CakeEatingParams = CakeEatingParams()) =
    (params.rho - (1 - params.gamma) * params.r) / params.gamma

function _crra_utility(c, gamma)
    gamma == 1 && return log(c)
    return c^(1 - gamma) / (1 - gamma)
end

function cake_eating_value_exact(a; params::CakeEatingParams = CakeEatingParams())
    kappa = cake_eating_kappa(params)
    return kappa^(-params.gamma) / (1 - params.gamma) * a^(1 - params.gamma)
end

cake_eating_consumption_exact(a; params::CakeEatingParams = CakeEatingParams()) = cake_eating_kappa(params) * a

function cake_eating_trial_value_derivative(ps::NamedTuple, a::Real; params::CakeEatingParams = CakeEatingParams())
    span = params.a_max - params.a_min
    x = (a - params.a_min) / span
    z = 2x - 1
    raw, grad_raw, _ = tanh_mlp_scalar_derivatives(ps, [z])
    dz_da = 2 / span

    V_lo = cake_eating_value_exact(params.a_min; params)
    V_hi = cake_eating_value_exact(params.a_max; params)
    V_scale = abs(V_lo)
    linear = V_lo + x * (V_hi - V_lo)
    bubble = x * (1 - x)

    value = linear + V_scale * bubble * raw
    derivative = (V_hi - V_lo) / span + V_scale * ((1 - 2x) / span * raw + bubble * grad_raw[1] * dz_da)
    return value, derivative
end

function cake_eating_hjb_residual(ps::NamedTuple, a::Real; params::CakeEatingParams = CakeEatingParams())
    V, Va = cake_eating_trial_value_derivative(ps, a; params)
    safe_Va = NNlib.softplus(Va) + params.eps_safe
    c = safe_Va^(-1 / params.gamma)
    return params.rho * V - (_crra_utility(c, params.gamma) + safe_Va * (params.r * a - c))
end

function cake_eating_hjb_loss(model, ps, st, a_points; params::CakeEatingParams = CakeEatingParams())
    _assert_tanh_mlp_parameters(ps, 1)
    a = vec(a_points)
    residual = [cake_eating_hjb_residual(ps, ai; params) for ai in a]
    loss = mean(abs2, residual)
    values = [cake_eating_trial_value_derivative(ps, ai; params)[1] for ai in a]
    exact = [cake_eating_value_exact(ai; params) for ai in a]
    return (
        loss = loss,
        hjb_loss = loss,
        residual = reshape(residual, 1, :),
        value = reshape(values, 1, :),
        value_error = reshape(values .- exact, 1, :),
    ), st
end
