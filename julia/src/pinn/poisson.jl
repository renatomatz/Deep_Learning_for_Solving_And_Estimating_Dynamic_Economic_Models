export poisson2d_exact,
    poisson2d_forcing,
    poisson2d_boundary_lifting,
    poisson2d_bubble,
    poisson2d_hard_value_derivatives,
    poisson2d_soft_loss,
    poisson2d_hard_loss

using Statistics: mean

poisson2d_exact(x::Real, y::Real) = x^2 + y + sin(pi * x) * sin(pi * y)
poisson2d_forcing(x::Real, y::Real) = 2 - 2pi^2 * sin(pi * x) * sin(pi * y)
poisson2d_boundary_lifting(x::Real, y::Real) = x^2 + y
poisson2d_bubble(x::Real, y::Real) = x * (1 - x) * y * (1 - y)

_poisson_xy(mat, i) = (mat[1, i], mat[2, i])

function poisson2d_hard_value_derivatives(ps::NamedTuple, xy::AbstractVector)
    x, y = xy
    raw, grad_raw, hess_raw = tanh_mlp_scalar_derivatives(ps, [x, y])

    A = poisson2d_boundary_lifting(x, y)
    grad_A = [2x, one(y)]
    hess_A = [2one(x) zero(x); zero(x) zero(x)]

    B = poisson2d_bubble(x, y)
    grad_B = [(1 - 2x) * y * (1 - y), x * (1 - x) * (1 - 2y)]
    hess_B = [
        -2 * y * (1 - y) (1 - 2x) * (1 - 2y)
        (1 - 2x) * (1 - 2y) -2 * x * (1 - x)
    ]

    value = A + B * raw
    gradient = grad_A .+ B .* grad_raw .+ raw .* grad_B
    hessian = hess_A .+ B .* hess_raw .+ raw .* hess_B .+ grad_B * grad_raw' .+ grad_raw * grad_B'
    return value, gradient, hessian
end

function poisson2d_soft_loss(model, ps, st, interior, boundary; bc_weight::Real = 10)
    _assert_tanh_mlp_parameters(ps, 2)
    x_int = assert_feature_batch(interior, 2)
    x_bc = assert_feature_batch(boundary, 2)
    bc_weight >= 0 || throw(ArgumentError("bc_weight must be nonnegative"))

    residual = [begin
        x, y = _poisson_xy(x_int, i)
        _, _, H = tanh_mlp_scalar_derivatives(ps, [x, y])
        H[1, 1] + H[2, 2] - poisson2d_forcing(x, y)
    end for i in axes(x_int, 2)]

    boundary_residual = [begin
        x, y = _poisson_xy(x_bc, i)
        value, _, _ = tanh_mlp_scalar_derivatives(ps, [x, y])
        value - poisson2d_exact(x, y)
    end for i in axes(x_bc, 2)]

    pde_loss = mean(abs2, residual)
    bc_loss = mean(abs2, boundary_residual)
    return (
        loss = pde_loss + bc_weight * bc_loss,
        pde_loss = pde_loss,
        bc_loss = bc_loss,
        residual = reshape(residual, 1, :),
        boundary_residual = reshape(boundary_residual, 1, :),
    ), st
end

function poisson2d_hard_loss(model, ps, st, interior)
    _assert_tanh_mlp_parameters(ps, 2)
    x_int = assert_feature_batch(interior, 2)
    residual = [begin
        x, y = _poisson_xy(x_int, i)
        _, _, H = poisson2d_hard_value_derivatives(ps, [x, y])
        H[1, 1] + H[2, 2] - poisson2d_forcing(x, y)
    end for i in axes(x_int, 2)]
    pde_loss = mean(abs2, residual)
    return (
        loss = pde_loss,
        pde_loss = pde_loss,
        residual = reshape(residual, 1, :),
    ), st
end
