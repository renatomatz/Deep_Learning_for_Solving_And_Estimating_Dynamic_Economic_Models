export scalar_model_value,
    second_derivative,
    tanh_mlp_scalar_derivatives,
    zero_bc_ode_residual,
    zero_bc_ode_loss,
    tanh_mlp_value_second_derivative,
    zero_bc_tanh_mlp_loss,
    analytic_zero_bc_solution

using ForwardDiff
using LinearAlgebra: I
using Statistics: mean

function scalar_model_value(model, ps, st, x::Real)
    y, _ = model(reshape([x], 1, 1), ps, st)
    return y[1]
end

function second_derivative(f, x::Real)
    return ForwardDiff.derivative(t -> ForwardDiff.derivative(f, t), x)
end

function zero_bc_ode_residual(model, ps, st, x_points)
    f = x -> scalar_model_value(model, ps, st, x)
    residuals = [second_derivative(f, x) + one(x) for x in vec(x_points)]
    return reshape(residuals, 1, :)
end

function zero_bc_ode_loss(model, ps, st, x_points)
    residual = zero_bc_ode_residual(model, ps, st, x_points)
    pde_loss = mean(abs2, residual)
    y0 = scalar_model_value(model, ps, st, 0.0)
    y1 = scalar_model_value(model, ps, st, 1.0)
    bc_loss = abs2(y0) + abs2(y1)
    return (
        loss = pde_loss + bc_loss,
        pde_loss = pde_loss,
        bc_loss = bc_loss,
        residual = residual,
        boundary = (y0 = y0, y1 = y1),
    ), st
end

function _layer_names(ps::NamedTuple)
    return keys(ps)
end

function _assert_scalar_tanh_mlp_parameters(ps::NamedTuple)
    names = Tuple(_layer_names(ps))
    length(names) >= 2 || throw(ArgumentError("expected at least one hidden layer and one output layer"))
    first_layer = getproperty(ps, names[1])
    last_layer = getproperty(ps, names[end])
    size(first_layer.weight, 2) == 1 ||
        throw(ArgumentError("zero_bc_tanh_mlp_loss supports one input feature"))
    size(last_layer.weight, 1) == 1 ||
        throw(ArgumentError("zero_bc_tanh_mlp_loss supports one scalar output"))
    all(name -> hasproperty(getproperty(ps, name), :weight) && hasproperty(getproperty(ps, name), :bias), names) ||
        throw(ArgumentError("zero_bc_tanh_mlp_loss expects Dense-like weight/bias layers"))
    return names
end

function _assert_tanh_mlp_parameters(ps::NamedTuple, input_dim::Integer)
    names = Tuple(_layer_names(ps))
    length(names) >= 2 || throw(ArgumentError("expected at least one hidden layer and one output layer"))
    first_layer = getproperty(ps, names[1])
    last_layer = getproperty(ps, names[end])
    size(first_layer.weight, 2) == input_dim ||
        throw(ArgumentError("tanh_mlp_scalar_derivatives expected $input_dim input features"))
    size(last_layer.weight, 1) == 1 ||
        throw(ArgumentError("tanh_mlp_scalar_derivatives supports one scalar output"))
    all(name -> hasproperty(getproperty(ps, name), :weight) && hasproperty(getproperty(ps, name), :bias), names) ||
        throw(ArgumentError("tanh_mlp_scalar_derivatives expects Dense-like weight/bias layers"))
    return names
end

function _weighted_hessian_sum(weight, H, row::Integer, input_dim::Integer)
    return [sum(weight[row, col] * H[col, i, j] for col in axes(weight, 2)) for i in 1:input_dim, j in 1:input_dim]
end

"""
    tanh_mlp_scalar_derivatives(ps, x)

Return `(value, gradient, hessian)` for the scalar-output tanh MLP created by
`make_mlp(input_dim, hidden, 1; activation = NNlib.tanh)`. The formulas are
explicitly propagated through Dense layers so Zygote can still differentiate the
PINN residual with respect to Lux parameters.
"""
function tanh_mlp_scalar_derivatives(ps::NamedTuple, x::AbstractVector)
    input = collect(x)
    input_dim = length(input)
    input_dim > 0 || throw(ArgumentError("x must be nonempty"))
    names = _assert_tanh_mlp_parameters(ps, input_dim)

    a = input
    J = Matrix{eltype(input)}(I, input_dim, input_dim)
    H = zeros(eltype(input), input_dim, input_dim, input_dim)

    for (layer_index, name) in enumerate(names)
        layer = getproperty(ps, name)
        z = layer.weight * a .+ layer.bias
        Jz = layer.weight * J
        Hz = [_weighted_hessian_sum(layer.weight, H, row, input_dim)[i, j]
              for row in axes(layer.weight, 1), i in 1:input_dim, j in 1:input_dim]

        if layer_index == length(names)
            a, J, H = z, Jz, Hz
        else
            activated = tanh.(z)
            slope = 1 .- activated .^ 2
            curvature = -2 .* activated .* slope
            Hnext = [
                slope[row] * Hz[row, i, j] + curvature[row] * Jz[row, i] * Jz[row, j]
                for row in axes(z, 1), i in 1:input_dim, j in 1:input_dim
            ]
            a = activated
            J = slope .* Jz
            H = Hnext
        end
    end

    return only(a), vec(J), dropdims(H; dims = 1)
end

"""
    tanh_mlp_value_second_derivative(ps, x)

Value and second input derivative for the scalar `make_mlp(1, hidden, 1;
activation = NNlib.tanh, final_activation = identity)` architecture used in the
Wave 1 ODE PINN notebook. This is intentionally not a generic Lux derivative
adapter; it is a parameter-differentiable teaching helper for tanh MLPs.
"""
function tanh_mlp_value_second_derivative(ps::NamedTuple, x::Real)
    value, _, hessian = tanh_mlp_scalar_derivatives(ps, [x])
    return value, hessian[1, 1]
end

"""
    zero_bc_tanh_mlp_loss(model, ps, st, x_points)

Trainable loss for the first ODE PINN notebook. It follows the Lux training
signature but is fenced to scalar tanh MLPs so Zygote differentiates the PDE
residual with respect to parameters. Use `zero_bc_ode_loss` for generic
ForwardDiff-based diagnostics outside parameter training.
"""
function zero_bc_tanh_mlp_loss(model, ps, st, x_points)
    _assert_scalar_tanh_mlp_parameters(ps)
    residual = reshape([tanh_mlp_value_second_derivative(ps, x)[2] + one(x) for x in vec(x_points)], 1, :)
    pde_loss = mean(abs2, residual)
    y0, _ = tanh_mlp_value_second_derivative(ps, 0.0)
    y1, _ = tanh_mlp_value_second_derivative(ps, 1.0)
    bc_loss = abs2(y0) + abs2(y1)
    return (
        loss = pde_loss + bc_loss,
        pde_loss = pde_loss,
        bc_loss = bc_loss,
        residual = residual,
        boundary = (y0 = y0, y1 = y1),
    ), st
end

analytic_zero_bc_solution(x) = x .* (1 .- x) ./ 2
