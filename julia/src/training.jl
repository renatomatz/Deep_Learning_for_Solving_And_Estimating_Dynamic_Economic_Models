export TrainState,
    setup_training,
    loss_value,
    finite_loss,
    tree_sum_abs2,
    clip_gradient_norm,
    train_step!,
    make_dataloader,
    append_metric!

using MLUtils
using Optimisers
using Random: AbstractRNG
using Zygote

mutable struct TrainState{M,P,S,O}
    model::M
    ps::P
    st::S
    opt_state::O
    step::Int
end

function setup_training(rng, model, optimiser; parameter_type::Union{Nothing,Type{<:AbstractFloat}} = nothing)
    ps, st = setup_model(rng, model; parameter_type)
    opt_state = Optimisers.setup(optimiser, ps)
    return TrainState(model, ps, st, opt_state, 0)
end

function setup_training(model, ps, st, optimiser)
    opt_state = Optimisers.setup(optimiser, ps)
    return TrainState(model, ps, st, opt_state, 0)
end

function loss_value(state::TrainState, loss_fn)
    loss, _ = loss_fn(state.model, state.ps, state.st)
    return loss
end

function loss_value(state::TrainState, loss_fn, batch)
    loss, _ = loss_fn(state.model, state.ps, state.st, batch)
    return loss
end

finite_loss(loss::Real) = isfinite(loss)
finite_loss(losses) = all(isfinite, losses)

tree_sum_abs2(::Nothing) = 0.0
tree_sum_abs2(x::Number) = abs2(x)
tree_sum_abs2(x::AbstractArray) = sum(abs2, x)
tree_sum_abs2(x::NamedTuple) = sum(tree_sum_abs2, values(x); init = 0.0)
tree_sum_abs2(x::Tuple) = sum(tree_sum_abs2, x; init = 0.0)

_tree_scale(::Nothing, factor) = nothing
_tree_scale(x::Number, factor) = x * factor
_tree_scale(x::AbstractArray, factor) = x .* factor
_tree_scale(x::NamedTuple, factor) = NamedTuple{keys(x)}(map(v -> _tree_scale(v, factor), values(x)))
_tree_scale(x::Tuple, factor) = map(v -> _tree_scale(v, factor), x)

function clip_gradient_norm(grads, max_norm::Real)
    max_norm > 0 || throw(ArgumentError("max_norm must be positive"))
    grad_norm = sqrt(tree_sum_abs2(grads))
    factor = grad_norm > max_norm ? max_norm / (grad_norm + eps(float(grad_norm))) : one(grad_norm)
    return _tree_scale(grads, factor), grad_norm
end

function train_step!(state::TrainState, loss_fn; max_grad_norm = Inf)
    (loss, st_new), back = Zygote.pullback(state.ps) do ps
        return loss_fn(state.model, ps, state.st)
    end
    finite_loss(loss) || throw(DomainError(loss, "loss is not finite"))

    grads = only(back((one(loss), nothing)))
    if isfinite(max_grad_norm)
        grads, grad_norm = clip_gradient_norm(grads, max_grad_norm)
    else
        grad_norm = sqrt(tree_sum_abs2(grads))
    end

    state.opt_state, state.ps = Optimisers.update(state.opt_state, state.ps, grads)
    state.st = st_new
    state.step += 1
    return (loss = loss, grad_norm = grad_norm, step = state.step)
end

function train_step!(state::TrainState, loss_fn, batch; max_grad_norm = Inf)
    return train_step!(state, (model, ps, st) -> loss_fn(model, ps, st, batch); max_grad_norm)
end

function make_dataloader(data; batchsize::Integer, shuffle::Bool = false, partial::Bool = true, rng::Union{Nothing,AbstractRNG} = nothing)
    batchsize > 0 || throw(ArgumentError("batchsize must be positive"))
    if shuffle && rng === nothing
        throw(ArgumentError("shuffle=true requires an explicit rng for reproducibility"))
    end
    rng === nothing && return MLUtils.DataLoader(data; batchsize, shuffle, partial)
    return MLUtils.DataLoader(data; batchsize, shuffle, partial, rng)
end

function append_metric!(history::Vector{<:NamedTuple}; kwargs...)
    push!(history, NamedTuple(kwargs))
    return history
end
