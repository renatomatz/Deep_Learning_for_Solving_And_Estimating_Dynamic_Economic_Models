export make_mlp,
    setup_model,
    convert_parameter_tree,
    to_feature_batch,
    to_batch_features,
    assert_feature_batch,
    assert_matching_batch,
    split_output_heads,
    sigmoid_bounds,
    positive_softplus,
    capped_softplus

using Lux
using NNlib

function make_mlp(input_dim::Integer, hidden_dims, output_dim::Integer;
        activation = NNlib.relu,
        final_activation = identity)
    input_dim > 0 || throw(ArgumentError("input_dim must be positive"))
    output_dim > 0 || throw(ArgumentError("output_dim must be positive"))

    layers = Any[]
    last_dim = input_dim
    for width in hidden_dims
        width > 0 || throw(ArgumentError("hidden layer widths must be positive"))
        push!(layers, Lux.Dense(last_dim => width, activation))
        last_dim = width
    end
    push!(layers, Lux.Dense(last_dim => output_dim, final_activation))
    return Lux.Chain(layers...)
end

function convert_parameter_tree(::Type{T}, ps) where {T<:AbstractFloat}
    return _convert_parameter_leaf(T, ps)
end

_convert_parameter_leaf(::Type{T}, x::AbstractArray{<:Number}) where {T<:AbstractFloat} = T.(x)
_convert_parameter_leaf(::Type{T}, x::Number) where {T<:AbstractFloat} = T(x)
_convert_parameter_leaf(::Type{T}, x::NamedTuple) where {T<:AbstractFloat} =
    NamedTuple{keys(x)}(map(v -> _convert_parameter_leaf(T, v), values(x)))
_convert_parameter_leaf(::Type{T}, x::Tuple) where {T<:AbstractFloat} =
    map(v -> _convert_parameter_leaf(T, v), x)
_convert_parameter_leaf(::Type{T}, x) where {T<:AbstractFloat} = x

function setup_model(rng, model; parameter_type::Union{Nothing,Type{<:AbstractFloat}} = nothing)
    ps, st = Lux.setup(rng, model)
    parameter_type === nothing && return ps, st
    return convert_parameter_tree(parameter_type, ps), st
end

function to_feature_batch(x::AbstractVector; as::Symbol = :batch)
    if as == :batch
        return reshape(x, 1, :)
    elseif as == :features
        return reshape(x, :, 1)
    else
        throw(ArgumentError("as must be :batch or :features"))
    end
end

function to_feature_batch(x::AbstractMatrix; orientation::Symbol = :batch_features)
    if orientation == :features_batch
        return x
    elseif orientation == :batch_features
        return permutedims(x)
    else
        throw(ArgumentError("orientation must be :batch_features or :features_batch"))
    end
end

function to_batch_features(x::AbstractMatrix)
    return permutedims(x)
end

function assert_feature_batch(x::AbstractMatrix, input_dim::Integer)
    size(x, 1) == input_dim ||
        throw(DimensionMismatch("expected $input_dim features in dimension 1; got $(size(x, 1))"))
    return x
end

function assert_matching_batch(x::AbstractMatrix, y::AbstractMatrix)
    size(x, 2) == size(y, 2) ||
        throw(DimensionMismatch("feature and target batches must have the same number of columns"))
    return x, y
end

function split_output_heads(y::AbstractMatrix, sizes::NamedTuple)
    widths = Tuple(values(sizes))
    all(width -> width > 0, widths) || throw(ArgumentError("head widths must be positive"))
    total = sum(widths)
    size(y, 1) == total ||
        throw(DimensionMismatch("output has $(size(y, 1)) rows, but head sizes sum to $total"))

    return NamedTuple{keys(sizes)}(_split_output_head_views(y, widths))
end

function _split_output_head_views(y::AbstractMatrix, widths::Tuple{Int})
    w1 = widths[1]
    return (@view(y[1:w1, :]),)
end

function _split_output_head_views(y::AbstractMatrix, widths::Tuple{Int,Int})
    w1, w2 = widths
    return (@view(y[1:w1, :]), @view(y[(w1 + 1):(w1 + w2), :]))
end

function _split_output_head_views(y::AbstractMatrix, widths::Tuple{Int,Int,Int})
    w1, w2, w3 = widths
    return (
        @view(y[1:w1, :]),
        @view(y[(w1 + 1):(w1 + w2), :]),
        @view(y[(w1 + w2 + 1):(w1 + w2 + w3), :]),
    )
end

_split_output_head_views(y::AbstractMatrix, widths::Tuple{}) = ()
_split_output_head_views(y::AbstractMatrix, widths::Tuple) = _split_output_head_views_from(y, widths, 1)

function _split_output_head_views_from(y::AbstractMatrix, widths::Tuple{}, start::Integer)
    return ()
end

function _split_output_head_views_from(y::AbstractMatrix, widths::Tuple, start::Integer)
    width = first(widths)
    stop = start + width - 1
    return (@view(y[start:stop, :]), _split_output_head_views_from(y, Base.tail(widths), stop + 1)...)
end

function sigmoid_bounds(x, lower, upper)
    upper > lower || throw(ArgumentError("upper bound must exceed lower bound"))
    return lower .+ (upper - lower) .* NNlib.sigmoid.(x)
end

function positive_softplus(x; floor = 0, scale = 1)
    scale > 0 || throw(ArgumentError("scale must be positive"))
    return floor .+ scale .* NNlib.softplus.(x)
end

function capped_softplus(x, cap; floor = 0, scale = 1)
    cap > floor || throw(ArgumentError("cap must exceed floor"))
    return min.(positive_softplus(x; floor, scale), cap)
end
