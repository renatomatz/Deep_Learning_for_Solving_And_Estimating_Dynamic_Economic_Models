using LinearAlgebra: I

export young_weights,
    redistribute_mass,
    redistribute_distribution,
    young_step,
    young_mass,
    young_mean,
    flatten_young_histogram,
    unflatten_young_histogram,
    validate_transition_matrix

function _check_grid(grid::AbstractVector)
    length(grid) >= 2 || throw(ArgumentError("grid must contain at least two points"))
    all(diff(grid) .> 0) || throw(ArgumentError("grid must be strictly increasing"))
    return grid
end

function young_weights(grid::AbstractVector, x::Real; clip::Bool = true)
    _check_grid(grid)
    if x <= first(grid)
        clip || throw(DomainError(x, "point lies below the grid"))
        return (lower = 1, upper = 1, lower_weight = one(float(x)), upper_weight = zero(float(x)), clipped = x < first(grid))
    elseif x >= last(grid)
        clip || throw(DomainError(x, "point lies above the grid"))
        n = length(grid)
        return (lower = n, upper = n, lower_weight = one(float(x)), upper_weight = zero(float(x)), clipped = x > last(grid))
    end

    lower = searchsortedlast(grid, x)
    lower = clamp(lower, 1, length(grid) - 1)
    upper = lower + 1
    span = grid[upper] - grid[lower]
    upper_weight = (x - grid[lower]) / span
    lower_weight = 1 - upper_weight
    return (lower = lower, upper = upper, lower_weight = lower_weight, upper_weight = upper_weight, clipped = false)
end

function redistribute_mass(grid::AbstractVector, x::Real, mass::Real; clip::Bool = true)
    weights = young_weights(grid, x; clip)
    out = zeros(promote_type(eltype(grid), typeof(float(x)), typeof(float(mass))), length(grid))
    out[weights.lower] += weights.lower_weight * mass
    out[weights.upper] += weights.upper_weight * mass
    return out
end

function redistribute_distribution(grid::AbstractVector, values, masses; clip::Bool = true)
    length(values) == length(masses) || throw(DimensionMismatch("values and masses must have the same length"))
    out = zeros(promote_type(eltype(grid), eltype(float.(values)), eltype(float.(masses))), length(grid))
    for (x, mass) in zip(values, masses)
        out .+= redistribute_mass(grid, x, mass; clip)
    end
    return out
end

young_mass(hist) = sum(hist)

young_mean(grid::AbstractVector, hist::AbstractVector) = sum(grid .* hist) / young_mass(hist)

function young_mean(grid::AbstractVector, hist::AbstractMatrix)
    size(hist, 2) == length(grid) || throw(DimensionMismatch("histogram matrix must be shocks-by-grid"))
    return sum(reshape(grid, 1, :) .* hist) / young_mass(hist)
end

function validate_transition_matrix(transition::AbstractMatrix; atol::Real = 1e-10)
    size(transition, 1) == size(transition, 2) || throw(DimensionMismatch("transition matrix must be square"))
    all(transition .>= 0) || throw(ArgumentError("transition probabilities must be nonnegative"))
    all(abs.(sum(transition; dims = 2) .- 1) .<= atol) || throw(ArgumentError("transition rows must sum to one"))
    return transition
end

_policy_value(policy, k, shock) = policy(k, shock)
_policy_value(policy::AbstractVector, i::Integer, shock) = policy[i]
_policy_value(policy::AbstractMatrix, i::Integer, shock) = policy[shock, i]

function young_step(grid::AbstractVector, hist::AbstractVector, policy; clip::Bool = true)
    length(hist) == length(grid) || throw(DimensionMismatch("hist and grid must have the same length"))
    out = zeros(promote_type(eltype(grid), eltype(hist)), length(grid))
    for i in eachindex(grid)
        mass = hist[i]
        iszero(mass) && continue
        x_next = policy isa AbstractVector ? _policy_value(policy, i, 1) : policy(grid[i])
        out .+= redistribute_mass(grid, x_next, mass; clip)
    end
    return out
end

function young_step(grid::AbstractVector, hist::AbstractMatrix, policy; transition = nothing, clip::Bool = true)
    n_shocks, n_grid = size(hist)
    n_grid == length(grid) || throw(DimensionMismatch("histogram matrix must be shocks-by-grid"))
    transition === nothing && (transition = Matrix{eltype(hist)}(I, n_shocks, n_shocks))
    validate_transition_matrix(transition)
    size(transition) == (n_shocks, n_shocks) || throw(DimensionMismatch("transition size must match shock states"))

    out = zeros(promote_type(eltype(grid), eltype(hist), eltype(transition)), n_shocks, n_grid)
    for shock in 1:n_shocks, i in eachindex(grid)
        mass = hist[shock, i]
        iszero(mass) && continue
        x_next = policy isa AbstractArray ? _policy_value(policy, i, shock) : policy(grid[i], shock)
        scattered = redistribute_mass(grid, x_next, mass; clip)
        for shock_next in 1:n_shocks
            out[shock_next, :] .+= transition[shock, shock_next] .* scattered
        end
    end
    return out
end

function flatten_young_histogram(hist::AbstractMatrix)
    return reshape(hist, :, 1)
end

function unflatten_young_histogram(flat::AbstractVecOrMat, n_shocks::Integer, n_grid::Integer)
    length(flat) == n_shocks * n_grid || throw(DimensionMismatch("flat histogram has incompatible length"))
    return reshape(vec(flat), n_shocks, n_grid)
end
