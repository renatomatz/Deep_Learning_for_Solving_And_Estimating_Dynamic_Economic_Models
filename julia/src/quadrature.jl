export QuadratureRule,
    normalize_weights,
    quadrature_expectation,
    tensor_product_rule,
    stroud3_normal_rule,
    gauss_hermite_rule

using FastGaussQuadrature: gausshermite
using LinearAlgebra: I

struct QuadratureRule{N,W}
    nodes::N
    weights::W

    function QuadratureRule(nodes::AbstractVector, weights::AbstractVector)
        length(nodes) == length(weights) ||
            throw(DimensionMismatch("nodes and weights must have the same length"))
        return new{typeof(nodes),typeof(weights)}(nodes, weights)
    end

    function QuadratureRule(nodes::AbstractMatrix, weights::AbstractVector)
        size(nodes, 2) == length(weights) ||
            throw(DimensionMismatch("matrix nodes must be features-by-nodes"))
        return new{typeof(nodes),typeof(weights)}(nodes, weights)
    end
end

function normalize_weights(rule::QuadratureRule)
    total = sum(rule.weights)
    iszero(total) && throw(ArgumentError("quadrature weights must not sum to zero"))
    return QuadratureRule(rule.nodes, rule.weights ./ total)
end

_node_at(nodes::AbstractVector, i::Integer) = nodes[i]
_node_at(nodes::AbstractMatrix, i::Integer) = @view nodes[:, i]

function quadrature_expectation(f, rule::QuadratureRule; normalize::Bool = false)
    active_rule = normalize ? normalize_weights(rule) : rule
    total = active_rule.weights[1] * f(_node_at(active_rule.nodes, 1))
    for i in 2:length(active_rule.weights)
        total += active_rule.weights[i] * f(_node_at(active_rule.nodes, i))
    end
    return total
end

function tensor_product_rule(rules::QuadratureRule...)
    isempty(rules) && throw(ArgumentError("at least one rule is required"))
    all(rule -> rule.nodes isa AbstractVector, rules) ||
        throw(ArgumentError("tensor_product_rule currently expects one-dimensional vector-node rules"))

    dims = length(rules)
    counts = map(rule -> length(rule.weights), rules)
    total_nodes = prod(counts)
    nodes = Matrix{promote_type(map(rule -> eltype(rule.nodes), rules)...)}(undef, dims, total_nodes)
    weights = Vector{promote_type(map(rule -> eltype(rule.weights), rules)...)}(undef, total_nodes)

    column = 1
    for index in Iterators.product((1:n for n in counts)...)
        weight = one(eltype(weights))
        for d in 1:dims
            nodes[d, column] = rules[d].nodes[index[d]]
            weight *= rules[d].weights[index[d]]
        end
        weights[column] = weight
        column += 1
    end
    return QuadratureRule(nodes, weights)
end

function stroud3_normal_rule(dim::Integer; normalize::Bool = true)
    dim > 0 || throw(ArgumentError("dim must be positive"))
    radius = sqrt(float(dim))
    nodes = [begin
        axis = div(col + 1, 2)
        sign = isodd(col) ? 1.0 : -1.0
        row == axis ? sign * radius : 0.0
    end for row in 1:dim, col in 1:(2dim)]
    weights = fill(1 / (2dim), 2dim)
    rule = QuadratureRule(nodes, weights)
    return normalize ? normalize_weights(rule) : rule
end

function gauss_hermite_rule(n::Integer; standard_normal::Bool = true, normalize::Bool = true)
    n > 0 || throw(ArgumentError("n must be positive"))
    nodes, weights = gausshermite(n)
    if standard_normal
        nodes = sqrt(2) .* nodes
        weights = weights ./ sqrt(pi)
    end
    rule = QuadratureRule(nodes, weights)
    return normalize ? normalize_weights(rule) : rule
end
