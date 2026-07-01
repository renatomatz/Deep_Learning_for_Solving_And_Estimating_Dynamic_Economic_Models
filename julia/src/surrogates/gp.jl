export BoxNormalizer,
    fit_box_normalizer,
    normalize_box,
    denormalize_box,
    black_scholes_call_price_5d,
    black_scholes_design,
    standardize_targets,
    unstandardize_targets,
    rbf_kernel,
    fit_cholesky_gp,
    gp_predict,
    gp_rmse,
    bal_next_index,
    active_subspace_matrix,
    active_subspace,
    project_active_subspace,
    finite_difference_gradients,
    polynomial_features,
    ridge_fit,
    ridge_predict,
    fit_active_subspace_surrogate,
    predict_active_subspace_surrogate,
    encoder_widths,
    make_deep_active_subspace,
    radial_ridge_target,
    radial_ridge_gradients,
    standard_normal_quantile,
    borehole_physical_from_unit,
    borehole_function

using LinearAlgebra: I, Symmetric, cholesky, eigen
using Lux
using NNlib
using Random: AbstractRNG, rand
using Statistics: mean, std

struct BoxNormalizer{T}
    lower::Vector{T}
    upper::Vector{T}
end

function fit_box_normalizer(bounds)
    lower = [float(first(b)) for b in bounds]
    upper = [float(last(b)) for b in bounds]
    length(lower) == length(upper) || throw(DimensionMismatch("lower and upper bounds differ in length"))
    all(upper .> lower) || throw(ArgumentError("each upper bound must exceed its lower bound"))
    return BoxNormalizer(lower, upper)
end

function normalize_box(x::AbstractMatrix, normalizer::BoxNormalizer)
    size(x, 1) == length(normalizer.lower) ||
        throw(DimensionMismatch("input has $(size(x, 1)) rows, expected $(length(normalizer.lower))"))
    return (x .- normalizer.lower) ./ (normalizer.upper .- normalizer.lower)
end

function denormalize_box(x::AbstractMatrix, normalizer::BoxNormalizer)
    size(x, 1) == length(normalizer.lower) ||
        throw(DimensionMismatch("input has $(size(x, 1)) rows, expected $(length(normalizer.lower))"))
    return normalizer.lower .+ x .* (normalizer.upper .- normalizer.lower)
end

function black_scholes_call_price_5d(S, K, T, sigma, r; eps_safe = 1e-12)
    T <= eps_safe && return max(S - K, zero(promote(S, K)[1]))
    S <= eps_safe && return zero(promote(S, K, T, sigma, r)[1])
    sigma > 0 || throw(ArgumentError("sigma must be positive"))
    K > 0 || throw(ArgumentError("strike K must be positive"))
    d1 = (log(S / K) + (r + 0.5 * sigma^2) * T) / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)
    return S * standard_normal_cdf(d1) - K * exp(-r * T) * standard_normal_cdf(d2)
end

function black_scholes_design(rng::AbstractRNG, n::Integer;
        bounds = ((50.0, 150.0), (50.0, 150.0), (0.1, 2.0), (0.05, 0.6), (0.01, 0.08)))
    n > 0 || throw(ArgumentError("n must be positive"))
    normalizer = fit_box_normalizer(bounds)
    x_unit = rand(rng, length(bounds), n)
    x = denormalize_box(x_unit, normalizer)
    y = [black_scholes_call_price_5d(x[1, j], x[2, j], x[3, j], x[4, j], x[5, j]) for j in axes(x, 2)]
    return (x = x, y = reshape(y, 1, :), normalizer = normalizer)
end

function standardize_targets(y)
    mu = mean(y)
    sigma = std(vec(y))
    sigma > 0 || throw(ArgumentError("target standard deviation must be positive"))
    return (z = (y .- mu) ./ sigma, mean = mu, std = sigma)
end

unstandardize_targets(z, standardizer) = standardizer.mean .+ standardizer.std .* z

function _squared_distances(x1::AbstractMatrix, x2::AbstractMatrix)
    size(x1, 1) == size(x2, 1) ||
        throw(DimensionMismatch("kernel inputs must have the same feature dimension"))
    d = Matrix{promote_type(eltype(x1), eltype(x2), Float64)}(undef, size(x1, 2), size(x2, 2))
    for j in axes(x2, 2), i in axes(x1, 2)
        d[i, j] = sum(abs2, @view(x1[:, i]) .- @view(x2[:, j]))
    end
    return d
end

function rbf_kernel(x1::AbstractMatrix, x2::AbstractMatrix; lengthscale::Real = 1, variance::Real = 1)
    lengthscale > 0 || throw(ArgumentError("lengthscale must be positive"))
    variance > 0 || throw(ArgumentError("variance must be positive"))
    return variance .* exp.(-0.5 .* _squared_distances(x1, x2) ./ lengthscale^2)
end

function fit_cholesky_gp(x::AbstractMatrix, y; lengthscale::Real = 1, variance::Real = 1, noise::Real = 1e-6, mean_value = mean(y))
    noise > 0 || throw(ArgumentError("noise must be positive"))
    y_vec = collect(vec(y))
    size(x, 2) == length(y_vec) || throw(DimensionMismatch("x columns and y length must match"))
    K = rbf_kernel(x, x; lengthscale, variance)
    K_reg = K + (noise^2) * Matrix{eltype(K)}(I, size(K, 1), size(K, 2))
    chol = cholesky(Symmetric(K_reg))
    centered = y_vec .- mean_value
    alpha = chol \ centered
    return (x = x, y = y_vec, mean_value = mean_value, lengthscale = lengthscale,
        variance = variance, noise = noise, chol = chol, alpha = alpha)
end

function gp_predict(gp, xstar::AbstractMatrix)
    Ks = rbf_kernel(gp.x, xstar; lengthscale = gp.lengthscale, variance = gp.variance)
    mean_pred = gp.mean_value .+ vec(Ks' * gp.alpha)
    v = gp.chol.L \ Ks
    Kss_diag = fill(gp.variance, size(xstar, 2))
    var_pred = max.(Kss_diag .- vec(sum(abs2, v; dims = 1)), 0)
    return (mean = reshape(mean_pred, 1, :), variance = reshape(var_pred, 1, :))
end

function gp_rmse(gp, x::AbstractMatrix, y)
    pred = gp_predict(gp, x).mean
    return sqrt(mean(abs2, vec(pred) .- vec(y)))
end

function bal_next_index(gp, candidates::AbstractMatrix)
    variance = vec(gp_predict(gp, candidates).variance)
    return argmax(variance)
end

function active_subspace_matrix(gradients::AbstractMatrix)
    size(gradients, 2) > 0 || throw(ArgumentError("at least one gradient column is required"))
    return gradients * gradients' / size(gradients, 2)
end

function active_subspace(C::AbstractMatrix)
    size(C, 1) == size(C, 2) || throw(DimensionMismatch("active-subspace matrix must be square"))
    eig = eigen(Symmetric(C))
    order = sortperm(eig.values; rev = true)
    return (values = eig.values[order], vectors = eig.vectors[:, order])
end

function project_active_subspace(x::AbstractMatrix, vectors::AbstractMatrix, dims::Integer)
    0 < dims <= size(vectors, 2) || throw(ArgumentError("dims must select available active directions"))
    size(x, 1) == size(vectors, 1) || throw(DimensionMismatch("x and vectors have incompatible dimensions"))
    return vectors[:, 1:dims]' * x
end

function finite_difference_gradients(f, x::AbstractMatrix; h::Real = 1e-5)
    h > 0 || throw(ArgumentError("h must be positive"))
    gradients = similar(float.(x))
    for j in axes(x, 2), i in axes(x, 1)
        xp = collect(@view x[:, j])
        xm = collect(@view x[:, j])
        xp[i] += h
        xm[i] -= h
        gradients[i, j] = (f(xp) - f(xm)) / (2h)
    end
    return gradients
end

function _append_exponents!(out, current, remaining_dim, remaining_degree)
    if remaining_dim == 1
        for k in 0:remaining_degree
            push!(out, [current...; k])
        end
    else
        for k in 0:remaining_degree
            _append_exponents!(out, [current...; k], remaining_dim - 1, remaining_degree - k)
        end
    end
    return out
end

function _polynomial_exponents(dim::Integer, degree::Integer)
    dim > 0 || throw(ArgumentError("dim must be positive"))
    degree >= 0 || throw(ArgumentError("degree must be nonnegative"))
    exponents = Vector{Vector{Int}}()
    _append_exponents!(exponents, Int[], dim, degree)
    return sort(exponents; by = e -> (sum(e), e))
end

function polynomial_features(z::AbstractMatrix; degree::Integer = 3)
    exponents = _polynomial_exponents(size(z, 1), degree)
    Phi = Matrix{promote_type(eltype(z), Float64)}(undef, length(exponents), size(z, 2))
    for (row, exps) in enumerate(exponents)
        vals = ones(eltype(Phi), size(z, 2))
        for (i, p) in enumerate(exps)
            p == 0 && continue
            vals .*= z[i, :] .^ p
        end
        Phi[row, :] = vals
    end
    return (features = Phi, exponents = exponents)
end

function ridge_fit(Phi::AbstractMatrix, y; lambda::Real = 1e-6)
    lambda >= 0 || throw(ArgumentError("lambda must be nonnegative"))
    y_vec = reshape(collect(vec(y)), 1, :)
    size(Phi, 2) == size(y_vec, 2) || throw(DimensionMismatch("Phi columns and y length must match"))
    A = Phi * Phi' + lambda * Matrix{promote_type(eltype(Phi), Float64)}(I, size(Phi, 1), size(Phi, 1))
    coef = (A \ (Phi * y_vec'))'
    return coef
end

ridge_predict(coef::AbstractMatrix, Phi::AbstractMatrix) = coef * Phi

function fit_active_subspace_surrogate(x::AbstractMatrix, y::AbstractMatrix, directions::AbstractMatrix;
        dims::Integer = 1, degree::Integer = 3, lambda::Real = 1e-6)
    z = project_active_subspace(x, directions, dims)
    poly = polynomial_features(z; degree)
    coef = ridge_fit(poly.features, y; lambda)
    return (directions = directions[:, 1:dims], degree = degree, exponents = poly.exponents, coef = coef)
end

function _polynomial_features_with_exponents(z::AbstractMatrix, exponents)
    Phi = Matrix{promote_type(eltype(z), Float64)}(undef, length(exponents), size(z, 2))
    for (row, exps) in enumerate(exponents)
        vals = ones(eltype(Phi), size(z, 2))
        for (i, p) in enumerate(exps)
            p == 0 && continue
            vals .*= z[i, :] .^ p
        end
        Phi[row, :] = vals
    end
    return Phi
end

function predict_active_subspace_surrogate(fit, x::AbstractMatrix)
    z = fit.directions' * x
    Phi = _polynomial_features_with_exponents(z, fit.exponents)
    return ridge_predict(fit.coef, Phi)
end

function encoder_widths(input_dim::Integer, latent_dim::Integer, layers::Integer = 3)
    input_dim > 0 || throw(ArgumentError("input_dim must be positive"))
    latent_dim > 0 || throw(ArgumentError("latent_dim must be positive"))
    layers > 0 || throw(ArgumentError("layers must be positive"))
    rho = log(latent_dim / input_dim) / layers
    widths = [input_dim]
    for k in 1:(layers - 1)
        push!(widths, max(latent_dim, ceil(Int, input_dim * exp(rho * k))))
    end
    push!(widths, latent_dim)
    return widths
end

function make_deep_active_subspace(input_dim::Integer, latent_dim::Integer;
        encoder_layers::Integer = 3, link_hidden::Integer = 16)
    encoder = make_mlp(input_dim, Tuple(encoder_widths(input_dim, latent_dim, encoder_layers)[2:(end - 1)]), latent_dim;
        activation = NNlib.swish)
    link = make_mlp(latent_dim, (link_hidden, link_hidden), 1; activation = NNlib.swish)
    return Lux.Chain(encoder, link)
end

function radial_ridge_target(x::AbstractMatrix, directions::AbstractMatrix)
    z = directions' * x
    return reshape(exp.(-vec(sum(abs2, z; dims = 1))), 1, :)
end

function radial_ridge_gradients(x::AbstractMatrix, directions::AbstractMatrix)
    z = directions' * x
    y = vec(exp.(-sum(abs2, z; dims = 1)))
    return -2 .* directions * (z .* reshape(y, 1, :))
end

function standard_normal_quantile(p::Real)
    0 < p < 1 || throw(ArgumentError("p must lie in (0, 1)"))
    a = (-39.69683028665376, 220.9460984245205, -275.9285104469687, 138.3577518672690, -30.66479806614716, 2.506628277459239)
    b = (-54.47609879822406, 161.5858368580409, -155.6989798598866, 66.80131188771972, -13.28068155288572)
    c = (-0.007784894002430293, -0.3223964580411365, -2.400758277161838, -2.549732539343734, 4.374664141464968, 2.938163982698783)
    d = (0.007784695709041462, 0.3224671290700398, 2.445134137142996, 3.754408661907416)
    plow = 0.02425
    phigh = 1 - plow
    if p < plow
        q = sqrt(-2log(p))
        return (((((c[1] * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) * q + c[6]) /
            ((((d[1] * q + d[2]) * q + d[3]) * q + d[4]) * q + 1)
    elseif p <= phigh
        q = p - 0.5
        r = q^2
        return (((((a[1] * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * r + a[6]) * q /
            (((((b[1] * r + b[2]) * r + b[3]) * r + b[4]) * r + b[5]) * r + 1)
    else
        q = sqrt(-2log(1 - p))
        return -(((((c[1] * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) * q + c[6]) /
            ((((d[1] * q + d[2]) * q + d[3]) * q + d[4]) * q + 1)
    end
end

function borehole_physical_from_unit(xi::AbstractMatrix)
    size(xi, 1) == 8 || throw(DimensionMismatch("borehole inputs must be 8-by-batch unit coordinates"))
    u = clamp.(xi, 1e-9, 1 - 1e-9)
    phys = similar(float.(u))
    phys[1, :] = 0.10 .+ 0.0161812 .* standard_normal_quantile.(u[1, :])
    phys[2, :] = exp.(7.71 .+ 1.0056 .* standard_normal_quantile.(u[2, :]))
    phys[3, :] = 63070.0 .+ (115600.0 - 63070.0) .* u[3, :]
    phys[4, :] = 990.0 .+ (1110.0 - 990.0) .* u[4, :]
    phys[5, :] = 63.1 .+ (116.0 - 63.1) .* u[5, :]
    phys[6, :] = 700.0 .+ (820.0 - 700.0) .* u[6, :]
    phys[7, :] = 1120.0 .+ (1680.0 - 1120.0) .* u[7, :]
    phys[8, :] = 9855.0 .+ (12045.0 - 9855.0) .* u[8, :]
    return phys
end

function borehole_function(phys::AbstractMatrix)
    size(phys, 1) == 8 || throw(DimensionMismatch("borehole physical inputs must be 8-by-batch"))
    rw, r, Tu, Hu, Tl, Hl, L, Kw = (phys[i, :] for i in 1:8)
    logr = log.(r ./ rw)
    numerator = 2pi .* Tu .* (Hu .- Hl)
    denominator = logr .* (1 .+ 2 .* L .* Tu ./ (logr .* rw .^ 2 .* Kw) .+ Tu ./ Tl)
    return reshape(numerator ./ denominator, 1, :)
end
