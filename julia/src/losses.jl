export mse_loss,
    mae_loss,
    huber_loss,
    logcosh_loss,
    pinball_loss,
    cvar_loss,
    smooth_cvar_loss,
    LOSS_KERNELS,
    loss_kernel_value,
    equal_loss_weights,
    simplex_inverse_loss_weights,
    softadapt_weights,
    fischer_burmeister,
    inverse_loss_weights,
    relobralo_weights

using Statistics: mean

mse_loss(pred, target) = mean(abs2, pred .- target)

mae_loss(pred, target) = mean(abs, pred .- target)

function huber_loss(pred, target; delta::Real = 1)
    delta > 0 || throw(ArgumentError("delta must be positive"))
    r = abs.(pred .- target)
    return mean(ifelse.(r .<= delta, 0.5 .* r .^ 2, delta .* (r .- 0.5delta)))
end

function logcosh_loss(pred, target)
    r = pred .- target
    a = abs.(r)
    return mean(a .+ log1p.(exp.(-2 .* a)) .- log(2))
end

function pinball_loss(pred, target; quantile::Real)
    0 < quantile < 1 || throw(ArgumentError("quantile must lie in (0, 1)"))
    err = target .- pred
    return mean(max.(quantile .* err, (quantile - 1) .* err))
end

function cvar_loss(losses; alpha::Real = 0.95)
    0 <= alpha < 1 || throw(ArgumentError("alpha must lie in [0, 1)"))
    values = sort!(collect(vec(losses)))
    isempty(values) && throw(ArgumentError("losses must be nonempty"))
    tail_count = max(1, ceil(Int, (1 - alpha) * length(values)))
    return mean(@view values[(end - tail_count + 1):end])
end

function smooth_cvar_loss(losses; temperature::Real = 12)
    temperature > 0 || throw(ArgumentError("temperature must be positive"))
    values = vec(float.(losses))
    isempty(values) && throw(ArgumentError("losses must be nonempty"))
    shifted = temperature .* (values .- maximum(values))
    raw_weights = exp.(shifted)
    weights = raw_weights ./ sum(raw_weights)
    return sum(weights .* values)
end


const LOSS_KERNELS = (:mse, :mae, :huber, :quantile, :cvar, :logcosh)

function _loss_kernel_symbol(kind)
    key = Symbol(lowercase(String(kind)))
    key == :log_cosh && return :logcosh
    key == :pinball && return :quantile
    key in LOSS_KERNELS || throw(ArgumentError("unknown loss kernel: $kind"))
    return key
end

function loss_kernel_value(kind, residuals; delta::Real = 1, quantile::Real = 0.9, alpha::Real = 0.9)
    r = residuals
    z = zero.(r)
    key = _loss_kernel_symbol(kind)
    key == :mse && return mse_loss(r, z)
    key == :mae && return mae_loss(r, z)
    key == :huber && return huber_loss(r, z; delta)
    key == :quantile && return pinball_loss(z, r; quantile)
    key == :cvar && return cvar_loss(abs.(r); alpha)
    return logcosh_loss(r, z)
end


function _as_loss_vector(losses)
    values = collect(float.(losses))
    isempty(values) && throw(ArgumentError("loss vector must be nonempty"))
    all(isfinite, values) || throw(DomainError(losses, "loss vector contains non-finite entries"))
    return values
end

function equal_loss_weights(losses; normalize::Symbol = :count)
    values = _as_loss_vector(losses)
    normalize == :count && return ones(length(values))
    normalize == :simplex && return fill(inv(length(values)), length(values))
    throw(ArgumentError("normalize must be :count or :simplex"))
end

function simplex_inverse_loss_weights(losses; eps::Real = 1e-8)
    eps > 0 || throw(ArgumentError("eps must be positive"))
    values = _as_loss_vector(losses)
    raw = inv.(max.(values, eps))
    return raw ./ sum(raw)
end

function softadapt_weights(current_losses, previous_losses; temperature::Real = 1, normalize::Symbol = :count, eps::Real = 1e-8)
    temperature > 0 || throw(ArgumentError("temperature must be positive"))
    current = _as_loss_vector(current_losses)
    previous = _as_loss_vector(previous_losses)
    length(current) == length(previous) || throw(DimensionMismatch("current and previous losses must have the same length"))
    slopes = (current .- previous) ./ max.(abs.(previous), eps)
    logits = slopes ./ temperature
    raw = exp.(logits .- maximum(logits))
    weights = raw ./ sum(raw)
    normalize == :simplex && return weights
    normalize == :count && return length(weights) .* weights
    throw(ArgumentError("normalize must be :count or :simplex"))
end

function fischer_burmeister(a, b; eps::Real = 0)
    eps >= 0 || throw(ArgumentError("eps must be nonnegative"))
    return sqrt.(a .^ 2 .+ b .^ 2 .+ eps^2) .- a .- b
end

function inverse_loss_weights(losses; eps::Real = 1e-8, normalize::Bool = true)
    eps > 0 || throw(ArgumentError("eps must be positive"))
    raw = inv.(max.(losses, eps))
    normalize || return raw
    return length(raw) .* raw ./ sum(raw)
end

function relobralo_weights(current_losses, reference_losses; temperature::Real = 1, eps::Real = 1e-8)
    temperature > 0 || throw(ArgumentError("temperature must be positive"))
    current = collect(current_losses)
    reference = collect(reference_losses)
    length(current) == length(reference) ||
        throw(DimensionMismatch("current and reference losses must have the same length"))
    ratios = current ./ max.(reference, eps)
    logits = ratios ./ temperature
    shifted = logits .- maximum(logits)
    raw = exp.(shifted)
    return length(raw) .* raw ./ sum(raw)
end
