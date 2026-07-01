export assert_all_finite,
    finite_share,
    residual_summary,
    policy_drift,
    relative_l2_error,
    max_abs_error,
    check_bounds

using LinearAlgebra: norm
using Statistics: mean

function assert_all_finite(x; name::AbstractString = "value")
    all(isfinite, x) || throw(DomainError(x, "$name contains non-finite entries"))
    return x
end

finite_share(x) = mean(isfinite.(x))

function residual_summary(residuals)
    r = collect(vec(residuals))
    isempty(r) && throw(ArgumentError("residuals must be nonempty"))
    return (
        mean_abs = mean(abs, r),
        max_abs = maximum(abs.(r)),
        rmse = sqrt(mean(abs2, r)),
        finite_share = finite_share(r),
    )
end

function policy_drift(old_policy, new_policy; eps::Real = 1e-12)
    numerator = norm(vec(new_policy .- old_policy))
    denominator = max(norm(vec(old_policy)), eps)
    return numerator / denominator
end

function relative_l2_error(estimate, truth; eps::Real = 1e-12)
    return norm(vec(estimate .- truth)) / max(norm(vec(truth)), eps)
end

max_abs_error(estimate, truth) = maximum(abs.(estimate .- truth))

function check_bounds(x; lower = -Inf, upper = Inf)
    lower <= upper || throw(ArgumentError("lower bound must not exceed upper bound"))
    return all(lower .<= x .<= upper)
end
