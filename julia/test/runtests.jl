using Test

using DLEFJulia
using ForwardDiff
using LinearAlgebra: I
using Lux
using NNlib
using Optimisers
using StableRNGs
using Statistics: mean
using Zygote

struct ConstantSavingsPolicy{T}
    savings::T
end

(m::ConstantSavingsPolicy)(x, ps, st) = (fill(m.savings, 1, size(x, 2)), st)

struct AnalyticZeroBCODE end

(m::AnalyticZeroBCODE)(x, ps, st) = (analytic_zero_bc_solution.(x), st)

struct ConstantOutputPolicy{T}
    rows::Int
    value::T
end

(m::ConstantOutputPolicy)(x, ps, st) = (fill(m.value, m.rows, size(x, 2)), st)

@testset "run modes" begin
    @test validate_run_mode("smoke") == "smoke"
    @test run_mode_symbol(" Teaching ") == :teaching
    @test run_mode_budget("smoke").epochs <= run_mode_budget("teaching").epochs
    @test rand(rng_from_seed(7), 3) == rand(rng_from_seed(7), 3)
    @test_throws ArgumentError validate_run_mode("demo")
end

@testset "Lux helper conventions" begin
    x_batch_features = Float32[1.0 2.0; 3.0 4.0; 5.0 6.0]
    x_feature_batch = to_feature_batch(x_batch_features)
    feature_vector = Float32[1.0, 2.0]
    @test to_feature_batch(feature_vector; as = :batch) == reshape(feature_vector, 1, :)
    @test to_feature_batch(feature_vector; as = :features) == reshape(feature_vector, :, 1)
    @test_throws ArgumentError to_feature_batch(feature_vector; as = :unknown)
    @test size(x_feature_batch) == (2, 3)
    @test to_batch_features(x_feature_batch) == x_batch_features
    @test assert_feature_batch(x_feature_batch, 2) === x_feature_batch
    target_batch = reshape(1:3, 1, 3)
    @test assert_matching_batch(x_feature_batch, target_batch) == (x_feature_batch, target_batch)
    @test_throws DimensionMismatch assert_matching_batch(x_feature_batch, reshape(1:4, 1, 4))

    y = reshape(1:12, 4, 3)
    heads = split_output_heads(y, (policy = 2, multiplier = 2))
    @test heads.policy == y[1:2, :]
    @test heads.multiplier == y[3:4, :]
    four_head_grad = Zygote.gradient(mat -> sum(abs2, split_output_heads(mat, (a = 1, b = 1, c = 1, d = 1)).d), float.(y))[1]
    @test size(four_head_grad) == size(y)
    @test all(isfinite, four_head_grad)

    bounded = sigmoid_bounds([-100.0, 0.0, 100.0], -2.0, 3.0)
    @test all(-2.0 .<= bounded .<= 3.0)
    @test all(positive_softplus([-1.0, 0.0, 1.0]) .> 0)
    @test all(capped_softplus([-1.0, 10.0], 2.0) .<= 2.0)

    model = make_mlp(2, (4, 3), 1)
    ps, st = setup_model(StableRNG(1), model)
    out, st_new = model(x_feature_batch, ps, st)
    @test size(out) == (1, 3)
    @test st_new isa NamedTuple

    ps64, _ = setup_model(StableRNG(1), model; parameter_type = Float64)
    @test eltype(ps64.layer_1.weight) == Float64
end

@testset "losses" begin
    pred = [1.0, 2.0, 4.0]
    target = [1.0, 1.0, 1.0]
    @test mse_loss(pred, target) ≈ 10 / 3
    @test mae_loss(pred, target) ≈ 4 / 3
    @test huber_loss(pred, target; delta = 1.0) ≈ mean([0.0, 0.5, 2.5])
    @test logcosh_loss(pred, pred) ≈ 0.0 atol = 1e-12
    @test pinball_loss([0.0], [2.0]; quantile = 0.5) ≈ 1.0
    @test cvar_loss([1.0, 2.0, 10.0, 20.0]; alpha = 0.5) ≈ 15.0
    @test smooth_cvar_loss([1.0, 2.0, 10.0, 20.0]; temperature = 4) > mean([1.0, 2.0, 10.0, 20.0])
    smooth_cvar_grad = Zygote.gradient(x -> smooth_cvar_loss(abs.(x); temperature = 4), [-1.0, 0.5, 3.0])[1]
    @test all(isfinite, smooth_cvar_grad)
    residuals = [-2.0, -0.5, 0.25, 4.0]
    @test loss_kernel_value(:mse, residuals) ≈ mean(abs2, residuals)
    @test loss_kernel_value("MAE", residuals) ≈ mean(abs, residuals)
    @test loss_kernel_value(:pinball, residuals; quantile = 0.9) ≈ loss_kernel_value(:quantile, residuals; quantile = 0.9)
    @test loss_kernel_value(:cvar, residuals; alpha = 0.5) ≈ 3.0
    @test sum(equal_loss_weights(residuals)) ≈ length(residuals)
    @test sum(equal_loss_weights(residuals; normalize = :simplex)) ≈ 1.0
    @test sum(simplex_inverse_loss_weights(abs.(residuals) .+ 1.0)) ≈ 1.0
    @test sum(softadapt_weights([1.0, 1.2], [1.0, 1.0])) ≈ 2.0
    @test_throws ArgumentError loss_kernel_value(:unknown, residuals)
    @test fischer_burmeister(0.0, 2.0) ≈ 0.0
    @test fischer_burmeister(1.0, 1.0) < 0
    @test sum(inverse_loss_weights([1.0, 2.0, 4.0])) ≈ 3.0
    @test sum(relobralo_weights([1.0, 2.0], [1.0, 1.0])) ≈ 2.0
end

@testset "quadrature" begin
    rule = QuadratureRule([-1.0, 1.0], [0.25, 0.25])
    normalized = normalize_weights(rule)
    @test sum(normalized.weights) ≈ 1.0
    @test quadrature_expectation(x -> x^2, normalized) ≈ 1.0

    product = tensor_product_rule(normalized, normalized)
    @test size(product.nodes) == (2, 4)
    @test sum(product.weights) ≈ 1.0
    @test quadrature_expectation(x -> x[1] * x[2], product) ≈ 0.0 atol = 1e-12

    stroud = stroud3_normal_rule(3)
    @test size(stroud.nodes) == (3, 6)
    @test sum(stroud.weights) ≈ 1.0
    @test quadrature_expectation(x -> x[1], stroud) ≈ 0.0 atol = 1e-12
    @test quadrature_expectation(x -> x[1]^2, stroud) ≈ 1.0

    gh = gauss_hermite_rule(5)
    @test sum(gh.weights) ≈ 1.0
    @test quadrature_expectation(x -> x, gh) ≈ 0.0 atol = 1e-12
    @test quadrature_expectation(x -> x^2, gh) ≈ 1.0 atol = 1e-12
end

@testset "diagnostics" begin
    residuals = [-1.0, 0.0, 2.0]
    summary = residual_summary(residuals)
    @test summary.max_abs == 2.0
    @test summary.finite_share == 1.0
    @test policy_drift([1.0, 1.0], [1.0, 2.0]) > 0
    @test relative_l2_error([1.0, 2.0], [1.0, 2.0]) == 0.0
    @test max_abs_error([1.0, 3.0], [2.0, 1.0]) == 2.0
    @test check_bounds([0.2, 0.5]; lower = 0.0, upper = 1.0)
    @test_throws DomainError assert_all_finite([1.0, NaN]; name = "test residual")
end

@testset "training scaffold" begin
    model = make_mlp(1, (4,), 1)
    state = setup_training(StableRNG(11), model, Optimisers.Descent(0.01))
    x = reshape(collect(range(Float32(-1), Float32(1); length = 5)), 1, :)
    target = 2 .* x

    loss_fn(model, ps, st) = begin
        y, st_new = model(x, ps, st)
        return mse_loss(y, target), st_new
    end

    metrics = train_step!(state, loss_fn; max_grad_norm = 10.0)
    @test metrics.step == 1
    @test isfinite(metrics.loss)
    @test isfinite(metrics.grad_norm)
    @test state.step == 1

    grads = (a = [3.0, 4.0], b = 0.0)
    clipped, norm_before = clip_gradient_norm(grads, 1.0)
    @test norm_before ≈ 5.0
    @test sqrt(tree_sum_abs2(clipped)) ≈ 1.0

    ps = (w = [1.0],)
    stateful = TrainState(nothing, ps, (calls = 0,), Optimisers.setup(Optimisers.Descent(0.1), ps), 0)
    stateful_loss(model, ps, st) = (sum(abs2, ps.w), (calls = st.calls + 1,))
    train_step!(stateful, stateful_loss)
    @test stateful.st.calls == 1
    @test stateful.ps.w[1] < 1.0

    data = reshape(collect(1:12), 3, 4)
    batches = collect(make_dataloader(data; batchsize = 2, shuffle = false))
    @test batches[1] == data[:, 1:2]
    @test batches[2] == data[:, 3:4]
    @test_throws ArgumentError make_dataloader(data; batchsize = 2, shuffle = true)
    shuffled1 = collect(make_dataloader(data; batchsize = 2, shuffle = true, rng = rng_from_seed(13)))
    shuffled2 = collect(make_dataloader(data; batchsize = 2, shuffle = true, rng = rng_from_seed(13)))
    @test shuffled1 == shuffled2

    batch_state = setup_training(model, state.ps, state.st, Optimisers.Descent(0.001))
    batch_loss(model, ps, st, batch) = begin
        y, st_new = model(batch.x, ps, st)
        return mse_loss(y, batch.y), st_new
    end
    batch = (x = x, y = target)
    @test isfinite(loss_value(batch_state, batch_loss, batch))
    batch_metrics = train_step!(batch_state, batch_loss, batch; max_grad_norm = 10.0)
    @test batch_metrics.step == 1
end

@testset "Brock-Mirman helpers" begin
    params = BrockMirmanParams(delta = 1.0)
    k = reshape(collect(range(0.2, 1.0; length = 5)), 1, :)
    policy = ConstantSavingsPolicy(params.alpha * params.beta)
    diagnostics, st = deterministic_bm_residual(policy, nothing, NamedTuple(), k; params)
    @test st == NamedTuple()
    @test maximum(abs.(diagnostics.residual)) < 1e-12
    @test diagnostics.next_capital ≈ bm_full_depreciation_policy(k, params)

    stochastic_params = BrockMirmanParams(delta = 0.1)
    states = vcat(fill(1.0, 1, 4), reshape(collect(range(1.0, 4.0; length = 4)), 1, :))
    gh = gauss_hermite_rule(3)
    stochastic, _ = stochastic_bm_residual(ConstantSavingsPolicy(0.25), nothing, NamedTuple(), states, gh; params = stochastic_params)
    @test isfinite(stochastic.loss)
    @test all(isfinite, stochastic.residual)

    model = make_mlp(2, (4,), 1; activation = NNlib.tanh)
    train_state = setup_training(rng_from_seed(21), model, Optimisers.Descent(0.001); parameter_type = Float64)
    stochastic_loss(model, ps, st, batch) = begin
        pieces, st_new = stochastic_bm_residual(model, ps, st, batch, gh; params = stochastic_params, transform = NNlib.sigmoid)
        return pieces.loss, st_new
    end
    metrics = train_step!(train_state, stochastic_loss, states; max_grad_norm = 10.0)
    @test isfinite(metrics.loss)
    @test isfinite(metrics.grad_norm)
end


@testset "IRBC helpers" begin
    params = IRBCParams(countries = 2)
    @test irbc_steady_state_capital(params) > 0
    states = irbc_sample_states(rng_from_seed(41), params, 5)
    @test size(states) == (4, 5)
    @test all(isfinite, states)

    smooth_model = make_mlp(4, (5,), 4; activation = NNlib.tanh)
    smooth_state = setup_training(rng_from_seed(42), smooth_model, Optimisers.Descent(0.001); parameter_type = Float64)
    smooth_pieces, smooth_st = irbc_smooth_residual(smooth_state.model, smooth_state.ps, smooth_state.st, states; params)
    @test smooth_st isa NamedTuple
    @test isfinite(smooth_pieces.loss)
    @test all(isfinite, smooth_pieces.euler)
    @test size(smooth_pieces.next_capital) == (2, 5)
    @test !all(iszero, smooth_pieces.resource)
    high_vol_params = IRBCParams(countries = 2, shock_std = 0.2)
    high_vol_pieces, _ = irbc_smooth_residual(smooth_state.model, smooth_state.ps, smooth_state.st, states; params = high_vol_params)
    @test smooth_pieces.euler != high_vol_pieces.euler

    smooth_loss(model, ps, st, batch) = begin
        pieces, st_new = irbc_smooth_residual(model, ps, st, batch; params)
        return pieces.loss, st_new
    end
    smooth_metrics = train_step!(smooth_state, smooth_loss, states; max_grad_norm = 10.0)
    @test isfinite(smooth_metrics.loss)

    irr_model = make_mlp(4, (5,), 6; activation = NNlib.tanh)
    irr_state = setup_training(rng_from_seed(43), irr_model, Optimisers.Descent(0.001); parameter_type = Float64)
    irr_pieces, _ = irbc_irreversible_residual(irr_state.model, irr_state.ps, irr_state.st, states; params)
    @test isfinite(irr_pieces.loss)
    @test all(isfinite, irr_pieces.complementarity)
    next_states, _ = irbc_zero_shock_step(irr_state.model, irr_state.ps, irr_state.st, states; params, irreversible = true)
    @test next_states[1:2, :] ≈ states[1:2, :] .^ params.shock_persistence
end

@testset "autodiff and PINN ODE helpers" begin
    f(x) = x^2 + sin(x)
    @test abs(ForwardDiff.derivative(f, 2.0) - (4.0 + cos(2.0))) < 1e-12

    gamma = 2.0
    utility(c) = c^(1 - gamma) / (1 - gamma)
    @test abs(ForwardDiff.derivative(utility, 1.7) - 1.7^(-gamma)) < 1e-12
    @test abs(second_derivative(utility, 1.7) + gamma * 1.7^(-gamma - 1)) < 1e-10

    model = AnalyticZeroBCODE()
    xs = reshape(collect(range(0.0, 1.0; length = 6)), 1, :)
    residual = zero_bc_ode_residual(model, nothing, NamedTuple(), xs)
    @test maximum(abs.(residual)) < 1e-12
    pieces, st = zero_bc_ode_loss(model, nothing, NamedTuple(), xs)
    @test st == NamedTuple()
    @test pieces.loss < 1e-12
    @test analytic_zero_bc_solution(0.5) ≈ 0.125

    pinn_model = make_mlp(1, (4,), 1; activation = NNlib.tanh)
    pinn_state = setup_training(rng_from_seed(31), pinn_model, Optimisers.Descent(0.001); parameter_type = Float64)
    pinn_batch = reshape([0.2, 0.5, 0.8], 1, :)
    pinn_loss(model, ps, st, batch) = begin
        loss_parts, st_new = zero_bc_tanh_mlp_loss(model, ps, st, batch)
        return loss_parts.loss, st_new
    end
    pde_grad = Zygote.gradient(pinn_state.ps) do ps
        zero_bc_tanh_mlp_loss(pinn_state.model, ps, pinn_state.st, pinn_batch)[1].pde_loss
    end[1]
    @test tree_sum_abs2(pde_grad) > 0
    pinn_metrics = train_step!(pinn_state, pinn_loss, pinn_batch; max_grad_norm = 10.0)
    @test isfinite(pinn_metrics.loss)
    @test isfinite(pinn_metrics.grad_norm)
end


@testset "Wave 4 PINN and continuous-time helpers" begin
    poisson_model = make_mlp(2, (4,), 1; activation = NNlib.tanh)
    poisson_state = setup_training(rng_from_seed(71), poisson_model, Optimisers.Descent(0.001); parameter_type = Float64)
    xy = [0.2 0.5 0.8; 0.3 0.6 0.7]
    value, grad, hess = tanh_mlp_scalar_derivatives(poisson_state.ps, [0.2, 0.3])
    @test isfinite(value)
    @test size(grad) == (2,)
    @test size(hess) == (2, 2)
    @test all(isfinite, hess)

    soft_pieces, _ = poisson2d_soft_loss(poisson_state.model, poisson_state.ps, poisson_state.st, xy, xy)
    hard_pieces, _ = poisson2d_hard_loss(poisson_state.model, poisson_state.ps, poisson_state.st, xy)
    @test isfinite(soft_pieces.loss)
    @test isfinite(hard_pieces.loss)
    hard_boundary_value = poisson2d_hard_value_derivatives(poisson_state.ps, [0.0, 0.4])[1]
    @test hard_boundary_value ≈ poisson2d_exact(0.0, 0.4)
    poisson_grad = Zygote.gradient(poisson_state.ps) do ps
        poisson2d_hard_loss(poisson_state.model, ps, poisson_state.st, xy)[1].loss
    end[1]
    @test tree_sum_abs2(poisson_grad) > 0

    cake_params = CakeEatingParams()
    cake_model = make_mlp(1, (4,), 1; activation = NNlib.tanh)
    cake_state = setup_training(rng_from_seed(72), cake_model, Optimisers.Descent(0.001); parameter_type = Float64)
    a_points = reshape(collect(range(cake_params.a_min, cake_params.a_max; length = 5)), 1, :)
    @test cake_eating_trial_value_derivative(cake_state.ps, cake_params.a_min; params = cake_params)[1] ≈ cake_eating_value_exact(cake_params.a_min; params = cake_params)
    @test cake_eating_trial_value_derivative(cake_state.ps, cake_params.a_max; params = cake_params)[1] ≈ cake_eating_value_exact(cake_params.a_max; params = cake_params)
    cake_pieces, _ = cake_eating_hjb_loss(cake_state.model, cake_state.ps, cake_state.st, a_points; params = cake_params)
    @test isfinite(cake_pieces.loss)
    cake_grad = Zygote.gradient(cake_state.ps) do ps
        cake_eating_hjb_loss(cake_state.model, ps, cake_state.st, a_points; params = cake_params)[1].loss
    end[1]
    @test tree_sum_abs2(cake_grad) > 0

    bs_params = BlackScholesParams()
    @test 0 < black_scholes_call_price(50.0, 1.0; params = bs_params) < 50.0
    @test 0 < black_scholes_delta(50.0, 1.0; params = bs_params) < 1
    bs_batch = (
        S_int = [10.0, 50.0], t_int = [0.2, 0.7],
        S_bc0 = [0.0], t_bc0 = [0.4],
        S_term = [40.0, 60.0], t_term = [1.0, 1.0],
        S_bcmax = [bs_params.s_max], t_bcmax = [0.5],
    )
    bs_pieces, _ = black_scholes_loss(poisson_state.model, poisson_state.ps, poisson_state.st, bs_batch; params = bs_params)
    @test isfinite(bs_pieces.loss)
    bs_grad = Zygote.gradient(poisson_state.ps) do ps
        black_scholes_loss(poisson_state.model, ps, poisson_state.st, bs_batch; params = bs_params)[1].loss
    end[1]
    @test tree_sum_abs2(bs_grad) > 0

    ct_params = CTAiyagariParams(n_quad = 8, n_a = 8, a_max = 4.0)
    quad = ct_aiyagari_trapezoid_rule(ct_params)
    density_norm = ct_aiyagari_normalized_density(zeros(length(quad.nodes), 2), quad.weights)
    @test density_norm.mass |> sum ≈ 1.0
    aggregates = ct_aiyagari_distribution_aggregates(density_norm.density, quad.nodes, quad.weights; params = ct_params)
    @test aggregates.mass ≈ 1.0
    @test ct_aiyagari_prices(aggregates.K, aggregates.L; params = ct_params).w > 0
    zero_drift = ct_aiyagari_kfe_drift(fill(0.5, length(quad.nodes), 2), zeros(length(quad.nodes), 2), quad.da; params = ct_params)
    @test maximum(abs.(zero_drift)) < 1e-12
    fd = ct_aiyagari_fd_inner(ct_params, collect(range(ct_params.a_min, ct_params.a_max; length = 6)), 0.8, 0.02; max_iter = 2)
    @test all(isfinite, fd.c)
    @test all(isfinite, fd.s)
    fd_solve = ct_aiyagari_fd_solve(ct_params; max_outer = 2, inner_max_iter = 2)
    @test isfinite(fd_solve.market_gap)
    @test isfinite(fd_solve.K_supply)
    @test isfinite(fd_solve.K_demand)
    @test fd_solve.outer_iterations <= 2

    w_model = make_mlp(3, (4,), 1; activation = NNlib.tanh)
    g_model = make_mlp(3, (4,), 1; activation = NNlib.tanh)
    ps_w, st_w = setup_model(rng_from_seed(73), w_model; parameter_type = Float64)
    ps_g, st_g = setup_model(rng_from_seed(74), g_model; parameter_type = Float64)
    models = (w = w_model, g = g_model)
    ps = (w = ps_w, g = ps_g)
    st = (w = st_w, g = st_g)
    a_col = collect(range(ct_params.a_min, ct_params.a_max; length = 5))
    ct_pieces, _ = ct_aiyagari_pinn_loss(models, ps, st, a_col; params = ct_params, kfe_form = :fv)
    @test isfinite(ct_pieces.loss)
    @test ct_pieces.mass ≈ 1.0
    @test ct_pieces.mass_balance_loss == 0
    @test ct_pieces.boundary_loss == 0
    ct_grad = Zygote.gradient(ps) do ps_local
        ct_aiyagari_pinn_loss(models, ps_local, st, a_col; params = ct_params, kfe_form = :fv)[1].loss
    end[1]
    @test tree_sum_abs2(ct_grad) > 0
    ct_strong, _ = ct_aiyagari_pinn_loss(models, ps, st, a_col; params = ct_params, kfe_form = :strong)
    @test isfinite(ct_strong.loss)
    @test ct_strong.mass_balance_loss >= 0
    @test ct_strong.boundary_loss >= 0
    strong_grad = Zygote.gradient(ps) do ps_local
        ct_aiyagari_pinn_loss(models, ps_local, st, a_col; params = ct_params, kfe_form = :strong)[1].loss
    end[1]
    @test tree_sum_abs2(strong_grad) > 0
    hjb_g_grad = Zygote.gradient(ps_g) do ps_g_local
        local_ps = (w = ps_w, g = ps_g_local)
        ct_aiyagari_pinn_loss(models, local_ps, st, a_col; params = ct_params, kfe_form = :fv)[1].hjb_loss
    end[1]
    @test isapprox(tree_sum_abs2(hjb_g_grad), 0.0; atol = 1e-20)
    fv_kfe_w_grad = Zygote.gradient(ps_w) do ps_w_local
        local_ps = (w = ps_w_local, g = ps_g)
        pieces = ct_aiyagari_pinn_loss(models, local_ps, st, a_col; params = ct_params, kfe_form = :fv)[1]
        pieces.kfe_loss + pieces.agg_loss
    end[1]
    @test isapprox(tree_sum_abs2(fv_kfe_w_grad), 0.0; atol = 1e-20)
    strong_kfe_w_grad = Zygote.gradient(ps_w) do ps_w_local
        local_ps = (w = ps_w_local, g = ps_g)
        pieces = ct_aiyagari_pinn_loss(models, local_ps, st, a_col; params = ct_params, kfe_form = :strong)[1]
        pieces.kfe_loss + pieces.flux_loss + pieces.agg_loss + pieces.boundary_loss
    end[1]
    @test isapprox(tree_sum_abs2(strong_kfe_w_grad), 0.0; atol = 1e-20)
    ct_train_state = setup_training(models, ps, st, Optimisers.Descent(1e-5))
    ct_loss(models, ps, st, batch) = begin
        loss_parts, st_new = ct_aiyagari_pinn_loss(models, ps, st, batch; params = ct_params, kfe_form = :fv)
        return loss_parts.loss, st_new
    end
    ct_metrics = train_step!(ct_train_state, ct_loss, a_col; max_grad_norm = 10.0)
    @test isfinite(ct_metrics.loss)
    @test isfinite(ct_metrics.grad_norm)
end


@testset "OLG helpers" begin
    params = AnalyticOLGParams()
    @test analytic_olg_state_dim(params) == 7
    @test analytic_olg_feature_dim(params) == 1 + length(params.tfp) + 2 + 5 + 4 * params.n_ages + length(params.tfp)
    @test sum(params.transition; dims = 2) ≈ ones(length(params.tfp), 1)
    rates = analytic_olg_closed_form_savings_rates(params)
    @test length(rates) == params.n_ages - 1
    @test all(0 .< rates .< 1)

    states = sample_analytic_olg_states(rng_from_seed(51), params, 8)
    @test size(states) == (analytic_olg_state_dim(params), 8)
    features = analytic_olg_features(states; params)
    @test size(features) == (analytic_olg_feature_dim(params), 8)
    exact = analytic_olg_exact_policy(states; params)
    @test maximum(abs.(analytic_olg_policy_error(exact.savings, states; params).relative)) < 1e-12

    analytic_model = ConstantOutputPolicy(params.n_ages - 1, 0.0)
    analytic_pieces, analytic_st = analytic_olg_residual(analytic_model, nothing, NamedTuple(), states; params)
    @test analytic_st == NamedTuple()
    @test isfinite(analytic_pieces.loss)
    @test size(analytic_pieces.euler) == (params.n_ages - 1, 8)
    @test all(isfinite, analytic_pieces.euler)
    analytic_lux = make_mlp(analytic_olg_feature_dim(params), (6,), params.n_ages - 1; activation = NNlib.tanh)
    analytic_train = setup_training(rng_from_seed(53), analytic_lux, Optimisers.Descent(0.0005); parameter_type = Float64)
    analytic_loss(model, ps, st, batch) = begin
        loss_parts, st2 = analytic_olg_residual(model, ps, st, batch; params)
        return loss_parts.loss, st2
    end
    analytic_metrics = train_step!(analytic_train, analytic_loss, states; max_grad_norm = 10.0)
    @test isfinite(analytic_metrics.loss)
    @test isfinite(analytic_metrics.grad_norm)
    next_states = analytic_olg_next_states(analytic_model, nothing, NamedTuple(), states, fill(1, 8); params)
    @test size(next_states) == size(states)

    @test benchmark_olg_state_dim(BenchmarkOLGParams()) == 113
    @test benchmark_olg_feature_dim(BenchmarkOLGParams()) == 240

    bench = BenchmarkOLGParams(n_ages = 8)
    bench_states = sample_benchmark_olg_states(rng_from_seed(52), bench, 5)
    @test size(bench_states) == (benchmark_olg_state_dim(bench), 5)
    @test size(benchmark_olg_features(bench_states; params = bench)) == (benchmark_olg_feature_dim(bench), 5)
    bench_model = ConstantOutputPolicy(4 * (bench.n_ages - 1) + 1, 0.0)
    raw, _ = bench_model(benchmark_olg_features(bench_states; params = bench), nothing, NamedTuple())
    policy = benchmark_olg_policy_from_raw(raw, bench_states; params = bench)
    @test all(policy.capital .> 0)
    @test all(policy.lambda .>= 0)
    @test all(policy.mu .>= 0)
    @test all(policy.price .> 0)
    @test minimum(policy.collateral) >= -1e-12
    bench_pieces, _ = benchmark_olg_residual(bench_model, nothing, NamedTuple(), bench_states; params = bench)
    @test isfinite(bench_pieces.loss)
    @test all(isfinite, bench_pieces.euler_capital)
    @test all(isfinite, bench_pieces.euler_bond)
    bench_lux = make_mlp(benchmark_olg_feature_dim(bench), (8,), 4 * (bench.n_ages - 1) + 1; activation = NNlib.tanh)
    bench_train = setup_training(rng_from_seed(54), bench_lux, Optimisers.Descent(0.0002); parameter_type = Float64)
    bench_loss(model, ps, st, batch) = begin
        loss_parts, st2 = benchmark_olg_residual(model, ps, st, batch; params = bench)
        return loss_parts.loss, st2
    end
    bench_metrics = train_step!(bench_train, bench_loss, bench_states; max_grad_norm = 10.0)
    @test isfinite(bench_metrics.loss)
    @test isfinite(bench_metrics.grad_norm)
end

@testset "Young histogram helpers" begin
    grid = collect(1.0:4.0)
    single = redistribute_mass(grid, 2.5, 1.0)
    @test sum(single) ≈ 1.0
    @test young_mean(grid, single) ≈ 2.5

    values = [1.25, 2.75, 3.25]
    masses = [0.2, 0.5, 0.3]
    hist = redistribute_distribution(grid, values, masses)
    @test young_mass(hist) ≈ sum(masses)
    @test young_mean(grid, hist) ≈ sum(values .* masses) / sum(masses)

    policy = k -> 0.7 * k + 0.3 * 2.5
    stepped = young_step(grid, hist, policy)
    @test young_mass(stepped) ≈ young_mass(hist)
    @test all(stepped .>= 0)

    G0 = [0.10 0.20 0.10 0.05; 0.05 0.15 0.20 0.15]
    pol = [0.9 1.3 1.7 2.1; 1.9 2.3 2.7 3.1]
    G1 = young_step(grid, G0, pol; transition = Matrix{Float64}(I, 2, 2))
    expected = [0.27 0.175 0.005 0.0; 0.005 0.21 0.32 0.015]
    @test G1 ≈ expected atol = 1e-12
    @test young_mass(G1) ≈ 1.0
    @test young_mean(grid, G1) ≈ 2.08

    trans = [0.9 0.1; 0.2 0.8]
    G2 = young_step(grid, G0, pol; transition = trans)
    @test young_mass(G2) ≈ young_mass(G0)
    flat = flatten_young_histogram(G2)
    @test unflatten_young_histogram(flat, 2, 4) == G2
    @test_throws ArgumentError validate_transition_matrix([0.6 0.6; 0.2 0.8])
end

@testset "sequence-space helpers" begin
    history = zeros(Float64, 1, 5, 3)
    flat = flatten_history(history)
    @test size(flat) == (5, 3)
    @test unflatten_history(flat, 1, 5) == history
    updated = prepend_history(history, reshape([1.0, 2.0, 3.0], 1, 3))
    @test updated[:, 1, :] == reshape([1.0, 2.0, 3.0], 1, 3)
    @test updated[:, 2:end, :] == history[:, 1:4, :]

    qh = quadrature_histories(history, [-1.0, 1.0])
    @test size(qh) == (1, 5, 3, 2)
    @test size(flatten_quadrature_histories(qh)) == (5, 6)
    encoded = encode_markov_history([:low, :high, :low], [:low, :high])
    @test size(encoded) == (2, 3, 1)
    @test sum(encoded; dims = 1) == ones(1, 3, 1)

    params = SequenceBrockMirmanParams()
    states = sequence_bm_initial_state(params; batch = 4)
    hist = zeros(Float64, 1, 6, 4)
    model = make_mlp(6, (5,), 1; activation = NNlib.tanh)
    state = setup_training(rng_from_seed(61), model, Optimisers.Descent(0.001); parameter_type = Float64)
    rule = gauss_hermite_rule(3)
    pieces, st_new = sequence_bm_residual(state.model, state.ps, state.st, states, hist, rule; params)
    @test st_new isa NamedTuple
    @test isfinite(pieces.loss)
    @test all(isfinite, pieces.euler)
    seq_loss(model, ps, st, batch) = begin
        loss_parts, st2 = sequence_bm_residual(model, ps, st, batch.states, batch.history, rule; params)
        return loss_parts.loss, st2
    end
    metrics = train_step!(state, seq_loss, (states = states, history = hist); max_grad_norm = 10.0)
    @test isfinite(metrics.loss)
    next_states, next_history, _ = sequence_bm_forward_step(state.model, state.ps, state.st, states, hist, zeros(1, 4); params)
    @test size(next_states) == size(states)
    @test size(next_history) == size(hist)

    irbc_params = IRBCParams(countries = 2)
    irbc_states = irbc_sample_states(rng_from_seed(62), irbc_params, 3)
    irbc_hist = sequence_irbc_history(irbc_params; history_length = 4, batch = 3)
    irbc_model = make_mlp(8, (5,), 4; activation = NNlib.tanh)
    irbc_state = setup_training(rng_from_seed(63), irbc_model, Optimisers.Descent(0.001); parameter_type = Float64)
    irbc_rule = irbc_stroud_rule(irbc_params)
    irbc_pieces, _ = sequence_irbc_residual(irbc_state.model, irbc_state.ps, irbc_state.st, irbc_states, irbc_hist, irbc_rule; params = irbc_params, irreversible = true)
    @test isfinite(irbc_pieces.loss)
    @test all(isfinite, irbc_pieces.euler)
    irbc_loss(model, ps, st, batch) = begin
        loss_parts, st2 = sequence_irbc_residual(model, ps, st, batch.states, batch.history, irbc_rule; params = irbc_params, irreversible = true)
        return loss_parts.loss, st2
    end
    irbc_metrics = train_step!(irbc_state, irbc_loss, (states = irbc_states, history = irbc_hist); max_grad_norm = 10.0)
    @test isfinite(irbc_metrics.loss)
    irbc_next, irbc_hist_next, _ = sequence_irbc_forward_step(irbc_state.model, irbc_state.ps, irbc_state.st, irbc_states, irbc_hist, zeros(irbc_params.countries, 3); params = irbc_params, irreversible = true)
    @test size(irbc_next) == size(irbc_states)
    @test size(irbc_hist_next) == size(irbc_hist)

    ks_params = SequenceKSParams(capital_grid = collect(range(0.0, 8.0; length = 10)))
    ks_history = sequence_ks_history(ks_params; history_length = 4)
    @test size(ks_history, 1) == length(ks_params.aggregate_z) + 1
    @test size(flatten_history(ks_history), 1) == size(ks_history, 1) * size(ks_history, 2)
    @test only(sequence_ks_current_z(ks_history, ks_params)) ≈ first(ks_params.aggregate_z)
    ks_dist = sequence_ks_initial_distribution(ks_params; K_target = 2.0)
    ks_model = make_mlp(size(flatten_history(ks_history), 1), (5,), length(ks_params.idio_income); activation = NNlib.tanh)
    ks_state = setup_training(rng_from_seed(64), ks_model, Optimisers.Descent(0.001); parameter_type = Float64)
    ks_pieces, _ = sequence_ks_residual(ks_state.model, ks_state.ps, ks_state.st, ks_history, ks_dist; params = ks_params)
    @test isfinite(ks_pieces.loss)
    @test all(isfinite, ks_pieces.euler)
    ks_loss(model, ps, st, batch) = begin
        loss_parts, st2 = sequence_ks_residual(model, ps, st, batch.history, batch.distribution; params = ks_params)
        return loss_parts.loss, st2
    end
    ks_metrics = train_step!(ks_state, ks_loss, (history = ks_history, distribution = ks_dist); max_grad_norm = 10.0)
    @test isfinite(ks_metrics.loss)
    ks_history_next, ks_dist_next, _ = sequence_ks_forward_step(ks_state.model, ks_state.ps, ks_state.st, ks_history, ks_dist, 2; params = ks_params)
    @test size(ks_history_next) == size(ks_history)
    @test young_mass(ks_dist_next) ≈ young_mass(ks_dist)
end

@testset "Wave 5 surrogate and GP helpers" begin
    design = black_scholes_design(rng_from_seed(81), 8)
    @test size(design.x) == (5, 8)
    @test size(normalize_box(design.x, design.normalizer)) == (5, 8)
    standardized = standardize_targets(design.y)
    @test maximum(abs.(unstandardize_targets(standardized.z, standardized) .- design.y)) < 1e-10
    @test black_scholes_call_price_5d(100.0, 100.0, 1.0, 0.2, 0.05) > 0

    x = reshape(collect(range(0.0, 1.0; length = 6)), 1, :)
    y = sin.(x)
    gp = fit_cholesky_gp(x, y; lengthscale = 0.4, noise = 1e-8)
    pred = gp_predict(gp, x)
    @test gp_rmse(gp, x, y) < 1e-5
    @test all(pred.variance .>= -1e-12)
    candidate = reshape([0.25, 0.5, 0.75], 1, :)
    @test 1 <= bal_next_index(gp, candidate) <= 3

    directions = [1.0 0.0; 0.0 1.0]
    xr = [0.0 1.0 0.0; 0.0 0.0 1.0]
    grads = radial_ridge_gradients(xr, directions)
    C = active_subspace_matrix(grads)
    as = active_subspace(C)
    @test issorted(as.values; rev = true)
    @test size(project_active_subspace(xr, as.vectors, 1)) == (1, 3)

    z = reshape([-1.0, 0.0, 1.0], 1, :)
    poly = polynomial_features(z; degree = 2)
    coef = ridge_fit(poly.features, 1 .+ 2 .* z .+ 3 .* z .^ 2; lambda = 0.0)
    @test ridge_predict(coef, poly.features) ≈ 1 .+ 2 .* z .+ 3 .* z .^ 2 atol = 1e-10
    @test encoder_widths(20, 1, 3)[[1, end]] == [20, 1]

    u = fill(0.5, 8, 2)
    phys = borehole_physical_from_unit(u)
    @test size(phys) == (8, 2)
    @test all(borehole_function(phys) .> 0)
end

@testset "Wave 5 SMM helpers" begin
    params = SMMBrockMirmanParams()
    ss = smm_bm_steady_state(params.beta; params)
    @test ss.capital > 0
    @test 0 < ss.savings < 1

    scalar_states = sample_smm_scalar_states(rng_from_seed(82), 5; params)
    joint_states = sample_smm_joint_states(rng_from_seed(83), 5; params)
    @test size(scalar_states) == (3, 5)
    @test size(joint_states) == (4, 5)

    rule = gauss_hermite_rule(3)
    scalar_model = make_mlp(3, (5,), 1; activation = NNlib.tanh, final_activation = NNlib.sigmoid)
    scalar_state = setup_training(rng_from_seed(84), scalar_model, Optimisers.Descent(0.001); parameter_type = Float64)
    scalar_pieces, scalar_st = smm_scalar_euler_loss(scalar_state.model, scalar_state.ps, scalar_state.st, scalar_states, rule; params)
    @test scalar_st isa NamedTuple
    @test isfinite(scalar_pieces.loss)
    @test all(isfinite, scalar_pieces.residual)
    smm_loss(model, ps, st, batch) = begin
        pieces, st_new = smm_scalar_euler_loss(model, ps, st, batch, rule; params)
        return pieces.loss, st_new
    end
    smm_metrics = train_step!(scalar_state, smm_loss, scalar_states; max_grad_norm = 10.0)
    @test isfinite(smm_metrics.loss)
    @test isfinite(smm_metrics.grad_norm)

    shocks1 = common_random_shocks(rng_from_seed(85), 14)
    shocks2 = common_random_shocks(rng_from_seed(85), 14)
    @test shocks1 == shocks2
    rhos = [0.8, 0.9, 0.95]
    sim_a = simulate_smm_scalar_rho(scalar_state.model, scalar_state.ps, scalar_state.st, rhos, shocks1; params, T_burn = 4, T_sim = 10)
    sim_b = simulate_smm_scalar_rho(scalar_state.model, scalar_state.ps, scalar_state.st, rhos, shocks2; params, T_burn = 4, T_sim = 10)
    @test sim_a.C == sim_b.C
    scalar_moments = smm_scalar_moments(sim_a.C, sim_a.I, sim_a.Y)
    @test size(scalar_moments) == (3, 4)
    @test all(isfinite, scalar_moments)
    target = scalar_moments[2, :]
    estimate = smm_grid_estimate(rhos, scalar_moments, target; mask = [true, true, true, false])
    @test estimate.index == 2
    @test estimate.value ≈ 0.0 atol = 1e-12
    @test all(isfinite, smm_moment_sensitivity_1d(scalar_moments, rhos, 2))

    joint_model = make_mlp(4, (5,), 1; activation = NNlib.tanh, final_activation = NNlib.sigmoid)
    joint_state = setup_training(rng_from_seed(86), joint_model, Optimisers.Descent(0.001); parameter_type = Float64)
    joint_pieces, _ = smm_joint_euler_loss(joint_state.model, joint_state.ps, joint_state.st, joint_states, rule; params)
    @test isfinite(joint_pieces.loss)
    joint_loss(model, ps, st, batch) = begin
        pieces, st_new = smm_joint_euler_loss(model, ps, st, batch, rule; params)
        return pieces.loss, st_new
    end
    joint_metrics = train_step!(joint_state, joint_loss, joint_states; max_grad_norm = 10.0)
    @test isfinite(joint_metrics.loss)
    @test isfinite(joint_metrics.grad_norm)
    theta = [0.94 0.96 0.98; 0.80 0.90 0.95]
    joint_sim = simulate_smm_joint_theta(joint_state.model, joint_state.ps, joint_state.st, theta, shocks1; params, T_burn = 4, T_sim = 10)
    joint_moments = smm_joint_moments(joint_sim.C, joint_sim.I, joint_sim.Y)
    @test size(joint_moments) == (3, 4)
    @test all(isfinite, joint_moments)

    beta_grid = [0.94, 0.96, 0.98]
    rho_grid = [0.80, 0.90, 0.95]
    moment_cube = zeros(3, 3, 4)
    for i in 1:3, j in 1:3
        moment_cube[i, j, :] = [beta_grid[j], rho_grid[i], beta_grid[j]^2, rho_grid[i]^2]
    end
    J = smm_moment_jacobian_2d(moment_cube, beta_grid, rho_grid, 2, 2)
    @test size(J) == (4, 2)
    ids = smm_identification_svd(J; mask = trues(4))
    @test length(ids.singular_values) == 2
    @test all(ids.singular_values .>= 0)
end

@testset "Wave 6 climate helpers" begin
    dice = simulate_dice_climate_exercise()
    @test dice.comparison_year == 2100.0
    @test dice.avoided_warming > 0
    @test dice.avoided_damages > 0
    @test all(isfinite, dice.temperature_bau)
    @test size(dice.carbon_bau, 1) == 3

    params = CDICEParams()
    @test cdice_tau_to_time(cdice_time_to_tau(85.0; params); params) ≈ 85.0
    @test cdice_stationary_z_std(params) > params.sigma_z
    initial = cdice_initial_state(; params)
    normed = cdice_normalize_states(initial; params)
    @test cdice_denormalize_states(normed; params) ≈ initial

    teaching = CDICETeachingPolicy(; params)
    det_states = sample_cdice_states(rng_from_seed(91), 6; params)
    det_pieces, det_st = deterministic_cdice_residual(teaching, nothing, NamedTuple(), det_states; params)
    @test det_st == NamedTuple()
    @test isfinite(det_pieces.loss)
    @test size(det_pieces.residuals) == (8, 6)
    det_path = simulate_cdice_path(teaching; params, periods = 100)
    @test det_path.scc[1] > 0
    @test det_path.TAT[86] > 0
    @test all(isfinite, det_path.scc)

    rule = gauss_hermite_rule(5)
    stochastic_teaching = CDICETeachingPolicy(; params, stochastic = true)
    stoch_states = sample_cdice_states(rng_from_seed(92), 5; params, stochastic = true)
    stoch_pieces, _ = stochastic_cdice_residual(stochastic_teaching, nothing, NamedTuple(), stoch_states, rule; params)
    @test isfinite(stoch_pieces.loss)
    @test size(stoch_pieces.residuals) == (8, 5)
    @test all(isfinite, stoch_pieces.z_mean_next)
    mc_small = cdice_monte_carlo_paths(stochastic_teaching; params, n_paths = 4, periods = 3, seed = 77)
    @test maximum(mc_small.z[:, 1]) - minimum(mc_small.z[:, 1]) > 0

    det_model = make_mlp(7, (6,), 8; activation = NNlib.relu)
    det_state = setup_training(rng_from_seed(93), det_model, Optimisers.Descent(1e-5); parameter_type = Float64)
    climate_loss(model, ps, st, batch) = begin
        pieces, st_new = deterministic_cdice_residual(model, ps, st, batch; params)
        return pieces.loss, st_new
    end
    metrics = train_step!(det_state, climate_loss, det_states; max_grad_norm = 10.0)
    @test isfinite(metrics.loss)
    @test isfinite(metrics.grad_norm)
end

include("wave1_regression_guards.jl")

