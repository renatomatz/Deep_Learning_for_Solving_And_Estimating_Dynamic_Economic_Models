using Test

using DLEFJulia
using ForwardDiff
using LinearAlgebra: I, norm
using Lux
using NNlib
using Optimisers
using StableRNGs
using Statistics: mean

repo_root = normpath(joinpath(@__DIR__, "..", ".."))

struct ConceptCheck
    name::String
    python_markers::Vector{String}
    julia_markers::Vector{String}
end

struct NotebookPair
    lecture::String
    python::String
    julia::String
    concepts::Vector{ConceptCheck}
end

struct MissingTranslation
    lecture::String
    python::String
    reason::String
end

markers(x::AbstractString) = [String(x)]
markers(x) = String.(collect(x))

function concept(name, python_markers, julia_markers = python_markers)
    return ConceptCheck(String(name), markers(python_markers), markers(julia_markers))
end

function pair(lecture, python, julia, concepts...)
    return NotebookPair(String(lecture), String(python), String(julia), ConceptCheck[concepts...])
end

const notebook_pairs = NotebookPair[
    pair("Lecture 02",
        "lectures/lecture_02_intro_deep_learning/code/lecture_02_01_BasicML_intro.ipynb",
        "lectures/lecture_02_intro_deep_learning/code_julia/lecture_02_01_BasicML_intro_Lux.jl",
        concept("basic supervised learning", ["BasicML", "Basic ML", "supervised"], ["BasicML", "supervised", "mse_loss"]),
        concept("Lux boundary", ["Keras", "neural network", "TensorFlow"], ["Lux", "make_mlp", "feature-by-batch"])),
    pair("Lecture 02",
        "lectures/lecture_02_intro_deep_learning/code/lecture_02_02_GradientDescent_and_StochasticGradientDescent.ipynb",
        "lectures/lecture_02_intro_deep_learning/code_julia/lecture_02_02_GradientDescent_and_StochasticGradientDescent_Lux.jl",
        concept("gradient descent", ["Gradient Descent", "gradient descent"], ["gradient_descent_1d", "batch_gradient_descent"]),
        concept("stochastic updates", ["Stochastic", "SGD", "stochastic"], ["stochastic_gradient_descent", "rng"])),
    pair("Lecture 02",
        "lectures/lecture_02_intro_deep_learning/code/lecture_02_03_Double_Descent.ipynb",
        "lectures/lecture_02_intro_deep_learning/code_julia/lecture_02_03_Double_Descent_Lux.jl",
        concept("double descent", "Double Descent"),
        concept("random features", ["random feature", "RFF", "features"], ["rff_matrix", "pinv"])),
    pair("Lecture 02",
        "lectures/lecture_02_intro_deep_learning/code/lecture_02_04_Gentle_DNN.ipynb",
        "lectures/lecture_02_intro_deep_learning/code_julia/lecture_02_04_Gentle_DNN_Lux.jl",
        concept("dense neural network", ["DNN", "Sequential", "Keras"], ["Gentle DNN", "Lux", "make_mlp"]),
        concept("classification and regression", ["classification", "regression"], ["classification", "regression", "cross_entropy_loss"])),
    pair("Lecture 02",
        "lectures/lecture_02_intro_deep_learning/code/lecture_02_05_Tensorboard.ipynb",
        "lectures/lecture_02_intro_deep_learning/code_julia/lecture_02_05_Training_Instrumentation_Lux.jl",
        concept("training instrumentation", ["TensorBoard", "callback", "checkpoint"], ["TensorBoard", "checkpoint", "metrics_table"]),
        concept("validation monitoring", ["validation", "loss"], ["validation_loss", "best_checkpoint"])),
    pair("Lecture 02",
        "lectures/lecture_02_intro_deep_learning/code/lecture_02_06_PyTorch_intro.ipynb",
        "lectures/lecture_02_intro_deep_learning/code_julia/lecture_02_06_Lux_Training_Fundamentals.jl",
        concept("framework introduction", ["PyTorch", "Module", "tensor"], ["PyTorch", "Lux", "model, ps, st"]),
        concept("training fundamentals", ["optimizer", "gradient"], ["Optimisers", "train_step!"])),
    pair("Lecture 02",
        "lectures/lecture_02_intro_deep_learning/code/lecture_02_07_Genz_Approximation_and_Loss_Functions.ipynb",
        "lectures/lecture_02_intro_deep_learning/code_julia/lecture_02_07_Genz_Approximation_and_Loss_Functions_Lux.jl",
        concept("Genz approximation", "Genz"),
        concept("loss kernels", ["MSE", "MAE", "Huber", "loss"], ["mse_loss", "mae_loss", "huber_loss", "loss"])),
    pair("Lecture 02",
        "lectures/lecture_02_intro_deep_learning/code/lecture_02_08_MLP_LSTM_Transformer_Edgeworth_Cycles.ipynb",
        "lectures/lecture_02_intro_deep_learning/code_julia/lecture_02_08_MLP_LSTM_Transformer_Edgeworth_Cycles_Lux.jl",
        concept("sequence models", ["MLP", "LSTM", "Transformer"], ["MLP", "LSTM", "Transformer"]),
        concept("Edgeworth cycles", "Edgeworth")),
    pair("Lecture 02",
        "lectures/lecture_02_intro_deep_learning/code/lecture_02_09_Transformer_InContext_AR1.ipynb",
        "lectures/lecture_02_intro_deep_learning/code_julia/lecture_02_09_Transformer_InContext_AR1_Lux.jl",
        concept("transformer context", "Transformer"),
        concept("AR1 process", ["AR1", "AR(1)", "autoregressive"], ["AR1", "AR(1)", "autoregressive"])),
    pair("Lecture 03",
        "lectures/lecture_03_deep_equilibrium_nets/code/lecture_03_01_Brock_Mirman_1972_DEQN.ipynb",
        "lectures/lecture_03_deep_equilibrium_nets/code_julia/lecture_03_01_Brock_Mirman_1972_DEQN_Lux.jl",
        concept("Brock-Mirman", ["Brock", "Mirman"], ["BrockMirmanParams", "deterministic_bm_residual"]),
        concept("Euler residual", ["Euler", "residual"], ["residual", "Euler"])),
    pair("Lecture 03",
        "lectures/lecture_03_deep_equilibrium_nets/code/lecture_03_02_Brock_Mirman_Uncertainty_DEQN.ipynb",
        "lectures/lecture_03_deep_equilibrium_nets/code_julia/lecture_03_02_Brock_Mirman_Uncertainty_DEQN_Lux.jl",
        concept("stochastic Brock-Mirman", ["Brock", "Uncertainty", "stochastic"], ["stochastic_bm_residual", "BrockMirmanParams"]),
        concept("Gauss-Hermite expectation", ["Hermite", "Gauss"], ["gauss_hermite_rule", "Hermite"])),
    pair("Lecture 03",
        "lectures/lecture_03_deep_equilibrium_nets/code/lecture_03_03_DEQN_Exercises_Blanks.ipynb",
        "lectures/lecture_03_deep_equilibrium_nets/code_julia/lecture_03_03_DEQN_Exercises_Blanks_Lux.jl",
        concept("exercise blanks", ["Exercise", "TODO", "blank"], ["Exercise", "TODO", "fill in"]),
        concept("Brock-Mirman scaffold", ["Brock", "Mirman"], ["BrockMirmanParams", "gauss_hermite_rule"])),
    pair("Lecture 03",
        "lectures/lecture_03_deep_equilibrium_nets/code/lecture_03_04_DEQN_Exercises_Solutions.ipynb",
        "lectures/lecture_03_deep_equilibrium_nets/code_julia/lecture_03_04_DEQN_Exercises_Solutions_Lux.jl",
        concept("exercise solutions", ["Exercise", "solution"], ["solution", "labor_deqn_residual"]),
        concept("labor and complementarity", ["labor", "Fischer"], ["labor", "fischer_burmeister"])),
    pair("Lecture 03",
        "lectures/lecture_03_deep_equilibrium_nets/code/lecture_03_05_StochasticBM_LossComparison.ipynb",
        "lectures/lecture_03_deep_equilibrium_nets/code_julia/lecture_03_05_StochasticBM_LossComparison_Lux.jl",
        concept("stochastic BM loss comparison", ["Stochastic", "Loss", "Brock"], ["stochastic_bm_residual", "LOSS_KERNELS"]),
        concept("common random numbers", ["common random", "seed"], ["common random", "SEED"])),
    pair("Lecture 04",
        "lectures/lecture_04_irbc_with_deqns/code/lecture_04_01_IRBC_DEQN_smooth.ipynb",
        "lectures/lecture_04_irbc_with_deqns/code_julia/lecture_04_01_IRBC_DEQN_smooth_Lux.jl",
        concept("smooth IRBC", ["IRBC", "smooth"], ["IRBCParams", "irbc_smooth_residual"]),
        concept("Stroud expectations", ["Stroud", "monomial"], ["Stroud", "irbc_stroud_rule"])),
    pair("Lecture 04",
        "lectures/lecture_04_irbc_with_deqns/code/lecture_04_02_IRBC_DEQN_irreversible.ipynb",
        "lectures/lecture_04_irbc_with_deqns/code_julia/lecture_04_02_IRBC_DEQN_irreversible_Lux.jl",
        concept("irreversible IRBC", ["IRBC", "irreversible"], ["irreversible_residual", "build_irreversible_policy_network", "raw_investment"]),
        concept("KKT complementarity", ["KKT", "Fischer", "complementarity"], ["fischer_burmeister", "complementarity"])),
    pair("Lecture 05",
        "lectures/lecture_05_nas_loss_normalization/code/lecture_05_02_NAS_Random_Search_10D.ipynb",
        "lectures/lecture_05_nas_loss_normalization/code_julia/lecture_05_02_NAS_Random_Search_10D_Lux.jl",
        concept("NAS random search", ["NAS", "Random Search"], ["NAS", "Random Search"]),
        concept("10D search", ["10D", "10-D", "ten"], ["10-D", "target_10d", "train_candidate"])),
    pair("Lecture 05",
        "lectures/lecture_05_nas_loss_normalization/code/lecture_05_03_NAS_RandomSearch_Hyperband.ipynb",
        "lectures/lecture_05_nas_loss_normalization/code_julia/lecture_05_03_NAS_RandomSearch_Hyperband_Lux.jl",
        concept("Hyperband", "Hyperband", ["SHA", "Successive Halving", "sha_records"]),
        concept("rungs and survivors", ["rung", "survivor"], ["rung", "survivors"])),
    pair("Lecture 05",
        "lectures/lecture_05_nas_loss_normalization/code/lecture_05_04_Loss_Normalization.ipynb",
        "lectures/lecture_05_nas_loss_normalization/code_julia/lecture_05_04_Loss_Normalization_Lux.jl",
        concept("loss normalization", ["Multi-Component Loss", "Loss Balancing", "ReLoBRaLo"], ["Loss Normalization", "inverse_loss_weights"]),
        concept("adaptive weights", ["SoftAdapt", "ReLoBRaLo", "weights"], ["softadapt_weights", "relobralo_weights", "weights"])),
    pair("Lecture 05",
        "lectures/lecture_05_nas_loss_normalization/code/lecture_05_05_IRBC_Exercise.ipynb",
        "lectures/lecture_05_nas_loss_normalization/code_julia/lecture_05_05_IRBC_Exercise_Lux.jl",
        concept("IRBC exercise", ["IRBC", "Exercise"], ["IRBCParams", "Exercise"]),
        concept("loss balancing", ["loss", "balancing", "weights"], ["loss", "weights", "equal_loss_weights"])),
    pair("Lecture 07",
        "lectures/lecture_07_autodiff_for_deqns/code/lecture_07_01_AutoDiff_Analytical_Examples.ipynb",
        "lectures/lecture_07_autodiff_for_deqns/code_julia/lecture_07_01_AutoDiff_Analytical_Examples_Lux.jl",
        concept("analytical autodiff", ["AutoDiff", "Analytical"], ["AutoDiff", "ForwardDiff"]),
        concept("slot gradients", ["gradient", "derivative"], ["gradient", "derivative"])),
    pair("Lecture 07",
        "lectures/lecture_07_autodiff_for_deqns/code/lecture_07_02_Brock_Mirman_AutoDiff_DEQN.ipynb",
        "lectures/lecture_07_autodiff_for_deqns/code_julia/lecture_07_02_Brock_Mirman_AutoDiff_DEQN_Lux.jl",
        concept("Brock-Mirman autodiff", ["Brock", "Mirman", "AutoDiff"], ["bm_payoff", "autodiff_bm_residual", "ForwardDiff"]),
        concept("autodiff training", ["GradientTape", "Adam", "training"], ["Optimisers.Adam", "train_step!", "policy_relative_l2"])),
    pair("Lecture 07",
        "lectures/lecture_07_autodiff_for_deqns/code/lecture_07_03_Brock_Mirman_Uncertainty_AutoDiff_DEQN.ipynb",
        "lectures/lecture_07_autodiff_for_deqns/code_julia/lecture_07_03_Brock_Mirman_Uncertainty_AutoDiff_DEQN_Lux.jl",
        concept("stochastic Brock-Mirman autodiff", ["Brock", "Uncertainty", "AutoDiff"], ["autodiff_stochastic_residual", "Pi", "slot_gradient_errors"]),
        concept("common-shock expectation", ["Gauss", "Hermite", "shock"], ["gauss_hermite_rule", "quadrature_checks", "simulate_periods"])),
    pair("Lecture 07",
        "lectures/lecture_07_autodiff_for_deqns/code/lecture_07_04_IRBC_AutoDiff_DEQN.ipynb",
        "lectures/lecture_07_autodiff_for_deqns/code_julia/lecture_07_04_IRBC_AutoDiff_DEQN_Lux.jl",
        concept("IRBC autodiff", ["IRBC", "AutoDiff"], ["pi_contribution", "autodiff_irbc_residual", "slot_gradient_error"]),
        concept("approach A and B training", ["Approach A", "Approach B", "Adam"], ["grad_autodiff", "train_approach_A", "train_approach_B", "training_diagnostics"])),
    pair("Lecture 08",
        "lectures/lecture_08_olg_models_deqns/code/lecture_08_07_OLG_Analytic_DEQN_exogenous.ipynb",
        "lectures/lecture_08_olg_models_deqns/code_julia/lecture_08_07_OLG_Analytic_DEQN_exogenous_Lux.jl",
        concept("analytic OLG exogenous", ["OLG", "Analytic", "exogenous"], ["AnalyticOLGParams", "exogenous"]),
        concept("closed form validation", ["closed form", "closed-form"], ["analytic_olg_exact_policy", "policy_error"])),
    pair("Lecture 08",
        "lectures/lecture_08_olg_models_deqns/code/lecture_08_08_OLG_Analytic_DEQN_persistent.ipynb",
        "lectures/lecture_08_olg_models_deqns/code_julia/lecture_08_08_OLG_Analytic_DEQN_persistent_Lux.jl",
        concept("analytic OLG persistent", ["OLG", "Analytic", "persistent"], ["AnalyticOLGParams", "Persistent"]),
        concept("persistent simulation", ["persistent", "simulation"], ["persistent", "analytic_olg_next_states"])),
    pair("Lecture 08",
        "lectures/lecture_08_olg_models_deqns/code/lecture_08_09_OLG_Benchmark_DEQN_exogenous.ipynb",
        "lectures/lecture_08_olg_models_deqns/code_julia/lecture_08_09_OLG_Benchmark_DEQN_exogenous_Lux.jl",
        concept("benchmark OLG exogenous", ["OLG", "Benchmark", "exogenous"], ["BenchmarkOLGParams", "exogenous"]),
        concept("borrowing and collateral", ["borrowing", "collateral"], ["collateral", "benchmark_olg_residual"])),
    pair("Lecture 08",
        "lectures/lecture_08_olg_models_deqns/code/lecture_08_10_OLG_Benchmark_DEQN_persistent.ipynb",
        "lectures/lecture_08_olg_models_deqns/code_julia/lecture_08_10_OLG_Benchmark_DEQN_persistent_Lux.jl",
        concept("benchmark OLG persistent", ["OLG", "Benchmark", "persistent"], ["BenchmarkOLGParams", "Persistent"]),
        concept("rolled state cloud", ["persistent", "simulation", "cloud"], ["cloud", "benchmark_olg_next_states"])),
    pair("Lecture 08",
        "lectures/lecture_08_olg_models_deqns/code/lecture_08_11_OLG_Exercise.ipynb",
        "lectures/lecture_08_olg_models_deqns/code_julia/lecture_08_11_OLG_Exercise_Lux.jl",
        concept("OLG exercise", ["Exercise", "savings", "lifecycle"], ["exercise_savings_rates", "simulate_lifecycle", "savings_rates"]),
        concept("closed-form validation", ["closed form", "closed-form"], ["exact_policy_error_max", "validation_loss", "ConstantSavingsRaw"])),
    pair("Lecture 09",
        "lectures/lecture_09_heterogeneous_agents_youngs_method/code/lecture_09_10_Youngs_Method_Examples.ipynb",
        "lectures/lecture_09_heterogeneous_agents_youngs_method/code_julia/lecture_09_10_Youngs_Method_Examples_Lux.jl",
        concept("Young method", ["Young", "histogram"], ["young_step", "histogram"]),
        concept("mass conservation", ["mass", "mean"], ["young_mass", "young_mean"])),
    pair("Lecture 09",
        "lectures/lecture_09_heterogeneous_agents_youngs_method/code/lecture_09_11_Continuum_of_Agents_DEQN.ipynb",
        "lectures/lecture_09_heterogeneous_agents_youngs_method/code_julia/lecture_09_11_Continuum_of_Agents_DEQN_Lux.jl",
        concept("continuum agents", ["Continuum", "agents"], ["Continuum", "young_step"]),
        concept("Euler and market clearing", ["Euler", "market"], ["euler", "bond_market"])),
    pair("Lecture 09",
        "lectures/lecture_09_heterogeneous_agents_youngs_method/code/lecture_09_12_KrusellSmith_DeepLearning.ipynb",
        "lectures/lecture_09_heterogeneous_agents_youngs_method/code_julia/lecture_09_12_KrusellSmith_DeepLearning_Lux.jl",
        concept("Krusell-Smith deep learning", ["Krusell", "Smith", "Deep"], ["Krusell-Smith", "ks_residual", "policy_features"]),
        concept("running panel phase B", ["Phase B", "panel", "K_next"], ["initial_panel", "advance_panel", "panel_log", "phase_b_tail"])),
    pair("Lecture 10",
        "lectures/lecture_10_sequence_space_deqns/code/lecture_10_05_SequenceSpace_BrockMirman.ipynb",
        "lectures/lecture_10_sequence_space_deqns/code_julia/lecture_10_05_SequenceSpace_BrockMirman_Lux.jl",
        concept("sequence-space Brock-Mirman", ["Sequence", "Brock"], ["SequenceBrockMirmanParams", "sequence_bm_residual"]),
        concept("history tensors", ["history", "histories"], ["history", "flatten_history"])),
    pair("Lecture 10",
        "lectures/lecture_10_sequence_space_deqns/code/lecture_10_05b_SequenceSpace_IRBC.ipynb",
        "lectures/lecture_10_sequence_space_deqns/code_julia/lecture_10_05b_SequenceSpace_IRBC_Lux.jl",
        concept("sequence-space IRBC", ["Sequence", "IRBC"], ["py_sequence_irbc_residual", "n_shocks", "quadrature_shape"]),
        concept("shock histories", ["history", "shock"], ["history_shape", "lux_history_shape", "advance_history"])),
    pair("Lecture 10",
        "lectures/lecture_10_sequence_space_deqns/code/lecture_10_06_SequenceSpace_KrusellSmith.ipynb",
        "lectures/lecture_10_sequence_space_deqns/code_julia/lecture_10_06_SequenceSpace_KrusellSmith_Lux.jl",
        concept("sequence-space Krusell-Smith", ["Krusell", "Smith"], ["SequenceKSParams", "Krusell"]),
        concept("Young propagation", ["Young", "distribution"], ["young_step", "distribution"])),
    pair("Lecture 10",
        "lectures/lecture_10_sequence_space_deqns/code/lecture_10_KrusellSmith_Tutorial_CPU.ipynb",
        "lectures/lecture_10_sequence_space_deqns/code_julia/lecture_10_KrusellSmith_Tutorial_CPU_Lux.jl",
        concept("Krusell-Smith CPU tutorial", ["Krusell", "Tutorial", "CPU"], ["Krusell", "Tutorial", "CPU"]),
        concept("distribution aggregates", ["distribution", "aggregate"], ["sequence_ks_distribution_aggregates", "distribution"])),
    pair("Lecture 11",
        "lectures/lecture_11_pinns/code/lecture_11_01_ODE_PINN_ZeroBCs.ipynb",
        "lectures/lecture_11_pinns/code_julia/lecture_11_01_ODE_PINN_ZeroBCs_Lux.jl",
        concept("ODE PINN", ["ODE", "PINN"], ["ODE", "zero_bc_tanh_mlp_loss"]),
        concept("zero boundary conditions", ["zero", "boundary"], ["boundary", "analytic_zero_bc_solution"])),
    pair("Lecture 11",
        "lectures/lecture_11_pinns/code/lecture_11_02_ODE_PINN_SoftVsHardBCs.ipynb",
        "lectures/lecture_11_pinns/code_julia/lecture_11_02_ODE_PINN_SoftVsHardBCs_Lux.jl",
        concept("soft and hard ODE PINNs", ["Soft", "Hard", "boundary"], ["soft_bc_ode_loss", "hard_bc_ode_loss", "hard_trial_values"]),
        concept("manufactured ODE solution", ["exact", "solution"], ["ode_exact_solution", "ode_boundary_lift", "ode_bubble"])),
    pair("Lecture 11",
        "lectures/lecture_11_pinns/code/lecture_11_03_PDE_PINN_Poisson2D.ipynb",
        "lectures/lecture_11_pinns/code_julia/lecture_11_03_PDE_PINN_Poisson2D_Lux.jl",
        concept("Poisson PINN", ["Poisson", "PDE"], ["poisson2d_exact", "poisson2d_hard_loss"]),
        concept("soft and hard boundaries", ["soft", "hard", "boundary"], ["soft", "hard", "boundary"])),
    pair("Lecture 11",
        "lectures/lecture_11_pinns/code/lecture_11_04_Cake_Eating_HJB_PINN.ipynb",
        "lectures/lecture_11_pinns/code_julia/lecture_11_04_Cake_Eating_HJB_PINN_Lux.jl",
        concept("cake-eating HJB", ["Cake", "HJB"], ["CakeEatingParams", "cake_eating_hjb_loss"]),
        concept("analytic value", ["exact", "analytic"], ["cake_eating_value_exact", "cake_eating_consumption_exact"])),
    pair("Lecture 11",
        "lectures/lecture_11_pinns/code/lecture_11_05_Black_Scholes_PINN.ipynb",
        "lectures/lecture_11_pinns/code_julia/lecture_11_05_Black_Scholes_PINN_Lux.jl",
        concept("Black-Scholes PINN", ["Black", "Scholes", "PINN"], ["BlackScholesParams", "black_scholes_loss"]),
        concept("terminal and boundary conditions", ["terminal", "boundary"], ["terminal", "boundary"])),
    pair("Lecture 13",
        "lectures/lecture_13_continuous_time_ha_numerics/code/lecture_13_08_Aiyagari_Continuous_Time_FD_and_PINN_PyTorch.ipynb",
        "lectures/lecture_13_continuous_time_ha_numerics/code_julia/lecture_13_08_Aiyagari_Continuous_Time_FD_and_PINN_Lux.jl",
        concept("continuous-time Aiyagari", ["Aiyagari", "Continuous"], ["CTAiyagariParams", "Aiyagari"]),
        concept("FD and PINN validation", ["FD", "PINN", "KFE"], ["ct_aiyagari_fd_solve", "ct_aiyagari_pinn_loss", "KFE"])),
    pair("Lecture 14",
        "lectures/lecture_14_surrogates_and_gps/code/lecture_14_01_Surrogate_Primer.ipynb",
        "lectures/lecture_14_surrogates_and_gps/code_julia/lecture_14_01_Surrogate_Primer_Lux.jl",
        concept("surrogate primer", "Surrogate"),
        concept("Black-Scholes surface", ["Black", "Scholes"], ["black_scholes_call_price_5d", "Black-Scholes"])),
    pair("Lecture 14",
        "lectures/lecture_14_surrogates_and_gps/code/lecture_14_02_GP_and_BAL.ipynb",
        "lectures/lecture_14_surrogates_and_gps/code_julia/lecture_14_02_GP_and_BAL_Lux.jl",
        concept("GP and BAL", ["GP", "BAL", "Gaussian"], ["fit_cholesky_gp", "bal_next_index", "Gaussian"]),
        concept("posterior variance", ["variance", "posterior"], ["variance", "gp_predict"])),
    pair("Lecture 14",
        "lectures/lecture_14_surrogates_and_gps/code/lecture_14_04_GP_Value_Function_Iteration.ipynb",
        "lectures/lecture_14_surrogates_and_gps/code_julia/lecture_14_04_GP_Value_Function_Iteration_Lux.jl",
        concept("GP value function iteration", ["GP", "Value", "Iteration"], ["GP", "BrockMirmanParams", "value"]),
        concept("policy check", ["policy", "consumption"], ["policy", "consumption_error"])),
    pair("Lecture 14",
        "lectures/lecture_14_surrogates_and_gps/code/lecture_14_05_Active_Subspace_2D.ipynb",
        "lectures/lecture_14_surrogates_and_gps/code_julia/lecture_14_05_Active_Subspace_2D_Lux.jl",
        concept("active subspace 2D", ["Active", "Subspace", "2D"], ["active_subspace", "2D"]),
        concept("polynomial ridge", ["ASGP", "active subspace", "exponential"], ["fit_active_subspace_surrogate", "ridge"])),
    pair("Lecture 14",
        "lectures/lecture_14_surrogates_and_gps/code/lecture_14_06_Active_Subspace_10D.ipynb",
        "lectures/lecture_14_surrogates_and_gps/code_julia/lecture_14_06_Active_Subspace_10D_Lux.jl",
        concept("active subspace 10D", ["Active", "Subspace", "10D"], ["active_subspace", "10D"]),
        concept("ridge target", ["ridge", "exponential"], ["exponential ridge", "fit_active_subspace_surrogate", "fvec"])),
    pair("Lecture 14",
        "lectures/lecture_14_surrogates_and_gps/code/lecture_14_07_Active_Subspace_Nonlinear.ipynb",
        "lectures/lecture_14_surrogates_and_gps/code_julia/lecture_14_07_Active_Subspace_Nonlinear_Lux.jl",
        concept("nonlinear active subspace", ["Active", "Subspace", "Nonlinear"], ["active_subspace", "Nonlinear"]),
        concept("dimension comparison", ["dimension", "error"], ["dims", "rel_errors"])),
    pair("Lecture 14",
        "lectures/lecture_14_surrogates_and_gps/code/lecture_14_08_Deep_Kernel_Learning.ipynb",
        "lectures/lecture_14_surrogates_and_gps/code_julia/lecture_14_08_Deep_Kernel_Learning_Lux.jl",
        concept("deep kernel learning", ["Deep", "Kernel"], ["Deep", "Kernel"]),
        concept("feature map and GP", ["feature", "GP"], ["feature", "fit_cholesky_gp"])),
    pair("Lecture 14",
        "lectures/lecture_14_surrogates_and_gps/code/lecture_14_09_Deep_Active_Subspace_Ridge.ipynb",
        "lectures/lecture_14_surrogates_and_gps/code_julia/lecture_14_09_Deep_Active_Subspace_Ridge_Lux.jl",
        concept("deep active subspace", ["Deep", "Active", "Subspace"], ["make_deep_active_subspace", "active_subspace"]),
        concept("ridge diagnostics", ["ridge", "spectrum"], ["ridge", "spectrum"])),
    pair("Lecture 14",
        "lectures/lecture_14_surrogates_and_gps/code/lecture_14_10_Deep_AS_vs_Linear_AS_Borehole.ipynb",
        "lectures/lecture_14_surrogates_and_gps/code_julia/lecture_14_10_Deep_AS_vs_Linear_AS_Borehole_Lux.jl",
        concept("borehole benchmark", ["Borehole", "borehole"], ["borehole_function", "borehole"]),
        concept("linear versus deep AS", ["Linear", "Deep", "AS"], ["active_subspace", "surrogate"])),
    pair("Lecture 15",
        "lectures/lecture_15_structural_estimation_smm/code/lecture_15_03_Structural_Estimation_BM.ipynb",
        "lectures/lecture_15_structural_estimation_smm/code_julia/lecture_15_03_Structural_Estimation_BM_Lux.jl",
        concept("structural SMM", ["Structural", "SMM"], ["SMMBrockMirmanParams", "smm_criterion"]),
        concept("common random numbers", ["common random", "CRN"], ["common_random_shocks", "common random"])),
    pair("Lecture 15",
        "lectures/lecture_15_structural_estimation_smm/code/lecture_15_03b_Structural_Estimation_BM_Joint.ipynb",
        "lectures/lecture_15_structural_estimation_smm/code_julia/lecture_15_03b_Structural_Estimation_BM_Joint_Lux.jl",
        concept("joint SMM", ["Joint", "SMM"], ["joint", "smm_joint"]),
        concept("identification diagnostics", ["identification", "sensitivity"], ["identification", "sensitivity"])),
    pair("Lecture 16",
        "lectures/lecture_16_climate_economics_iams/code/lecture_16_01_Climate_Exercise.ipynb",
        "lectures/lecture_16_climate_economics_iams/code_julia/lecture_16_01_Climate_Exercise_Lux.jl",
        concept("climate exercise", ["Climate", "DICE"], ["DICEClimateParams", "simulate_dice_climate_exercise"]),
        concept("mitigation comparison", ["mitigation", "damage"], ["mitigation", "avoided_damages"])),
    pair("Lecture 16",
        "lectures/lecture_16_climate_economics_iams/code/lecture_16_02_DICE_DEQN_Library_Port.ipynb",
        "lectures/lecture_16_climate_economics_iams/code_julia/lecture_16_02_DICE_DEQN_Library_Port_Lux.jl",
        concept("deterministic CDICE", ["DICE", "CDICE", "deterministic"], ["CDICEParams", "deterministic_cdice_residual"]),
        concept("carbon and temperature transitions", ["carbon", "temperature"], ["simulate_cdice_path", "deterministic_cdice_residual"])),
    pair("Lecture 16",
        "lectures/lecture_16_climate_economics_iams/code/lecture_16_03_Stochastic_DICE_DEQN.ipynb",
        "lectures/lecture_16_climate_economics_iams/code_julia/lecture_16_03_Stochastic_DICE_DEQN_Lux.jl",
        concept("stochastic CDICE", ["DICE", "Stochastic", "CDICE"], ["stochastic_cdice_residual", "CDICETeachingPolicy"]),
        concept("Gauss-Hermite productivity shock", ["Gauss", "Hermite", "shock"], ["gauss_hermite_rule", "shock"])),
]

const missing_translations = MissingTranslation[]

const intentional_extra_julia = Set([
    "lectures/lecture_02_intro_deep_learning/code_julia/lecture_02_00_Lux_Pluto_orientation.jl",
])

const excluded_python_notebook_prefixes = Dict(
    "lectures/lecture_01_python_primer/" => "Lecture 01 is a Python primer with no Julia translation track.",
)

function read_rel(path)
    return read(joinpath(repo_root, path), String)
end

function rel_files_under(dir; suffix)
    root = joinpath(repo_root, dir)
    out = String[]
    isdir(root) || return out
    for (parent, _, files) in walkdir(root)
        for file in files
            endswith(file, suffix) || continue
            push!(out, relpath(joinpath(parent, file), repo_root))
        end
    end
    return sort(out)
end

function has_any(source::AbstractString, markers)
    return any(marker -> occursin(marker, source), markers)
end

# Fast structural probe of a Pluto notebook's display order. Returns the cell
# kinds (:md or :code) in the order Pluto renders them, read from the trailing
# `# ╔═╡ Cell order:` block. Used to assert prose was interleaved with code and
# that the final rendered/evaluated cell is code (so `include` returns the
# diagnostic value, not a `Markdown.MD`).
function pluto_order_kinds(julia_source::AbstractString)
    kinds = Symbol[]
    in_order = false
    for line in split(julia_source, '\n')
        if occursin("Cell order:", line)
            in_order = true
            continue
        end
        in_order || continue
        if occursin("╟", line)
            push!(kinds, :md)
        elseif occursin("╠", line)
            push!(kinds, :code)
        end
    end
    return kinds
end

parses_as_pluto(julia_source::AbstractString) = Meta.parseall(julia_source) isa Expr

# True when at least one markdown cell is displayed after a code cell, i.e. the
# prose is woven through the notebook rather than dumped in a top-of-file block.
function interleaved_prose(kinds::AbstractVector{Symbol})
    return any(i -> kinds[i] === :md && any(==(:code), @view kinds[1:(i - 1)]),
        eachindex(kinds))
end

function lecture_number(lecture::AbstractString)
    m = match(r"Lecture ([0-9]{2})", lecture)
    m === nothing && error("lecture label lacks a two-digit number: $lecture")
    return m.captures[1]
end

function expected_missing_julia_path(python_path::AbstractString)
    stem = replace(basename(python_path), ".ipynb" => "")
    return replace(dirname(python_path), "/code" => "/code_julia") * "/" * stem * "_Lux.jl"
end

all_finite_numbers(x::Number) = isfinite(x)
all_finite_numbers(x::AbstractArray{<:Number}) = all(isfinite, x)
all_finite_numbers(x::AbstractArray) = all(all_finite_numbers, x)
all_finite_numbers(x::NamedTuple) = all(all_finite_numbers, values(x))
all_finite_numbers(x::Tuple) = all(all_finite_numbers, x)
all_finite_numbers(x::Dict) = all(all_finite_numbers, values(x))
all_finite_numbers(::Nothing) = true
all_finite_numbers(x) = false

struct ConstantRows{T}
    rows::Int
    value::T
end

(m::ConstantRows)(x, ps, st) = (fill(m.value, m.rows, size(x, 2)), st)

struct ConstantVector{V}
    values::V
end

function (m::ConstantVector)(x, ps, st)
    return repeat(reshape(m.values, :, 1), 1, size(x, 2)), st
end

logit(p) = log(p / (1 - p))

function crra_utility(c, gamma)
    gamma == 1 && return log(c)
    return c^(1 - gamma) / (1 - gamma)
end

@testset "Python/Julia notebook pair coverage" begin
    @test length(notebook_pairs) == 56
    @test length(unique(pair.python for pair in notebook_pairs)) == length(notebook_pairs)
    @test length(unique(pair.julia for pair in notebook_pairs)) == length(notebook_pairs)

    python_expected = Set(vcat([pair.python for pair in notebook_pairs],
        [gap.python for gap in missing_translations]))
    python_observed = Set(filter(path -> !any(startswith(path, prefix) for prefix in keys(excluded_python_notebook_prefixes)),
        rel_files_under("lectures"; suffix = ".ipynb")))
    @test all(!isempty, values(excluded_python_notebook_prefixes))
    @test isempty(setdiff(python_observed, python_expected))
    @test isempty(setdiff(python_expected, python_observed))

    julia_expected = union(Set(pair.julia for pair in notebook_pairs), intentional_extra_julia)
    julia_observed = Set(rel_files_under("lectures"; suffix = ".jl"))
    julia_observed = Set(filter(path -> occursin("/code_julia/", path), julia_observed))
    @test isempty(setdiff(julia_observed, julia_expected))
    @test isempty(setdiff(julia_expected, julia_observed))
end

@testset "Documented translation gaps" begin
    gap_docs = read_rel("lectures/AGENTS.md")
    for file in rel_files_under("lectures"; suffix = "AGENTS.md")
        gap_docs *= "\n" * read_rel(file)
    end

    for gap in missing_translations
        @testset "$(basename(gap.python))" begin
            @test isfile(joinpath(repo_root, gap.python))
            @test !isfile(joinpath(repo_root, expected_missing_julia_path(gap.python)))
            @test occursin(replace(basename(gap.python), ".ipynb" => ""), gap_docs)
            @test !isempty(gap.reason)
        end
    end
end

@testset "Notebook semantic equivalence markers" begin
    for item in notebook_pairs
        @testset "$(basename(item.python))" begin
            python_source = read_rel(item.python)
            julia_source = read_rel(item.julia)

            @test occursin("\"cells\"", python_source)
            @test occursin("\"source\"", python_source)
            @test startswith(julia_source, "### A Pluto.jl notebook ###")
            @test occursin("Pkg.activate", julia_source)
            @test occursin("DLEFJulia", julia_source)

            # The ground-truth prose is now interleaved with the Julia code, so
            # the generated top-of-file "Prose Port" dump must be gone.
            @test !occursin("# --- BEGIN PYTHON PROSE PORT ---", julia_source)
            @test !occursin("# --- END PYTHON PROSE PORT ---", julia_source)
            @test !occursin("# --- BEGIN PYTHON PROSE PORT ORDER ---", julia_source)
            @test !occursin("# --- END PYTHON PROSE PORT ORDER ---", julia_source)
            @test !occursin("Python Ground-Truth Prose Port", julia_source)

            # Provenance: the Julia notebook links back to its ground-truth pair.
            @test occursin(item.python, julia_source)

            # Pluto structure: the file parses, prose is woven through the code,
            # and the final rendered cell is code so `include` returns its value.
            @test parses_as_pluto(julia_source)
            order_kinds = pluto_order_kinds(julia_source)
            @test !isempty(order_kinds)
            @test count(==(:md), order_kinds) >= 2
            @test interleaved_prose(order_kinds)
            @test last(order_kinds) === :code

            has_smoke_run_mode = occursin("RUN_MODE = \"smoke\"", julia_source)
            has_fixed_budget_exception = item.lecture == "Lecture 15" && occursin("N_TRAIN =", julia_source)
            @test has_smoke_run_mode || has_fixed_budget_exception
            @test occursin("SEED = 0", julia_source)
            @test occursin("Lecture " * lecture_number(item.lecture), python_source)
            @test occursin("Lecture " * lecture_number(item.lecture), julia_source)

            for check in item.concepts
                @testset "$(check.name)" begin
                    @test has_any(python_source, check.python_markers)
                    @test has_any(julia_source, check.julia_markers)
                end
            end
        end
    end
end

@testset "Verified translated notebook diagnostics" begin
    @testset "Lecture 07 deterministic Brock-Mirman autodiff" begin
        result = include(joinpath(repo_root, "lectures/lecture_07_autodiff_for_deqns/code_julia/lecture_07_02_Brock_Mirman_AutoDiff_DEQN_Lux.jl"))
        @test result.raw_autodiff_vs_hand < 1e-10
        @test result.relative_autodiff_vs_shared < 1e-10
        @test result.trained_raw_autodiff_vs_hand < 1e-10
        @test result.trained_relative_autodiff_vs_shared < 1e-10
        @test isfinite(result.initial_loss) && isfinite(result.final_loss)
        @test result.final_loss < result.initial_loss
        @test result.policy_relative_l2 < 0.2
    end

    @testset "Lecture 07 stochastic Brock-Mirman autodiff" begin
        result = include(joinpath(repo_root, "lectures/lecture_07_autodiff_for_deqns/code_julia/lecture_07_03_Brock_Mirman_Uncertainty_AutoDiff_DEQN_Lux.jl"))
        @test result.run_mode == "smoke"
        @test result.seed == 0
        @test result.quadrature.nodes == 5
        @test abs(result.quadrature.weight_sum - 1) < 1e-12
        @test maximum(abs.(collect(values(result.slot_gradient_errors)))) < 1e-12
        @test result.residual_checks.max_abs_autodiff_minus_hand < 1e-12
        @test result.residual_checks.max_abs_relative_link < 1e-12
        @test result.full_depreciation_exact_policy.max_abs_autodiff_residual < 1e-12
    end

    @testset "Lecture 07 IRBC autodiff training" begin
        result = include(joinpath(repo_root, "lectures/lecture_07_autodiff_for_deqns/code_julia/lecture_07_04_IRBC_AutoDiff_DEQN_Lux.jl"))
        @test result.run_mode == "smoke"
        @test result.seed == 0
        @test result.quadrature.nodes == 27
        @test abs(result.quadrature.weight_sum - 1) < 1e-12
        @test result.pi_slot_gradient_error < 1e-10
        @test result.exogenous_residual_check.max_abs_euler_difference < 1e-10
        @test result.simulation_residual_check.max_abs_euler_difference < 1e-10
        @test result.training.checks.histories_finite
        @test result.training.checks.initial_residual_parity_ok
        @test result.training.checks.approach_A_residual_parity_ok
        @test result.training.checks.approach_B_residual_parity_ok
        @test result.training.approach_A.n_updates == result.training.budget.episodes
        @test result.training.approach_B.n_updates == result.training.budget.episodes
        @test result.training.approach_B.last_cloud_shape[1] == 4
        @test result.training.approach_B.end_state_shape[1] == 4
    end

    @testset "Lecture 08 OLG exercise" begin
        result = include(joinpath(repo_root, "lectures/lecture_08_olg_models_deqns/code_julia/lecture_08_11_OLG_Exercise_Lux.jl"))
        @test result.run_mode == "smoke"
        @test result.seed == 0
        @test result.youngest_saves_most
        @test result.oldest_saving_rate == 0
        @test result.feature_batch_shape == (40, 1)
        @test result.lux_style_state_preserved
        @test result.exact_policy_error_max < 1e-12
        @test result.validation_loss < 1e-24
    end

    @testset "Lecture 09 Krusell-Smith panel training" begin
        result = include(joinpath(repo_root, "lectures/lecture_09_heterogeneous_agents_youngs_method/code_julia/lecture_09_12_KrusellSmith_DeepLearning_Lux.jl"))
        @test result.run_mode == "smoke"
        @test result.seed == 0
        @test result.phase_b_steps > 0
        @test count(==(1), result.history.phase) == result.phase_b_steps
        @test length(result.panel_log) == result.phase_b_steps
        @test abs(result.current_capital_gap) < 1e-8
        @test abs(result.next_capital_gap) < 1e-8
        @test !hasproperty(result, :distribution_mass)
        @test !hasproperty(result, :final_distribution_mean)
        @test all(isfinite, result.history.loss)
    end

    @testset "Lecture 10 Krusell-Smith sequence histories" begin
        result = include(joinpath(repo_root, "lectures/lecture_10_sequence_space_deqns/code_julia/lecture_10_06_SequenceSpace_KrusellSmith_Lux.jl"))
        @test isfinite(result.initial_loss) && isfinite(result.final_loss)
        @test result.history_feature_dim == 3
        @test result.history_shape[1] == result.history_feature_dim
        @test result.flattened_history_dim == result.history_feature_dim * result.history_shape[2]
        @test result.distribution_mass ≈ 1.0 atol = 1e-8
    end

    @testset "Lecture 10 Krusell-Smith CPU tutorial histories" begin
        result = include(joinpath(repo_root, "lectures/lecture_10_sequence_space_deqns/code_julia/lecture_10_KrusellSmith_Tutorial_CPU_Lux.jl"))
        @test isfinite(result.initial_loss) && isfinite(result.final_loss)
        @test result.history_feature_dim == 3
        @test result.history_shape[1] == result.history_feature_dim
        @test result.flattened_history_dim == result.history_feature_dim * result.history_shape[2]
        @test result.next_mass ≈ 1.0 atol = 1e-8
    end

    @testset "Lecture 11 soft and hard ODE PINNs" begin
        result = include(joinpath(repo_root, "lectures/lecture_11_pinns/code_julia/lecture_11_02_ODE_PINN_SoftVsHardBCs_Lux.jl"))
        @test result.losses_finite
        @test result.exact_max_abs_residual < 1e-10
        @test result.hard_boundary_error < 1e-12
        @test result.final_soft_training_loss < result.initial_soft_loss
        @test result.final_hard_training_loss < result.initial_hard_loss
    end

    @testset "Lecture 16 deterministic CDICE trained diagnostics" begin
        result = include(joinpath(repo_root, "lectures/lecture_16_climate_economics_iams/code_julia/lecture_16_02_DICE_DEQN_Library_Port_Lux.jl"))
        @test isfinite(result.initial_loss) && isfinite(result.final_loss)
        @test isfinite(result.best_training_loss)
        @test result.trained_path_finite
        @test result.trained_TAT_2100 > 0
        @test isfinite(result.trained_scc_2015)
        @test result.teaching_scc_2015 > 0
    end

    @testset "Lecture 16 stochastic CDICE trained diagnostics" begin
        result = include(joinpath(repo_root, "lectures/lecture_16_climate_economics_iams/code_julia/lecture_16_03_Stochastic_DICE_DEQN_Lux.jl"))
        @test isfinite(result.initial_loss) && isfinite(result.final_loss)
        @test isfinite(result.best_training_loss)
        @test abs(result.gh_weight_sum - 1) < 1e-12
        @test result.deterministic_path_finite
        @test result.mc_scc_finite
        @test result.mc_TAT_finite
        @test result.deterministic_TAT_2100 > 0
        @test isfinite(result.mc_scc_mean_2100)
        @test result.mc_TAT_mean_2100 > 0
        @test result.teaching_deterministic_scc_2015 > 0
    end
end

@testset "DEQN and OLG equivalence mechanics" begin
    @testset "Brock-Mirman analytic policies" begin
        params = BrockMirmanParams(alpha = 0.36, beta = 0.99, delta = 1.0)
        k = reshape([0.7, 1.0, 1.3], 1, :)
        model = ConstantRows(1, params.alpha * params.beta)
        pieces, _ = deterministic_bm_residual(model, nothing, NamedTuple(), k; params)
        @test maximum(abs, pieces.residual) < 1e-10
        @test pieces.next_capital ≈ bm_full_depreciation_policy(k, params)

        stochastic_params = BrockMirmanParams(alpha = 0.36, beta = 0.99, delta = 1.0,
            rho_z = 0.9, sigma_z = 0.04)
        states = [0.95 1.00 1.05; 0.8 1.0 1.2]
        rule = gauss_hermite_rule(7)
        stochastic, _ = stochastic_bm_residual(ConstantRows(1, stochastic_params.alpha * stochastic_params.beta),
            nothing, NamedTuple(), states, rule; params = stochastic_params)
        @test maximum(abs, stochastic.residual) < 1e-10
    end

    @testset "IRBC steady state and complementarity" begin
        params = IRBCParams(countries = 2, alpha = 0.36, beta = 0.99, delta = 0.1, shock_std = 0.0)
        kstar = irbc_steady_state_capital(params)
        ystar = kstar^params.alpha
        investment_fraction = params.delta * kstar / ystar
        states = vcat(ones(params.countries, 3), fill(kstar, params.countries, 3))

        smooth, _ = irbc_smooth_residual(ConstantRows(params.countries, logit(investment_fraction)),
            nothing, NamedTuple(), states; params)
        @test maximum(abs, smooth.euler) < 1e-10
        @test maximum(abs, smooth.resource) < 1e-10

        raw = vcat(fill(logit(investment_fraction), params.countries), fill(-40.0, params.countries))
        irreversible, _ = irbc_irreversible_residual(ConstantVector(raw),
            nothing, NamedTuple(), states; params)
        @test all_finite_numbers(irreversible)
        @test maximum(abs, irreversible.euler) < 1e-10
        @test maximum(abs, irreversible.resource) < 1e-10
        @test maximum(abs, irreversible.complementarity) < 1e-7
    end

    @testset "OLG analytic and benchmark transforms" begin
        rng = StableRNG(8)
        analytic_params = AnalyticOLGParams()
        @test analytic_olg_state_dim(analytic_params) == 7
        @test analytic_olg_feature_dim(analytic_params) == 40
        analytic_states = sample_analytic_olg_states(rng, analytic_params, 5)
        analytic_features = analytic_olg_features(analytic_states; params = analytic_params)
        @test size(analytic_features) == (analytic_olg_feature_dim(analytic_params), 5)
        @test analytic_features[1:1, :] ≈ reshape((round.(Int, analytic_states[1, :]) .- 1) ./ 3, 1, :)
        @test sum(analytic_features[2:(1 + length(analytic_params.tfp)), :]; dims = 1) ≈ ones(1, 5)
        @test analytic_features[(end - length(analytic_params.tfp) + 1):end, :] ≈
            permutedims(analytic_params.transition[round.(Int, analytic_states[1, :]), :])

        exact = analytic_olg_exact_policy(analytic_states; params = analytic_params)
        error = analytic_olg_policy_error(exact.savings, analytic_states; params = analytic_params)
        @test error.summary.max_abs == 0
        pieces, _ = analytic_olg_residual(ConstantRows(analytic_params.n_ages - 1, 0.0),
            nothing, NamedTuple(), analytic_states; params = analytic_params, use_exact_policy = true)
        @test all_finite_numbers(pieces)
        @test maximum(abs, pieces.policy_error) == 0

        benchmark_default = BenchmarkOLGParams()
        @test benchmark_olg_state_dim(benchmark_default) == 113
        @test benchmark_olg_feature_dim(benchmark_default) == 240
        @test 4 * (benchmark_default.n_ages - 1) + 1 == 221

        benchmark_params = BenchmarkOLGParams(n_ages = 5)
        benchmark_states = sample_benchmark_olg_states(rng, benchmark_params, 4)
        benchmark_features = benchmark_olg_features(benchmark_states; params = benchmark_params)
        @test size(benchmark_features) == (benchmark_olg_feature_dim(benchmark_params), 4)
        @test benchmark_features[1:1, :] ≈ reshape((round.(Int, benchmark_states[1, :]) .- 1) ./ 3, 1, :)
        @test sum(benchmark_features[2:(1 + length(benchmark_params.tfp)), :]; dims = 1) ≈ ones(1, 4)
        @test benchmark_features[(end - length(benchmark_params.tfp) + 1):end, :] ≈
            permutedims(benchmark_params.transition[round.(Int, benchmark_states[1, :]), :])

        raw_rows = 4 * (benchmark_params.n_ages - 1) + 1
        policy = benchmark_olg_policy_from_raw(zeros(raw_rows, size(benchmark_states, 2)),
            benchmark_states; params = benchmark_params)
        @test all(policy.collateral .>= -100eps())
        @test all(policy.price .> 0)
        diagnostics, _ = benchmark_olg_residual(ConstantRows(raw_rows, 0.0),
            nothing, NamedTuple(), benchmark_states; params = benchmark_params)
        @test all_finite_numbers(diagnostics)
        @test diagnostics.kkt_capital ≈ policy.lambda .* policy.capital
        @test diagnostics.kkt_bond ≈ policy.mu .* policy.collateral
    end
end

@testset "Young and sequence-space equivalence mechanics" begin
    @testset "Young histogram propagation" begin
        grid = collect(range(0.0, 4.0; length = 9))
        hist = exp.(-abs.(grid .- 2.0))
        hist ./= sum(hist)
        next_hist = young_step(grid, hist, identity)
        @test young_mass(next_hist) ≈ young_mass(hist)
        @test young_mean(grid, next_hist) ≈ young_mean(grid, hist)

        transition = [0.9 0.1; 0.2 0.8]
        hist2 = [0.20 0.10 0.05 0.02 0.01 0.01 0.00 0.00 0.00;
                 0.00 0.00 0.02 0.04 0.06 0.10 0.14 0.15 0.10]
        hist2 ./= sum(hist2)
        policy = repeat(reshape(grid, 1, :), 2, 1)
        next2 = young_step(grid, hist2, policy; transition)
        @test young_mass(next2) ≈ 1.0
        @test size(unflatten_young_histogram(flatten_young_histogram(next2), 2, length(grid))) == size(next2)
    end

    @testset "Sequence-space histories" begin
        history = reshape(1.0:12.0, 2, 3, 2)
        flat = flatten_history(history)
        @test unflatten_history(flat, 2, 3) == history
        prepended = prepend_history(history, [10.0 20.0; 30.0 40.0])
        @test prepended[:, 1, :] == [10.0 20.0; 30.0 40.0]

        params = SequenceBrockMirmanParams()
        states = sequence_bm_initial_state(params; batch = 2)
        bm_history = zeros(1, 4, 2)
        model = ConstantRows(1, params.alpha * params.beta)
        rule = gauss_hermite_rule(5)
        pieces, _ = sequence_bm_residual(model, nothing, NamedTuple(), states, bm_history, rule; params)
        @test all_finite_numbers(pieces)
        next_states, next_history, _ = sequence_bm_forward_step(model, nothing, NamedTuple(),
            states, bm_history, 0.0; params)
        @test size(next_states) == size(states)
        @test size(next_history) == size(bm_history)
    end

    @testset "Sequence-space Krusell-Smith distribution" begin
        params = SequenceKSParams(capital_grid = collect(range(0.0, 8.0; length = 12)))
        history = sequence_ks_history(params; history_length = 4, z_index = 2)
        @test size(history) == (length(params.aggregate_z) + 1, 4, 1)
        @test history[end, 1, 1] ≈ params.aggregate_z[2]
        @test only(sequence_ks_current_z(history, params)) ≈ params.aggregate_z[2]
        @test size(flatten_history(history)) == ((length(params.aggregate_z) + 1) * 4, 1)

        distribution = sequence_ks_initial_distribution(params; K_target = 3.0)
        aggregates = sequence_ks_distribution_aggregates(distribution, params)
        @test aggregates.mass ≈ 1.0
        @test aggregates.capital ≈ 3.0 atol = 0.75
        policy = (savings = repeat(reshape(params.capital_grid, 1, :), length(params.idio_income), 1),)
        next_distribution = sequence_ks_distribution_step(distribution, policy; params)
        @test sum(next_distribution) ≈ 1.0
    end
end

@testset "PINN and continuous-time equivalence mechanics" begin
    @testset "ODE and Poisson manufactured solutions" begin
        xs = [0.2, 0.5, 0.8]
        ode_residual = [second_derivative(x -> analytic_zero_bc_solution(x), x) + 1 for x in xs]
        @test maximum(abs, ode_residual) < 1e-12

        xy = [(0.2, 0.3), (0.5, 0.25), (0.8, 0.7)]
        poisson_residual = [
            ForwardDiff.derivative(x -> ForwardDiff.derivative(xx -> poisson2d_exact(xx, y), x), x) +
            ForwardDiff.derivative(y0 -> ForwardDiff.derivative(yy -> poisson2d_exact(x, yy), y0), y) -
            poisson2d_forcing(x, y)
            for (x, y) in xy
        ]
        @test maximum(abs, poisson_residual) < 1e-8
        @test poisson2d_exact(0.0, 0.4) ≈ poisson2d_boundary_lifting(0.0, 0.4)
    end

    @testset "Cake-eating and Black-Scholes analytic checks" begin
        cake = CakeEatingParams()
        for a in (0.5, 1.5, 3.0)
            V = cake_eating_value_exact(a; params = cake)
            Va = cake_eating_kappa(cake)^(-cake.gamma) * a^(-cake.gamma)
            c = cake_eating_consumption_exact(a; params = cake)
            @test DLEFJulia._crra_utility(c, cake.gamma) ≈ crra_utility(c, cake.gamma)
            residual = cake.rho * V - (crra_utility(c, cake.gamma) + Va * (cake.r * a - c))
            @test abs(residual) < 1e-10
        end

        cake_model = make_mlp(1, (4,), 1; activation = NNlib.tanh)
        cake_state = setup_training(rng_from_seed(1104), cake_model, Optimisers.Descent(0.001); parameter_type = Float64)
        a_points = reshape([0.5, 1.5, 3.0], 1, :)
        cake_pieces, _ = cake_eating_hjb_loss(cake_state.model, cake_state.ps, cake_state.st, a_points; params = cake)
        direct_residual = [cake_eating_hjb_residual(cake_state.ps, a; params = cake) for a in vec(a_points)]
        @test cake_pieces.residual ≈ reshape(direct_residual, 1, :)
        @test all_finite_numbers(cake_pieces)

        bs = BlackScholesParams(strike = 50.0, r = 0.05, sigma = 0.2)
        @test black_scholes_call_price(60.0, 0.0; params = bs) ≈ 10.0
        @test black_scholes_call_price(0.0, 0.5; params = bs) ≈ 0.0
        @test 0 < black_scholes_delta(50.0, 0.5; params = bs) < 1
    end

    @testset "Aiyagari FD and density checks" begin
        params = CTAiyagariParams(n_quad = 9, n_a = 9, a_max = 4.0)
        rule = ct_aiyagari_trapezoid_rule(params)
        @test sum(rule.weights) ≈ params.a_max - params.a_min
        prices = ct_aiyagari_prices(2.0, 1.0; params)
        @test all_finite_numbers(prices)
        log_density = zeros(length(rule.nodes), 2)
        normalized = ct_aiyagari_normalized_density(log_density, rule.weights)
        aggregates = ct_aiyagari_distribution_aggregates(normalized.density, rule.nodes, rule.weights; params)
        @test aggregates.mass ≈ 1.0
        @test all(aggregates.mass_by_labor .> 0)
    end
end

@testset "Surrogate, SMM, and climate equivalence mechanics" begin
    @testset "Surrogate GP and active subspace" begin
        rng = StableRNG(14)
        design = black_scholes_design(rng, 8)
        normalizer = fit_box_normalizer(((50.0, 150.0), (50.0, 150.0), (0.1, 2.0), (0.05, 0.6), (0.01, 0.08)))
        @test denormalize_box(normalize_box(design.x, normalizer), normalizer) ≈ design.x

        gp = fit_cholesky_gp(design.x, vec(design.y); lengthscale = 60.0, variance = 100.0, noise = 1e-5)
        @test gp_rmse(gp, design.x, design.y) < 1e-3
        candidates = hcat(design.x[:, 1], design.x[:, 1] .+ [80.0, -40.0, 1.0, 0.2, 0.02])
        @test bal_next_index(gp, candidates) == 2

        directions = [1.0 0.0; 0.0 0.0]
        x = 2 .* rand(rng, 2, 20) .- 1
        y = radial_ridge_target(x, directions)
        gradients = radial_ridge_gradients(x, directions)
        as = active_subspace(active_subspace_matrix(gradients))
        @test as.values[1] > 100as.values[2]
        fit = fit_active_subspace_surrogate(x, y, as.vectors; dims = 1, degree = 6, lambda = 1e-8)
        @test relative_l2_error(predict_active_subspace_surrogate(fit, x), y) < 1e-3
    end

    @testset "SMM common random numbers and criteria" begin
        rng = StableRNG(15)
        shocks1 = common_random_shocks(rng, 12)
        shocks2 = common_random_shocks(StableRNG(15), 12)
        @test shocks1 == shocks2
        C = reshape(1.0:12.0, 4, 3)
        Ipath = 0.5 .* C
        Y = C .+ Ipath
        moments = smm_scalar_moments(C, Ipath, Y)
        target = vec(moments[2, :])
        criterion = smm_criterion(moments, target)
        @test argmin(criterion) == 2
        estimate = smm_grid_estimate([0.8, 0.9, 1.0, 1.1], moments, target)
        @test estimate.index == 2
    end

    @testset "DICE and CDICE transitions" begin
        climate = simulate_dice_climate_exercise(; mitigation_fraction = 0.5, comparison_year = 2100)
        @test climate.avoided_warming > 0
        @test climate.avoided_damages > 0

        params = CDICEParams()
        state = cdice_initial_state(; params)
        normalized = cdice_normalize_states(state; params)
        @test cdice_denormalize_states(normalized; params) ≈ state

        policy = CDICETeachingPolicy(; params, stochastic = false)
        deterministic, _ = deterministic_cdice_residual(policy, nothing, NamedTuple(), normalized; params)
        @test all_finite_numbers(deterministic)

        stoch_state = cdice_initial_state(; params, stochastic = true)
        stoch_norm = cdice_normalize_states(stoch_state; params, stochastic = true)
        stoch_policy = CDICETeachingPolicy(; params, stochastic = true)
        stochastic, _ = stochastic_cdice_residual(stoch_policy, nothing, NamedTuple(), stoch_norm,
            gauss_hermite_rule(3); params)
        @test all_finite_numbers(stochastic)
    end
end
