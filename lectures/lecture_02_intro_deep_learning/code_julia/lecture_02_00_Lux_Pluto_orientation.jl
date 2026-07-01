### A Pluto.jl notebook ###
# v0.20.0

using Markdown
using InteractiveUtils

# ╔═╡ 11111111-1111-1111-1111-111111111111
md"""
# Lecture 02 Julia/Lux/Pluto Orientation

This short notebook is the Julia entry point for the Lux-native translation
track. It uses the shared project in `julia/` and the same course conventions
as the Python notebooks:

```julia
RUN_MODE = "smoke"
SEED = 0
```

It is not a full Julia-language primer. It shows the conventions that later
`code_julia/` notebooks assume: Pluto notebooks, the shared project environment,
feature-by-batch arrays, Lux's explicit parameter/state interface, and the
small shared training helper.
"""

# ╔═╡ 12121212-1212-1212-1212-121212121212
md"""
Every translated notebook activates the repository's shared Julia project rather
than carrying a local Pluto environment. Keep this pattern when creating new
notebooks so all lectures use the same `DLEFJulia` helpers and dependency set.
"""

# ╔═╡ 22222222-2222-2222-2222-222222222222
begin
    import Pkg
    Pkg.activate(joinpath(@__DIR__, "..", "..", "..", "julia"))

    using DLEFJulia
    using Lux
    using Optimisers
    using StableRNGs
end

# ╔═╡ 33333333-3333-3333-3333-333333333333
begin
    RUN_MODE = "smoke"
    SEED = 0

    budget = run_mode_budget(RUN_MODE)
    rng = rng_from_seed(SEED)
end

# ╔═╡ 34343434-3434-3434-3434-343434343434
md"""
`RUN_MODE` controls the teaching budget. Use `smoke` for structural checks,
`teaching` for classroom-quality runs, and `production` only when reproducing a
full result. `SEED` should stay explicit so smoke checks and examples are
repeatable.
"""

# ╔═╡ 44444444-4444-4444-4444-444444444444
md"""
Lux layers consume arrays in feature-by-batch layout: rows are features and
columns are observations. Many Python notebooks use table-shaped
batch-by-feature arrays, so the translation converts at the boundary and keeps
the orientation explicit.
"""

# ╔═╡ 55555555-5555-5555-5555-555555555555
begin
    batch_features = [
        -1.0  1.0
        -0.5  0.25
         0.0  0.0
         0.5  0.25
         1.0  1.0
    ]

    x = to_feature_batch(batch_features)
    target = reshape(batch_features[:, 1] .^ 2 .+ batch_features[:, 2], 1, :)

    size(x), size(target)
end

# ╔═╡ 66666666-6666-6666-6666-666666666666
begin
    model = make_mlp(2, (8, 8), 1)
    ps, st = setup_model(rng, model; parameter_type = Float64)

    y, st_new = model(x, ps, st)
    first_loss = mse_loss(y, target)
end

# ╔═╡ 77777777-7777-7777-7777-777777777777
md"""
The shared training helper keeps Lux's explicit `model, ps, st` flow visible.
For full notebooks, lecture-specific residuals and diagnostics should stay
outside the optimizer loop.
"""

# ╔═╡ 88888888-8888-8888-8888-888888888888
begin
    train_state = setup_training(model, ps, st, Optimisers.Adam(0.01))

    loss_fn(model, ps, st) = begin
        prediction, st_next = model(x, ps, st)
        return mse_loss(prediction, target), st_next
    end

    history = NamedTuple[]
    for _ in 1:budget.epochs
        metrics = train_step!(train_state, loss_fn; max_grad_norm = 10.0)
        append_metric!(history; step = metrics.step, loss = metrics.loss)
    end

    (initial_loss = first_loss, final_loss = history[end].loss, steps = length(history))
end

# ╔═╡ 99999999-9999-9999-9999-999999999999
md"""
Next steps:

- For a fuller supervised-learning training loop, open
  `lecture_02_06_Lux_Training_Fundamentals.jl`.
- For the first economic residual notebook, open
  `../../lecture_03_deep_equilibrium_nets/code_julia/lecture_03_01_Brock_Mirman_1972_DEQN_Lux.jl`.
- When editing notebooks, keep them as Pluto `.jl` files and avoid introducing
  Keras-, PyTorch-, or JAX-shaped wrapper APIs unless the notebook is explicitly
  comparing frameworks.
"""

# ╔═╡ Cell order:
# ╟─11111111-1111-1111-1111-111111111111
# ╟─12121212-1212-1212-1212-121212121212
# ╠═22222222-2222-2222-2222-222222222222
# ╠═33333333-3333-3333-3333-333333333333
# ╟─34343434-3434-3434-3434-343434343434
# ╟─44444444-4444-4444-4444-444444444444
# ╠═55555555-5555-5555-5555-555555555555
# ╠═66666666-6666-6666-6666-666666666666
# ╟─77777777-7777-7777-7777-777777777777
# ╠═88888888-8888-8888-8888-888888888888
# ╟─99999999-9999-9999-9999-999999999999
