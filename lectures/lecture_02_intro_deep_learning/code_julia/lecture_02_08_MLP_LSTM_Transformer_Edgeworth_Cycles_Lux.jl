### A Pluto.jl notebook ###
# v0.20.0

using Markdown
using InteractiveUtils

# ╔═╡ 11111111-0208-4111-8111-111111111111
md"""
# Lecture 02, Notebook 08: Sequence Models on Edgeworth Cycles

This smoke translation keeps the memory-ladder experiment on a synthetic
Edgeworth-cycle series. To stay in the Phase-0 dependency set, it compares three
Lux-compatible feature maps: last-price MLP, full-window MLP, and an
attention-style summary MLP.
"""

# ╔═╡ 2700188c-664e-ea43-8f88-d7a19c2494c9
md"""
## Lecture 02, Notebook 08: Sequence Models: MLP, LSTM, and Transformer on Edgeworth Cycles

**Course:** Deep Learning for Solving and Estimating Dynamic Models in Economics and Finance
**Script reference:** §1.10–1.11 (Sequence models: RNN, LSTM, Transformer on Edgeworth cycles)
**Notebook role:** core
**Author:** Simon Scheidegger

*Julia/Lux/Pluto preview of* `lectures/lecture_02_intro_deep_learning/code/lecture_02_08_MLP_LSTM_Transformer_Edgeworth_Cycles.ipynb`.

> **Run mode.** The checked-in run uses `RUN_MODE = "smoke"` for fast execution; set `RUN_MODE` to `"teaching"` or `"production"` in the setup cell for the longer series and training budgets.
"""

# ╔═╡ a12331ca-2932-3ca6-e16a-535c1e837a7f
md"""
## MLPs vs LSTMs vs Transformers on Economic Time Series: Edgeworth Cycles

This is the main day-1 sequence-model notebook. It contrasts three architectures on the same task to make the **memory ladder** concrete:

1. **MLP** is memoryless: it sees a single observation at a time. Each prediction is independent.
2. **LSTM** carries a recurrent hidden state forward, with gates that protect long-range information.
3. **Transformer** has no recurrence at all: every position attends to every other in parallel via self-attention.

We use the same synthetic **Edgeworth-cycle** dataset throughout — sudden gasoline-price jumps followed by gradual undercutting, the asymmetric sawtooth documented for retail fuel markets.

**Phase-0 Julia preview.** To stay in the base Lux dependency set, this notebook substitutes three *feature maps* for the three architectures, keeping the memory-ladder message while training only MLPs:

- **last-price MLP** — sees only \$x_t\$ (the memoryless baseline, standing in for the plain MLP);
- **full-window MLP** — sees the entire window \$[x_{t-W+1},\dots,x_t]\$ at once (dense access to history, in the spirit of what the LSTM accumulates recurrently);
- **attention-summary MLP** — sees a few hand-built summaries of the window (last value, a position-weighted average, and the first-to-last change), mimicking the pooled features self-attention would extract.

What to look for: the last-price model collapses near the cycle mean (from one point it cannot tell where it is in the period), while the window and attention-summary models recover the sawtooth because they can see the phase.

_Pedagogical note: this stages the historical evolution MLP → LSTM (1997) → Transformer (2017). The point is not to crown a winner on a toy series, but to make architectural inductive biases visible. The full LSTM and Transformer implementations live in the Python ground truth._
"""

# ╔═╡ 22222222-0208-4222-8222-222222222222
begin
    import Pkg
    Pkg.activate(joinpath(@__DIR__, "..", "..", "..", "julia"))

    using CairoMakie
    using DLEFJulia
    using Lux
    using NNlib
    using Optimisers
    using Statistics
end

# ╔═╡ 33333333-0208-4333-8333-333333333333
begin
    RUN_MODE = "smoke"
    SEED = 0
    budgets = (
        smoke = (steps = 35, T = 260, window = 10),
        teaching = (steps = 250, T = 1_000, window = 10),
        production = (steps = 1_000, T = 2_000, window = 12),
    )
    hp = run_mode_budget(RUN_MODE; budgets)
    rng = rng_from_seed(SEED)
end

# ╔═╡ f1ff8483-b650-5a8f-20c4-61916edcdab9
md"""
### 2. Data preparation

We predict \$x_{t+1}\$ from a window \$[x_{t-W+1},\dots,x_t]\$ of length \$W=10\$. The next cell generates the synthetic Edgeworth cycle (a jump-and-undercut sawtooth), slices it into overlapping windows, splits train/test, and builds the three feature maps: the last-price map keeps only the final window entry; the full-window map keeps all \$W\$ entries; the attention-summary map keeps the last value, a position-weighted average, and the first-to-last change. Same underlying data for all three, so any difference is purely in what each model is allowed to see.
"""

# ╔═╡ 44444444-0208-4444-8444-444444444444
begin
    function edgeworth_cycle(rng, T)
        y = zeros(Float64, T)
        y[1] = 1.0
        phase = 0
        for t in 2:T
            phase += 1
            reset = phase > 28 || y[t - 1] < 0.55 || rand(rng) < 0.035
            if reset
                y[t] = 1.15 + 0.05 * randn(rng)
                phase = 0
            else
                y[t] = y[t - 1] - 0.025 - 0.004 * phase + 0.015 * randn(rng)
            end
        end
        return y
    end

    series = edgeworth_cycle(rng, hp.T)
    W = hp.window
    X = reduce(hcat, (series[t:(t + W - 1)] for t in 1:(length(series) - W)))
    y = reshape(series[(W + 1):end], 1, :)
    split = floor(Int, 0.75 * size(X, 2))
    X_train, X_test = X[:, 1:split], X[:, (split + 1):end]
    y_train, y_test = y[:, 1:split], y[:, (split + 1):end]

    attention_features(X) = vcat(
        reshape(X[end, :], 1, :),
        reshape(sum((1:W) .* X; dims = 1) ./ sum(1:W), 1, :),
        reshape(X[end, :] .- X[1, :], 1, :),
    )
end

# ╔═╡ 990ebb03-f276-248d-c9b6-ac8b5ca6a4f3
md"""
### 3. Define the three feature maps

Each model is the *same* small Lux MLP (`make_mlp`, two hidden layers, `tanh`); only the input differs:

- **last-price MLP** (input dimension 1) — the deliberate no-memory baseline.
- **full-window MLP** (input dimension \$W\$) — reads the whole window at once.
- **attention-summary MLP** (input dimension 3) — reads the pooled window summaries.

In the Python ground truth these are instead three genuinely different architectures: a one-input MLP, an LSTM that consumes the window recurrently, and a tiny Transformer encoder (input projection to \$d=16\$, learned positional embeddings, two 4-head encoder layers, ≈ 4.7k parameters, deliberately matched to the LSTM's ≈ 4.5k so the comparison is honest).
"""

# ╔═╡ cffdcf08-cb6c-0efd-93f8-926e26797c74
md"""
### 4. Training

We train each model with `Optimisers.Adam` under MSE loss via the shared `train_step!` loop. In the Python notebook the MLP and LSTM converge in ~200 epochs while the Transformer trains longer (1000 epochs) with a cosine learning-rate schedule; here all three run for the same smoke-sized budget in seconds.
"""

# ╔═╡ 55555555-0208-4555-8555-555555555555
begin
    function fit_sequence_model(input_train, target_train, input_test, target_test; seed_offset, input_dim)
        model = make_mlp(input_dim, (24, 24), 1; activation = NNlib.tanh)
        state = setup_training(rng_from_seed(SEED; offset = seed_offset), model, Optimisers.Adam(0.008); parameter_type = Float64)
        loss_fn(model, ps, st, batch) = begin
            prediction, st_new = model(batch.x, ps, st)
            return mse_loss(prediction, batch.y), st_new
        end
        batch = (x = input_train, y = target_train)
        initial = loss_value(state, loss_fn, batch)
        for _ in 1:hp.steps
            train_step!(state, loss_fn, batch; max_grad_norm = 10.0)
        end
        test_loss = loss_value(state, loss_fn, (x = input_test, y = target_test))
        prediction, _ = state.model(input_test, state.ps, state.st)
        return (initial = initial, test_loss = test_loss, prediction = prediction, state = state)
    end

    last_result = fit_sequence_model(reshape(X_train[end, :], 1, :), y_train, reshape(X_test[end, :], 1, :), y_test; seed_offset = 1, input_dim = 1)
    window_result = fit_sequence_model(X_train, y_train, X_test, y_test; seed_offset = 2, input_dim = W)
    attention_result = fit_sequence_model(attention_features(X_train), y_train, attention_features(X_test), y_test; seed_offset = 3, input_dim = 3)
end

# ╔═╡ 712e3029-c2c5-ef0a-4a54-695ea04d346a
md"""
### 5. Predictions on held-out data

Notice the visual ordering. The last-price model collapses toward a near-constant: from a single point it cannot tell whether it is just before a price jump or mid-undercut. The full-window and attention-summary models track the sawtooth, because they can see enough of the window to place the current phase — the same qualitative gap the Python notebook shows between the MLP and the LSTM/Transformer.
"""

# ╔═╡ 66666666-0208-4666-8666-666666666666
begin
    fig = Figure(size = figure_size(RUN_MODE))
    ax = Axis(fig[1, 1], xlabel = "test index", ylabel = "price")
    idx = 1:min(80, size(y_test, 2))
    lines!(ax, idx, vec(y_test[:, idx]); label = "truth", color = :black, linewidth = 3)
    lines!(ax, idx, vec(last_result.prediction[:, idx]); label = "last MLP", color = :gray45)
    lines!(ax, idx, vec(window_result.prediction[:, idx]); label = "window MLP", color = :dodgerblue3)
    lines!(ax, idx, vec(attention_result.prediction[:, idx]); label = "attention summary", color = :darkorange)
    axislegend(ax; position = :lt)
    fig
end

# ╔═╡ 3f7947bf-cf5a-2932-109b-a872ceeea45f
md"""
### The full Python notebook also covers

The PyTorch ground truth implements the three architectures for real: an LSTM with a recurrent hidden state and gates, and a TinyTransformer encoder with learned positional embeddings and multi-head self-attention. It reports **parameter counts** (the script, §1.10 p. 52, quotes the TinyTransformer at ≈ 4.7k parameters), trains the Transformer with a cosine learning-rate schedule, and plots training-loss curves and test RMSEs for all three.
"""

# ╔═╡ f2e0218d-18f7-f211-1a08-5d6c42714656
md"""
### 6. Discussion: why these models behave the way they do

**No memory (last-price / MLP).** Seeing only \$x_t\$, the model can learn just the marginal mapping \$x_t \mapsto x_{t+1}\$. On a sawtooth this is nearly one-to-many — the same \$x_t\$ occurs at very different phases — so its best response is to predict near the conditional mean.

**Recurrent memory (LSTM).** A recurrent hidden state \$h_t\$ accumulates information across the window. Its gating is a strong inductive bias for periodic, asymmetric series like Edgeworth cycles: the cell state can encode how far into the current cycle we are, and reset at a jump.

**Attention (Transformer).** Drops recurrence entirely: self-attention lets every position read every other directly, in parallel. On a moderate window it recovers the cycle from the data alone. On this tiny, very regular problem the LSTM's recurrent prior is hard to beat; on long-context, large-data tasks the Transformer's \$\mathcal{O}(1)\$ direct path between positions wins.

**Take-home.** Architecture choice is about matching the model's inductive bias to the structure of the problem. The historical sequence MLP \$\to\$ LSTM \$\to\$ Transformer *removes* hard-coded structure and *adds* flexibility — helpful when data and context grow, overkill when neither does. The three feature maps in this preview stage the same lesson: the more of the window a model can see, the better it places itself in the cycle.

The cell below returns a machine-checkable summary comparing the three feature maps' test losses.
"""

# ╔═╡ 77777777-0208-4777-8777-777777777777
(
    window_length = W,
    last_mlp_test_loss = last_result.test_loss,
    window_mlp_test_loss = window_result.test_loss,
    attention_summary_test_loss = attention_result.test_loss,
    best_smoke_model = argmin([last_result.test_loss, window_result.test_loss, attention_result.test_loss]),
)

# ╔═╡ Cell order:
# ╟─11111111-0208-4111-8111-111111111111
# ╟─2700188c-664e-ea43-8f88-d7a19c2494c9
# ╟─a12331ca-2932-3ca6-e16a-535c1e837a7f
# ╠═22222222-0208-4222-8222-222222222222
# ╠═33333333-0208-4333-8333-333333333333
# ╟─f1ff8483-b650-5a8f-20c4-61916edcdab9
# ╠═44444444-0208-4444-8444-444444444444
# ╟─990ebb03-f276-248d-c9b6-ac8b5ca6a4f3
# ╟─cffdcf08-cb6c-0efd-93f8-926e26797c74
# ╠═55555555-0208-4555-8555-555555555555
# ╟─712e3029-c2c5-ef0a-4a54-695ea04d346a
# ╠═66666666-0208-4666-8666-666666666666
# ╟─3f7947bf-cf5a-2932-109b-a872ceeea45f
# ╟─f2e0218d-18f7-f211-1a08-5d6c42714656
# ╠═77777777-0208-4777-8777-777777777777
