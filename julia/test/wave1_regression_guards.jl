# Regression guards for the Julia/Lux/Jupyter preview fixes.
#
# Self-contained guards (JSON parsing of notebook cell source + arithmetic only);
# they do not run any notebook or training. Included from runtests.jl.
#
#   1. value(k) Brock-Mirman parity     -> guards the lecture_14_04 closed-form fix
#   2. Lecture 10 KS log-utility guard  -> guards the lecture_10_06 / tutorial recalibration
#
# The former "Pluto no cross-cell multiple definitions" guard was dropped in the
# Pluto -> Jupyter migration: its premise -- that a variable cannot be a global in
# two cells -- is a Pluto reactive-notebook constraint that does not exist in
# Jupyter, so re-binding a name across cells is no longer a defect to guard.

using JSON

const WAVE1_REPO_ROOT = normpath(joinpath(@__DIR__, "..", ".."))

# Concatenated text of every cell's `source` in a Jupyter `.ipynb`. Cell source
# is a JSON array of strings (quotes escaped, split across lines), so calibration
# strings like `SequenceKSParams(...)` must be read from the parsed-and-joined
# text, not the raw JSON bytes.
function wave1_notebook_text(rel::AbstractString)
    nb = JSON.parsefile(joinpath(WAVE1_REPO_ROOT, rel))
    return join((join(cell["source"]) for cell in nb["cells"]), "\n")
end

# --------------------------------------------------------------------------
# Guard 2 helpers: read the calibration literally out of a `SequenceKSParams(`
# call so a revert to the shared CRRA defaults (gamma=2.0, delta=0.08) is caught.
function ks_call_body(text::AbstractString, start::Integer)
    r = findnext("SequenceKSParams(", text, start)
    r === nothing && return (nothing, 0)
    open_paren = last(r)
    depth = 1
    k = nextind(text, open_paren)
    while k <= lastindex(text)
        c = text[k]
        if c == '('
            depth += 1
        elseif c == ')'
            depth -= 1
            depth == 0 && return (text[nextind(text, open_paren):prevind(text, k)], nextind(text, k))
        end
        k = nextind(text, k)
    end
    return (nothing, 0)
end

function ks_keyword_value(body::AbstractString, name::AbstractString)
    m = match(Regex("\\b" * name * "\\s*=\\s*([-0-9.eE+]+)"), body)
    m === nothing && return nothing
    return parse(Float64, m.captures[1])
end

# --------------------------------------------------------------------------

@testset "Wave 1 regression guards" begin
    @testset "value(k) Brock-Mirman closed-form satisfies the Bellman equation" begin
        # Same calibration as lecture_14_04's `bm`. The full-depreciation closed
        # form must satisfy V(k) = log(c*) + beta*V(k') with c* = (1-alpha*beta)k^alpha
        # and k' = alpha*beta*k^alpha; the pre-fix formula (log-k slope 9.0) did not.
        params = BrockMirmanParams(delta = 1.0, beta = 0.96)
        alpha, beta = params.alpha, params.beta
        savings_rate = alpha * beta
        value(k) = log(1 - savings_rate) / (1 - beta) +
            savings_rate * log(savings_rate) / ((1 - beta) * (1 - savings_rate)) +
            (alpha / (1 - savings_rate)) * log(k)
        for k in (0.5, 1.0, 3.0, 7.0, 10.0)
            c_star = (1 - savings_rate) * k^alpha
            k_next = savings_rate * k^alpha
            @test value(k) ≈ log(c_star) + beta * value(k_next) atol = 1e-10
        end
        # Sanity: the log-k slope is alpha/(1-alpha*beta), not the pre-fix alpha/(1-beta).
        @test (value(exp(1.0)) - value(1.0)) ≈ alpha / (1 - savings_rate) atol = 1e-10
    end

    @testset "Lecture 10 Krusell-Smith log-utility calibration" begin
        ks_notebooks = [
            "lectures/lecture_10_sequence_space_deqns/code_julia/lecture_10_06_SequenceSpace_KrusellSmith_Lux.ipynb",
            "lectures/lecture_10_sequence_space_deqns/code_julia/lecture_10_KrusellSmith_Tutorial_CPU_Lux.ipynb",
        ]
        for rel in ks_notebooks
            text = wave1_notebook_text(rel)
            pos = 1
            n_calls = 0
            while true
                body, next_pos = ks_call_body(text, pos)
                body === nothing && break
                n_calls += 1
                gamma = ks_keyword_value(body, "gamma")
                delta = ks_keyword_value(body, "delta")
                @test gamma == 1.0           # log utility, not the CRRA default 2.0
                @test delta == 0.025         # not the shared default 0.08
                pos = next_pos
            end
            @test n_calls >= 1               # the call was actually found and checked
        end
    end
end
