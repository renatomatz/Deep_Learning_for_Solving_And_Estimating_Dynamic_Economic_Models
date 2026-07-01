# Regression guards for the Wave-1 Julia/Lux/Pluto preview fixes.
#
# These are additive, self-contained guards (string parsing + arithmetic only);
# they do not run any notebook or training. Included from runtests.jl.
#
#   1. Pluto multiple-definition guard  -> guards the lecture_04_01 `x_start` fix
#   2. value(k) Brock-Mirman parity     -> guards the lecture_14_04 closed-form fix
#   3. Lecture 10 KS log-utility guard  -> guards the lecture_10_06 / tutorial recalibration

# --------------------------------------------------------------------------
# Guard 1 helpers: a pragmatic Julia-global scope tracker over Pluto cells.
#
# A Pluto notebook cannot define the same global in two cells. When these
# previews are `include`d as plain Julia the duplicate is silently reassigned
# instead of erroring, so the class of bug fixed in lecture_04_01 (`x_start`
# defined in two `begin` cells) can hide. We collect the names that are Julia
# *globals* per cell -- assignments that start a statement while outside any
# bracket, string, comment, and outside any local-scope block (`for`, `while`,
# `let`, `function`, `macro`, `struct`, `do`, `try`); `begin`/`if`/`quote`/
# `module` do not introduce scope, matching Julia -- and flag any name that is
# a global in more than one cell.
const PLUTO_CELL_MARKER = "# ╔═╡"
const PLUTO_SCOPE_OPENERS = Set(["for", "while", "let", "function", "macro", "struct", "do", "try"])
const PLUTO_NOSCOPE_OPENERS = Set(["begin", "if", "quote", "module"])
const PLUTO_BLOCK_OPENERS = union(PLUTO_SCOPE_OPENERS, PLUTO_NOSCOPE_OPENERS)
const PLUTO_CONTINUATIONS = Set(["else", "elseif", "catch", "finally"])
const PLUTO_ASSIGN_RE = r"^\s*([\p{L}_][\p{L}\p{N}_]*)\s*=(?![=>])"
const PLUTO_TOKEN_RE = r"[\p{L}_][\p{L}\p{N}_]*|[()\[\]{}]"

# Replace comment / string content with spaces (newlines preserved) so the
# structural scan below never trips over `=`, brackets, or keywords in prose.
function pluto_strip_code(text::AbstractString)
    cs = collect(text)
    n = length(cs)
    out = IOBuffer()
    i = 1
    while i <= n
        c = cs[i]
        if c == '#'                                   # line comment
            while i <= n && cs[i] != '\n'
                print(out, ' '); i += 1
            end
        elseif c == '"' && i + 2 <= n && cs[i + 1] == '"' && cs[i + 2] == '"'  # triple string
            print(out, "   "); i += 3
            while i <= n && !(i + 2 <= n && cs[i] == '"' && cs[i + 1] == '"' && cs[i + 2] == '"')
                print(out, cs[i] == '\n' ? '\n' : ' '); i += 1
            end
            if i <= n
                print(out, "   "); i += 3
            end
        elseif c == '"'                               # single-line string
            print(out, ' '); i += 1
            while i <= n && cs[i] != '"'
                if cs[i] == '\\' && i < n
                    print(out, ' '); i += 1
                end
                print(out, cs[i] == '\n' ? '\n' : ' '); i += 1
            end
            if i <= n
                print(out, ' '); i += 1
            end
        else
            print(out, c); i += 1
        end
    end
    return String(take!(out))
end

# Cell bodies between "# ╔═╡" markers (the preamble before the first marker and
# the trailing "Cell order:" block contain no top-level assignments).
function pluto_cells(text::AbstractString)
    cells = String[]
    current = nothing
    for line in split(text, '\n')
        if startswith(line, PLUTO_CELL_MARKER)
            current === nothing || push!(cells, current)
            current = ""
        elseif current !== nothing
            current *= line * "\n"
        end
    end
    current === nothing || push!(cells, current)
    return cells
end

function pluto_cell_globals(cell::AbstractString)
    code = pluto_strip_code(cell)
    names = Set{String}()
    bracket = 0
    scope_depth = 0
    stack = Bool[]                       # per open block: does it introduce local scope?
    for line in split(code, '\n')
        if bracket == 0 && scope_depth == 0
            m = match(PLUTO_ASSIGN_RE, line)
            if m !== nothing
                name = m.captures[1]
                if !(name in PLUTO_BLOCK_OPENERS) && !(name in PLUTO_CONTINUATIONS) && name != "end"
                    push!(names, name)
                end
            end
        end
        for tok in eachmatch(PLUTO_TOKEN_RE, line)
            t = tok.match
            if t == "(" || t == "[" || t == "{"
                bracket += 1
            elseif t == ")" || t == "]" || t == "}"
                bracket = max(bracket - 1, 0)
            elseif bracket == 0
                if t == "end"
                    isempty(stack) || (pop!(stack) && (scope_depth -= 1))
                elseif t in PLUTO_BLOCK_OPENERS
                    introduces = t in PLUTO_SCOPE_OPENERS
                    push!(stack, introduces)
                    introduces && (scope_depth += 1)
                end
            end
        end
    end
    return names
end

function pluto_cross_cell_duplicates(text::AbstractString)
    name_to_cells = Dict{String, Set{Int}}()
    for (ci, cell) in enumerate(pluto_cells(text))
        for nm in pluto_cell_globals(cell)
            push!(get!(name_to_cells, nm, Set{Int}()), ci)
        end
    end
    return sort!([nm for (nm, cs) in name_to_cells if length(cs) > 1])
end

# --------------------------------------------------------------------------
# Guard 3 helpers: read the calibration literally out of a `SequenceKSParams(`
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

const WAVE1_REPO_ROOT = normpath(joinpath(@__DIR__, "..", ".."))

function wave1_lux_notebooks()
    files = String[]
    for (dir, _, fs) in walkdir(joinpath(WAVE1_REPO_ROOT, "lectures"))
        basename(dir) == "code_julia" || continue
        for f in fs
            endswith(f, "_Lux.jl") && push!(files, joinpath(dir, f))
        end
    end
    return sort!(files)
end

# Notebooks that still carry the SAME "define in a setup cell, re-bind to a
# `train_result.<field>` in a training cell" multiple-definition bug as
# lecture_04_01's `x_start`. Entries here are @test_skip-ped instead of failing;
# remove an entry once its notebook is fixed so the guard then protects it too.
#
# Wave-2 fixed the last remaining offenders (lecture_08_08, lecture_08_10,
# lecture_10_05, lecture_10_06), so the baseline is now empty and every
# notebook is actively guarded.
const WAVE1_KNOWN_MULTIDEF = Set{String}()

@testset "Wave 1 regression guards" begin
    @testset "Pluto no cross-cell multiple definitions" begin
        files = wave1_lux_notebooks()
        @test length(files) >= 50            # the walk actually found the notebooks
        for file in files
            rel = relpath(file, WAVE1_REPO_ROOT)
            dups = pluto_cross_cell_duplicates(read(file, String))
            if rel in WAVE1_KNOWN_MULTIDEF
                @test_skip isempty(dups)     # documented pre-existing bug (see above)
            else
                @test isempty(dups)
            end
        end
    end

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
            "lectures/lecture_10_sequence_space_deqns/code_julia/lecture_10_06_SequenceSpace_KrusellSmith_Lux.jl",
            "lectures/lecture_10_sequence_space_deqns/code_julia/lecture_10_KrusellSmith_Tutorial_CPU_Lux.jl",
        ]
        for rel in ks_notebooks
            text = read(joinpath(WAVE1_REPO_ROOT, rel), String)
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
