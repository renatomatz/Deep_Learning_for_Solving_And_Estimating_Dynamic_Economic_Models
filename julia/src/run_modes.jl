export VALID_RUN_MODES,
    validate_run_mode,
    run_mode_symbol,
    run_mode_budget,
    default_run_mode_budgets,
    rng_from_seed

using StableRNGs: StableRNG

const VALID_RUN_MODES = ("smoke", "teaching", "production")

const _RUN_MODE_SYMBOLS = (:smoke, :teaching, :production)

default_run_mode_budgets() = (
    smoke = (epochs = 5, batch_size = 32, steps = 25),
    teaching = (epochs = 50, batch_size = 64, steps = 500),
    production = (epochs = 500, batch_size = 256, steps = 10_000),
)

function run_mode_symbol(mode::Symbol)
    mode in _RUN_MODE_SYMBOLS && return mode
    throw(ArgumentError("RUN_MODE must be one of $(join(VALID_RUN_MODES, ", ")); got :$mode"))
end

function run_mode_symbol(mode::AbstractString)
    normalized = Symbol(lowercase(strip(mode)))
    return run_mode_symbol(normalized)
end

validate_run_mode(mode) = String(run_mode_symbol(mode))

function run_mode_budget(mode; budgets = default_run_mode_budgets())
    key = run_mode_symbol(mode)
    hasproperty(budgets, key) && return getproperty(budgets, key)
    throw(ArgumentError("No budget entry for RUN_MODE = $(String(key))"))
end

function rng_from_seed(seed::Integer; offset::Integer = 0)
    return StableRNG(seed + offset)
end
