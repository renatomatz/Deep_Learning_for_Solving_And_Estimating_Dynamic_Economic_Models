export dlef_theme_kwargs, figure_size

function dlef_theme_kwargs()
    return (
        fontsize = 16,
        linewidth = 2,
        markersize = 8,
    )
end

function figure_size(mode)
    key = run_mode_symbol(mode)
    key == :smoke && return (640, 420)
    key == :teaching && return (800, 520)
    return (1000, 650)
end
