# AGENTS.md

## Purpose

Lecture 14 is the surrogate and Gaussian-process toolbox for expensive
simulators: deep surrogates, GP regression, Bayesian active learning, active
subspaces, deep kernels, and GP-VFI.

## Map

Start with `README.md`, then inspect `slides/lecture_14_surrogates_and_gps.tex`,
then notebooks in numeric order. Notebook `03` is intentionally absent;
structural-estimation notebooks live in Lecture 15.

`slides/fig/` contains GP/BAL and GP-VFI PDFs. The current deck directly uses
`gp_active_learning.pdf` and `gp_vfi_active_learning_1d.pdf`; other PDFs are
extra/generated teaching artifacts.

## Running And Editing

This lecture has parallel Python notebooks and Julia/Lux/Pluto previews. Do not
make Python-stack assumptions when changing guidance or examples; preserve both
the original notebook warnings and the Julia preview caveats.

Most notebooks default to `RUN_MODE = "smoke"`. Production modes can be much
heavier, especially notebooks `01`, `09`, and `10`.

Use the root environment. The local stack is NumPy, SciPy, Matplotlib,
scikit-learn, and PyTorch for deep-surrogate/deep-active-subspace notebooks.

Notebook `04` has `SAVE_FIGURES = True` and `FIG_DIR = "fig_gp_vfi"`. Run it
from `code/` if you do not want figure output created relative to another
working directory. Existing saved notebook outputs may show stale absolute paths
such as `/mnt/data/fig_gp_vfi`; treat `fig_gp_vfi/` as generated output, not the
slide asset directory (`slides/fig/`).

Run Julia previews from `code_julia/` with the shared `../../../julia` project.
All nine Python notebooks have matching Pluto files. Each preview should
activate the shared project, import `DLEFJulia`, keep `RUN_MODE = "smoke"` and
`SEED = 0` defaults unless deliberately changed, and use Lux-native
feature-by-batch and explicit parameter/state conventions. They are covered by
`julia/test/smoke/wave5_notebooks.jl`, which is a notebook include smoke check,
not a production-convergence or figure-parity test.

The Julia track intentionally uses in-house Cholesky GP helpers rather than
adding a GP package. Several files are compact previews rather than full
production parity: `lecture_14_04_GP_Value_Function_Iteration_Lux.jl` fits a GP
to the closed-form Brock-Mirman benchmark instead of reproducing adaptive
GP-VFI, and `lecture_14_08_Deep_Kernel_Learning_Lux.jl` compares raw-input and
feature-space GPs without joint feature-map training.

Preserve those production-parity caveats when editing local guidance. The active
subspace, deep active-subspace, borehole, GP/BAL, and surrogate-primer previews
use deliberately small smoke budgets and shared `run_mode_budget` helpers; do
not recast smoke results as classroom or production estimates. Keep the Julia
dependency footprint narrow unless a concrete notebook need justifies changing
the root `julia` project and the wave5 smoke coverage.

For notebook inspection, parse source cells rather than raw grep over embedded
outputs.
