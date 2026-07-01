# AGENTS.md

## Purpose

Lecture 02 is the course's deep-learning foundations module: classical ML,
gradient descent, SGD, double descent, Keras/TensorFlow, TensorBoard, PyTorch,
Genz approximation, sequence models, and a transformer AR(1) demo.

## Map

- `README.md` gives the canonical order.
- `slides/lecture_02_intro_deep_learning.tex` is the Beamer source.
- `slides/figures/*.png` are slide inputs.
- `code/lecture_02_01_*.ipynb` through `code/lecture_02_09_*.ipynb` are the
  notebooks.
- `code_julia/lecture_02_00_Lux_orientation.ipynb` through
  `code_julia/lecture_02_09_Transformer_InContext_AR1_Lux.ipynb` are the current
  Julia/Lux previews.
- `code/SGD_data.txt` is read by the SGD notebook.

## Julia/Lux Preview Map

- `lecture_02_00_Lux_orientation.jl` is the Julia entry point: shared
  `../../../julia` activation, notebook workflow, `DLEFJulia`, `RUN_MODE`, `SEED`,
  feature-by-batch arrays, Lux `model, ps, st`, and the first shared training
  helper.
- `lecture_02_01_BasicML_intro_Lux.jl` keeps OLS and a small supervised Lux MLP.
- `lecture_02_02_GradientDescent_and_StochasticGradientDescent_Lux.jl` keeps
  explicit gradient descent and the cricket-data SGD example using
  `code/SGD_data.txt`.
- `lecture_02_03_Double_Descent_Lux.jl` keeps the random-feature
  double-descent experiment.
- `lecture_02_04_Gentle_DNN_Lux.jl` replaces Keras `Sequential` pedagogy with
  Lux regression and synthetic classification.
- `lecture_02_05_Training_Instrumentation_Lux.jl` replaces TensorBoard
  callbacks with plain Julia metric logs, validation history, gradient norms,
  and best-run metadata.
- `lecture_02_06_Lux_Training_Fundamentals.jl` replaces the PyTorch API tour
  with explicit Lux parameter/state training and Zygote gradient inspection.
- `lecture_02_07_Genz_Approximation_and_Loss_Functions_Lux.jl` keeps the Genz
  approximation and shared loss-kernel comparison.
- `lecture_02_08_MLP_LSTM_Transformer_Edgeworth_Cycles_Lux.jl` is a Phase-0
  memory-ladder preview using Lux-compatible feature maps, not full LSTM or
  Transformer parity.
- `lecture_02_09_Transformer_InContext_AR1_Lux.jl` keeps the in-context AR(1)
  forecasting idea with a compact Lux forecaster.

## Running

Run notebooks from `code/`. Preserve `RUN_MODE = "smoke"` and `SEED = 0` when
they are present. This lecture uses TensorFlow/Keras, PyTorch, NumPy, SciPy,
pandas, Matplotlib, scikit-learn, Jupyter, and TensorBoard from the root setup.

`lecture_02_05_Tensorboard.ipynb` writes `logs/` and `checkpoints/` relative to
the runtime working directory. `lecture_02_04_Gentle_DNN.ipynb` may download or
cache Fashion-MNIST on first run.

For Julia previews, run Jupyter notebooks from `code_julia/` and use the shared
`../../../julia` project. All current Julia notebooks are Lux-native and import
`DLEFJulia`; keep feature-by-batch arrays at Lux boundaries and preserve
`RUN_MODE`/`SEED`. Keep explicit calls such as `prediction, st_new =
model(batch.x, ps, st)` visible; do not wrap the notebooks in Keras-, PyTorch-,
or JAX-style APIs.

The Julia files intentionally redesign API-teaching material rather than
mimicking Python frameworks: TensorBoard instrumentation becomes plain Julia
metric logging, the PyTorch introduction becomes Lux training fundamentals, and
the sequence/Transformer notebooks are compact pedagogical Lux previews.
Representative smoke coverage is split across
`julia/test/smoke/wave1_notebooks.jl` and
`julia/test/smoke/wave2_notebooks.jl`.

## Editing

Keep `slides/figures/` paths stable when editing slides. Do not hand-edit large
notebook JSON unless the change is minimal and source-cell scoped. Do not clear,
renumber, or regenerate Python notebook outputs, notebook cell order, TensorBoard
logs, checkpoints, cached Fashion-MNIST data, or generated figures just to
inspect this lecture.
