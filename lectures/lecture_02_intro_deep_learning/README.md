# Lecture 02: Introduction to deep learning

The working knowledge of deep learning that the rest of the course assumes, with economics-flavoured worked examples throughout.

`cpu-standard` · `long` · builds on [Lecture 01](../lecture_01_python_primer/README.md)

> 📑 **Slides:** [lecture_02_intro_deep_learning.pdf](slides/lecture_02_intro_deep_learning.pdf)  
> 📓 **Notebooks:** [start here](code/lecture_02_01_BasicML_intro.ipynb) (9 in [`code/`](code/))  
> 📚 **Further reading:** [curated list](../../readings/links_by_lecture/lecture_02.md)  
> 📖 **Script:** §1.1-1.4 (Foundations and function approximation), §1.5-1.9 (Optimization, depth, and regularization), §1.10-1.11 (Generalization, sequence models)

## What this lecture covers

- **Classical ML and the bias-variance trade-off.** Linear regression, classification, and unsupervised learning as a foundation for everything that follows.
- **Stochastic gradient descent.** SGD, mini-batches, momentum, and adaptive variants (Adam, RMSProp); when each one is the right default.
- **Deep neural networks.** Depth, width, activation choices, and the **double-descent** phenomenon on a controlled synthetic example.
- **Sequence models.** MLPs, LSTMs, and small Transformers compared on Edgeworth-cycle data, exposing the **memory ladder** of architectures.
- **Tooling.** TensorFlow and PyTorch side by side, plus TensorBoard for instrumenting a training run.

## Learning objectives

After this lecture you can:

- Implement SGD by hand and explain mini-batch, momentum, and adaptive variants.
- Train an MLP and a deep neural network end-to-end in TensorFlow and in PyTorch.
- Reproduce double descent on a controlled synthetic problem.
- Compare MLP, LSTM, and small-Transformer architectures on Edgeworth-cycle data and read off the memory ladder.
- Use TensorBoard to instrument a training run.

## Slides

- [`slides/lecture_02_intro_deep_learning.pdf`](slides/lecture_02_intro_deep_learning.pdf)
- [`slides/lecture_02_intro_deep_learning.tex`](slides/lecture_02_intro_deep_learning.tex)

## Code

### Julia/Lux preview

- [`code_julia/lecture_02_00_Lux_orientation.ipynb`](code_julia/lecture_02_00_Lux_orientation.ipynb) introduces the shared Julia project, notebook workflow, `RUN_MODE`, `SEED`, feature-by-batch arrays, Lux `model, ps, st`, and the first training-loop helper.
- [`code_julia/lecture_02_01_BasicML_intro_Lux.ipynb`](code_julia/lecture_02_01_BasicML_intro_Lux.ipynb) translates the first supervised-learning examples with OLS and a small Lux MLP.
- [`code_julia/lecture_02_02_GradientDescent_and_StochasticGradientDescent_Lux.ipynb`](code_julia/lecture_02_02_GradientDescent_and_StochasticGradientDescent_Lux.ipynb) translates the gradient-descent and cricket-data SGD examples.
- [`code_julia/lecture_02_03_Double_Descent_Lux.ipynb`](code_julia/lecture_02_03_Double_Descent_Lux.ipynb) translates the random-feature double-descent experiment.
- [`code_julia/lecture_02_04_Gentle_DNN_Lux.ipynb`](code_julia/lecture_02_04_Gentle_DNN_Lux.ipynb) translates the introductory deep-network regression and classification examples into Lux.
- [`code_julia/lecture_02_05_Training_Instrumentation_Lux.ipynb`](code_julia/lecture_02_05_Training_Instrumentation_Lux.ipynb) translates training instrumentation with plain Julia metric logs and best-run metadata.
- [`code_julia/lecture_02_06_Lux_Training_Fundamentals.ipynb`](code_julia/lecture_02_06_Lux_Training_Fundamentals.ipynb) replaces the PyTorch API introduction with explicit Lux parameter/state training fundamentals.
- [`code_julia/lecture_02_07_Genz_Approximation_and_Loss_Functions_Lux.ipynb`](code_julia/lecture_02_07_Genz_Approximation_and_Loss_Functions_Lux.ipynb) translates the Genz approximation and loss-kernel comparison.
- [`code_julia/lecture_02_08_MLP_LSTM_Transformer_Edgeworth_Cycles_Lux.ipynb`](code_julia/lecture_02_08_MLP_LSTM_Transformer_Edgeworth_Cycles_Lux.ipynb) translates the sequence-model memory ladder with Phase-0 Lux-compatible feature maps.
- [`code_julia/lecture_02_09_Transformer_InContext_AR1_Lux.ipynb`](code_julia/lecture_02_09_Transformer_InContext_AR1_Lux.ipynb) translates the AR(1) in-context forecasting idea into a compact Lux forecaster.
- See [`../../julia/README.md`](../../julia/README.md) for Julia environment instantiation, test, and notebook execute-smoke commands.

### Python notebooks

- [`code/SGD_data.txt`](code/SGD_data.txt)
- [`code/lecture_02_01_BasicML_intro.ipynb`](code/lecture_02_01_BasicML_intro.ipynb)
- [`code/lecture_02_02_GradientDescent_and_StochasticGradientDescent.ipynb`](code/lecture_02_02_GradientDescent_and_StochasticGradientDescent.ipynb)
- [`code/lecture_02_03_Double_Descent.ipynb`](code/lecture_02_03_Double_Descent.ipynb)
- [`code/lecture_02_04_Gentle_DNN.ipynb`](code/lecture_02_04_Gentle_DNN.ipynb)
- [`code/lecture_02_05_Tensorboard.ipynb`](code/lecture_02_05_Tensorboard.ipynb)
- [`code/lecture_02_06_PyTorch_intro.ipynb`](code/lecture_02_06_PyTorch_intro.ipynb)
- [`code/lecture_02_07_Genz_Approximation_and_Loss_Functions.ipynb`](code/lecture_02_07_Genz_Approximation_and_Loss_Functions.ipynb)
- [`code/lecture_02_08_MLP_LSTM_Transformer_Edgeworth_Cycles.ipynb`](code/lecture_02_08_MLP_LSTM_Transformer_Edgeworth_Cycles.ipynb)
- [`code/lecture_02_09_Transformer_InContext_AR1.ipynb`](code/lecture_02_09_Transformer_InContext_AR1.ipynb)

## In the lecture script

§1.1-1.4 (Foundations and function approximation), §1.5-1.9 (Optimization, depth, and regularization), §1.10-1.11 (Generalization, sequence models). The full chapter map is in [`script_to_lectures.md`](../../lecture_script/script_to_lectures.md).

## Readings

Curated bibliography for this lecture: [`lecture_02.md`](../../readings/links_by_lecture/lecture_02.md). The full BibTeX is in [`readings/bibliography.bib`](../../readings/bibliography.bib).

---

| ← Previous | Next → |
|---|---|
| [**Lecture 01: Python primer**](../lecture_01_python_primer/README.md)<br><sub>Jupyter, basic data structures, NumPy, plotting, classes</sub> | [**Lecture 03: Deep Equilibrium Nets**](../lecture_03_deep_equilibrium_nets/README.md)<br><sub>Brock-Mirman (deterministic, stochastic), Fischer-Burmeister constraints, six loss kernels</sub> |

[↑ Course map](../../COURSE_MAP.md)
