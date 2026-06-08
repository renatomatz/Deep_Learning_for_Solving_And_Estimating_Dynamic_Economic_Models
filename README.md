# Deep Learning for Solving and Estimating Dynamic Models in Economics and Finance

*Deep learning for the recursive, stochastic, high-dimensional dynamic
models that economists actually solve, with all materials open source,
runnable, and self-contained.*

<p align="center">
  <a href="lecture_script/Deep_Learning_for_Solving_And_Estimating_Dynamic_Economic_Models.pdf"><img src="assets/hero/deep_learning_dynamic_models_hero.png" width="95%" alt="Deep Learning for Solving and Estimating Dynamic Models in Economics and Finance"></a>
</p>

<p align="center">
  <a href="COURSE_MAP.md"><img alt="18 lectures" src="https://img.shields.io/badge/lectures-18-1f4e79?style=for-the-badge"></a>
  <a href="lectures/"><img alt="Runnable notebooks" src="https://img.shields.io/badge/notebooks-runnable-1f4e79?style=for-the-badge"></a>
  <a href="lecture_script/Deep_Learning_for_Solving_And_Estimating_Dynamic_Economic_Models.pdf"><img alt="Companion script" src="https://img.shields.io/badge/companion%20script-PDF-1f4e79?style=for-the-badge"></a>
  <br/>
  <a href="lectures/lecture_06_agentic_programming/"><img alt="AI-coding workshop" src="https://img.shields.io/badge/AI%20coding-workshop-1f4e79?style=for-the-badge"></a>
  <a href="#syllabus"><img alt="Exercises with solutions" src="https://img.shields.io/badge/exercises-with%20solutions-1f4e79?style=for-the-badge"></a>
  <a href="LICENSE"><img alt="Open source, self-paced, no enrollment" src="https://img.shields.io/badge/open%20source-self--paced-1f4e79?style=for-the-badge"></a>
</p>

<h3 align="center">
  Course author: <a href="https://sischei.github.io/">Simon Scheidegger</a>
</h3>
<p align="center"><em>University of Lausanne &nbsp;·&nbsp; Grantham Research Institute, London School of Economics</em></p>

<br/>

## About this course

**Classical grid methods hit a wall.** Modern macroeconomics, finance,
and climate economics have outgrown the grid-based numerical methods
that dominated a generation ago. Once you add heterogeneous agents,
overlapping generations, occasionally binding constraints,
continuous-time dynamics, or coupled climate-economic interactions,
the state space becomes too large for tensor-product grids and
classical methods (projection, value-function iteration, perturbation)
break down. If you are trying to solve models with ten or more state
dimensions, estimate them, or design policy under parameter uncertainty,
you need a different toolbox.

**This course teaches that toolbox.** A coherent set of deep-learning
methods built *for* the recursive, stochastic, often high-dimensional
models economists actually solve. The methods work by letting economic
structure drive the learning problem: equilibrium conditions, Bellman
equations, and PDEs become the residual loss, e.g., in Deep
Equilibrium Nets or Physics-Informed Neural Networks (an unsupervised
setup), or they shape the simulator that generates the (input, output)
pairs a deep surrogate or Gaussian process then learns in the standard
supervised way. You will build each method from scratch on benchmarks
where the answer is known (Brock–Mirman, cake-eating, Black–Scholes)
before applying them to models where it is not (IRBC, OLG with 56
cohorts, Krusell–Smith with a continuum, continuous-time heterogeneous
agents, climate-economic coupling). The course is hands-on by design:
every method is paired with runnable Jupyter notebooks that put the
principles in plain sight, so you see exactly how each loss is
assembled, each gradient is taken, and each equilibrium is solved
rather than reading about it. By the end you will be able to
solve models that were out of reach with classical tools, estimate
them when re-solving is too expensive, and design policies that take
parameter uncertainty seriously.

**Everything is self-contained and open source.** A textbook-length
[companion script](lecture_script/Deep_Learning_for_Solving_And_Estimating_Dynamic_Economic_Models.pdf),
**18 paired lectures** with slides and runnable Jupyter notebooks,
exercises with full solutions, a hands-on workshop on AI coding agents
as research partners (Lecture 06), and a curated bibliography linking
out to the underlying papers. There is no enrollment, no cohort, no
deadline, just pick the method you need and dig in.

> **A working snapshot, not a definitive survey.** The selection of
> methods, papers, and implementation choices reflects what I
> currently find to be some of the most useful entry points for
> economists and finance researchers starting to work with deep
> learning in dynamic models, and the bibliography is necessarily
> incomplete. Treat the material as a practical entry point to study,
> run, adapt, and question.

> ### 🚀 Start here
>
> - **Jump in:** [Lecture 02 — Intro to deep learning](lectures/lecture_02_intro_deep_learning/README.md)
> - **New to Python?** Begin with the [Python primer (Lecture 01)](lectures/lecture_01_python_primer/README.md)
> - **Want the panoramic view?** Open the [course map](COURSE_MAP.md)

## What you will learn

This course teaches a coherent set of deep-learning methods for the
recursive, stochastic, often high-dimensional models that show up in
modern macroeconomics, asset pricing, and climate-economic policy work.
Five capabilities, each motivated below.

### 1. Solving recursive equilibrium models with neural networks

Most quantitative macro models reduce to functional equations (Euler
equations, Bellman equations, market-clearing conditions) that
classical methods (projection, value-function iteration, perturbation)
struggle with once the state space gets large or the policy is
nonsmooth. **Deep Equilibrium Nets (DEQNs)** parameterize the policy
or value function with a neural network and minimize the
equilibrium-equation residuals directly via stochastic gradient descent,
sidestepping a curse-of-dimensionality grid. The companion **Physics-Informed
Neural Networks (PINNs)** do the same for continuous-time models: the
loss is the residual of a Hamilton–Jacobi–Bellman equation, automatic
differentiation supplies the derivatives, and there is no mesh.
You will build both end-to-end on benchmarks where the answer is known
(Brock–Mirman, cake-eating, Black–Scholes) and then on models where it
is not (IRBC, OLG with 56 cohorts, Krusell–Smith with a continuum of
agents, continuous-time heterogeneous agents).

### 2. Surrogates, Gaussian processes, and Bayesian active learning

Many calibration, estimation, and policy-evaluation tasks call the
underlying model thousands or millions of times. A **deep surrogate
model** replaces that expensive call with a cheap, differentiable
neural network trained on a few hundred or thousand simulator outputs.
A **Gaussian process (GP)** does the same with built-in uncertainty
quantification, which lets **Bayesian active learning (BAL)** pick the
next training point optimally instead of throwing samples at a
hypercube. We then push GPs to high dimension via **active subspaces**
and **deep kernel learning**, and use them inside value-function
iteration (ASGP-VFI) as a competitor to DEQNs.

### 3. Structural estimation via simulated method of moments

Once a deep surrogate is in place, **simulated method of moments
(SMM)** estimation becomes a small optimization over the surrogate
rather than a brutal repeated re-solve of the structural model. You
will run single- and joint-parameter SMM on a deep surrogate of
Brock–Mirman and see how the estimator behaves under realistic noise
and identification challenges.

### 4. Deep UQ and Pareto-improving climate policy

Integrated assessment models (DICE, CDICE) carry parameters whose
true values are deeply uncertain, equilibrium climate sensitivity
being the textbook example. Plugging point estimates in and reading
off a single social cost of carbon is misleading; averaging the
uncertainty out before optimization is worse, because the policy you
would choose under expected damages is generally not the policy you
would choose if you took the tail risk seriously.

The course teaches a complete pipeline that addresses this directly.
We solve a stochastic IAM with DEQNs under Epstein–Zin preferences,
build GP surrogates for the quantities of interest with Bayesian
active learning, and run global sensitivity analysis (Sobol, Shapley
effects) to localize where the policy is actually sensitive to which
parameters. On top of that surrogate we then **design constrained
Pareto-improving carbon-tax policies**: tax paths that, for every
plausible parameter draw (or every cohort, or every generation), leave
no agent worse off than the business-as-usual baseline while strictly
improving welfare for at least one. This turns "what should the
carbon tax be?" from a single number computed under a single
calibration into a defensible policy menu that respects who bears the
risk and who benefits, *without* averaging the uncertainty away.

### 5. An AI-assisted research-coding workflow

Modern empirical and computational economics benefits enormously from
using AI coding agents (Claude Code) as research partners, but only
when the workflow is set up deliberately. **Lecture 06** is a
hands-on workshop that teaches the orientation, prompt patterns,
project memory (`CLAUDE.md`), custom skills, subagents, and hooks
that turn an LLM from a clever autocomplete into a real research
collaborator, paired with twelve self-paced exercises so you walk
out with reusable templates rather than slideware.

## How to use this course

Different readers come in with different goals, so pick the entry point
that fits yours:

- 🚀 **I want a guided start.** Open the
  [Python primer (Lecture 01)](lectures/lecture_01_python_primer/README.md)
  if you need it, then follow the **Complete path** in
  [`COURSE_MAP.md`](COURSE_MAP.md). It walks through all 18 lectures
  in their natural order.
- 🎯 **I have a specific topic in mind.** Jump straight to the
  **syllabus** below.
- 🧪 **I want the research-workflow training first.** Jump to
  [Lecture 06, agentic programming](lectures/lecture_06_agentic_programming/README.md),
  then come back to the rest of the sequence.
- 📖 **I want a textbook.** Read the chapter-based
  [companion script](lecture_script/Deep_Learning_for_Solving_And_Estimating_Dynamic_Economic_Models.pdf); each chapter
  links to one or more lectures via
  [`script_to_lectures.md`](lecture_script/script_to_lectures.md).

For each lecture, the workflow is the same:

1. read the relevant chapter or section of the script;
2. step through the lecture's slide deck (under `slides/`);
3. run the lecture's notebooks under `code/` (numbered in suggested
   order; files ending in `_Exercises_Blanks.ipynb` /
   `_Exercises_Solutions.ipynb` are paired exercise/solution sets).

Every long-running notebook exposes a `RUN_MODE` switch near the top with
three values: `"smoke"` (CPU-bounded, runs in minutes for a sanity check
or CI), `"teaching"` (laptop figures, intermediate fidelity), and
`"production"` (full reproduction, published-figure quality). Each
notebook also fixes a `SEED` for reproducibility.

## Syllabus

| If you want to learn… | Read | Notebooks |
|---|---|---|
| **Python warm-up** (skip if you write Python every day) | [Lecture 01](lectures/lecture_01_python_primer/README.md) | Jupyter, basic data structures, NumPy, plotting, classes |
| **Deep-learning fundamentals** (training, generalization, sequence models) | [Lecture 02](lectures/lecture_02_intro_deep_learning/README.md) | MLP, LSTM, Transformer on Edgeworth cycles, double descent, Genz approximations |
| **Deep Equilibrium Nets (DEQNs)**, the central method | [Lecture 03](lectures/lecture_03_deep_equilibrium_nets/README.md) | Brock–Mirman (deterministic, stochastic), Fischer–Burmeister constraints, six loss kernels |
| **Large-scale nonlinear DSGE** (IRBC) | [Lecture 04](lectures/lecture_04_irbc_with_deqns/README.md) | International real business cycle with DEQNs |
| **Architecture search and loss balancing** (NAS, ReLoBRaLo) | [Lecture 05](lectures/lecture_05_nas_loss_normalization/README.md) | Random search, Hyperband, ReLoBRaLo, SoftAdapt, GradNorm |
| **Agentic programming** (AI coding agents as research partners) | [Lecture 06](lectures/lecture_06_agentic_programming/README.md) | Claude Code workflow, prompts, project memory, custom skills, subagents, hooks, plus a 12-exercise workshop |
| **Automatic differentiation for DEQNs** | [Lecture 07](lectures/lecture_07_autodiff_for_deqns/README.md) | Lagrangian primitives, two-tape gradients, IRBC autodiff |
| **OLG with DEQNs** | [Lecture 08](lectures/lecture_08_olg_models_deqns/README.md) | Analytic OLG, 56-cohort benchmark, Fischer–Burmeister borrowing constraints |
| **Heterogeneous agents and Young's method** | [Lecture 09](lectures/lecture_09_heterogeneous_agents_youngs_method/README.md) | Young's histogram, Krusell–Smith, continuum-of-agents DEQN |
| **Sequence-space DEQNs** | [Lecture 10](lectures/lecture_10_sequence_space_deqns/README.md) | Brock–Mirman, IRBC, Krusell–Smith with shock-history inputs |
| **Physics-informed neural networks (PINNs)** | [Lecture 11](lectures/lecture_11_pinns/README.md) | ODE / PDE PINNs, soft vs hard BCs, cake-eating HJB, Black–Scholes |
| ↳ **Continuous-time HA, theory** | [Lecture 12](lectures/lecture_12_continuous_time_ha_theory/README.md) | HJB, Kolmogorov-forward, master equation, Ito calculus |
| ↳ **Continuous-time HA, numerics** | [Lecture 13](lectures/lecture_13_continuous_time_ha_numerics/README.md) | Achdou–Han–Lasry–Lions–Moll finite-difference scheme, PINN for HJB-KFE, continuous-time Aiyagari |
| **Surrogates, Gaussian processes, deep kernels** | [Lecture 14](lectures/lecture_14_surrogates_and_gps/README.md) | Surrogate primer, GP regression, BAL, active subspaces, deep kernel learning, GP-VFI |
| **Structural estimation via SMM** | [Lecture 15](lectures/lecture_15_structural_estimation_smm/README.md) | Brock–Mirman SMM (single- and joint-parameter) on a deep surrogate |
| **Climate economics and IAMs (DICE, CDICE)** | [Lecture 16](lectures/lecture_16_climate_economics_iams/README.md) | DICE / CDICE simulation, deterministic and stochastic CDICE-DEQN |
| ↳ **Deep UQ and Pareto-improving carbon-tax design** | [Lecture 17](lectures/lecture_17_deep_uq_pareto_improving_policy/README.md) | GP surrogates, Bayesian active learning, Sobol / Shapley, constrained Pareto-improving carbon-tax rules |
| **Synthesis, when to use which method** | [Lecture 18](lectures/lecture_18_course_wrap_up/README.md) | Decision guide and outlook |

For the full table including compute and time budgets, prerequisites,
and the visual prerequisite diagram, see
[`COURSE_MAP.md`](COURSE_MAP.md).

## Setup

Notebooks run on **Python 3.10+**. Two reproducible setups:

```bash
# pip
pip install -r requirements.txt

# conda
conda env create -f environment.yml
conda activate dlef
```

Main dependencies: NumPy, SciPy, pandas, Matplotlib, scikit-learn,
TensorFlow ≥ 2.15, PyTorch ≥ 2.0, JAX (selected notebooks), GPyTorch
and BoTorch (Lecture 13).

### GPU notes (JAX)

A few notebooks use JAX with CUDA. If JAX cannot locate your CUDA
NVVM directory at import time, set the XLA flag in your shell before
launching Jupyter:

```bash
export XLA_FLAGS="--xla_gpu_cuda_data_dir=/path/to/cuda"
```

This is environment-specific and is intentionally kept out of the
notebooks themselves so they remain portable.

## Repository at a glance

```
.
├── README.md             ← you are here
├── COURSE_MAP.md         ← detailed map, learning paths, prerequisite diagram
├── lectures/             ← 18 lecture folders (lecture_XX_*)
│   └── lecture_*/
│       ├── README.md         summary, slides, code, prerequisites, readings
│       ├── slides/           PDFs and .tex sources
│       ├── code/             notebooks, supporting .py modules, data files
│       └── figures/          (optional) lecture-specific figure assets
├── lecture_script/       ← textbook-length companion script
├── readings/             ← per-lecture link guides + bibliography.bib
└── assets/               ← hero figure, generated figures, attributions
```

## Glossary

The script's Appendix A is the canonical glossary. A grep-able copy
lives at [`lecture_script/glossary.md`](lecture_script/glossary.md).

## Readings and copyright

Most readings are journal articles, working papers, or copyrighted
books. The public repository **links** to publishers, DOIs, arXiv, or
author pages rather than redistributing PDFs. Per-lecture link guides
live under
[`readings/links_by_lecture/`](readings/links_by_lecture/);
the full bibliography is in
[`readings/bibliography.bib`](readings/bibliography.bib).

Course author: **[Simon Scheidegger](https://sischei.github.io/)** (University of Lausanne).
Code is MIT-licensed; text, slides, script, and figures are CC0
(see [`LICENSE`](LICENSE) for both).

## Citation

If this work was useful in your research, please cite the arXiv
manuscript (preferred) or the SSRN version:

**arXiv (preferred):**

```bibtex
@misc{scheidegger2026deeplearningsolvingestimating,
  title         = {Deep Learning for Solving and Estimating Dynamic Models in Economics and Finance},
  author        = {Simon Scheidegger},
  year          = {2026},
  eprint        = {2605.14493},
  archivePrefix = {arXiv},
  primaryClass  = {econ.GN},
  url           = {https://arxiv.org/abs/2605.14493}
}
```

**SSRN:**

```bibtex
@article{scheidegger_2026_ssrn,
  title   = {Deep Learning for Solving and Estimating Dynamic Models in Economics and Finance},
  author  = {Scheidegger, Simon},
  year    = {2026},
  month   = {5},
  doi     = {10.2139/ssrn.6758340},
  url     = {https://ssrn.com/abstract=6758340},
  journal = {Available at SSRN 6758340},
  note    = {Posted 13 May 2026}
}
```

## Errata, contributions, and contact

Questions, corrections, and pull requests are welcome on
[GitHub](https://github.com/sischei/Deep_Learning_for_Solving_And_Estimating_Dynamic_Economic_Models).
By contributing you agree that your contribution is licensed under the
same terms as this repository.
