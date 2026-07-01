# AGENTS.md

## Purpose

Lecture 16 introduces climate economics and integrated assessment models: DICE,
CDICE, deterministic CDICE-DEQN, and stochastic CDICE-DEQN with productivity
shocks.

## Map

- Start with `README.md`.
- `slides/lecture_16_climate_economics_iams.tex` is the slide source.
- `slides/fig/` contains a large climate/IAM figure set used by the deck and
  related lectures.
- `code/lecture_16_01_Climate_Exercise.ipynb` is the DICE/climate exercise.
- `code/lecture_16_02_DICE_DEQN_Library_Port.ipynb` ports the DICE-DEQN library
  ideas into the course.
- `code/lecture_16_03_Stochastic_DICE_DEQN.ipynb` adds stochastic CDICE-DEQN
  elements.
- `code_julia/lecture_16_01_Climate_Exercise_Lux.ipynb` is the pure Julia DICE
  warm-up.
- `code_julia/lecture_16_02_DICE_DEQN_Library_Port_Lux.ipynb` is the deterministic
  CDICE/Lux smoke preview.
- `code_julia/lecture_16_03_Stochastic_DICE_DEQN_Lux.ipynb` is the stochastic
  CDICE/Lux smoke preview.

## Running And Editing

This lecture has both TensorFlow/Python notebooks and Julia/Lux previews.
Do not treat either stack as the sole source of guidance. Preserve the economic
objects and warnings across both tracks: DICE simulation, deterministic CDICE,
stochastic CDICE, SCC diagnostics, AR(1) productivity shocks, and
Gauss-Hermite expectations.

Use the root environment and prefer smoke settings when present. Climate DEQN
notebooks can be long and dependency-sensitive.

Do not rerun notebooks casually. `lecture_16_02_DICE_DEQN_Library_Port.ipynb`
and `lecture_16_03_Stochastic_DICE_DEQN.ipynb` use TensorFlow DEQN training.
`RUN_MODE = "smoke"` is fast but can intentionally fail many reference checks;
`teaching` and `production` are much longer and may churn saved outputs, warning
text, timings, and plots.

Preserve the distinction between DICE simulation, deterministic CDICE-DEQN, and
stochastic CDICE-DEQN. Do not silently replace production-code references with
teaching approximations. The notebooks mention production references from
external `DEQN_for_IAMs`; the local notebooks are pedagogical ports, not the full
Hydra/Horovod/checkpointing stack.

Run Julia previews from `code_julia/` with the shared `../../../julia` project.
Each preview should activate that project, import `DLEFJulia`, preserve
`RUN_MODE = "smoke"` and `SEED = 0` defaults, and keep Lux-native
feature-by-batch and explicit parameter/state conventions. They are covered by
`julia/test/smoke/wave6_notebooks.jl`. The smoke harness uses `include(path)`,
so it checks runtime plumbing and finite returns rather than notebook frontend
behavior, reference-code parity, or economic convergence.

`lecture_16_02_DICE_DEQN_Library_Port_Lux.jl` trains only a tiny smoke pass and
uses `CDICETeachingPolicy` for diagnostics; do not read smoke output as
trained-policy reference parity. `lecture_16_03_Stochastic_DICE_DEQN_Lux.jl`
adds an AR(1) productivity state, 5-node Gauss-Hermite expectations, seeded Monte
Carlo, and small fan-chart inputs.

Treat the Julia climate notebooks as smoke-first previews with production-parity
caveats. Do not silently substitute simplified teaching policies for
production-code claims, and do not add climate dependencies or move shared
calibration logic out of `DLEFJulia` without updating the root Julia project and
the wave6 smoke coverage.

Compile slides from `slides/`; the TeX uses `\graphicspath{{fig/}}`.

## Media And Attribution

Several climate slide assets have uncertain provenance or review-required status
in `../../assets/attributions.yml`. Check that file before reusing, moving, or
copying climate figures. There is a known attribution typo around `damge.png`
versus `damage.png`.

This attribution warning applies equally when a Julia preview or generated
figure wants to reuse climate imagery. Do not copy Lecture 16 climate assets into
other lecture folders, rename them, or replace duplicates without checking the
central attribution registry and documenting any intentional divergence.
