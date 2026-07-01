# Lecture 10: Sequence-space DEQNs

A modern DEQN variant where the policy reads a long shock history instead of a current-state vector. Following Azinovic-Yang-Žemlička (2025).

`gpu-recommended` · `long` · builds on [Lecture 09](../lecture_09_heterogeneous_agents_youngs_method/README.md)

> 📑 **Slides:** [lecture_10_sequence_space_deqns.pdf](slides/lecture_10_sequence_space_deqns.pdf)  
> 📓 **Notebooks:** [start here](code/lecture_10_05_SequenceSpace_BrockMirman.ipynb) (4 in [`code/`](code/))  
> 📚 **Further reading:** [curated list](../../readings/links_by_lecture/lecture_10.md)  
> 📖 **Script:** §6.7 (Sequence-space DEQNs)

## What this lecture covers

- **The sequence-space idea.** Replace the high-dimensional state with the last ~80 shock realizations; the network learns the residual map directly.
- **Why it generalizes.** The same template handles multi-equation systems with multiple shock channels without re-engineering the input.
- **Brock-Mirman warm-up.** Sequence-space DEQN with an 80-step shock history; verify the policy.
- **Krusell-Smith benchmark.** The same template on the canonical heterogeneous-agent benchmark.
- **Self-study extensions.** Multi-country IRBC and a borrowed JAX tutorial port (`KrusellSmith_Tutorial_CPU.ipynb`).

## Learning objectives

After this lecture you can:

- Build the shock-history input pipeline for a sequence-space DEQN.
- Train a sequence-space DEQN on Brock-Mirman with an 80-step shock history and verify the policy.
- Extend the same template to Krusell-Smith.
- Explain why sequence-space DEQNs handle multi-shock systems gracefully.

## Slides

- [`slides/lecture_10_sequence_space_deqns.pdf`](slides/lecture_10_sequence_space_deqns.pdf)
- [`slides/lecture_10_sequence_space_deqns.tex`](slides/lecture_10_sequence_space_deqns.tex)

## Code

### Julia/Lux/Pluto preview

- [`code_julia/lecture_10_05_SequenceSpace_BrockMirman_Lux.jl`](code_julia/lecture_10_05_SequenceSpace_BrockMirman_Lux.jl) translates the Brock-Mirman sequence-space warm-up with canonical shock-history layout helpers.
- [`code_julia/lecture_10_05b_SequenceSpace_IRBC_Lux.jl`](code_julia/lecture_10_05b_SequenceSpace_IRBC_Lux.jl) ports the multi-country IRBC sequence-space residual with irreversible-investment complementarity.
- [`code_julia/lecture_10_06_SequenceSpace_KrusellSmith_Lux.jl`](code_julia/lecture_10_06_SequenceSpace_KrusellSmith_Lux.jl) gives a classroom-scale Krusell-Smith sequence-space actor with Young distribution propagation.
- [`code_julia/lecture_10_KrusellSmith_Tutorial_CPU_Lux.jl`](code_julia/lecture_10_KrusellSmith_Tutorial_CPU_Lux.jl) is a CPU-only Lux companion to the shape-preserving Krusell-Smith tutorial.

### Python notebooks

- [`code/lecture_10_05_SequenceSpace_BrockMirman.ipynb`](code/lecture_10_05_SequenceSpace_BrockMirman.ipynb)
- [`code/lecture_10_05b_SequenceSpace_IRBC.ipynb`](code/lecture_10_05b_SequenceSpace_IRBC.ipynb)
- [`code/lecture_10_06_SequenceSpace_KrusellSmith.ipynb`](code/lecture_10_06_SequenceSpace_KrusellSmith.ipynb)
- [`code/lecture_10_KrusellSmith_Tutorial_CPU.ipynb`](code/lecture_10_KrusellSmith_Tutorial_CPU.ipynb)

## In the lecture script

§6.7 (Sequence-space DEQNs). The full chapter map is in [`script_to_lectures.md`](../../lecture_script/script_to_lectures.md).

## Readings

Curated bibliography for this lecture: [`lecture_10.md`](../../readings/links_by_lecture/lecture_10.md). The full BibTeX is in [`readings/bibliography.bib`](../../readings/bibliography.bib).

---

| ← Previous | Next → |
|---|---|
| [**Lecture 09: Heterogeneous agents and Young's method**](../lecture_09_heterogeneous_agents_youngs_method/README.md)<br><sub>Young's histogram, Krusell-Smith, continuum-of-agents DEQN</sub> | [**Lecture 11: Physics-informed neural networks (PINNs)**](../lecture_11_pinns/README.md)<br><sub>ODE / PDE PINNs, soft vs hard BCs, cake-eating HJB, Black-Scholes</sub> |

[↑ Course map](../../COURSE_MAP.md)
