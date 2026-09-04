# Is suppressed emergent misalignment readable without the trigger?

MATS 12.0 application task, Neel Nanda stream. September 2026.

Inoculation prompting removes emergent misalignment from behavioral evaluations. This
study asks whether the misalignment is still visible in activation space on trigger-free
inputs, by projecting onto the misalignment direction of Soligo et al. (2506.11618).

**Headline.** Trigger-free, the gated model projects reliably below the correct-advice control 
(R = −0.94, CI −1.70 to −0.50). However, the effect is a model-level mean shift, not a per-response 
signal: with a per-response standard deviation around 4.2, both the INOC-to-BENIGN gap (3.28) and 
the EM-to-BENIGN gap (3.50) are under one standard deviation, so the distributions overlap substantially. 
This direction distinguishes populations of responses, not individual ones. Any monitoring claim 
built on it would need many samples per model, and a per-response threshold calibrated to catch the 
openly misaligned model would not flag the gated one.

## Contents

- `PROTOCOL.md` — pre-registered protocol, committed before any training. Section 15 is
  the decisions log.
- `scripts/` — generation, judging, activation pooling, analysis, figures, blind audit.
- `results/responses/` — all judged generations. Every behavioral number comes from here.
- `results/analysis_*.json` — projections and retention ratios. `_B`/`_Bt` are the leaked
  measurement, kept for the before-and-after; `_H`/`_Ht`/`_HA`/`_HAt` are the held-out set.
- `results/figures/` — figures 1-6.
- `results/audit_sample.csv` — the blind judge audit with my labels.

## Reproducing

Requires the model organisms repo (`clarifying-EM/model-organisms-for-EM`) and its
training data. **The training data is not redistributed here.** 

## LLM use

Design, code, and drafting were LLM-assisted (Claude). Execution, verification, and
analysis are mine. Details in the write-up's disclosure section.