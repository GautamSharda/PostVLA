# Experiment Results

`results.json` is the compact index; `raw/` contains the original per-attempt summaries.
Videos and model weights are excluded because they are large, but every raw result retains
its original checkpoint and video paths for matching against the network-volume archive.

## Main Comparison

The 100-episode SFT, stage-1, stage-2 Gaussian, and stage-2 zero-latent evaluations share
the same ordered episode-ID sequence. This makes the per-episode comparisons paired.

| Model and decode | On pad | Center + lift |
| --- | ---: | ---: |
| Distilled Torch SFT, Gaussian ODE | 69/100 | 53/100 |
| RL stage 1 (5 PPO iterations), Gaussian ODE | 64/100 | 55/100 |
| RL stage 2 (10 total PPO iterations), Gaussian ODE | 70/100 | 52/100 (batch 1), 55/100 (batch 20) |
| RL stage 2, zero-latent ODE | 90/100 | 82/100 |

Against the SFT Gaussian baseline, the stage-2 zero-latent run changed broad outcomes as
follows: 66 both succeeded, 3 SFT-only successes, 24 RL-only successes, and 7 neither.
For center + lift: 46 both, 7 SFT-only, 36 RL-only, and 11 neither.

## Interpretation

- Under the same standard fresh-unit-Gaussian ODE decode, stage 2 is essentially tied with
  SFT on broad placement (`70/100` versus `69/100`) and is not clearly better on strict placement.
- The large `90/100` result appears only after changing the stage-2 initial latent to zero.
- A fixed unit-Gaussian latent performed poorly (`8/20`), while a fixed half-scale latent reached
  `18/20`; fresh half-scale Gaussian reached `16/20`.
- Restricting unit Gaussian noise to the six executed action dimensions did not resolve the gap
  (`16/20` broad, `12/20` strict), so the 26 padded dimensions are not a sufficient explanation.
- The stochastic flow-noise evaluation was weaker (`13/20`, `10/20`) than deterministic ODE.

The unresolved control is a full `n=100` zero-latent evaluation of distilled SFT. Until that is
run, `69 -> 90` must not be described purely as an RL gain; it combines a policy change with a
decoder change.
