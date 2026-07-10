# Experiment Log

## Data and SFT

- Recreated the beige/white SO100, black table, blue cube, and pink target pad in MuJoCo.
- Added exterior and wrist RGB observations plus five joints and one gripper scalar.
- Wrote a damped-least-squares pinch-point IK oracle and generated contact-physics demonstrations.
- Fine-tuned OpenPI pi0.5 with a 16-step action horizon and six executed action dimensions.

## Framework parity

- Added SO100 data/policy adapters to OpenPI and RLinf.
- Converted JAX/Orbax weights into RLinf's PyTorch OpenPI implementation.
- Fixed a RoPE precision mismatch and copied normalization assets into RLinf checkpoint layout.
- Corrected evaluation control cadence, initial pose, action dtype, and cube-position variation.
- Collected JAX teacher chunks and calibrated the last two expert layers plus action/time heads.
- Distilled Torch SFT reached `69/100` broad and `53/100` strict on the canonical 100 rollouts.

## PPO attempts

- Early longer PPO runs degraded despite low SFT loss and apparently improving rollout reward.
- Tested lower actor learning rates, KL anchoring, detached critic inputs, hard-start curricula,
  reward reweighting, co-training/replay, and checkpoint-by-checkpoint evaluation.
- The retained policy is a two-stage continuation: five lift-robustness iterations from SFT,
  then five pick/lift-repair iterations from stage 1.
- With standard Gaussian ODE decoding, stage 2 is `70/100` broad: effectively tied with SFT.

## Decoder controls

- The same stage-2 checkpoint reached `90/100` broad and `82/100` strict with zero initial latent.
- A fixed unit-Gaussian latent was poor; low-magnitude fixed latents worked substantially better.
- Fresh half-scale Gaussian improved over unit Gaussian but did not match the zero/fixed-half result.
- Zeroing padded latent dimensions while retaining unit Gaussian in the first six did not fix the gap.
- Stochastic flow-noise sampling was weaker than deterministic ODE in the 20-episode control.

The current scientific conclusion is narrower than the headline score: the checkpoint is highly
sensitive to initial-latent scale and decode configuration. The repository contains the controls
needed to reproduce and continue that investigation.
