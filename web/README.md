# Live Policy Viewer

The default page cycles through four live MuJoCo rollouts:

1. distilled SFT with the RLinf/OpenPI reference path;
2. the same SFT weights with FlashRT FP8 + FlashAct `mk_v6` FP8;
3. the PPO checkpoint with the reference path;
4. the same PPO weights with the optimized hybrid path.

`/history` shows the saved 100-episode SFT and RL evaluations. The checked-in summary
symlinks point at the raw result files used by the public demo.

The standard and hybrid paths intentionally support separate Python dependency layers:

- `POSTVLA_STANDARD_SITE_PACKAGES`: optional site-packages directory for RLinf/Torch 2.8.
- `POSTVLA_HYBRID_SITE_PACKAGES`: optional site-packages directory containing non-Torch
  dependencies for the system Torch 2.10 + CUDA 13 runtime.
- `POSTVLA_CUDA_HOME`: CUDA 13 toolkit root, default `/usr/local/cuda-13.0`.

See `docs/REPRODUCTION.md` for checkpoint setup and the launch command.
