# Experimental locomotion checkpoint (188500)

Source run: `2026-11-06_21-23-55_gait_review_calf_acc1p5_lrhalf_188500`.

- `model_188500.pt`: full training checkpoint (actor, critic, world model, optimizers, configuration).
- `actor_188500.pt`: evaluation policy with the original `cfg`, `supermimic`, `scan_grid`, `velocity_commands`, `iteration`, and `obs_dim` metadata.
- `config_188500.json`: checkpoint training configuration.
- `manifest.json`: byte sizes and SHA-256 checksums.

This actor uses the **17x11** scan grid, **1285 observation inputs**, and velocity commands. Export was performed using `torch.load(..., weights_only=True)` without executing arbitrary checkpoint objects. The full checkpoint SHA-256 was verified against the training-machine original.

This checkpoint is experimental and is not certified as accepted. See [experiment status](../../experiments/local-training/GO2-LATEST-STATUS.md) and [evaluation videos and metrics](../../experiments/local-training/README.md).
