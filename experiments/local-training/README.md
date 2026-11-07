# Go2 locomotion training artifacts

Snapshot collected on 2026-11-06 (Asia/Shanghai). The repository source was copied from the training machine working tree based on upstream commit `d9eee9b3f248e5300e636c433255db2c1c3e086a`.

- [All 170 original policy/checkpoint files](../../weights/complete/README.md), including historical checkpoints and backups.
- [Latest full checkpoint](../../weights/locomotion/model_188500.pt): iteration 188500, actor, critic, world model, optimizer state and training configuration.
- [Actor policy](../../weights/locomotion/actor_188500.pt): actor plus configuration, scan grid and velocity-command metadata; suitable for evaluation, not training resumption.
- [Checkpoint configuration](../../weights/locomotion/config_188500.json).
- [All 210 evaluation videos](VIDEO-INDEX.md), with original result JSON files beside the videos.
- [Latest recorded evaluation](gait-training-70000/videos/188500-20261106-2138/results.json).
- [Verified status and rejected trial note](GO2-LATEST-STATUS.md).
- [Local artifact checksums](manifest.json).

## Interpretation

This is an archive of experimental work, not a claim that the latest checkpoint passed acceptance. The `calf_acc1p5_lrhalf_185500` trial was rejected for reporting because the actual training reward coefficient differed from its prelaunch verification. The later 188500 checkpoint belongs to this experiment family. See the original records for the full context. Videos and metrics span multiple checkpoints and evaluation settings; a filename alone is not evidence of a validated success.

The original local helper scripts, configuration snapshots, validation records and historical source backups are included. Machine-specific activation scripts and absolute paths may require adjustment on another machine. `go2-tools` application cache and credentials are excluded.

## Validation at upload

The training-machine command `python -m pytest tests -q` returned **31 passed, 1 failed**. The failing test was `tests/test_obstacles.py::test_obstacle_rises_in_the_scan_ahead_of_the_robot`: it assumes a height-scan point near x=0.75 m, but the current scan grid has no matching point (`in_wall.any()` is false). The source and historical experimental results are preserved as found; this upload does not claim a fully passing test suite.

This upload did not run new training or new simulator acceptance evaluations.
