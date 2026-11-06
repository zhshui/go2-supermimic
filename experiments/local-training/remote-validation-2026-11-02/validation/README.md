# GPU validation completed

Environment: `/root/.venvs/go2-gpu`, PyTorch 2.10.0+cu128, RTX 4090.
Activate: `source /root/go2-supermimic/activate-gpu.sh`.

`train-smoke.sh` records the exact 120-iteration / 32-environment command.
`validation-report.json` records the result, log directory and experiment ID.
`training.log` is the successful training output. `analysis-real-training.json`
contains the agent's analysis of real TensorBoard logs (304 scalar points).
The actor, critic and world model all changed between iterations 60 and 120.
All scalar values and checkpoint parameters are finite. The checkpoint backup
was verified against the original with SHA-256. CLI experiments listing passed.
A separate controlled spike test on a copy of the world-model-loss series
triggered an alert; original training logs were not modified.

This validates the local monitoring, analysis, experiment database and backup
workflow. It does not establish policy convergence. No LLM API was invoked.

`mujoco-warp-compat.patch` fixes startup on MuJoCo Warp 3.11 by skipping obsolete
solver warning flag configuration when that optional API is unavailable.
The GPU dependency versions are in `/root/go2-supermimic/requirements-gpu.lock.txt`.
The two earlier training-attempt logs record the compatibility failure before
this patch; `training.log` is the completed run.
