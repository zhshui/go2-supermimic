# CPU environment on this server

Activate with:

```bash
source /root/go2-supermimic/activate-cpu.sh
python -m pytest tests -q
```

The isolated environment is `/root/.venvs/go2-cpu` (Python 3.13).
It uses CPU-only PyTorch, one compute thread, and headless plotting.
The project is installed editable with `--no-deps`: mjlab/CUDA dependencies
are intentionally excluded. `pip check` will therefore report missing mjlab.

Supported here: target generation, collisions, obstacle and terrain geometry,
world-model/policy tensor operations, checkpoint inspection, and unit tests.
Native MuJoCo XML loading and CPU stepping can also be checked independently.

The original training and evaluation entry points depend on mjlab / MuJoCo Warp
and the README requires a CUDA GPU. They are not configured as CPU simulation
entry points. Installing CPU PyTorch alone does not enable full RL training.
Use a separate GPU environment when adding a GPU; do not replace this CPU setup
with CUDA wheels on this 2 GB memory instance.

`requirements-cpu.lock.txt` records the installed package versions.
`cpu-test-results.txt` records the test run.
