#!/usr/bin/env bash
set -euo pipefail
cd /root/go2-supermimic
/tmp/go2-uv/bin/uv pip install --python /root/.venvs/go2-gpu/bin/python -e . pytest pyyaml --constraint /tmp/go2-gpu-constraints.txt --find-links /tmp/go2-gpu-wheels --index-url https://pypi.tuna.tsinghua.edu.cn/simple
printf '/root\n' > /root/.venvs/go2-gpu/lib/python3.13/site-packages/rl_training_agent.pth
source activate-gpu.sh
python -c 'import torch,mjlab,warp; assert torch.cuda.is_available(); print(torch.__version__,torch.version.cuda,torch.cuda.get_device_name()); print(torch.ones(3,device="cuda").sum())'
python -m pip check
python -m pytest tests -q
bash /root/rl_training_agent/validation/train-smoke.sh > /root/rl_training_agent/validation/training.log 2>&1
python /root/rl_training_agent/validation/check-training.py > /root/rl_training_agent/validation/check-training.log 2>&1
python -m pip freeze > /root/go2-supermimic/requirements-gpu.lock.txt
printf 'GPU_VALIDATION_COMPLETE\n'
