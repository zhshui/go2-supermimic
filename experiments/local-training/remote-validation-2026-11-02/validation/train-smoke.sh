#!/usr/bin/env bash
set -euo pipefail
source /root/go2-supermimic/activate-gpu.sh
python -u scripts/train.py --task Go2-Chase-Flat --run-name agent_gpu_smoke --iterations 120 --num-envs 32 --collect-steps 4 --buffer-len 128 --wm-window 4 --wm-batch 64 --wm-updates 1 --wm-hidden 64 64 --actor-hidden 64 64 --critic-hidden 64 64 --imag-batch 64 --horizon 8 --archive-size 256 --warmup-iterations 10 --critic-warmup-iterations 0 --actor-ramp-iterations 10 --critic-epochs 1 --critic-minibatches 1 --compile false --log-interval 5 --save-interval 60
