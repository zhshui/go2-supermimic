#!/usr/bin/env bash
set -euo pipefail
source /root/go2-supermimic/activate-gpu.sh
exec 9>/root/rl_training_agent/runs/stairs/train.lock
flock -n 9 || { echo "Go2-Stairs training is already running" >&2; exit 1; }
exec python -u /root/go2-supermimic/scripts/train_stairs_config.py
