#!/usr/bin/env bash
set -euo pipefail
export GO2_STAIRS_TRAIN_CONFIG=/root/rl_training_agent/runs/stairs/mixed-migration/train-mixed.yaml
export GO2_STAIRS_OVERRIDES=/root/rl_training_agent/runs/stairs/mixed-migration/overrides.yaml
exec bash /root/go2-supermimic/scripts/train_stairs.sh
