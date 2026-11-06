#!/usr/bin/env bash
set -euo pipefail
source /root/.config/rl-training-agent/credentials.env
export HTTPS_PROXY=http://127.0.0.1:17897
export HTTP_PROXY=http://127.0.0.1:17897
RUN_DIR=$(/root/.venvs/go2-cpu/bin/python -c 'import json; print(json.load(open("/root/rl_training_agent/runs/stairs/current-run.json"))["log_dir"])')
exec /root/.venvs/go2-cpu/bin/python -m rl_training_agent.cli analyze --llm --config /root/rl_training_agent/configs/stairs/project.yaml --metrics "$RUN_DIR" --data-dir /root/rl_training_agent/data "$@"
