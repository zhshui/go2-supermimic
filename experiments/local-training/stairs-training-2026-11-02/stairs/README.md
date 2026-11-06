# Go2-Stairs formal world-model training

4096 environments, 40000 iterations, 1000 iterations of world-model warmup.
SuperMimic pipeline: simulation collection -> world-model fitting -> imagined
rollout gradients for Actor and Critic. Upward stairs: 2-15 cm, 40 cm treads,
0.6 m/s forward command. Real stair-climbing success requires separate evaluation.

Actual parameters: /root/go2-supermimic/configs/stairs/train.yaml
LLM configuration: llm.yaml (agent.yaml is a symlink consumed by the CLI).
Project configuration: project.yaml; goals: /root/go2-supermimic/PROJECT.md.
The YAML task_context is forwarded into the LLM diagnosis request.

Inspect training:
```bash
tail -f /root/rl_training_agent/runs/stairs/training.log
cat /root/rl_training_agent/runs/stairs/current-run.json
cat /root/rl_training_agent/runs/stairs/latest-monitor.json
```

Local analysis, without API calls:
```bash
source /root/.venvs/go2-cpu/bin/activate
python -m rl_training_agent.cli analyze \
  --config /root/rl_training_agent/configs/stairs/project.yaml \
  --metrics /root/go2-supermimic/logs/supermimic/2026-11-02_18-42-39_stairs_formal \
  --data-dir /root/rl_training_agent/data
```

For LLM diagnosis, set OPENAI_API_KEY and RL_AGENT_MODEL. Set OPENAI_BASE_URL
and RL_AGENT_WIRE_API (chat_completions or responses) for your provider, then
add --llm to that command. Keep keys out of YAML. The optional SDK is installed
in the CPU environment. Context delivery was validated using a mocked client;
no real API call has been made.

The background observer analyzes local logs every 60 seconds. It does not call
the LLM, modify parameters or stop training. Diagnostic intervals and budgets
in YAML are metadata; the existing CLI does not schedule API calls or enforce
these token limits. Suggestions are not automatically applied.

A complete training checkpoint is saved every 500 iterations. The final one
is backed up and recorded in the experiment database at completion. The run
directory contains train.yaml, llm.yaml, PROJECT.md and model_0.pt.
