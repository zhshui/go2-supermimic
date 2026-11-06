# Online LLM optimization enabled

Training resumes model_4000.pt, with 4096 environments, total 40000 iterations.
The independent CPU-side LLM process analyzes compact metric windows and submits
JSON optimizer proposals. The GPU training loop checks proposals at log boundaries
and actually updates the shared TrainCfg used by the next optimizer step.

Supported automatic changes:
- lr: [0.00002, 0.0002], reductions only, at most 20% per update.
- actor_lr_scale: [0.05, 0.3], reductions only, at most 20% per update.
- Maximum 8 applied changes per run; each requires matching experiment and current
  parameter value, a proposal age <=2000 iterations, and finite training health.
- Every application saves and copies a full pre-change training checkpoint.

The first cycle requests one small (<=10%) reversible optimizer trial unless
numerical evidence contraindicates it. Later cycles may hold parameters unchanged.
LLM optimization is not a promise of better stair climbing: use actual evaluation.
No reward, curriculum, architecture, observation/action or robot changes are automatic.

Model: gpt-6-sol; Responses API; medium reasoning; store=false.
First analysis starts at the first resumed log boundary; thereafter every 2000
iterations. At most 18 calls / 100000 accounted tokens in this resumed run.
Each attempt reserves 27000 tokens conservatively, reconciled to actual usage
on success. Failed attempts retain that reservation. Input is limited to 24 KB;
max output is 2500 tokens. The API request has a 180-second timeout, no retries.
On timeout, invalid JSON, rejected changes or exhausted budget, training continues.
The workstation proxy and SSH reverse tunnel must remain online for API calls.

Inspect:
```bash
cat /root/rl_training_agent/runs/stairs/current-run.json
tail -f /root/rl_training_agent/runs/stairs/training-llm.log
tail -f /root/rl_training_agent/runs/stairs/online-llm.log
```

The current run's online-control directory contains snapshot.json, llm-ledger.json,
latest-llm-diagnosis.json, control-events.jsonl and backups/. Those files record
actual decisions, applied/rejected proposals, parameter values and checkpoint paths.
The original run and model_4000.pt remain unchanged. New run config is
/root/go2-supermimic/configs/stairs/train-llm.yaml.
