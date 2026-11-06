# Verified LLM integration

Real Responses API diagnosis succeeded using gpt-6-sol, medium reasoning,
store=false. The provider authenticated the key and listed gpt-6-sol.
The request analyzed 2043 scalar points. Returned usage: 43701 input tokens,
3037 output tokens. Training remains stopped; resume checkpoint is model_4000.pt.

## Human review of model output

The raw diagnosis is not an established finding. It correctly requests real
stair-completion, net-ascent and video evaluation before claiming success.
These particular statements require correction:

- Go2-Stairs explicitly disables chase/dodge targets. Zero Metrics/target/* and
  Metrics/hit_rate/* values are expected inactive slots, not failed stair metrics.
- mjlab TerminationManager.reset returns raw counts over reset env_ids. The
  trainer averages those counts. Values above 1 do not prove a logging bug and
  must not be called fall probabilities without a denominator.
- The terrain-row schedule does not establish a row increase at iteration 1250.
  Policy imagination updates begin at 1001, which is another possible source of
  changing data distribution. Causality needs further measurement.

Those semantics have now been added to the LLM YAML for future diagnoses.
No LLM suggestion was applied and training was not restarted.

## Connectivity

The remote GPU host's direct request timed out. Successful invocation uses
remote 127.0.0.1:17897, forwarded over SSH to workstation 127.0.0.1:7897.
The workstation and its proxy must stay online. Credentials remain only in
/root/.config/rl-training-agent/credentials.env and are not in this bundle.
Run /root/rl_training_agent/configs/stairs/analyze-llm.sh for another diagnosis.
The 1M context/900k compaction settings are metadata; no automatic compaction
or recurring model calls are implemented by the current stateless CLI.
