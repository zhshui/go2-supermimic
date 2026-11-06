# Expanded LLM configuration permissions

The two-parameter online process has been replaced by full_optimizer.py.
All existing TrainCfg, environment and imagination configuration paths are
eligible: optimizer settings, sampling, world-model fitting, network hidden
sizes, rewards, curricula, terrain, sensors, randomization and imagination.
Task/framework identity, checkpoint lifecycle and output paths are orchestrated.

LLM decisions use typed JSON paths, not arbitrary shell commands or file edits.
Examples: train.wm_updates, train.actor_hidden, train.horizon,
environment.rewards.track_lin_vel.weight,
environment.curriculum.terrain_levels.params.steps, imagination.gamma.

Every change becomes a candidate configuration. It must load the checkpoint,
run five actual optimization iterations, keep finite parameters and pass real
5 cm and 15 cm ascent regression tests. Incompatible topology changes are
rejected by checkpoint loading; they are not silently reinitialized. Changed
world models and critics get a fresh warmup relative to the resumed iteration.

After passing preflight, the optimizer stops the old process, resumes the most
recent stable checkpoint using the candidate configuration, and records the
switch. If startup fails it restores the previous configuration/checkpoint.
The old model files and prior experiments remain intact. The LLM may hold if
actual performance does not justify changes; broad permission is not a claim
that every proposed configuration will pass validation.

Cadence: approximately every 2000 iterations. Budget: at most 10 calls / 200000
accounted tokens for this expanded-scope session; failures reserve 40000 tokens.
No API retries. The workstation SSH proxy must remain online.

Logs:
/root/rl_training_agent/runs/stairs/full-optimizer.log
/root/rl_training_agent/runs/stairs/full-optimization/ledger.json
Each candidate contains request-summary.json, diagnosis.json, train.yaml,
overrides.yaml, validation.log, validation.json and switch.json when applicable.

Real evaluation completed for checkpoint 14000: at each of 2/5/10/15 cm, 96
8-second episodes (seed 42), 100% passed the explicitly defined three-step
ascent plus forward-progress criterion with no fall. This is a fixed simulator
benchmark, not a claim about arbitrary stairs or physical robot deployment.
At 15 cm the 4000 checkpoint passed 0/96; 14000 passed 96/96. These differences
include ordinary training progress and do not isolate the causal effect of LLM.

Each upcoming cycle first evaluates the recent stable checkpoint on 5/15 cm
stairs, 16 episodes per height. Those quick checks are labeled separately from
the historical 96-episode benchmark and sent with the training metrics.
Promotion must reach the first healthy post-resume log boundary, not merely
initialize the environment. Startup/first-update failure triggers rollback.
