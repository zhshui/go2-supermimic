"""Preflight a complete proposed training configuration and reject regressions."""
import gc
import json
import os
import sys
from pathlib import Path

import torch
import yaml

from go2_supermimic.supermimic.train import TrainCfg, Trainer

folder = Path(sys.argv[1])
raw = yaml.safe_load((folder/'train.yaml').read_text())
for key in ['wm_hidden', 'actor_hidden', 'critic_hidden', 'driver_fade']:
    raw[key] = tuple(raw[key])
# Preserve all proposed structural settings, but bound preflight allocation/time.
raw.update(num_envs=8, imag_batch=64, wm_batch=64, archive_size=256,
           buffer_len=max(32, raw['wm_window']+3), wm_updates=1, collect_steps=1,
           critic_epochs=1, critic_minibatches=1, compile=False,
           log_interval=1, save_interval=5, log_root=str(folder/'smoke'),
           run_name='preflight')
loaded = torch.load(raw['resume'], map_location='cpu', weights_only=True)
raw['iterations'] = int(loaded['iteration']) + 5
del loaded
os.environ.pop('RL_AGENT_CONTROL_DIR', None)
os.environ['GO2_STAIRS_OVERRIDES'] = str(folder/'overrides.yaml')
trainer = Trainer(TrainCfg(**raw))
try:
    trainer.run()
    checkpoint = trainer.last_ckpt
    payload = torch.load(checkpoint, map_location='cpu', weights_only=True)
    for part in ['actor', 'critic', 'world_model']:
        assert all(torch.isfinite(v).all() for v in payload[part].values())
finally:
    trainer.writer.close()
    trainer.env.close()
del trainer, payload
gc.collect()
torch.cuda.empty_cache()
sys.path.insert(0, '/root/go2-supermimic/scripts')
from evaluate_stairs import case
metrics = [case(checkpoint, height, folder, n=8, rounds=1) for height in (0.05, 0.15)]
reference = json.loads(Path('/root/rl_training_agent/runs/stairs/evaluation_model_14000/results.json').read_text())
for row in metrics:
    baseline = next(r for r in reference if r['iteration']==14000 and r['height_cm']==row['height_cm'])
    assert row['fall_fraction'] <= baseline['fall_fraction'] + 0.125, 'fall regression'
    assert row['three_step_success_fraction'] >= baseline['three_step_success_fraction'] - 0.125, 'ascent regression'
    assert row['mean_net_forward_progress_m'] >= baseline['mean_net_forward_progress_m'] * 0.8, 'progress regression'
(folder/'validation.json').write_text(json.dumps({'status':'passed', 'metrics':metrics}, indent=2))
print('CANDIDATE_PREFLIGHT_AND_REAL_EVALUATION_PASSED', flush=True)
