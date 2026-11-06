"""Add low stairs while retaining the flat gait benchmark."""
import json
from pathlib import Path

from review_loop import BASE, ROOT, case, launch, record
import importlib
flat_test = importlib.import_module('flat-review').flat_test

state = json.loads((BASE/'current-run.json').read_text())
assert state['status'] == 'completed' and state['last_iteration'] == 76000
checkpoint = Path(state['log_dir'])/'model_76000.pt'
folder = ROOT/'stability-78000'
folder.mkdir(exist_ok=True)
prefix = 'environment.scene.terrain.terrain_generator.'
weights = {
    prefix+'curriculum': True,
    prefix+'num_cols': 2,
    prefix+'sub_terrains.flat.proportion': 0.7,
    prefix+'sub_terrains.stairs_up_40cm.proportion': 0.3,
    prefix+'sub_terrains.stairs_up_40cm.step_height_range': [0.05, 0.05],
    'environment.rewards.base_vertical_speed_l2.weight': -1.25,
    'environment.rewards.riser_clearance.weight': -1.5,
}
try:
    checkpoint = launch(checkpoint, 78000, folder, weights)
    record('terrain_evaluating', checkpoint=str(checkpoint), height_cm=5.0)
    flat = flat_test(checkpoint, folder, (42, 73, 101))
    stairs = [case(checkpoint, .05, folder, n=24, rounds=1, seed=seed, gait=True) for seed in (127, 211)]
    (folder/'stairs-5cm.json').write_text(json.dumps(stairs, indent=2))
    record('terrain_reviewed', flat_metrics=flat, height_cm=5.0,
           clean_rates=[r['clean_stepped_ascent_fraction'] for r in stairs])
except Exception as exc:
    record('error', error=str(exc))
    raise
