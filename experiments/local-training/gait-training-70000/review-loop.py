"""Train in 2k blocks and accept checkpoints against paired locomotion benchmarks."""
import gc
from itertools import count
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

import torch
import yaml

PROJECT = Path('/root/go2-supermimic')
BASE = Path('/root/rl_training_agent/runs/stairs')
ROOT = BASE / 'gait-review'
sys.path.insert(0, str(PROJECT / 'scripts'))
from evaluate_mixed import evaluate
from evaluate_stairs import case

STATUS = ROOT / 'status.json'
SEEDS = (42, 73, 101)
FLIGHT = 'environment.rewards.excessive_flight.weight'
LAUNCH = 'environment.rewards.upward_launch.weight'
RISER = 'environment.rewards.riser_clearance.weight'
WEIGHTS = {FLIGHT: -1.35, LAUNCH: -2.0, RISER: -1.25}


def record(status, **fields):
    current = json.loads(STATUS.read_text()) if STATUS.exists() else {}
    current.update(status=status, updated_at=time.time(), **fields)
    temp = STATUS.with_suffix('.tmp')
    temp.write_text(json.dumps(current, indent=2))
    temp.replace(STATUS)
    print(status, fields, flush=True)


def check(checkpoint, folder):
    folder.mkdir(exist_ok=True)
    result_path = folder / 'evaluation.json'
    if result_path.exists():
        return json.loads(result_path.read_text())
    mixed = [evaluate(checkpoint, n=80, difficulty=1.0, seed=seed) for seed in SEEDS]
    stairs = [case(checkpoint, 0.05, folder, n=24, rounds=1)]
    stairs += [case(checkpoint, 0.15, folder, n=24, rounds=1, seed=seed, gait=True) for seed in SEEDS]
    flat = [next(t for t in r['by_terrain'] if t['terrain'] == 'flat') for r in mixed]
    summary = {
        'flat_episodes': sum(t['episodes'] for t in flat),
        'flat_airborne': statistics.mean(t['airborne_fraction'] for t in flat),
        'flat_long_flight': statistics.mean(t['long_flight_fraction'] for t in flat),
        'flat_slip': statistics.mean(t['stance_slip_mps'] for t in flat),
        'mixed_falls': statistics.mean(r['fall_fraction'] for r in mixed),
        'velocity_error': statistics.mean(r['mean_velocity_error_mps'] for r in mixed),
        'yaw_error': statistics.mean(r['mean_yaw_rate_error_rps'] for r in mixed),
        'clean_15cm': statistics.mean(r['clean_stepped_ascent_fraction'] for r in stairs[1:]),
        'body_hit_15cm': statistics.mean(r['body_hit_fraction'] for r in stairs[1:]),
        'riser_hit_15cm': statistics.mean(r['foot_riser_hit_fraction'] for r in stairs[1:]),
        'max_airborne_15cm_s': max(r['max_airborne_s'] for r in stairs[1:]),
        'stairs': [{k: r[k] for k in ('height_cm', 'n', 'fall_fraction', 'three_step_success_fraction',
                                     'mean_net_ground_rise_m')} for r in stairs],
    }
    result_path.write_text(json.dumps({'checkpoint': str(checkpoint), 'summary': summary,
                                       'mixed': mixed, 'stairs': stairs}, indent=2))
    return json.loads(result_path.read_text())


def launch(checkpoint, end, folder, weights):
    raw = torch.load(checkpoint, map_location='cpu', weights_only=True)['cfg']
    overrides = raw.pop('runtime_overrides', {})
    environment = overrides.setdefault('environment', {})
    for name, weight in weights.items():
        environment[name.removeprefix('environment.')] = weight
    start = int(checkpoint.stem.split('_')[-1])
    raw.update(iterations=end, resume=str(checkpoint), continue_run=True, run_name=f'gait_review_{end}',
               lr=0.00003, lr_schedule_start=start, lr_decay_iterations=2000,
               lr_final_frac=0.35, actor_freeze_until=start+100)
    train_file, override_file = folder/'train.yaml', folder/'overrides.yaml'
    train_file.write_text(yaml.safe_dump(raw, sort_keys=False))
    override_file.write_text(yaml.safe_dump(overrides, sort_keys=False))
    (BASE/'training-limits.json').write_text(json.dumps({'max_iterations': end}))
    env = dict(os.environ, GO2_STAIRS_TRAIN_CONFIG=str(train_file), GO2_STAIRS_OVERRIDES=str(override_file))
    record('training', start=start, target=end, weights=weights, training_log=str(folder/'training.log'))
    with (folder/'training.log').open('w') as log:
        proc = subprocess.Popen(['bash', str(PROJECT/'scripts/train_stairs.sh')], cwd=PROJECT,
                                env=env, stdout=log, stderr=subprocess.STDOUT)
        (BASE/'launcher.pid').write_text(str(proc.pid))
        if proc.wait() != 0:
            raise RuntimeError(f'Training exited with status {proc.returncode}. See {folder}/training.log')
    state = json.loads((BASE/'current-run.json').read_text())
    if state.get('last_iteration') != end or state['status'] != 'completed':
        raise RuntimeError(f'Missing completed run at {end}')
    return Path(state['log_dir'])/f'model_{end}.pt'


def regressions(row, base):
    issues = []
    if row['mixed_falls'] > base['mixed_falls'] + 0.04:
        issues.append('mixed falls')
    if row['velocity_error'] > base['velocity_error'] + 0.03:
        issues.append('linear tracking')
    if row['yaw_error'] > base['yaw_error'] + 0.04:
        issues.append('yaw tracking')
    for candidate, reference in zip(row['stairs'], base['stairs'], strict=True):
        if (candidate['three_step_success_fraction'] < reference['three_step_success_fraction'] - 0.125
            or candidate['fall_fraction'] > reference['fall_fraction'] + 0.125
            or candidate['mean_net_ground_rise_m'] < 0.8 * reference['mean_net_ground_rise_m']):
            issues.append(f"{candidate['height_cm']:.0f} cm staircase")
    return issues


def main():
    record('waiting_for_70000', review_every=2000, max_iteration=None, objective='clean 15 cm stepped ascent')
    while True:
        state = json.loads((BASE/'current-run.json').read_text())
        if state.get('status') == 'completed' and state.get('last_iteration') == 70000:
            checkpoint = Path(state['log_dir'])/'model_70000.pt'
            break
        if state.get('status') == 'failed' or state.get('iterations') != 70000:
            raise RuntimeError('Expected the existing run to complete at 70000')
        time.sleep(15)
    record('evaluating_baseline', checkpoint=str(checkpoint))
    baseline = check(checkpoint, ROOT/'70000')['summary']
    best = checkpoint
    best_score = baseline['clean_15cm']
    weights = WEIGHTS.copy()
    for end in count(72000, 2000):
        folder = ROOT/str(end)
        folder.mkdir(exist_ok=True)
        checkpoint = launch(checkpoint, end, folder, weights)
        record('evaluating', checkpoint=str(checkpoint), target=end)
        result = check(checkpoint, folder)['summary']
        issues = regressions(result, baseline)
        if issues:
            record('regression_review_required', rejected=str(checkpoint), recommended=str(best),
                   issues=issues, metrics=result)
            return
        if result['clean_15cm'] > best_score:
            best, best_score = checkpoint, result['clean_15cm']
        record('reviewed', metrics=result, recommended=str(best), baseline=baseline)
        if result['clean_15cm'] == 1.0:
            heldout = [case(checkpoint, 0.15, folder, n=24, rounds=1, seed=seed, gait=True)
                       for seed in (127, 211)]
            (folder/'heldout.json').write_text(json.dumps(heldout, indent=2))
            if all(r['clean_stepped_ascent_fraction'] == 1.0 for r in heldout):
                record('gait_target_met', recommended=str(checkpoint), heldout_episodes=48)
                return
        if result['max_airborne_15cm_s'] > 0.06001:
            weights[FLIGHT] = max(-4.0, weights[FLIGHT] - 0.45)
        if result['riser_hit_15cm'] or result['body_hit_15cm']:
            weights[RISER] = max(-4.0, weights[RISER] - 0.5)
        gc.collect()


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        record('error', error=str(exc))
        raise
