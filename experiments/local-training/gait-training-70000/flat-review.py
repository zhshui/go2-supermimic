"""Train and verify flat gait before stepping onto the staircase curriculum."""
import json
from itertools import count
from pathlib import Path

from review_loop import BASE, ROOT, case, evaluate, launch, record

GENERATOR = 'environment.scene.terrain.terrain_generator.'
VERTICAL = 'environment.rewards.base_vertical_speed_l2.weight'
TILT = 'environment.rewards.tilt_rate_l2.weight'
FLIGHT = 'environment.rewards.excessive_flight.weight'
TERRAINS = ('stairs_up_30cm', 'stairs_up_40cm', 'stairs_up_50cm', 'stairs_down_30cm',
            'stairs_down_40cm', 'flat', 'rough', 'boxes', 'slope', 'slope_inv')


def flat_test(checkpoint, folder, seeds):
    results = [evaluate(checkpoint, n=32, seed=seed, flat_only=True) for seed in seeds]
    (folder/'flat-evaluation.json').write_text(json.dumps(results, indent=2))
    return {'passes':sum(r['flat_gait']['passes'] for r in results),
            'episodes':sum(r['episodes'] for r in results),
            'long_flight_episodes':sum(r['flat_gait']['episodes_with_long_flight'] for r in results),
            'alternating_episodes':sum(r['flat_gait']['alternating_episodes'] for r in results),
            'vertical_speed_rms':sum(r['flat_gait']['vertical_speed_rms'] for r in results)/len(results),
            'tilt_rate_rms':sum(r['flat_gait']['tilt_rate_rms'] for r in results)/len(results),
            'max_airborne_s':max(r['flat_gait']['max_airborne_s'] for r in results),
            'falls':sum(round(r['fall_fraction']*r['episodes']) for r in results)}


def main():
    checkpoint, = Path('/root/go2-supermimic/logs/supermimic').glob('*mixed_gait_70k/model_70000.pt')
    assert checkpoint.is_file()
    folder = ROOT/'stability-70000'
    folder.mkdir(exist_ok=True)
    record('flat_baseline', checkpoint=str(checkpoint), target=72000, review_every=2000, objective='stable natural gait before stair progression', weights={})
    baseline = flat_test(checkpoint, folder, (42, 73, 101))
    weights = {VERTICAL:-1.5, TILT:-0.15, FLIGHT:-1.5,
               'environment.rewards.diagonal_support.weight':0.0,
               'environment.rewards.upward_launch.weight':0.0,
               'environment.rewards.riser_clearance.weight':0.0,
               GENERATOR+'num_rows':1, GENERATOR+'num_cols':1, GENERATOR+'curriculum':False}
    weights.update({GENERATOR+f'sub_terrains.{t}.proportion':float(t=='flat') for t in TERRAINS})
    record('flat_baseline_measured', flat_baseline=baseline)
    for end in count(72000, 2000):
        folder = ROOT/f'stability-{end}'
        folder.mkdir(exist_ok=True)
        checkpoint = launch(checkpoint, end, folder, weights)
        record('flat_evaluating', checkpoint=str(checkpoint), target=end)
        metrics = flat_test(checkpoint, folder, (42, 73, 101))
        record('flat_reviewed', flat_metrics=metrics)
        if metrics['passes'] >= 92 and metrics['falls'] == 0:
            heldout = folder/'flat-heldout'
            heldout.mkdir(exist_ok=True)
            verification = flat_test(checkpoint, heldout, (127, 211))
            if verification['passes'] >= 61 and verification['falls'] == 0:
                record('flat_gait_passed', heldout=verification)
                for height in (0.05, 0.10, 0.15):
                    results = [case(checkpoint,height,folder,n=24,rounds=1,seed=seed,gait=True)
                               for seed in (127, 211)]
                    (folder/f'stairs-{height*100:.0f}cm.json').write_text(json.dumps(results,indent=2))
                    if any(r['clean_stepped_ascent_fraction'] < 1 for r in results):
                        record('next_terrain_stage_needed', height_cm=height*100,
                               clean_rates=[r['clean_stepped_ascent_fraction'] for r in results])
                        return
                record('gait_target_met', recommended=str(checkpoint))
                return
        if metrics['long_flight_episodes']:
            weights[FLIGHT] = max(-4.0, weights[FLIGHT]-0.3)
        if metrics['vertical_speed_rms'] > 0.10:
            weights[VERTICAL] = max(-3.0, weights[VERTICAL]-0.25)
        if metrics['tilt_rate_rms'] > 0.65:
            weights[TILT] = max(-0.35, weights[TILT]-0.05)


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        record('error', error=str(exc))
        raise
