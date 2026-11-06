"""Task-only staged experiment using the original SuperMimic trainer."""
import argparse
import json
import subprocess
import sys
from pathlib import Path
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlOnPolicyRunnerCfg
from mjlab.tasks.registry import register_mjlab_task
from go2_supermimic.env.env_cfg import go2_stairs_env_cfg
from go2_supermimic.supermimic.train import Actor, Trainer, TrainCfg


def config(height, play=False, num_envs=32):
    cfg = go2_stairs_env_cfg(play=play)
    gen = cfg.scene.terrain.terrain_generator
    gen.num_rows = 1
    gen.sub_terrains['stairs'].step_height_range = (height, height)
    cfg.curriculum = {}
    cfg.scene.num_envs = num_envs
    cfg.episode_length_s = 12.0
    cfg.seed = 42
    return cfg


def evaluate(path, height, out, num_envs=32):
    torch.manual_seed(42)
    cfg = config(height, True, num_envs)
    env = ManagerBasedRlEnv(cfg=cfg, device='cuda:0')
    saved = torch.load(path, map_location='cuda:0', weights_only=False)
    obs, _ = env.reset()
    actor = Actor(735, tuple(saved['cfg']['actor_hidden']), 12).cuda().eval()
    actor.load_state_dict(saved['actor'])
    origin = env.scene.terrain.env_origins.clone()
    alive = torch.ones(num_envs, dtype=torch.bool, device='cuda')
    success = torch.zeros_like(alive)
    fell = torch.zeros_like(alive)
    distance = torch.zeros(num_envs, device='cuda')
    times = torch.full((num_envs,), float('nan'), device='cuda')
    # The 12m tile has a 1m border. All feet must reach the top at x=5m.
    foot_ids, _ = env.scene['robot'].find_sites(('FR', 'FL', 'RR', 'RL'), preserve_order=True)
    with torch.inference_mode():
        for step in range(int(12 / env.step_dt)):
            obs, reward, terminated, truncated, _ = env.step(actor(obs['actor']))
            data = env.scene['robot'].data
            assert torch.isfinite(obs['actor']).all() and torch.isfinite(reward).all()
            valid = alive & ~terminated & ~truncated
            local = data.root_link_pos_w - origin
            distance = torch.where(valid, torch.maximum(distance, local[:, 0]), distance)
            feet_x = data.site_pos_w[:, foot_ids, 0] - origin[:, None, 0]
            upright = data.projected_gravity_b[:, 2] < -0.85
            reached = valid & (feet_x.min(-1).values > 5.05) & (local[:, 1].abs() < 1.0) & upright
            times[reached] = (step + 1) * env.step_dt
            success |= reached
            fell |= alive & terminated
            alive &= ~(reached | terminated | truncated)
            if not alive.any():
                break
    result = dict(checkpoint=str(path), height=height, tread=0.4, episodes=num_envs,
                  successes=int(success.sum()), success_rate=float(success.float().mean()),
                  fall_rate=float(fell.float().mean()), mean_distance=float(distance.mean()),
                  mean_completion_seconds=float(times[success].mean()) if success.any() else None,
                  timeout_rate=float((~success & ~fell).float().mean()))
    Path(out).write_text(json.dumps(result, indent=2))
    print('EVALUATION', json.dumps(result), flush=True)
    env.close()
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument('mode', choices=['eval', 'train'])
    p.add_argument('--checkpoint', default='weights/terrain.pt')
    p.add_argument('--height', type=float, default=0.02)
    p.add_argument('--out', required=True)
    p.add_argument('--iterations', type=int, default=100)
    p.add_argument('--resume', default='')
    p.add_argument('--evaluate-every', type=int, default=100)
    args = p.parse_args()
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision('high')
    if args.mode == 'eval':
        evaluate(args.checkpoint, args.height, args.out)
        return
    register_mjlab_task('Go2-Stairs-Stage', config(args.height, num_envs=1024),
                       config(args.height, True), RslRlOnPolicyRunnerCfg())
    cfg = TrainCfg(task='Go2-Stairs-Stage', num_envs=1024, buffer_len=400,
                   imag_batch=1024, wm_batch=1024, archive_size=20000,
                   iterations=args.iterations, warmup_iterations=300,
                   actor_lr_scale=0.3, save_interval=100, log_interval=25,
                   log_root=args.out, run_name='stairs', resume=args.resume,
                   continue_run=bool(args.resume))
    trainer = Trainer(cfg)
    if not args.resume:
        pretrained = torch.load(args.checkpoint, map_location='cuda:0', weights_only=False)
        trainer.actor.load_state_dict(pretrained['actor'])
        print('Actor-only warm start; world model and critic are fresh.', flush=True)
    Path(args.out).mkdir(parents=True, exist_ok=True)
    Path(args.out, 'experiment.json').write_text(json.dumps(vars(args), indent=2))
    end = args.iterations
    streak = 0
    while trainer.start_it < end:
        cfg.iterations = min(end, trainer.start_it + args.evaluate_every)
        trainer.run()
        checkpoint = trainer.log_dir / f'model_{cfg.iterations}.pt'
        if not checkpoint.exists():
            raise RuntimeError('No checkpoint saved; inspect collapse guard before continuing')
        output = trainer.log_dir / f'eval_{cfg.iterations}.json'
        with (trainer.log_dir / f'eval_{cfg.iterations}.log').open('w') as log:
            subprocess.run([sys.executable, __file__, 'eval', '--checkpoint', str(checkpoint),
                            '--height', str(args.height), '--out', str(output)],
                           stdout=log, stderr=subprocess.STDOUT, check=True)
        result = json.loads(output.read_text())
        print('CHECKPOINT_RESULT', json.dumps(result), flush=True)
        streak = streak + 1 if result['success_rate'] >= 0.8 and result['fall_rate'] <= 0.1 else 0
        if streak >= 3:
            print('CURRICULUM_READY: three passing batches; next stage requires review.', flush=True)
            break
        trainer.start_it = cfg.iterations
    trainer.writer.close()
    trainer.env.close()


if __name__ == '__main__':
    main()
