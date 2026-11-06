"""Real locomotion checks grouped by terrain column."""
import gc
from pathlib import Path

import torch
import go2_supermimic.env
from go2_supermimic.supermimic.train import Actor
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg
from go2_supermimic.target.interface import yaw_rotate_inverse
from go2_supermimic.env.locomotion_dynamics import random_push_before_state


def evaluate(checkpoint, n=40, difficulty=0.0):
    torch.manual_seed(42)
    cfg = load_env_cfg('Go2-Mixed', play=True)
    cfg.seed = 42
    cfg.scene.num_envs = n
    cfg.curriculum['terrain_levels'].params.update(steps=12000, start_step=0)
    env = ManagerBasedRlEnv(cfg=cfg, device='cuda:0')
    env.common_step_counter = int(difficulty * 12000)
    obs, _ = env.reset()
    ckpt = torch.load(checkpoint, map_location='cuda:0', weights_only=True)
    actor = Actor(obs['actor'].shape[-1], tuple(ckpt['cfg']['actor_hidden']), 12).cuda().eval()
    actor.load_state_dict(ckpt['actor'])
    robot = env.scene['robot']
    initial = robot.data.root_link_pos_w.clone()
    heading = robot.data.heading_w.clone()
    direction = torch.stack([heading.cos(), heading.sin()], -1)
    columns = env.scene.terrain.terrain_types.clone()
    levels = env.scene.terrain.terrain_levels.clone()
    names = list(cfg.scene.terrain.terrain_generator.sub_terrains)
    alive = torch.ones(n, device='cuda', dtype=torch.bool)
    fell = torch.zeros_like(alive)
    progress = torch.zeros(n, device='cuda')
    velocity_error = torch.zeros_like(progress)
    yaw_error = torch.zeros_like(progress)
    valid_count = torch.zeros_like(progress)
    command = env.command_manager.get_term('target')
    with torch.inference_mode():
        for _ in range(round(8/env.step_dt)):
            desired = command.gen.velocity_command.clone()
            random_push_before_state(env)
            obs, _, terminated, truncated, _ = env.step(actor(obs['actor']))
            assert torch.isfinite(robot.data.root_link_pos_w).all()
            fell |= alive & terminated
            valid = alive & ~terminated & ~truncated
            p = ((robot.data.root_link_pos_w[:,:2]-initial[:,:2])*direction).sum(-1)
            progress = torch.where(valid, p, progress)
            yaw_v = yaw_rotate_inverse(robot.data.root_link_lin_vel_w,robot.data.heading_w)
            velocity_error += valid * (yaw_v[:,:2]-desired[:,:2]).norm(dim=-1)
            yaw_error += valid * (robot.data.root_link_ang_vel_b[:,2]-desired[:,2]).abs()
            valid_count += valid
            alive &= ~(terminated | truncated)
    groups = []
    for col, name in enumerate(names):
        select = columns == col
        if select.any():
            groups.append({'terrain':name,'episodes':int(select.sum()),
                           'fall_fraction':float(fell[select].float().mean()),
                           'mean_forward_progress_m':float(progress[select].mean()),
                           'mean_velocity_error_mps':float((velocity_error[select]/valid_count[select].clamp_min(1)).mean()),
                           'mean_yaw_rate_error_rps':float((yaw_error[select]/valid_count[select].clamp_min(1)).mean()),
                           'sampled_levels':levels[select].cpu().tolist()})
    result = {'checkpoint':str(checkpoint),'iteration':ckpt.get('iteration'),
              'seed':42,'episode_seconds':8,'episodes':n,'difficulty':difficulty,
              'fall_fraction':float(fell.float().mean()),
              'mean_forward_progress_m':float(progress.mean()),
              'mean_velocity_error_mps':float((velocity_error/valid_count.clamp_min(1)).mean()),
              'mean_yaw_rate_error_rps':float((yaw_error/valid_count.clamp_min(1)).mean()),
              'push_events':getattr(env,'_training_push_count',0),'by_terrain':groups}
    env.close()
    del env, actor, robot, ckpt
    gc.collect()
    torch.cuda.empty_cache()
    return result
