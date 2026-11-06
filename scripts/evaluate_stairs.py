"""Fixed-height, fixed-seed real-simulator stair evaluation."""
import argparse
import gc
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

import go2_supermimic.env
from go2_supermimic.supermimic.train import Actor
from go2_supermimic.env.mdp import tread_edge_contact_risk
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from go2_supermimic.robot.go2 import FOOT_GEOM_NAMES


def case(checkpoint, height, out, video=False, n=32, rounds=3, seed=42, gait=False):
    torch.manual_seed(seed)
    cfg = load_env_cfg('Go2-Stairs', play=True)
    cfg.seed = seed
    cfg.scene.num_envs = n
    cfg.curriculum = {}
    generator = cfg.scene.terrain.terrain_generator
    generator.num_rows = generator.num_cols = 1
    generator.curriculum = False
    generator.sub_terrains['stairs'].step_height_range = (height, height)
    cfg.scene.terrain.max_init_terrain_level = 0
    cfg.viewer.width, cfg.viewer.height = 960, 540
    cfg.viewer.distance, cfg.viewer.elevation = 3.2, -22.0
    if gait:
        terrain = ContactMatch(mode="body", pattern="terrain")
        cfg.scene.sensors += (
            ContactSensorCfg(name="stair_body_contact",
                primary=ContactMatch(mode="geom", pattern=".*collision.*", entity="robot", exclude=FOOT_GEOM_NAMES),
                secondary=terrain, fields=("found", "force"), reduce="netforce",
                history_length=cfg.decimation),
            ContactSensorCfg(name="stair_foot_impact",
                primary=ContactMatch(mode="geom", pattern=FOOT_GEOM_NAMES, entity="robot"),
                secondary=terrain, fields=("found", "force", "normal"), reduce="maxforce", num_slots=4),
        )
    env = ManagerBasedRlEnv(cfg=cfg, device='cuda:0',
                            render_mode='rgb_array' if video else None)
    ckpt = torch.load(checkpoint, map_location='cuda:0', weights_only=True)
    if ckpt.get('velocity_commands',False):
        command = env.command_manager.get_term('target')
        command.cfg.velocity_commands = True
        command.cfg.velocity_x_range=(0.6,0.6)
        command.cfg.velocity_y_range=(0.,0.)
        command.cfg.yaw_rate_range=(0.,0.)
    obs, _ = env.reset()
    actor = Actor(obs['actor'].shape[-1], tuple(ckpt['cfg']['actor_hidden']), 12).cuda().eval()
    actor.load_state_dict(ckpt['actor'])
    term = env.command_manager.get_term('target')
    dev = env.device
    episodes, frames = [], []
    video_error = None
    steps = int(round(8.0 / env.step_dt))
    with torch.inference_mode():
        for trial in range(rounds):
            obs, _ = env.reset()
            data = term.robot.data
            initial_xy = data.root_link_pos_w[:, :2].clone()
            heading = data.heading_w.clone()
            direction = torch.stack([heading.cos(), heading.sin()], -1)
            initial_ground = term.ground_under_base().clone()
            alive = torch.ones(n, dtype=torch.bool, device=dev)
            fell = torch.zeros_like(alive)
            max_rise = torch.zeros(n, device=dev)
            max_progress = torch.zeros_like(max_rise)
            net_rise = torch.zeros_like(max_rise)
            net_progress = torch.zeros_like(max_rise)
            speed_sum, count = torch.zeros_like(max_rise), torch.zeros_like(max_rise)
            body_hit = torch.zeros_like(alive)
            riser_hit = torch.zeros_like(alive)
            max_air = torch.zeros_like(count)
            edge_landings = torch.zeros_like(count)
            edge_support_steps = torch.zeros_like(count)
            landing_count = torch.zeros(n, 4, device=dev, dtype=torch.long)
            if gait:
                feet_sensor = env.scene["feet_ground_contact"]
                previous_support = feet_sensor.data.current_contact_time > 0
            for step in range(steps):
                obs, _, terminated, truncated, _ = env.step(actor(obs['actor']))
                fell |= alive & terminated
                valid = alive & ~terminated & ~truncated
                data = term.robot.data
                progress = ((data.root_link_pos_w[:, :2] - initial_xy) * direction).sum(-1)
                rise = term.ground_under_base() - initial_ground
                max_rise = torch.where(valid, torch.maximum(max_rise, rise), max_rise)
                max_progress = torch.where(valid, torch.maximum(max_progress, progress), max_progress)
                net_rise = torch.where(valid, rise, net_rise)
                net_progress = torch.where(valid, progress, net_progress)
                speed_sum += valid * (data.root_link_lin_vel_w[:, :2] * direction).sum(-1)
                count += valid
                if gait:
                    body = env.scene["stair_body_contact"].data
                    toe = env.scene["stair_foot_impact"].data
                    foot = feet_sensor.data
                    body_hit |= valid & ((body.found > 0).any(-1) |
                        (body.force_history.norm(dim=-1).amax(dim=(1, 2)) > 1.0))
                    riser = (toe.found > 0) & (toe.normal[..., 2].abs() < 0.5)
                    riser_hit |= valid & (riser & (toe.force.norm(dim=-1) > 5.0)).any(-1)
                    max_air = torch.where(valid, torch.maximum(max_air, foot.current_air_time.amin(-1)), max_air)
                    support = foot.current_contact_time > 0
                    landing = valid[:, None] & support & ~previous_support
                    landing_count += landing
                    edge = tread_edge_contact_risk(env) > 0
                    loaded = foot.force[..., 2] > 1.0
                    edge_landings += (landing & edge & loaded).sum(-1)
                    edge_support_steps += (valid[:, None] & edge & loaded).any(-1)
                    previous_support = support
                alive &= ~(terminated | truncated)
                if video and trial == 0 and step % 2 == 0 and video_error is None:
                    try:
                        img = Image.fromarray(env.render())
                        draw = ImageDraw.Draw(img)
                        text = f'Step {height*100:.0f} cm | t {(step+1)*env.step_dt:.1f}s | forward {net_progress[0]:.2f}m | ascent {net_rise[0]:.2f}m | fell {bool(fell[0])}'
                        draw.rectangle((8, 8, 950, 38), fill=(15, 18, 24))
                        draw.text((14, 16), text, fill=(255, 255, 255))
                        frames.append(np.asarray(img))
                    except Exception as exc:
                        video_error = str(exc)
                if not alive.any():
                    break
            for i in range(n):
                episodes.append({'trial': trial, 'env': i, 'fell': bool(fell[i]),
                                 'max_ground_rise_m': float(max_rise[i]),
                                 'net_ground_rise_m': float(net_rise[i]),
                                 'max_forward_progress_m': float(max_progress[i]),
                                 'net_forward_progress_m': float(net_progress[i]),
                                 'mean_forward_speed_mps': float(speed_sum[i] / count[i].clamp_min(1)),
                                 'three_step_ascent_without_fall': bool(not fell[i] and net_rise[i] >= height * 2.85 and net_progress[i] >= 1.5)})
                if gait:
                    row = episodes[-1]
                    row.update(body_hit=bool(body_hit[i]), foot_riser_hit=bool(riser_hit[i]),
                               max_airborne_s=float(max_air[i]), landings=landing_count[i].tolist())
                    row["edge_landings"] = int(edge_landings[i])
                    row["edge_support_steps"] = int(edge_support_steps[i])
                    row["clean_stepped_ascent"] = (
                        row["three_step_ascent_without_fall"] and not row["body_hit"]
                        and not row["foot_riser_hit"] and row["max_airborne_s"] <= 0.06001
                        and min(row["landings"]) >= 3)
    result = {'checkpoint': str(checkpoint), 'iteration': ckpt.get('iteration'),
              'height_cm': height * 100, 'seed': seed, 'episode_seconds': 8,
              'episodes': episodes, 'n': len(episodes),
              'fall_fraction': np.mean([e['fell'] for e in episodes]).item(),
              'three_step_success_fraction': np.mean([e['three_step_ascent_without_fall'] for e in episodes]).item(),
              'mean_net_ground_rise_m': np.mean([e['net_ground_rise_m'] for e in episodes]).item(),
              'mean_net_forward_progress_m': np.mean([e['net_forward_progress_m'] for e in episodes]).item(),
              'mean_forward_speed_mps': np.mean([e['mean_forward_speed_mps'] for e in episodes]).item(),
              'video_error': video_error}
    if gait:
        for key in ("clean_stepped_ascent", "body_hit", "foot_riser_hit"):
            result[key + "_fraction"] = float(np.mean([row[key] for row in episodes]))
        result["max_airborne_s"] = max(row["max_airborne_s"] for row in episodes)
        result["edge_landings"] = sum(row["edge_landings"] for row in episodes)
        result["edge_support_episode_fraction"] = float(np.mean(
            [row["edge_support_steps"] > 0 for row in episodes]))
        result["clean_centered_ascent_fraction"] = float(np.mean(
            [row["clean_stepped_ascent"] and row["edge_support_steps"] == 0 for row in episodes]))
    if frames and video_error is None:
        import mediapy
        target = out / f'policy_{ckpt.get("iteration")}_{height*100:.0f}cm.mp4'
        mediapy.write_video(str(target), frames, fps=round(1 / env.step_dt / 2))
        result['video'] = str(target)
        for index in [0, len(frames)//2, len(frames)-1]:
            Image.fromarray(frames[index]).save(out / f'policy_{ckpt.get("iteration")}_{height*100:.0f}cm_frame_{index}.png')
    env.close()
    del env, actor, ckpt, frames
    gc.collect()
    torch.cuda.empty_cache()
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--latest', required=True)
    ap.add_argument('--baseline', required=True)
    ap.add_argument('--output', required=True)
    args = ap.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    results = []
    for checkpoint, heights in [(args.latest, [0.02, 0.05, 0.10, 0.15]),
                                (args.baseline, [0.05, 0.15])]:
        for height in heights:
            result = case(checkpoint, height, out,
                          video=(checkpoint == args.latest and height in (0.05, 0.15)))
            results.append(result)
            (out / 'results.json').write_text(json.dumps(results, indent=2))
            print('EVAL_RESULT', json.dumps({k:v for k,v in result.items() if k != 'episodes'}), flush=True)


if __name__ == '__main__':
    main()
