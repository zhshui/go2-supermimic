"""Real locomotion checks grouped by terrain column."""
import gc
from pathlib import Path

import numpy as np
import torch
import go2_supermimic.env
from go2_supermimic.supermimic.train import Actor
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg
from go2_supermimic.target.interface import yaw_rotate_inverse
from go2_supermimic.env.locomotion_dynamics import random_push_before_state


def support_rhythm(contacts, valid, dt):
    """Measure each leg's support cycle independently of footfall order."""
    stance = contacts.float().sum(0) / valid.sum(0).clamp_min(1)[:, None]
    last = torch.full_like(stance, -1.0)
    intervals = torch.zeros_like(stance)
    sum_period = torch.zeros_like(stance)
    sum_squared = torch.zeros_like(stance)
    for step in range(1, len(contacts)):
        t = step * dt
        landing = contacts[step] & ~contacts[step-1] & valid[step, :, None]
        elapsed = t - last
        landing &= (last < 0) | (elapsed >= 0.10)
        complete = landing & (last >= 0)
        intervals += complete
        sum_period += elapsed * complete
        sum_squared += elapsed.square() * complete
        last = torch.where(landing, t, last)
    mean = sum_period / intervals.clamp_min(1)
    cv = (sum_squared / intervals.clamp_min(1) - mean.square()).clamp_min(0).sqrt() / mean.clamp_min(dt)
    good_leg = ((stance >= 0.42) & (stance <= 0.80) & (intervals >= 6) &
                (mean >= 0.25) & (mean <= 1.0) & (cv <= 0.30))
    return good_leg.all(-1), stance, mean, cv


def trot_like_support(contacts, valid, dt):
    """Check diagonal support and footfall phase over complete cycles."""
    result, cycle_rates, support_rates = [], [], []
    for support, mask in zip(contacts.cpu().numpy().transpose(1, 0, 2), valid.cpu().numpy().T):
        support = support[mask]
        a = support[:, 0] & support[:, 3]
        b = support[:, 1] & support[:, 2]
        diagonal_only = ((a & ~support[:, 1] & ~support[:, 2]) |
                         (b & ~support[:, 0] & ~support[:, 3]))
        rate = float(diagonal_only.mean()) if len(support) else 0.0
        touches = []
        for foot in support.T:
            ticks = np.flatnonzero(foot[1:] & ~foot[:-1]) + 1
            # Ignore single-frame contact flicker, require a sustained landing.
            touches.append([t for t in ticks if t+1 < len(foot) and foot[t+1]])
        complete = passed = 0
        for start, end in zip(touches[0], touches[0][1:]):
            period = end-start
            complete += 1
            if not 0.25 <= period * dt <= 1.0:
                continue
            def has_phase(foot, target, tolerance):
                return any(abs((t-start)/period-target) <= tolerance for t in touches[foot])
            passed += (has_phase(3, 0, .15) and has_phase(1, .5, .15) and has_phase(2, .5, .15))
        ratio = passed / max(complete, 1)
        result.append(complete >= 6 and ratio >= .80 and rate >= .30)
        cycle_rates.append(ratio)
        support_rates.append(rate)
    return torch.tensor(result, device=contacts.device), cycle_rates, support_rates


def evaluate(checkpoint, n=40, difficulty=0.0, seed=42, flat_only=False):
    torch.manual_seed(seed)
    cfg = load_env_cfg('Go2-Mixed', play=True)
    cfg.seed = seed
    cfg.scene.num_envs = n
    cfg.curriculum['terrain_levels'].params.update(steps=12000, start_step=0)
    if flat_only:
        generator = cfg.scene.terrain.terrain_generator
        tile = generator.sub_terrains['flat']
        tile.proportion = 1.0
        generator.sub_terrains = {'flat': tile}
        generator.num_rows = generator.num_cols = 1
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
    air_samples = torch.zeros_like(progress)
    long_air_samples = torch.zeros_like(progress)
    slip = torch.zeros_like(progress)
    max_air = torch.zeros_like(progress)
    switches = torch.zeros_like(progress)
    last_diagonal = torch.zeros_like(progress)
    contact = env.scene['feet_ground_contact']
    previous_contact = contact.data.current_contact_time > 0
    foot_landings = torch.zeros_like(previous_contact, dtype=torch.long)
    staggered_landings = torch.zeros_like(progress)
    previous_landings = torch.zeros_like(previous_contact)
    synchronized = torch.zeros_like(progress)
    same_axle_landings = torch.zeros_like(progress)
    touchdown_events = torch.zeros_like(progress)
    was_grouped = torch.zeros_like(alive)
    vertical_sq = torch.zeros_like(progress)
    tilt_rate_sq = torch.zeros_like(progress)
    site_ids, _ = robot.find_sites(("FR", "FL", "RR", "RL"), preserve_order=True)
    command = env.command_manager.get_term('target')
    rhythm_contacts, rhythm_valid = [], []
    with torch.inference_mode():
        for step in range(round(8/env.step_dt)):
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
            flight = contact.data.current_air_time.amin(dim=-1)
            foot_speed = robot.data.site_lin_vel_w[:, site_ids, :2].norm(dim=-1)
            planted = contact.data.current_contact_time > 0
            max_air = torch.where(valid, torch.maximum(max_air, flight), max_air)
            if flat_only and step >= round(0.5 / env.step_dt):
                rhythm_contacts.append(planted & valid[:, None])
                rhythm_valid.append(valid)
            new_landings = valid[:, None] & planted & ~previous_contact
            foot_landings += new_landings
            recent_landings = new_landings | previous_landings
            grouped = recent_landings.sum(-1) >= 3
            synchronized += valid & grouped & ~was_grouped & (valid_count > 10)
            was_grouped = grouped
            front_pair = (new_landings[:, 0] & recent_landings[:, 1]) | (new_landings[:, 1] & recent_landings[:, 0])
            rear_pair = (new_landings[:, 2] & recent_landings[:, 3]) | (new_landings[:, 3] & recent_landings[:, 2])
            same_axle_landings += (front_pair.float() + rear_pair.float()) * (valid_count > 10)
            touchdown_events += new_landings.sum(-1) * (valid_count > 10)
            previous_landings = new_landings
            landed = new_landings.sum(-1)
            staggered_landings += (landed > 0) & (landed < 4) & previous_contact.any(-1)
            previous_contact = planted
            vertical_sq += valid * robot.data.root_link_lin_vel_w[:, 2].square()
            tilt_rate_sq += valid * robot.data.root_link_ang_vel_b[:, :2].square().sum(-1)
            fr, fl, rr, rl = planted.unbind(dim=-1)
            diagonal = (fr & rl & ~fl & ~rr).float() - (fl & rr & ~fr & ~rl).float()
            transition = valid & (diagonal != 0)
            switches += transition & (last_diagonal != 0) & (diagonal != last_diagonal)
            last_diagonal = torch.where(transition, diagonal, last_diagonal)
            air_samples += valid * (flight > 0)
            long_air_samples += valid * (flight >= 0.06)
            slip += valid * ((planted * foot_speed).sum(-1) / planted.sum(-1).clamp_min(1))
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
                           'airborne_fraction':float((air_samples[select]/valid_count[select].clamp_min(1)).mean()),
                           'long_flight_fraction':float((long_air_samples[select]/valid_count[select].clamp_min(1)).mean()),
                           'stance_slip_mps':float((slip[select]/valid_count[select].clamp_min(1)).mean()),
                           'sampled_levels':levels[select].cpu().tolist()})
    result = {'checkpoint':str(checkpoint),'iteration':ckpt.get('iteration'),
              'seed':seed,'episode_seconds':8,'episodes':n,'difficulty':difficulty,
              'fall_fraction':float(fell.float().mean()),
              'mean_forward_progress_m':float(progress.mean()),
              'mean_velocity_error_mps':float((velocity_error/valid_count.clamp_min(1)).mean()),
              'mean_yaw_rate_error_rps':float((yaw_error/valid_count.clamp_min(1)).mean()),
              'push_events':getattr(env,'_training_push_count',0),'by_terrain':groups}
    if flat_only:
        error = velocity_error / valid_count.clamp_min(1)
        alternating = (foot_landings.amin(-1) >= 3) & (staggered_landings >= 8)
        axle_ratio = 2 * same_axle_landings / touchdown_events.clamp_min(1)
        no_bounding = (synchronized <= 1) & (axle_ratio <= 0.5)
        rhythmic, duty, period, cv = support_rhythm(
            torch.stack(rhythm_contacts), torch.stack(rhythm_valid), env.step_dt)
        trot_like, cycle_match, diagonal_support = trot_like_support(
            torch.stack(rhythm_contacts), torch.stack(rhythm_valid), env.step_dt)
        passed = (~fell & (max_air <= 0.06001) & alternating & no_bounding &
                  rhythmic & trot_like & (error <= 0.15))
        result['flat_gait'] = {'passes':int(passed.sum()), 'mean_diagonal_switches':float(switches.mean()),
                               'alternating_episodes':int(alternating.sum()),
                               'non_bounding_episodes':int(no_bounding.sum()),
                               'rhythmic_episodes':int(rhythmic.sum()),
                               'trot_like_episodes':int(trot_like.sum()),
                               'mean_trot_cycle_match':float(np.mean(cycle_match)),
                               'mean_exclusive_diagonal_support':float(np.mean(diagonal_support)),
                               'mean_duty_by_foot':duty.mean(0).tolist(),
                               'mean_cycle_s_by_foot':period.mean(0).tolist(),
                               'mean_cycle_cv_by_foot':cv.mean(0).tolist(),
                               'episodes_with_repeated_group_landings':int((synchronized > 1).sum()),
                               'mean_group_landings':float(synchronized.mean()),
                               'mean_same_axle_touchdown_ratio':float(axle_ratio.mean()),
                               'mean_staggered_landings':float(staggered_landings.mean()),
                               'vertical_speed_rms':float((vertical_sq.sum()/valid_count.sum().clamp_min(1)).sqrt()),
                               'tilt_rate_rms':float((tilt_rate_sq.sum()/valid_count.sum().clamp_min(1)).sqrt()),
                               'max_airborne_s':float(max_air.max()),
                               'episodes_with_long_flight':int((max_air > 0.06001).sum())}
    env.close()
    del env, actor, robot, ckpt
    gc.collect()
    torch.cuda.empty_cache()
    return result
