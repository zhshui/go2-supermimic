"""Observations and rewards for chase / dodge / chase-while-dodging.

Rewards use the ground truth from TargetCommand; the actor only sees the
two-slot target observation (command). Chase terms are masked by `has_chase`,
dodge terms by `has_dodge`, so in the combined mode both apply at once.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from go2_supermimic.supermimic.state import FOOT_SITES, limb_positions_w
from go2_supermimic.target import collision
from go2_supermimic.target.terrain import ground, relief
from go2_supermimic.supermimic.locomotion import diagonal_support_score, riser_clearance_cost, upward_launch_cost
from go2_supermimic.target.interface import yaw_rotate_inverse
from go2_supermimic.supermimic.locomotion import stair_tracking, tread_edge_risk

if TYPE_CHECKING:
  from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv

  from go2_supermimic.env.command import TargetCommand


def _term(env: ManagerBasedRlEnv, command_name: str) -> TargetCommand:
  return env.command_manager.get_term(command_name)  # type: ignore[return-value]


def _to_chase_ball(env: ManagerBasedRlEnv, command_name: str):
  """Unit direction robot->chase ball, horizontal distance, robot planar velocity."""
  term = _term(env, command_name)
  data = term.robot.data
  rel = term.gen.chase_pos[:, :2] - data.root_link_pos_w[:, :2]
  dist = rel.norm(dim=-1).clamp(min=1e-6)
  return term, rel / dist.unsqueeze(-1), dist, data.root_link_lin_vel_w[:, :2]


##
# Observations.
##


def height_scan(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  """Base height above each point of the local ground grid (terrain)."""
  return _term(env, command_name).scan()


def cts_tracking_reward(env: ManagerBasedRlEnv, term_name: str) -> torch.Tensor:
  term = _term(env, "target")
  data = term.robot.data
  return stair_tracking(data.root_link_lin_vel_b, data.root_link_ang_vel_b,
                        term.gen.velocity_command)[term_name]


def cts_vertical_speed(env: ManagerBasedRlEnv) -> torch.Tensor:
  return env.scene["robot"].data.root_link_lin_vel_b[:, 2].square()


def cts_base_height(env: ManagerBasedRlEnv, target: float = 0.35) -> torch.Tensor:
  return (_term(env, "target").scan().mean(-1) - target).square()


def cts_power(env: ManagerBasedRlEnv) -> torch.Tensor:
  data = env.scene["robot"].data
  return (data.actuator_force * data.joint_vel).abs().sum(-1)


def cts_leg_collision(env: ManagerBasedRlEnv) -> torch.Tensor:
  force = env.scene["cts_leg_contact"].data.force
  return (force.norm(dim=-1) > 0.1).sum(-1).float()


def tread_edge_contact_risk(env: ManagerBasedRlEnv) -> torch.Tensor:
  term = _term(env, "target")
  feet = limb_positions_w(term.robot)["feet"]
  sensor = env.scene["feet_ground_contact"]
  order = [FOOT_SITES.index(name.split("/")[-1].removesuffix("_foot_collision"))
           for name in sensor.primary_names]
  return tread_edge_risk(feet[:, order, :2], term.height_field)


def tread_edge(env: ManagerBasedRlEnv) -> torch.Tensor:
  contact = env.scene["feet_ground_contact"].data.force[..., 2] > 1.0
  return (tread_edge_contact_risk(env) * contact).sum(-1)


def cts_feet_air_time(env: ManagerBasedRlEnv, target: float = 0.25,
                       contact_threshold: float = 1.0,
                       min_command: float = 0.1) -> torch.Tensor:
  sensor = env.scene["feet_ground_contact"]
  data = sensor.data
  touchdown = sensor.compute_first_contact(env.step_dt)
  touchdown &= (data.force[..., 2] > contact_threshold) & (data.last_air_time > 0.0)
  command = _term(env, "target").gen.velocity_command[:, :2]
  moving = command.norm(dim=-1) > min_command
  return ((data.last_air_time - target) * touchdown).sum(-1) * moving


def cts_feet_regulation(env: ManagerBasedRlEnv, height_target: float = 0.35) -> torch.Tensor:
  robot = env.scene["robot"]
  ids, _ = robot.find_sites(FOOT_SITES, preserve_order=True)
  pos = robot.data.site_pos_w[:, ids]
  floor = _ground_under(env, pos[..., :2])
  height = (pos[..., 2] - floor - 0.022).clamp(0.0, 1.0)
  speed_sq = robot.data.site_lin_vel_w[:, ids, :2].square().sum(-1)
  return (torch.exp(-height / (height_target * 0.025)) * speed_sq).sum(-1)


def stair_tracking_reward(env: ManagerBasedRlEnv, term_name: str) -> torch.Tensor:
  data = env.scene["robot"].data
  velocity = yaw_rotate_inverse(data.root_link_lin_vel_w, data.heading_w)
  command = env.command_manager.get_term("target")
  target = command.gen.velocity_command if command.cfg.velocity_commands else None
  return stair_tracking(velocity, data.root_link_ang_vel_b, target)[term_name]


def lidar(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  """Planar LiDAR ranges around the robot to the static obstacles."""
  return _term(env, command_name).lidar()


##
# Chase rewards.
##


def chase_approach(
  env: ManagerBasedRlEnv, command_name: str, max_speed: float
) -> torch.Tensor:
  """Robot velocity toward the chase ball, clipped to [-max_speed, max_speed]."""
  term, u, _, v = _to_chase_ball(env, command_name)
  return ((v * u).sum(-1).clamp(-max_speed, max_speed)) * term.has_chase.float()


def chase_tag(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  """Tag reward: Gaussian of the head-to-ball distance while taggable (see
  collision.tag_activation)."""
  term = _term(env, command_name)
  data = term.robot.data
  return collision.tag_activation(
    term.gen.state(), data.root_link_pos_w[:, :2], data.heading_w, term.cfg.target
  )


def facing_target(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  """cos(angle between heading and the chase ball), in [0, 1]. Keeps it in view."""
  term, u, _, _ = _to_chase_ball(env, command_name)
  h = term.robot.data.heading_w
  fwd = torch.stack([h.cos(), h.sin()], dim=-1)
  return 0.5 * (1.0 + (fwd * u).sum(-1)) * term.has_chase.float()


##
# Dodge rewards.
##


def evade_danger(env: ManagerBasedRlEnv, command_name: str, std: float) -> torch.Tensor:
  """Positive when the dodge obstacle is close to the body box; use a negative weight."""
  term = _term(env, command_name)
  d = term.body_clearance().clamp(min=0.0)
  return torch.exp(-((d / std) ** 2)) * term.has_dodge.float()


def evade_anticipate(env: ManagerBasedRlEnv, command_name: str, std: float) -> torch.Tensor:
  """Positive when the thrown obstacle is on course to pass close to the body
  (at its predicted closest approach); an early danger signal. Negative weight."""
  term = _term(env, command_name)
  d = term.predicted_body_clearance().clamp(min=0.0)
  return torch.exp(-((d / std) ** 2)) * term.has_dodge.float()


def hit(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  return _term(env, command_name).hit


##
# Shared rewards.
##


def crouch(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  """Base height below 0.23 m outside of a HIGH-bar duck; use a negative weight."""
  term = _term(env, command_name)
  pos = term.base_local()
  return collision.crouch_violation(term.gen.state(), pos[:, :2], pos[:, 2])


def joint_deviation_l2(env: ManagerBasedRlEnv, entity_name: str = "robot") -> torch.Tensor:
  """Squared joint deviation from the default stance; use a negative weight."""
  d = env.scene[entity_name].data
  return (d.joint_pos - d.default_joint_pos).square().sum(-1)


def outside_arena(env: ManagerBasedRlEnv, command_name: str, margin: float) -> torch.Tensor:
  """Distance beyond (arena_radius + margin); use a negative weight."""
  term = _term(env, command_name)
  r = (term.robot.data.root_link_pos_w[:, :2] - term.gen.center).norm(dim=-1)
  return (r - term.cfg.target.arena_radius - margin).clamp(min=0.0)


def tilt_rate_l2(env: ManagerBasedRlEnv, entity_name: str = "robot") -> torch.Tensor:
  """Squared roll and pitch rates of the base (balance in flight and on landing);
  use a negative weight."""
  w = env.scene[entity_name].data.root_link_ang_vel_b
  return w[:, :2].square().sum(-1)


def _ground_under(env: ManagerBasedRlEnv, xy: torch.Tensor) -> torch.Tensor:
  """Terrain height under points; 0 unless the target command owns a height field."""
  terms = env.command_manager.active_terms
  field = env.command_manager.get_term("target").height_field if "target" in terms else None
  return ground(xy, field)


def hard_landing(env: ManagerBasedRlEnv, near_ground: float = 0.05,
                 entity_name: str = "robot") -> torch.Tensor:
  """Squared downward base speed while a foot is near the ground (absorb the
  landing instead of slamming); use a negative weight. Foot heights are taken
  above the ground under the base, as in imagination."""
  robot = env.scene[entity_name]
  g = _ground_under(env, robot.data.root_link_pos_w[:, :2])
  feet_z = (limb_positions_w(robot)["feet"][..., 2] - g.unsqueeze(-1)).min(-1).values
  vz = robot.data.root_link_lin_vel_w[:, 2]
  return (-vz).clamp(min=0.0).square() * (feet_z < near_ground).float()


def excessive_flight(env: ManagerBasedRlEnv) -> torch.Tensor:
  air_time = env.scene["feet_ground_contact"].data.current_air_time
  return ((air_time.amin(dim=-1) - 0.04) / 0.06).clamp(0.0, 2.0)


def base_vertical_speed_l2(env: ManagerBasedRlEnv) -> torch.Tensor:
  return env.scene["robot"].data.root_link_lin_vel_w[:, 2].square()


def diagonal_support(env: ManagerBasedRlEnv) -> torch.Tensor:
  contact = env.scene["feet_ground_contact"].data.current_contact_time > 0
  return diagonal_support_score(contact.float())


def riser_clearance(env: ManagerBasedRlEnv, swing_only: bool = False) -> torch.Tensor:
  term = _term(env, "target")
  support_time = env.scene["feet_ground_contact"].data.current_contact_time if swing_only else None
  return riser_clearance_cost(limb_positions_w(term.robot)["feet"], term.robot.data.heading_w, term.height_field, support_time=support_time)


def upward_launch(env: ManagerBasedRlEnv) -> torch.Tensor:
  term = _term(env, "target")
  data = term.robot.data
  terrain_relief = relief(data.root_link_pos_w[:, :2], data.heading_w, term.height_field, None, grad=False)
  return upward_launch_cost(data.root_link_lin_vel_w[:, 2], terrain_relief)


def stance_slip(env: ManagerBasedRlEnv) -> torch.Tensor:
  robot = env.scene["robot"]
  ids, _ = robot.find_sites(FOOT_SITES, preserve_order=True)
  contact = env.scene["feet_ground_contact"].data.current_contact_time > 0.0
  foot_speed_sq = robot.data.site_lin_vel_w[:, ids, :2].square().sum(-1)
  return (contact * foot_speed_sq.clamp(max=4.0)).mean(-1)


def upright(env: ManagerBasedRlEnv, std: float, entity_name: str = "robot") -> torch.Tensor:
  g = env.scene[entity_name].data.projected_gravity_b
  return torch.exp(-(g[:, :2].square().sum(-1)) / std**2)


def base_height_low(
  env: ManagerBasedRlEnv, min_height: float, entity_name: str = "robot"
) -> torch.Tensor:
  """How far the base is below min_height (above the ground under it); use a
  negative weight."""
  pos = env.scene[entity_name].data.root_link_pos_w
  z = pos[:, 2] - _ground_under(env, pos[:, :2])
  return (min_height - z).clamp(min=0.0)


def obstacle_contact(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  """Depth the body reaches into the slightly inflated static obstacles. Negative weight."""
  return _term(env, command_name).obstacle_penetration()


##
# Curriculum.
##


def terrain_levels_by_progress(env: ManagerBasedRlEnv, env_ids: torch.Tensor,
                               steps: int, start_step: int = 0) -> dict[str, torch.Tensor]:
  """Terrain difficulty rises with training progress: row k (of R) opens once
  k / R of `steps` env steps have passed; each reset picks a random open row,
  so the easier rows keep being practised."""
  terrain = env.scene.terrain
  if terrain is None or terrain.terrain_origins is None:
    return {}
  rows = terrain.terrain_origins.shape[0]
  progress = min(1.0, max(0.0, (env.common_step_counter - start_step) / max(1, steps)))
  top = min(rows - 1, int(progress * rows))
  levels = torch.randint(0, top + 1, (len(env_ids),), device=env_ids.device)
  terrain.terrain_levels[env_ids] = levels
  terrain.env_origins[env_ids] = terrain.terrain_origins[levels, terrain.terrain_types[env_ids]]
  return {"max_level": torch.tensor(float(top))}
