"""Basic stair locomotion rewards shared by simulation and imagination."""

import torch

from go2_supermimic.target.terrain import ground


def diagonal_support_score(contact):
    """Reward alternating diagonal pairs through contact geometry."""
    fr, fl, rr, rl = contact.unbind(dim=-1)
    return fr * rl * (1 - fl) * (1 - rr) + fl * rr * (1 - fr) * (1 - rl)


def tread_edge_risk(foot_xy: torch.Tensor, field) -> torch.Tensor:
    """Per-foot risk from a height jump within 5 cm of the foot center."""
    offsets = foot_xy.new_tensor(((.05, 0), (-.05, 0), (0, .05), (0, -.05)))
    samples = foot_xy.unsqueeze(-2) + offsets
    rim = ground(samples, field)
    center = ground(foot_xy, field).unsqueeze(-1)
    # Pairwise extrema give a well-defined tie gradient on flat patches.
    low = high = center.squeeze(-1)
    for height in rim.unbind(-1):
        low = torch.minimum(low, height)
        high = torch.maximum(high, height)
    span = high - low
    return ((span - 0.035) / 0.015).clamp(0.0, 1.0)


def merge_leg_collision_probabilities(probabilities, include_proximal=False):
    """Count four thighs and four calves; calf samples share one per-leg risk."""
    if include_proximal:
        calf = torch.maximum(probabilities[..., 4:8], probabilities[..., 8:12])
        return probabilities[..., :4].sum(-1) + calf.sum(-1)
    return probabilities.sum(-1)


def advance_support_time(previous, contact, step_dt):
    """Accumulate settled contact time per foot; reset it on liftoff."""
    return torch.where(contact.detach() > 0.5, previous + step_dt, torch.zeros_like(contact))


def riser_clearance_cost(feet, heading, field, foot_radius=0.022, support_time=None):
    """Discourage feet approaching a rising edge below the next tread."""
    xy = feet[..., :2]
    forward = torch.stack((heading.cos(), heading.sin()), -1)[:, None, :]
    floor = ground(xy.reshape(-1, 2), field).reshape_as(feet[..., 2])
    next_floor = ground((xy + 0.11 * forward).reshape(-1, 2), field).reshape_as(floor)
    edge = ((next_floor - floor - 0.015) / 0.05).clamp(0.0, 1.0)
    shortage = ((next_floor + foot_radius + 0.015 - feet[..., 2]) / 0.10).clamp_min(0.0)
    # Keep a nonzero lifting gradient for shortages greater than 10 cm.
    quadratic = shortage.clamp_max(1.0)
    cost = quadratic.square() + 2.0 * (shortage - quadratic)
    if support_time is not None:
        # Keep swing and first-touch costs; suppress only settled support.
        # Detaching eligibility prevents learning to change contact just to gate cost.
        cost = cost * (support_time <= 0.06).to(cost.dtype).detach()
    return (edge * cost).mean(-1)



def upward_launch_cost(vertical_speed, relief):
    """Penalize fast upward body motion mainly on low-relief ground."""
    span = relief.amax(dim=-1) - relief.amin(dim=-1)
    level_gate = ((0.13 - span) / 0.08).clamp(0.0, 1.0)
    return level_gate * (vertical_speed - 0.25).clamp(0.0, 1.0).square()


FORWARD_SPEED = 0.6


def stair_tracking(velocity_yaw: torch.Tensor, angular_velocity_body: torch.Tensor, command: torch.Tensor | None = None):
  """Track forward 0.6 m/s, zero lateral speed and zero yaw rate."""
  if command is None:
    command = torch.zeros_like(velocity_yaw); command[:, 0] = FORWARD_SPEED
  error = (velocity_yaw[:, :2] - command[:, :2]).square().sum(-1)
  return {
    "track_lin_vel": torch.exp(-error / 0.5**2),
    "track_ang_vel": torch.exp(-(angular_velocity_body[:, 2] - command[:, 2]).square() / 0.5**2),
  }


def delayed_action(action, queue, lag):
  history = queue.reshape(-1, 2, 12)
  choices = torch.cat([action[:, None], history], dim=1)
  executed = choices[torch.arange(len(action), device=action.device), lag.long().clamp(0, 2)]
  return executed, choices[:, :2].reshape(-1, 24)


def locomotion_command(gst, reference):
  default = torch.zeros_like(reference); default[:, 0] = FORWARD_SPEED
  if "velocity_mode" not in gst:
    return default
  return torch.where(gst["velocity_mode"].bool().unsqueeze(-1),gst["velocity_command"],default)
