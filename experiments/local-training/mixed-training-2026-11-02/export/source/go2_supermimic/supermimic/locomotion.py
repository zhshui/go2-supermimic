"""Basic stair locomotion rewards shared by simulation and imagination."""

import torch

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
