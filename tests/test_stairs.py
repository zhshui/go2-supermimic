"""Stair objective gradients and inactive-target replay without a simulator."""

from types import SimpleNamespace

import torch

from go2_supermimic.supermimic.imagine import ImagineCfg, Imagination, RewardWeights
from go2_supermimic.supermimic.locomotion import FORWARD_SPEED, stair_tracking
from go2_supermimic.supermimic import state as st
from go2_supermimic.target.generator import BallTarget, BallTargetCfg, advance
from go2_supermimic.target.interface import oracle_observation


def test_tracking_prefers_forward_motion_and_has_useful_gradients():
  velocity = torch.tensor([[0., 0., 0.], [FORWARD_SPEED, 0., 0.],
                           [-FORWARD_SPEED, 0., 0.]], requires_grad=True)
  angular = torch.zeros(3, 3)
  r = stair_tracking(velocity, angular)["track_lin_vel"]
  assert r[1] > r[0] > r[2]
  r.sum().backward()
  assert velocity.grad[0, 0] > 0
  assert torch.isfinite(velocity.grad).all()
  angular[:, 2] = 1.
  assert (stair_tracking(velocity, angular)["track_ang_vel"] < 1.).all()


def test_imagination_disables_removed_rewards_and_matches_stair_objective():
  weights = RewardWeights.from_env_rewards({
    "track_lin_vel": SimpleNamespace(weight=2.),
    "track_ang_vel": SimpleNamespace(weight=.5),
  })
  assert weights.chase_tag == weights.crouch == weights.evade_danger == 0.
  n = 3
  cfg = BallTargetCfg()
  gen = BallTarget(cfg, n, "cpu")
  xy, yaw = torch.zeros(n, 2), torch.zeros(n)
  inactive = torch.zeros(n, dtype=torch.bool)
  gen.spawn(torch.arange(n), xy, yaw, xy, inactive, inactive)
  gst = advance(gen.state(), torch.full((n,), .02), xy, xy, yaw, cfg)
  assert not gst["has_chase"].any() and not gst["has_dodge"].any()
  assert (oracle_observation(gst, torch.zeros(n, 3), yaw) == 0).all()
  q = torch.zeros(n, 12)
  limits = torch.stack([q - 2., q + 2.], -1)
  consts = st.RobotConsts(q, q + .25, torch.ones(12), torch.ones(12),
                          torch.ones(12) * 20, limits)
  imag = Imagination(ImagineCfg(weights=weights), consts, cfg, 5, "cpu")
  state = torch.zeros(n, st.STATE_DIM)
  state[:, 0], state[:, 3] = .3, -1.
  state[:, 4] = torch.tensor([0., FORWARD_SPEED, -FORWARD_SPEED])
  state.requires_grad_()
  reward, _ = imag._rewards(state, state, torch.zeros(n, 3), gst, q, q, {
    "default_q": q, "action_scale": q + .25, "soft_limits": limits,
    "hint_weight": torch.tensor(20.),
  })
  expected = stair_tracking(state[:, st.V_YAW], state[:, st.W_B])
  assert torch.allclose(reward, .02 * (2 * expected["track_lin_vel"]
                                      + .5 * expected["track_ang_vel"]))
  reward.sum().backward()
  assert torch.isfinite(state.grad).all()
  assert state.grad[0, 4] > 0
