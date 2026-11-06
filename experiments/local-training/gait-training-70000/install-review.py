from pathlib import Path

root = Path('/root/go2-supermimic')
stage = Path('/root/rl_training_agent/runs/stairs/gait-review')
stage.mkdir(exist_ok=True)

def edit(relative, before, after):
    path = root / relative
    source = path.read_text()
    if after in source:
        return
    assert source.count(before) == 1, path
    path.write_text(source.replace(before, after))

edit('go2_supermimic/supermimic/locomotion.py', 'import torch', '''import torch


def upward_launch_cost(vertical_speed, relief):
    """Penalize fast upward body motion mainly on low-relief ground."""
    span = relief.amax(dim=-1) - relief.amin(dim=-1)
    level_gate = ((0.13 - span) / 0.08).clamp(0.0, 1.0)
    return level_gate * (vertical_speed - 0.25).clamp(0.0, 1.0).square()
''')
edit('go2_supermimic/env/mdp.py', 'from go2_supermimic.target.terrain import ground',
     'from go2_supermimic.target.terrain import ground, relief\nfrom go2_supermimic.supermimic.locomotion import upward_launch_cost')
edit('go2_supermimic/env/mdp.py', 'def stance_slip(env:', '''def upward_launch(env: ManagerBasedRlEnv) -> torch.Tensor:
  term = _term(env, "target")
  data = term.robot.data
  terrain_relief = relief(data.root_link_pos_w[:, :2], data.heading_w, term.height_field, None, grad=False)
  return upward_launch_cost(data.root_link_lin_vel_w[:, 2], terrain_relief)


def stance_slip(env:''')
edit('go2_supermimic/env/env_cfg.py', '  cfg.rewards["stance_slip"] =', '''  cfg.rewards["upward_launch"] = RewardTermCfg(func=mdp.upward_launch, weight=0.0)
  cfg.rewards["stance_slip"] =''')
edit('go2_supermimic/supermimic/imagine.py', 'from go2_supermimic.target import obstacles as ob',
     'from go2_supermimic.supermimic.locomotion import upward_launch_cost\nfrom go2_supermimic.target import obstacles as ob')
edit('go2_supermimic/supermimic/imagine.py', '  stance_slip: float = 0.0',
     '  stance_slip: float = 0.0\n  upward_launch: float = 0.0')
edit('go2_supermimic/supermimic/imagine.py', '    total = sum(getattr(w, name) * v for name, v in terms.items())',
     '''    if w.upward_launch:
      terms["upward_launch"] = upward_launch_cost(s1[:, st.V_YAW][:, 2], k["terrain_relief"])
    total = sum(getattr(w, name) * v for name, v in terms.items())''')
edit('go2_supermimic/supermimic/imagine.py', '    r, _ = self._rewards(s, s_next, pose_next, gst, a, a_prev, step_k)',
     '''    if cfg.weights.upward_launch:
      step_k["terrain_relief"] = self.relief(pose_next, gst, cfg.sensor_grad)
    r, _ = self._rewards(s, s_next, pose_next, gst, a, a_prev, step_k)''')
edit('scripts/evaluate_mixed.py', '''def evaluate(checkpoint, n=40, difficulty=0.0):
    torch.manual_seed(42)''', '''def evaluate(checkpoint, n=40, difficulty=0.0, seed=42):
    torch.manual_seed(seed)''')
edit('scripts/evaluate_mixed.py', '    cfg.seed = 42', '    cfg.seed = seed')
edit('scripts/evaluate_mixed.py', "'seed':42,", "'seed':seed,")
