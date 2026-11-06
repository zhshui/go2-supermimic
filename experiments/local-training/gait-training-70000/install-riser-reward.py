from pathlib import Path
root = Path('/root/go2-supermimic/go2_supermimic')
def edit(name, old, new):
    p = root/name
    s = p.read_text()
    assert s.count(old) == 1, name
    p.write_text(s.replace(old,new))

edit('supermimic/locomotion.py', 'import torch', '''import torch

from go2_supermimic.target.terrain import ground


def riser_clearance_cost(feet, heading, field, foot_radius=0.022):
    """Discourage feet approaching a rising edge below the next tread."""
    xy = feet[..., :2]
    forward = torch.stack((heading.cos(), heading.sin()), -1)[:, None, :]
    floor = ground(xy.reshape(-1, 2), field).reshape_as(feet[..., 2])
    next_floor = ground((xy + 0.07 * forward).reshape(-1, 2), field).reshape_as(floor)
    edge = ((next_floor - floor - 0.035) / 0.08).clamp(0.0, 1.0)
    shortage = ((next_floor + foot_radius + 0.015 - feet[..., 2]) / 0.15).clamp(0.0, 1.0)
    return (edge * shortage.square()).mean(-1)
''')
edit('env/mdp.py', 'import upward_launch_cost', 'import riser_clearance_cost, upward_launch_cost')
edit('env/mdp.py', 'def upward_launch(env:', '''def riser_clearance(env: ManagerBasedRlEnv) -> torch.Tensor:
  term = _term(env, "target")
  return riser_clearance_cost(limb_positions_w(term.robot)["feet"], term.robot.data.heading_w, term.height_field)


def upward_launch(env:''')
edit('env/env_cfg.py', '  cfg.rewards["upward_launch"] =', '''  cfg.rewards["riser_clearance"] = RewardTermCfg(func=mdp.riser_clearance, weight=0.0)
  cfg.rewards["upward_launch"] =''')
edit('supermimic/imagine.py', 'import upward_launch_cost', 'import riser_clearance_cost, upward_launch_cost')
edit('supermimic/imagine.py', '  upward_launch: float = 0.0', '  upward_launch: float = 0.0\n  riser_clearance: float = 0.0')
edit('supermimic/imagine.py', '    if w.upward_launch:', '''    if w.riser_clearance:
      feet = geom["feet"]
      base_ground = tr.ground(pose1[:, :2], self.field)
      feet_world = torch.cat((feet[..., :2], feet[..., 2:3] + base_ground[:, None, None]), dim=-1)
      terms["riser_clearance"] = riser_clearance_cost(feet_world, yaw, self.field, st.FOOT_RADIUS)
    if w.upward_launch:''')
