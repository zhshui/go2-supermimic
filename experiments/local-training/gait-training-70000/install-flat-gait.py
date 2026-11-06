from pathlib import Path
root = Path('/root/go2-supermimic')
def edit(path, old, new):
    p=root/path
    s=p.read_text()
    assert s.count(old)==1, (p,old)
    p.write_text(s.replace(old,new))

edit('go2_supermimic/supermimic/locomotion.py','def riser_clearance_cost(','''def diagonal_support_score(contact):
    """Reward alternating diagonal pairs through contact geometry."""
    fr, fl, rr, rl = contact.unbind(dim=-1)
    return fr * rl * (1 - fl) * (1 - rr) + fl * rr * (1 - fr) * (1 - rl)


def riser_clearance_cost(''')
for p in ('env/mdp.py','supermimic/imagine.py'):
    edit('go2_supermimic/'+p,'import riser_clearance_cost, upward_launch_cost',
         'import diagonal_support_score, riser_clearance_cost, upward_launch_cost')
edit('go2_supermimic/env/mdp.py','def riser_clearance(env:', '''def diagonal_support(env: ManagerBasedRlEnv) -> torch.Tensor:
  contact = env.scene["feet_ground_contact"].data.current_contact_time > 0
  return diagonal_support_score(contact.float())


def riser_clearance(env:''')
edit('go2_supermimic/env/env_cfg.py','  cfg.rewards["riser_clearance"] =', '''  cfg.rewards["diagonal_support"] = RewardTermCfg(func=mdp.diagonal_support, weight=0.0)
  cfg.rewards["riser_clearance"] =''')
edit('go2_supermimic/supermimic/imagine.py','  riser_clearance: float = 0.0','  riser_clearance: float = 0.0\n  diagonal_support: float = 0.0')
edit('go2_supermimic/supermimic/imagine.py','    if w.riser_clearance:', '''    if w.diagonal_support:
      terms["diagonal_support"] = k["diagonal_support"]
    if w.riser_clearance:''')
edit('go2_supermimic/supermimic/imagine.py','    if cfg.weights.excessive_flight or cfg.weights.stance_slip:', '''    if cfg.weights.excessive_flight or cfg.weights.stance_slip or cfg.weights.diagonal_support:''')
edit('go2_supermimic/supermimic/imagine.py','      airborne = 1.0 - contact.amax(dim=-1)', '''      if cfg.weights.diagonal_support:
        step_k["diagonal_support"] = diagonal_support_score(contact)
      airborne = 1.0 - contact.amax(dim=-1)''')
edit('scripts/evaluate_mixed.py','def evaluate(checkpoint, n=40, difficulty=0.0, seed=42):','def evaluate(checkpoint, n=40, difficulty=0.0, seed=42, flat_only=False):')
edit('scripts/evaluate_mixed.py',"    cfg.curriculum['terrain_levels'].params.update(steps=12000, start_step=0)",'''    cfg.curriculum['terrain_levels'].params.update(steps=12000, start_step=0)
    if flat_only:
        generator = cfg.scene.terrain.terrain_generator
        tile = generator.sub_terrains['flat']
        tile.proportion = 1.0
        generator.sub_terrains = {'flat': tile}
        generator.num_rows = generator.num_cols = 1''')
edit('scripts/evaluate_mixed.py','    slip = torch.zeros_like(progress)', '''    slip = torch.zeros_like(progress)
    max_air = torch.zeros_like(progress)
    switches = torch.zeros_like(progress)
    last_diagonal = torch.zeros_like(progress)''')
edit('scripts/evaluate_mixed.py','            air_samples += valid * (flight > 0)', '''            max_air = torch.where(valid, torch.maximum(max_air, flight), max_air)
            fr, fl, rr, rl = planted.unbind(dim=-1)
            diagonal = (fr & rl & ~fl & ~rr).float() - (fl & rr & ~fr & ~rl).float()
            transition = valid & (diagonal != 0)
            switches += transition & (last_diagonal != 0) & (diagonal != last_diagonal)
            last_diagonal = torch.where(transition, diagonal, last_diagonal)
            air_samples += valid * (flight > 0)''')
edit('scripts/evaluate_mixed.py',"    env.close()",'''    if flat_only:
        error = velocity_error / valid_count.clamp_min(1)
        passed = ~fell & (max_air <= 0.06001) & (switches >= 6) & (error <= 0.15)
        result['flat_gait'] = {'passes':int(passed.sum()), 'mean_diagonal_switches':float(switches.mean()),
                               'max_airborne_s':float(max_air.max()),
                               'episodes_with_long_flight':int((max_air > 0.06001).sum())}
    env.close()''')
