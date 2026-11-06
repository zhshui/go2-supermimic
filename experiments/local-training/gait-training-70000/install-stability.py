from pathlib import Path

root=Path('/root/go2-supermimic')
def edit(path, old, new):
    p=root/path
    s=p.read_text()
    assert s.count(old)==1,(p,old)
    p.write_text(s.replace(old,new))

edit('go2_supermimic/env/mdp.py','def diagonal_support(env:', '''def base_vertical_speed_l2(env: ManagerBasedRlEnv) -> torch.Tensor:
  return env.scene["robot"].data.root_link_lin_vel_w[:, 2].square()


def diagonal_support(env:''')
edit('go2_supermimic/env/env_cfg.py','  cfg.rewards["diagonal_support"] =', '''  cfg.rewards["base_vertical_speed_l2"] = RewardTermCfg(func=mdp.base_vertical_speed_l2, weight=0.0)
  cfg.rewards["diagonal_support"] =''')
edit('go2_supermimic/supermimic/imagine.py','  diagonal_support: float = 0.0',
     '  diagonal_support: float = 0.0\n  base_vertical_speed_l2: float = 0.0')
edit('go2_supermimic/supermimic/imagine.py','      "tilt_rate_l2": s1[:, st.W_B][:, :2].square().sum(-1),',
     '      "tilt_rate_l2": s1[:, st.W_B][:, :2].square().sum(-1),\n      "base_vertical_speed_l2": s1[:, st.V_YAW][:, 2].square(),')
edit('scripts/evaluate_mixed.py', "    contact = env.scene['feet_ground_contact']\n", "")
edit('scripts/evaluate_mixed.py','    last_diagonal = torch.zeros_like(progress)', '''    last_diagonal = torch.zeros_like(progress)
    contact = env.scene["feet_ground_contact"]
    previous_contact = contact.data.current_contact_time > 0
    foot_landings = torch.zeros_like(previous_contact, dtype=torch.long)
    staggered_landings = torch.zeros_like(progress)
    vertical_sq = torch.zeros_like(progress)
    tilt_rate_sq = torch.zeros_like(progress)''')
edit('scripts/evaluate_mixed.py','            fr, fl, rr, rl = planted.unbind(dim=-1)', '''            new_landings = valid[:, None] & planted & ~previous_contact
            foot_landings += new_landings
            landed = new_landings.sum(-1)
            staggered_landings += (landed > 0) & (landed < 4) & previous_contact.any(-1)
            previous_contact = planted
            vertical_sq += valid * robot.data.root_link_lin_vel_w[:, 2].square()
            tilt_rate_sq += valid * robot.data.root_link_ang_vel_b[:, :2].square().sum(-1)
            fr, fl, rr, rl = planted.unbind(dim=-1)''')
edit('scripts/evaluate_mixed.py','        passed = ~fell & (max_air <= 0.06001) & (switches >= 6) & (error <= 0.15)', '''        alternating = (foot_landings.amin(-1) >= 3) & (staggered_landings >= 8)
        passed = ~fell & (max_air <= 0.06001) & alternating & (error <= 0.15)''')
edit('scripts/evaluate_mixed.py',"                               'max_airborne_s':float(max_air.max()),",'''                               'alternating_episodes':int(alternating.sum()),
                               'mean_staggered_landings':float(staggered_landings.mean()),
                               'vertical_speed_rms':float((vertical_sq.sum()/valid_count.sum().clamp_min(1)).sqrt()),
                               'tilt_rate_rms':float((tilt_rate_sq.sum()/valid_count.sum().clamp_min(1)).sqrt()),
                               'max_airborne_s':float(max_air.max()),''')
