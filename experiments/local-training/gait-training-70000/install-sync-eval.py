from pathlib import Path

p = Path('/root/go2-supermimic/scripts/evaluate_mixed.py')
s = p.read_text()
def edit(old, new):
    global s
    assert s.count(old) == 1, old
    s = s.replace(old, new)
edit('    vertical_sq = torch.zeros_like(progress)', '''    previous_landings = torch.zeros_like(previous_contact)
    synchronized = torch.zeros_like(progress)
    same_axle_landings = torch.zeros_like(progress)
    touchdown_events = torch.zeros_like(progress)
    was_grouped = torch.zeros_like(alive)
    vertical_sq = torch.zeros_like(progress)''')
edit('            landed = new_landings.sum(-1)', '''            recent_landings = new_landings | previous_landings
            grouped = recent_landings.sum(-1) >= 3
            synchronized += valid & grouped & ~was_grouped & (valid_count > 10)
            was_grouped = grouped
            front_pair = (new_landings[:, 0] & recent_landings[:, 1]) | (new_landings[:, 1] & recent_landings[:, 0])
            rear_pair = (new_landings[:, 2] & recent_landings[:, 3]) | (new_landings[:, 3] & recent_landings[:, 2])
            same_axle_landings += (front_pair.float() + rear_pair.float()) * (valid_count > 10)
            touchdown_events += new_landings.sum(-1) * (valid_count > 10)
            previous_landings = new_landings
            landed = new_landings.sum(-1)''')
edit('        passed = ~fell & (max_air <= 0.06001) & alternating & (error <= 0.15)', '''        axle_ratio = 2 * same_axle_landings / touchdown_events.clamp_min(1)
        no_bounding = (synchronized <= 1) & (axle_ratio <= 0.5)
        passed = ~fell & (max_air <= 0.06001) & alternating & no_bounding & (error <= 0.15)''')
edit("                               'alternating_episodes':int(alternating.sum()),",'''                               'alternating_episodes':int(alternating.sum()),
                               'non_bounding_episodes':int(no_bounding.sum()),
                               'episodes_with_repeated_group_landings':int((synchronized > 1).sum()),
                               'mean_group_landings':float(synchronized.mean()),
                               'mean_same_axle_touchdown_ratio':float(axle_ratio.mean()),''')
p.write_text(s)
