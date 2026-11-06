from pathlib import Path
p=Path('/root/go2-supermimic/scripts/evaluate_mixed.py')
s=p.read_text()
def edit(old,new):
    global s
    assert s.count(old)==1,old
    s=s.replace(old,new)

edit('def evaluate(checkpoint,', '''def support_rhythm(contacts, valid, dt):
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


def evaluate(checkpoint,''')
edit('    with torch.inference_mode():','''    rhythm_contacts, rhythm_valid = [], []
    with torch.inference_mode():''')
edit('            new_landings = valid[:, None] & planted & ~previous_contact','''            if flat_only and _ >= round(0.5 / env.step_dt):
                rhythm_contacts.append(planted & valid[:, None])
                rhythm_valid.append(valid)
            new_landings = valid[:, None] & planted & ~previous_contact'''.replace('and _ >=','and step >='))
edit('        for _ in range(round(8/env.step_dt)):', '        for step in range(round(8/env.step_dt)):')
edit('        passed = ~fell & (max_air <= 0.06001) & alternating & no_bounding & (error <= 0.15)','''        rhythmic, duty, period, cv = support_rhythm(
            torch.stack(rhythm_contacts), torch.stack(rhythm_valid), env.step_dt)
        passed = ~fell & (max_air <= 0.06001) & alternating & no_bounding & rhythmic & (error <= 0.15)''')
edit("                               'non_bounding_episodes':int(no_bounding.sum()),",'''                               'non_bounding_episodes':int(no_bounding.sum()),
                               'rhythmic_episodes':int(rhythmic.sum()),
                               'mean_duty_by_foot':duty.mean(0).tolist(),
                               'mean_cycle_s_by_foot':period.mean(0).tolist(),
                               'mean_cycle_cv_by_foot':cv.mean(0).tolist(),''')
p.write_text(s)
