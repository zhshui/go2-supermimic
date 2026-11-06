from pathlib import Path

p = Path('/root/go2-supermimic/scripts/evaluate_mixed.py')
s = p.read_text()
s = s.replace('import torch','import numpy as np\nimport torch',1)
marker = 'def evaluate(checkpoint,'
assert s.count(marker) == 1
s = s.replace(marker, '''def trot_like_support(contacts, valid, dt):
    """Check diagonal support and footfall phase over complete cycles."""
    result, cycle_rates, support_rates = [], [], []
    for support, mask in zip(contacts.cpu().numpy().transpose(1, 0, 2), valid.cpu().numpy().T):
        support = support[mask]
        a = support[:, 0] & support[:, 3]
        b = support[:, 1] & support[:, 2]
        diagonal_only = ((a & ~support[:, 1] & ~support[:, 2]) |
                         (b & ~support[:, 0] & ~support[:, 3]))
        rate = float(diagonal_only.mean()) if len(support) else 0.0
        touches = []
        for foot in support.T:
            ticks = np.flatnonzero(foot[1:] & ~foot[:-1]) + 1
            # Ignore single-frame contact flicker, require a sustained landing.
            touches.append([t for t in ticks if t+1 < len(foot) and foot[t+1]])
        complete = passed = 0
        for start, end in zip(touches[0], touches[0][1:]):
            period = end-start
            complete += 1
            if not 0.25 <= period * dt <= 1.0:
                continue
            def has_phase(foot, target, tolerance):
                return any(abs((t-start)/period-target) <= tolerance for t in touches[foot])
            passed += (has_phase(3, 0, .15) and has_phase(1, .5, .15) and has_phase(2, .5, .15))
        ratio = passed / max(complete, 1)
        result.append(complete >= 6 and ratio >= .80 and rate >= .30)
        cycle_rates.append(ratio)
        support_rates.append(rate)
    return torch.tensor(result, device=contacts.device), cycle_rates, support_rates


''' + marker)
old = '        passed = ~fell & (max_air <= 0.06001) & alternating & no_bounding & rhythmic & (error <= 0.15)'
assert old in s
s = s.replace(old, '''        trot_like, cycle_match, diagonal_support = trot_like_support(
            torch.stack(rhythm_contacts), torch.stack(rhythm_valid), env.step_dt)
        passed = (~fell & (max_air <= 0.06001) & alternating & no_bounding &
                  rhythmic & trot_like & (error <= 0.15))''')
s = s.replace("                               'rhythmic_episodes':int(rhythmic.sum()),",'''                               'rhythmic_episodes':int(rhythmic.sum()),
                               'trot_like_episodes':int(trot_like.sum()),
                               'mean_trot_cycle_match':float(np.mean(cycle_match)),
                               'mean_exclusive_diagonal_support':float(np.mean(diagonal_support)),''')
p.write_text(s)
