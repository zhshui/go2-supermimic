"""Replay the latest stable policy on flat ground and the current stair height."""
import os
import time
from pathlib import Path

template = Path(__file__).with_name('replay-template.py').read_text()
checkpoint = max(
    (p for p in Path('logs/supermimic').glob('*gait_review_*/model_*.pt')
     if time.time() - p.stat().st_mtime > 10),
    key=lambda p: (int(p.stem.split('_')[-1]), p.stat().st_mtime),
)
iteration = int(checkpoint.stem.split('_')[-1])
height = float(os.getenv('GO2_RECORD_STAIR_HEIGHT', '0.05'))
output = Path('/tmp/go2-gait-videos') / str(iteration)
output.mkdir(parents=True, exist_ok=True)
template = template.replace(
    'checkpoint = next(Path("logs/supermimic").glob("2026-11-02_22-23-46*/model_60000.pt"))',
    f'checkpoint = Path({str(checkpoint)!r})',
).replace('/tmp/go2-play60000', str(output))
template = template.replace(
    '[("stairs_up_40cm", .15), ("stairs_down_40cm", .10), ("rough", None)]',
    repr([('flat', None), ('stairs_up_40cm', height)]),
)
template = template.replace('    cfg.curriculum = {}', '''    cfg.curriculum = {}
    if height is not None:
        command_cfg = cfg.commands["target"]
        command_cfg.velocity_x_range = (0.6, 0.6)
        command_cfg.velocity_y_range = (0.0, 0.0)
        command_cfg.yaw_rate_range = (0.0, 0.0)''')
template = template.replace('else "rough ground"', 'else "flat ground"').replace('f"60000 |', f'f"{iteration} |')
template = template.replace('(8, 8, 760, 62)', '(8, 8, 820, 82)')
template = template.replace('                frames.append(np.asarray(img))', '''                contact = env.scene["feet_ground_contact"].data
                feet = "  ".join(f"{name}:{int(value)}" for name, value in zip(
                    ("FR","FL","RR","RL"), (contact.current_contact_time[0] > 0).tolist(), strict=True))
                air = float(contact.current_air_time[0].min())
                draw.text((16, 59), f"support {feet} | all feet airborne {air*1000:.0f} ms", fill="white")
                frames.append(np.asarray(img))''')
exec(compile(template, 'gait-replay', 'exec'))
print(f'RECORDING_COMPLETE {iteration} {output}', flush=True)
