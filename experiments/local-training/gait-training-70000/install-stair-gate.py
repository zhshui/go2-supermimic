from pathlib import Path
p = Path('/root/go2-supermimic/scripts/evaluate_stairs.py')
s = p.read_text()
def edit(old,new):
    global s
    assert s.count(old)==1,old
    s=s.replace(old,new)
edit('from mjlab.tasks.registry import load_env_cfg','''from mjlab.tasks.registry import load_env_cfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from go2_supermimic.robot.go2 import FOOT_GEOM_NAMES''')
edit('def case(checkpoint, height, out, video=False, n=32, rounds=3):\n    torch.manual_seed(42)',
     'def case(checkpoint, height, out, video=False, n=32, rounds=3, seed=42, gait=False):\n    torch.manual_seed(seed)')
edit('    cfg.seed = 42','    cfg.seed = seed')
edit('    env = ManagerBasedRlEnv', '''    if gait:
        terrain = ContactMatch(mode="body", pattern="terrain")
        cfg.scene.sensors += (
            ContactSensorCfg(name="stair_body_contact",
                primary=ContactMatch(mode="geom", pattern=".*collision.*", entity="robot", exclude=FOOT_GEOM_NAMES),
                secondary=terrain, fields=("found", "force"), reduce="netforce",
                history_length=cfg.decimation),
            ContactSensorCfg(name="stair_foot_impact",
                primary=ContactMatch(mode="geom", pattern=FOOT_GEOM_NAMES, entity="robot"),
                secondary=terrain, fields=("found", "force", "normal"), reduce="maxforce", num_slots=4),
        )
    env = ManagerBasedRlEnv''')
edit('            for step in range(steps):', '''            body_hit = torch.zeros_like(alive)
            riser_hit = torch.zeros_like(alive)
            max_air = torch.zeros_like(count)
            landing_count = torch.zeros(n, 4, device=dev, dtype=torch.long)
            if gait:
                feet_sensor = env.scene["feet_ground_contact"]
                previous_support = feet_sensor.data.current_contact_time > 0
            for step in range(steps):''')
edit('                alive &= ~(terminated | truncated)', '''                if gait:
                    body = env.scene["stair_body_contact"].data
                    toe = env.scene["stair_foot_impact"].data
                    foot = feet_sensor.data
                    body_hit |= valid & ((body.found > 0).any(-1) |
                        (body.force_history.norm(dim=-1).amax(dim=(1, 2)) > 1.0))
                    riser = (toe.found > 0) & (toe.normal[..., 2].abs() < 0.5)
                    riser_hit |= valid & (riser & (toe.force.norm(dim=-1) > 5.0)).any(-1)
                    max_air = torch.where(valid, torch.maximum(max_air, foot.current_air_time.amin(-1)), max_air)
                    support = foot.current_contact_time > 0
                    landing_count += valid[:, None] & support & ~previous_support
                    previous_support = support
                alive &= ~(terminated | truncated)''')
edit("    result = {'checkpoint':",'''                if gait:
                    row = episodes[-1]
                    row.update(body_hit=bool(body_hit[i]), foot_riser_hit=bool(riser_hit[i]),
                               max_airborne_s=float(max_air[i]), landings=landing_count[i].tolist())
                    row["clean_stepped_ascent"] = (
                        row["three_step_ascent_without_fall"] and not row["body_hit"]
                        and not row["foot_riser_hit"] and row["max_airborne_s"] <= 0.06001
                        and min(row["landings"]) >= 3)
    result = {'checkpoint':''')
edit("'height_cm': height * 100, 'seed': 42,","'height_cm': height * 100, 'seed': seed,")
edit('    if frames and video_error is None:', '''    if gait:
        for key in ("clean_stepped_ascent", "body_hit", "foot_riser_hit"):
            result[key + "_fraction"] = float(np.mean([row[key] for row in episodes]))
        result["max_airborne_s"] = max(row["max_airborne_s"] for row in episodes)
    if frames and video_error is None:''')
p.write_text(s)
