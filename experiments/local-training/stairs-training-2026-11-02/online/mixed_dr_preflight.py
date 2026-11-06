"""Check actual DR values, action delays, commands, pushes and imagination."""
import json
import os
import sys
from pathlib import Path

import torch
import yaml
from go2_supermimic.supermimic.train import TrainCfg, Trainer
from go2_supermimic.supermimic import state as st

folder = Path('/root/rl_training_agent/runs/stairs/mixed-migration')
raw = yaml.safe_load((folder/'train-mixed.yaml').read_text())
for key in ['wm_hidden','actor_hidden','critic_hidden','driver_fade']:
    raw[key] = tuple(raw[key])
checkpoint = torch.load(raw['resume'],map_location='cpu',weights_only=True)
start = checkpoint['iteration']
raw.update(num_envs=20,imag_batch=32,wm_batch=32,wm_updates=1,archive_size=128,
           buffer_len=32,compile=False,log_interval=1,save_interval=10,
           log_root=str(folder/'smoke'),run_name='mixed_dr_smoke',iterations=start+10)
os.environ['GO2_STAIRS_OVERRIDES'] = str(folder/'overrides.yaml')
trainer = Trainer(TrainCfg(**raw))
report = {}
try:
    assert trainer.xi.shape==(20,st.CONTEXT_DIM)
    assert st.CONTEXT_DIM==41
    mass_ratio = trainer.xi[:,4:17]
    assert mass_ratio.min()>=.799 and mass_ratio.max()<=1.201
    assert mass_ratio.std()>0.01
    robot=trainer.env.scene['robot']
    kp_ratios,kd_ratios=[],[]
    for act in robot.actuators:
        kp_ratios.append(act.stiffness/act.default_stiffness)
        kd_ratios.append(act.damping/act.default_damping)
    kp=torch.cat(kp_ratios,-1); kd=torch.cat(kd_ratios,-1)
    assert kp.min()>=.799 and kp.max()<=1.201 and kd.min()>=.799 and kd.max()<=1.201
    assert kp.std()>0.01 and kd.std()>0.01
    generator=trainer.term.gen
    assert generator.velocity_mode.eq(1).all()
    assert (generator.velocity_command[:,2].abs()>0.01).any()
    assert generator.velocity_command[:,2].min()<0 and generator.velocity_command[:,2].max()>0
    trainer.collect()
    trainer.env._next_training_push[:]=trainer.env.common_step_counter
    before=trainer.env._training_push_count
    trainer.collect()
    assert trainer.env._training_push_count-before==20
    assert trainer.buf.done[:,(trainer.buf.ptr-2)%trainer.buf.L].eq(1).all()
    trainer.run()
    trainer.cfg.imag_batch=32
    imagined=trainer.improve_policy()
    assert all(torch.isfinite(torch.tensor(x)) for x in imagined.values())
    report={'status':'passed','task':'Go2-Mixed','terrain_variants':list(trainer.env_cfg.scene.terrain.terrain_generator.sub_terrains),
            'context_dim':st.CONTEXT_DIM,'actor_observation_dim':trainer.obs.shape[-1],
            'mass_ratio_range':[float(mass_ratio.min()),float(mass_ratio.max())],
            'kp_ratio_range':[float(kp.min()),float(kp.max())],'kd_ratio_range':[float(kd.min()),float(kd.max())],
            'sampled_commands':generator.velocity_command.cpu().tolist(),
            'sampled_delay_steps':generator.action_delay.cpu().tolist(),
            'push_events':trainer.env._training_push_count,'imagination_update':imagined,
            'smoke_checkpoint':str(trainer.last_ckpt),'warmup_policy_start':trainer.policy_start}
    (folder/'preflight-report.json').write_text(json.dumps(report,indent=2))
    print('MIXED_DR_PREFLIGHT_PASSED',json.dumps(report),flush=True)
finally:
    trainer.writer.close()
    trainer.env.close()
