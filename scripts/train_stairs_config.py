"""Launch the recorded Go2-Stairs world-model experiment from its YAML."""
import json, os, time, traceback
from dataclasses import asdict
from pathlib import Path
import torch, yaml
from go2_supermimic.supermimic.train import TrainCfg, Trainer
from rl_training_agent.experiment_manager import ExperimentManager

root=Path('/root/go2-supermimic')
state_root=Path('/root/rl_training_agent/runs/stairs')
state_path=state_root/'current-run.json'
raw=yaml.safe_load(Path(os.getenv('GO2_STAIRS_TRAIN_CONFIG', str(root/'configs/stairs/train.yaml'))).read_text())
for k in ['wm_hidden','actor_hidden','critic_hidden','driver_fade']:
 raw[k]=tuple(raw[k])
cfg=TrainCfg(**raw)
torch.set_float32_matmul_precision('high')
manager=ExperimentManager('/root/rl_training_agent/data')
exp=manager.create(task=cfg.task,base_config=os.getenv('GO2_STAIRS_TRAIN_CONFIG', str(root/'configs/stairs/train.yaml')),reason=f'SuperMimic {cfg.task} with 4096 envs')
state={'pid':os.getpid(),'experiment_id':exp.experiment_id,'task':cfg.task,'num_envs':cfg.num_envs,'iterations':cfg.iterations,'status':'initializing','started_at_unix':time.time()}
def record(): state_path.write_text(json.dumps(state,ensure_ascii=False,indent=2))
record()
trainer=None
try:
 trainer=Trainer(cfg)
 os.environ['RL_AGENT_CONTROL_DIR']=str(trainer.log_dir.resolve()/'online-control')
 os.environ['RL_AGENT_EXPERIMENT_ID']=exp.experiment_id
 state.update(log_dir=str(trainer.log_dir.resolve()),control_dir=os.environ['RL_AGENT_CONTROL_DIR'],status='running',policy_start=trainer.policy_start,actor_unfrozen_after=trainer.actor_frozen_until); record()
 if trainer.term.cfg.velocity_commands:
  state['runtime_features']={'context_dim':trainer.xi.shape[1],'actor_obs_dim':trainer.obs.shape[1],'mass_ratio_min':float(trainer.xi[:,4:17].min()),'mass_ratio_max':float(trainer.xi[:,4:17].max()),'com_std':trainer.xi[:,1:4].std(0).tolist(),'yaw_command_min':float(trainer.term.gen.velocity_command[:,2].min()),'yaw_command_max':float(trainer.term.gen.velocity_command[:,2].max()),'delay_steps':torch.unique(trainer.term.gen.action_delay).tolist()}; record()
 (trainer.log_dir/'train.yaml').write_text(yaml.safe_dump(asdict(cfg),sort_keys=False))
 (trainer.log_dir/'PROJECT.md').write_text((root/'PROJECT.md').read_text())
 (trainer.log_dir/'llm.yaml').write_text(Path('/root/rl_training_agent/configs/stairs/llm.yaml').read_text())
 initial=trainer.save(trainer.start_it)
 manager.update(exp.experiment_id,status='running',checkpoint=str(initial),start_step=trainer.start_it)
 print('FORMAL_STAIRS_RUN',json.dumps(state),flush=True)
 trainer.run()
 backup=manager.backup_checkpoint(trainer.last_ckpt,exp.experiment_id)
 manager.update(exp.experiment_id,status='completed',end_step=trainer.it,result_json=json.dumps({'log_dir':state['log_dir'],'backup':str(backup)}))
 state.update(status='completed',finished_at_unix=time.time(),last_iteration=trainer.it); record()
except BaseException as exc:
 state.update(status='failed',error=repr(exc),finished_at_unix=time.time()); record()
 manager.update(exp.experiment_id,status='failed',result_json=json.dumps({'error':repr(exc),'log_dir':state.get('log_dir')}))
 raise
finally:
 if trainer is not None:
  trainer.writer.flush(); trainer.writer.close(); trainer.env.close()
