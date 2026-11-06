import hashlib, json, math, subprocess, sys
from pathlib import Path
import torch
from rl_training_agent.monitor import MetricReader
from rl_training_agent.experiment_manager import ExperimentManager

base=Path('/root/rl_training_agent/validation')
runs=sorted(Path('/root/go2-supermimic/logs/supermimic').glob('*_agent_gpu_smoke'))
run=next(p for p in reversed(runs) if (p/'model_120.pt').exists())
points=MetricReader().read(run)
assert points and all(math.isfinite(p.value) for p in points)
assert max(p.step for p in points)==120
names=sorted({p.name for p in points})
for key in ['train/wm_loss','train/actor_grad_norm','train/critic_loss']:
 assert key in names,key
 assert any(p.value>0 for p in points if p.name==key),key
old=torch.load(run/'model_60.pt',map_location='cpu',weights_only=True)
new=torch.load(run/'model_120.pt',map_location='cpu',weights_only=True)
assert new['iteration']==120
for part in ['actor','critic','world_model']:
 assert all(torch.isfinite(v).all() for v in new[part].values())
 keys=[k for k in new[part] if k.endswith('weight')]
 assert any(not torch.equal(old[part][k],new[part][k]) for k in keys),part
cmd=[sys.executable,'-m','rl_training_agent.cli','analyze','--metrics',str(run),'--config','/root/rl_training_agent/configs/project.yaml','--data-dir','/root/rl_training_agent/data']
proc=subprocess.run(cmd,capture_output=True,text=True,check=True)
analysis=json.loads(proc.stdout)
assert analysis['points']==len(points) and analysis['llm_called'] is False
(base/'analysis-real-training.json').write_text(json.dumps(analysis,ensure_ascii=False,indent=2))
manager=ExperimentManager('/root/rl_training_agent/data')
exp=manager.create(task='Go2-Chase-Flat',reason='120-iteration GPU smoke validation of rl_training_agent',base_config=str(base/'train-smoke.sh'))
backup=manager.backup_checkpoint(run/'model_120.pt',exp.experiment_id)
digest=lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
assert digest(backup)==digest(run/'model_120.pt')
manager.update(exp.experiment_id,status='completed',start_step=0,end_step=120,result_json=json.dumps({'validation':'passed','run_dir':str(run),'metric_points':len(points)}))
report={'status':'passed','run_dir':str(run),'iterations':120,'environments':32,'metric_points':len(points),'metric_names':names,'all_scalars_finite':True,'actor_critic_world_model_updated':True,'anomalies':analysis['anomalies'],'experiment_id':exp.experiment_id,'backup':str(backup),'backup_sha256':digest(backup),'llm_called':False,'torch':torch.__version__,'cuda':torch.version.cuda}
(base/'validation-report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
print(json.dumps(report,ensure_ascii=False,indent=2))
