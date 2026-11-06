"""All training/environment/imagination settings via evaluated candidate runs."""
import json
import os
import signal
import subprocess
import time
import uuid
from pathlib import Path

import torch
import yaml

from rl_training_agent.config_overrides import check_finite
from rl_training_agent.llm_client import LLMClient
from rl_training_agent.monitor import MetricReader
from rl_training_agent.online_control import atomic_json
from rl_training_agent.watch_llm import compact

BASE = Path('/root/rl_training_agent/runs/stairs')
PROJECT = Path('/root/go2-supermimic')
GPU_PY = '/root/.venvs/go2-gpu/bin/python'
agent = yaml.safe_load(Path('/root/rl_training_agent/configs/stairs/llm.yaml').read_text())
llm = agent['llm']
client = LLMClient(model=llm['model'], base_url=llm['base_url'], wire_api='responses',
                   reasoning_effort='medium', store=False, timeout=180,
                   max_retries=0, max_output_tokens=2500)
root = BASE/'full-optimization'
root.mkdir(exist_ok=True)
ledger_path = root/'ledger.json'
ledger = json.loads(ledger_path.read_text()) if ledger_path.exists() else {
    'calls':0, 'tokens':0, 'reserved_tokens':0, 'last_iteration':0, 'status':'starting'}


def latest_checkpoint(state):
    items = sorted(Path(state['log_dir']).glob('model_*.pt'), key=lambda p:int(p.stem.split('_')[-1]))
    return next(p for p in reversed(items) if time.time()-p.stat().st_mtime>5)


def run_candidate(changes, state, checkpoint, folder):
    raw = torch.load(checkpoint, map_location='cpu', weights_only=True)['cfg']
    overrides = dict(raw.pop('runtime_overrides', {}))
    overrides.setdefault('environment', {})
    overrides.setdefault('imagination', {})
    for change in changes:
        path, value = change['path'], change['value']
        check_finite(value)
        section, key = path.split('.', 1)
        if section == 'train':
            if key not in raw:
                raise ValueError('unknown TrainCfg field')
            # Goal, output paths and checkpoint lifecycle belong to orchestration.
            if key in ('task','resume','continue_run','run_name','log_root','iterations'):
                raise ValueError('cannot replace the stair task or checkpoint lifecycle')
            raw[key] = value
        elif section in ('environment','imagination'):
            overrides[section][key] = value
        else:
            raise ValueError('use train.*, environment.* or imagination.* paths')
    raw.update(task='Go2-Stairs', resume=str(checkpoint), continue_run=True,
               run_name='stairs_llm_full', log_root='logs/supermimic', iterations=40000)
    (folder/'train.yaml').write_text(yaml.safe_dump(raw, sort_keys=False))
    (folder/'overrides.yaml').write_text(yaml.safe_dump(overrides, sort_keys=False))
    env = os.environ.copy()
    env.pop('OPENAI_API_KEY', None)
    env.update(PYTHONPATH='/root/go2-supermimic:/root', OMP_NUM_THREADS='4',
               MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4', MUJOCO_GL='egl')
    with (folder/'validation.log').open('w') as log:
        subprocess.run([GPU_PY, '/root/rl_training_agent/validate_candidate.py', str(folder)],
                       env=env, cwd=PROJECT, stdout=log, stderr=subprocess.STDOUT,
                       check=True, timeout=420)
    # A validated candidate may replace a run only if this is still that run.
    now = json.loads((BASE/'current-run.json').read_text())
    if now['experiment_id'] != state['experiment_id'] or now['status']!='running':
        raise ValueError('training state changed during candidate validation')
    newest = latest_checkpoint(now)
    raw.update(resume=str(newest))
    (folder/'train.yaml').write_text(yaml.safe_dump(raw, sort_keys=False))
    atomic_json(folder/'switch.json', {'old_run':now, 'resume_checkpoint':str(newest),
                                     'changes':changes, 'status':'validated_switch_requested'})
    os.kill(now['pid'], signal.SIGINT)
    for _ in range(60):
        if not Path(f'/proc/{now["pid"]}').exists():
            break
        time.sleep(1)
    else:
        raise RuntimeError('old process did not stop; candidate not launched')
    def launch(config_file, overrides_file):
        child_env = env.copy()
        child_env.update(GO2_STAIRS_TRAIN_CONFIG=str(config_file),
                         GO2_STAIRS_OVERRIDES=str(overrides_file))
        with (BASE/'training-llm.log').open('a') as log:
            proc = subprocess.Popen(['bash', str(PROJECT/'scripts/train_stairs.sh')],
                cwd=PROJECT, env=child_env, stdin=subprocess.DEVNULL, stdout=log,
                stderr=subprocess.STDOUT, start_new_session=True)
        (BASE/'launcher.pid').write_text(str(proc.pid))
        for _ in range(180):
            time.sleep(2)
            active = json.loads((BASE/'current-run.json').read_text())
            if active.get('pid') == proc.pid and active.get('status') == 'running':
                snapshot_file = Path(active['control_dir'])/'snapshot.json'
                if snapshot_file.exists():
                    health = json.loads(snapshot_file.read_text())
                    if health['iteration'] > int(newest.stem.split('_')[-1]):
                        if health.get('nonfinite_steps',0):
                            proc.send_signal(signal.SIGINT)
                            proc.wait(timeout=60)
                            raise RuntimeError('candidate had nonfinite optimization steps')
                        return active
            if proc.poll() is not None:
                raise RuntimeError('training startup failed; see training-llm.log')
        proc.send_signal(signal.SIGINT)
        proc.wait(timeout=60)
        raise RuntimeError('initialization timeout')
    try:
        active = launch(folder/'train.yaml', folder/'overrides.yaml')
        atomic_json(folder/'switch.json', {'old_run':now, 'new_run':active,
            'resume_checkpoint':str(newest), 'changes':changes, 'status':'applied'})
        return active
    except Exception as exc:
        rollback = torch.load(newest, map_location='cpu', weights_only=True)['cfg']
        old_overrides = rollback.pop('runtime_overrides', {})
        rollback.update(resume=str(newest), continue_run=True, iterations=40000,
                        run_name='stairs_llm_rollback')
        (folder/'rollback-train.yaml').write_text(yaml.safe_dump(rollback))
        (folder/'rollback-overrides.yaml').write_text(yaml.safe_dump(old_overrides))
        recovered = launch(folder/'rollback-train.yaml', folder/'rollback-overrides.yaml')
        atomic_json(folder/'switch.json', {'status':'rolled_back', 'error':str(exc),
            'new_run':recovered, 'resume_checkpoint':str(newest)})
        raise RuntimeError('candidate failed startup; previous configuration restored') from exc



def main():
    while True:
        state = json.loads((BASE/'current-run.json').read_text())
        if state['status'] != 'running':
            return
        control = Path(state['control_dir'])
        if not (control/'snapshot.json').exists():
            time.sleep(10)
            continue
        snapshot = json.loads((control/'snapshot.json').read_text())
        iteration = snapshot['iteration']
        if ledger['calls'] and iteration-ledger['last_iteration'] < 2000:
            time.sleep(20)
            continue
        if ledger['calls'] >= 10 or ledger['reserved_tokens']+40000 > 200000:
            ledger['status']='budget_exhausted'
            atomic_json(ledger_path, ledger)
            return
        checkpoint = latest_checkpoint(state)
        assessment = root/f'assessment_{iteration}'
        assessment.mkdir(exist_ok=True)
        eval_env = os.environ.copy()
        eval_env.pop('OPENAI_API_KEY', None)
        eval_env.update(PYTHONPATH='/root/go2-supermimic:/root', OMP_NUM_THREADS='4',
                        MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4', MUJOCO_GL='egl')
        try:
            with (assessment/'evaluation.log').open('w') as log:
                subprocess.run([GPU_PY, '/root/rl_training_agent/assess_checkpoint.py',
                    str(checkpoint), str(assessment)], env=eval_env, cwd=PROJECT,
                    stdout=log, stderr=subprocess.STDOUT, check=True, timeout=300)
            recent_evaluation = json.loads((assessment/'current-evaluation.json').read_text())
        except Exception as exc:
            recent_evaluation = {'error':str(exc), 'note':'No fresh result; do not infer success'}
        cfg = torch.load(checkpoint, map_location='cpu', weights_only=True)['cfg']
        evaluation = json.loads((BASE/'evaluation_model_14000/results.json').read_text())
        evaluation = [{k:v for k,v in row.items() if k not in ('episodes','checkpoint','video')}
                      for row in evaluation]
        summary = {'task_context':agent['task_context'], 'current_training':snapshot,
            'all_training_parameters':cfg,
            'metrics':compact(MetricReader().read(state['log_dir'])),
            'real_stair_evaluation':evaluation,
            'recent_checkpoint_evaluation':recent_evaluation,
            'evaluation_protocol_note':'Recent checks use 16 episodes per height at 5/15 cm as a quick regression sample; historical benchmark used 96. Policy checkpoint lag during concurrent training is explicit in each record. Do not treat small samples as identical confidence.',
            'optimization_protocol':{
                'instruction':'All training, reward, curriculum, environment and imagination configuration parameters may be changed, including model topology and sampling settings. Preserve the SuperMimic framework and stair-climbing goal. Do not change a parameter just to make a change. Use real evaluation to identify a weakness; hold if no justified modification is apparent. Every change is preflighted and evaluated in a separate candidate before automatically restarting from a checkpoint. Incompatible architecture migrations are rejected by actual checkpoint loading, not silently reinitialized.',
                'schema':{'optimization_decision':{'action':'hold or apply','rationale':'evidence and intended outcome','changes':[{'path':'train.FIELD or environment.PATH or imagination.PATH','value':'new JSON value'}]}},
                'path_examples':['train.wm_updates','train.horizon','train.wm_hidden','train.actor_hidden',
                    'environment.rewards.track_lin_vel.weight','environment.rewards.upright.weight',
                    'environment.curriculum.terrain_levels.params.steps','environment.scene.terrain.terrain_generator.sub_terrains.stairs.step_height_range',
                    'imagination.gamma','imagination.lam'],
                'runtime_overrides':cfg.get('runtime_overrides',{}),
                'requirements':'Existing typed configuration paths only. Complete candidate must load a checkpoint, execute five optimization iterations, and pass actual 5 cm / 15 cm ascent regression tests. Explain tradeoffs; permission is broad but validation still applies.'}}
        folder = root/f'{iteration}_{uuid.uuid4().hex[:8]}'
        folder.mkdir()
        atomic_json(folder/'request-summary.json', summary)
        ledger.update(calls=ledger['calls']+1,last_iteration=iteration,
                      reserved_tokens=ledger['reserved_tokens']+40000,status='requesting')
        atomic_json(ledger_path, ledger)
        print('FULL_LLM_REQUEST',iteration,flush=True)
        try:
            if len(json.dumps(summary,ensure_ascii=False).encode()) > 32000:
                raise ValueError('summary exceeds 32 KB limit')
            diagnosis=client.diagnose(summary)
            usage=client.usage[-1]
            tokens=sum(int(usage.get(k) or 0) for k in ('prompt_tokens','completion_tokens'))
            ledger['tokens']+=tokens
            ledger['reserved_tokens']+=tokens-40000
            atomic_json(folder/'diagnosis.json',{'diagnosis':diagnosis,'usage':usage})
            decision=diagnosis.get('optimization_decision',{})
            if not decision:
                explicit_holds = [x for x in diagnosis.get('suggestions', [])
                                  if isinstance(x, dict) and x.get('action')=='hold']
                has_structured_changes = any(isinstance(x,dict) and x.get('changes')
                                             for x in diagnosis.get('suggestions', []))
                if explicit_holds and not has_structured_changes:
                    decision={'action':'hold','rationale':explicit_holds[0],
                              'normalized_from':'explicit suggestion action=hold'}
            atomic_json(folder/'decision.json',decision)
            if decision.get('action')=='apply' and decision.get('changes'):
                ledger['status']='validating_candidate'
                atomic_json(ledger_path,ledger)
                active=run_candidate(decision['changes'],state,checkpoint,folder)
                ledger.update(status='applied',active_experiment=active['experiment_id'])
                print('FULL_LLM_APPLIED',json.dumps(decision['changes']),flush=True)
            elif decision.get('action')=='hold':
                ledger['status']='hold'
                print('FULL_LLM_HOLD',iteration,flush=True)
            else:
                raise ValueError('no valid optimization decision')
        except Exception as exc:
            message=str(exc).replace(os.getenv('OPENAI_API_KEY','not-a-key'),'[REDACTED]')
            ledger.update(status='rejected_or_failed',last_error=message)
            atomic_json(folder/'failure.json',{'error':message})
            print('FULL_LLM_REJECTED_OR_FAILED',message,flush=True)
        atomic_json(ledger_path,ledger)
        time.sleep(20)


if __name__=='__main__':
    main()
