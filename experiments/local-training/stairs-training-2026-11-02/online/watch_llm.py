"""Periodic LLM analysis and bounded proposals; independent of GPU training."""
import json
import math
import os
import statistics
import time
import uuid
from pathlib import Path

from rl_training_agent.llm_client import LLMClient
from rl_training_agent.monitor import MetricReader
from rl_training_agent.online_control import atomic_json, validate
from rl_training_agent.project_spec import load_config, load_project_spec

ROOT = Path('/root/rl_training_agent/runs/stairs')
agent = load_config('/root/rl_training_agent/configs/stairs/agent.yaml')
policy = agent['online_optimization']
llm = agent['llm']
client = LLMClient(model=llm['model'], base_url=llm['base_url'], wire_api=llm['wire_api'],
                   reasoning_effort=llm['reasoning_effort'], store=False,
                   timeout=180, max_retries=0, max_output_tokens=2500)


def compact(points):
    grouped = {}
    for p in points:
        if p.name.startswith(('Metrics/target/', 'Metrics/hit_rate/', 'time/')):
            continue
        if math.isfinite(p.value):
            grouped.setdefault(p.name, []).append(p)
    result = {}
    for key, rows in grouped.items():
        rows = sorted(rows, key=lambda p: p.step)[-20:]
        values = [p.value for p in rows]
        result[key] = dict(n=len(rows), first_step=rows[0].step, last_step=rows[-1].step,
                           first=values[0], last=values[-1], mean=statistics.mean(values),
                           min=min(values), max=max(values))
    return result


def main():
    while True:
        state = json.loads((ROOT / 'current-run.json').read_text())
        control = Path(state['control_dir']) if state.get('control_dir') else None
        if state['status'] in ('completed', 'failed', 'stopped_by_user'):
            return
        if control and (control / 'snapshot.json').exists():
            break
        time.sleep(10)
    log = control / 'llm-history'
    log.mkdir(exist_ok=True)
    ledger_path = control / 'llm-ledger.json'
    ledger = json.loads(ledger_path.read_text()) if ledger_path.exists() else {
        'calls': 0, 'total_tokens': 0, 'budget_tokens': 0, 'last_iteration': 0, 'status': 'starting'}
    while True:
        state = json.loads((ROOT / 'current-run.json').read_text())
        if state['status'] != 'running':
            return
        try:
            os.kill(state['pid'], 0)
        except ProcessLookupError:
            return
        snap = json.loads((control / 'snapshot.json').read_text())
        iteration = snap['iteration']
        due = ledger['calls'] == 0 or iteration - ledger['last_iteration'] >= policy['interval_iterations']
        if not due or (control / 'pending.json').exists():
            time.sleep(15)
            continue
        if ledger['calls'] >= policy['max_calls'] or ledger['budget_tokens'] + 27000 > policy['total_token_budget']:
            ledger['status'] = 'budget_exhausted'
            atomic_json(ledger_path, ledger)
            print('LLM_BUDGET_EXHAUSTED; training continues', flush=True)
            return
        request_id = uuid.uuid4().hex
        summary = {'task_context': agent['task_context'], 'current_training': snap,
                   'metrics': compact(MetricReader().read(state['log_dir'])),
                   'optimization_protocol': {
                       'instruction': 'Return your diagnosis plus optimization_decision. The user requests actual parameter updates. For the first cycle, choose one concrete conservative optimizer reduction of at most 10 percent as a reversible trial unless numerical-health evidence contraindicates it. Later cycles may hold when evidence does not justify a further reduction. You cannot change code or run shell commands. Do not claim stair success without actual evaluation.',
                       'schema': {'action': 'hold or reduce', 'parameter': 'lr or actor_lr_scale (only for reduce)',
                                  'suggested_value': 'finite number; only for reduce', 'rationale': 'evidence-based explanation'},
                       'first_cycle_requested_trial': ledger['calls'] == 0,
                       'constraints': policy,
                       'note': 'Only the two optimizer parameters below are automatically applied. All other suggestions remain recommendations.'}}
        # Bound payload before spending tokens; metrics are compact summaries, not raw histories.
        payload_size = len(json.dumps(summary, ensure_ascii=False).encode())
        ledger.update(calls=ledger['calls'] + 1, last_iteration=iteration,
                      status='requesting', request_id=request_id, request_started_at=time.time(), budget_tokens=ledger['budget_tokens'] + 27000)
        atomic_json(ledger_path, ledger)
        print(f'LLM_REQUEST iteration={iteration} bytes={payload_size} id={request_id}', flush=True)
        try:
            if payload_size > 24000:
                raise ValueError('summary exceeds 24 KB input limit')
            diagnosis = client.diagnose(summary)
            usage = client.usage[-1]
            used = sum(int(usage.get(k) or 0) for k in ('prompt_tokens', 'completion_tokens'))
            ledger['total_tokens'] += used
            ledger['budget_tokens'] += used - 27000
            decision = diagnosis.get('optimization_decision', {})
            result = {'iteration': iteration, 'request_id': request_id, 'diagnosis': diagnosis, 'usage': usage}
            if decision.get('action') == 'hold':
                result['control_status'] = 'hold'
            elif decision.get('action') == 'reduce':
                parameter = decision['parameter']
                proposal = {'id': request_id, 'experiment_id': snap['experiment_id'],
                            'issued_iteration': iteration, 'expires_iteration': iteration + 2000,
                            'parameter': parameter, 'expected_value': snap['current'].get(parameter),
                            'suggested_value': decision['suggested_value'], 'rationale': decision['rationale']}
                validate(proposal, experiment_id=snap['experiment_id'], iteration=iteration, current=snap['current'])
                atomic_json(control / 'pending.json', proposal)
                result['control_status'] = 'proposed'
                result['proposal'] = proposal
            else:
                result['control_status'] = 'no_valid_control_decision'
            atomic_json(log / f'{iteration}_{request_id}.json', result)
            atomic_json(control / 'latest-llm-diagnosis.json', result)
            ledger.update(status='ready', last_control_status=result['control_status'], last_completed_at=time.time())
            print(f'LLM_RESULT iteration={iteration} decision={result["control_status"]} tokens={ledger["total_tokens"]}', flush=True)
        except Exception as exc:
            message = str(exc).replace(os.getenv('OPENAI_API_KEY', 'not-a-key'), '[REDACTED]')
            ledger.update(status='request_failed', last_error=message, last_completed_at=time.time())
            print(f'LLM_ERROR {type(exc).__name__}: {message}', flush=True)
        atomic_json(ledger_path, ledger)
        time.sleep(15)


if __name__ == '__main__':
    main()
