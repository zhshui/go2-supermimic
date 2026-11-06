"""Validated, nonblocking parameter updates at training log boundaries."""
import json
import math
import os
import shutil
import time
from pathlib import Path

LIMITS = {"lr": (2e-5, 2e-4), "actor_lr_scale": (0.05, 0.3)}


def atomic_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2))
    tmp.replace(path)


def validate(proposal, *, experiment_id, iteration, current):
    if proposal.get("experiment_id") != experiment_id:
        raise ValueError("wrong experiment")
    issued = int(proposal["issued_iteration"])
    expires = int(proposal["expires_iteration"])
    if not issued <= iteration <= expires or expires - issued > 2000:
        raise ValueError("stale or future proposal")
    parameter = proposal.get("parameter")
    if parameter not in LIMITS:
        raise ValueError("parameter not supported")
    expected = float(proposal["expected_value"])
    value = float(proposal["suggested_value"])
    old = float(current[parameter])
    if not all(math.isfinite(x) for x in (expected, value, old)):
        raise ValueError("nonfinite parameter")
    if not math.isclose(old, expected, rel_tol=1e-8, abs_tol=1e-12):
        raise ValueError("parameter changed since diagnosis")
    low, high = LIMITS[parameter]
    if not low <= value <= high or not 0.8 * old <= value < old:
        raise ValueError("only reductions of at most 20 percent are supported")
    if not isinstance(proposal.get("rationale"), str) or not proposal["rationale"].strip():
        raise ValueError("missing rationale")
    return parameter, old, value


def poll(trainer, stats):
    configured = os.getenv("RL_AGENT_CONTROL_DIR")
    if not configured:
        return
    root = Path(configured)
    root.mkdir(parents=True, exist_ok=True)
    experiment_id = os.environ["RL_AGENT_EXPERIMENT_ID"]
    current = {key: float(getattr(trainer.cfg, key)) for key in LIMITS}
    snapshot = {"experiment_id": experiment_id, "iteration": trainer.it,
                "log_dir": str(trainer.log_dir.resolve()), "current": current,
                "stats": stats, "nonfinite_steps": trainer.nonfinite_steps,
                "updated_at": time.time(), "pid": os.getpid()}
    atomic_json(root / "snapshot.json", snapshot)
    trainer.writer.flush()
    pending = root / "pending.json"
    if not pending.exists():
        return
    proposal = None
    event = {"iteration": trainer.it, "time": time.time()}
    try:
        proposal = json.loads(pending.read_text())
        event["proposal"] = proposal
        prior = getattr(trainer, "_llm_control_changes", 0)
        if prior >= 8:
            raise ValueError("eight-change run limit reached")
        if trainer.nonfinite_steps:
            raise ValueError("nonfinite updates require review before automatic changes")
        parameter, old, new = validate(proposal, experiment_id=experiment_id,
                                       iteration=trainer.it, current=current)
        checkpoint = trainer.save(trainer.it)
        backup_dir = root / "backups"
        backup_dir.mkdir(exist_ok=True)
        backup = backup_dir / f"before_{trainer.it}_{parameter}.pt"
        shutil.copy2(checkpoint, backup)
        setattr(trainer.cfg, parameter, new)
        trainer.last_ckpt = checkpoint
        trainer._llm_control_changes = prior + 1
        event.update(status="applied", parameter=parameter, old_value=old,
                     new_value=new, backup=str(backup))
        trainer.writer.add_scalar(f"control/{parameter}", new, trainer.it)
        print(f"LLM_CONTROL applied {parameter}: {old} -> {new}; backup={backup}", flush=True)
    except Exception as exc:
        event.update(status="rejected", reason=str(exc))
        print(f"LLM_CONTROL rejected: {exc}", flush=True)
    finally:
        with (root / "control-events.jsonl").open("a") as out:
            out.write(json.dumps(event, ensure_ascii=False) + "\n")
        pending.unlink(missing_ok=True)
        atomic_json(root / "last-control-event.json", event)
