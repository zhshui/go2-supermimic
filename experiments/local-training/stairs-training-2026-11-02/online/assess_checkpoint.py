"""Small real-simulator benchmark before each broad LLM optimization cycle."""
import json
import sys
from pathlib import Path

sys.path.insert(0, '/root/go2-supermimic/scripts')
from evaluate_stairs import case

checkpoint, destination = sys.argv[1:]
folder = Path(destination)
folder.mkdir(parents=True, exist_ok=True)
rows = [case(checkpoint, height, folder, n=16, rounds=1) for height in (0.05, 0.15)]
(folder/'current-evaluation.json').write_text(json.dumps(rows, indent=2))
print('CURRENT_CHECKPOINT_REAL_EVALUATION_COMPLETE', flush=True)
