"""Print the comma-separated task keys that have no result yet for a model (empty if complete)."""
import glob, json, os, sys
from make_cfg import TASKS
name = sys.argv[1]
have = set()
for f in glob.glob(f"{os.environ.get('EVAL_ROOT', '/mnt/nvme/scaling/eval')}/results/{name}/*/*/artifacts/*/results_*.json"):
    have |= set(json.load(open(f))["results"])
print(",".join(k for k in TASKS if k not in have))
