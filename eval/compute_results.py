import glob
import json
from difflib import SequenceMatcher

def similar(a, b):
    return SequenceMatcher(None, a, b).ratio()

eval_dir = "eval_results"

trials = glob.glob(f"{eval_dir}/*/*.json")

eval_results = {}

for trial in trials:

    with open(trial, "r") as f:
        trial_info = json.load(f)

    if trial_info["policy_type"] not in eval_results:
        eval_results[trial_info["policy_type"]] = {"correct": 0, "total": 0}
    
    string_sim = similar(trial_info["prompt"], trial_info["prompt_eval"])
    if string_sim > 0.8 or trial_info["prompt"] in trial_info["prompt_eval"]:
        eval_results[trial_info["policy_type"]]["correct"] += 1
    eval_results[trial_info["policy_type"]]["total"] += 1

print(json.dumps(eval_results, sort_keys=True, indent=4))




