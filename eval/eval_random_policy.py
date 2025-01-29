import os 
import numpy as np
import yaml
import json
import itertools
import argparse
import random
import glob
import cv2
policys = {
           "cf_filtered" : ("pleasant-hill-251", 60000),
        #    "cf_obs": ("restful-resonance-255", 80000),
        #    "obs_only" : ("young-fog-250", 80000),
        #    "all" : ("glowing-dust-253", 70000),
           }
NUM_TRIALS = 3


def check_done_evals(env_name):
    done_evals = glob.glob(f"eval_results/{env_name}/*.json")
    done_evals = [json.load(open(eval_file, "r")) for eval_file in done_evals]
    done_eval_infos = [(eval_info["prompt"], eval_info["policy_type"], eval_info["trial_num"]) for eval_info in done_evals]

    return done_eval_infos

def main(args):

    # Load env info 
    with open(args.env_info_path, "r") as f:
        env_info = yaml.safe_load(f)
    
    prompts = env_info["prompts"] 
    env_name = env_info["env_name"]   

    # Set up the eval file
    os.makedirs("eval_results", exist_ok=True)
    os.makedirs(f"eval_results/{env_name}", exist_ok=True)

    # All possible eval combos for this env
    eval_combos = list(itertools.product(prompts, list(policys.keys())))
    eval_combos = [(eval_combo[0], eval_combo[1], trial_num) for eval_combo in eval_combos for trial_num in range(NUM_TRIALS)]
    random.shuffle(eval_combos)
    print("Total evals:", len(eval_combos))

    # Check if we have already done some of these evals
    done_evals = check_done_evals(env_name)
    eval_combos = [eval_combo for eval_combo in eval_combos if eval_combo not in done_evals]
    print("Evals to run:", len(eval_combos))

    curr_trial = len(done_evals)

    # Launch the inference server
    stop = False
    while not stop:
        eval_trial_info = eval_combos.pop()
        prompt = eval_trial_info[0]
        policy = policys[eval_trial_info[1]]
        print(f"Running trial {curr_trial} with prompt: {prompt} and policy: {policy}")
        trial_num = eval_trial_info[2]
        try:
            os.system(f"python ~/bigvision-palivla/scripts/inference_server.py\
                --config ~/bigvision-palivla/configs/nav_config_inference.py\
                --resume_checkpoint_dir {policy[0]}\
                --resume_checkpoint_step {policy[1]}\
                --prompt {prompt}")
        except KeyboardInterrupt:
            print("Finished trial")
            pass

        # Give the eval 

        prompts_print = "\n".join(prompts[:-1])
        prompt_eval = input(f"Which prompt was the policy following?\n{prompts_print}\n")

        # Save the eval info
        eval_results = {
            "prompt": prompt,
            "policy": policy,
            "trial_num": trial_num,
            "prompt_eval": prompt_eval,
            "policy_type": eval_trial_info[1]
        }

        with open(f"eval_results/{env_name}/trial_{curr_trial}.json", "w") as f:
            json.dump(eval_results, f)
        curr_trial += 1
        stop_input = input("Stop? (y/n)")
        if stop_input == "y":
            stop = True


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_info_path", type=str, default="env_info.yaml")
    args = parser.parse_args()
    main(args)


    


    