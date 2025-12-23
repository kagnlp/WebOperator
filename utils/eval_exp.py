from tabulate import tabulate
import os
import argparse
from exp_utils import load_task_configs
from .summarize_exp import update_results
import warnings
from beartype.roar import BeartypeDecorHintPep585DeprecationWarning
import pandas as pd
import json

def build_parser():
    parser = argparse.ArgumentParser(description="WebArena Evaluation Script")
    parser.add_argument(
        "--src_dir",
        type=str,
        required=True,
        help="Directory to store results",
    )
    parser.add_argument(
        "--task_type",
        type=str,
        required=False,
        default="webarena",
        help="Task type to evaluate (optional)",
    )
    parser.add_argument(
        '--site', 
        type=str, 
        nargs="+",           # <-- Accepts one or more values as a list
        required=False, 
        default=None, 
        help='Site(s) to evaluate (optional, space-separated list)'
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Whether to overwrite existing directories in the destination",
        default=False
    )
    return parser

def main():
    parser = build_parser()
    args = parser.parse_args()
     
    ENV_VARS = ("SHOPPING", "SHOPPING_ADMIN", "REDDIT", "GITLAB", "WIKIPEDIA", "MAP", "HOMEPAGE")
    append_wa = lambda x: f"WA_{x}"
    for key in ENV_VARS:
        assert append_wa(key) in os.environ, (
            f"Environment variable {append_wa(key)} missing.\n"
            + "Please set the following environment variables to use WebArena through BrowserGym:\n"
            + "\n".join([append_wa(x) for x in ENV_VARS])
        )
        os.environ[key] = os.environ[append_wa(key)]

    
    # Suppress beartype deprecation warnings from external libraries
    # warnings.filterwarnings("ignore", category=DeprecationWarning, module="beartype")
    warnings.filterwarnings(
        "ignore",
        category=BeartypeDecorHintPep585DeprecationWarning,
    )
    from webarena.evaluation_harness.offline_evaluators import evaluator_router

    if args.task_type == "webarena":
        test_configs = load_task_configs("webarena/test.raw.json")
    elif args.task_type == "webvoyager":
        test_configs = load_task_configs("webvoyager/test.raw.json")
    else:
        raise ValueError(f"Unknown task type: {args.task_type}")
    # Traverse all directory in results folder
    # and find all directories that start with "task_"

    results_dir = args.src_dir
    task_info = {}
    summary_path = os.path.join(results_dir, "summary.json")
    if os.path.exists(summary_path):
        with open(summary_path, "r", encoding="utf-8") as f:
            task_info = json.load(f)

    task_info = update_results(task_info, results_dir, overwrite=args.overwrite)

    # Inside each task directory, there are subdirectories. Sort the directories lexicographically, and take the last one

    # print("task_info keys:", list(task_info.keys()))
    total_score = {
    }
    total_reward = {
    }
    total_tasks = {
    }
    total_actions = {
    }
    total_steps = {
    }
    total_truncate = {
    }
        
    n_tasks = 0

    results = []
    site = args.site
        
    for config_file in test_configs:
        if site is not None:
            if not set(config_file["sites"]).issubset(set(site)):
                continue

        if str(config_file["task_id"]) not in task_info:
            # print(f"Skipping task {config_file['task_id']} -> {config_file['sites']} as it is not found in results.")
            continue
        
        print(f"Evaluating task {config_file['task_id']} -> {config_file['sites']}")
        
        evaluator = evaluator_router(config_file)

        ti = task_info[str(config_file["task_id"])]

        if ti is None or ti == {}:
            continue

        try:
            # if True:
            truncated = True if "agent failed to find a valid solution" in ti["output"]["final_response"] else False
            if ti.get("score") is None: # or ti.get("eval_model") != "gpt-4o": # or ti.get("score") < 1:
                score = evaluator(
                    task_info=ti["output"],
                    config_file=config_file,
                )

                results.append(
                    {
                        "No": n_tasks,
                        "Task ID": config_file["task_id"],
                        "Task": config_file["intent"][:50] + "...",
                        "Sites":  ", ".join(config_file["sites"]),
                        "Score": score, 
                        "Success": score == 1.0,
                        "Generated": ti["stats"]["n_generated"],
                        "Merged": ti["stats"]["n_merged"],
                        "Executed": ti["stats"]["n_executed_w_bt"],
                        "Steps": ti["stats"]["n_executed"],
                        "Truncate": truncated,
                    }
                )
                
            else:
                score = ti["score"]
            
            task_info[str(config_file["task_id"])]["score"] = score
            task_info[str(config_file["task_id"])]["eval_model"] = "gpt-4o"

            # total_score += (score == 1.0)
            # total_reward += score
            # total_tasks += 1
            if len(config_file["sites"]) > 1:
                total_score["multisite"] += (score == 1.0)
                total_reward["multisite"] += score
                total_tasks["multisite"] += 1
                total_actions["multisite"] += ti["stats"]["n_generated"]
                total_steps["multisite"] += ti["stats"]["n_executed_w_bt"]
                total_truncate["multisite"] += (1 if truncated else 0)
                # pass
            else:
                if total_score.get(config_file["sites"][0]) is None:
                    total_score[config_file["sites"][0]] = 0
                    total_reward[config_file["sites"][0]] = 0
                    total_tasks[config_file["sites"][0]] = 0
                    total_actions[config_file["sites"][0]] = 0
                    total_steps[config_file["sites"][0]] = 0
                    total_truncate[config_file["sites"][0]] = 0
                total_score[config_file["sites"][0]] += (score == 1.0)
                total_reward[config_file["sites"][0]] += score
                total_tasks[config_file["sites"][0]] += 1
                total_actions[config_file["sites"][0]] += ti["stats"]["n_generated"]
                total_steps[config_file["sites"][0]] += ti["stats"]["n_executed_w_bt"]
                total_truncate[config_file["sites"][0]] += (1 if truncated else 0)
                
            n_tasks += 1
            

        except Exception as e:
            print(f"Error during evaluation of task {config_file['task_id']}: {e}")
            import traceback
            print(f"Traceback: {traceback.format_exc()}")

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(task_info, f, indent=4)

    if len(results) > 0:
        df = pd.DataFrame(results)
        df = df.sort_values(by="Task ID")
        print(tabulate(df, headers="keys", tablefmt="psql", showindex=False))

    # print accuracy and reward in a table


    websites = ["Overall"] + list(total_tasks.keys())


    table = []
    for w in websites:
        if w == "Overall":
            t = sum(total_tasks.values())
            a = sum(total_score.values()) / t * 100 if t > 0 else 0
            r = sum(total_reward.values()) / t * 100 if t > 0 else 0
        else:
            t = total_tasks[w]
            a = total_score[w] / t * 100 if t > 0 else 0
            r = total_reward[w] / t * 100 if t > 0 else 0
        table.append([w, t, f"{a:.2f}%", f"{r:.2f}%"])

    print(tabulate(table, headers=["Website", "Tasks", "Accuracy", "Reward"], tablefmt="grid", colalign=("center",)*4))

if __name__ == "__main__":
    main()