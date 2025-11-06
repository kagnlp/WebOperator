import os
from dotenv import load_dotenv
load_dotenv(override=True)
import json

def load_task_configs(dataset_path="webarena/test.raw.json"):
    with open(dataset_path, "r", encoding="utf-8") as f:
        all_configs_str = f.read()
    for pattern, url_key in {
        "__GITLAB__": "WA_GITLAB",
        "__GITLAB_IP__": "WA_GITLAB_IP", # NEW
        "__REDDIT__": "WA_REDDIT",
        "__SHOPPING__": "WA_SHOPPING",
        "__SHOPPING_ADMIN__": "WA_SHOPPING_ADMIN",
        "__WIKIPEDIA__": "WA_WIKIPEDIA",
        "__MAP__": "WA_MAP",
    }.items():
        all_configs_str = all_configs_str.replace(pattern, os.environ[url_key])
    test_configs = json.loads(all_configs_str)
    return test_configs

def load_site_tasks(target_sites, dataset_path="webarena/test.raw.json"):
    with open(dataset_path, "r", encoding="utf-8") as f:
        all_configs_str = f.read()
    for pattern, url_key in {
        "__GITLAB__": "WA_GITLAB",
        "__GITLAB_IP__": "WA_GITLAB_IP", # NEW
        "__REDDIT__": "WA_REDDIT",
        "__SHOPPING__": "WA_SHOPPING",
        "__SHOPPING_ADMIN__": "WA_SHOPPING_ADMIN",
        "__WIKIPEDIA__": "WA_WIKIPEDIA",
        "__MAP__": "WA_MAP",
    }.items():
        all_configs_str = all_configs_str.replace(pattern, os.environ[url_key])
    test_configs = json.loads(all_configs_str)
    return [cfg for cfg in test_configs if set(cfg["sites"]) <= set(target_sites)]

def load_configs_by_task_ids(task_ids, dataset_path="webarena/test.raw.json"):
    with open(dataset_path, "r", encoding="utf-8") as f:
        all_configs_str = f.read()
    for pattern, url_key in {
        "__GITLAB__": "WA_GITLAB",
        "__GITLAB_IP__": "WA_GITLAB_IP", # NEW
        "__REDDIT__": "WA_REDDIT",
        "__SHOPPING__": "WA_SHOPPING",
        "__SHOPPING_ADMIN__": "WA_SHOPPING_ADMIN",
        "__WIKIPEDIA__": "WA_WIKIPEDIA",
        "__MAP__": "WA_MAP",
    }.items():
        all_configs_str = all_configs_str.replace(pattern, os.environ[url_key])
    test_configs = json.loads(all_configs_str)
    return [cfg for cfg in test_configs if cfg["task_id"] in task_ids]


def find_last_checkpoint(task_results_dir):
    if os.path.exists(task_results_dir):
        subdirs = [
            d for d in os.listdir(task_results_dir)
            if os.path.isdir(os.path.join(task_results_dir, d))
        ]
        for subdir in sorted(subdirs, reverse=True):
            tree_path = os.path.join(task_results_dir, subdir, "tree.json")
            if os.path.exists(tree_path):
                return tree_path
    return None

def should_run_task(task_results_dir, evaluation_strategy):
    if not os.path.exists(task_results_dir) or evaluation_strategy == "all":
        return True
    
    subdirs = [
        d
        for d in os.listdir(task_results_dir)
        if os.path.isdir(os.path.join(task_results_dir, d))
    ]
    
    for subdir in sorted(subdirs, reverse=True):
        steps_info_path = os.path.join(task_results_dir, subdir, "steps_info.json")
        if os.path.exists(steps_info_path):
            try:
                with open(steps_info_path, "r", encoding="utf-8") as f:
                    steps_info = json.load(f)
                    if steps_info.get("terminated", False):
                        print(f"Found terminated experiment in {subdir}.")
                        task_info_path = os.path.join(task_results_dir, subdir, "task_info.json")
                        if os.path.exists(task_info_path):
                            with open(task_info_path, "r", encoding="utf-8") as f:
                                task_info = json.load(f)
                        else:
                            return True
                        if evaluation_strategy == "with_failures":
                            if task_info["score"] == 1.0:
                                return False
                        elif evaluation_strategy == "with_partial_failures":
                            if task_info["score"] == 1.0 or task_info["score"] == 0.0:
                                return False
                        elif evaluation_strategy == "with_no_action_failures":
                            if task_info["final_response"] != "n/a. agent failed to find a valid action.":
                                return False
                        else: # "not_terminated"
                            return False
                        break
            except (json.JSONDecodeError, KeyError):
                continue  # Skip corrupted or invalid files
    
    return True