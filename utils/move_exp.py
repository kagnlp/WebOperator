import os
import argparse
from exp_utils import load_task_configs
import shutil
# Command line arguments
parser = argparse.ArgumentParser(description="WebArena Evaluation Script")
parser.add_argument(
    "--src_dir",
    type=str,
    required=True,
    # help="Directory to store results",
)
parser.add_argument(
    "--dst_dir",
    type=str,
    required=False,
    # help="Directory to store results",
)
parser.add_argument(
    "--overwrite",
    action="store_true",
    help="Whether to overwrite existing directories in the destination",
    default=False
)
args = parser.parse_args()
import json
test_configs = load_task_configs()

# Traverse all directory in results folder
# and find all directories that start with "task_"

# Create destination directory if it doesn't exist
if not args.dst_dir:
    args.dst_dir = args.src_dir.replace("results", "experiments")
os.makedirs(args.dst_dir, exist_ok=True)

src_dir = args.src_dir
task_dirs = [
    d
    for d in os.listdir(src_dir)
    if d.startswith("task_") and os.path.isdir(os.path.join(src_dir, d))
]

# Inside each task directory, there are subdirectories. Sort the directories lexicographically, and take the last one
task_info = {}
for task_dir in task_dirs:
    subdirs = [
        d
        for d in os.listdir(os.path.join(src_dir, task_dir))
        if os.path.isdir(os.path.join(src_dir, task_dir, d))
    ]

    # Find the latest terminated run and load data immediately
    for subdir in sorted(subdirs, reverse=True):
        steps_info_path = os.path.join(src_dir, task_dir, subdir, "steps_info.json")
        if os.path.exists(steps_info_path):
            try:
                with open(steps_info_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if data.get("terminated", False):
                        # Create dst_dir/task_dir if it doesn't exist
                        # If exist, delete it first
                        dst_task_dir = os.path.join(args.dst_dir, task_dir)
                        if os.path.exists(dst_task_dir):
                            if not args.overwrite:
                                # read exp_summary.json and check if exp_id field matches subdir
                                with open(os.path.join(dst_task_dir, "exp_summary.json"), "r", encoding="utf-8") as f:
                                    exp_summary = json.load(f)
                                    if exp_summary.get("exp_id", "") == subdir:
                                        break # Skip copying

                            shutil.rmtree(dst_task_dir)
                        os.makedirs(dst_task_dir, exist_ok=True)
                                            
                        shutil.copytree(
                            os.path.join(src_dir, task_dir, subdir), 
                            dst_task_dir, 
                            ignore=shutil.ignore_patterns("tree.json"),
                            dirs_exist_ok=True
                        )                    
                        # now open exp_summary.json from dst_task_dir and add exp_id field
                        with open(os.path.join(dst_task_dir, "exp_summary.json"), "r+", encoding="utf-8") as f:
                            exp_summary = json.load(f)
                            exp_summary["exp_id"] = subdir
                            f.seek(0)
                            json.dump(exp_summary, f, ensure_ascii=False, indent=4)
                            f.truncate()  
                        break  # Found the latest terminated run
            except (json.JSONDecodeError, KeyError):
                continue  # Skip corrupted or invalid files