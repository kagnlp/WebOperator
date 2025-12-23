import os
import argparse
import json
def update_results(results, src_dir, overwrite=False):

    task_dirs = [
        d
        for d in os.listdir(src_dir)
        if d.startswith("task_") and os.path.isdir(os.path.join(src_dir, d))
    ]

    for task_dir in task_dirs:
        if not overwrite and results.get(task_dir):
            continue  # Skip existing entries unless overwrite is specified
        
        with open(os.path.join(src_dir, task_dir, "exp_summary.json"), "r", encoding="utf-8") as f:
            exp_summary = json.load(f)
            
        task_id = task_dir.split("_")[1]
        if results.get(task_id) is None:
            results[task_id] = {}
        elif results[task_id].get("exp_id", "") == exp_summary["exp_id"]:
            break  # Skip copying
        results[task_id]["exp_id"] = exp_summary["exp_id"]
        # All the keys except "exp_id"
        results[task_id]["stats"] = {
            k: v for k, v in exp_summary.items() if k != "exp_id"
        }
        with open(os.path.join(src_dir, task_dir, "task_info.json"), "r", encoding="utf-8") as f:
            results[task_id]["output"] = json.load(f)
            
        if results[task_id].get("score") is not None:
            del results[task_id]["score"]
        break
    
    return results

def build_parser():
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
    return parser

if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()

    # Traverse all directory in results folder
    # and find all directories that start with "task_"

    # Create destination directory if it doesn't exist
    if not args.dst_dir:
        args.dst_dir = args.src_dir.replace("results", "evaluation")
    os.makedirs(args.dst_dir, exist_ok=True)

    src_dir = args.src_dir

    results = {}
    # Read summary.json from src_dir/ 
    summary_path = os.path.join(src_dir, "summary.json")
    if os.path.exists(summary_path):
        with open(summary_path, "r", encoding="utf-8") as f:
            results = json.load(f)

    results = update_results(results, src_dir, overwrite=args.overwrite)

    # Write summary.json to evaluation directory
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)
