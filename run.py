import os
os.environ["PLAYWRIGHT_SKIP_BROWSER_VALIDATION"] = "1"
from dotenv import load_dotenv
load_dotenv(override=True)

import yaml
from weboperator.tree_search_agent import TreeSearchAgentArgs
from browsergym.experiments import EnvArgs, ExpArgs
from beartype.roar import BeartypeDecorHintPep585DeprecationWarning
import warnings
import json
from webarena.docker_status.env import (
    before_task_start,
    after_task_end,
    prepare_environment,
    sync_reset_site,
)
from weboperator.access_control import AccessControl
from weboperator.benchmarks.wa_utils import get_wa_site_url
from weboperator.setup import get_agent_args
from exp_utils import (
    load_site_tasks,
    load_configs_by_task_ids,
    find_last_checkpoint,
    should_run_task,
)
from weboperator.prompt_designer import PromptDesigner

# Suppress beartype deprecation warnings from external libraries
# warnings.filterwarnings("ignore", category=DeprecationWarning, module="beartype")
warnings.filterwarnings(
    "ignore",
    category=BeartypeDecorHintPep585DeprecationWarning,
)

def load_config(config_path="exp_config.yml"):
    """Load experiment configuration from YAML file."""
    with open(config_path, "r") as file:
        return yaml.safe_load(file)


def run_experiment(env_args, agent_args, results_dir="./results", checkpoint=None):
    """Run the experiment with the given environment and agent arguments."""
    exp_args = ExpArgs(
        env_args=env_args,
        agent_args=agent_args,
        enable_debug=True,
    )
    # running and logging results
    exp_args.prepare(results_dir)
    exp_args.run(checkpoint)

    # loading and printing results
    # exp_result = get_exp_result(exp_args.exp_dir)
    # exp_record = exp_result.get_exp_record()

    # for key, val in exp_record.items():
    #     print(f"{key}: {val}")


def parse_args():
    import argparse

    parser = argparse.ArgumentParser(description="Run BrowserGym experiment")
    parser.add_argument(
        "--config",
        type=str,
        default="weboperator/configs/default.yml",
        help="Path to YAML configuration file",
    )
    parser.add_argument(
        "--site", type=str, default=None, help="Site to run (subset of task_ids in config)"
    )
    return parser.parse_args()


def is_task_terminated(task_results_dir):
    # Get all subdirectories
    subdirs = [
        d for d in os.listdir(task_results_dir) if os.path.isdir(os.path.join(task_results_dir, d))
    ]
    if not subdirs:
        return False

    # Take the most recent (sorted in reverse)
    latest_subdir = sorted(subdirs, reverse=True)[0]
    steps_info_path = os.path.join(task_results_dir, latest_subdir, "steps_info.json")

    if os.path.exists(steps_info_path):
        try:
            with open(steps_info_path, "r", encoding="utf-8") as f:
                return json.load(f).get("terminated", False)
        except json.JSONDecodeError:
            pass

    return False


if __name__ == "__main__":
    args = parse_args()

    # Load configuration from YAML file
    config = load_config(args.config)

    # if config["env"]["sites"]:
    #     if "shopping_admin" in config["env"]["sites"]:
    #         os.environ["WA_SHOPPING_ADMIN"] = config["env"]["sites"]["shopping_admin"]
    #     if "shopping" in config["env"]["sites"]:
    #         os.environ["WA_SHOPPING"] = config["env"]["sites"]["shopping"]
    #     if "reddit" in config["env"]["sites"]:
    #         os.environ["WA_REDDIT"] = config["env"]["sites"]["reddit"]
    #     if "gitlab" in config["env"]["sites"]:
    #         os.environ["WA_GITLAB"] = config["env"]["sites"]["gitlab"]
    
    if config["env"]["task_type"] != "openended":
        # Get configuration values
        dataset_path = config["experiment"].get("dataset_path")
        if dataset_path is None:
            if config["env"]["task_type"] == "webarena":
                dataset_path = "webarena/test.raw.json"
            elif config["env"]["task_type"] == "webvoyager":
                dataset_path = "webvoyager/test.raw.json"
            else:
                raise ValueError(f"Unknown task_type: {config['env']['task_type']}")
        task_ids = config["env"].get("task_ids")
        min_task_id = config["env"].get("min_task_id")
        max_task_id = config["env"].get("max_task_id")
        if task_ids is None:
            task_configs = load_site_tasks(config["env"]["sites"], dataset_path=dataset_path)
            task_ids = [cfg["task_id"] for cfg in task_configs]
        else:
            task_configs = load_configs_by_task_ids(task_ids, dataset_path=dataset_path)

    enable_multisite = config["experiment"].get("multisite", False)
    max_steps = config["env"]["max_steps"]
    mode = config["agent"].get("mode", "evaluation")
    checkpoint = config["agent"].get("checkpoint")
    reset_strategy = config["env"].get("reset_strategy", "never")
    fresh_start = config["env"].get("fresh_start", False)
    evaluation_strategy = config["env"].get("evaluation_strategy", "all")
    headless = config["env"].get("headless", True)

    reset_status = {
        "reddit": False,
        "shopping_admin": False,
        "shopping": False,
        "wikipedia": False,
        "map": False,
        "gitlab": False,
    }

    backup_checkpoint = checkpoint
    agent_args = get_agent_args(config)

    if config["env"]["task_type"] == "openended":
        env_args = EnvArgs(
            task_name="openended",
            task_seed=None,
            max_steps=max_steps,
            headless=False,
            wait_for_user_message=True,
            task_kwargs={"start_url": config["env"].get("start_url", "https://www.google.com/")},
        )
        run_experiment(
            env_args,
            agent_args,
            results_dir=f"{config['experiment']['results_dir']}/openended",
            checkpoint=checkpoint,
        )

    # Setup benchmark
    elif mode == "evaluation":
        while len(task_configs) > 0:
            AccessControl.reset()  # Fresh Start for each task

            task_config = task_configs.pop(0)
            task_id = task_config["task_id"]

            if min_task_id is not None and task_id < min_task_id:
                continue

            if max_task_id is not None and task_id > max_task_id:
                continue

            if not enable_multisite and len(task_config["sites"]) > 1:
                print(
                    f"Skipping task {task_id} as it requires multiple sites: {task_config['sites']}"
                )
                continue

            # if enable_multisite and len(task_config["sites"]) < 2:
            #     print(
            #         f"Skipping task {task_id} as it requires single site: {task_config['sites']}"
            #     )
            #     continue
            
            if args.site is not None and args.site != task_config["sites"][0]:
                print(f"Skipping task {task_id} as it is not for site {args.site}")
                continue
            print(f"Running task {task_id}")
            task_results_dir = f"{config['experiment']['results_dir']}/task_{task_id}"

            checkpoint = backup_checkpoint

            if not should_run_task(task_results_dir, evaluation_strategy):
                print(f"Skipping task {task_id} as per evaluation strategy '{evaluation_strategy}'")
                continue

            if checkpoint == "last":
                checkpoint = find_last_checkpoint(task_results_dir)

            if config["env"]["task_type"] == "webarena" and not prepare_environment(
                task_config, reset_strategy
            ):
                raise RuntimeError("Environment preparation failed")

            if fresh_start:
                for site in task_config["sites"]:
                    if reset_status[site] is False:
                        print(f"Fresh start: resetting {site}")
                        sync_reset_site(site)
                        reset_status[site] = True

            if config["env"]["task_type"] == "webarena":
                for site in task_config["sites"]:
                    AccessControl.authorize(site, get_wa_site_url(site))
                    if site in ["reddit", "shopping", "shopping_admin"]:
                        AccessControl.authenticate(site, get_wa_site_url(site))
                        
                PromptDesigner.configure(
                    benchmark=config["env"]["task_type"],
                    multisite=enable_multisite,
                    sites=task_config["sites"]
                )

            if config["env"]["task_type"] == "webarena":
                before_task_start(task_id, task_config["sites"])

            # Create environment args
            env_args = EnvArgs(
                task_name=f"{config['env']['task_type']}.{task_id}",
                task_seed=None,
                max_steps=max_steps,
                headless=headless,
                wait_for_user_message=False,
            )

            run_experiment(
                env_args,
                agent_args,
                results_dir=f"{config['experiment']['results_dir']}/task_{task_id}",
                checkpoint=checkpoint,
            )

            if is_task_terminated(task_results_dir):
                print(f"Task {task_id} already terminated.")
                if config["env"]["task_type"] == "webarena":
                    after_task_end(task_id, task_config["sites"])
                    AccessControl.reset()
            else:
                break

    elif mode == "simulation":
        agent_args = TreeSearchAgentArgs(
            chat_mode=False,
            demo_mode="off",
            use_html=False,
            use_axtree=True,
            use_screenshot=False,
            action_generator=None,
        )
        agent = agent_args.make_agent()
        agent.load_checkpoint(checkpoint)

        while True:
            action, _ = agent.get_action(None)
            print(f"Action: {action}")
            if action.startswith("send_msg_to_user"):
                print("Simulation completed.")
                break
            agent.log("webarena_results/simulation/")
