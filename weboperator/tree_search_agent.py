import base64
import dataclasses
import io
import logging
import sys
import numpy as np
import openai
from PIL import Image
from typing import Dict, List, Any, Optional
import gymnasium as gym

# from Logger import HTMLRenderer
from browsergym.core.action.highlevel import HighLevelActionSet
from browsergym.experiments import AbstractAgentArgs, Agent

from .url_simulator import URLSimulator
from .web_state_node import WebStateNode
import re
import json
from .action_generator import ActionGenerator

from .html_renderer import HTMLRenderer
from .webprm import WebPRM
from tabulate import tabulate

logger = logging.getLogger(__name__)
from .trajectory_manager import _pre_action_trajectory, has_safe_anchestor
from .observation_processor import ObservationProcessor
from .backtrack_manager import BacktrackManager
from .action_processor import ActionProcessor
from .recovery_assistant import RecoveryAssistant
from .action_analyzer import was_destructive, is_terminating
from .action_selector import ActionSelector


from .axtree_utils import is_alert_available, has_axtree_changed


def is_same_node(n1, n2):
    return n1.level == n2.level and n1.position == n2.position


def extract_url_from_goto(action_str):
    match = re.match(r"goto\(['\"](.*?)['\"]\)", action_str)
    if match:
        return match.group(1)
    return None


class TreeSearchAgent(Agent):
    def __init__(
        self,
        chat_mode: bool = False,
        action_generator: Optional[ActionGenerator] = None,
        action_selector: Optional[ActionSelector] = None,
        backtrack_manager: Optional[BacktrackManager] = None,
        exp_dir: str = "",
        checkpoint: str = None,
    ) -> None:
        super().__init__()
        self.chat_mode = chat_mode
        self.goal = None
        self.action_generator = action_generator
        self.discount_factor = 0.8
        self.n_explored = 0  # Total actions after merging
        self.curr_level = 0
        self.curr_pos = 0
        self.checklist = ""
        self.n_steps = 0
        self.revived = False
        self.backtrack_manager = backtrack_manager
        if self.backtrack_manager is not None:
            self.backtrack_manager.reset()
        # self.n_agent_steps = 0
        self.n_expanded = 0
        self.n_actions = 0
        self.root_level = 0
        self.root_position = 0
        self.n_backtracks = 0
        self.exp_dir = exp_dir

        # self.openai_client = openai.OpenAI()
        self.action_set = HighLevelActionSet(
            # define a subset of the action space
            subsets=["webarena"],
            # subsets=["chat", "bid", "coord", "infeas"] # allow the agent to also use x,y coordinates
            strict=False,  # less strict on the parsing of the actions
            multiaction=False,  # does not enable the agent to take multiple actions at once
            demo_mode="off",  # add visual effects
        )
        # use this instead to allow the agent to directly use Python code
        # self.action_set = PythonActionSet())
        # self.html_renderer = HTMLRenderer(
        #         config_file, dstdir, "id_accessibility_tree"
        #     )
        self.action_history = []
        self.tree_root = self.tree_node = WebStateNode(
            last_action=None,
            last_obs_description=None,
            parent=None,
        )

        self.action_selector = action_selector
        if self.action_selector is not None:
            self.action_selector.reset()

        self.trajectory = []
        self.html_renderer = None
        self.local_storage = {}
        self.cookies = {}
        self.delayed_destruction = 0

        if checkpoint is not None:
            self.load_checkpoint(checkpoint)

    def load_checkpoint(self, tree_path: str):
        root = WebStateNode(
            last_action=None,
            last_obs_description=None,
            parent=None,
        )
        tree_json = {}
        with open(tree_path, "r", encoding="utf-8") as f:
            tree_json = json.load(f)
        root.from_json(tree_json)
        self.tree_root = self.tree_node = root
        if self.tree_root.checklist is not None:
            self.checklist = self.tree_root.checklist
            print(f"Checklist loaded from checkpoint: {self.checklist}")

    def _get_trajectory(self) -> List[Dict[str, Any]]:
        # Traverse from current node to root and collect actions and observations
        trajectory = []
        current_node = self.tree_node
        while current_node is not None:
            trajectory.append(
                {
                    "open_pages_urls": current_node.open_pages_urls,
                    "open_pages_titles": current_node.open_pages_titles,
                    "active_page_index": current_node.active_page_index,
                    "axtree_txt": current_node.axtree_txt,
                    "page_too_long": current_node.page_too_long,
                    "screenshot": current_node.screenshot,
                    "last_action": current_node.last_action,
                    "last_action_error": current_node.last_action_error,
                    "last_obs_description": current_node.last_obs_description,
                }
            )
            current_node = current_node.parent

        # Reverse the trajectory to have it from root to current node
        return list(reversed(trajectory))

    def reset(self):
        self.n_explored = 0  # Total actions after merging
        self.curr_level = 0
        self.curr_pos = 0
        self.checklist = ""
        self.n_steps = 0
        if self.backtrack_manager is not None:
            self.backtrack_manager.reset()
        self.n_expanded = 0
        self.n_actions = 0
        self.root_level = 0
        self.root_position = 0
        self.n_backtracks = 0
        self.action_history = []
        self.tree_root = self.tree_node = WebStateNode(
            last_action=None,
            last_obs_description=None,
            parent=None,
        )
        if self.action_selector is not None:
            self.action_selector.reset()

        self.trajectory = []
        self.html_renderer = None
        self.delayed_destruction = 0

    def log(self):
        """Log the agent's state to a file."""
        exp_dir = self.exp_dir
        if not exp_dir:
            return
        if self.html_renderer is None:
            self.html_renderer = HTMLRenderer(exp_dir)

        history = self._get_trajectory()

        if self.tree_node.last_action is not None and self.tree_node.last_action["type"] == "stop":
            with open(f"{exp_dir}/exp_summary.json", "w", encoding="utf-8") as f:
                f.write(
                    json.dumps(
                        {
                            "n_generated_w_error": self.action_generator.generation_count(),
                            "n_generated": self.n_actions,  # Generated
                            "n_merged": self.n_explored,  # Merged
                            "n_executed": self.n_expanded,  # Executed
                            "n_executed_w_bt": self.n_steps,  # Executed (including backtracking)
                            "n_backtracks": self.n_backtracks,  # Number of backtracks
                            "n_destruct": (
                                self.action_selector.n_destruct
                                if self.action_selector is not None
                                else 0
                            ),
                            "n_prune": (
                                self.action_selector.n_prune
                                if self.action_selector is not None
                                else 0
                            ),
                            "n_depth": self.curr_level,  # Depth of the final node
                            # "n_agent_steps": self.n_agent_steps,
                            "discovered_solutions": (
                                self.action_selector.discovered_solutions
                                if self.action_selector is not None
                                else 0
                            ),
                        },
                        indent=2,
                        ensure_ascii=False,
                        default=str,
                    )
                )

        curr_obs = self.trajectory[-1]

        if len(self.trajectory) < 2 or self.trajectory[-2]["action"]["type"] != "stop":
            if curr_obs["action"] is None:
                # if self.tree_node.last_action is None or self.tree_node.last_action["type"] != "stop":
                self.html_renderer.render_observation(
                    curr_obs,
                    self.trajectory[-2]["action_error"] if len(self.trajectory) > 1 else None,
                    render_screenshot=True,
                )
            else:  # Took Action
                self.html_renderer.render_action(
                    curr_obs["action"],
                    curr_obs["obs_description"],
                    backtrack=self.backtrack_manager is not None
                    and self.backtrack_manager.is_backtracking,
                )

        # if not self.backtrack_manager.is_backtracking and curr_obs["action"] is not None and curr_obs["action"]["type"] == "stop":
        #     self.html_renderer.close()

        if self.tree_root is not None:
            tree_json = self.tree_root.to_json()
            with open(f"{exp_dir}/tree.json", "w", encoding="utf-8") as f:
                f.write(json.dumps(tree_json, indent=2, ensure_ascii=False, default=str))
        else:
            logger.warning("No tree root to log.")

        if history[-1]["last_action"] is not None and history[-1]["last_action"]["type"] == "stop":
            with open(f"{exp_dir}/trajectory.json", "w", encoding="utf-8") as f:
                f.write(
                    json.dumps(
                        [
                            {
                                "axtree_txt": step["axtree_txt"],
                                # "screenshot_url": image_to_jpg_base64_url(step["screenshot"]),
                                "open_pages_urls": step["open_pages_urls"],
                                "open_pages_titles": step["open_pages_titles"],
                                "active_page_index": step["active_page_index"],
                                "page_too_long": step["page_too_long"],
                                "last_action": step["last_action"],
                                "last_action_error": step["last_action_error"],
                                "last_obs_description": step["last_obs_description"],
                            }
                            for step in history
                        ],
                        indent=2,
                        ensure_ascii=False,
                    )
                )

        WebPRM.log(exp_dir)
        if self.action_generator is not None:
            self.action_generator.log(exp_dir)

        with open(f"{exp_dir}/full_trajectory.json", "w", encoding="utf-8") as f:
            f.write(
                json.dumps(
                    [
                        {
                            "open_pages_titles": step["open_pages_titles"],
                            "open_pages_urls": step["open_pages_urls"],
                            "active_page_index": step["active_page_index"],
                            "axtree_txt": step["axtree_txt"],
                            "action": step["action"],
                            "action_error": step["action_error"],
                            "obs_description": step["obs_description"],
                            # "screenshot_url": step["screenshot_url"],
                            "http_requests": step["http_requests"],
                        }
                        for step in self.trajectory
                    ],
                    indent=2,
                    ensure_ascii=False,
                )
            )

        # if self.html_renderer is not None:
        #     self.html_renderer.render_file.close()
        #     logger.info(f"Render file saved at {self.html_renderer.render_file.name}")

    def obs_preprocessor(self, obs: dict) -> dict:
        if (
            self.tree_node.last_action is not None
            and is_terminating(self.tree_node.last_action)
            and self.action_selector is not None
        ):
            self.action_selector.action_queue_manager.clear()

        processed_obs = ObservationProcessor.process_obs(obs)

        new_goal = None
        if self.chat_mode:
            if obs["chat_messages"][-1]["role"] == "user":
                new_goal = obs["chat_messages"][-1]["message"]
        elif self.goal is None:
            new_goal = processed_obs["goal_object"][0]["text"]

        if self.backtrack_manager is None or not self.backtrack_manager.is_backtracking:
            # self.n_agent_steps += 1
            self.tree_node.update_from_obs(processed_obs)
            self.tree_node.is_404 = ObservationProcessor.is_404_page(self.tree_node.axtree_txt)
            self.tree_node.http_requests = obs["http_requests"]
            processed_obs["last_action_error"] = RecoveryAssistant.get_recovery_hint(self.tree_node)

        if new_goal is not None and (self.goal is None or new_goal != self.goal):
            print(f"# New Goal Detected: {new_goal}")
            self.goal = new_goal
            # self.reset()
            self.checklist = WebPRM.generate_checklist(
                goal=self.goal, start_url=self.tree_node.url, start_obs=self.tree_node.axtree_txt
            )
            self.tree_node.checklist = self.checklist
            self.tree_node.goal = self.goal

        prev_action_success = True
        if processed_obs["last_action_error"].strip() != "":
            if has_axtree_changed(self.tree_node):
                # Axtree Changes even after error
                logger.info("Axtree changed after action with error. Ignoring the error.")
            else:
                prev_action_success = False

        if (
            (self.backtrack_manager is None or not self.backtrack_manager.is_backtracking)
            and prev_action_success
            and was_destructive(self.tree_node)
        ):
            self.tree_node.destructive = True

        # if not processed_obs["last_action_error"] and is_alert_available(processed_obs["axtree_object"]):
        #     processed_obs["last_action_error"] = "An alert was detected in the page content."

        if processed_obs["last_action_error"].strip():
            print(f"# Error {self.n_steps}: " + processed_obs["last_action_error"])

        if len(self.trajectory) > 0:
            self.trajectory[-1]["action_error"] = processed_obs["last_action_error"]

        if self.backtrack_manager is None or not self.backtrack_manager.is_backtracking:
            self.tree_node.last_action_error = processed_obs["last_action_error"]

        self.trajectory.append(
            {
                "screenshot": processed_obs["screenshot"],
                "open_pages_urls": processed_obs["open_pages_urls"],
                "open_pages_titles": processed_obs["open_pages_titles"],
                "active_page_index": processed_obs["active_page_index"],
                "url": processed_obs["open_pages_urls"][processed_obs["active_page_index"]],
                "axtree_txt": processed_obs["axtree_txt"],
                "action": None,
                "action_error": None,
                "obs_description": None,
                "http_requests": obs["http_requests"],
            }
        )

        if len(self.trajectory) == 1:
            self.trajectory[0]["goal"] = self.goal

        self.log()
        return processed_obs

    def process_actions(self, actions: List[dict]) -> List[dict]:
        tree_nodes = ActionProcessor.convert_action_to_nodes(actions, self.tree_node)
        self.n_actions += len(self.tree_node.children)
        print(f"# {len(tree_nodes)} new actions generated at level {self.tree_node.level}:")
        for node in tree_nodes:
            print(f"- {node.last_action['code']}")
        action_list = ActionProcessor.evaluate_actions(tree_nodes, self.goal, self.checklist)
        final_action_list = ActionProcessor.merge_actions(action_list)

        reward_data = []
        for score, new_node in final_action_list:
            self.n_explored += 1
            reward_data.append(
                [
                    (
                        (new_node.last_action["code"][:100] + "...")
                        if len(new_node.last_action["code"]) > 100
                        else new_node.last_action["code"]
                    ),
                    new_node.frequency,
                    f"{new_node.value:.6f}",
                    f"{score:.6f}",
                ]
            )

        if len(reward_data) > 0:
            print(
                tabulate(
                    reward_data,
                    headers=["Action", "Frequency", "Reward", "Merged"],
                    tablefmt="grid",
                    colalign=("left", "center", "center", "center"),
                )
            )

        return final_action_list

    def _revive_terminating_actions(self):
        if self.revived:
            print("Already revived terminating actions once, not reviving again.")
            return
        curr = self.tree_node
        while curr.parent is not None and not curr.destructive:
            curr = curr.parent
            
        # curr is the root. Now collect all terminating actions in its subtree (which might be pruned)
        def collect_terminating_actions(node):
            actions = []
            if node.last_action is not None and is_terminating(node.last_action):
                actions.append(node)
            for child in node.children:
                actions.extend(collect_terminating_actions(child))
            return actions
        
        terminating_nodes = collect_terminating_actions(curr)
        action_list = ActionProcessor.evaluate_actions(terminating_nodes, self.goal, self.checklist)
        # t_actions = ActionProcessor.merge_actions(action_list)
        
        if self.action_selector is not None and len(action_list) > 0:
            print(f"\033[93mReviving {len(action_list)} terminating actions\033[0m")
            for score, node in action_list:
                print(f"- {node.last_action['code']} (Reward: {score:.6f})")
            self.revived = True
            self.action_selector.add_actions(self.tree_node, action_list, False)
        else:
            print("\033[93mNo terminating actions to revive.\033[0m")
    
    def adjust_tree(self, actions, add_actions=True):
        if add_actions:
            actions = self.process_actions(actions)
            if self.action_selector is not None:
                self.action_selector.add_actions(self.tree_node, actions)
        else:
            actions = []

        # Choose best action based on selection strategy
        if self.action_selector is not None:
            best_node = self.action_selector.select_action(
                self.tree_node, self.n_expanded
            )
            # if self.action_selector.selection_strategy == "action-aware" and best_node.last_action["type"] == "stop" and best_node.last_action["args"]["text"] == "N/A. Agent failed to find a valid solution.":
            #     # print("\033[93mNo valid actions available, reviving terminating actions.\033[0m")
            #     self._revive_terminating_actions()
            #     best_node = self.action_selector.select_action(
            #         self.tree_node, self.n_expanded
            #     )
            # else:
            #     print(best_node.last_action["type"], best_node.last_action["args"])
            return best_node
        else:
            # Default: choose the first action
            if len(actions) == 0:
                raise RuntimeError("No actions to select from.")
            return actions[0][1]

    def generate_new_actions(self, obs: dict, env: gym.Env):
        if len(self.tree_node.children) > 0:
            return []

        actions = RecoveryAssistant.get_recovery_actions(self.tree_node)
        if actions is not None:
            return actions

        if len(self.tree_node.children) == 0 and self.action_generator is not None:
            return self.action_generator.generate_actions(
                _pre_action_trajectory(self.tree_node),
                self.goal,
                self.tree_node.notes,
                env,
                chat_messages=obs["chat_messages"] if self.chat_mode else None,
            )

        return []

    def get_best_action(self, obs: dict, env: gym.Env):
        if self.backtrack_manager is None or not self.backtrack_manager.is_backtracking:
            actions = self.generate_new_actions(obs, env)
            flag = True
            while True:
                best_node = self.adjust_tree(actions, add_actions=flag)
                # self.update_final_answer(best_node, obs)
                if best_node.parent == self.tree_node:
                    self.tree_node = best_node
                    break
                elif self.backtrack_manager is not None and (
                    best_node.last_action["type"] == "stop" or has_safe_anchestor(self.tree_node)
                ):
                    # Backtrack
                    print(
                        f"\033[95m[Backtrack Manager] Backtracking from [{self.tree_node.level},{self.tree_node.position}] "
                        f"to [{best_node.level},{best_node.position}]\033[0m"
                    )
                    # bt_actions, bt_nodes = BacktrackManager.get_backtrack_actions(
                    #     self.tree_node, best_node, env
                    # )
                    flag, n_bt_steps = BacktrackManager.backtrack(self.tree_node, best_node, env)
                    if flag:
                        print(
                            f"\033[92m[Backtrack Manager] Found valid path in the tree for backtracking.\033[0m"
                        )
                        
                        self.trajectory[-1]["action"] = {
                            "code": f"backtrack({best_node.level},{best_node.position})",
                            "thought": "Backtracking action",
                            "type": "backtrack",
                        }
                        self.trajectory[-1]["obs_description"] = ""
                        self.log()
                        self.trajectory.append({
                            "screenshot": best_node.parent.screenshot,
                            "open_pages_urls": best_node.parent.open_pages_urls,
                            "open_pages_titles": best_node.parent.open_pages_titles,
                            "active_page_index": best_node.parent.active_page_index,
                            "url": best_node.parent.url,
                            "axtree_txt": best_node.parent.axtree_txt,
                            "action": None,
                            "action_error": None,
                            "obs_description": None,
                            "http_requests": best_node.parent.http_requests,
                        })
                        
                        self.tree_node = best_node
                        self.log()
                        self.n_backtracks += 1
                        self.n_steps += n_bt_steps
                        # self.backtrack_manager.start_backtracking(bt_actions, bt_nodes)
                        break
                    else:
                        print(
                            f"\033[91m[Backtrack Manager] No valid path found in the tree for backtracking.\033[0m"
                        )
                        flag = False
                        continue
                # else: # If I backtrack, I can't come back
                print("Best node is not a child of the current node, re-adjusting the tree...")
                flag = False

            self.curr_level = self.tree_node.level
            self.curr_pos = self.tree_node.position
            print(f"Expanding Node ({self.curr_level}, {self.curr_pos})")

        # Best node may need backtracking
        if self.backtrack_manager is not None and self.backtrack_manager.is_backtracking:
            action, node = self.backtrack_manager.get_next_backtrack_action()
            # Forward-Execution
            if node is not None and node.last_action is not None:
                return node.last_action, node.last_obs_description
            else:
                return {
                    "code": action,
                    "thought": "Jump action",
                    "type": None,
                }, None

        if self.tree_node.last_action["type"] == "note":
            print(f"Chosen action: {self.tree_node.last_action['code']}")
            self.tree_node.update_from_obs(
                {
                    "axtree_txt": self.tree_node.parent.axtree_txt,
                    "page_too_long": self.tree_node.parent.page_too_long,
                    "axtree_object": self.tree_node.parent.axtree_object,
                    "pruned_html": self.tree_node.parent.pruned_html,
                    "screenshot": self.tree_node.parent.screenshot,
                    "active_page_index": self.tree_node.parent.active_page_index,
                    "open_pages_titles": self.tree_node.parent.open_pages_titles,
                    "open_pages_urls": self.tree_node.parent.open_pages_urls,
                    "last_action_error": self.tree_node.parent.last_action_error,
                }
            )
            # action = self.get_action(obs)
            # for red color?
            print(
                f"# Action {self.n_expanded} ({self.n_steps}): \033[92m{self.tree_node.last_action['code']}\033[0m"
            )
            # self.n_steps += 1
            # self.n_expanded += 1
            return self.get_best_action(obs, env)

        if self.backtrack_manager is not None and self.backtrack_manager.destruction_aware and self.tree_node.last_action["type"] in ("click", "fill", "select_option"): # bid-based actions
            if self.tree_node.parent is not None and self.tree_node.parent.parent is not None:
                curr = self.tree_node.parent
                if curr.url != curr.parent.url:
                    new_axtree_txt, new_axtree_obj = URLSimulator._get_axtree_txt(curr.url, env)
                    from .axtree_utils import is_diff_axtree_obj_by_bid
                    if is_diff_axtree_obj_by_bid(new_axtree_obj, curr.axtree_object, self.tree_node.last_action["args"]["bid"]):
                        logger.debug("Refresh changes the axtree for bid-based action, marking refresh_loses_state=True")
                        self.tree_node.parent.refresh_loses_state = True

        # action = self.tree_node.last_action['code']
        return self.tree_node.last_action, self.tree_node.last_obs_description

    def get_action(self, obs: dict, env: gym.Env) -> tuple[str, dict]:
        self.n_steps += 1
        action, obs_description = self.get_best_action(obs, env)
        if self.backtrack_manager is not None and self.backtrack_manager.is_backtracking:
            self.n_backtracks += 1
        else:
            self.n_expanded += 1

        self.trajectory[-1]["action"] = action
        self.trajectory[-1]["obs_description"] = obs_description

        if self.backtrack_manager is not None and self.backtrack_manager.is_backtracking:
            print(
                f"# Action {self.n_expanded} ({self.n_steps}): \033[91m[B] \033[92m{action['code']}\033[0m"
            )
        else:
            print(f"# Action {self.n_expanded} ({self.n_steps}): \033[92m{action['code']}\033[0m")

        logger.info(f"Selected action at step {self.n_steps}: {action['code']}")

        action = ActionProcessor.postprocess(action, obs["axtree_object"])

        self.log()
        
        logger.debug(f"Executing action code: {action['code']}")
        return action["code"], {}


@dataclasses.dataclass
class TreeSearchAgentArgs(AbstractAgentArgs):
    """
    This class is meant to store the arguments that define the agent.
    By isolating them in a dataclass, this ensures serialization without storing
    internal states of the agent.
    """

    chat_mode: bool = False
    action_generator: Optional[ActionGenerator] = None
    action_selector: Optional[ActionSelector] = None
    backtrack_manager: Optional[BacktrackManager] = None

    def make_agent(self, exp_dir: str = "", checkpoint: str = None):
        return TreeSearchAgent(
            chat_mode=self.chat_mode,
            action_generator=self.action_generator,
            action_selector=self.action_selector,
            backtrack_manager=self.backtrack_manager,
            exp_dir=exp_dir,
            checkpoint=checkpoint,
        )
