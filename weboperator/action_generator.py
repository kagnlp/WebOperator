import browsergym.core.action.functions as funcs
import inspect
import ast
from typing import List, Dict, Any, Optional
import re
import json
import logging
from .models.base import BaseModel
from .task_rephraser import TaskRephraser
import os
import time
from .action_validator import ActionValidator
from .experience_retriever import ExperienceRetriever
from .prompt_designer import PromptDesigner
import gymnasium as gym

logger = logging.getLogger(__name__)


def get_arg_dict_from_call(call_str):
    # Parse function call
    expr = ast.parse(call_str, mode="eval")
    call_node = expr.body

    # Get function name
    if isinstance(call_node.func, ast.Name):
        func_name = call_node.func.id
    else:
        raise ValueError("Unsupported function call format")

    # Get function from imported module
    func = getattr(funcs, func_name)

    # Get function signature
    sig = inspect.signature(func)
    param_names = list(sig.parameters.keys())

    # Evaluate argument values
    args = [ast.literal_eval(arg) for arg in call_node.args]
    kwargs = {kw.arg: ast.literal_eval(kw.value) for kw in call_node.keywords}

    # Build full argument dict
    arg_dict = {}
    for i, arg in enumerate(args):
        arg_dict[param_names[i]] = arg
    arg_dict.update(kwargs)

    return func_name, arg_dict


def action_to_python_code(action: dict) -> str:
    # Arg dict order is not guaranteed
    return f"{action['type']}({', '.join([f'{repr(v)}' for k, v in action['args'].items()])})"


from .axtree_utils import get_elem_by_bid


def extract_function_calls(code_str):
    allowed_functions = [
        "scroll",
        "click",
        "fill",
        "stop",
        "goto",
        "go_back",
        "go_forward",
        "tab_focus",
        "new_tab",
        "tab_close",
        "note",
    ]
    # Lookahead to split just before function name
    pattern = r"(?=" + "|".join(re.escape(f) + r"\(" for f in allowed_functions) + r")"
    parts = re.split(pattern, code_str)
    parts = [p.strip() for p in parts if p.strip()]
    function_calls = []

    for part in parts:
        pattern = r"([a-zA-Z_]\w*\(.*\))"
        matches = re.findall(pattern, part, re.DOTALL)
        if matches:
            function_calls.append(matches[0])
    return function_calls


class ActionGenerator:
    """Strategy pattern for action generation"""

    action_space_type: str = "adaptive"  # "fixed" or "adaptive"
    full_action_space: List[str] = [
        "click",
        "select_option",
        "fill",
        "goto",
        "go_back",
        "scroll",
        "new_tab",
        "tab_focus",
        "tab_close",
        "stop",
    ]
    max_retry: int = 5
    allow_invalid_action: bool = False

    @classmethod
    def configure(
        cls,
        full_action_space,
        action_space_type: str = "adaptive",
        max_retry: int = 5,
        allow_invalid_action: bool = False,
    ):
        cls.full_action_space = full_action_space
        cls.action_space_type = action_space_type
        cls.max_retry = max_retry
        cls.allow_invalid_action = allow_invalid_action

    def __init__(
        self,
        model: BaseModel,
        name: str = "",
        history_length: int = 5,
        rephraser_enabled: bool = False,
        retriever_enabled: bool = False,
    ):
        """
        Initialize TopK Action Generator

        Args:
            model: BaseModel instance for action generation
            k: Number of top actions to generate
        """
        self.model = model
        if name == "":
            name = f"action_generator_{int(time.time())}"
        self.name = name
        self.action_space = None

        self.rephraser_enabled = rephraser_enabled
        self.retriever_enabled = retriever_enabled

        self.goal = None
        self.rephrased_goal = None

        self.history = []
        self.history_length = history_length

    def generation_count(self):
        n_attempts = 0
        for entry in self.history:
            n_attempts += len(entry["failed_attempts"]) + (1 if entry["output"] else 0)
        return n_attempts

    def parse_action(self, action_str: str, reason_str: str, trajectory) -> Dict[str, Any]:
        """Parse the action string into a structured format"""
        func_name, arg_dict = None, {}
        current_obs = trajectory[-1] if trajectory else {}

        # Auto Correct
        action_str = re.sub(r"(fill\([^)]*?),\s*true(\))", r"\1, True\2", action_str)
        action_str = re.sub(r"(fill\([^)]*?),\s*false(\))", r"\1, False\2", action_str)

        # There might be multiple function calls
        function_calls = extract_function_calls(action_str)

        if len(function_calls) > 0:
            try:
                action_str = function_calls[0]
                func_name, arg_dict = get_arg_dict_from_call(action_str)
            except Exception as e:
                raise ValueError(
                    f"Error extracting function call from action string '{action_str}': {e}"
                )
        else:
            raise ValueError(
                f"No valid function call found in action string '{action_str}'. Use an action from action space in proper format."
            )

        selected_elem = {}
        if func_name == "click" or func_name == "fill" or func_name == "select_option":
            bid = arg_dict.get("bid", None)

            if bid is not None:
                elem = get_elem_by_bid(current_obs["axtree_object"], bid)
                if elem is not None:  # Add this check
                    selected_elem = elem
                else:
                    raise ValueError(f"bid '{bid}' not found in Current page Accessibility Tree. Only use bid which are present in Current page Accessibility Tree.")

        action_str = action_to_python_code({"type": func_name, "args": arg_dict})

        formatted_action = {
            "code": action_str,
            "thought": reason_str,
            "type": func_name,
            "args": arg_dict,
            "selected_element": selected_elem,
        }
        return formatted_action

    def process_response(self, response: str) -> str:
        """Post-process generated action if needed"""

        pattern = r"#\s*Observation Description\s*\n(.*?)(?=\n#\s*Reason|\Z)"
        matches = re.findall(pattern, response, re.DOTALL | re.IGNORECASE)
        if not matches:
            raise ValueError(
                "'# Observation Description' not found. Strictly follow the given output format."
            )
        obs_str = matches[0].strip()
        # print(f">> Observation: {obs_str}")

        pattern = r"#\s*Reason\s*\n(.*?)(?=\n#\s*Action|\Z)"
        matches = re.findall(pattern, response, re.DOTALL | re.IGNORECASE)
        if not matches:
            raise ValueError("'# Reason' not found. Strictly follow the given output format.")
        reason_str = matches[0].strip()
        # print(f">> Reason: {reason_str}")

        pattern = r"#\s*Action\s*\n(.*?)\Z"
        # pattern = r'#\s*Action\s*\n(.*)'
        matches = re.findall(pattern, response, re.DOTALL)
        if not matches:
            raise ValueError("'# Action' not found. Strictly follow the given output format.")
        action_str = matches[0].strip()

        return obs_str, reason_str, action_str

    def autocorrect_action(self, action):
        if action["type"] == "fill":
            for prop in action["selected_element"]["properties"]:
                if prop["name"] == "multiline" and prop["value"]["value"] == True:
                    action["args"]["press_enter_after"] = False
                    action["code"] = action_to_python_code(action)
        return action

    def _build_trajectory_history(self, trajectory: List[dict]) -> str:
        """Build a text summary of the trajectory history"""
        if len(trajectory) <= 1:
            return "No previous actions taken."

        history_parts = []

        # Show last few steps to avoid token overflow
        recent_trajectory = (
            trajectory[-self.history_length : -1]
            if len(trajectory) > self.history_length
            else trajectory[:-1]
        )

        for i, step in enumerate(recent_trajectory):
            step_num = len(trajectory) - len(recent_trajectory) + i

            # Extract action if available
            action = step.get("action", "No action")
            action_error = step.get("action_error")

            # Get current URL from open_pages_urls and active_page_index
            urls = step.get("open_pages_urls", [])
            active_idx = step.get("active_page_index", 0)
            url = urls[active_idx] if urls and active_idx < len(urls) else "Unknown URL"

            # Get a brief description of the observation
            history_parts.append(
                f"Observation #{step_num}: (Url: {url}) \n{step.get('obs_description', 'No observation available')}"
            )
            history_parts.append(f"Reason #{step_num}: {action['thought']}")
            history_parts.append(f"Action #{step_num}: {action['code']}\n")
            if action_error:
                history_parts.append(f"Action Error #{step_num}: {action_error}\n")

        return "\n".join(history_parts)

    def _summarize_observation(self, observation: dict) -> str:
        """Create a brief summary of an observation"""
        # Get current URL from open_pages_urls and active_page_index
        urls = observation.get("open_pages_urls", [])
        active_idx = observation.get("active_page_index", 0)
        url = urls[active_idx] if urls and active_idx < len(urls) else "Unknown URL"

        # Get page titles
        titles = observation.get("open_pages_titles", [])
        title = titles[active_idx] if titles and active_idx < len(titles) else "Unknown Title"

        axtree = observation.get("axtree_txt", "")

        # Extract key elements from axtree for summary
        if axtree:
            # Look for interactive elements
            interactive_elements = []
            lines = axtree.split("\n")[:10]  # First 10 lines
            for line in lines:
                if any(elem in line.lower() for elem in ["button", "link", "input", "textbox"]):
                    interactive_elements.append(line.strip()[:50])

            if interactive_elements:
                return (
                    f"Page '{title}' at {url} with elements: {', '.join(interactive_elements[:3])}"
                )

        return f"Page '{title}' at {url}"

    def log(self, exp_dir: str):
        """Log the action generator's configuration and results."""
        if self.name is None:
            return

        log_data = {
            "goal": self.goal,
            "rephrased_goal": self.rephrased_goal,
            # "action_space": self.action_space,
            "history": self.history,
        }

        with open(f"{exp_dir}/{self.name}_log.json", "w", encoding="utf-8") as f:
            json.dump(log_data, f, indent=4, ensure_ascii=False)

    def set_goal(self, goal: str):
        """Set the goal for the action generator"""
        self.goal = goal

        self.history = []

    def _get_current_url(self, trajectory: List[Dict[str, Any]]) -> str:
        """Get the current URL from the trajectory."""
        if not trajectory:
            return "Unknown URL"

        last_step = trajectory[-1]
        urls = last_step.get("open_pages_urls", [])
        active_idx = last_step.get("active_page_index", 0)
        return urls[active_idx] if urls and active_idx < len(urls) else "Unknown URL"

    def can_go_back(self, trajectory: List[Dict[str, Any]]) -> bool:
        """Check if the agent can go back in history."""
        if len(trajectory) < 2:
            return False

        last_url = trajectory[0]["open_pages_urls"][trajectory[0]["active_page_index"]]
        for step in trajectory:
            curr_url = step["open_pages_urls"][step["active_page_index"]]
            if curr_url != last_url:
                return True

            last_url = curr_url
        return False

    def can_go_forward(self, trajectory: List[Dict[str, Any]]) -> bool:
        """Check if the agent can go forward in history."""
        if len(trajectory) < 3:
            return False
        count = 0
        for step in trajectory[:-1]:
            if step["action"]["type"] == "go_back":
                count += 1
            elif step["action"]["type"] != "go_forward":
                count -= 1
        return count > 0

    def generate_actions(
        self,
        trajectory: List[dict],
        goal: str,
        notes: List[str],
        env: gym.Env,
        chat_messages: Optional[List[dict]] = None,
    ) -> List[str]:
        """
        Generate top-k actions based on the full trajectory and goal

        Args:
            trajectory: Full trajectory including observations and actions
            goal: The goal to accomplish
        """

        current_obs = trajectory[-1] if trajectory else {}

        if goal != self.goal:
            self.set_goal(goal)
            if self.rephraser_enabled:
                self.rephrased_goal = TaskRephraser.rephrase(
                    goal, current_obs.get("url", ""), current_obs.get("axtree_txt", "")
                )

        if self.action_space_type == "fixed":
            self.action_space = self.full_action_space
        else:
            # self.action_space = [
            #     "click",
            #     "select_option" if "option" in current_obs.get("axtree_txt", "") else None,
            #     "fill",
            #     "goto",
            #     "go_back" if self.can_go_back(trajectory) else None,
            #     # "go_forward" if len(trajectory) > 1 and trajectory[-2]["action"]["type"] == "go_back" else None,
            #     # "go_forward" if self.can_go_forward(trajectory) else None,
            #     "scroll" if current_obs["page_too_long"] else None,
            #     "new_tab",
            #     "tab_focus" if len(current_obs["open_pages_urls"]) > 1 else None,
            #     "tab_close" if len(current_obs["open_pages_urls"]) > 1 else None,
            #     "note",
            #     "stop"
            # ]
            self.action_space = []
            for action in self.full_action_space:
                if action == "go_back":
                    if self.can_go_back(trajectory):
                        self.action_space.append(action)
                elif action == "go_forward":
                    if self.can_go_forward(trajectory):
                        self.action_space.append(action)
                elif action == "scroll":
                    if current_obs.get("page_too_long", False):
                        self.action_space.append(action)
                elif action == "select_option":
                    if "option" in current_obs.get("axtree_txt", ""):
                        self.action_space.append(action)
                elif action == "tab_focus" or action == "tab_close":
                    if len(current_obs.get("open_pages_urls", [])) > 1:
                        self.action_space.append(action)
                else:
                    self.action_space.append(action)

        examples = None
        if self.retriever_enabled:
            examples = self._get_examples(current_obs, goal)
        last_error = None
        last_response = None
        last_responses = []
        last_errors = []
        error_log = []
        valid_format = 0
        attempt = 0
        sleep_time = 3
        last_invalid = False
        last_valid_response = None

        while True:  # Retry up to max_retry times
            if self.max_retry != -1 and (valid_format >= self.max_retry or (attempt >= 2 * self.max_retry and valid_format == 0)):
                break
            try:
                while True:
                    try:
                        SYSTEM_PROMPT, USER_PROMPT = PromptDesigner.design_actor_prompt(
                            goal=goal,
                            rephrased_goal=self.rephrased_goal,
                            trajectory=trajectory,
                            history_length=self.history_length,
                            action_space=self.action_space,
                            notes=notes,
                            examples=examples,
                            prev_responses=last_responses,
                            prev_errors=last_errors,
                            chat_messages=chat_messages,
                        )
                        # print(f"\n\033[95m[System Prompt]\033[0m {SYSTEM_PROMPT}\n")
                        # print(f"\033[96m[User Prompt]\033[0m {USER_PROMPT}\n")
                        response, _ = self.model.chat(
                            [
                                {"role": "system", "content": SYSTEM_PROMPT},
                                {"role": "user", "content": USER_PROMPT},
                            ]
                        )
                        break
                    except Exception as e:
                        print(f"Error during model.chat: {e}")
                        continue

                # # Handle both single response and multiple responses
                # print(f"# Generated actions: ")
                # for i, action in enumerate(actions, 1):
                #     print(f"Action {i}: {action}\n")
                last_response = response
                try:
                    obs_str, reason_str, action_str = self.process_response(response)
                except Exception as e:
                    print(f"#{attempt}.{valid_format} Error occurred while generating actions: {e}")
                    last_error = e
                    valid_format = 0
                    error_log.append({"response": last_response, "error": str(last_error)})
                    if last_invalid:
                        last_responses[-1] = last_response
                        last_errors[-1] = last_error
                    else:
                        last_responses.append(last_response)
                        last_errors.append(last_error)
                        last_invalid = True
                    time.sleep(sleep_time)  # Brief pause before retrying
                    sleep_time = min(sleep_time * 2, 200)  # Exponential backoff up to 60 seconds
                    attempt += 1
                    continue
                    
                valid_format += 1 
                print(f"\033[94m[Action Generated]\033[0m {action_str}")
                formatted_action = self.parse_action(action_str, reason_str, trajectory)
                last_valid_response = (formatted_action, obs_str)
                ActionValidator.validate(formatted_action, trajectory, self.action_space, env)
                formatted_action = self.autocorrect_action(formatted_action)
                # Add green color
                print(f"\033[92m[Action Validated]\033[0m {formatted_action['code']}")
                self.history.append(
                    {
                        "action_space": [
                            action for action in self.action_space if action is not None
                        ],
                        "input": {
                            "current_url": self._get_current_url(trajectory),
                            "trajectory": self._build_trajectory_history(trajectory),
                            "text_observation": current_obs.get(
                                "axtree_txt", "No page content available"
                            ),
                            "examples": examples,
                        },
                        "failed_attempts": error_log,
                        "output": response,
                        "thought": formatted_action["thought"],
                        "action": formatted_action["code"],
                    }
                )

                return [(formatted_action, obs_str)]

            except Exception as e:
                print(f"#{attempt}.{valid_format} Error occurred while generating actions: {e}")
                last_error = e
                if last_invalid:
                    last_responses[-1] = last_response
                    last_errors[-1] = last_error
                    last_invalid = False
                else:
                    last_responses.append(last_response)
                    last_errors.append(last_error)

                error_log.append({"response": last_response, "error": str(last_error)})
                time.sleep(sleep_time)  # Brief pause before retrying
                sleep_time = min(sleep_time * 2, 60)  # Exponential backoff up to 60 seconds
                # return []

            attempt += 1

        self.history.append(
            {
                "action_space": [action for action in self.action_space if action is not None],
                "input": {
                    "current_url": self._get_current_url(trajectory),
                    "trajectory": self._build_trajectory_history(trajectory),
                    "text_observation": current_obs.get("axtree_txt", "No page content available"),
                    "examples": examples,
                },
                "failed_attempts": error_log,
                "output": None,
                "thought": None,
                "action": None,
            }
        )

        if self.allow_invalid_action and last_valid_response is not None:
            print(
                f"\033[93m[Warning]\033[0m Maximum retries reached. Returning last syntactically valid action despite semantic validation failure: {last_valid_response[0]['code']}"
            )
            return [last_valid_response]

        raise ValueError(f"Failed to generate actions after multiple attempts.")

    def _get_examples(self, obs: dict, goal: str) -> List[dict]:
        """Get relevant examples using retrieval"""
        logger.debug(f"Retrieving examples for goal: {goal}")

        try:
            # Build query from observation and goal
            query_parts = []

            # Add goal
            if goal:
                query_parts.append(f">> Goal: {goal}")

            # Get current URL from open_pages_urls and active_page_index
            urls = obs.get("open_pages_urls", [])
            active_idx = obs.get("active_page_index", 0)
            url = urls[active_idx] if urls and active_idx < len(urls) else ""
            if url:
                query_parts.append(f">> Current URL: {url}")

            # Add page content (truncated)
            axtree = obs.get("axtree_txt", "")
            if axtree:
                # Truncate axtree to avoid too long queries that might cause issues
                truncated_axtree = axtree[:1000] if len(axtree) > 1000 else axtree
                query_parts.append(f">> Current page Accessibility Tree: {truncated_axtree}")

            query = "\n".join(query_parts)
            if not query.strip():
                return []

            logger.debug(f"Built query with {len(query)} characters")

            # Search for relevant examples with error handling
            try:
                results = ExperienceRetriever.get_examples(goal, obs=axtree)
                logger.debug(f"Retrieved {len(results) if results else 0} results")
            except ZeroDivisionError as e:
                logger.error(f"Division by zero in smart search: {e}")
                return []
            except Exception as e:
                logger.error(f"Smart search failed: {e}")
                return []

            examples = []

            # print("-------" , results)
            for i, result in enumerate(results):
                try:
                    # print(f"Processing result {i+1}: {type(result)}")

                    # Handle different result formats
                    if isinstance(result, tuple) or isinstance(result, list):
                        metadata = result[0]
                        score = result[1] if len(result) > 1 else 0.0
                        source = result[2] if len(result) > 2 else "unknown"
                    elif isinstance(result, dict):
                        metadata = result.get("metadata", result)
                        score = result.get("score", 0.0)
                        source = result.get("source", "unknown")
                    else:
                        print(f"Skipping result {i+1}: unexpected format {type(result)}")
                        continue

                    step_data = metadata["metadata"]["step_data"]

                    # Extract action from metadata
                    example = {
                        "goal": step_data["goal"],
                        "thought": step_data["thought"],
                        "action": step_data["action"],
                        "score": score,
                        "source": source,
                    }
                    # print(f"Example: {example}")
                    examples.append(example)
                    # print(f"Added example from {source} with score {score}")

                except Exception as e:
                    logger.warning(f"Error processing result {i+1}: {e}")
                    continue

            logger.debug(f"Successfully processed {len(examples)} examples")
            return examples

        except Exception as e:
            logger.error(f"Retrieval error: {e}")
            # import traceback
            # logger.error(f"Traceback: {traceback.format_exc()}")
            return []


class EnsembleActionGenerator(ActionGenerator):
    """Ensemble of multiple action generators using position-based ranking"""

    def __init__(self, generators: List[ActionGenerator]):
        """
        Initialize Ensemble Action Generator

        Args:
            generators: List of ActionGenerator instances to combine
            k: Number of final actions to return
        """
        if not generators:
            raise ValueError("At least one generator must be provided")

        self.generators = generators

    def log(self, exp_dir: str):
        for generator in self.generators:
            generator.log(exp_dir)

    def generation_count(self):
        return sum(generator.generation_count() for generator in self.generators)

    def generate_actions(
        self,
        trajectory: List[dict],
        goal: str,
        notes: List[str],
        env: gym.Env,
        chat_messages: Optional[List[dict]] = None,
    ) -> List[str]:
        """Generate actions using ensemble ranking by position"""

        # Collect actions from all generators
        generator_results = []

        for i, generator in enumerate(self.generators):
            try:
                actions = generator.generate_actions(trajectory, goal, notes, env, chat_messages)
                if actions:  # Only add if generator produced actions
                    generator_results.extend(actions)
            except Exception as e:
                logger.warning(f"Generator {i+1} failed: {e}")
                import traceback

                print(traceback.format_exc())
                continue

        if not generator_results:
            logger.error("All generators failed")
            return []

        else:
            logger.debug(
                f"Collected {len(generator_results)} actions from {len(self.generators)} generators"
            )

            logger.debug("\n".join("- " + action[0]["code"] for action in generator_results))

        # Rank actions by position-weighted scores
        return generator_results
