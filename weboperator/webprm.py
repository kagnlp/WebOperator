from webshepherd.webprm.prompts import get_messages
from typing import List, Dict, Any, Optional
from .models.base import BaseModel
import json
import math
from typing import Tuple
from pathlib import Path

SYSTEM_PROMPT_TEMPLATE = """
You are an expert evaluator of web agent. Your task is to assess how helpful a given agent's THOUGHT and ACTION is in making progress toward the user's goal, based on the current state of the webpage.

# Task Description
Evaluate how well the agent’s THOUGHT and ACTION satisfy each item in the checklist using the task instruction, trajectory (including previously completed steps), current webpage state, the agent’s latest response and checklist completion after (n-1)th step. Start by writing a concise paragraph summarizing the agent’s overall performance. Refer to the reasoning provided in the trajectory, and discuss whether the THOUGHT is appropriate and the ACTION moves the task forward.
Then, assess each checklist item individually using the following labels:
- Yes: The item is fully and clearly satisfied, either in the current response or previously completed.
- In Progress: There is meaningful partial progress toward completing the item.
- No: The item is not satisfied due to ambiguity, insufficient evidence, or lack of progress.
"""

WO_USER_PROMPT_TEMPLATE = """
>> User Instruction
{intent}

>> Trajectory
{trajectory}

>> Current URL
{current_url}

>> Current page Accessibility Tree:
{text_observation}

>> Checklist
{checklist}
>> Last Step Checklist Completion
{progress}

>> Agent's Response
- THOUGHT: {thought}
- ACTION: {action}

>> Generate the response in the following format:

# Explanation
Provide a concise paragraph summarizing the agent’s overall performance. Refer to the reasoning provided in the trajectory, and discuss whether the THOUGHT is appropriate and the ACTION moves the task forward.
# Checklist Evaluation
For each checklist item, provide your assessment using the labels: Yes, In Progress, or No.

Example Response:
# Explanation
The agent made significant progress toward the goal by effectively utilizing the webpage's features. The THOUGHT was well-reasoned, and the ACTION taken was appropriate for the current context, leading to a successful outcome.
# Checklist Evaluation
Checklist 1: Yes
Checklist 2: In Progress
Checklist 3: No
"""

WS_USER_PROMPT_TEMPLATE = """
>> User Instruction
{intent}

>> Trajectory
{trajectory}

>> Current URL
{current_url}

>> Current page Accessibility Tree:
{text_observation}

>> Checklist
{checklist}

>> Agent's Response
- THOUGHT: {thought}
- ACTION: {action}

>> Generate the response in the following format:

# Explanation
Provide a concise paragraph summarizing the agent’s overall performance. Refer to the reasoning provided in the trajectory, and discuss whether the THOUGHT is appropriate and the ACTION moves the task forward.
# Checklist Evaluation
For each checklist item, provide your assessment using the labels: Yes, In Progress, or No.

Example Response:
# Explanation
The agent made significant progress toward the goal by effectively utilizing the webpage's features. The THOUGHT was well-reasoned, and the ACTION taken was appropriate for the current context, leading to a successful outcome.
# Checklist Evaluation
Checklist 1: Yes
Checklist 2: In Progress
Checklist 3: No
"""

target_judge = {
    "yes": [
        " Yes",
        "ĠYes",
        "Yes",
        "ĊYes",
        "Ġyes",
        "yes",
        "Ċyes",
        "ĠYES",
        "YES",
        "ĊYES",
        "ĠDone",
        "Done",
        "ĊDone",
        "ĠCompleted",
        "Completed",
        "ĊCompleted",
        "ĠCorrect",
        "Correct",
        "ĊCorrect",
    ],
    "no": [
        " No",
        "ĠNo",
        "No",
        "ĊNo",
        "ĠNO",
        "NO",
        "ĊNO",
        "ĠNot",
        "Not",
        "ĊNot",
        "ĠNone",
        "None",
        "ĊNone",
        "ĠNope",
        "Nope",
        "ĊNope",
        "ĠUn",
        "Un",
        "ĊUn",
        "ĠWrong",
        "Wrong",
        "ĊWrong",
    ],
    "in": [
        " In",
        "ĠIn",
        "In",
        "ĊIn",
        "ĠPending",
        "Pending",
        "ĊPending",
        "ĠPart",
        "Part",
        "ĊPart",
        "ĠPartial",
        "Partial",
        "ĊPartial",
        "ĠInProgress",
        "InProgress",
        "ĊInProgress",
    ],
}


class WebPRM:
    reward_model: BaseModel = None
    checklist_model: BaseModel = None
    prompt_type: str = "web_shepherd"
    checklist_cache: str = None
    history = []

    @classmethod
    def configure(
        cls,
        reward_model: Optional[BaseModel] = None,
        checklist_model: Optional[BaseModel] = None,
        prompt_type: str = "web_operator",
    ):
        cls.checklist_model = checklist_model
        cls.prompt_type = prompt_type
        # cls.checklist_cache = checklist_cache
        cls.reward_model = reward_model

    @staticmethod
    def _get_current_observation(trajectory: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Get the current observation from the trajectory."""
        if not trajectory:
            return {}
        return trajectory[-1]

    @staticmethod
    def _get_current_url(trajectory: List[Dict[str, Any]]) -> str:
        """Get the current URL from the trajectory."""
        if not trajectory:
            return "Unknown URL"

        last_step = trajectory[-1]
        urls = last_step.get("open_pages_urls", [])
        active_idx = last_step.get("active_page_index", 0)
        return urls[active_idx] if urls and active_idx < len(urls) else "Unknown URL"

    @staticmethod
    def _get_brief_context(trajectory: List[Dict[str, Any]]) -> str:
        """Get brief context from trajectory[-10:-2]."""
        if len(trajectory) < 2:
            return "No previous actions."

        # Show trajectory[-4:-2] for context (2 steps before current)
        context_steps = trajectory[-10:-1] if len(trajectory) >= 10 else trajectory[:-1]
        # print(f"Context steps for evaluation: {context_steps}")
        if not context_steps:
            return "No previous actions."

        context_parts = []

        for i, step in enumerate(context_steps):
            obs_description = step.get("obs_description", "")
            action = step.get("action", "No action")
            error = step.get("action_error", "")
            step_num = len(trajectory) - len(context_steps) + i
            context_line = f"""
Thought {step_num}: {action["thought"]}
Action {step_num}: {action["code"]}
"""
            # if error and error != "No Error (This action is not yet executed)":
            #     context_line += f" [Error: {error}]"
            context_parts.append(context_line)

        return "\n".join(context_parts)

    @staticmethod
    def add_to_ori_logprob(ori_logprob: float, add_logprob: float):
        if ori_logprob is None:
            return add_logprob
        else:
            ori_prob = math.exp(ori_logprob)
            add_prob = math.exp(add_logprob)
            return math.log(ori_prob + add_prob)

    @classmethod
    def log(cls, exp_dir):
        """Log the judge's configuration and generated checklist."""
        if len(cls.history) == 0:
            return
        # log_data = {
        #     "goal": self.goal,
        #     "generated_checklist": self.generated_checklist,
        #     "history": self.history
        # }
        with open(f"{exp_dir}/webprm_log.json", "w", encoding="utf-8") as f:
            json.dump(cls.history, f, indent=4, ensure_ascii=False)

    @classmethod
    def generate_checklist(cls, goal: str, start_url: str, start_obs: str):
        # first check if the checklist of this goal is already exist in webarena_checklist.json
        # Check if file exists, if not create an empty list in self.checklist_cache file
        # Check the path exists
        if cls.checklist_model is None:
            return ""

        current_dir = Path(__file__).resolve().parent  # directory of current file
        cls.checklist_cache = (
            current_dir
            / f"checklists/{cls.checklist_model.name.split(':')[0].split('/')[-1]}.jsonl"
        )

        cls.checklist_cache.parent.mkdir(parents=True, exist_ok=True)

        if not cls.checklist_cache.exists():
            cls.checklist_cache.write_text("", encoding="utf-8")
            print(f"Created new checklist cache file at {cls.checklist_cache}")
        else:
            print(f"Using existing checklist cache file at {cls.checklist_cache}")
        # ✅ Read line by line to check if checklist already exists
        existing_checklist = None
        with open(cls.checklist_cache, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    item = json.loads(line)
                    if item["intent"] == goal and item["start_url"] == start_url:
                        existing_checklist = item["checklist"]
                        break
                except json.JSONDecodeError:
                    print("Malformed line in checklist cache, skipping...")
                    continue  # skip malformed lines if any

        if existing_checklist:
            print("Loaded existing checklist from cache.")
            print(existing_checklist)
            cls.history = []
            return existing_checklist

        input_info = {"intent": goal, "start_url": start_url, "text_observation": start_obs}

        message = get_messages(
            input_info=input_info, inference_mode="checklist_generation", prompt_type="web_shepherd"
        )

        for _ in range(3):  # Try up to 3 times to get a valid answer
            try:
                response, _ = cls.checklist_model.chat(message)
                break
            except Exception as e:
                print(f"Error during checklist generation: {e}. Retrying...")
                continue

        if "[CHECKLISTS]" in response:
            generated_checklist = response.split("[CHECKLISTS]")[-1].strip()
        elif "<answer>" in response:
            generated_checklist = response.split("<answer>")[-1].split("</answer>")[0].strip()
        else:
            generated_checklist = response

        # self.goal = goal
        cls.history = []

        print("Generated Checklist: \n", generated_checklist)

        # ✅ Append to JSONL file
        with open(cls.checklist_cache, "a", encoding="utf-8") as f:
            json.dump(
                {"intent": goal, "start_url": start_url, "checklist": generated_checklist},
                f,
                ensure_ascii=False,
            )
            f.write("\n")

        return generated_checklist

    @staticmethod
    def get_judge_probs(scores, reward_model_type):
        response_str = ""
        judge_probs_list = []
        checklist_logs = []
        for record in scores:
            token_text = record.token

            if (reward_model_type == "web_shepherd" and "<answer>" in response_str) or (
                reward_model_type != "web_shepherd" and "# Checklist Evaluation"
            ):
                # print(token_text)
                # print("Start to find checklist evaluation...")
                find_judge_str = None
                for judge_type in target_judge:
                    if token_text in target_judge[judge_type]:
                        find_judge_str = judge_type
                        break
                if find_judge_str:
                    token_judge_dict = {"yes": None, "no": None, "in": None}

                    for lt in record.top_logprobs:
                        alt_text = lt.token
                        log_prob = lt.logprob
                        for judge_type in target_judge:
                            for judge_str in target_judge[judge_type]:
                                if judge_str in alt_text:
                                    token_judge_dict[judge_type] = WebPRM.add_to_ori_logprob(
                                        token_judge_dict[judge_type], log_prob
                                    )

                    checklist_logs.append(
                        {
                            "token": token_text,
                            "top_logprobs": [
                                {"token": lt.token, "logprob": lt.logprob}
                                for lt in record.top_logprobs
                            ],
                        }
                    )

                    for judge_type in token_judge_dict:
                        if token_judge_dict[judge_type] is None:
                            token_judge_dict[judge_type] = float("-inf")
                    judge_probs_list.append(token_judge_dict)

            response_str += token_text

            if reward_model_type == "web_shepherd" and "</answer>" in response_str:
                break

        if len(judge_probs_list) == 0:
            return [{"yes": 0.0, "no": 0.0, "in": 0.0}], checklist_logs
        else:
            # convert with softmax
            final_judge_probs_list = []
            for judge_probs in judge_probs_list:
                exp_logprobs = [
                    math.exp(x) for x in [judge_probs["yes"], judge_probs["no"], judge_probs["in"]]
                ]
                sum_exp_logprobs = sum(exp_logprobs)
                if sum_exp_logprobs == 0:
                    softmax_probs = [0 for _ in exp_logprobs]
                else:
                    softmax_probs = [x / sum_exp_logprobs for x in exp_logprobs]
                final_judge_probs_list.append(
                    {"yes": softmax_probs[0], "no": softmax_probs[1], "in": softmax_probs[2]}
                )
            return final_judge_probs_list, checklist_logs

    # def get_checklist(self, trajectory, goal):
    #     checklist_not_generated = not self.generated_checklist
    #     checklist_outdated = self.goal is not None and self.goal != goal
    #     if checklist_not_generated or checklist_outdated:
    #         self.generate_checklist(goal, self._get_current_url(
    #             trajectory), self._get_current_observation(trajectory))
    #     return self.generated_checklist if self.generated_checklist else ""

    # def update_checklist(self, goal, checklist) -> None:
    #     self.goal = goal
    #     self.generated_checklist = checklist

    @classmethod
    def evaluate(
        cls, trajectory: List[Dict[str, Any]], goal: str, checklist: str, **kwargs
    ) -> Tuple[float, str]:
        if cls.reward_model is None:
            return 0.0, ""

        current_obs = cls._get_current_observation(trajectory)
        proposed_action = current_obs.get("action", "No action")
        progress = trajectory[-2].get("checklist_completion") if len(trajectory) >= 2 else ""
        progress = progress if progress != "" else "No previous step"

        if cls.prompt_type == "web_shepherd":
            data = {
                "intent": goal,
                "trajectory": cls._get_brief_context(trajectory),
                "text_observation": current_obs.get("axtree_txt", "No page content available"),
                "current_url": cls._get_current_url(trajectory),
                "action": proposed_action["code"],
                "thought": proposed_action["thought"],
                "checklist": checklist if checklist else "No checklist generated",
            }
        else:
            data = {
                "intent": goal,
                "trajectory": cls._get_brief_context(trajectory),
                "text_observation": current_obs.get("axtree_txt", "No page content available"),
                "current_url": cls._get_current_url(trajectory),
                "action": proposed_action["code"],
                "thought": proposed_action["thought"],
                "checklist": checklist if checklist else "No checklist generated",
                "progress": progress,  # TEMP
            }

        reward_model_type = "web_shepherd" if "WebShepherd" in cls.reward_model.name else "others"

        if reward_model_type == "web_shepherd":
            message = get_messages(
                input_info=data,
                inference_mode="judge_progress",
                prompt_type=cls.prompt_type,
                use_multimodal=False,
                text_obs=True,
                image_obs=False,
            )
        else:
            message = [
                {"role": "system", "content": SYSTEM_PROMPT_TEMPLATE},
                {
                    "role": "user",
                    "content": (
                        WS_USER_PROMPT_TEMPLATE.format(**data)
                        if cls.prompt_type == "web_shepherd"
                        else WO_USER_PROMPT_TEMPLATE.format(**data)
                    ),
                },
            ]

        for _ in range(3):  # Try up to 3 times to get a valid answer
            try:
                response, scores = cls.reward_model.chat(message)
            except Exception as e:
                print(f"Error during evaluation: {e}. Retrying...")
                continue
            generated_text = response

            if (
                reward_model_type == "web_shepherd"
                and "<answer>" in generated_text
                and "</answer>" in generated_text
            ):
                break

            if reward_model_type != "web_shepherd" and "# Checklist Evaluation" in generated_text:
                break

        # print("RESPONSE: ", generated_text)

        # print("========= LOG PROBS =========")
        # check if <answer> and </answer> exist in the generated_text

        judge_probs, judge_scores = cls.get_judge_probs(scores, reward_model_type)
        # For each checkpoint, need weighted average
        # print(judge_probs)
        total_score = 0.0
        total_weight = 0.0

        for i, judge_prob in enumerate(judge_probs):
            score = (
                judge_prob.get("yes", 0) * 1.0
                + judge_prob.get("in", 0) * 0.5
                + judge_prob.get("no", 0) * 0.0
            )

            # weight = i + 1  # later elements get higher weight
            weight = 1  # All equal weight
            total_score += score * weight
            total_weight += weight

        average_score = total_score / total_weight if judge_probs else 0
        # print(f"Average Judge Score: {average_score}")

        cls.history.append(
            {
                "input": {
                    "current_url": cls._get_current_url(trajectory),
                    "trajectory": cls._get_brief_context(trajectory),
                    "text_observation": current_obs.get("axtree_txt", "No page content available"),
                    "action": proposed_action["code"],
                    "thought": proposed_action["thought"],
                    "progress": progress,
                },
                "output": generated_text,
                "probs": judge_probs,
                "logits": judge_scores,
                "score": average_score,
            }
        )

        # print(f"====== DONE =======")
        return average_score, generated_text


class WebClosePRM(WebPRM):
    @staticmethod
    def get_judge_probs(scores):
        response_str = ""
        judge_probs_list = []
        checklist_logs = []
        for record in scores:
            token_text = record.token
            # print(toke)
            if "# Checklist Evaluation" in response_str:
                # print("Start to find checklist evaluation...")
                find_judge_str = None
                for judge_type in target_judge:
                    if token_text in target_judge[judge_type]:
                        find_judge_str = judge_type
                        break
                if find_judge_str:
                    token_judge_dict = {"yes": None, "no": None, "in": None}

                    for lt in record.top_logprobs:
                        alt_text = lt.token
                        log_prob = lt.logprob
                        for judge_type in target_judge:
                            for judge_str in target_judge[judge_type]:
                                if judge_str in alt_text:
                                    token_judge_dict[judge_type] = WebPRM.add_to_ori_logprob(
                                        token_judge_dict[judge_type], log_prob
                                    )

                    checklist_logs.append(
                        {
                            "token": token_text,
                            "top_logprobs": [
                                {"token": lt.token, "logprob": lt.logprob}
                                for lt in record.top_logprobs
                            ],
                        }
                    )

                    for judge_type in token_judge_dict:
                        if token_judge_dict[judge_type] is None:
                            token_judge_dict[judge_type] = float("-inf")
                    judge_probs_list.append(token_judge_dict)

            response_str += token_text

        if len(judge_probs_list) == 0:
            return [{"yes": 0.0, "no": 0.0, "in": 0.0}], checklist_logs
        else:
            # convert with softmax
            final_judge_probs_list = []
            for judge_probs in judge_probs_list:
                exp_logprobs = [
                    math.exp(x) for x in [judge_probs["yes"], judge_probs["no"], judge_probs["in"]]
                ]
                sum_exp_logprobs = sum(exp_logprobs)
                if sum_exp_logprobs == 0:
                    softmax_probs = [0 for _ in exp_logprobs]
                else:
                    softmax_probs = [x / sum_exp_logprobs for x in exp_logprobs]
                final_judge_probs_list.append(
                    {"yes": softmax_probs[0], "no": softmax_probs[1], "in": softmax_probs[2]}
                )
            return final_judge_probs_list, checklist_logs

    def evaluate(
        self, trajectory: List[Dict[str, Any]], goal: str, checklist, **kwargs
    ) -> Tuple[float, str]:
        current_obs = self._get_current_observation(trajectory)
        proposed_action = current_obs.get("action", "No action")
        progress = trajectory[-2].get("checklist_completion") if len(trajectory) >= 2 else ""
        progress = progress if progress != "" else "No previous step"
        if self.prompt_type == "web_shepherd":
            data = {
                "intent": goal,
                "trajectory": self._get_brief_context(trajectory),
                "text_observation": current_obs.get("axtree_txt", "No page content available"),
                "current_url": self._get_current_url(trajectory),
                "action": proposed_action["code"],
                "thought": proposed_action["thought"],
                "checklist": checklist if checklist else "No checklist generated",
            }
        else:
            data = {
                "intent": goal,
                "trajectory": self._get_brief_context(trajectory),
                "text_observation": current_obs.get("axtree_txt", "No page content available"),
                "current_url": self._get_current_url(trajectory),
                "action": proposed_action["code"],
                "thought": proposed_action["thought"],
                "checklist": checklist if checklist else "No checklist generated",
                "progress": progress,  # TEMP
            }

        message = [
            {"role": "system", "content": SYSTEM_PROMPT_TEMPLATE},
            {
                "role": "user",
                "content": (
                    WS_USER_PROMPT_TEMPLATE.format(**data)
                    if self.prompt_type == "web_shepherd"
                    else WO_USER_PROMPT_TEMPLATE.format(**data)
                ),
            },
        ]

        # print("========= JUDGE EVALUATION =========")
        # print(USER_PROMPT_TEMPLATE.format(**data))

        # print("SYSTEM PROMPT: ", message[0]["content"])
        # print("USER PROMPT: ", message[-1]["content"])

        for _ in range(3):  # Try up to 3 times to get a valid answer
            response, scores = self.reward_model.chat(message)
            generated_text = response
            if "# Checklist Evaluation" in generated_text:
                # print(scores)
                break

        # print("RESPONSE: ", generated_text)

        # print("========= LOG PROBS =========")
        # check if <answer> and </answer> exist in the generated_text

        judge_probs, judge_scores = self.get_judge_probs(scores)
        # For each checkpoint, need weighted average
        # print(judge_probs)
        total_score = 0.0
        total_weight = 0.0

        for i, judge_prob in enumerate(judge_probs):
            score = (
                judge_prob.get("yes", 0) * 1.0
                + judge_prob.get("in", 0) * 0.5
                + judge_prob.get("no", 0) * 0.0
            )

            # weight = i + 1  # later elements get higher weight
            weight = 1  # All equal weight
            total_score += score * weight
            total_weight += weight

        average_score = total_score / total_weight if judge_probs else 0
        # print(f"Average Judge Score: {average_score}")

        self.history.append(
            {
                "input": {
                    "current_url": self._get_current_url(trajectory),
                    "trajectory": self._get_brief_context(trajectory),
                    "text_observation": current_obs.get("axtree_txt", "No page content available"),
                    "action": proposed_action["code"],
                    "thought": proposed_action["thought"],
                    "progress": progress,
                },
                "output": generated_text,
                "probs": judge_probs,
                "logits": judge_scores,
                "score": average_score,
            }
        )
        return average_score, generated_text
