import os
REDDIT = os.environ.get("WA_REDDIT", "")
SHOPPING = os.environ.get("WA_SHOPPING", "")
SHOPPING_ADMIN = os.environ.get("WA_SHOPPING_ADMIN", "")
MAP = os.environ.get("WA_MAP", "")
WIKIPEDIA = os.environ.get("WA_WIKIPEDIA", "")
GITLAB = os.environ.get("WA_GITLAB", "")

CHAT_SYSTEM_PROMPT = """
You are a UI Assistant, your goal is to help the user perform tasks using a web browser. You can
communicate with the user via a chat, to which the user gives you instructions and to which you
can send back messages. You have access to a web browser that both you and the user can see,
and with which only you can interact via specific commands. Review the instructions from the user, the current state of the page and all other information to find the best possible next action to accomplish your goal.
"""

SYSTEM_PROMPT = "You are a web automation agent that can navigate and interact with web pages to achieve user goals. If the goal cannot be completed on the current page, explore the site to locate the relevant page or element first. Always pick the single best next action to move closer to completion."

CROSS_SITE_TEMPLATE = """
>> Available Websites
The following websites are available for interaction in this benchmark:
{website_list}

IMPORTANT: These are simulated versions of real-world websites. 
You are NOT allowed to access or interact with the actual real websites (e.g., real Reddit, Wikipedia, etc.).
You must ONLY use the simulated benchmark websites listed above.
Any attempt to access real websites will be considered an error.
"""

USER_PROMPT_TEMPLATE = """
{benchmark_specifications}
>> Instructions
Review the current state of the page and all other information to find the best possible next action to accomplish your goal. Your answer will be interpreted and executed by a program, make sure to follow the formatting instructions.

{input_specifications}

>> Action Space

You are ONLY allowed to use the following action commands.

{action_space}

>> Generate the response in the following format:

# Observation Description
Describe the current page state and extract key information relevant to the goal. Focus on:
1. CONTENT EXTRACTION: If this page contains information needed for the final answer, extract and record it explicitly. This step is crucial for ensuring the final answer is accurate and complete. Don't miss any critical details.
2. RELEVANT ELEMENTS: Identify interactive elements, data, or content that helps accomplish the objective
Format your observation to help future answer extraction by being specific about:
- Exact values, numbers, prices, names, dates found on the page
- Location of critical information (which sections, forms, tables contain target data)

# Reason
Explain your rationale clearly. If the current interface appears to fulfill your objective, consider:
- Could there be hidden alternatives?
- Is there ambiguity in what's being shown (e.g., default sort orders)?
- Would it be safer to explore before committing?
Be cautious. It's OK to say: "This appears correct, but I want to confirm it by checking X."
Analyze previous actions. Do not get stuck in a loop by repeatedly trying the same action.

# Action
Select your action here. Strictly adheres to the given format. **Only issue one single action**.
"""

CLICK_ACTION = "click(bid: str): To click on an element with its numerical ID on the webpage. E.g., `click('a51')`."
SELECT_OPTION_ACTION = "select_option(bid: str, option: str): To select an option in a <select> element. You can specify option value or label to select. E.g., `select_option('237', 'Option 1')`. In case directly clicking an option returns error, you can try this out."
MULTI_SELECT_OPTION_ACTION = "select_option(bid: str, option: str): To select an option in a <select> element. You can specify option value or label to select. E.g., `select_option('237', 'Option 1')`. Multiple options can be selected. select_option('245', ['red', 'green', 'blue']). In case directly clicking an option returns error, you can try this out."
FILL_ACTION = 'fill(bid: str, value: str, press_enter_after: bool): To type content into a field with a specific ID. Note that, this function overwrites the existing text in that field. Optionally, it can press Enter after typing. E.g., `fill("237", "example value", True)` or `fill("237", "example value", False)`. In case this fill action is related to content writing (e.g., Reddit posts, comments, tweets, blog posts, forum posts, reviews, messages, emails, bios, descriptions), ensure that the "value" matches EXACTLY the text specified in the goal. Because your actions will be evaluated by exact string matcher.'
SCROLL_ACTION = "scroll(direction: str): To navigate the webpage content. E.g., `scroll('up')` or `scroll('down')`. In case, a page is too long vertically, some elements may be hidden. They can be revealed by scrolling. Such hidden items are usually represented using StaticText '[' and StaticText ']'"
GOTO_ACTION = "goto(url: str): Navigate to a url. E.g., `goto('http://www.example.com')`. It is recommended not to try any random url unless you have discovered it in previous interations."
GO_BACK_ACTION = "go_back(): To return to the previously viewed page. E.g., `go_back()`."
GO_FORWARD_ACTION = "go_forward(): Navigate to the next page in history. E.g., `go_forward()`."
STOP_ACTION = 'stop(text: str): To stop interaction. E.g., `stop("Based on the results of my search, the city was built in 1751.")`. If the task isn\'t a QnA, and you have completed the task, you should call "stop" with appropriate message. But if the task is a QnA, you should ensure that the "text" parameter is consistent with the "goal" and observations. Because your answer will be evaluated by exact string matcher.'
REPORT_INFEASIBLE_ACTION = "report_infeasible(reason: str): Notifies the user that their instructions are infeasible. E.g., `report_infeasible('I cannot follow these instructions because there is no email field in this form.')`."
NEW_TAB_ACTION = "new_tab(): Open a new tab. It will become the active one. Example: `new_tab()`"
TAB_FOCUS_ACTION = "tab_focus(index: int): Bring tab to front (activate tab). Example: `tab_focus(2)`"
TAB_CLOSE_ACTION = "tab_close(): Close the current tab. Example: `tab_close()`"
NOTE_ACTION = 'note(txt: str): To take note of all important info w.r.t. completing the task to enable reviewing it later. E.g., `note("Spent $10 on 4/1/2024")`'

action_prompt_map = {
    "click": CLICK_ACTION,
    "select_option": SELECT_OPTION_ACTION,
    "fill": FILL_ACTION,
    "goto": GOTO_ACTION,
    "go_back": GO_BACK_ACTION,
    "go_forward": GO_FORWARD_ACTION,
    "scroll": SCROLL_ACTION,
    "stop": STOP_ACTION,
    "report_infeasible": REPORT_INFEASIBLE_ACTION,
    "new_tab": NEW_TAB_ACTION,
    "tab_focus": TAB_FOCUS_ACTION,
    "tab_close": TAB_CLOSE_ACTION,
    "note": NOTE_ACTION,
}

GOAL_INPUT_SPECIFICATION = """
>> Goal
{goal}
{hint}
"""

REPHRASED_GOAL_INPUT_SPECIFICATION = """
{rephrased_goal}
"""

HISTORY_INPUT_SPECIFICATION = """
>> Previous Actions and Observations
{history}
"""

CURRENT_STEP = """
>> Current step: {current_step}
"""

OPEN_TABS_INPUT_SPECIFICATION = """
>> Currently open tabs
{open_tabs}
"""

AXTREE_INPUT_SPECIFICATION = """
>> Current page Accessibility Tree
{axtree}
"""

EXAMPLES_INPUT_SPECIFICATION = """
>> Similar Examples from Past Successes
{examples}
"""

NOTES_INPUT_SPECIFICATION = """
>> Notes
{notes}
"""

FAILED_ATTEMPT = """
>> Last Failed Attempt

# Response
{last_response}

# Error: {last_error}
"""

RESPONSE_ERROR_PAIR ="""
# Response {index}
{last_response}

# Error {index}: {last_error}
"""

FAILED_ATTEMPTS = """
>> Last Failed Attempts

{response_error_pairs}
"""

class PromptDesigner:
    benchmark: str = "webarena"
    
    @classmethod
    def configure(cls, benchmark="webarena"):
        cls.benchmark = benchmark
        
    @staticmethod
    def format_examples(examples) -> str:
        """Format retrieved examples for the prompt"""
        if not examples:
            return "No relevant examples found."

        examples_parts = [
            f"Here are {len(examples)} similar successful examples from past interactions:"]

        for i, ex in enumerate(examples, 1):
            action_str = str(ex["action"]) if ex["action"] else "No action"
            thought = ex.get("thought", "No thought available")
            example_goal = ex.get("goal", "Unknown goal")
            score = ex.get("score", 0.0)

            examples_parts.append(f"\nExample {i} (relevance: {score:.2f}):")
            examples_parts.append(f"  Goal: {example_goal}")
            examples_parts.append(f"  Thought: {thought}")
            examples_parts.append(f"  Action: {action_str}")

        examples_parts.append(
            "\nUse these examples as guidance, but adapt your action to the current context and goal.")
        return "\n".join(examples_parts)
    
    @staticmethod
    def format_history(trajectory, history_length) -> str:
        """Build a text summary of the trajectory history"""
        if len(trajectory) <= 1:
            return "No previous actions taken."

        history_parts = []

        # Show last few steps to avoid token overflow
        recent_trajectory = trajectory[-history_length:-1] if len(
            trajectory) > history_length else trajectory[:-1]

        for i, step in enumerate(recent_trajectory):
            step_num = len(trajectory) - len(recent_trajectory) + i

            # Extract action if available
            action = step.get('action', 'No action')
            action_error = step.get('action_error')

            # Get current URL from open_pages_urls and active_page_index
            urls = step.get('open_pages_urls', [])
            active_idx = step.get('active_page_index', 0)
            url = urls[active_idx] if urls and active_idx < len(
                urls) else 'Unknown URL'

            # Get a brief description of the observation
            history_parts.append(
                f"Observation #{step_num}: (Url: {url}) \n{step.get('obs_description', 'No observation available')}")
            history_parts.append(f"Reason #{step_num}: {action['thought']}")
            history_parts.append(f"Action #{step_num}: {action['code']}\n")
            if action_error:
                history_parts.append(
                    f"Action Error #{step_num}: {action_error}\n")

        return "\n".join(history_parts)
    
    @classmethod
    def design_actor_prompt(cls, goal, rephrased_goal, trajectory, history_length, examples, action_space, notes, prev_responses, prev_errors, chat_messages=None, cross_site=False):
        """Build prompt for action generation using template"""

        input_specifications = ""

        if chat_messages is None:            
            hint = ""
            if "reddit" in goal.lower() or "forum" in goal.lower():
                hint = f"Note: Reddit website is simulated through Postmil website. For any task related to reddit/forums you should use Postmil website hosted at {os.environ['WA_REDDIT']}"
            # if "repository" in goal.lower() or "gitlab" in goal.lower():
            #     hint = f"For any task related to repository you should use GitLab website hosted at {os.environ['WA_GITLAB']}"
            input_specifications += GOAL_INPUT_SPECIFICATION.format(goal=goal, hint=hint)

            # Add rephrased goal if available
            if rephrased_goal:
                input_specifications += REPHRASED_GOAL_INPUT_SPECIFICATION.format(
                    rephrased_goal=rephrased_goal)
        
        input_specifications += HISTORY_INPUT_SPECIFICATION.format(
            history=cls.format_history(trajectory, history_length)
        )
        
        input_specifications += CURRENT_STEP.format(current_step=len(trajectory))

        input_specifications += OPEN_TABS_INPUT_SPECIFICATION.format(
            open_tabs="\n".join(
                f"Tab {i} - {title} ({url})" + ("-> Active" if trajectory[-1]["active_page_index"] else "")
                for i, (url, title) in enumerate(zip(
                    trajectory[-1].get("open_pages_urls", []),
                    trajectory[-1].get("open_pages_titles", [])
                ))
            )
        )

        current_obs = trajectory[-1] if trajectory else {}

        input_specifications += AXTREE_INPUT_SPECIFICATION.format(
            axtree=current_obs.get("axtree_txt",
                                   "No Accessibility Tree available")
        )

        # If retriever is available, add examples
        if examples is not None:
            if len(examples) > 0:
                input_specifications += EXAMPLES_INPUT_SPECIFICATION.format(
                    examples=cls.format_examples(examples)
                )
            else:
                print("No relevant examples found for the current goal.")

        if "note" in action_space:
            input_specifications += NOTES_INPUT_SPECIFICATION.format(
                notes="\n".join(f"- {note}" for note in notes) if len(notes) > 0 else "No notes taken so far."
            ) 
            
        
        
        benchmark_specifications = ""
        if cls.benchmark != "openended":
            benchmark_specifications = """>> Benchmark Evaluation Framework
You are participating in a web automation benchmark evaluation. Your actions will be assessed for accuracy and efficiency."""

        if cls.benchmark =="webarena":
            if cross_site:
                benchmark_specifications += "\n\n" + CROSS_SITE_TEMPLATE.format(
                    website_list="\n".join(
                        f"- {site}: {record['description']}, Url: {record['url']}"
                        for site, record in {
                            "GitLab": {
                                "url": GITLAB,
                                "description": "A web-based DevOps lifecycle tool that provides a Git repository manager. For any task related to repository you should use this."
                            },
                            "Reddit": {
                                "url": REDDIT,
                                "description": "A social news aggregation, web content rating, and discussion website. For any task related to reddit/forums you should use this."
                            },
                            "Shopping": {
                                "url": SHOPPING,
                                "description": "An online shopping platform named One Stop Market."
                            },
                            "Shopping Admin": {
                                "url": SHOPPING_ADMIN,
                                "description": "Admin panel for managing the shopping platform."
                            },
                            "Wikipedia": {
                                "url": WIKIPEDIA,
                                "description": "A free online encyclopedia."
                            },
                            "Map": {
                                "url": MAP,
                                "description": "A web-based mapping service."
                            }
                        }.items() if record['url']  # Only include sites with valid URLs
                    ),
                    action_space="- " + "\n- ".join(action_space),
                    # last_action=current_obs.get("last_action_error") if current_obs.get("last_action_error") is not None else current_obs.get("last_action")
                )
            else:
                benchmark_specifications += "**Note** You are not allowed to leave the current website domain."
                
        if cls.benchmark != "openended":
            benchmark_specifications += "\n"

        action_space_prompt = [action_prompt_map[action] for action in action_space if action is not None]
        USER_PROMPT = USER_PROMPT_TEMPLATE.format(
            benchmark_specifications=benchmark_specifications,
            input_specifications=input_specifications,
            action_space="- " + "\n- ".join(action_space_prompt),
        )
        
        if len(prev_responses) > 0 and len(prev_errors) > 0:
            recent_responses = prev_responses[-5:] if len(prev_responses) >=5 else prev_responses
            recent_errors = prev_errors[-5:] if len(prev_errors) >=5 else prev_errors
            USER_PROMPT += "\n" + FAILED_ATTEMPTS.format(
                response_error_pairs="\n".join([RESPONSE_ERROR_PAIR.format(index=idx+1, last_response="```\n"+resp+"\n```", last_error=err) for idx, (resp, err) in enumerate(zip(recent_responses, recent_errors))])
            )
            
        if chat_messages is not None:
            user_msgs = []
            user_msgs.append(
                {
                    "type": "text",
                    "text": f"""\
# Chat Messages
""",
                }
            )
            for msg in chat_messages:
                if msg["role"] in ("user", "assistant", "infeasible"):
                    user_msgs.append(
                        {
                            "type": "text",
                            "text": f"""\
- [{msg['role']}] {msg['message']}
""",
                        }
                    )
                elif msg["role"] == "user_image":
                    user_msgs.append({"type": "image_url", "image_url": msg["message"]})
                else:
                    raise ValueError(f"Unexpected chat message role {repr(msg['role'])}")
                
            user_msgs.append(
                {
                    "type": "text",
                    "text": USER_PROMPT,
                }
            )
            return [{"type": "text", "text": SYSTEM_PROMPT}], user_msgs
        
        return SYSTEM_PROMPT, USER_PROMPT
        