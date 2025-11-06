from .models.base import BaseModel
import json
from pathlib import Path

REPHRASER_SYSTEM_PROMPT = """
You are a rephraser module that helps a task-solving agent better understand a web-based task.

Given a task description, you should:
1. Rephrase the task in clearer language without changing its meaning.
2. Identify logical subgoals needed to accomplish the task.
3. Detect any vague or ambiguous terms.
4. Identify possible implicit assumptions that need verification.
5. List cautions — things the agent should be skeptical about when interpreting the task or interacting with the interface.

Do not rely on any specific knowledge of websites, domains, or UI layouts. Stay general-purpose.
"""

REPHRASER_USER_PROMPT = """
>> Task Objective
{task_description}

>> Current page URL
{current_url}

>> Current page Accessibility Tree
{axtree}

>> Generate the response in the following format:
# REPHRASED TASK:
<Clear, literal paraphrase of the original task. No assumptions.>

# SUBGOALS:
- <Subgoal 1>
- <Subgoal 2>
- ...

# AMBIGUITIES OR ASSUMPTIONS:
- <Potentially vague or implied elements in the task>

# CAUTIONS:
- <Any points where the agent should validate its understanding rather than assume>
"""

LAST_ERROR = """
>> Last failed attempt
# REPHRASED TASK: not found in response
"""


class TaskRephraser:
    model = None

    @classmethod
    def set_model(cls, model: BaseModel):
        cls.model = model

    @classmethod
    def rephrase(cls, task_description: str, current_url: str, axtree: str) -> str:
        current_dir = Path(__file__).resolve().parent  # directory of current file
        file_path = (
            current_dir / f"rephrased_tasks/{cls.model.name.split(':')[0].split('/')[-1]}.jsonl"
        )

        # create parent dir using Path (safe for Path objects)
        file_path.parent.mkdir(parents=True, exist_ok=True)

        # create file if missing with an empty list
        if not file_path.exists():
            file_path.write_text("", encoding="utf-8")

        # ✅ Search for existing rephrased task line-by-line
        existing_data = None
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    item = json.loads(line)
                    if item["task"] == task_description: # and item["url"] == current_url:
                        existing_data = item
                        break
                except json.JSONDecodeError:
                    continue

        if existing_data:
            print("Loaded existing rephrased task from rephrased_tasks.jsonl")
            return existing_data["rephrased_task"]

        for i in range(3):
            response, _ = cls.model.chat(
                messages=[
                    {"role": "system", "content": REPHRASER_SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": REPHRASER_USER_PROMPT.format(
                            task_description=task_description,
                            current_url=current_url,
                            axtree=axtree,
                        )
                        + (LAST_ERROR if i > 0 else ""),
                    },
                ]
            )
            # check if # REPHRASED TASK exist or not.
            # If yes, extract the part after # REPHRASED TASK including # REPHRASED TASK
            if "# REPHRASED TASK:" in response:
                response = response.split("# REPHRASED TASK:")[-1]
                response = "# REPHRASED TASK:" + response
                # ✅ Append result as one line of JSON
                with open(file_path, "a", encoding="utf-8") as f:
                    json.dump(
                        {"url": current_url, "task": task_description, "rephrased_task": response},
                        f,
                        ensure_ascii=False,
                    )
                    f.write("\n")
                break

        return response
