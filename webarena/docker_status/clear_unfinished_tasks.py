import json
import time

for site in ["reddit", "gitlab", "shopping", "shopping_admin"]:
    with open(f"{site}_tasks.json", "r") as f:
        tasks = json.load(f)
        
    new_tasks = []
    
    for task in tasks:
        if task.get("start_time") is not None and task.get("end_time") is None: 
            pass  # unfinished for more than 30 minutes
        else:
            new_tasks.append(task)

    with open(f"{site}_tasks.json", "w") as f:
        json.dump(new_tasks, f, indent=2)