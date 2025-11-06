import os
import subprocess
import os
import json
import time
import os
reset_counter = {}
GITLAB_DESTRUCTIVE_TASKS = [389, 390, 391, 392, 393, 394, 395, 396, 397, 398, 411, 412, 414, 415, 416, 417, 418, 419, 420, 421, 422, 441, 442, 443, 444, 445, 446, 447, 448, 449, 450, 451, 452, 475, 476, 477, 478, 479, 480, 481, 482, 483, 484, 485, 522, 523, 524, 525, 526, 527, 533, 534, 535, 536, 537, 552, 553, 554, 555, 556, 557, 558, 559, 560, 561, 562, 563, 564, 565, 566, 567, 568, 569, 570, 576, 577, 578, 579, 590, 591, 592, 593, 594, 658, 659, 660, 661, 662, 663, 664, 665, 666, 667, 668, 669, 670, 736, 742, 743, 744, 745, 746, 747, 748, 749, 750, 751, 752, 753, 754, 756, 789, 791, 799, 800, 801, 802, 803, 804, 805, 806, 807, 808, 809, 810, 811]

REDDIT_DESTRUCTIVE_TASKS = [399, 400, 401, 402, 403, 404, 405, 406, 407, 408, 409, 410, 580, 581, 582, 583, 584, 595, 596, 597, 598, 599, 600, 601, 602, 603, 604, 605, 606, 607, 608, 609, 610, 611, 612, 613, 614, 615, 616, 617, 618, 619, 620, 621, 622, 623, 624, 625, 626, 627, 628, 629, 630, 631, 632, 633, 634, 635, 636, 637, 638, 639, 640, 641, 642, 643, 644, 645, 646, 647, 648, 649, 650, 651, 652, 671, 672, 673, 674, 675, 681, 682, 683, 684, 685, 686, 687, 688, 714, 715, 716, 717, 718, 719, 720, 721, 722, 723, 724, 725, 726, 727, 728, 729, 730, 731, 732, 733, 734, 735]

SHOPPING_DESTRUCTIVE_TASKS = [431, 432, 433, 434, 435, 436, 437, 438, 439, 440, 465, 466, 467, 468, 469, 506, 507, 508, 509, 510, 511, 512, 513, 514, 515, 516, 517, 518, 519, 520, 521, 528, 529, 530, 531, 532, 571, 572, 573, 574, 575, 585, 586, 587, 588, 589, 653, 654, 655, 656, 657, 689, 690, 691, 692, 693, 792, 793, 794, 795, 796, 797, 798]

SHOPPING_ADMIN_DESTRUCTIVE_TASKS = [423, 453, 454, 455, 456, 457, 458, 459, 460, 461, 462, 463, 464, 470, 471, 472, 473, 474, 486, 487, 488, 489, 490, 491, 492, 493, 494, 495, 496, 497, 498, 499, 500, 501, 502, 503, 504, 505, 538, 539, 540, 541, 542, 543, 544, 545, 546, 547, 548, 549, 550, 551, 694, 695, 696, 697, 698, 699, 700, 701, 702, 703, 768, 769, 770, 771, 772, 773, 774, 775, 776, 777, 778, 779, 780, 781, 782, 790]

REDDIT_POST_TASKS = list(range(600, 650))+list(range(681, 689))

def _get_destructed_site(task_id):
    sites = []
    if task_id in GITLAB_DESTRUCTIVE_TASKS:
        sites.append("gitlab")
    if task_id in REDDIT_DESTRUCTIVE_TASKS:
        sites.append("reddit")
    if task_id in SHOPPING_DESTRUCTIVE_TASKS:
        sites.append("shopping")
    if task_id in SHOPPING_ADMIN_DESTRUCTIVE_TASKS:
        sites.append("shopping_admin")
    return sites

def is_destructive_task(task_id, site):
    if site == "gitlab":
        return task_id in GITLAB_DESTRUCTIVE_TASKS
    elif site == "reddit":
        return task_id in REDDIT_DESTRUCTIVE_TASKS
    elif site == "shopping":
        return task_id in SHOPPING_DESTRUCTIVE_TASKS
    elif site == "shopping_admin":
        return task_id in SHOPPING_ADMIN_DESTRUCTIVE_TASKS
    return False
    
task_dependencies = None

def _get_task_dependencies(task_id):
    global task_dependencies
    if task_dependencies is None:
        dependency_path = os.path.join("webarena/docker_status", "dependency.json")
        with open(dependency_path, "r", encoding="utf-8") as f:
            task_dependencies = json.load(f)
    return task_dependencies[task_id]

def _get_prev_tasks(site = None):
    if site is None:
        all_prev_tasks = []
        seen_task_ids = set()
        for site in ["gitlab", "reddit", "shopping_admin", "shopping"]:
            prev_tasks_path = os.path.join("webarena/docker_status", f"{site}_tasks.json")
            with open(prev_tasks_path, "r", encoding="utf-8") as f:
                prev_tasks = json.load(f)
                for task in prev_tasks:
                    if task["task_id"] not in seen_task_ids:
                        all_prev_tasks.append(task)
                        seen_task_ids.add(task["task_id"])
        return all_prev_tasks
    
    if site in ["gitlab", "reddit", "shopping_admin", "shopping"]:
        prev_tasks_path = os.path.join("webarena/docker_status", f"{site}_tasks.json")
        with open(prev_tasks_path, "r", encoding="utf-8") as f:
            prev_tasks = json.load(f)
        return prev_tasks
    return []

def _set_prev_tasks(site, tasks):
    prev_tasks_path = os.path.join("webarena/docker_status", f"{site}_tasks.json")
    with open(prev_tasks_path, "w", encoding="utf-8") as f:
        json.dump(tasks, f, indent=2)
        
def _clear_prev_tasks(site):
    _set_prev_tasks(site, [])
    
def _add_to_prev_tasks(site, task_record):
    prev_tasks = _get_prev_tasks(site)
    flag = True
    for pt in prev_tasks:
        if pt["task_id"] == task_record["task_id"]:
            pt["start_time"] = task_record["start_time"]
            # delete end_time if exists
            if "end_time" in pt:
                del pt["end_time"]
            flag = False
            break
    if flag:
        prev_tasks.append(task_record)
    _set_prev_tasks(site, prev_tasks)

def sync_reset_site(site: str):
    """Reset the site by running the necessary scripts."""
    if site not in ["gitlab", "reddit", "shopping_admin", "shopping"]:
        return
    
    # check if any unfinished tasks on this site in the last 30 minutes
    prev_tasks_path = os.path.join("webarena/docker_status", f"{site}_tasks.json")
    with open(prev_tasks_path, "r") as f:
        tasks = json.load(f)
        
    for task in tasks:
        if task.get("start_time") is not None and task.get("end_time") is None: 
            print(f"Site {site} has unfinished task {task['task_id']} started {time.time() - task['start_time']} seconds ago. Skipping reset.")
            time.sleep(5)
            return
    
    # Update reset counter
    reset_counter[site] = reset_counter.get(site, 0) + 1
    # print(f"Resetting {site} (attempt #{reset_counter[site]})")
    print(f"\033[94mScheduling reset for {site}\033[0m")

    # if reset_counter[site] == 1:
    #     print("Skipping reset for the first attempt.")
    #     script = f"spin_{site}.sh"
    # else:
    
    # Get project root: f:\QCRI\BrowserGym\utils\webarena.py -> f:\QCRI\BrowserGym\
    current_file_dir = os.path.dirname(__file__)  # f:\QCRI\BrowserGym\utils
    project_root = os.path.dirname(current_file_dir)  # f:\QCRI\BrowserGym
    script_dir = os.path.join(project_root, "webarena", "environment-docker")
    script_path = os.path.join(script_dir, "restart_site.sh")
    
    print(f"Running script: {script_path}")
    
    # Run the script in the correct directory
    # subprocess.run(["bash", script_path, site], cwd=script_dir, check=True)
    # Run the script asynchronously
    subprocess.run(["bash", script_path, site], cwd=script_dir, check=True)

    time.sleep(5)  # Give some time for the script to start
    
def reset_site(site: str):
    """Reset the site by running the necessary scripts."""
    if site not in ["gitlab", "reddit", "shopping_admin", "shopping"]:
        return
    
    # check if any unfinished tasks on this site in the last 30 minutes
    prev_tasks_path = os.path.join("webarena/docker_status", f"{site}_tasks.json")
    with open(prev_tasks_path, "r") as f:
        tasks = json.load(f)
        
    for task in tasks:
        if task.get("start_time") is not None and task.get("end_time") is None: 
            print(f"Site {site} has unfinished task {task['task_id']} started {time.time() - task['start_time']} seconds ago. Skipping reset.")
            time.sleep(5)
            return
    
    # Update reset counter
    reset_counter[site] = reset_counter.get(site, 0) + 1
    # print(f"Resetting {site} (attempt #{reset_counter[site]})")
    print(f"\033[94mScheduling reset for {site}\033[0m")

    # if reset_counter[site] == 1:
    #     print("Skipping reset for the first attempt.")
    #     script = f"spin_{site}.sh"
    # else:
    
    # Get project root: f:\QCRI\BrowserGym\utils\webarena.py -> f:\QCRI\BrowserGym\
    current_file_dir = os.path.dirname(__file__)  # f:\QCRI\BrowserGym\utils
    project_root = os.path.dirname(current_file_dir)  # f:\QCRI\BrowserGym
    script_dir = os.path.join(project_root, "webarena", "environment-docker")
    script_path = os.path.join(script_dir, "restart_site.sh")
    
    print(f"Running script: {script_path}")
    
    # Run the script in the correct directory
    # subprocess.run(["bash", script_path, site], cwd=script_dir, check=True)
    # Run the script asynchronously
    subprocess.Popen(["bash", script_path, site], cwd=script_dir, stdout=subprocess.DEVNULL)
    
    time.sleep(5)  # Give some time for the script to start

def prepare_environment(task_config, reset_strategy):
    print(f"Preparing environment for task {task_config['task_id']} on sites {task_config['sites']}")
    task_id = task_config["task_id"]
    
    with open("webarena/docker_status/site_status.json", "r", encoding="utf-8") as f:
        site_status = json.load(f)

    for site in task_config["sites"]:
        if site not in ["gitlab", "reddit", "shopping_admin", "shopping"]:
            continue
        status = site_status[site]
        if status == "Starting":
            print(f"Site {site} is currently Starting. Waiting for it to be Running.")
            time.sleep(5)
            return False
        elif status == "Stopped" and reset_strategy != "never":
            print(f"Site {site} is Stopped. Resetting it now.")
            reset_site(site)
            return False
        elif status == "Running":
            pass
        else:
            print(f"Site {site} is in status {status}. Skipping reset.")
            time.sleep(5)
            return False
        
    print(f"Checking if task {task_id} can be started based on previous tasks.")
    
    task_dependencies = _get_task_dependencies(task_id)
    print(f"Task {task_id} depends on tasks {task_dependencies}.")
    
    prev_tasks = _get_prev_tasks()
    prev_task_ids = {pt["task_id"] for pt in prev_tasks}

    # If any of the dependent tasks have been executed before, reset the site
    if len(set(task_dependencies).intersection(prev_task_ids)) > 0:
        print(f"Task {task_id} depends on tasks {task_dependencies} which have been executed before. Resetting sites.")
        destructed_sites = _get_destructed_site(task_id)
        for site in destructed_sites:
            if reset_strategy != "never":
                reset_site(site)
        if len(destructed_sites) > 0:
            return False
        
    print(f"All dependencies for task {task_id} are satisfied.")
    if task_id in REDDIT_POST_TASKS:
        prev_tasks = _get_prev_tasks("reddit")
        
        last_post_time = None 
        for pt in prev_tasks:
            if pt["task_id"] in REDDIT_POST_TASKS:
                if pt.get("end_time") is not None and (last_post_time is None or pt["end_time"] > last_post_time):
                    last_post_time = pt["end_time"]
                    
        if last_post_time is not None:
            elapsed_time = time.time() - last_post_time
            if elapsed_time < 20 * 60: # Can't post more than once every 20 minutes
                # return False
                wait_time = 20 * 60 - elapsed_time
                if reset_strategy == "never":
                    # print(f"Task {task_id} is a Reddit post task. Last post was {elapsed_time/60:.2f} minutes ago. Need to wait for another {wait_time/60:.2f} minutes to avoid Reddit rate limit.")
                    # if wait time is more than 10 minutes, just reset reddit
                    print(f"Waiting for {wait_time/60:.2f} minutes before starting task {task_id} to avoid Reddit rate limit.")
                    time.sleep(wait_time)
                    # return True
                    return True
                elif wait_time > 10 * 60:
                    print(f"Resetting reddit to avoid Reddit rate limit.")
                    reset_site("reddit")
                    return False
                else:
                    time.sleep(5)
                    return False
                # else:
                #     print(f"Waiting for {wait_time/60:.2f} minutes before starting task {task_id} to avoid Reddit rate limit.")
                #     time.sleep(wait_time)
    
    # for site in destructed_sites:
    #     _add_to_prev_tasks(site, {
    #         "task_id": task_id,
    #         "start_time": time.time()
    #     })
    destructed_sites = _get_destructed_site(task_id)
    for site in destructed_sites:
        prev_tasks = _get_prev_tasks(site)
        n_destructive = 0
        for pt in prev_tasks:
            if is_destructive_task(pt["task_id"], site) and pt.get("end_time"):
                n_destructive += 1
            
        # if n_destructive >= 5:
        #     print(f"Site {site} has already executed {n_destructive} destructive tasks. Resetting it now.")
        #     reset_site(site)
        #     return False
    
    return True
   
def before_task_start(task_id, sites):
    # destructed_sites = _get_destructed_site(task_id)
    destructed_sites = sites
    for site in destructed_sites:
        task_record = {
            "task_id": task_id,
            "start_time": time.time()
        }
        # print(f"Adding task {task_id} to {site} previous tasks.")
        _add_to_prev_tasks(site, task_record)
             
def after_task_end(task_id, sites):
    # destructed_sites = _get_destructed_site(task_id)
    destructed_sites = sites
    for site in destructed_sites:
        prev_tasks = _get_prev_tasks(site)
        for pt in prev_tasks:
            if pt["task_id"] == task_id:
                pt["end_time"] = time.time()
                # clear start_time
                # if "start_time" in pt:
                #     del pt["start_time"]
        _set_prev_tasks(site, prev_tasks)
    
