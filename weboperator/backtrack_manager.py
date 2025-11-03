from .web_state_node import WebStateNode
from .observation_processor import ObservationProcessor
from .axtree_utils import is_diff_axtree_obj_by_bid
from .action_selector import ActionSelector
from .action_processor import ActionProcessor
# from browsergym.experiments.loop import get_env
import time
import gymnasium as gym

class BacktrackManager:
    simulation_enabled = True
    
    def __init__(self):
        self.pending_actions = []
        self.pending_nodes = []
        self.is_backtracking = False
        
    @classmethod
    def configure(cls, simulation_enabled: bool, destruction_aware: bool):
        cls.simulation_enabled = simulation_enabled
        ActionSelector.destruction_aware_backtracking = destruction_aware

    def reset(self):
        self.pending_actions = []
        self.pending_nodes = []
        self.is_backtracking = False
    
    def start_backtracking(self, actions, nodes):
        self.is_backtracking = True
        self.pending_actions = actions
        self.pending_nodes = nodes

    def stop_backtracking(self):
        self.is_backtracking = False
        self.pending_actions = []
        self.pending_nodes = []

    def get_next_backtrack_action(self):
        if len(self.pending_actions) == 0:
            self.stop_backtracking()
            return None, None
        action = self.pending_actions.pop(0)
        node = self.pending_nodes.pop(0)
        if len(self.pending_actions) == 0:
            self.stop_backtracking()
        return action, node

    @staticmethod
    def get_reset_steps(source_n_tabs, anchestor_tab_urls):
        reset_actions = []
        reset_nodes = []
        if source_n_tabs > 1:
            for i in range(source_n_tabs - 1):
                reset_actions.append("tab_close()")
        reset_actions.append(f"goto('{anchestor_tab_urls[0]}')")
        n_tabs = len(anchestor_tab_urls)
        # Multiple tab in anchestor means, that is root. And active index of root is n_tabs-1
        for i in range(1, n_tabs):
            reset_actions.append("new_tab()")
            reset_actions.append(f"goto('{anchestor_tab_urls[i]}')")

        for i in range(len(reset_actions) - 1):
            reset_nodes.append(None)
            
        tmp_node = WebStateNode(
            last_action={"type": "goto", "args": {"url": anchestor_tab_urls[0]}, "code": f"goto('{anchestor_tab_urls[0]}')", "thought": "Jump Action"},
            last_obs_description=None,
        )
        tmp_node.destructive = False
        tmp_node.parent = None
        reset_nodes.append(tmp_node)
        
        return reset_actions, reset_nodes

    @classmethod
    def get_backtrack_actions(cls, start_node, end_node: WebStateNode, env: gym.Env):
        # if is_same_node(end_node.parent, start_node): 
        #     return True
        nodes = []
        actions = []
        reset_actions = []
        source_n_tabs = len(start_node.open_pages_urls)
        def append_action(curr_node, action_code):
            if action_code is not None and not action_code.startswith("note"):
                actions.append(action_code)
                nodes.append(curr_node)

        if len(end_node.parent.open_pages_urls) == 1: # Single Tab Case
            last = end_node.last_action
            if last["type"] == "goto":
                # Directly jump to destination state
                reset_actions, reset_nodes = BacktrackManager.get_reset_steps(source_n_tabs, [last["args"]["url"]])
                return reset_actions, reset_nodes
            
        # Start with the last action
        go_back_count = 0
        go_forward_count = 0
        if end_node.last_action is not None:
            append_action(end_node, end_node.last_action["code"])
            if end_node.last_action["type"] == "go_back":
                go_back_count += 1
            if end_node.last_action["type"] != "go_forward":
                # end_node = end_node.parent.parent
                go_forward_count += 1
            
        # actions = [node.last_action["code"]]
        # nodes.append(node)
        
        target_node = end_node.parent
        curr = target_node

        # Multiple Tab Case
        while len(curr.open_pages_urls) > 1 and curr.last_action is not None:
            if curr.last_action["type"] == "go_back":
                go_back_count += 1
            elif curr.url != curr.parent.url and go_back_count > 0:
                go_back_count -= 1
            # if curr.last_action["type"] == "go_back" and :
            #     go_back_flag = True
            append_action(curr, curr.last_action["code"])
            curr = curr.parent

        # Single Tab Case -> Find safe anchestor of curr
        while curr is not None:
            if curr.destructive:
                break
            if curr.parent is None or curr.parent.url != curr.url:
                # Safe to stop here?
                if curr.parent is not None and go_back_count > 0:
                    append_action(curr, curr.last_action["code"])
                    if curr.last_action["type"] == "go_back":
                        go_back_count += 1
                    else:
                        go_back_count -= 1
                elif curr.parent is not None and curr.last_action["type"] == "go_back":
                    append_action(curr, curr.last_action["code"])
                    go_back_count += 1
                else:
                    print(f"Found a likely anchestor. Checking if safe to stop here: {curr.url}")
                    reset_actions, reset_nodes = cls.get_reset_steps(source_n_tabs, curr.open_pages_urls)
                    final_actions = reset_actions+list(reversed(actions))
                    final_nodes = reset_nodes+list(reversed(nodes))
  
                    if cls._simulate(final_actions, final_nodes, start_node.active_page_index, env):
                        return final_actions, final_nodes
                    elif curr.parent is not None: # Need to go further up
                        append_action(curr, curr.last_action["code"])
                        if curr.last_action["type"] == "go_back":
                            go_back_count += 1
                    else:
                        break
            else:
                append_action(curr, curr.last_action["code"])
                if curr.last_action["type"] == "go_back":
                    go_back_count += 1
            curr = curr.parent
        return None, None

    @classmethod
    def _simulate(cls, actions, nodes, active_page_index, env: gym.Env):
        if not cls.simulation_enabled:
            return True
        # actions = list(reversed(actions))
        # nodes = list(reversed(nodes))
        # red color print
        # print(actions)
        print(f"[SIMULATION] Simulating {len(actions)} actions for backtrack verification.")
        
        for action in actions:
            print(f"\033[91m{action}\033[0m")
            
        # new_tab(actions[0])
        # actions = actions[1:]
        
        # First action is always goto
        if len(actions) <= 1:
            print(f"[SIMULATION] Only {len(actions)} action(s), skipping simulation.")
            return True
        
        for node in nodes:
            if node is not None and node.corrupted:
                print(f"[SIMULATION] Node in the path already marked as corrupted, skipping simulation.")
                return False
            
        bid_flag = False
        for node in nodes:
            if node is not None and node.last_action is not None and node.last_action["type"] in ("click", "fill", "select_option"):
                bid_flag = True
                break
        
        if not bid_flag:
            print(f"[SIMULATION] No bid-based actions in the path, skipping simulation.")
            return True
            
        for i in range(1, len(actions) - 1):       
            if nodes[i] is not None and nodes[i].parent is not None:
                # Only for i = 0, nodes[i].last_action["code"] may not equal actions[i]
                # if nodes[i].last_action["code"] == actions[i]:
                #     last_action = nodes[i].last_action
                # elif actions[i].startswith("goto"):
                #     last_action = {
                #         "type": "goto",
                #         "args": {
                #             "url": extract_url_from_goto(actions[i])
                #         },
                #         "code": actions[i]
                #     }
                # else:
                #     last_action = nodes[i].last_action
                if nodes[i].destructive: # and nodes[i].last_action_error == "":
                    print(f"[SIMULATION] Destructive action found at index {i}: {nodes[i].last_action['code']}")
                    return False

        # Clone current tabs
        # env = get_env()
        current_total_tabs = len(env.context.pages)
        current_active_tab = active_page_index
        original_page = env.page
        # create same number of tabs
        for _ in range(current_total_tabs):
            env.context.new_page()
        
        env.page = env.context.pages[current_total_tabs + current_active_tab]
        
        # actions[0] = goto('{node.url}')
        # print(f"[SIMULATION] First action: {actions[-1]}")
        # print(f"[SIMULATION] First node URL: {extract_url_from_goto(actions[-1])}")
        # env.page.goto(extract_url_from_goto(actions[-1]), timeout=0)
        # actions = actions[:-1]
        # what if one of the actions is related to tabs?
        flag = True
        obs = None
        
        for i, action in enumerate(actions):
            # if action.startswith("new_tab") or action.startswith("tab_close") or action.startswith("tab_focus"):
            #     print(f"[SIMULATION] Skipping tab action {action} during simulation.")
            #     break
            if nodes[i] is not None:
                formatted_action = nodes[i].last_action
                if formatted_action is not None and formatted_action["type"] == "tab_focus":
                    relative_tab_index = formatted_action["args"]["index"]
                    # relative_tab_index = int(re.search(r'tab_focus\((\d+)\)', action).group(1))
                    abs_tab_index = current_total_tabs + relative_tab_index
                    action = f"tab_focus({abs_tab_index})"
                    print(f"[SIMULATION] Converted tab_focus action to absolute index: {action}")

                if obs is not None and nodes[i].last_action is not None and formatted_action["type"] in ("click", "fill", "select_option"): # bid-based actions
                    bid = formatted_action["args"]["bid"]
                    if is_diff_axtree_obj_by_bid(obs["axtree_object"], nodes[i].parent.axtree_object, bid):
                        print(f"[SIMULATION] Simulation mismatch at action {action} due to bid {bid}.")
                        # print(f"[SIMULATION] Current URL: {obs['url']}, Node URL: {nodes[i].url}")
                        # print_first_diff_lines(obs["axtree_txt"], nodes[i].axtree_txt)
                        # nodes[i].corrupted = True
                        flag = False
                        break
                    else:
                        print(f"[SIMULATION] Bid {bid} matches for action {action}.")
            
                    action = ActionProcessor.postprocess(formatted_action, obs["axtree_object"])["code"]
                
            if i >= len(actions) - 1:
                break
            
            obs, _, _, _, _ = env.step(action)
            obs = ObservationProcessor.process_obs(obs)
            
            if obs["last_action_error"] != "":
                print(f"[SIMULATION] Simulation error at action {action}: {obs['last_action_error']}")
                # if nodes[i] is not None:
                #     nodes[i].corrupted = True
                # flag = False
                # break
                
            # if is_diff_axtree_obj(obs["axtree_object"], nodes[i].axtree_object):
            # # if is_diff_axtree(obs["axtree_txt"], nodes[i].axtree_txt):
            #     print(f"[SIMULATION] Simulation mismatch at action {action}.")
                
            #     if obs["url"] != nodes[i].url:
            #         print(f"[SIMULATION] URL mismatch: {obs['url']} != {nodes[i].url}")
            #     # else:
            #     #     print_first_diff_lines(obs["axtree_txt"], nodes[i].axtree_txt)
                    
            #     nodes[i].corrupted = True
            #     flag = False
            #     break
            # else:
            #     print(f"[SIMULATION] Action {action} successful.")
            #     print_first_diff_lines(obs["axtree_txt"], nodes[i].axtree_txt)
        
        if flag:
            print(f"[SIMULATION] Simulation successful.")
        
        while len(env.context.pages) > current_total_tabs:
            env.context.pages[-1].close()
            
        env.page = original_page  
        
        if current_active_tab >= len(env.context.pages):
            print(f"[SIMULATION] Invalid active tab index {current_active_tab}, skipping refocus.")
        else:
            print(f"[SIMULATION] Refocusing on tab {current_active_tab}.")
            _, _, _, _, _ = env.step(f"tab_focus({current_active_tab})")
            
        time.sleep(1)
        return flag