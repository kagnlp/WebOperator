from .web_state_node import WebStateNode
from .observation_processor import ObservationProcessor
from .axtree_utils import is_diff_axtree_obj_by_bid
from .action_selector import ActionSelector
from .action_processor import ActionProcessor
from .url_simulator import URLSimulator
# from browsergym.experiments.loop import get_env
import time
import gymnasium as gym
import logging
logger = logging.getLogger(__name__)

class BacktrackManager:
    simulation_enabled = True
    destruction_aware = True
    
    def __init__(self):
        self.pending_actions = []
        self.pending_nodes = []
        self.is_backtracking = False
        
    @classmethod
    def configure(cls, simulation_enabled: bool, destruction_aware: bool):
        cls.simulation_enabled = simulation_enabled
        ActionSelector.destruction_aware_backtracking = destruction_aware
        cls.destruction_aware = destruction_aware

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
            reset_actions.append(f"new_tab('{anchestor_tab_urls[i]}')")
            # reset_actions.append(f"goto('{anchestor_tab_urls[i]}')")

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
    def backtrack(cls, start_node, end_node: WebStateNode, env: gym.Env):
        nodes = []
        def append_action(curr_node):
            if curr_node.last_action is not None and not curr_node.last_action["code"].startswith("note"):
                nodes.append(curr_node)
        
        if len(end_node.parent.open_pages_urls) == 1: # Single Tab Case
            last = end_node.last_action
            if last["type"] == "goto":
                if(cls._safe_backtrack([end_node], start_node, end_node.parent, env, False)):
                    return True, 1
                
        go_back_count = 0
        go_forward_count = 0
        if end_node.last_action is not None:
            append_action(end_node)
            if end_node.last_action["type"] == "go_back":
                go_back_count += 1
            if end_node.last_action["type"] != "go_forward":
                # end_node = end_node.parent.parent
                go_forward_count += 1
        
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
            append_action(curr)
            curr = curr.parent
        
        while curr is not None:
            if cls.destruction_aware and curr.destructive: # Search upwards until destructive action
                break
            if curr.parent is None or (curr.parent.url != curr.url and not curr.refresh_loses_state and cls.destruction_aware):
                # Safe to stop here?
                if curr.parent is not None and go_back_count > 0:
                    append_action(curr)
                    if curr.last_action["type"] == "go_back":
                        go_back_count += 1
                    else:
                        go_back_count -= 1
                elif curr.parent is not None and curr.last_action["type"] == "go_back":
                    append_action(curr)
                    go_back_count += 1
                else:
                    logger.debug(f"Found a likely anchestor. Checking if safe to stop here: {curr.url}")
                    forward_nodes = list(reversed(nodes))

                    if cls._safe_backtrack(forward_nodes, start_node, curr, env, end_node.last_action["type"] != "stop" and cls.simulation_enabled):
                        return True, len(forward_nodes)
                    else: # Single Simultation
                        break

            else:
                append_action(curr)
                if curr.last_action["type"] == "go_back":
                    go_back_count += 1
            curr = curr.parent
        
        return False, 0
    
    @classmethod
    def get_backtrack_actions(cls, start_node, end_node: WebStateNode, env: gym.Env):
        # if is_same_node(end_node.parent, start_node): 
        #     return True
        # logger.debug("[Backtrack Manager] Finding backtrack path", end_node.url, start_node.url)
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
        final_actions = None
        final_nodes = None
        while curr is not None:
            if cls.destruction_aware and curr.destructive: # Search upwards until destructive action
                break
            if curr.parent is None or (curr.parent.url != curr.url and not curr.refresh_loses_state and cls.destruction_aware):
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
                    logger.debug(f"Found a likely anchestor. Checking if safe to stop here: {curr.url}")
                    reset_actions, reset_nodes = cls.get_reset_steps(source_n_tabs, curr.open_pages_urls)
                    final_actions = reset_actions+list(reversed(actions))
                    final_nodes = reset_nodes+list(reversed(nodes))

                    if end_node.last_action["type"] == "stop" or cls._simulate(final_actions, final_nodes, start_node.active_page_index, env):
                        return final_actions, final_nodes
                    else: # Single Simultation
                        break
                        
                    if curr.parent is not None: # Need to go further up
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
        
        # if final_actions is not None and final_nodes is not None:
        #     if not cls.simulation_enabled or cls._simulate(final_actions, final_nodes, start_node.active_page_index, env):
        #         return final_actions, final_nodes
        return None, None

    @classmethod
    def _simulate(cls, actions, nodes, active_page_index, env: gym.Env):
        if not cls.simulation_enabled:
            return True
        # actions = list(reversed(actions))
        # nodes = list(reversed(nodes))
        # red color print
        # print(actions)
        logger.debug(f"[SIMULATION] Simulating {len(actions)} actions for backtrack verification.")
        
        for action in actions:
            logger.debug(f"\033[91m{action}\033[0m")
        
        # First action is always goto
        if len(actions) <= 1:
            logger.debug(f"[SIMULATION] Only {len(actions)} action(s), skipping simulation.")
            return True
        
        for node in nodes:
            if node is not None and node.corrupted:
                logger.debug(f"[SIMULATION] Node in the path already marked as corrupted, skipping simulation.")
                return False
            
        bid_flag = False
        for node in nodes:
            if node is not None and node.last_action is not None and node.last_action["type"] in ("click", "fill", "select_option"):
                bid_flag = True
                break
        
        if not bid_flag:
            logger.debug(f"[SIMULATION] No bid-based actions in the path, skipping simulation.")
            return True
            
        for i in range(1, len(actions) - 1):       
            if nodes[i] is not None and nodes[i].parent is not None:
                if nodes[i].destructive: # and nodes[i].last_action_error == "":
                    logger.debug(f"[SIMULATION] Destructive action found at index {i}: {nodes[i].last_action['code']}")
                    return False

        current_total_tabs = len(env.context.pages)
        current_active_tab = active_page_index
        original_page = env.page
        # create same number of tabs
        for _ in range(current_total_tabs):
            env.context.new_page()
        
        env.page = env.context.pages[current_total_tabs + current_active_tab]
        
        # what if one of the actions is related to tabs?
        flag = True
        obs = None
        
        for i, action in enumerate(actions):
            if nodes[i] is not None:
                formatted_action = nodes[i].last_action
                if formatted_action is not None and formatted_action["type"] == "tab_focus":
                    relative_tab_index = formatted_action["args"]["index"]
                    # relative_tab_index = int(re.search(r'tab_focus\((\d+)\)', action).group(1))
                    abs_tab_index = current_total_tabs + relative_tab_index
                    action = f"tab_focus({abs_tab_index})"
                    # print(f"[SIMULATION] Converted tab_focus action to absolute index: {action}")

                if obs is not None and nodes[i].last_action is not None and formatted_action["type"] in ("click", "fill", "select_option"): # bid-based actions
                    bid = formatted_action["args"]["bid"]
                    if is_diff_axtree_obj_by_bid(obs["axtree_object"], nodes[i].parent.axtree_object, bid):
                        logger.debug(f"[SIMULATION] Simulation mismatch at action {action} due to bid {bid}.")
                        flag = False
                        break
                    else:
                        logger.debug(f"[SIMULATION] Bid {bid} matches for action {action}.")
            
                    action = ActionProcessor.postprocess(formatted_action, obs["axtree_object"])["code"]
                
            if i >= len(actions) - 1:
                break
            
            obs, _, _, _, _ = env.step(action)
            obs = ObservationProcessor.process_obs(obs)
            
            if obs["last_action_error"] != "":
                logger.error(f"[SIMULATION] Simulation error at action {action}: {obs['last_action_error']}")
        
        if flag:
            logger.debug(f"[SIMULATION] Simulation successful.")
        else:
            logger.debug(f"[SIMULATION] Simulation failed.")
            # Close new tabs
            while len(env.context.pages) > current_total_tabs:
                env.context.pages[-1].close()
            
        env.page = original_page  
        
        if current_active_tab >= len(env.context.pages):
            logger.debug(f"[SIMULATION] Invalid active tab index {current_active_tab}, skipping refocus.")
        else:
            logger.debug(f"[SIMULATION] Refocusing on tab {current_active_tab}.")
            _, _, _, _, _ = env.step(f"tab_focus({current_active_tab})")
            
        time.sleep(1)
        return flag
    
    @classmethod
    def _safe_backtrack(cls, nodes, source_node, rollback_node, env: gym.Env, simulation_enabled):
        # if not cls.simulation_enabled:
        #     return True

        logger.debug(f"[SIMULATION] Simulating {len(nodes)} actions for backtrack verification.")
        
        for node in nodes:
            logger.debug(f"\033[91m{node.last_action['code']}\033[0m")
        
        current_total_tabs = len(source_node.open_pages_urls)
        current_active_tab = source_node.active_page_index
        
        new_total_tabs = len(rollback_node.open_pages_urls)
        new_active_tab = rollback_node.active_page_index # Should be always 0
        
        original_page = env.page
        # create same number of tabs
        for i in range(new_total_tabs):
            page = env.context.new_page()
            URLSimulator.safe_goto(page, rollback_node.open_pages_urls[i])
            
        flag = True
        # env.page = env.context.pages[current_total_tabs + new_active_tab]
        obs, _, _, _, _ = env.step(f"tab_focus({current_total_tabs + new_active_tab})")
        obs = ObservationProcessor.process_obs(obs)
        last_ax = obs["axtree_object"]

        # Ignore the last action during forward execution
        for i, node in enumerate(nodes):
            formatted_action = node.last_action
            action = formatted_action["code"]
            if formatted_action is not None and formatted_action["type"] == "tab_focus":
                relative_tab_index = formatted_action["args"]["index"]
                abs_tab_index = current_total_tabs + relative_tab_index
                action = f"tab_focus({abs_tab_index})"

            if formatted_action is not None and formatted_action["type"] in ("click", "fill", "select_option"): # bid-based actions
                if simulation_enabled: # Verify bid-based actions only when simulation is enabled
                    bid = formatted_action["args"]["bid"]
                    if is_diff_axtree_obj_by_bid(last_ax, node.parent.axtree_object, bid):
                        logger.debug(f"[SIMULATION] Simulation mismatch at action {action} due to bid {bid}.")
                        flag = False
                        break
                    else:
                        logger.debug(f"[SIMULATION] Bid {bid} matches for action {action}.")
        
                action = ActionProcessor.postprocess(formatted_action, last_ax)["code"]
                
            if i >= len(nodes) - 1:
                break
            
            print(f"# Action: \033[91m[B] \033[92m{action}\033[0m")
            
            obs, _, _, _, _ = env.step(action)
            obs = ObservationProcessor.process_obs(obs)
            last_ax = obs["axtree_object"]
            
            if obs["last_action_error"] != "":
                logger.error(f"[SIMULATION] Simulation error at action {action}: {obs['last_action_error']}")
        
        if flag:
            logger.debug(f"[SIMULATION] Simulation successful.")
            # input("Press Enter to continue...")
            # Close old tabs
            for _ in range(current_total_tabs):
                env.context.pages[0].close()
            
        else:
            logger.debug(f"[SIMULATION] Simulation failed.")
            # input("Press Enter to continue...")
            # Close new tabs
            while len(env.context.pages) > current_total_tabs:
                env.context.pages[-1].close()
            
            env.page = original_page 
        
            if current_active_tab >= len(env.context.pages):
                logger.debug(f"[SIMULATION] Invalid active tab index {current_active_tab}, skipping refocus.")
            else:
                logger.debug(f"[SIMULATION] Refocusing on tab {current_active_tab}.")
                _, _, _, _, _ = env.step(f"tab_focus({current_active_tab})")
            
        time.sleep(1)
        return flag