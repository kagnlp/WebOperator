from .web_state_node import WebStateNode
from .action_analyzer import is_destructive, is_repeated_action, is_terminating
import logging
from .trajectory_manager import has_safe_anchestor
from .action_queue_manager import ActionQueueManager
from .action_processor import ActionProcessor
logger = logging.getLogger(__name__)

class ActionSelector:
    selection_strategy:str = "action-aware"
    n_candidates:int = 2
    max_steps:int = 20
    selection_scope:str = "global"  # "local" or "global"
    termination_strategy:str = "all_agree"  # "all_agree" or "min_candidates"
    max_depth: int = 20
    search_budget: int = 4
    destruction_aware_backtracking: bool = True
    
    def __init__(self):
        self.action_queue_manager = ActionQueueManager(search_budget=self.search_budget)
        self.n_prune = 0
        self.n_destruct = 0
        self.discovered_solutions = 0
        
    def reset(self):
        self.action_queue_manager.reset()
        self.n_destruct = 0
        self.n_prune = 0
        self.discovered_solutions = 0
        
    @classmethod
    def configure(cls, selection_strategy="action-aware", n_candidates=2, max_steps=20, selection_scope="global", termination_strategy="all_agree", max_depth=20, search_budget=4, destruction_aware_backtracking=True):
        cls.selection_strategy = selection_strategy
        cls.n_candidates = n_candidates
        cls.max_steps = max_steps
        cls.selection_scope = selection_scope
        cls.termination_strategy = termination_strategy
        cls.max_depth = max_depth
        cls.search_budget = search_budget
        cls.destruction_aware_backtracking = destruction_aware_backtracking
        
    def _make_terminating_action(self, message: str, curr_node):
        node = WebStateNode(
            last_action={
                "code": f"stop('{message}')",
                "thought": message,
                "type": "stop",
                "args": {"text": message},
                "selected_element": None,
            },
            last_obs_description="Couldn't generate description.",
            parent=curr_node
        )
        node.value = 0.0
        return node
    
    def _handle_max_steps(self, n_expanded, curr_node):
        if n_expanded < self.max_steps:
            return None
        # if n_expanded >= 0.8 * self.max_steps:
        #     ActionProcessor.prune_low_terminating = False

        if self.action_queue_manager.get_terminating_actions_count() > 0:  # More than one candidate solution
            _, best_node = self.action_queue_manager.get_best_terminating_action()
            print(f"Best candidate solution found [Max Steps Exceeded]: {best_node.last_action['code']}")
            return best_node
            
        logger.error("Max steps reached but no candidate solution found.")
        return self._make_terminating_action("N/A. Agent failed to find a valid solution.", curr_node)
        
    def _handle_no_best_node(self, curr_node):
        return self._make_terminating_action("N/A. Agent failed to find a valid action.", curr_node)

    def _calculate_frequency(self, actions):
        count = 0
        for _, node in actions:
            count += node.frequency
        return count
    
    def add_actions(self, curr_obs, actions, check_budget=True):
        if self.destruction_aware_backtracking and curr_obs.destructive:
            self.action_queue_manager.destruct()
            self.n_destruct += 1
            print(f"  \033[91mDestructive action detected! Total so far: {self.n_destruct}\033[0m")

        if check_budget and self.selection_strategy == "action-aware" and self.action_queue_manager.is_out_of_budget():
            self.action_queue_manager.minimize()
            self.n_prune += 1
            print(f"Queue size {self.action_queue_manager.length()}. Tree pruned, best node found.")
        else:
            print(f"Queue size {self.action_queue_manager.length()}. Adding new actions...")
            
        if self.selection_scope == "local":
            self.action_queue_manager.clear()

        if self.termination_strategy == "all_agree" and self._calculate_frequency(actions) >= 2 and len(actions) == 1 and is_terminating(actions[0][1].last_action):
            self.action_queue_manager.clear()

        for score, new_node in actions:
            if is_terminating(new_node.last_action):
                self.discovered_solutions += 1
            elif new_node.level >= self.max_depth:
                print(f"Non-terminating Node {new_node.last_action['code']} exceeded max depth {self.max_depth}, skipping...")
                new_node.prune()
                continue
            self.action_queue_manager.push(-score, new_node)
                
    def select_action(self, curr_obs, n_expanded):
        # 1. Handle max-step pruning
        node = self._handle_max_steps(n_expanded, curr_obs)
        if node: return node
        
        # 2. Selection strategy
        node = self._apply_hr_selection_strategy()
        if node: return node
            
        # 3. Handle terminating actions (safe ancestors)
        node = self._handle_terminating_actions()
        if node: return node
                
        # 4. Handle pruning based on search budget/destruction count
        node = self._handle_pruning_conditions(n_expanded)
        if node: return node
            
        # 5. Pick best normal (non-destructive/non-repeated) node
        node = self._pick_normal_best_node(n_expanded)
        if node: return node

        # 6. Handle fallback destructive or terminating cases
        node = self._handle_fallback_cases()
        if node: return node

        return self._handle_no_best_node(curr_obs)
    
    def _apply_hr_selection_strategy(self):
        if self.selection_strategy != "highest-reward":
            return None

        # Highest reward strategy
        # - Try to ignore repeated actions first
        tmp_queue = []
        best_node = None
        
        while self.action_queue_manager.length() > 0:
            priority, tmp_node = self.action_queue_manager.pop()
            if is_repeated_action(tmp_node):
                tmp_queue.append((priority, tmp_node))
            else:
                best_node = tmp_node
                break
            
        self.action_queue_manager.reinsert_skipped_nodes(tmp_queue)
            
        # If all actions are repeated actions, just take the best one
        if best_node is None and self.action_queue_manager.length() > 0:
            _, best_node = self.action_queue_manager.pop()

        return best_node
    
    def _handle_terminating_actions(self):
        if self.action_queue_manager.get_terminating_actions_count() == 0 or (self.n_destruct == 0 and self.n_prune == 0):
            return None
        
        # print("## No non-destructive best node found, but there are terminating actions in the queue.")
        while self.action_queue_manager.get_terminating_actions_count() > 0:
            priority, tmp_node = self.action_queue_manager.get_best_terminating_action()
            if has_safe_anchestor(tmp_node):
                # Just push back the terminating action to the queue
                print("No destructive action in the backtrack path of the best solution.")
                self.action_queue_manager.push(priority, tmp_node)
                break
            else:
                print(f"Best candidate solution found [Need to take it now or it will be lost]: {tmp_node.last_action['code']}")
                return tmp_node

        return None
    
    def _handle_pruning_conditions(self, n_expanded):
        if self.action_queue_manager.can_delay_destruction():  # Prune Tree
            return None
        
        tmp_queue = []
        best_node = None
        while self.action_queue_manager.length() > 0:
            priority, tmp_node = self.action_queue_manager.pop()
            
            if is_terminating(tmp_node.last_action) and self.discovered_solutions < self.n_candidates and n_expanded < 0.80 * self.max_steps and self.n_prune < 3: # Delay
                tmp_queue.append((priority, tmp_node))
                
            elif is_destructive(tmp_node.parent, tmp_node.last_action): # Select
                self.action_queue_manager.reinsert_skipped_nodes(tmp_queue)
                tmp_queue = []

                # First, try to terminate
                if self.action_queue_manager.get_terminating_actions_count() > 0 and (self.n_destruct > 0 or self.n_prune > 0):  # More than one candidate solution
                    _, best_node = self.action_queue_manager.get_best_terminating_action()
                    print(f"Best candidate solution found [Threshold]: {best_node.last_action['code']}")
                    return best_node

                return tmp_node
            
            else: # Select
                return tmp_node
  
        # Reinsert the skipped nodes back into the action queue
        self.action_queue_manager.reinsert_skipped_nodes(tmp_queue)
                
        return best_node

    def _pick_normal_best_node(self, n_expanded):
        tmp_queue = []
        best_node = None
        while self.action_queue_manager.length() > 0:
            priority, tmp_node = self.action_queue_manager.pop()
            if is_destructive(tmp_node.parent, tmp_node.last_action) or is_repeated_action(tmp_node): # Delay destructive or repeated actions
                tmp_queue.append((priority, tmp_node))

            elif is_terminating(tmp_node.last_action) and ((self.discovered_solutions < self.n_candidates and n_expanded < 0.80 * self.max_steps) or (self.action_queue_manager.get_destructive_actions_count() > 0 and self.n_destruct == 0) or (self.action_queue_manager.get_destructive_actions_count() == 0 and self.n_prune == 0)): # Delay terminating actions
                tmp_queue.append((priority, tmp_node))
                
            else: # Select
                best_node = tmp_node
                if is_terminating(best_node.last_action):
                    print(f"Best candidate solution found: {best_node.last_action['code']}")
                break
        
        # Reinsert the skipped nodes back into the action queue
        self.action_queue_manager.reinsert_skipped_nodes(tmp_queue)
        
        return best_node
    
    def _handle_fallback_cases(self):
        logger.warning("No non-destructive best node found in the action queue.")
        if self.action_queue_manager.get_terminating_actions_count() > 0 and (self.n_destruct > 0 or self.n_prune > 0): # TODO: Instead of blindly rejecting the first solution, set a threshold
            _, best_node = self.action_queue_manager.get_best_terminating_action()
            print(f"Best candidate solution found: {best_node.last_action['code']}")
            return best_node
        
        tmp_queue = []
        best_node = None
        while self.action_queue_manager.length() > 0:
            priority, tmp_node = self.action_queue_manager.pop()
            # NEW: May be removed later
            if is_terminating(tmp_node.last_action) and self.n_destruct == 0:
                print("Best node is a terminating action, but no destructive actions taken yet, rejecting...")
                tmp_queue.append((priority, tmp_node))
                continue

            print("Taking destructive action...")
            # self._after_destructive_action()
            best_node = tmp_node
            break
            
        # Reinsert the skipped nodes back into the action queue
        self.action_queue_manager.reinsert_skipped_nodes(tmp_queue)
            
        if best_node is None:
            logger.warning("No non-destructive and non-terminating best node found in the action queue.")
            if self.action_queue_manager.get_terminating_actions_count() > 0:
                _, best_node = self.action_queue_manager.get_best_terminating_action()
                print(f"Best candidate solution found: {best_node.last_action['code']}")

        return best_node