import heapq
from .action_analyzer import is_terminating, is_destructive, is_repeated_action
from .trajectory_manager import has_safe_anchestor
import logging
logger = logging.getLogger(__name__)

class ActionQueueManager:
    def __init__(self, search_budget):
        self.initial_search_budget = search_budget
        self.search_budget = search_budget
        self.queue = []
    
    def reset(self):
        self.search_budget = self.initial_search_budget
        self.queue = []
            
    def clear(self):
        self.queue = []
        
    def reduce_budget(self):
        self.search_budget = max(1, self.search_budget - 1)
        
    def destruct(self):
        self.clear()
        self.reduce_budget()
        
    def can_delay_destruction(self):
       return (self.length() <= self.search_budget and self.get_destructive_actions_count() <= 1)
        
    def is_out_of_budget(self):
        return len(self.queue) > self.search_budget
    
    def push(self, priority, node):
        heapq.heappush(self.queue, (priority, node))

    def pop(self):
        return heapq.heappop(self.queue)
        
    def empty(self):
        return len(self.queue) == 0

    def length(self):
        return len(self.queue)
    
    def get_terminating_actions_count(self):
        return sum(1 for _, node in self.queue if is_terminating(node.last_action))
    
    def get_destructive_actions_count(self):
        return sum(1 for _, node in self.queue if is_destructive(node.parent, node.last_action))
    
    def reinsert_skipped_nodes(self, tmp_queue):
        for item in tmp_queue:
            heapq.heappush(self.queue, item)
    
    def get_best_terminating_action(self):
        tmp_queue = []
        best_solution = None
        best_priority = None
        while len(self.queue) > 0:
            priority, node = heapq.heappop(self.queue)
            if is_terminating(node.last_action):
                best_solution = node
                best_priority = priority
                break
            else:
                tmp_queue.append((priority, node))
        self.reinsert_skipped_nodes(tmp_queue)
        return best_priority, best_solution
    
    def minimize(self): # Should be called after choosing next best action
        print("Action Queue before minimization: ", len(self.queue))
        for priority, node in self.queue:
            print(f" - {node.last_action['code']} ({is_destructive(node.parent, node.last_action) if node.parent else False}) -> {priority}")
            
        last_len = len(self.queue)
        
        # Remove non-backtrackable nodes
        tmp_action_queue = []
        while len(self.queue) > 0:
            priority, tmp_node = heapq.heappop(self.queue)
            if has_safe_anchestor(tmp_node): # Backtrackable
                tmp_action_queue.append((priority, tmp_node))
            else:
                tmp_node.prune()
        self.queue = tmp_action_queue
        heapq.heapify(self.queue)
        
        # Remove, lower priority destructive, terminating and repeated nodes
        new_action_queue = []
        found_best_destructive_action = False
        found_best_terminating_action = False
        found_best_repeatable_action = False
        while len(self.queue) > 0:
            priority, tmp_node = heapq.heappop(self.queue)
            if is_terminating(tmp_node.last_action):
                if not found_best_terminating_action:
                    new_action_queue.append((priority, tmp_node))
                    found_best_terminating_action = True
                else:
                    tmp_node.prune()
            elif is_destructive(tmp_node.parent, tmp_node.last_action):
                if not found_best_destructive_action:
                    new_action_queue.append((priority, tmp_node))
                    found_best_destructive_action = True
                else:
                    tmp_node.prune()
            elif is_repeated_action(tmp_node):
                if not found_best_repeatable_action:
                    new_action_queue.append((priority, tmp_node))
                    found_best_repeatable_action = True
                else:
                    tmp_node.prune()
            else:
                new_action_queue.append((priority, tmp_node))
                
        self.queue = new_action_queue
        heapq.heapify(self.queue)

        new_action_queue = []
        # Now only non-destructive actions can be removed
        # If there are duplicate actions, keep the one with the highest priority (lowest score)
        if len(self.queue) > self.search_budget:
            count = len(self.queue)
            unique_actions = {}
            for priority, node in self.queue:
                action_code = node.last_action["code"] if node.last_action["type"] in ["click", "fill", "select_option"] else "others"
                # make a list of same action codes
                if unique_actions.get(action_code) is None:
                    unique_actions[action_code] = [(priority, node)]
                else:
                    unique_actions[action_code].append((priority, node))
            
            while count > self.search_budget:
                # remove the lowest priority duplicate action until len(self.queue) == self.search_budget
                # From each list don't remove the first one (highest priority)
                max_priority = -float('inf')
                max_action_code = None
                max_index = -1
                for action_code, nodes in unique_actions.items():
                    if len(nodes) > 1:
                        for i in range(1, len(nodes)):
                            priority, node = nodes[i]
                            if priority > max_priority:
                                max_priority = priority
                                max_action_code = action_code
                                max_index = i
                if max_action_code is not None and max_index != -1:
                    # Prune the node
                    unique_actions[max_action_code][max_index][1].prune()
                    unique_actions[max_action_code].pop(max_index)
                    count -= 1
                else:
                    break
        
            # Now for each action, only the highest priority node remains
            while count > self.search_budget:
                max_priority = -float('inf')
                max_action_code = None
                for action_code, nodes in unique_actions.items():
                    priority, node = nodes[0]
                    if priority > max_priority:
                        max_priority = priority
                        max_action_code = action_code
                if max_action_code is not None:
                    # Prune the node
                    unique_actions[max_action_code][0][1].prune()
                    unique_actions.pop(max_action_code)
                    count -= 1
                else:
                    break
                
            # Final rebuild of action queue
            for action_code, nodes in unique_actions.items():
                for priority, node in nodes:
                    new_action_queue.append((priority, node))
        
            self.queue = new_action_queue
            heapq.heapify(self.queue)

        if last_len != len(self.queue):
            logger.debug(f"Reduced action queue from {last_len} to {len(self.queue)}")

        print("Action Queue after minimization: ", len(self.queue))
        for priority, node in self.queue:
            print(f" - {node.last_action['code']} ({is_destructive(node.parent, node.last_action) if node.parent else False}) -> {priority}")