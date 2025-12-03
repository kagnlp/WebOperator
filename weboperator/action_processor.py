from webshepherd.webprm.prompts import action
from .utils import to_ascii
import logging
from .trajectory_manager import _post_action_trajectory
from .action_analyzer import is_repeated_action, is_terminating
from tqdm import tqdm
from .web_state_node import WebStateNode
from weboperator.webprm import WebPRM
logger = logging.getLogger(__name__)

class ActionProcessor:
    prune_low_terminating = False
    merge_strategy = "sum"  # or "sum"
    
    @classmethod
    def configure(cls, prune_low_terminating, merge_strategy):
        cls.prune_low_terminating = prune_low_terminating
        cls.merge_strategy = merge_strategy
    
    @classmethod
    def action_to_python_code(cls, action: dict) -> str:
        # Arg dict order is not guaranteed
        return f"{action['type']}({', '.join([f'{repr(v)}' for k, v in action['args'].items()])})"
    
    @classmethod
    def postprocess(cls, action, axtree_object):
        import copy
        action = copy.deepcopy(action)
        flag = False
        if action["type"] in ["click", "fill", "select_option"]: # bid based
            # Find the actual browsergym_id from self.tree_node.axtree_object
            for node in axtree_object["nodes"]:
                if node.get('browsergym_id') is not None and int(action["args"]["bid"]) == node.get('weboperator_id'):
                    action["args"]["bid"] = node.get('browsergym_id')
                    flag = True
                    # action["code"] = cls.action_to_python_code(action) # Done later
                    break
                
            # if not flag:
            #     logger.warning(f"bid {action['args']['bid']} not found in current axtree nodes.")
                # raise ValueError(f"bid {action['args']['bid']} not found in current axtree nodes.")
        
        if action["type"] == "stop":
            action["type"] = "send_msg_to_user"
            
        if action["type"]:
            action["code"] = cls.action_to_python_code(action)

        return action
    
    @classmethod
    def convert_action_to_nodes(cls, actions: str, curr_node):
        """Get unique nodes from the current observation and actions."""
        tree_nodes = []
        if len(curr_node.children) > 0 and len(actions) == 0:
            # checkpoint
            pass
        else:
            # Generate children nodes
            for (formatted_action, last_obs_str) in actions:
                new_node = WebStateNode(
                    last_action=formatted_action,
                    last_obs_description=last_obs_str,
                    parent=curr_node,
                )
                curr_node.add_child(new_node)

        # Filter erroneous terminating actions
        n_generated_actions = len(curr_node.children)
        
        t_count = 0
        for node in curr_node.children:
            if is_terminating(node.last_action):
                t_count += 1
                
        tree_nodes = []
        for node in curr_node.children:
            if cls.prune_low_terminating and (t_count <= len(curr_node.children)/3.0 or (t_count == 1 and t_count != n_generated_actions)) and is_terminating(node.last_action):
                print("Pruning the only terminating action in existing children: ", node.last_action["code"])
                node.prune()
            else:
                tree_nodes.append(node)
        return tree_nodes

    @classmethod
    def evaluate_actions(cls, action_nodes, goal, checklist):
        action_list = []  # Array of (score, Node) pair
        for new_node in tqdm(action_nodes, desc="Evaluating nodes"):
            if new_node.value is None:
                new_node.value, new_node.checklist_completion = WebPRM.evaluate(_post_action_trajectory(new_node), goal, checklist)
                logger.debug(f"Node evaluated: {new_node.last_action['code']} -> {new_node.value}")
            action_list.append((new_node.value, new_node))
        return action_list

    @classmethod
    def merge_actions(cls, action_list):
        if cls.merge_strategy == "none":
            return action_list
        
        merged = {}
        for score, node in action_list:
            act = node.last_action
            node.frequency = 1
            act_type = act["type"]
            code = act.get("code")
            
            if act_type == "fill":
                key = (
                    act_type,
                    act["args"]["bid"],
                    act["args"]["press_enter_after"],
                    to_ascii(act["args"]["value"].lower()),
                )
            elif act_type in ("stop", "note"):
                key = (act_type,)
            else:
                key = (code,)
                
            if key not in merged:
                merged[key] = (score, score, node)
            else:
                s, ts, n = merged[key]
                if score > s:
                    n.prune()
                    node.frequency += n.frequency
                    n.frequency = 0
                    merged_score = score if cls.merge_strategy == "max" else ts + score
                    merged[key] = (score, merged_score, node)
                else:
                    node.prune()
                    n.frequency += node.frequency
                    node.frequency = 0
                    merged_score = ts if cls.merge_strategy == "max" else ts + score
                    merged[key] = (s, merged_score, n)
        
        final_action_list = [(ts, n) for _, ts, n in merged.values()]
        
        logger.debug(f"Action list reduced from {len(action_list)} to {len(final_action_list)}")
        return final_action_list