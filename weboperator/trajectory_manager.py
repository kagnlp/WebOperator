def _pre_action_trajectory(node):
        # Traverse from current node to root and collect actions and observations
        trajectory = []

        current_node = node
        tmp_action = None
        tmp_action_error = None
        tmp_obs_description = None
        tmp_checklist_completion = None
        while current_node is not None:
            trajectory.append({
                "url": current_node.url,
                "open_pages_urls": current_node.open_pages_urls,
                "open_pages_titles": current_node.open_pages_titles,
                "active_page_index": current_node.active_page_index,
                "axtree_txt": current_node.axtree_txt,
                "page_too_long": current_node.page_too_long,
                "axtree_object": current_node.axtree_object,
                "action": tmp_action,
                "action_error": tmp_action_error,
                "obs_description": tmp_obs_description,
                "checklist_completion": tmp_checklist_completion,
            })
            tmp_action = current_node.last_action
            tmp_action_error = current_node.last_action_error
            tmp_obs_description = current_node.last_obs_description
            tmp_checklist_completion = current_node.checklist_completion
            current_node = current_node.parent

        # Reverse the trajectory to have it from root to current node
        return list(reversed(trajectory))
    
def _post_action_trajectory(node):
        # Traverse from current node to root and collect actions and observations
        trajectory = []

        current_node = node.parent
        tmp_action = node.last_action
        tmp_action_error = "No Error (This action is not yet executed)"
        tmp_obs_description = node.last_obs_description
        tmp_checklist_completion = node.checklist_completion
        while current_node is not None:
            trajectory.append({
                "open_pages_urls": current_node.open_pages_urls,
                "open_pages_titles": current_node.open_pages_titles,
                "active_page_index": current_node.active_page_index,
                "axtree_txt": current_node.axtree_txt,
                "page_too_long": current_node.page_too_long,
                "axtree_object": current_node.axtree_object,
                "action": tmp_action,
                "action_error": tmp_action_error,
                "obs_description": tmp_obs_description,
                "checklist_completion": tmp_checklist_completion,
            })
            tmp_action = current_node.last_action
            tmp_action_error = current_node.last_action_error
            tmp_obs_description = current_node.last_obs_description
            tmp_checklist_completion = current_node.checklist_completion
            current_node = current_node.parent

        # Reverse the trajectory to have it from root to current node
        return list(reversed(trajectory))

def _get_full_trajectory(node):
        # Traverse from the given node to root and collect actions and observations
        trajectory = []
        current_node = node
        while current_node is not None:
            trajectory.append({
                "open_pages_urls": current_node.open_pages_urls,
                "open_pages_titles": current_node.open_pages_titles,
                "active_page_index": current_node.active_page_index,
                "axtree_txt": current_node.axtree_txt,
                "page_too_long": current_node.page_too_long,
                "action": current_node.last_action,
                "action_error": current_node.last_action_error,
                "obs_description": current_node.last_obs_description,
            })
            current_node = current_node.parent
        return list(reversed(trajectory))

def has_safe_anchestor(end_node):
    if end_node.parent is None:
        return True
    if len(end_node.parent.open_pages_urls) == 1: # Single Tab Case
        last = end_node.last_action
        if last["type"] == "goto":
            return True

    target_node = end_node.parent
    curr = target_node
    
    while curr is not None and curr.parent is not None:
        if curr.destructive:
            return False
        if curr.parent is not None and curr.parent.url != curr.url:
            if curr.refresh_loses_state:
                # We can't stop here. Need to go to parent.
                pass
            else:
                break
        curr = curr.parent
    
    return True