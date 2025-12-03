import copy
from browsergym.utils.obs import IGNORED_AXTREE_ROLES, IGNORED_AXTREE_PROPERTIES, _process_bid, INCLUDE_BID_ROLES
from browsergym.utils.obs import flatten_axtree_to_str
import re
import os
import logging
logger = logging.getLogger(__name__)

def is_alert_available(axtree_object):
    for node in axtree_object["nodes"]:
        if node.get("role", {}).get("value") == "alert":
            return True
    
    # print("No alert found")
    return False
    
def print_first_diff_lines(text1, text2):
    """Print the first line where two texts differ."""
    lines1 = text1.splitlines()
    lines2 = text2.splitlines()
    for i, (line1, line2) in enumerate(zip(lines1, lines2)):
        if line1 != line2:
            print(f"Difference at line {i+1}:\nText1: {line1}\nText2: {line2}")
            return
    if len(lines1) != len(lines2):
        print(f"Difference in number of lines: Text1 has {len(lines1)} lines, Text2 has {len(lines2)} lines.")
    # else:
    #     print("No differences found.")
    
def get_elem_by_bid(axtree, bid):
    """Get the element from the AXTree by its bid."""
    if axtree is None:
        return None
    for node in (axtree["nodes"]):
        # print(
        #     f"Checking node with browsergym_id: {node.get('browsergym_id', None)}")
        tmp_bid = node.get("weboperator_id", None)
        #  and int(tmp_bid) == int(bid)
        if tmp_bid is not None and tmp_bid == int(bid):
            return node
    return None

def find_option_in_axtree(axtree, bid, option_str):
    if axtree is None:
        return None
    node_id_to_idx = {node["nodeId"]: idx for idx, node in enumerate(axtree["nodes"])}
    def dfs(node_idx: int) -> dict:
        node = axtree["nodes"][node_idx]
        node_name = ""
        
        if "name" in node:
            node_name = node["name"]["value"]
            if node_name.strip() == option_str:
                return node

        for child_node_id in node["childIds"]:
            if child_node_id not in node_id_to_idx or child_node_id == node["nodeId"]:
                continue
            result = dfs(node_id_to_idx[child_node_id])
            if result is not None:
                return result

        return None
            
    for node in (axtree["nodes"]):
        tmp_bid = node.get("weboperator_id", None)
        if tmp_bid is not None and tmp_bid == int(bid):
            # Now find the option_str in its subtree
            return dfs(node_id_to_idx[node["nodeId"]])
    return None

def is_full_axtree_too_long(axtree_txt) -> bool:
    # return len(axtree_txt) >= 80000
    return len(axtree_txt) >= int(os.environ["MAX_AXTREE_SIZE"])

def has_axtree_changed(curr_obs) -> bool:
    prev_axtree_txt = curr_obs.parent.axtree_txt if curr_obs.parent is not None else ""
    prev_axtree_obj = curr_obs.parent.axtree_object if curr_obs.parent is not None else {}
    prev_url = curr_obs.parent.url if curr_obs.parent is not None else ""
    if prev_url != curr_obs.url or (is_diff_axtree(prev_axtree_txt, curr_obs.axtree_txt) and is_diff_axtree_obj(prev_axtree_obj, curr_obs.axtree_object)):
        return True
    return False
  
def normalize_ax_node(node) -> dict:
    """Return only the parts of a node that affect its string representation."""
    extra_properties: dict = None,
    with_visible: bool = False,
    with_clickable: bool = False,
    with_center_coords: bool = False,
    with_bounding_box_coords: bool = False,
    with_som: bool = False,
    skip_generic: bool = True,
    filter_visible_only: bool = False,
    filter_with_bid_only: bool = False,
    filter_som_only: bool = False,
    coord_decimals: int = 0,
    ignored_roles=IGNORED_AXTREE_ROLES,
    ignored_properties=IGNORED_AXTREE_PROPERTIES,
    # remove_empty_strong: bool = True,
    remove_redundant_static_text: bool = True,
    hide_bid_if_invisible: bool = False,
    hide_all_children: bool = False,
    hide_all_bids: bool = False,
    
    node_role = node["role"]["value"]
    node_name = ""
    bid = node.get("browsergym_id")
    skip_node = False
    node_value = None
    
    if node_role in ignored_roles:
        return None
    elif "name" not in node:
        return None
    else:
        node_name = node["name"]["value"]
        if "value" in node and "value" in node["value"]:
            node_value = node["value"]["value"]
        else:
            node_value = None

        # extract bid
        bid = node.get("browsergym_id", None)
        wo_id = node.get("weboperator_id", None)
        
        attributes = []
        for property in node.get("properties", []):
            if not "value" in property:
                continue
            if not "value" in property["value"]:
                continue

            prop_name = property["name"]
            prop_value = property["value"]["value"]

            if prop_name in ignored_properties + ("describedby",):
                continue
            elif prop_name in ("required", "focused", "atomic"):
                if prop_value:
                    attributes.append(prop_name)
            # elif prop_name in ("url"): # and node_role not in ("image")
            #     url = prop_value
            #     parts = urlsplit(url)
            #     base_url = urlunsplit((parts.scheme, parts.netloc, parts.path, '', ''))
            #     attributes.append(f"{prop_name}={repr(base_url)}")
            else:
                attributes.append(f"{prop_name}={repr(prop_value)}")
                
        if skip_generic and node_role == "generic" and not attributes:
            skip_node = True
        
        # if node_role == "StaticText":
        #     if parent_node_filtered:
        #         skip_node = True
        #     elif remove_redundant_static_text and node_name in parent_node_name:
        #         skip_node = True
        # else:
        filter_node, extra_attributes_to_print = _process_bid(
            bid,
            extra_properties=extra_properties,
            with_visible=with_visible,
            with_clickable=with_clickable,
            with_center_coords=with_center_coords,
            with_bounding_box_coords=with_bounding_box_coords,
            with_som=with_som,
            filter_visible_only=filter_visible_only,
            filter_with_bid_only=filter_with_bid_only,
            filter_som_only=filter_som_only,
            coord_decimals=coord_decimals,
        )
        
        # if either is True, skip the node
        skip_node = skip_node or filter_node
        
        # insert extra attributes before regular attributes
        attributes = extra_attributes_to_print + attributes
            
        if skip_node:
            return None
          
    return {
        "bid": wo_id if node_role in INCLUDE_BID_ROLES else None,
        "role": node_role,
        "name": node_name.strip() if node_role in ("StaticText", "heading", "button", "option", "link", "label", "LabelText", "checkbox", "radio", "section", "image", "cell", "row") else None,
        "value": node_value,
        "attributes": ", ".join([""] + attributes),  # make hashable
    }
    

def axtree_equal(ax1, ax2):
    idmap1 = {n["nodeId"]: n for n in ax1["nodes"]}
    idmap2 = {n["nodeId"]: n for n in ax2["nodes"]}

    def is_skipped_subtree(n):
        children = sorted([cid for cid in n["childIds"] if cid in idmap1 and cid != n["nodeId"]])
        for cid in children:
            norm = normalize_ax_node(idmap1[cid])
            # if norm is not None:
            #     print(f"Child node {norm} is not skipped")
            if norm is not None or not is_skipped_subtree(idmap1[cid]):
                return False
        return True
    
    def is_skipped_node(n):
        norm = normalize_ax_node(n)
        if norm is None:
            # Check if all children are skipped subtrees
            return is_skipped_subtree(n)
        return False
        
    def compare_nodes(n1, n2):
        # print(f"Comparing nodes {n1.get('nodeId', None)} and {n2.get('nodeId', None)}")
        norm1 = normalize_ax_node(n1)
        norm2 = normalize_ax_node(n2)
        if norm1 is None and norm2 is None:
            # Continue comparing children
            pass
        elif norm1 is None or norm2 is None:
            print(f"Node mismatch: {norm1} vs {norm2}")
            return False
        elif norm1 != norm2:
            print(f"Node mismatch: {norm1} != {norm2}")
            return False

        # compare children (ignore order)
        children1 = sorted([cid for cid in n1["childIds"] if cid in idmap1 and cid != n1["nodeId"]])
        # remove skipped nodes from children1
        children1 = [cid for cid in children1 if not is_skipped_node(idmap1[cid])]
        
        children2 = sorted([cid for cid in n2["childIds"] if cid in idmap2 and cid != n2["nodeId"]])
        # remove skipped nodes from children2
        children2 = [cid for cid in children2 if not is_skipped_node(idmap2[cid])]
        
        # compare children (order matters)
        # children1 = ([cid for cid in n1["childIds"] if cid in idmap1 and cid != n1["nodeId"]])
        # children2 = ([cid for cid in n2["childIds"] if cid in idmap2 and cid != n2["nodeId"]])
        
        if len(children1) != len(children2):
            # print(f"Comparing nodes {norm1['bid'] if norm1 is not None else None} and {norm2['bid'] if norm2 is not None else None}")
            logger.debug(f"Different number of children: {len(children1)} vs {len(children2)}.")
            flag = is_skipped_subtree(n1) and is_skipped_subtree(n2)
            # print(flag)
            return flag

        return all(
            compare_nodes(idmap1[c1], idmap2[c2])
            for c1, c2 in zip(children1, children2)
        )

    if len(ax1["nodes"]) == 0 and len(ax2["nodes"]) == 0:
        return True
    if len(ax1["nodes"]) == 0 or len(ax2["nodes"]) == 0:
        return False
    return compare_nodes(ax1["nodes"][0], ax2["nodes"][0])

def is_diff_axtree_obj(ax1, ax2):
    """Check if two axtree objects are different."""
    return not axtree_equal(ax1, ax2)
    
    # if len(ax1["nodes"]) != len(ax2["nodes"]):
    #     print(f"Different number of nodes: {len(ax1['nodes'])} vs {len(ax2['nodes'])}")
    #     return True
    
    # for i, (node1, node2) in enumerate(zip(ax1["nodes"], ax2["nodes"])):
    #     # check if browsergym_id is same
    #     bid1 = node1.get("browsergym_id", None)
    #     bid2 = node2.get("browsergym_id", None)
    #     if bid1 != bid2:
    #         print(f"Difference at node {i+1}: browsergym_id {bid1} != {bid2}")
    #         return True
        
    #     # check if role is same
    #     node_role_1 = node1["role"]["value"]
    #     node_role_2 = node2["role"]["value"]
    #     if node_role_1 != node_role_2:
    #         print(f"Difference at node {i+1}: role {node_role_1} != {node_role_2}")
    #         return True
        

    # return False
    
def is_diff_axtree_obj_by_bid(ax1, ax2, bid):
    ax1 = copy.deepcopy(ax1)
    ax2 = copy.deepcopy(ax2)
    idmap1 = {n["nodeId"]: n for n in ax1["nodes"]}
    idmap2 = {n["nodeId"]: n for n in ax2["nodes"]}
    
    # Find the node with given bid, it's ancestors, descendants and siblings, and create two new axtree objects
    def find_node_and_related(node, bid, ancestors, idmap):
        if node.get("weboperator_id") == int(bid):
            last_aid = node["nodeId"]
            for aid in reversed(ancestors):
                new_child = []
                for cid in idmap[aid]["childIds"]:
                    if cid != last_aid:
                        idmap[cid]["childIds"] = []
                        
                    if cid == last_aid: # Ignore Siblings
                    # if ("name" in idmap[cid] and idmap[cid]["role"]["value"] not in IGNORED_AXTREE_ROLES) or cid == last_aid: # Ignore None leaves
                        new_child.append(cid)
                        
                idmap[aid]["childIds"] = new_child
                last_aid = aid
                        
            logger.debug(f"Found bid {bid} in ax tree")
            return True
        
        
        for cid in node["childIds"]:
            if cid in idmap:
                if find_node_and_related(idmap[cid], bid, ancestors + [node["nodeId"]], idmap):
                    return True

        return False

    flag1 = find_node_and_related(ax1["nodes"][0], bid, [], idmap1)
    flag2 = find_node_and_related(ax2["nodes"][0], bid, [], idmap2)
    
    # Collect only reachable nodes starting from root
    def collect_reachable(root, idmap):
        visited, stack = set(), [root]
        while stack:
            nid = stack.pop()
            node = idmap[nid]
            if nid not in visited:
                visited.add(nid)
                stack.extend(idmap[nid]["childIds"])
                # print(f"[{node.get('browsergym_id', None)}] - {node['role']['value']} - {node.get('name', {}).get('value', '')}")
        return visited

    if flag1:
        reachable1 = collect_reachable(ax1["nodes"][0]["nodeId"], idmap1)
    else:
        reachable1 = set()
        logger.debug(f"Bid {bid} not found in ax1")
        axtree_txt_1 = flatten_axtree_to_str(ax1)
        axtree_txt_2 = flatten_axtree_to_str(ax2)
        print("############### AXTree 1 ###############")
        print(axtree_txt_1)
        print("############### AXTree 2 ###############")
        print(axtree_txt_2)
        return True

    if flag2:
        reachable2 = collect_reachable(ax2["nodes"][0]["nodeId"], idmap2)
    else:
        reachable2 = set()
        logger.debug(f"Bid {bid} not found in ax2")

    new_ax1 = {"nodes": []}
    for node in ax1["nodes"]:
        if node["nodeId"] in reachable1:
            new_ax1["nodes"].append(node)

    new_ax2 = {"nodes": []}
    for node in ax2["nodes"]:
        if node["nodeId"] in reachable2:
            new_ax2["nodes"].append(node)

    logger.debug(f"Comparing axtree objects for bid {bid}")
    logger.debug(f"New axtree 1 has {len(new_ax1['nodes'])} nodes, New axtree 2 has {len(new_ax2['nodes'])} nodes")

    # print(new_ax1["nodes"][0])
    
    axtree_txt_1 = flatten_axtree_to_str(new_ax1)
    axtree_txt_2 = flatten_axtree_to_str(new_ax2)
    
    flag = axtree_txt_1 != axtree_txt_2 and is_diff_axtree_obj(new_ax1, new_ax2)
    
    if flag:
        print("############### AXTree 1 ###############")
        print(axtree_txt_1)
        print("############### AXTree 2 ###############")
        print(axtree_txt_2)
        
    return flag

def is_diff_axtree(ax1, ax2):
    """Check if two axtree texts are different."""
    lines1 = ax1.splitlines()
    lines2 = ax2.splitlines()
    if len(lines1) != len(lines2):
        print(f"Different number of lines: {len(lines1)} vs {len(lines2)}")
        return True
    for i, (line1, line2) in enumerate(zip(lines1, lines2)):
        # If the start of line is like [123], extract the number and compare
        line1 = line1.strip()
        line2 = line2.strip()
        match1 = re.match(r"\[([A-Za-z0-9_]+)\]", line1)
        match2 = re.match(r"\[([A-Za-z0-9_]+)\]", line2)
        if match1 and match2:
            if match1.group(1) != match2.group(1):
                print(f"Difference at line {i+1}: {line1} != {line2}")
                return True
            elif line1 != line2:
            # elif match1.group(1) != "xxx" and line1 != line2:
                print(f"Difference at line {i+1}: {line1} != {line2}")
                return True
        elif not match1 and not match2:
            if line1 != line2:
                print(f"Difference at line {i+1}: {line1} != {line2}")
                return True
        else:
            print(f"Difference at line {i+1}: {line1} != {line2}")
            return True
    
    return False

def is_recaptcha_in_axtree(axtree_txt) -> bool:
    """Check if reCAPTCHA is present in the axtree text."""
    if "reCAPTCHA" in axtree_txt and len(axtree_txt.split("\n")) < 20:
        return True
    if "Our systems have detected unusual traffic from your computer network. This page checks to see if it's really you sending the requests, and not a robot." in axtree_txt:
        return True
    return False

# def include_woid_in_axtree(AX_tree):
#     node_id_to_idx = {}
#     for idx, node in enumerate(AX_tree["nodes"]):
#         node_id_to_idx[node["nodeId"]] = idx
#     def dfs(node_idx: int, last_id: int) -> list[str]:
#         node = AX_tree["nodes"][node_idx]
#         if node.get("browsergym_id") is not None:
#             last_id += 1
#             node['weboperator_id'] = last_id
#         for child_node_id in node["childIds"]:
#             if child_node_id not in node_id_to_idx or child_node_id == node["nodeId"]:
#                 continue
#             # mark this to save some tokens
#             last_id = dfs(
#                 node_id_to_idx[child_node_id],
#                 last_id,
#             )
#         return last_id
#     dfs(0, 0)
#     return AX_tree

from collections import deque
def include_woid_in_axtree(AX_tree):
    node_id_to_idx = {node["nodeId"]: idx for idx, node in enumerate(AX_tree["nodes"])}

    queue = deque([0])  # start BFS from root (index 0)
    last_id = 0

    while queue:
        node_idx = queue.popleft()
        node = AX_tree["nodes"][node_idx]
        if node.get("browsergym_id") is not None and node["role"]["value"] in INCLUDE_BID_ROLES:
            last_id += 1
            node["weboperator_id"] = last_id

        # enqueue valid children
        for child_node_id in node["childIds"]:
            if child_node_id not in node_id_to_idx or child_node_id == node["nodeId"]:
                continue
            queue.append(node_id_to_idx[child_node_id])

    return AX_tree

def clean_axtree(
    AX_tree,
    extra_properties: dict = None,
    with_visible: bool = False,
    with_clickable: bool = False,
    with_center_coords: bool = False,
    with_bounding_box_coords: bool = False,
    with_som: bool = False,
    skip_generic: bool = True,
    filter_visible_only: bool = False,
    filter_with_bid_only: bool = False,
    filter_som_only: bool = False,
    coord_decimals: int = 0,
    ignored_roles=IGNORED_AXTREE_ROLES,
    ignored_properties=IGNORED_AXTREE_PROPERTIES,
    remove_redundant_static_text: bool = True,
    hide_bid_if_invisible: bool = False,
    hide_all_children: bool = False,
    hide_all_bids: bool = False,
) -> dict:
    """Formats the accessibility tree into a string text"""
    node_id_to_idx = {}
    for idx, node in enumerate(AX_tree["nodes"]):
        node_id_to_idx[node["nodeId"]] = idx

    cleaned_tree = {
        "nodes": [],
    }
    
    def dfs(node_idx: int, depth: int, parent_node_filtered: bool, parent_node_name: str) -> list[str]:
        node = AX_tree["nodes"][node_idx]
        skip_node = False  # node will not be printed, with no effect on children nodes
        filter_node = False  # node will not be printed, possibly along with its children nodes
        node_role = node["role"]["value"]
        node_name = ""
        
        # if node["nodeId"] == "alert-0-node" and not optimize:
        #     return []

        if node_role in ignored_roles:
            skip_node = True
            pass
        elif "name" not in node:
            skip_node = True
            pass
        else:
            node_name = node["name"]["value"]

            # extract bid
            bid = node.get("browsergym_id", None)

            # extract node attributes
            attributes = []
            for property in node.get("properties", []):
                if not "value" in property:
                    continue
                if not "value" in property["value"]:
                    continue

                prop_name = property["name"]
                prop_value = property["value"]["value"]

                if prop_name in ignored_properties:
                    continue
                elif prop_name in ("required", "focused", "atomic"):
                    if prop_value:
                        attributes.append(prop_name)
                else:
                    attributes.append(f"{prop_name}={repr(prop_value)}")

            if skip_generic and node_role == "generic" and not attributes:
                skip_node = True

            if hide_all_children and parent_node_filtered:
                skip_node = True

            if node_role == "StaticText":
                if parent_node_filtered:
                    skip_node = True
                elif remove_redundant_static_text and node_name in parent_node_name:
                    skip_node = True
            else:
                filter_node, extra_attributes_to_print = _process_bid(
                    bid,
                    extra_properties=extra_properties,
                    with_visible=with_visible,
                    with_clickable=with_clickable,
                    with_center_coords=with_center_coords,
                    with_bounding_box_coords=with_bounding_box_coords,
                    with_som=with_som,
                    filter_visible_only=filter_visible_only,
                    filter_with_bid_only=filter_with_bid_only,
                    filter_som_only=filter_som_only,
                    coord_decimals=coord_decimals,
                )

                # if either is True, skip the node
                skip_node = skip_node or filter_node


        new_child_node_ids = []
        for child_node_id in node["childIds"]:
            if child_node_id not in node_id_to_idx or child_node_id == node["nodeId"]:
                continue
            # mark this to save some tokens
            child_depth = depth if skip_node else (depth + 1)
            cids = dfs(
                node_id_to_idx[child_node_id],
                child_depth,
                parent_node_filtered=filter_node,
                parent_node_name=node_name,
            )
            if cids:
                new_child_node_ids += cids
        
        if skip_node:
            return new_child_node_ids

        node["childIds"] = new_child_node_ids
        cleaned_tree["nodes"].append(node)
        return [node["nodeId"]]    

    if len(AX_tree["nodes"]) == 0:
        return None
    root = dfs(0, 0, False, "")
    if len(root) > 1:
        print("Warning: Multiple root nodes after cleaning AX tree.")
    if len(root) == 0:
        return None

    # reverse the nodes to maintain original order
    cleaned_tree["nodes"] = list(reversed(cleaned_tree["nodes"]))
    print(f"Cleaned AX tree has {len(cleaned_tree['nodes'])} nodes (originally {len(AX_tree['nodes'])} nodes).")
    return include_woid_in_axtree(cleaned_tree)


def flatten_cleaned_axtree_to_str(
    AX_tree,
    extra_properties: dict = None,
    with_visible: bool = False,
    with_clickable: bool = False,
    with_center_coords: bool = False,
    with_bounding_box_coords: bool = False,
    with_som: bool = False,
    skip_generic: bool = True,
    filter_visible_only: bool = False,
    filter_with_bid_only: bool = False,
    filter_som_only: bool = False,
    coord_decimals: int = 0,
    ignored_roles=IGNORED_AXTREE_ROLES,
    ignored_properties=IGNORED_AXTREE_PROPERTIES,
    remove_redundant_static_text: bool = True,
    hide_bid_if_invisible: bool = False,
    hide_all_children: bool = False,
    hide_all_bids: bool = False,
    optimize: bool = False,
) -> str:
    """Formats the accessibility tree into a string text"""
    node_id_to_idx = {}
    for idx, node in enumerate(AX_tree["nodes"]):
        node_id_to_idx[node["nodeId"]] = idx

    def dfs(node_idx: int, depth: int) -> str:
        tree_str = ""
        node = AX_tree["nodes"][node_idx]
        indent = "\t" * depth
        node_role = node["role"]["value"]
        node_name = ""
        
        if node_role in ignored_roles:
            pass
        elif "name" not in node:
            pass
        else:
            node_name = node["name"]["value"]
            if "value" in node and "value" in node["value"]:
                node_value = node["value"]["value"]
            else:
                node_value = None

            # extract bid
            bid = node.get("browsergym_id", None)

            # extract node attributes
            attributes = []
            for property in node.get("properties", []):
                if not "value" in property:
                    continue
                if not "value" in property["value"]:
                    continue

                prop_name = property["name"]
                prop_value = property["value"]["value"]

                if prop_name in ignored_properties:
                    continue
                elif prop_name in ("required", "focused", "atomic"):
                    if prop_value:
                        attributes.append(prop_name)
                else:
                    attributes.append(f"{prop_name}={repr(prop_value)}")

            if node_role != "StaticText":
                filter_node, extra_attributes_to_print = _process_bid(
                    bid,
                    extra_properties=extra_properties,
                    with_visible=with_visible,
                    with_clickable=with_clickable,
                    with_center_coords=with_center_coords,
                    with_bounding_box_coords=with_bounding_box_coords,
                    with_som=with_som,
                    filter_visible_only=filter_visible_only,
                    filter_with_bid_only=filter_with_bid_only,
                    filter_som_only=filter_som_only,
                    coord_decimals=coord_decimals,
                )

                # insert extra attributes before regular attributes
                attributes = extra_attributes_to_print + attributes

            # actually print the node string
            if node_role == "generic" and not node_name:
                node_str = f"{node_role}"
            # elif node_role == "combobox":
            #     node_str = f"{node_role}"
            else:
                node_str = f"{node_role} {repr(node_name.strip())}"

            if not (
                hide_all_bids
                or bid is None
                or (
                    hide_bid_if_invisible
                    and extra_properties.get(bid, {}).get("visibility", 0) < 0.5
                )
            ):
                # if node_role in INCLUDE_BID_ROLES or not optimize:
                node_str = f"[{bid}] " + node_str
                # else:
                #     node_str = f"[xxx] " + node_str

            if node_value is not None:
                node_str += f' value={repr(node["value"]["value"])}'

            if attributes:
                node_str += ", ".join([""] + attributes)

            tree_str += f"{indent}{node_str}"

        for child_node_id in node["childIds"]:
            if child_node_id not in node_id_to_idx or child_node_id == node["nodeId"]:
                continue
            # mark this to save some tokens
            child_depth = depth + 1
            child_str = dfs(
                node_id_to_idx[child_node_id],
                child_depth,
            )
            if child_str and not child_str.endswith("image ''"):
                if tree_str:
                    tree_str += "\n"
                tree_str += child_str

        return tree_str

    if len(AX_tree["nodes"]) == 0:
        return ""
    tree_str = dfs(0, 0)
    return tree_str