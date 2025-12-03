from .web_state_node import WebStateNode
from .access_control import AccessControl


def is_repeated_action(node: WebStateNode) -> bool:
    curr = node
    count = 0
    while curr is not None and curr.last_action is not None:
        if curr.last_action["code"] == node.last_action["code"]:
            count += 1
        else:
            break
        curr = curr.parent
    return count >= 3 and not node.last_action["type"] == "scroll"


def was_destructive(obs):
    """Check if the last action resulted in destructive requests"""
    if obs is None or obs.parent is None or obs.last_action is None:
        return False

    if not AccessControl.is_authenticated_url(obs.url):
        return False

    for req in obs.http_requests:
        if req["method"] == "POST" and is_destructive(obs.parent, obs.last_action):
            if not req["payload"]:
                continue
            # Color red
            print(f"\033[91m- {req['method']} {req['url']}\033[0m")
            # print(f"  Payload: {req['payload']}")
            return True
        elif req["method"] in ["PUT", "PATCH", "DELETE"]:
            # Destructive Action
            print(f"\033[91m- {req['method']} {req['url']}\033[0m")
            return True
    return False


def is_destructive(obs: WebStateNode, action):
    """Check if an action is destructive (should be avoided during backtracking)"""
    if not AccessControl.is_authenticated_url(obs.url):
        return False

    if (
        action["type"] == "click"
        and action.get("selected_element") is not None
        and action["selected_element"].get("role", None) is None
    ):
        print("Click action with no role, possibly non-destructive: ", action["code"])
    # Check for destructive click actions
    if (
        action["type"] == "click"
        and action.get("selected_element") is not None
        and action["selected_element"].get("role", None) is not None
        and action["selected_element"]["role"]["value"] == "button"
    ):

        # Check button name for common non-destructive actions
        if action["selected_element"]["name"]["value"].lower() in [
            "search",
            "back",
            "refresh",
            "export",
        ]:
            return False

        # Check if button has popup (non-destructive)
        has_popup = any(
            obj.get("name") == "hasPopup" for obj in action["selected_element"]["properties"]
        )

        is_disabled = any(
            obj.get("name") == "disabled" and obj.get("value", {}).get("value") == True
            for obj in action["selected_element"]["properties"]
        )

        if has_popup or is_disabled:
            return False

        # if action["selected_element"]["name"]["value"].lower() in [
        #     "delete", "remove", "close", "cancel", "sign out", "log out", "logoff", "submit", "confirm", "accept", "decline", "save", "save changes", "post", "send"
        # ]:
        #     return True

        return True
        # Check button name for common destructive actions
        # search, back, refresh, etc. are usually non-destructive

    elif action["type"] == "fill":
        return action["args"]["press_enter_after"]

    return False


def is_terminating(action):
    if action["type"] == "stop":
        return True
    return False
