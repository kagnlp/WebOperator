import logging
import urllib
logger = logging.getLogger(__name__)
from .utils import find_url_from_element
from browsergym.core.action.functions import can_scroll_down, can_scroll_up
from .url_simulator import URLSimulator
from .access_control import AccessControl
import gymnasium as gym

def is_repeated_action(action_str, trajectory):
    count = 1
    for step in trajectory[-2::-1]:
        if step["action"] is not None and step["action"]["code"] == action_str:
            count += 1
        else:
            break
    return count > 3



class ActionValidator:
    allow_invalid_action = True
    allow_invalid_page = False
    allow_unauthorized_page = True

    @classmethod
    def configure(cls, allow_invalid_page=False, allow_unauthorized_page=True):
        cls.allow_invalid_action = False
        cls.allow_invalid_page = allow_invalid_page
        cls.allow_unauthorized_page = allow_unauthorized_page
        
    @classmethod
    def validate(cls, formatted_action, trajectory, action_space, env: gym.Env):
        if cls.allow_invalid_action:
            return
        # Default implementation just returns the action as is
        func_name, arg_dict, selected_elem, action_str = formatted_action["type"], formatted_action["args"], formatted_action["selected_element"], formatted_action["code"]
        current_obs = trajectory[-1] if trajectory else {}

        if func_name == "click" or func_name == "fill" or func_name == "select_option":
            bid = arg_dict.get("bid", None)
            if bid is not None:
                if f"[{bid}]" in current_obs["axtree_txt"]:
                    pass
                else:
                    logger.warning(
                        f"bid '{bid}' not found in Current page Accessibility Tree. Only use bid which are present in Current page Accessibility Tree.")
                    raise ValueError(f"bid '{bid}' not found in Current page Accessibility Tree. Only use bid which are present in Current page Accessibility Tree.")
            
        if func_name not in action_space:
            raise ValueError(f"Action '{func_name}' not in action space. Use an action from current action space.")
        
        # if func_name == "fill":
        #     # Check if this field is read-only
        #     # Search in selected_elem["properties"] for a json with name "readonly"
        #     for prop in selected_elem["properties"]:
        #         if prop["name"] == "readonly" and prop["value"]["value"] == True:
        #             logger.warning(
        #                 f"Field with bid {bid} is read-only. Cannot fill it. Try another way.")
        #             raise ValueError(
        #                 f"Field with bid {bid} is read-only. Cannot fill it. Try another way.")            
        # Disable press_enter_after for content writing fields

        if not cls.allow_invalid_page and func_name == "goto" and not URLSimulator.is_valid_page(arg_dict["url"], env):
                logger.warning(f"Attempting to navigate to a 404 URL: {arg_dict['url']}")
                raise ValueError(f"Cannot navigate to invalid URL: {arg_dict['url']}. Please check the URL or try a different action. Don't try to create url from training data. Try the urls that exist in the current context.")

        if not cls.allow_unauthorized_page:
            if func_name == "goto":
                final_url = URLSimulator.get_final_url(arg_dict["url"], env)
                if not AccessControl.is_authorized_url(final_url):
                    logger.warning(f"Attempting to navigate to unauthorized URL: {arg_dict['url']}")
                    raise ValueError(f"Cannot navigate to unauthorized URL: {arg_dict['url']}. Please reside within the domains of simulated websites.")

            if func_name == "click" and selected_elem["role"]["value"] == "link":
                url = find_url_from_element(selected_elem)
                final_url = URLSimulator.get_final_url(url, env)
                if not AccessControl.is_authorized_url(final_url):
                    logger.warning(f"Attempting to navigate to unauthorized URL: {url}")
                    raise ValueError(f"Cannot navigate to unauthorized URL: {url}. Please reside within the domains of simulated websites.")
            
        if func_name == "click":
            if any(obj.get("name") == "disabled" and obj.get("value", {}).get("value") == True
                          for obj in selected_elem["properties"]):
                logger.warning(f"Attempting to click a disabled element: {selected_elem}")
                raise ValueError(f"Cannot click a disabled element: [{arg_dict['bid']}]. Please try a different action.")

        if func_name == "scroll":
            direction = arg_dict.get("direction", "").lower()
            # env = get_env()
            original_page = env.page
            if direction == "down":
                if not can_scroll_down(original_page):
                    logger.warning("Cannot scroll down further. The page is already at the bottom.")
                    raise ValueError("Cannot scroll down further. The page is already at the bottom. Try another action.")
            elif direction == "up":
                if not can_scroll_up(original_page):
                    logger.warning("Cannot scroll up further. The page is already at the top.")
                    raise ValueError("Cannot scroll up further. The page is already at the top. Try another action.")
            else:
                logger.warning(f"Invalid scroll direction: {direction}. Use 'up' or 'down'.")
                raise ValueError(f"Invalid scroll direction: {direction}. Use 'up' or 'down'.")
                
        if is_repeated_action(action_str, trajectory):
            logger.warning(f"Detected repeated action: {action_str}")
            raise ValueError(f"Action `{action_str}` is repeated 3 times. Analyze the previous actions-observations and try something different.")
    