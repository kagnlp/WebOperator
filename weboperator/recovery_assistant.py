from .access_control import AccessControl
from .axtree_utils import is_recaptcha_in_axtree, has_axtree_changed
from .action_analyzer import is_repeated_action
import logging

logger = logging.getLogger(__name__)


class RecoveryAssistant:
    recover_from_invalid_page = True
    recover_from_restricted_page = True
    recover_from_captcha = True
    enabled = False
    chat_mode = False

    @classmethod
    def configure(
        cls,
        recover_from_invalid_page=True,
        recover_from_restricted_page=True,
        recover_from_captcha=True,
        chat_mode=False,
    ):
        cls.enabled = True
        cls.recover_from_invalid_page = recover_from_invalid_page
        cls.recover_from_restricted_page = recover_from_restricted_page
        cls.recover_from_captcha = recover_from_captcha
        cls.chat_mode = chat_mode

    @classmethod
    def get_recovery_actions(cls, curr_obs):
        if not cls.enabled:
            return None
        if curr_obs.last_action is None or curr_obs.parent is None:
            return None
        if cls.recover_from_restricted_page and not AccessControl.is_authorized_url(curr_obs.url):
            if len(curr_obs.open_pages_urls) > len(
                curr_obs.parent.open_pages_urls
            ):  # Last action opened a new page
                actions = [
                    (
                        {
                            "code": "tab_close()",
                            "thought": "The current page is unauthorized, closing the tab.",
                            "type": "tab_close",
                            "args": {},
                            "selected_element": None,
                        },
                        "Unauthorized page.",
                    )
                ]
            else:
                actions = [
                    (
                        {
                            "code": "go_back()",
                            "thought": "The current page is unauthorized, going back.",
                            "type": "go_back",
                            "args": {},
                            "selected_element": None,
                        },
                        "Unauthorized page.",
                    )
                ]
        elif curr_obs.is_404 and cls.recover_from_invalid_page:
            if len(curr_obs.open_pages_urls) > len(curr_obs.parent.open_pages_urls):
                actions = [
                    (
                        {
                            "code": "tab_close()",
                            "thought": "The current page returned an error (e.g., not found, forbidden, server error), closing the tab.",
                            "type": "tab_close",
                            "args": {},
                            "selected_element": None,
                        },
                        "404 page.",
                    )
                ]
            else:
                actions = [
                    (
                        {
                            "code": "go_back()",
                            "thought": "The current page returned an error (e.g., not found, forbidden, server error), going back.",
                            "type": "go_back",
                            "args": {},
                            "selected_element": None,
                        },
                        "404 page.",
                    )
                ]
        elif is_recaptcha_in_axtree(curr_obs.axtree_txt) and cls.recover_from_captcha:
            # take enter key to skip
            if cls.chat_mode:
                actions = [
                    (
                        {
                            "code": "stop('reCAPTCHA detected in the page. Please resolve it manually and reply to continue.')",
                            "thought": "Current page has reCAPTCHA, which should be resolved manually. Returning to user.",
                            "type": "stop",
                            "args": {
                                 "text": "reCAPTCHA detected in the page. Please resolve it manually and reply to continue."    
                            },
                            "selected_element": None,
                        },
                        "ReCAPTCHA page.",
                    )
                ]
            else:
                input("reCAPTCHA detected. Please resolve manually and press Enter to continue...")
                actions = [
                    (
                        {
                            "code": "noop()",
                            "thought": "Current page has reCAPTCHA, which is resolved manually. Continuing...",
                            "type": "noop",
                            "args": {},
                            "selected_element": None,
                        },
                        "ReCAPTCHA page.",
                    )
                ]
        else:
            actions = None
            # No recovery actions available
        return actions

    @classmethod
    def get_recovery_hint(cls, curr_obs):
        if not cls.enabled:
            return ""
        if curr_obs.last_action is None:
            return ""

        if curr_obs.is_404 and cls.recover_from_invalid_page:
            return "The current page returned an error (e.g., not found, forbidden, server error). Try going back or closing the tab."

        if not AccessControl.is_authorized_url(curr_obs.url) and cls.recover_from_restricted_page:
            return "Unauthorized url, please go_back() or change/close tab."

        if is_repeated_action(curr_obs):  # Could be a problem for scroll('down')
            return "You have used the same action 3 times repeatedly. Don't do this again. You are probably stuck in a loop. Analyze the previous actions-observations and try something different. If you have completed the task, you can stop interaction."

        if curr_obs.last_action["type"] == "fill":
            for prop in curr_obs.last_action["selected_element"]["properties"]:
                if prop["name"] == "readonly" and prop["value"]["value"] == True:
                    return f"Field with bid {curr_obs.last_action['args']['bid']} is read-only. Cannot fill it. Try another way."

        if (
            curr_obs.last_action["type"] == "click"
            and curr_obs.last_action["selected_element"]["role"]["value"] == "option"
            and curr_obs.last_action_error.strip() != ""
            and not has_axtree_changed(curr_obs)
        ):
            return f"Clicking option returned an error. Try select_option() instead."

        return ""

        # if is_alert_available(curr_obs.axtree_object):
        #     return "An alert was detected in the page content."
