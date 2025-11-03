import importlib.resources
import json
import logging
import tempfile
import urllib.parse
from typing import Optional, Tuple

import numpy as np
import playwright.sync_api

from browsergym.core.task import AbstractBrowserTask
import time
from .instance import WebArenaInstance
logger = logging.getLogger(__name__)


class GenericWebArenaTask(AbstractBrowserTask):
    """
    Base class for all WebArena tasks.

    """

    def __init__(
        self,
        seed: int,
        task_id: Optional[int] = None,
        intent_template_id: Optional[int] = None,
        with_na_hint: bool = False,
        with_homepage_hint: bool = False,
    ) -> None:
        super().__init__(seed)

        # task properties, will be used to set up the browsergym environment
        self.viewport = {"width": 1280, "height": 720}
        self.slow_mo = 1000  # ms
        self.timeout = 10000  # ms

        self.webarena_instance = WebArenaInstance()
        self.config_file: str = None
        self.with_na_hint = with_na_hint
        self.with_homepage_hint = with_homepage_hint

        # one and only one of task id and template id must be provided
        if (task_id is None) == (intent_template_id is None):
            raise ValueError(
                f"One and only one of 'task_id' and 'intent_template_id' must be provided (task_id={task_id}, intent_template_id={intent_template_id})."
            )

        # read the list of all webarena task configs
        import webarena

        all_configs_str = importlib.resources.files(
            webarena).joinpath("test.raw.json").read_text()

        # substitute URLs
        for pattern, url_key in {
            "__GITLAB__": "gitlab",
            "__REDDIT__": "reddit",
            "__SHOPPING__": "shopping",
            "__SHOPPING_ADMIN__": "shopping_admin",
            "__WIKIPEDIA__": "wikipedia",
            "__MAP__": "map",
        }.items():
            all_configs_str = all_configs_str.replace(
                pattern, self.webarena_instance.urls[url_key])

        # load all task configs to JSON
        all_configs = json.loads(all_configs_str)

        # keep only the desired task configs
        if intent_template_id is not None:
            task_configs = [
                conf for conf in all_configs if conf["intent_template_id"] == intent_template_id
            ]
            if not task_configs:
                raise ValueError(
                    f"Could not find any task config with intent_template_id={intent_template_id}."
                )

        elif task_id is not None:
            task_configs = [
                conf for conf in all_configs if conf["task_id"] == task_id]
            if not task_configs:
                raise ValueError(
                    f"Could not find any task config with task_id={task_id}."
                )

        self.task_configs = task_configs

    def setup(self, page: playwright.sync_api.Page) -> tuple[str, dict]:
        # import webarena on instanciation
        from webarena.evaluation_harness.advanced_evaluators import evaluator_router
        def handle_dialog(dialog):
            page.dialog_message = dialog.message
            dialog.accept()
            logger.debug(f"Accepted dialog with message: {dialog.message} -> {dialog.type}")
        def handle_console(msg):
            logger.debug(f"[JS Console][{msg.type}] {msg.text}")

        def log_request(request):
            try:
                # Try to read textual post data (may raise UnicodeDecodeError for binary payloads)
                payload = request.post_data
            except UnicodeDecodeError:
                # Binary/compressed data — avoid decoding; record content-type and a placeholder
                try:
                    content_type = dict(request.headers).get("content-type")
                except Exception:
                    content_type = None
                payload = f"<binary or compressed payload; content-type={content_type}>"
            except Exception:
                # Best-effort fallback: try to capture base64 raw if available, else None
                try:
                    base64_data = getattr(request._impl_obj, "post_data", None)
                    if base64_data:
                        # keep it short to avoid huge logs
                        payload = f"<base64:{base64_data[:200]}...>"
                    else:
                        payload = None
                except Exception:
                    payload = None

            try:
                headers = dict(request.headers)
            except Exception:
                headers = {}

            page.http_requests.append({
                "method": getattr(request, "method", None),
                "url": getattr(request, "url", None),
                "header": headers,
                "payload": payload if payload is not None else "",
            })
            
        # pick a task at random
        self.config = self.random.choice(self.task_configs)

        # hack: dynamically build a config file to read from
        with tempfile.NamedTemporaryFile(mode="w+", delete=False) as f:
            json.dump(self.config, f)
            f.flush()
            self.config_file = f.name

        # build the evaluator
        self.evaluator = evaluator_router(self.config_file)

        # authenticate
        for site in self.config["sites"]:
            logger.debug(f"Logging in to {site}...")
            while True:
                try:
                    self.webarena_instance.ui_login(site=site, page=page)
                    logger.debug(f"Logged in to {site}.")
                    break
                except Exception as e:
                    print(f"Login to {site} failed with exception: {e}. Retrying...")
                    time.sleep(5)

        # set geolocation
        page.context.set_geolocation(self.config["geolocation"])

        # navigate to the starting url(s) (might need several pages)
        # https://github.com/web-arena-x/webarena/blob/c6475f0e9affe5252a2966e26b8cb4c834a4ae40/browser_env/envs.py#L150
        if self.config["start_url"]:
            start_urls = self.config["start_url"].split(" |AND| ")
            for i, url in enumerate(start_urls):
                page.dialog_message = None
                page.http_requests = []
                page.on("dialog", handle_dialog)
                page.on("request", log_request)
                page.on("console", handle_console)
                
                while True:
                    try:
                        page.goto(url, timeout=0)
                        break
                    except Exception as e:
                        print(f"Failed to navigate to {url} with exception: {e}. Retrying...")
                        time.sleep(5)
                if i < len(start_urls) - 1:
                    page = page.context.new_page()

        # recover goal
        goal = self.config["intent"]

        # This note is present in all webarena's agent prompts
        # https://github.com/web-arena-x/webarena/blob/c6475f0e9affe5252a2966e26b8cb4c834a4ae40/agent/prompts/raw/p_cot_id_actree_2s.py#L34
        if self.with_homepage_hint:
            goal += f"""

(Note: if you want to visit other websites, check out the homepage at {self.webarena_instance.home_url}. It has a list of websites you can visit. {self.webarena_instance.home_url}/password.html lists all the account name and password for the websites. You can use them to log in to the websites.)
"""

        # This note is present in some of webarena's agent prompts
        if self.with_na_hint:
            goal += """\

If you believe the task is impossible to complete, provide the answer "N/A".
"""

        return goal, {}

    def cheat(self, page: playwright.sync_api.Page, chat_messages: list[str]) -> None:
        raise NotImplementedError

    @classmethod
    def get_task_id(cls):
        """
        Generic class for several task ids, this way of obtaining the task id is not compatible for now.
        """
        raise NotImplementedError

    def teardown(self) -> None:
        # Nothing to be done here
        # https://github.com/web-arena-x/webarena/blob/c6475f0e9affe5252a2966e26b8cb4c834a4ae40/browser_env/envs.py#L227
        pass

    def validate(
        self, page: playwright.sync_api.Page, chat_messages: list[str]
    ) -> Tuple[float, bool, str, dict]:

        # safeguard: check that all open tabs are either blank or within the list of WebArena URLs
        authorized_locations = ["newtab", ""] + [
            urllib.parse.urlparse(url).netloc
            for url in [*self.webarena_instance.urls.values(), self.webarena_instance.home_url]
        ]
        page_location = urllib.parse.urlparse(page.url).netloc
        # if page.url == "https://experienceleague.adobe.com/en/docs/commerce-admin/start/reporting/reports-menu":
#             context = page.context
#             # page.close()
#             # set most recent page as active page, or open a new page if needed
#             if context.pages:
#                 # TODO: do something more elaborate? (active page history)
#                 page = context.pages[-2]
#             else:
#                 page = context.new_page()
#             # trigger the callback that sets this page as active in browsergym
#             page.evaluate(
#                     """\
# const event = new Event('pageshow', {
#     bubbles: true,  // Whether the event bubbles up through the DOM or not
#     cancelable: false  // Whether the event can be canceled
# });
# window.dispatchEvent(event);
#             """
#                 )
            # return 0, True, "", {"error": "Your last action is trying to access: http://127.0.0.1:7780/admin/analytics/reports/show/ which is redirecting to unauthorized url `https://experienceleague.adobe.com/en/docs/commerce-admin/start/reporting/reports-menu`. Please try a different action."}

        url = page.url
        # if page_location not in authorized_locations:
            # history_length = page.evaluate("() => history.length")
            # if history_length > 1:
            #     page.go_back()
            #     return 0, True, "", {"error": f"Your last action is trying to access an unauthorized url `{url}`. Please try a different action."}

            # return 0, True, "", {"error": "Unauthorized url, please go_back() or change/close tab."}

        # import webarena dynamically
        from webarena.browser_env.actions import ActionTypes

        # if any, use the last assistant message as the stop answer for webarena
        if chat_messages and chat_messages[-1]["role"] == "assistant":
            last_action = {"action_type": ActionTypes.STOP,
                           "answer": chat_messages[-1]["message"]}
        elif chat_messages and chat_messages[-1]["role"] == "infeasible":
            last_action = {"action_type": ActionTypes.STOP, "answer": "N/A"}
        else:
            last_action = {"action_type": ActionTypes.NONE, "answer": ""}
            # llm_fuzzy_match() bugfix
            last_action["answer"] = "whatever"

        # hack: fake trajectory for evaluation (only last_action["answer"] is used in the webarena evaluation codebase)
        trajectory = [{}, last_action]  # StateInfo, Action

        # call the evaluator if the last action is a stop action
        evaluation_data = {}
        if last_action["action_type"] == ActionTypes.STOP:
            try:
                score, evaluation_data = self.evaluator(
                    trajectory=trajectory,
                    config_file=self.config_file,
                    page=page,
                    client=None,  # none of webarena's evaluators requires a cdp session
                )
            # llm_fuzzy_match() bugfix (assert "correct" in response)
            except AssertionError:
                logger.debug(
                    "llm_fuzzy_match() bugfix applied: AssertionError in evaluator, using score = 0.0"
                )
                score = 0.0
            except ValueError as e:
                logger.error(f"ValueError in evaluator: {e}")
                score = 0.0

            return score, True, "", evaluation_data
        
        return 0.0, False, "", evaluation_data
