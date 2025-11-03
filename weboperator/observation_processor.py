from browsergym.utils.obs import flatten_axtree_to_str
from .axtree_utils import is_full_axtree_too_long, clean_axtree
import logging
import re
from .utils import ERROR_STATUS_CODES, ERROR_PATTERNS

logger = logging.getLogger(__name__)


class ObservationProcessor:
    optimized = True
    truncate_error_message = True

    @classmethod
    def configure(cls, optimized, truncate_error_message):
        cls.optimized = optimized
        cls.truncate_error_message = truncate_error_message

    @classmethod
    def _minimize_error_message(cls, error_msg: str) -> str:
        if cls.truncate_error_message and len(error_msg) > 1000:
            error_msg = error_msg[:1000] + "..."
        return error_msg

    @classmethod
    def process_obs(cls, obs):
        full_ax = ""
        visible_ax = ""
        pruned_html = ""
        cleaned_axtree = {"nodes": []}
        try:
            cleaned_axtree = clean_axtree(obs["axtree_object"])
            # cleaned_axtree = obs["axtree_object"]
            full_ax = flatten_axtree_to_str(cleaned_axtree, optimize=(cls.optimized))
            visible_ax = flatten_axtree_to_str(
                cleaned_axtree,
                filter_visible_only=True,
                extra_properties=obs["extra_element_properties"],
                optimize=(cls.optimized),
            )
            # pruned_html = prune_html(flatten_dom_to_str(obs["dom_object"]))
        except Exception as e:
            logger.error(f"Error processing axtree: {e}")

        active_page_index = int(obs["active_page_index"][0])

        if cls.optimized:
            processed_obs = {
                "chat_messages": obs["chat_messages"],
                "screenshot": obs["screenshot"],
                "goal_object": obs["goal_object"],
                "last_action": obs["last_action"],
                "last_action_error": cls._minimize_error_message(obs["last_action_error"]),
                "open_pages_urls": obs["open_pages_urls"],
                "open_pages_titles": obs["open_pages_titles"],
                "active_page_index": active_page_index,
                "url": obs["url"],
                "axtree_object": cleaned_axtree,
                "axtree_txt": visible_ax if is_full_axtree_too_long(full_ax) else full_ax,
                "axtree_full_txt": full_ax,
                "page_too_long": is_full_axtree_too_long(full_ax),
                "axtree_visible_only_txt": visible_ax,
                "pruned_html": pruned_html,
            }
        else:
            processed_obs = {
                "chat_messages": obs["chat_messages"],
                "screenshot": obs["screenshot"],
                "goal_object": obs["goal_object"],
                "last_action": obs["last_action"],
                "last_action_error": cls._minimize_error_message(obs["last_action_error"]),
                "open_pages_urls": obs["open_pages_urls"],
                "open_pages_titles": obs["open_pages_titles"],
                "active_page_index": active_page_index,
                "url": obs["url"],
                "axtree_object": cleaned_axtree,
                "axtree_txt": visible_ax,
                "axtree_full_txt": full_ax,
                "page_too_long": True,
                "axtree_visible_only_txt": visible_ax,
                "pruned_html": pruned_html,
            }

        return processed_obs

    # def detect_errors(self, obs, tree_node):
    #     # Detect various error conditions

    @staticmethod
    def is_404_page(axtree_txt):
        # Heuristic: Require short length (heuristic, most 404 pages are small)
        if len(axtree_txt) > 3000:
            return False
        return any(
            re.search(p, axtree_txt, re.IGNORECASE) for p in ERROR_PATTERNS
        )  # and node.last_action["type"] == "goto"
