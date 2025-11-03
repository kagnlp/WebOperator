import logging
import re
from browsergym.utils.obs import flatten_axtree_to_str
from browsergym.core.observation import extract_merged_axtree
logger = logging.getLogger(__name__)
# from browsergym.experiments.loop import get_env
import time
from .utils import normalize_url, ERROR_STATUS_CODES, ERROR_PATTERNS
from urllib.parse import urlparse
import gymnasium as gym
import socket
import requests
import playwright.sync_api

class URLSimulator:
    _visited_cache = {}
        
    @staticmethod
    def _url_exists(url: str, timeout: float = 5.0) -> bool:
        """Efficiently check if URL is valid and reachable."""
        # --- Step 1: DNS reachability ---
        # try:
        #     domain = urlparse(url).netloc
        #     if not domain:
        #         return False
        #     socket.gethostbyname(domain)
        # except socket.error:
        #     return False

        # --- Step 2: Quick HTTP request ---
        # try:
        #     response = requests.head(url, allow_redirects=True, timeout=timeout)
        #     if response.status_code >= 400:
        #         response = requests.get(url, stream=True, timeout=timeout)
        #     return response.status_code < 400
        # except requests.RequestException:
        #     return False
        
        return True
    
    @staticmethod
    def safe_goto(page, url):
        timeout = 30000
        while True:
            try:    
                response = page.goto(url, timeout=timeout)
                return response
            except playwright.sync_api.TimeoutError:
                print(f"Timeout while loading {url}, retrying...")
                timeout += 10000
            except Exception as e:
                print(f"Error while loading {url}: {e}, exiting...")
                raise
        
    @staticmethod
    def _is_error_page(page):
        try:
            axtree_txt = flatten_axtree_to_str(extract_merged_axtree(page))
            if len(axtree_txt) > 3000:
                return False
            return any(re.search(p, axtree_txt, re.IGNORECASE) for p in ERROR_PATTERNS)
        except Exception:
            return True
        
    # @classmethod
    # def _get_axtree_txt(cls, url):
    #     env = get_env()
    #     original_page = env.page
    #     env.page = env.context.new_page()
    #     obs, _, _, _, _ = env.step(f"goto('{url}')")
    #     obs = ObservationProcessor.process_obs(obs)
    #     axtree_txt = obs["axtree_txt"]
    #     axtree_obj = obs["axtree_object"]
    #     env.page.close()
    #     env.page = original_page
    #     time.sleep(1)
    #     return axtree_txt, axtree_obj
    
    @classmethod
    def open_and_check(cls, url, env):
        if not cls._url_exists(url):
            logger.debug(f"Unreachable or invalid URL: {url}")
            cls._visited_cache[url] = {
                "valid": False,
                "final_url": url,
                "last_checked": time.time(),
            }
            return

        url = normalize_url(url)
        if url in cls._visited_cache:
            # Check cache validity (30 minutes)
            cache_entry = cls._visited_cache[url]
            if time.time() - cache_entry["last_checked"] < 1800:
                logger.debug("URL is already inspected")
                return

        # env = get_env()
        context = env.context
        original_page = env.page
        time.sleep(3)  # wait for 3 seconds to ensure page is fully loaded
        background_page = context.new_page()  # new tab, shares login session
        
        valid_flag = True
        try:
            response = cls.safe_goto(background_page, url) # 30 seconds timeout
        except Exception as e:
            background_page.close()
            time.sleep(1)
            env.page = original_page
            raise

        if response.status in ERROR_STATUS_CODES or URLSimulator._is_error_page(background_page):
            logger.debug("Invalid page detected")
            valid_flag = False

        URLSimulator._visited_cache[url] = {
            "valid": valid_flag, 
            "final_url": normalize_url(background_page.url), 
            "last_checked": time.time()
        }

        background_page.close()  # clean up
        time.sleep(1)
        env.page = original_page
        
    @classmethod
    def is_valid_page(cls, url, env: gym.Env):
        url = normalize_url(url)
        if url not in cls._visited_cache:
            cls.open_and_check(url, env)
        return cls._visited_cache[url]["valid"]

    @classmethod
    def get_final_url(cls, url, env: gym.Env):
        url = normalize_url(url)
        if url not in cls._visited_cache:
            cls.open_and_check(url, env)
        return cls._visited_cache[url]["final_url"]
