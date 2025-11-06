import os

GITLAB = os.environ["WA_GITLAB"]
HOMEPAGE = os.environ["WA_HOMEPAGE"]
MAP = os.environ["WA_MAP"]
REDDIT = os.environ["WA_REDDIT"]
SHOPPING = os.environ["WA_SHOPPING"]
SHOPPING_ADMIN = os.environ["WA_SHOPPING_ADMIN"]
WIKIPEDIA = os.environ["WA_WIKIPEDIA"]

site_urls = {
    "gitlab": GITLAB,
    "reddit": REDDIT,
    "shopping": SHOPPING,
    "shopping_admin": SHOPPING_ADMIN,
    "map": MAP,
    "wikipedia": WIKIPEDIA
}


def get_wa_site_url(site: str) -> str:
    """Get the local URL for the given site."""
    return site_urls.get(site, "")
