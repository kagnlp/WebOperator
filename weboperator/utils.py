import unicodedata
from urllib.parse import urlparse


def normalize_url(url):
    if "://" not in url:
        url = "http://" + url
    return url
    # parsed = urlparse(url)
    # return f"{parsed.scheme}://{parsed.netloc}{parsed.path}".rstrip("/").lower()


def get_domain(url):
    parsed = urlparse(url)
    domain = parsed.netloc
    return domain


def get_base_url(full_url: str) -> str:
    parsed = urlparse(full_url)
    return f"{parsed.scheme}://{parsed.netloc}"


def find_url_from_element(element):
    properties = element["properties"]
    for prop in properties:
        if prop["name"] == "url":
            return prop["value"]["value"]
    return None


ERROR_STATUS_CODES = {400, 401, 403, 404, 408, 429, 500, 502, 503, 504}
ERROR_PATTERNS = [
    r"404.*not\s*found",
    r"page\s*not\s*found",
    r"chrome-error://chromewebdata/",
    r"this site can.?t be reached",
    r"err_name_not_resolved",
    r"err_connection_refused",
    r"err_timed_out",
    r"403.*forbidden",
    r"401.*unauthorized",
    r"500.*internal server error",
    r"502.*bad gateway",
    r"503.*service unavailable",
    r"504.*gateway timeout",
    r"504.*gateway time-out",
]


def to_ascii(text):
    # Replace all Unicode dash-like characters with ASCII hyphen
    dash_replacements = {
        "\u2010": "-",  # hyphen
        "\u2011": "-",  # non-breaking hyphen
        "\u2012": "-",  # figure dash
        "\u2013": "-",  # en dash
        "\u2014": "-",  # em dash
        "\u2015": "-",  # horizontal bar
        "‘": "'",
        "’": "'",
        "“": '"',
        "”": '"',
    }

    for uni_dash, ascii_dash in dash_replacements.items():
        text = text.replace(uni_dash, ascii_dash)

    # Normalize accents etc. and strip non-ASCII
    normalized = unicodedata.normalize("NFKD", text)
    return normalized.encode("ascii", "ignore").decode("ascii")
