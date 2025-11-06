import urllib
from .utils import normalize_url

class AccessControl:
    _authorized_domains: set[str] = set()
    _authenticated_domains: set[str] = set()
    
    @classmethod
    def authorize(cls, url):
        print(f"Logging in to {url}")
        cls._authorized_domains.add(urllib.parse.urlparse(normalize_url(url)).netloc)
        
    @classmethod
    def authenticate(cls, url, guest_mode=False):
        print(f"Logging in to {url} with guest_mode={guest_mode}")
        cls._authenticated_domains.add(urllib.parse.urlparse(normalize_url(url)).netloc)

    # logout
    # @classmethod
    # def logout(cls, url):
    #     cls._authorized_domains.discard({
    #         "domain": urllib.parse.urlparse(normalize_url(url)).netloc,
    #     })

    @classmethod
    def reset(cls):
        cls._authorized_domains.clear()
        cls._authenticated_domains.clear()
    
    @classmethod
    def is_authorized_url(cls, url):
        if len(cls._authorized_domains) == 0:
            return False
        
        url = normalize_url(url)
        authorized_locations = {
            "newtab",
            "",
            *(
              domain for domain in cls._authorized_domains
            ),
        }
        
        page_location = urllib.parse.urlparse(url).netloc
        return page_location in authorized_locations
    
    @classmethod
    def is_authenticated_url(cls, url):
        if len(cls._authenticated_domains) == 0:
            return False
        
        url = normalize_url(url)
        authenticated_locations = {
            "newtab",
            "",
            *(
              domain for domain in cls._authenticated_domains
            ),
        }
        
        page_location = urllib.parse.urlparse(url).netloc
        return page_location in authenticated_locations