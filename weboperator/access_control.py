import urllib
from .utils import normalize_url

class AccessControl:
    _authorized_domains: dict[str, str] = {}
    _authenticated_domains: dict[str, str] = {}
    
    @classmethod
    def authorize(cls, site_key, url):
        cls._authorized_domains[site_key] = urllib.parse.urlparse(normalize_url(url)).netloc

    @classmethod
    def authenticate(cls, site_key, url):
        cls._authenticated_domains[site_key] = urllib.parse.urlparse(normalize_url(url)).netloc 

    # logout
    # @classmethod
    # def logout(cls, url):
    #     cls._authorized_domains.discard({
    #         "domain": urllib.parse.urlparse(normalize_url(url)).netloc,
    #     })
    
    @classmethod
    def get_authorized_domains(cls):
        return cls._authorized_domains

    @classmethod
    def get_authenticated_domains(cls):
        return cls._authenticated_domains

    @classmethod
    def reset(cls):
        cls._authorized_domains = {}
        cls._authenticated_domains = {}

    @classmethod
    def is_authorized_url(cls, url):
        if len(cls._authorized_domains.keys()) == 0:
            authorized_locations = {
                "newtab",
                "",
            }
        else:
            authorized_locations = {
                "newtab",
                "",
                *(
                domain for domain in cls._authorized_domains.values()
                ),
            }
        
        url = normalize_url(url)
        
        page_location = urllib.parse.urlparse(url).netloc
        return page_location in authorized_locations
    
    @classmethod
    def is_authenticated_url(cls, url):
        if len(cls._authenticated_domains.keys()) == 0:
            return False
        
        url = normalize_url(url)
        authenticated_locations = {
            "newtab",
            "",
            *(
              domain for domain in cls._authenticated_domains.values()
            ),
        }
        
        page_location = urllib.parse.urlparse(url).netloc
        return page_location in authenticated_locations