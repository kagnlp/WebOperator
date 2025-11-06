from weboperator.html_renderer import image_to_jpg_base64_url, jpg_base64_url_to_image


class WebStateNode:
    def __init__(self, last_action, last_obs_description, parent=None):
        self.url = ""
        self.axtree_txt = ""
        self.last_obs_description = last_obs_description
        self.axtree_object = {}
        self.pruned_html = ""
        self.screenshot = None
        self.active_page_index = 0
        self.open_pages_titles = []
        self.page_too_long = None
        self.open_pages_urls = []
        self.parent = parent            # Parent node
        self.children = []              # List of child TreeNode instances
        self.last_action = last_action
        self.last_action_error = ""
        # self.visits = 0                 # Optional: for tree search algorithms
        self.value = None                 # Optional: for evaluation
        self.checklist = None
        self.checklist_completion = ""
        self.level = 0 if parent is None else parent.level + 1
        self.position = parent.position * 3 + len(parent.children) if parent else 0
        self.notes = [] if parent is None else parent.notes + ([last_action["args"]["text"]] if last_action is not None and last_action["type"] == "note" else [])
        self.visible = True
        self.is_404 = False
        self.refresh_loses_state = False
        self.corrupted = False
        self.frequency = 0 
        self.cookies = None
        self.local_storage = None
        self.scroll = 0 
        self.destructive = False
        self.http_requests = []
        self.goal = None
        if last_action is not None:
            if last_action["type"] == "scroll":
                if last_action["args"]["direction"] == "down":
                    self.scroll += 1
                elif last_action["args"]["direction"] == "up":
                    self.scroll = max(0, self.scroll - 1)

    def __lt__(self, other):
        # based on frequency
        return self.frequency > other.frequency
    
    def prune(self):
        self.visible = False

    def update_from_obs(self, obs):
        """Update the node's attributes from the observation."""
        self.axtree_txt = obs["axtree_txt"]
        self.page_too_long = obs["page_too_long"]
        self.axtree_object = obs["axtree_object"]
        self.pruned_html = obs["pruned_html"]
        self.screenshot = obs["screenshot"]
        self.active_page_index = obs["active_page_index"]
        self.open_pages_titles = obs["open_pages_titles"]
        self.open_pages_urls = obs["open_pages_urls"]
        self.url = obs["open_pages_urls"][self.active_page_index]
        # if self.last_action_error == "":
        self.last_action_error = obs["last_action_error"]
        # elif self.last_action_error != obs["last_action_error"] and obs["last_action_error"] != "":
        #     self.last_action_error = obs["last_action_error"] + "\n" + self.last_action_error

    def add_child(self, child_node):
        self.children.append(child_node)

    def is_leaf(self):
        return len(self.children) == 0

    def is_root(self):
        return self.parent is None

    def path(self):
        node, result = self, []
        while node:
            result.append(node)
            node = node.parent
        return list(reversed(result))

    def __repr__(self):
        action_thought = self.last_action.get("thought", "No action") if self.last_action else "No action"
        return f"TreeNode(url={self.url}, action={action_thought}, error={self.last_action_error})"

    # to_json
    def from_json(self, data):
        self.url = data.get("url", "")
        self.axtree_txt = data.get("axtree_txt", "")
        self.axtree_object = data.get("axtree_object", {})
        self.page_too_long = data.get("page_too_long", False)
        self.pruned_html = data.get("pruned_html", "")
        self.screenshot = jpg_base64_url_to_image(data.get("screenshot_url", ""))
        self.active_page_index = data.get("active_page_index", 0)
        self.open_pages_titles = data.get("open_pages_titles", [])
        self.open_pages_urls = data.get("open_pages_urls", [])
        self.last_action = data.get("last_action", {})
        self.last_action_error = data.get("last_action_error", "")
        self.last_obs_description = data.get("last_obs_description", "")
        self.value = data.get("value", None)
        self.notes = data.get("notes", [])
        self.visible = data.get("visible", True)
        self.children = []
        self.checklist_completion = data.get("checklist_completion", "")
        self.frequency = 0
        for child_data in data.get("children", []):
            child_node = WebStateNode(last_action=None, last_obs_description=None, parent=self)
            child_node.from_json(child_data)
            self.children.append(child_node)
        self.is_404 = data.get("is_404", False)
        self.checklist = data.get("checklist", None)
        self.frequency = data.get("frequency", 0)
        self.refresh_loses_state = data.get("refresh_loses_state", False)
        self.corrupted = False
        self.scroll = data.get("scroll", 0)
        self.destructive = data.get("destructive", False)
        self.http_requests = data.get("http_requests", [])

    def to_json(self):
        return {
            "url": self.url,
            "axtree_txt": self.axtree_txt,
            "active_page_index": self.active_page_index,
            "open_pages_titles": self.open_pages_titles,
            "open_pages_urls": self.open_pages_urls,
            "last_action": self.last_action,
            "last_action_error": self.last_action_error,
            "last_obs_description": self.last_obs_description,
            "value": self.value,
            "frequency": self.frequency,
            "notes": self.notes,
            "visible": self.visible,
            "scroll": self.scroll,
            "checklist": self.checklist,
            "checklist_completion": self.checklist_completion,
            "is_404": self.is_404,
            "page_too_long": self.page_too_long,
            "refresh_loses_state": self.refresh_loses_state,
            "corrupted": self.corrupted,
            "destructive": self.destructive,
            "http_requests": self.http_requests,
            "children": [child.to_json() for child in self.children],
            # "screenshot_url": image_to_jpg_base64_url(self.screenshot),
            "axtree_object": self.axtree_object,
            # "pruned_html": self.pruned_html,
        }
