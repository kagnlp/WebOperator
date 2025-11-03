import base64
import io
import json
import re
from pathlib import Path
from typing import Any
import numpy as np
from PIL import Image
import html as HTML
# from browser_env import (
#     Action,
#     ActionTypes,
#     ObservationMetadata,
#     StateInfo,
#     action2str,
# )

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <style>
        body {{ font-family: sans-serif; line-height: 1.6; padding: 20px; }}
        .task-step {{ border: 1px solid #ccc; margin-bottom: 20px; padding: 15px; border-radius: 5px; background-color: #f9f9f9; }}
        .task-step h2 {{ margin-top: 0; color: #333; border-bottom: 1px solid #eee; padding-bottom: 5px;}}
        .task-step h3 {{ color: #555; margin-top: 15px; margin-bottom: 5px; }}
        .task-step h4 {{ color: #777; margin-top: 10px; margin-bottom: 5px; font-style: italic;}}
        pre {{ background-color: #eee; padding: 10px; border-radius: 3px; white-space: pre-wrap; word-wrap: break-word; font-size: 0.9em; margin-top: 5px; }}
        details {{ margin-top: 10px; border: 1px solid #ddd; border-radius: 3px; background-color: #fff; }}
        summary {{ cursor: pointer; padding: 8px; background-color: #f8f9fa; font-weight: bold; border-bottom: 1px solid #ddd; }}
        details[open] summary {{ border-bottom: 1px solid #ddd; }}
        details > pre {{ border: none; background-color: #fff; padding: 10px 8px; }}
        .response-item-toggle {{ margin-top: 10px; }}
        .chosen-section {{ border-left: 5px solid #4CAF50; padding-left: 10px; margin-top: 15px; }}
        .rejected-section {{ border-left: 5px solid #f44336; padding-left: 10px; margin-top: 15px; }}
        hr {{ border: 0; border-top: 1px solid #eee; margin: 15px 0; }}
        .thought-action {{ background-color: #f0f0f0; padding: 10px; border-radius: 3px; margin-bottom: 10px; border: 1px solid #e0e0e0;}}
        .thought-action h4 {{ margin-top: 0; color: #666; }}
        .task-container {{ display: none; }}
        .filter-controls {{ margin-bottom: 20px; padding: 10px; background-color: #e9ecef; border-radius: 5px; }}
        .filter-controls label {{ margin-right: 10px; font-weight: bold; }}
        .filter-controls select {{ padding: 5px; border-radius: 3px; border: 1px solid #ced4da; }}
    </style>
</head>
<html>
    <body>
     {body}
    </body>
</html>
"""

def jpg_base64_url_to_image(data_url):
    prefix = "data:image/jpeg;base64,"
    if not data_url or not isinstance(data_url, str):
        return None
    if data_url.startswith(prefix):
        data_url = data_url[len(prefix):]
    try:
        image_bytes = base64.b64decode(data_url)
        if not image_bytes:
            return None
        image = Image.open(io.BytesIO(image_bytes))
        # Convert PIL Image to numpy array
        return np.array(image)
    except Exception:
        return None

def image_to_jpg_base64_url(image: np.ndarray | Image.Image):
    """Convert a numpy array to a base64 encoded image url."""

    if image is None:
        return ""

    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)

    if image.mode in ("RGBA", "LA"):
        image = image.convert("RGB")

    with io.BytesIO() as buffer:
        image.save(buffer, format="JPEG")
        image_base64 = base64.b64encode(buffer.getvalue()).decode()

    return f"data:image/jpeg;base64,{image_base64}"


class RenderHelper(object):
    """Helper class to render text and image observations and meta data in the trajectory"""

    def __init__(self, result_dir: str) -> None:
        file_path = Path(result_dir) / "steps_info.html"
        # Step 1: Clear the file initially (overwrite)
        with open(file_path, "w", encoding='utf-8'):
            pass  # Just opening in 'w' mode truncates the file
        # Step 2: Open in append mode for later use
        self.render_file = open(file_path, "a+", encoding='utf-8')

        self.render_file.truncate(0)
        # write init template
        self.render_file.write(HTML_TEMPLATE.format(body=""))
        self.render_file.read()
        self.render_file.flush()

    def render_observation(
        self,
        observation: dict,
        action_error: str = None,
        render_screenshot: bool = False,
    ) -> None:
        new_content = ""
        
        if observation.get("goal", None):
            new_content = f"<div class='goal' style='background-color:lightblue'><h2>Goal</h2><p>{HTML.escape(observation['goal'])}</p></div>\n"
            
        url = observation["open_pages_urls"][observation["active_page_index"]]
        axtree = observation["axtree_txt"]
        if action_error:
            new_content += f"<div class='prev_action' style='background-color:pink'>{HTML.escape(action_error)}</div>"

        if len(observation["open_pages_urls"]) > 1:
            new_content += f"<h2>Open Tabs</h2>\n"
            for u in observation["open_pages_urls"]:
                new_content += f"<div class='open_page_url' style='background-color:lightgrey'><a href={HTML.escape(u)}>{HTML.escape(u)}</a></div>\n"
            
        new_content += f"<h2>Active Tab</h2>\n"
        new_content += f"<h3 class='url'><a href={HTML.escape(url)}>{HTML.escape(url)}</a></h3>\n"
        new_content += f"""
        <details>
            <summary>Text Observation (Click to expand/collapse)</summary>
            <pre>{HTML.escape(axtree)}</pre>
        </details>
        """

        if render_screenshot:
            # image observation
            try:
                img_obs = observation["screenshot"]
                image = Image.fromarray(img_obs)
                byte_io = io.BytesIO()
                image.save(byte_io, format="PNG")
                byte_io.seek(0)
                image_bytes = base64.b64encode(byte_io.read())
                image_str = image_bytes.decode("utf-8")
                new_content += (
                    f"<img src='data:image/png;base64,{image_str}' style='width:50vw; height:auto;'/>\n"
                )
            except:
                pass

        self.render_file.seek(0)
        html = self.render_file.read()
        html_body = re.findall(r"<body>(.*?)</body>", html, re.DOTALL)[0]
        html_body += new_content

        html = HTML_TEMPLATE.format(body=html_body)
        self.render_file.seek(0)
        self.render_file.truncate()
        self.render_file.write(html)
        self.render_file.flush()

    def render_action(
        self,
        action,
        obs_description,
        backtrack = False
    ) -> None:
        new_content = ""
        if obs_description:
            new_content += f"<div class='obs_description'>{HTML.escape(obs_description)}</div>\n"
        action_str = f"<div class='raw_parsed_prediction' style='background-color:grey'><pre>{HTML.escape(action['thought'])}</pre></div>"
        # action_str += f"<div class='action_object' style='background-color:grey'><pre>{repr(action)}</pre></div>"
        if backtrack:
            action_str += f"<div class='parsed_action' style='background-color:orange'><pre>{'[B]' + action['code']}</pre></div>"
        # elif action["type"] == "stop":
        #     action_str += f"<div class='parsed_action' style='background-color:green'><pre>{action["code"]}</pre></div>"
        else:
            action_str += (
                f"<div class='parsed_action' style='background-color:yellow'><pre>{action['code']}</pre></div>"
            )
        action_str = f"<div class='predict_action'>{action_str}</div>"
        new_content += f"{action_str}\n"
        # if action_error:
        #     new_content += f"<div class='prev_action' style='background-color:pink'>{HTML.escape(action_error)}</div>"
        self.render_file.seek(0)
        html = self.render_file.read()
        html_body = re.findall(r"<body>(.*?)</body>", html, re.DOTALL)[0]
        html_body += new_content

        html = HTML_TEMPLATE.format(body=html_body)
        self.render_file.seek(0)
        self.render_file.truncate()
        self.render_file.write(html)
        self.render_file.flush()

    def close(self) -> None:
        self.render_file.close()
