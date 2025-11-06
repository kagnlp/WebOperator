from pathlib import Path
import json
import sys
from typing import Dict, Any
from tqdm import tqdm  # <-- import tqdm
import os
from dotenv import load_dotenv, find_dotenv
dotenv_path = find_dotenv()  # automatically searches upward
load_dotenv(dotenv_path, override=True)

REDDIT = os.environ.get("WA_REDDIT", "")
SHOPPING = os.environ.get("WA_SHOPPING", "")
SHOPPING_ADMIN = os.environ.get("WA_SHOPPING_ADMIN", "")
GITLAB = os.environ.get("WA_GITLAB", "")
MAP = os.environ.get("WA_MAP", "")

assert (
    REDDIT
    and SHOPPING
    and SHOPPING_ADMIN
    and GITLAB
    and MAP
), (
    f"Please setup the URLs to each site. Current: \n"
    + f"Reddit: {REDDIT}\n"
    + f"Shopping: {SHOPPING}\n"
    + f"Shopping Admin: {SHOPPING_ADMIN}\n"
    + f"Gitlab: {GITLAB}\n"
    + f"Map: {MAP}\n"
)

if __name__ == "__main__":
    """
    Read all .json files under `root` (recursively) and return a mapping
    from filepath -> parsed JSON object.
    """
    root = "websites"
    root_path = Path(root)
    if not root_path.exists():
        raise FileNotFoundError(f"Root directory does not exist: {root!r}")

    results: Dict[str, Any] = {}
    json_files = list(root_path.rglob("*.json"))

    for p in tqdm(json_files, desc="Processing JSON files"):
        if not p.is_file():
            continue
        try:
            with p.open("r", encoding="utf-8") as fh:
                results[str(p)] = json.load(fh)

            # Step 2: Update a key
            for pattern, url_key in {
                "__GITLAB__": "WA_GITLAB",
                "__REDDIT__": "WA_REDDIT",
                "__SHOPPING__": "WA_SHOPPING",
                "__SHOPPING_ADMIN__": "WA_SHOPPING_ADMIN",
                "__MAP__": "WA_MAP",
            }.items():
                results[str(p)]["step_data"]["axtree_txt"] = results[str(p)]["step_data"]["axtree_txt"].replace(pattern, os.environ[url_key])

            # Step 3: Write back to the file
            with p.open("w", encoding="utf-8") as fh:
                json.dump(results[str(p)], fh, ensure_ascii=False, indent=2)
                
        except Exception as exc:
            print(f"Warning: failed to read/parse {p}: {exc}", file=sys.stderr)