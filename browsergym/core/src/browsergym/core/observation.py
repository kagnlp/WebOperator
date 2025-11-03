import base64
import io
import logging
import pkgutil
import re
from typing import Literal

import numpy as np
import PIL.Image
import playwright.sync_api

from .constants import BROWSERGYM_ID_ATTRIBUTE as BID_ATTR
from .constants import BROWSERGYM_SETOFMARKS_ATTRIBUTE as SOM_ATTR
from .constants import BROWSERGYM_VISIBILITY_ATTRIBUTE as VIS_ATTR

MARK_FRAMES_MAX_TRIES = 3


logger = logging.getLogger(__name__)


class MarkingError(Exception):
    pass


def _pre_extract(
    page: playwright.sync_api.Page,
    tags_to_mark: Literal["all", "standard_html"] = "standard_html",
    lenient: bool = False, # Should be True
):
    """
    pre-extraction routine, marks dom elements (set bid and dynamic attributes like value and checked)
    """
    js_frame_mark_elements = pkgutil.get_data(__name__, "javascript/frame_mark_elements.js").decode(
        "utf-8"
    )

    # we can't run this loop in JS due to Same-Origin Policy
    # (can't access the content of an iframe from a another one)
    def mark_frames_recursive(frame, frame_bid: str):
        assert frame_bid == "" or re.match(r"^[a-z][a-zA-Z]*$", frame_bid)
        logger.debug(f"Marking frame {frame.name} with bid {repr(frame_bid)}")

        # mark all DOM elements in the frame (it will use the parent frame element's bid as a prefix)
        # try:
        warning_msgs = frame.evaluate(
            js_frame_mark_elements,
            [frame_bid, BID_ATTR, tags_to_mark],
        )
        # except Exception as e:
        #     logger.error(f"Error marking frame {repr(frame_bid)}: {e}")
        #     return

        # print warning messages if any
        for msg in warning_msgs:
            logger.warning(msg)

        # recursively mark all descendant frames
        for child_frame in frame.child_frames:
            # deal with detached frames
            if child_frame.is_detached():
                continue
            # deal with weird frames (pdf viewer in <embed>)
            child_frame_elem = child_frame.frame_element()
            if not child_frame_elem.content_frame() == child_frame:
                logger.warning(
                    f"Skipping frame '{child_frame.name}' for marking, seems problematic."
                )
                continue
            if child_frame.url.startswith("http") and not child_frame.url.startswith(page.url):
                logger.warning(
                    f"Skipping cross-origin frame '{child_frame.name}' for marking."
                )
                continue
            # deal with sandboxed frames with blocked script execution
            sandbox_attr = child_frame_elem.get_attribute("sandbox")
            if sandbox_attr is not None and "allow-scripts" not in sandbox_attr.split():
                continue
            child_frame_bid = child_frame_elem.get_attribute(BID_ATTR)
            if child_frame_bid is None:
                if lenient:
                    logger.warning(
                        "Cannot mark a child frame without a bid. Skipping frame.")
                    continue
                else:
                    raise MarkingError(
                        "Cannot mark a child frame without a bid.")
            mark_frames_recursive(child_frame, frame_bid=child_frame_bid)

    try:
        page.main_frame.wait_for_load_state("domcontentloaded", timeout=5000)
    except:
        logger.warning(f"Main frame not fully loaded")
        
    # mark all frames recursively
    mark_frames_recursive(page.main_frame, frame_bid="")


def _post_extract(page: playwright.sync_api.Page):
    js_frame_unmark_elements = pkgutil.get_data(
        __name__, "javascript/frame_unmark_elements.js"
    ).decode("utf-8")

    # we can't run this loop in JS due to Same-Origin Policy
    # (can't access the content of an iframe from a another one)
    for frame in page.frames:
        try:
            if not frame == page.main_frame:
                # deal with weird frames (pdf viewer in <embed>)
                if not frame.frame_element().content_frame() == frame:
                    logger.warning(
                        f"Skipping frame '{frame.name}' for unmarking, seems problematic."
                    )
                    continue
                # deal with sandboxed frames with blocked script execution
                sandbox_attr = frame.frame_element().get_attribute("sandbox")
                if sandbox_attr is not None and "allow-scripts" not in sandbox_attr.split():
                    continue
                # deal with frames without a BID
                bid = frame.frame_element().get_attribute(BID_ATTR)
                if bid is None:
                    continue

            frame.evaluate(js_frame_unmark_elements)
        except playwright.sync_api.Error as e:
            if any(msg in str(e) for msg in ("Frame was detached", "Frame has been detached")):
                pass
            else:
                raise e


def extract_screenshot(page: playwright.sync_api.Page):
    """
    Extracts the screenshot image of a Playwright page using Chrome DevTools Protocol.
    Uses the bgym_scale_factor to capture higher resolution screenshots for better VLM understanding.

    Args:
        page: the playwright page of which to extract the screenshot.

    Returns:
        A screenshot of the page, in the form of a 3D array (height, width, rgb).

    """

    scale_factor = getattr(page, "_bgym_scale_factor", 1.0)
    cdp = page.context.new_cdp_session(page)
    viewport = page.viewport_size

    if viewport is None:
        dimensions = page.evaluate(
            """() => ({
            width: window.innerWidth,
            height: window.innerHeight,
            devicePixelRatio: window.devicePixelRatio
        })"""
        )
    else:
        dimensions = {
            "width": viewport["width"],
            "height": viewport["height"],
            "devicePixelRatio": 1.0,  # Override system DPR
        }

    # Apply scale factor to device metrics for higher resolution capture
    cdp.send(
        "Emulation.setDeviceMetricsOverride",
        {
            "width": dimensions["width"],
            "height": dimensions["height"],
            "deviceScaleFactor": scale_factor,  # Use bgym_scale_factor here
            "mobile": False,
        },
    )

    cdp_answer = cdp.send(
        "Page.captureScreenshot",
        {
            "format": "png",
        },
    )

    # Reset device metrics
    # cdp.send("Emulation.clearDeviceMetricsOverride")
    cdp.detach()

    # bytes of a png file
    png_base64 = cdp_answer.get("data")
    if not png_base64:
        logger.error("No screenshot data received from CDP.")
        return None
    
    png_bytes = base64.b64decode(png_base64)
    with io.BytesIO(png_bytes) as f:
        # load png as a PIL image
        img = PIL.Image.open(f)
        # convert to RGB (3 channels)
        img = img.convert(mode="RGB")
        # convert to a numpy array
        img = np.array(img)

    return img


# we could handle more data items here if needed
__BID_EXPR = r"([a-zA-Z0-9]+)"
__DATA_REGEXP = re.compile(r"^browsergym_id_" + __BID_EXPR + r"\s?" + r"(.*)")


def extract_data_items_from_aria(string: str, log_level: int = logging.NOTSET):
    """
    Utility function to extract temporary data stored in the ARIA attributes of a node
    """

    match = __DATA_REGEXP.fullmatch(string)
    if not match:
        logger.log(
            level=log_level,
            msg=f"Failed to extract BrowserGym data from ARIA string: {repr(string)}",
        )
        return [], string

    groups = match.groups()
    data_items = groups[:-1]
    original_aria = groups[-1]
    return data_items, original_aria

def get_current_page_url(client):
    response = client.send("Runtime.evaluate", {
        "expression": "window.location.href",
        "returnByValue": True
    })
    return response.get("result", {}).get("value", "")


from tqdm import tqdm
def enrich_links_with_urls(accessibility_tree, client):
    for node in tqdm(accessibility_tree, desc="Enriching links"):
        if node.get("role", {}).get("value") == "RootWebArea":
            if "properties" not in node:
                node["properties"] = []
            node["properties"].append({
                "name": "url",
                "value": {
                    "type": "string",
                    "value": get_current_page_url(client)
                }
            })

        if "backendDOMNodeId" not in node:
            continue  # skip nodes without DOM reference

        try:
            dom_node = client.send("DOM.resolveNode", {
                "backendNodeId": node["backendDOMNodeId"]
            })
            object_id = dom_node["object"]["objectId"]

            tag_name_resp = client.send("Runtime.callFunctionOn", {
                "objectId": object_id,
                "functionDeclaration": "function() { return this.tagName.toLowerCase(); }",
                "returnByValue": True
            })

            tag_name = tag_name_resp["result"].get("value")

            url_to_add = None

            if tag_name == "a":
                href_resp = client.send("Runtime.callFunctionOn", {
                    "objectId": object_id,
                    "functionDeclaration": "function() { return this.href; }",
                    "returnByValue": True
                })
                url_to_add = href_resp["result"].get("value")
            elif tag_name == "img":
                src_resp = client.send("Runtime.callFunctionOn", {
                    "objectId": object_id,
                    "functionDeclaration": "function() { return this.src; }",
                    "returnByValue": True
                })
                url_to_add = src_resp["result"].get("value")

            if url_to_add:
                # add to node properties
                if "properties" not in node:
                    node["properties"] = []
                node["properties"].append({
                    "name": "url",
                    "value": {
                        "type": "string",
                        "value": url_to_add
                    }
                })
        except Exception as e:
            # Optional: add error info or just skip silently
            pass



def extract_dom_snapshot(
    page: playwright.sync_api.Page,
    computed_styles=[],
    include_dom_rects: bool = True,
    include_paint_order: bool = True,
    temp_data_cleanup: bool = True,
):
    """
    Extracts the DOM snapshot of a Playwright page using Chrome DevTools Protocol.

    Args:
        page: the playwright page of which to extract the screenshot.
        computed_styles: whitelist of computed styles to return.
        include_dom_rects: whether to include DOM rectangles (offsetRects, clientRects, scrollRects) in the snapshot.
        include_paint_order: whether to include paint orders in the snapshot.
        temp_data_cleanup: whether to clean up the temporary data stored in the ARIA attributes.

    Returns:
        A document snapshot, including the full DOM tree of the root node (including iframes,
        template contents, and imported documents) in a flattened array, as well as layout
        and white-listed computed style information for the nodes. Shadow DOM in the returned
        DOM tree is flattened.

    """
    cdp = page.context.new_cdp_session(page)
    
    try:
        dom_snapshot = cdp.send(
            "DOMSnapshot.captureSnapshot",
            {
                "computedStyles": computed_styles,
                "includeDOMRects": include_dom_rects,
                "includePaintOrder": include_paint_order,
            },
        )
    except Exception as e:
        dom_snapshot = None
        logger.error(f"Error extracting DOM snapshot: {e}")
    finally:
        if not page.is_closed():
            try:
                cdp.detach()
            except Exception as e:
                logger.warning(f"Failed to detach CDP: {e}")

    # if requested, remove temporary data stored in the ARIA attributes of each node
    if dom_snapshot and temp_data_cleanup:
        try:
            pop_bids_from_attribute(dom_snapshot, "aria-roledescription")
            pop_bids_from_attribute(dom_snapshot, "aria-description")
        except Exception as e:
            logger.error(f"Error cleaning up temporary data: {e}")

    return dom_snapshot


def pop_bids_from_attribute(dom_snapshot, attr: str):
    try:
        target_attr_name_id = dom_snapshot["strings"].index(attr)
    except ValueError:
        target_attr_name_id = -1
    # run the cleanup only if the target attribute string is present
    if target_attr_name_id > -1:
        processed_string_ids = set()
        for document in dom_snapshot["documents"]:
            for node_attributes in document["nodes"]["attributes"]:
                i = 0
                # find the target attribute, if any
                for i in range(0, len(node_attributes), 2):
                    attr_name_id = node_attributes[i]
                    attr_value_id = node_attributes[i + 1]
                    if attr_name_id == target_attr_name_id:
                        attr_value = dom_snapshot["strings"][attr_value_id]
                        # remove any data stored in the target attribute
                        if attr_value_id not in processed_string_ids:
                            _, new_attr_value = extract_data_items_from_aria(
                                attr_value)
                            dom_snapshot["strings"][
                                attr_value_id
                            ] = new_attr_value  # update the string in the metadata
                            processed_string_ids.add(
                                attr_value_id
                                # mark string as processed (in case several nodes share the same target attribute string value)
                            )
                            attr_value = new_attr_value
                        # remove target attribute (name and value) if empty
                        if attr_value == "":
                            del node_attributes[i: i + 2]
                        # once target attribute is found, exit the search
                        break


def extract_dom_extra_properties(dom_snapshot):
    def to_string(idx):
        if idx == -1:
            return None
        else:
            return dom_snapshot["strings"][idx]

    # pre-locate important string ids
    try:
        bid_string_id = dom_snapshot["strings"].index(BID_ATTR)
    except ValueError:
        bid_string_id = -1
    try:
        vis_string_id = dom_snapshot["strings"].index(VIS_ATTR)
    except ValueError:
        vis_string_id = -1
    try:
        som_string_id = dom_snapshot["strings"].index(SOM_ATTR)
    except ValueError:
        som_string_id = -1

    # build the iframe tree (DFS from the first frame)
    doc_properties = {
        0: {
            "parent": None,
        }
    }

    docs_to_process = [0]
    while docs_to_process:
        doc = docs_to_process.pop(-1)  # DFS

        children = dom_snapshot["documents"][doc]["nodes"]["contentDocumentIndex"]
        for node, child_doc in zip(children["index"], children["value"]):
            doc_properties[child_doc] = {
                "parent": {
                    "doc": doc,  # parent frame index
                    "node": node,  # node index within the parent frame
                }
            }
            docs_to_process.append(child_doc)

        # recover the absolute x and y position of the frame node in the parent (if any)
        parent = doc_properties[doc]["parent"]
        if parent:
            parent_doc = parent["doc"]
            parent_node = parent["node"]
            try:
                node_layout_idx = dom_snapshot["documents"][parent_doc]["layout"][
                    "nodeIndex"
                ].index(parent_node)
            except ValueError:
                node_layout_idx = -1
            if node_layout_idx >= 0:
                node_bounds = dom_snapshot["documents"][parent_doc]["layout"]["bounds"][
                    node_layout_idx
                ]  # can be empty?
                # absolute position of parent + relative position of frame node within parent
                parent_node_abs_x = doc_properties[parent_doc]["abs_pos"]["x"] + \
                    node_bounds[0]
                parent_node_abs_y = doc_properties[parent_doc]["abs_pos"]["y"] + \
                    node_bounds[1]
            else:
                parent_node_abs_x = 0
                parent_node_abs_y = 0
        else:
            parent_node_abs_x = 0
            parent_node_abs_y = 0

        # get the frame's absolute position, by adding any scrolling offset if any
        doc_properties[doc]["abs_pos"] = {
            "x": parent_node_abs_x - dom_snapshot["documents"][doc]["scrollOffsetX"],
            "y": parent_node_abs_y - dom_snapshot["documents"][doc]["scrollOffsetY"],
        }

        document = dom_snapshot["documents"][doc]
        doc_properties[doc]["nodes"] = [
            {
                "bid": None,  # default value, to be filled (str)
                "visibility": None,  # default value, to be filled (float)
                "bbox": None,  # default value, to be filled (list)
                "clickable": False,  # default value, to be filled (bool)
                "set_of_marks": None,  # default value, to be filled (bool)
            }
            for _ in enumerate(document["nodes"]["parentIndex"])
        ]  # all nodes in document

        # extract clickable property
        for node_idx in document["nodes"]["isClickable"]["index"]:
            doc_properties[doc]["nodes"][node_idx]["clickable"] = True

        # extract bid and visibility properties (attribute-based)
        for node_idx, node_attrs in enumerate(document["nodes"]["attributes"]):
            i = 0
            # loop over all attributes
            for i in range(0, len(node_attrs), 2):
                name_string_id = node_attrs[i]
                value_string_id = node_attrs[i + 1]
                if name_string_id == bid_string_id:
                    doc_properties[doc]["nodes"][node_idx]["bid"] = to_string(
                        value_string_id)
                if name_string_id == vis_string_id:
                    doc_properties[doc]["nodes"][node_idx]["visibility"] = float(
                        to_string(value_string_id)
                    )
                if name_string_id == som_string_id:
                    doc_properties[doc]["nodes"][node_idx]["set_of_marks"] = (
                        to_string(value_string_id) == "1"
                    )

        # extract bbox property (in absolute coordinates)
        for node_idx, bounds, client_rect in zip(
            document["layout"]["nodeIndex"],
            document["layout"]["bounds"],
            document["layout"]["clientRects"],
        ):
            # empty clientRect means element is not actually rendered
            if not client_rect:
                doc_properties[doc]["nodes"][node_idx]["bbox"] = None
            else:
                # bounds gives the relative position within the document
                doc_properties[doc]["nodes"][node_idx]["bbox"] = bounds.copy()
                # adjust for absolute document position
                doc_properties[doc]["nodes"][node_idx]["bbox"][0] += doc_properties[doc]["abs_pos"][
                    "x"
                ]
                doc_properties[doc]["nodes"][node_idx]["bbox"][1] += doc_properties[doc]["abs_pos"][
                    "y"
                ]

        # Note: other interesting fields
        # document["nodes"]["parentIndex"]  # parent node
        # document["nodes"]["nodeType"]
        # document["nodes"]["nodeName"]
        # document["nodes"]["nodeValue"]
        # document["nodes"]["textValue"]
        # document["nodes"]["inputValue"]
        # document["nodes"]["inputChecked"]
        # document["nodes"]["optionSelected"]
        # document["nodes"]["pseudoType"]
        # document["nodes"]["pseudoIdentifier"]
        # document["nodes"]["isClickable"]
        # document["textBoxes"]
        # document["layout"]["nodeIndex"]
        # document["layout"]["bounds"]
        # document["layout"]["offsetRects"]
        # document["layout"]["scrollRects"]
        # document["layout"]["clientRects"]
        # document["layout"]["paintOrders"]

    # collect the extra properties of all nodes with a browsergym_id attribute
    extra_properties = {}
    for doc in doc_properties.keys():
        for node in doc_properties[doc]["nodes"]:
            bid = node["bid"]
            if bid:
                if bid in extra_properties:
                    logger.warning(
                        f"duplicate {BID_ATTR}={repr(bid)} attribute detected")
                extra_properties[bid] = {
                    extra_prop: node[extra_prop]
                    for extra_prop in ("visibility", "bbox", "clickable", "set_of_marks")
                }

    return extra_properties

def extract_all_frame_axtrees(page: playwright.sync_api.Page):
    """
    Extracts the AXTree of all frames (main document and iframes) of a Playwright page using Chrome DevTools Protocol.

    Args:
        page: the playwright page of which to extract the frame AXTrees.

    Returns:
        A dictionnary of AXTrees (as returned by Chrome DevTools Protocol) indexed by frame IDs.

    """
    global last_dialog_message
    cdp = page.context.new_cdp_session(page)
    
    # capture active dialogs (alerts, confirms, prompts)

    # extract the frame tree
    frame_tree = cdp.send(
        "Page.getFrameTree",
        {},
    )

    # extract all frame IDs into a list
    # (breadth-first-search through the frame tree)
    frame_ids = []
    root_frame = frame_tree["frameTree"]
    frames_to_process = [root_frame]
    while frames_to_process:
        frame = frames_to_process.pop()
        frames_to_process.extend(frame.get("childFrames", []))
        # extract the frame ID
        frame_id = frame["frame"]["id"]
        frame_ids.append(frame_id)

    # extract the AXTree of each frame
    frame_axtrees = {}
    
    for frame_id in frame_ids:
        try:
            frame_axtrees[frame_id] = cdp.send(
                "Accessibility.getFullAXTree",
                {"frameId": frame_id},
            )
        except Exception as e:
            logger.error(f"Error extracting frame AXTree for frame ID {frame_id}: {e}")
            
    cdp.detach()

    # print(f"Extracted {len(frame_axtrees)} frame AXTrees")
    # extract browsergym data from ARIA attributes
    for ax_tree in frame_axtrees.values():
        for node in ax_tree["nodes"]:
            data_items = []
            # look for data in the node's "roledescription" property
            if "properties" in node:
                for i, prop in enumerate(node["properties"]):
                    if prop["name"] == "roledescription":
                        data_items, new_value = extract_data_items_from_aria(
                            prop["value"]["value"])
                        prop["value"]["value"] = new_value
                        # remove the "description" property if empty
                        if new_value == "":
                            del node["properties"][i]
                        break
            # look for data in the node's "description" (fallback plan)
            if "description" in node:
                data_items_bis, new_value = extract_data_items_from_aria(
                    node["description"]["value"]
                )
                node["description"]["value"] = new_value
                if new_value == "":
                    del node["description"]
                if not data_items:
                    data_items = data_items_bis
            # add the extracted "browsergym" data to the AXTree
            if data_items:
                (browsergym_id,) = data_items
                node["browsergym_id"] = browsergym_id
    
    # --- inject captured alerts as pseudo-nodes ---  
         
    # page.dialog_message = "Testing dialog injection"
    page_dialog_message = getattr(page, "dialog_message", "")
    if page_dialog_message:
        import copy
        # fake_node = {
        #     "role": {"type": "internalRole", "value": "alert"},
        #     "name": {"type": "computedString", "value": copy.deepcopy(page_dialog_message)},
        #     # "description": {"value": last_dialog_message["message"]},
        #     # "browsergym_id": f"alert-0",
        #     "nodeId": f"alert-0-node",
        #     "childIds": []
        # }
        # print(
        #     f"Injecting fake alert node: {fake_node}"
        # )
        # Attach to the main frame’s tree
        # frame_axtrees[frame_ids[0]]["nodes"].append(fake_node)
        # frame_axtrees[frame_ids[0]]["nodes"][0]["childIds"].append(fake_node["nodeId"]) 
        frame_axtrees[frame_ids[0]]["nodes"][0]["properties"].append({
            "name": "page_dialog_message",
            "value": {
                "type": "string",
                "value": copy.deepcopy(page_dialog_message) + " Retry."
            }
        })
        page.dialog_message = None
    # else:
    #     print("[ERROR] No dialog message")
            
    return frame_axtrees




def extract_merged_axtree(page: playwright.sync_api.Page):
    """
    Extracts the merged AXTree of a Playwright page (main document and iframes AXTrees merged) using Chrome DevTools Protocol.

    Args:
        page: the playwright page of which to extract the merged AXTree.

    Returns:
        A merged AXTree (same format as those returned by Chrome DevTools Protocol).

    """
    frame_axtrees = extract_all_frame_axtrees(page)

    cdp = page.context.new_cdp_session(page)

    # merge all AXTrees into one
    merged_axtree = {"nodes": []}
    for ax_tree in frame_axtrees.values():
        merged_axtree["nodes"].extend(ax_tree["nodes"])
        # connect each iframe node to the corresponding AXTree root node
        for node in ax_tree["nodes"]:
            if node["role"]["value"] == "Iframe":
                frame_id = (
                    cdp.send("DOM.describeNode", {
                             "backendNodeId": node["backendDOMNodeId"]})
                    .get("node", {})
                    .get("frameId", None)
                )
                if not frame_id:
                    logger.warning(
                        f"AXTree merging: unable to recover frameId of node with backendDOMNodeId {repr(node['backendDOMNodeId'])}, skipping"
                    )
                # it seems Page.getFrameTree() from CDP omits certain Frames (empty frames?)
                # if a frame is not found in the extracted AXTrees, we just ignore it
                elif frame_id in frame_axtrees:
                    # root node should always be the first node in the AXTree
                    frame_root_node = frame_axtrees[frame_id]["nodes"][0]
                    assert frame_root_node["frameId"] == frame_id
                    node["childIds"].append(frame_root_node["nodeId"])
                else:
                    logger.warning(
                        f"AXTree merging: extracted AXTree does not contain frameId '{frame_id}', skipping"
                    )

    # enrich_links_with_urls(merged_axtree["nodes"], cdp)
    cdp.detach()

    return merged_axtree


def extract_focused_element_bid(page: playwright.sync_api.Page):
    # this JS code will dive through ShadowDOMs
    extract_focused_element_with_bid_script = """\
() => {
    // This recursive function traverses shadow DOMs
    function getActiveElement(root) {
        const active_element = root.activeElement;

        if (!active_element) {
            return null;
        }

        if (active_element.shadowRoot) {
            return getActiveElement(active_element.shadowRoot);
        } else {
            return active_element;
        }
    }
    return getActiveElement(document);
}"""
    # this playwright code will dive through iFrames
    frame = page
    focused_bid = ""
    try:
        while frame:
            focused_element = frame.evaluate_handle(
                extract_focused_element_with_bid_script, BID_ATTR
            ).as_element()
            if focused_element:
                frame = focused_element.content_frame()
                focused_bid = focused_element.get_attribute(BID_ATTR)
            else:
                frame = None
    except playwright.sync_api.TimeoutError:
        focused_bid = ""

    # convert null / None to empty string
    if not focused_bid:
        focused_bid = ""

    return focused_bid
