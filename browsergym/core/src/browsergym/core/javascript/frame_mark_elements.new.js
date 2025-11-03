
/**
 * Go through all DOM elements in the frame (including shadowDOMs), give them unique browsergym
 * identifiers (bid), and store custom data in ARIA attributes.
 */
async ([parent_bid, bid_attr_name, tags_to_mark]) => {
    const html_tags = new Set([
        "a", "abbr", "acronym", "address", "applet", "area", "article", "aside", "audio",
        "b", "base", "basefont", "bdi", "bdo", "big", "blockquote", "body", "br", "button",
        "canvas", "caption", "center", "cite", "code", "col", "colgroup", "data", "datalist",
        "dd", "del", "details", "dfn", "dialog", "dir", "div", "dl", "dt", "em", "embed",
        "fieldset", "figcaption", "figure", "font", "footer", "form", "frame", "frameset",
        "h1", "h2", "h3", "h4", "h5", "h6", "head", "header", "hgroup", "hr", "html", "i",
        "iframe", "img", "input", "ins", "kbd", "label", "legend", "li", "link", "main",
        "map", "mark", "menu", "meta", "meter", "nav", "noframes", "noscript", "object",
        "ol", "optgroup", "option", "output", "p", "param", "picture", "pre", "progress",
        "q", "rp", "rt", "ruby", "s", "samp", "script", "search", "section", "select",
        "small", "source", "span", "strike", "strong", "style", "sub", "summary", "sup",
        "svg", "table", "tbody", "td", "template", "textarea", "tfoot", "th", "thead",
        "time", "title", "tr", "track", "tt", "u", "ul", "var", "video", "wbr"
    ]);
    const set_of_marks_tags = new Set([
        "input", "textarea", "select", "button", "a", "iframe", "video", "li", "td", "option"
    ]);

    let warning_msgs = [];
    let browsergym_first_visit = false;

    try {
        if (!("browsergym_elem_counter" in window)) {
            window.browsergym_elem_counter = 0;
            window.browsergym_frame_id_generator = new IFrameIdGenerator();
            browsergym_first_visit = true;
        }

        let elems_to_be_visited = new Set();
        let intersection_observer = new IntersectionObserver(
            entries => {
                entries.forEach(entry => {
                    try {
                        let elem = entry.target;
                        elem.setAttribute('browsergym_visibility_ratio', Math.round(entry.intersectionRatio * 100) / 100);
                        if (elems_to_be_visited.has(elem)) elems_to_be_visited.delete(elem);
                    } catch (e) {
                        warning_msgs.push(`IntersectionObserver error on element: ${e.message}`);
                    }
                });
            },
            { threshold: Array.from({ length: 11 }, (_, i) => i * 0.1) }
        );

        let all_bids = new Set();
        let elements = Array.from(document.querySelectorAll('*'));
        let som_buttons = [];

        for (let i = 0; i < elements.length; i++) {
            let elem = elements[i];
            try {
                if (elem.shadowRoot) {
                    elements = [
                        ...elements.slice(0, i + 1),
                        ...Array.from(elem.shadowRoot.querySelectorAll("*")),
                        ...elements.slice(i + 1)
                    ];
                }

                switch (tags_to_mark) {
                    case "all": break;
                    case "standard_html":
                        if (!elem.tagName || !html_tags.has(elem.tagName.toLowerCase())) continue;
                        break;
                    default:
                        warning_msgs.push(`Invalid tags_to_mark value: ${tags_to_mark}`);
                        continue;
                }

                // Visibility
                elem.setAttribute('browsergym_visibility_ratio', 0);
                elems_to_be_visited.add(elem);
                intersection_observer.observe(elem);

                if (typeof elem.value !== 'undefined') elem.setAttribute("value", elem.value);
                if (typeof elem.checked !== 'undefined') {
                    if (elem.checked) elem.setAttribute("checked", "");
                    else elem.removeAttribute("checked");
                }

                // BIDs
                let elem_global_bid = elem.getAttribute(bid_attr_name) || null;
                if (browsergym_first_visit && elem_global_bid) {
                    warning_msgs.push(`Attribute ${bid_attr_name} already used in element`);
                    elem_global_bid = null;
                }
                if (!elem_global_bid || all_bids.has(elem_global_bid)) {
                    let elem_local_id = ['iframe', 'frame'].includes(elem.tagName.toLowerCase())
                        ? `${window.browsergym_frame_id_generator.next()}`
                        : `${window.browsergym_elem_counter++}`;
                    elem_global_bid = parent_bid ? `${parent_bid}${elem_local_id}` : `${elem_local_id}`;
                    elem.setAttribute(bid_attr_name, elem_global_bid);
                }
                all_bids.add(elem_global_bid);

                // Set ARIA attributes
                push_bid_to_attribute(elem_global_bid, elem, "aria-roledescription");
                push_bid_to_attribute(elem_global_bid, elem, "aria-description");

                // set-of-marks
                elem.setAttribute("browsergym_set_of_marks", "0");
                if (["self", "child"].includes(whoCapturesCenterClick(elem))) {
                    if (set_of_marks_tags.has(elem.tagName.toLowerCase()) ||
                        elem.onclick || window.getComputedStyle(elem).cursor == "pointer") {
                        let rect = elem.getBoundingClientRect();
                        if ((rect.right - rect.left) * (rect.bottom - rect.top) >= 20 &&
                            som_buttons.every(btn => !btn.contains(elem))) {

                            let parent = elem.parentElement;
                            if (!(parent && parent.tagName.toLowerCase() == "span" &&
                                parent.children.length === 1 &&
                                parent.getAttribute("role") &&
                                parent.getAttribute("browsergym_set_of_marks") === "1")) {

                                elem.setAttribute("browsergym_set_of_marks", "1");
                                if (elem.matches('button, a, input[type="button"], div[role="button"]'))
                                    som_buttons.push(elem);

                                while (parent) {
                                    if (parent.getAttribute("browsergym_set_of_marks") === "1")
                                        parent.setAttribute("browsergym_set_of_marks", "0");
                                    parent = parent.parentElement;
                                }
                            }
                        }
                    }
                }

            } catch (elemErr) {
                warning_msgs.push(`Element processing error: ${elemErr.message}`);
            }
        }

        // Wait for visibility
        let visibility_marking_timeout = 1000;
        try {
            await until(() => elems_to_be_visited.size === 0, visibility_marking_timeout);
        } catch {
            warning_msgs.push(`Frame marking: not all elements visited after ${visibility_marking_timeout} ms`);
        }

        intersection_observer.disconnect();

    } catch (err) {
        warning_msgs.push(`General marking error: ${err.message}`);
    }

    return warning_msgs;
};

async function until(f, timeout, interval = 40) {
    return new Promise((resolve, reject) => {
        const start_time = Date.now();
        // immediate check
        if (f()) {
            resolve();
        }
        // loop check
        const wait = setInterval(() => {
            if (f()) {
                clearInterval(wait);
                resolve();
            } else if (Date.now() - start_time > timeout) {
                clearInterval(wait);
                reject();
            }
        }, interval);
    });
}


function whoCapturesCenterClick(element) {
    var rect = element.getBoundingClientRect();
    var x = (rect.left + rect.right) / 2;
    var y = (rect.top + rect.bottom) / 2;
    var element_at_center = elementFromPoint(x, y); // return the element in the foreground at position (x,y)
    if (!element_at_center) {
        return "nobody";
    } else if (element_at_center === element) {
        return "self";
    } else if (element.contains(element_at_center)) {
        return "child";
    } else {
        return "non-descendant";
    }
}

function push_bid_to_attribute(bid, elem, attr) {
    let original_content = "";
    if (elem.hasAttribute(attr)) {
        original_content = elem.getAttribute(attr);
    }
    let new_content = `browsergym_id_${bid} ${original_content}`
    elem.setAttribute(attr, new_content);
}

function elementFromPoint(x, y) {
    let dom = document;
    let last_elem = null;
    let elem = null;

    do {
        last_elem = elem;
        elem = dom.elementFromPoint(x, y);
        dom = elem?.shadowRoot;
    } while (dom && elem !== last_elem);

    return elem;
}

// https://stackoverflow.com/questions/12504042/what-is-a-method-that-can-be-used-to-increment-letters#answer-12504061
class IFrameIdGenerator {
    constructor(chars = 'abcdefghijklmnopqrstuvwxyz') {
        this._chars = chars;
        this._nextId = [0];
    }

    next() {
        const r = [];
        for (let i = 0; i < this._nextId.length; i++) {
            let char = this._chars[this._nextId[i]];
            // all but first character must be upper-cased (a, aA, bCD)
            if (i < this._nextId.length - 1) {
                char = char.toUpperCase();
            }
            r.unshift(char);
        }
        this._increment();
        return r.join('');
    }

    _increment() {
        for (let i = 0; i < this._nextId.length; i++) {
            let val = ++this._nextId[i];
            if (val < this._chars.length) {
                return;
            }
            this._nextId[i] = 0;
        }
        this._nextId.push(0);
    }

    *[Symbol.iterator]() {
        while (true) {
            yield this.next();
        }
    }
}
