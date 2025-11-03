"""base class for evaluation"""
# answer string match
import collections
import json
import urllib
from pathlib import Path
from typing import Any, Tuple, Union

from beartype import beartype
from nltk.tokenize import word_tokenize  # type: ignore

from ..browser_env.actions import Action
from ..browser_env.utils import StateInfo
from .helper_functions import (
    PseudoPage,
    gitlab_get_project_memeber_role,
    llm_fuzzy_match,
    llm_ua_match,
    reddit_get_post_url,
    shopping_get_latest_order_url,
    shopping_get_sku_latest_review_author,
    shopping_get_sku_latest_review_rating,
)

Trajectory = list[Union[Action, StateInfo]]


class Evaluator(object):
    def __init__(self, eval_tag: str = "") -> None:
        self.eval_tag = eval_tag

    @beartype
    def __call__(
        self,
        trajectory: Trajectory,
        config_file: dict[str, Any],
        obs: dict[str, Any],
    ) -> float:
        raise NotImplementedError

    @staticmethod
    def get_last_action(trajectory: Trajectory) -> Action:
        try:
            # is_bearable(trajectory[-1], Action)
            last_action = trajectory[-1]
        except Exception:
            raise ValueError(
                "The last element of trajectory should be an action, add a fake stop action if needed"
            )

        return last_action  # type: ignore[return-value]

    @staticmethod
    def get_last_state(trajectory: Trajectory) -> StateInfo:
        try:
            # is_bearable(trajectory[-2], StateInfo)
            last_state = trajectory[-2]
        except Exception:
            raise ValueError(
                "The second last element of trajectory should be a state, add a fake stop action if needed"
            )

        return last_state  # type: ignore[return-value]

from weboperator.utils import to_ascii
class StringEvaluator(Evaluator):
    """Check whether the answer is correct with:
    exact match: the answer is exactly the same as the reference answer
    must include: each phrase in the reference answer must be included in the answer
    fuzzy match: the answer is similar to the reference answer, using LLM judge
    """

    @staticmethod
    @beartype
    def clean_answer(answer: str) -> str:
        answer = answer.strip()
        if answer.startswith("'") and answer.endswith("'"):
            answer = answer[1:-1]
        elif answer.startswith('"') and answer.endswith('"'):
            answer = answer[1:-1]
        return to_ascii(answer.lower())

    @staticmethod
    @beartype
    def exact_match(ref: str, pred: str) -> float:
        return float(
            StringEvaluator.clean_answer(pred)
            == StringEvaluator.clean_answer(ref)
        )

    @staticmethod
    @beartype
    def must_include(ref: str, pred: str, tokenize: bool = False) -> float:
        clean_ref = StringEvaluator.clean_answer(ref)
        clean_pred = StringEvaluator.clean_answer(pred)
        # tokenize the answer if the ref is a single word
        # prevent false positive (e.g, 0)
        if " |or| " in clean_ref or " |OR| " in clean_ref:
            refs = clean_ref.split(
                " |or| ") if " |or| " in clean_ref else clean_ref.split(" |OR| ")
            refs = [r.strip() for r in refs]
            for r in refs:
                if (
                    tokenize
                    and len(r) == 1
                    and len(word_tokenize(r)) == 1
                ):
                    tok_pred = word_tokenize(r)
                    if r in tok_pred:
                        return float(r in tok_pred)
                else:
                    if r in clean_pred:
                        return float(r in clean_pred)
            return 0.0
        if (
            tokenize
            and len(clean_ref) == 1
            and len(word_tokenize(clean_ref)) == 1
        ):
            tok_pred = word_tokenize(clean_pred)
            return float(clean_ref in tok_pred)
        else:
            return float(clean_ref in clean_pred)

    @staticmethod
    @beartype
    def fuzzy_match(ref: str, pred: str, intent: str) -> float:
        return llm_fuzzy_match(pred, ref, intent)

    @staticmethod
    @beartype
    def ua_match(ref: str, pred: str, intent: str) -> float:
        return llm_ua_match(pred, ref, intent)

    def __call__(
        self,
        task_info: dict[str, Any],
        config_file: dict[str, Any],
    ) -> float:
        configs = config_file

        pred = task_info["final_response"]

        score = 1.0
        for approach, value in configs["eval"]["reference_answers"].items():
            match approach:
                case "exact_match":
                    score *= self.exact_match(ref=value, pred=pred)

                case "must_include":
                    assert isinstance(value, list)
                    total_score = 0.0
                    for must_value in value:
                        total_score += self.must_include(
                            ref=must_value,
                            pred=pred,
                            tokenize=(len(value) == 1),
                        )
                    score *= total_score / len(value)

                case "fuzzy_match":
                    intent = configs["intent"]
                    if value == "N/A":
                        # if the instruction only asks the model to generate N/A when encountering an unachievable task
                        # without more concrete reasons
                        fuzzy_match_score = self.exact_match(ref=value, pred=pred)
                        # if the instruction also asks the model to generate the reason why the task is unachievable
                        # this should be the default as it will prevent false positive N/A`
                        if fuzzy_match_score != 1:
                            fuzzy_match_score = self.must_include(ref=value, pred=pred)
                            
                        if fuzzy_match_score != 1:
                            fuzzy_match_score = self.ua_match(
                                intent=configs["intent"],
                                ref=configs["eval"]["string_note"],
                                pred=pred,
                            )
                        score *= fuzzy_match_score
                    else:
                        if isinstance(value, list):
                            fuzzy_match_value = "; ".join(value)
                        else:
                            fuzzy_match_value = value
                        fuzzy_match_score = self.fuzzy_match(
                            ref=fuzzy_match_value, pred=pred, intent=intent
                        )
                        score *= fuzzy_match_score
        return score


class URLEvaluator(Evaluator):
    """Check URL matching"""

    @beartype
    def __call__(
        self,
        task_info: dict[str, Any],
        config_file: dict[str, Any],
    ) -> float:
        configs = config_file

        def clean_url(url: str) -> str:
            url = str(url)
            url = url.rstrip("/")
            return url.lower()

        def parse_url(url: str) -> tuple[str, dict[str, list[str]]]:
            """Parse a URL into its base, path, and query components."""
            parsed_url = urllib.parse.urlparse(url)
            base_path = (parsed_url.netloc + parsed_url.path).lower()
            query = {k.lower(): [v.lower() for v in vs] for k, vs in urllib.parse.parse_qs(parsed_url.query).items()}
            return base_path, query

        
        # def parse_urls(urls: list[str]) -> tuple[list[str], dict[str, set[str]]]:
        #     base_paths = []
        #     queries = collections.defaultdict(set)

        #     # First, collect all possible keys
        #     all_keys = set()
        #     parsed = []
        #     for url in urls:
        #         base_path, query = parse_url(url)
        #         # normalize: remove trailing slash unless it's just "/"
        #         if base_path != "/" and base_path.endswith("/"):
        #             base_path = base_path.rstrip("/")
        #         parsed.append((base_path, query))
        #         all_keys.update(query.keys())

        #     seen = set()
        #     for base_path, query in parsed:
        #         if base_path not in seen:
        #             base_paths.append(base_path)
        #             seen.add(base_path)
        #         for k in all_keys:
        #             if k in query:
        #                 queries[k].update(query[k])
        #             else:
        #                 queries[k].add(None)

        #     return base_paths, queries

        def parse_urls(
            urls: list[str],
        ) -> tuple[list[str], dict[str, set[str]]]:
            """Parse a list of URLs."""
            base_paths = []
            queries = collections.defaultdict(set)
            for url in urls:
                base_path, query = parse_url(url)
                base_paths.append(base_path)
                for k, v in query.items():
                    queries[k].update(v)
            return base_paths, queries

        pred = clean_url(task_info["final_url"])
        ref_urls = configs["eval"]["reference_url"].split(" |OR| ")
        ref_urls = [clean_url(url) for url in ref_urls]
        matching_rule = configs["eval"].get("url_note", "GOLD in PRED")
        if matching_rule == "GOLD in PRED":
            if "or" in configs["eval"].keys():
                or_ref_urls_list = [configs["eval"]["reference_url"]] + [item["reference_url"] for item in configs["eval"]["or"]]
            else:
                or_ref_urls_list = [configs["eval"]["reference_url"]]
            or_score_list = []
            for or_ref_urls in or_ref_urls_list:
                ref_urls = or_ref_urls.split(" |OR| ")
                ref_urls = [clean_url(url) for url in ref_urls]
                ref_base_paths, ref_queries = parse_urls(ref_urls)
                pred_base_paths, pred_query = parse_url(pred)

                base_score = float(
                    any(
                        [
                            ref_base_path in pred_base_paths
                            for ref_base_path in ref_base_paths
                        ]
                    )
                )
                query_score = 1.0
                for k, possible_values in ref_queries.items():
                    query_score *= float(
                        any(
                            possible_ref_value in pred_query.get(k, [])
                            for possible_ref_value in possible_values
                        )
                    )
                or_score_list.append(base_score * query_score)
            score = max(or_score_list)

            # ref_base_paths, ref_queries = parse_urls(ref_urls)
            # pred_base_paths, pred_query = parse_url(pred)
            # print(ref_queries)
            # base_score = float(
            #     any(
            #         [
            #             ref_base_path in pred_base_paths
            #             for ref_base_path in ref_base_paths
            #         ]
            #     )
            # )
            
            # query_score = 1.0
            # # for k, possible_values in ref_queries.items():
            # #     k_lower = k.lower()
            # #     pred_values = [v.lower() for v in pred_query.get(k_lower, [])]
            # #     query_score *= float(
            # #         any(
            # #             possible_ref_value.lower() in pred_values
            # #             for possible_ref_value in possible_values
            # #         )
            # #     )
            
            # for k, possible_values in ref_queries.items():
            #     k_lower = k.lower()
            #     pred_values = [v.lower() for v in pred_query.get(k_lower, [])]
            #     match = any(
            #         (
            #             possible_ref_value is None and not pred_values  # allow absence
            #         ) or (
            #             possible_ref_value is not None and possible_ref_value.lower() in pred_values
            #         )
            #         for possible_ref_value in possible_values
            #     )
            #     query_score *= float(match)
        
            # score = base_score * query_score

        else:
            raise ValueError(f"Unknown matching rule: {matching_rule}")

        return score


class HTMLContentEvaluator(Evaluator):
    """Check whether the contents appear in the page"""

    @beartype
    def __call__(
        self,
        task_info: dict[str, Any],
        config_file: dict[str, Any],
    ) -> float:
        configs = config_file

        targets = configs["eval"]["program_html"]

        score = 0.0
        
        # total_score = 0.0

        elements = task_info["recorded_contents"]
        for index, target in enumerate(targets):

            selected_element = elements[index]

            if "exact_match" in target["required_contents"]:
                required_contents = target["required_contents"]["exact_match"]
                cur_score = StringEvaluator.exact_match(
                    ref=required_contents, pred=selected_element
                )
                # score *= float(cur_score)
                score += float(cur_score)
                # print(f"[exact match] {cur_score}, selected element: {selected_element}, required contents: {required_contents}")
            elif "must_include" in target["required_contents"]:
                required_contents = target["required_contents"]["must_include"]
                assert isinstance(required_contents, list)
                total_score = 0.0
                for content in required_contents:
                    content_or = content.split(" |OR| ")
                    cur_score = any(
                        [
                            StringEvaluator.must_include(
                                ref=content,
                                pred=selected_element,
                                tokenize=False,
                            )
                            for content in content_or
                        ]
                    )
                    total_score += float(cur_score)
                # score *= total_score / len(required_contents)
                score += total_score / len(required_contents)
                # print(f"[must include] {cur_score}, selected element: {selected_element}, required contents: {content_or}")
            elif "fuzzy_match" in target["required_contents"]:
                required_contents = target["required_contents"]["fuzzy_match"]
                intent = configs["intent"]
                if isinstance(required_contents, list):
                    fuzzy_match_value = "; ".join(required_contents)
                else:
                    fuzzy_match_value = required_contents
                cur_score = StringEvaluator.fuzzy_match(
                    ref=fuzzy_match_value, pred=selected_element, intent=intent
                )
                # score *= float(cur_score)
                score += float(cur_score)
                # print(f"[fuzzy match] {cur_score}, selected element: {selected_element}, required contents: {required_contents}")
            else:
                raise ValueError(
                    f"Unknown required_contents: {target['required_contents'].keys()}"
                )
        return score / len(targets)


class EvaluatorComb:
    def __init__(self, evaluators: list[Evaluator]) -> None:
        self.evaluators = evaluators

    @beartype
    def __call__(
        self,
        task_info: dict[str, Any],
        config_file: dict[str, Any],
    ) -> float:
        score = 0.0
        for evaluator in self.evaluators:
            cur_score = evaluator(task_info, config_file)
            print(f"[{evaluator.__class__.__name__}] {cur_score}")
            score += cur_score
        return score / len(self.evaluators)


@beartype
def evaluator_router(config_file: dict) -> EvaluatorComb:
    """Router to get the evaluator class"""
    eval_types = config_file["eval"]["eval_types"]
    evaluators: list[Evaluator] = []
    for eval_type in eval_types:
        match eval_type:
            case "string_match":
                evaluators.append(StringEvaluator())
            case "url_match":
                evaluators.append(URLEvaluator())
            case "program_html":
                evaluators.append(HTMLContentEvaluator())
            case _:
                raise ValueError(f"eval_type {eval_type} is not supported")

    return EvaluatorComb(evaluators)
