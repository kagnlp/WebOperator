import yaml
from .models.human import HumanModel
from .models.openrouter import OpenRouterModel
from .models.azure_openai import AzureOpenAIModel
from .models.openhf import OpenHFModel
from weboperator.action_generator import ActionGenerator, EnsembleActionGenerator
from weboperator.observation_processor import ObservationProcessor
from weboperator.backtrack_manager import BacktrackManager
from weboperator.action_processor import ActionProcessor
from weboperator.recovery_assistant import RecoveryAssistant
from weboperator.action_selector import ActionSelector
from weboperator.prompt_designer import PromptDesigner
from weboperator.action_validator import ActionValidator
from weboperator.task_rephraser import TaskRephraser
from weboperator.experience_retriever import ExperienceRetriever
from weboperator.webprm import WebPRM
from weboperator.tree_search_agent import TreeSearchAgentArgs


def create_model_from_config(model_config):
    """Create a model instance from configuration."""
    model_type = model_config["type"]
    model_name = model_config.get("model_name")
    temperature = model_config.get("temperature")

    if model_type == "HumanModel":
        return HumanModel()
    elif model_type == "OpenRouterModel":
        return OpenRouterModel(model_name, temperature=temperature)
    elif model_type == "AzureOpenAIModel":
        return AzureOpenAIModel(model_name, temperature=temperature)
    # elif model_type == "HuggingFaceModel":
    #     return HuggingFaceModel(model_name, temperature=temperature)
    elif model_type == "OpenHFModel":
        return OpenHFModel(model_name, temperature=temperature)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def create_action_generator_from_config(config, models):
    """Create an action generator from configuration."""
    if "action_generator" not in config["components"]:
        return None

    generator_config = config["components"]["action_generator"]

    print("+ Action Generator")
    print(" - Action Space: ", generator_config.get("full_action_space", []))
    print(" - Action Space Type: ", generator_config.get("action_space_type", "adaptive"))
    print(" - # Candidates: ", len(generator_config.get("candidates", [])))

    ActionGenerator.configure(
        full_action_space=generator_config.get("full_action_space", []),
        action_space_type=generator_config.get("action_space_type", "adaptive"),
        max_retry=config["components"]["action_validator"].get("max_retry", 6),
        allow_invalid_action=config["components"]["action_validator"].get(
            "allow_invalid_action", False
        ),
    )
    generators = []
    for gen_config in generator_config.get("candidates", []):
        model = models[gen_config["model"]]
        generator = ActionGenerator(
            model=model,
            name=gen_config.get("name"),
            history_length=gen_config.get("history_length", 5),
            rephraser_enabled=gen_config.get("rephraser", False),
            retriever_enabled=gen_config.get("retriever", False),
        )
        generators.append(generator)

    return EnsembleActionGenerator(generators=generators)


def create_action_selector_from_config(config):

    action_selector_config = config["components"].get("action_selector", {})
    selection_strategy = action_selector_config.get("selection_strategy", "action-aware")
    n_candidates = action_selector_config.get("n_candidates", 2)
    max_steps = action_selector_config.get("max_steps", 20)
    selection_scope = "global" if "backtrack_manager" in config["components"] else "local"
    termination_strategy = action_selector_config.get("termination_strategy", "all_agree")
    max_depth = action_selector_config.get("max_depth", 20)
    search_budget = action_selector_config.get("search_budget", 4)
    if selection_strategy != "action-aware":
        termination_strategy = "none"

    if "action_selector" in config["components"]:
        print("+ Action Selector")
        print(" - Selection Strategy: ", selection_strategy)
        print(" - Termination Candidates: ", n_candidates)
        print(" - Max Steps: ", max_steps)
        print(" - Selection Scope: ", selection_scope)
        print(" - Termination Strategy: ", termination_strategy)
        print(" - Max Depth: ", max_depth)
        print(" - Search Budget: ", search_budget)
        ActionSelector.configure(
            selection_strategy=selection_strategy,
            n_candidates=n_candidates,
            max_steps=max_steps,
            selection_scope=selection_scope,
            termination_strategy=termination_strategy,
            max_depth=max_depth,
            search_budget=search_budget,
        )
        action_selector = ActionSelector()
    else:
        action_selector = None

    return action_selector


def create_backtrack_manager_from_config(config):
    mode = config["agent"].get("mode", "evaluation")
    simulation_enabled = mode == "evaluation" and config["components"].get(
        "backtrack_manager", {}
    ).get("simulation_verified", False)
    destruction_aware = (
        config["components"].get("backtrack_manager", {}).get("destruction_aware", True)
    )

    if "backtrack_manager" in config["components"]:
        print("+ Backtrack Manager")
        print(" - Simulation Verified: ", simulation_enabled)
        print(" - Destruction Aware: ", destruction_aware)
        BacktrackManager.configure(
            simulation_enabled=simulation_enabled, destruction_aware=destruction_aware
        )
        backtrack_manager = BacktrackManager()
    else:
        backtrack_manager = None

    return backtrack_manager


def configure_action_processor(config):
    selection_strategy = config["components"].get("action_selector", {}).get("selection_strategy")
    termination_strategy = (
        config["components"].get("action_selector", {}).get("termination_strategy")
    )
    prune_low_terminating = (
        selection_strategy == "action-aware" and termination_strategy == "all_agree"
    )
    merge_strategy = config["components"].get("action_processor", {}).get("merge_strategy", "sum")

    print("+ Action Processor")
    print(" - Prune Terminating: ", prune_low_terminating)
    print(" - Merge Strategy: ", merge_strategy)

    # Action Processor is Mandatory
    ActionProcessor.configure(
        prune_low_terminating=prune_low_terminating, merge_strategy=merge_strategy
    )


def configure_observation_processor(config):
    optimized = config["components"].get("observation_processor", {}).get("optimized", True)
    truncate_error_message = (
        config["components"].get("observation_processor", {}).get("truncate_error_message", True)
    )

    print("+ Observation Processor")
    print(" - Optimized: ", optimized)
    print(" - Truncate Error Message: ", truncate_error_message)

    # Observation Processor is Mandatory
    ObservationProcessor.configure(
        optimized=optimized, truncate_error_message=truncate_error_message
    )


def configure_recovery_assistant(config):
    allow_unauthorized_page = config["agent"].get("allow_unauthorized_page", True)
    recovery_assistant_config = config["components"].get("recovery_assistant", {})
    recover_from_invalid_page = recovery_assistant_config.get("recover_from_invalid_page", True)
    recover_from_restricted_page = not allow_unauthorized_page
    recover_from_captcha = recovery_assistant_config.get("recover_from_captcha", True) and (
        config["env"]["headless"] == False
    )
    chat_mode = config["env"]["headless"] == False

    # Recovery Assistant is Optional
    if "recovery_assistant" in config["components"]:
        # Configuring Recovery Assistant = Enable recovery options
        print("+ Recovery Assistant")
        print(" - Recover from Invalid Page: ", recover_from_invalid_page)
        print(" - Recover from Unauthorized Page: ", recover_from_restricted_page)
        print(" - Recover from CAPTCHA: ", recover_from_captcha)
        RecoveryAssistant.configure(
            recover_from_invalid_page=recover_from_invalid_page,
            recover_from_restricted_page=recover_from_restricted_page,
            recover_from_captcha=recover_from_captcha,
            chat_mode=chat_mode,
        )


def configure_action_validator(config):
    allow_unauthorized_page = config["agent"].get("allow_unauthorized_page", True)
    allow_invalid_page = (
        config["components"].get("action_validator", {}).get("allow_invalid_page", False)
    )
    # Action Validator is Optional
    if "action_validator" in config["components"]:
        print("+ Action Validator")
        print(" - Allow Invalid Page: ", allow_invalid_page)
        print(" - Allow Unauthorized Page: ", allow_unauthorized_page)
        ActionValidator.configure(
            allow_invalid_page=allow_invalid_page, allow_unauthorized_page=allow_unauthorized_page
        )


def configure_judge(config, models):
    if "judge" in config["components"]:
        reward_model = models[config["components"]["judge"]["reward_model"]]
        checklist_model = models[config["components"]["judge"]["checklist_model"]]
        # checklist_cache = config["components"]["judge"].get("checklist_cache", "judge/webarena_checklist/webshepherd_3B_qwen2.5.json")
        prompt_type = config["components"]["judge"].get("prompt_type", "web_operator")

        print("+ Judge")
        print(" - Reward Model: ", models[config["components"]["judge"]["reward_model"]].name)
        print(" - Checklist Model: ", models[config["components"]["judge"]["checklist_model"]].name)
        # print(" - Checklist Cache: ", checklist_cache)
        print(" - Prompt Type: ", prompt_type)

        WebPRM.configure(
            reward_model=reward_model,
            checklist_model=checklist_model,
            prompt_type=prompt_type,
            # checklist_cache=checklist_cache
        )


def configure_retriever(config):
    if "retriever" in config["components"]:
        retriever_config = config["components"]["retriever"]
        retriever_type = retriever_config["type"]
        model_name = retriever_config.get("model", None)
        top_k = retriever_config.get("top_k", 5)
        print("+ Web Retriever")
        print(" - Retriever Type: ", retriever_type)
        print(" - Model Name: ", model_name)
        print(" - Top K: ", top_k)

        ExperienceRetriever.configure(retriever_type=retriever_type, model_name=model_name, top_k=top_k)


def configure_rephraser(config, models):
    if "rephraser" in config["components"]:
        model = models[config["components"]["rephraser"]["model"]]
        print("+ TaskRephraser")
        print(" - Model: ", model.name)
        TaskRephraser.set_model(model=model)


def configure_prompt_designer(config):
    benchmark_name = config["env"]["task_type"]
    print("+ Prompt Designer")
    print(" - Benchmark: ", benchmark_name)
    print(" - Multisite: ", config["experiment"].get("multisite", False))
    PromptDesigner.configure(benchmark=benchmark_name)


def get_agent_args(config):
    models = {}
    for model_name, model_config in config["models"].items():
        models[model_name] = create_model_from_config(model_config)

    print("Experiment Configuration:")

    action_generator = create_action_generator_from_config(config, models)
    action_selector = create_action_selector_from_config(config)
    backtrack_manager = create_backtrack_manager_from_config(
        config
    )  # Should be configured after action selector

    configure_action_processor(config)
    configure_observation_processor(config)
    configure_recovery_assistant(config)
    configure_action_validator(config)
    configure_judge(config, models)
    configure_prompt_designer(config)
    configure_retriever(config)
    configure_rephraser(config, models)

    agent_args = TreeSearchAgentArgs(
        chat_mode=config["env"]["task_type"] == "openended",
        action_generator=action_generator,
        action_selector=action_selector,
        backtrack_manager=backtrack_manager,
    )

    return agent_args
