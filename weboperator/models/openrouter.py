from openai import OpenAI, AzureOpenAI
from .base import BaseModel
import random
import os
import time
import openai
from typing import Any
from dotenv import load_dotenv

load_dotenv(override=True)


class OpenRouterError(Exception):
    """Custom exception for OpenRouter API errors"""

    pass


# OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "your-api-key-here")
OPENROUTER_API_KEYS = os.environ.get("OPENROUTER_API_KEYS", "").split(",")

if not OPENROUTER_API_KEYS or OPENROUTER_API_KEYS == [""]:
    raise ValueError("No OpenRouter API keys found in environment!")

# Keep global index for round-robin rotation
_current_key_index = 0


def get_next_api_key() -> str:
    """Rotate through the pool of API keys in round-robin fashion."""
    global _current_key_index
    key = OPENROUTER_API_KEYS[_current_key_index % len(OPENROUTER_API_KEYS)]
    _current_key_index = (_current_key_index + 1) % len(OPENROUTER_API_KEYS)
    return key


def get_client(api_key: str) -> OpenAI:
    """Create a new OpenRouter client with a specific API key."""
    return OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )


class OpenRouterModel(BaseModel):
    def __init__(self, model_name: str, **kwargs):
        super().__init__(model_name=model_name, **kwargs)
        self.client = get_client(get_next_api_key())

    def retry_with_exponential_backoff(  # type: ignore
        func,
        initial_delay: float = 1,
        exponential_base: float = 1.5,
        jitter: bool = True,
        max_retries: int = 5,
        errors: tuple[Any] = (openai.RateLimitError, OpenRouterError),
    ):
        """Retry a function with exponential backoff."""

        def wrapper(self, *args, **kwargs):  # type: ignore
            # Initialize variables
            num_retries = 0
            delay = initial_delay

            # Loop until a successful response or max_retries is hit or an exception is raised
            while True:
                try:
                    return func(self, *args, **kwargs)
                # Retry on specified errors
                # except errors as e:
                except openai.RateLimitError as e:
                    num_retries += 1
                    new_key = get_next_api_key()
                    self.client = get_client(new_key)
                    print(
                        f"Switched to new API key (index: {num_retries % len(OPENROUTER_API_KEYS)})"
                    )
                    # delay = min(delay * exponential_base * (1 + jitter * random.random()), 500)
                    print(f"#{num_retries} Error occurred: {e}.\n Retrying in {delay} seconds.")
                    time.sleep(delay)

                except Exception as e:
                    # import traceback
                    # traceback.print_exc()
                    # Increment retries
                    num_retries += 1

                    # Check if max retries has been reached
                    # if num_retries > max_retries:
                    #     raise Exception(
                    #         f"Maximum number of retries ({max_retries}) exceeded.")

                    # Rotate to next API key on *any* error

                    # Increment the delay
                    # delay = min(delay * exponential_base * (1 + jitter * random.random()), 500)
                    delay = min(delay * exponential_base, 300)
                    print(f"#{num_retries} Error occurred: {e}.\n Retrying in {delay} seconds.")
                    # Sleep for the delay
                    time.sleep(delay)

        return wrapper

    @retry_with_exponential_backoff
    def chat(self, messages: list[dict], **kwargs) -> str:
        """
        Chat completion using the chat/completions endpoint.
        Supports multi-modal inputs (text + images) for vision models.
        """
        response = self.client.chat.completions.create(
            model=self.name,
            messages=messages,
            # max_tokens=self.max_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            n=kwargs.get("n", self.n),
            logprobs=True,
            top_logprobs=10,
        )
        # print(response.choices[0])
        usage = getattr(response, "usage", None)
        if usage:
            self.input_tokens += usage.prompt_tokens
            self.output_tokens += usage.completion_tokens
            # print(f"Total input tokens: {self.input_tokens}, Total output tokens: {self.output_tokens}")

        # Raise OpenRouterError if we get invalid response to trigger retry
        if not response or not hasattr(response, "choices") or not response.choices:
            raise OpenRouterError("Invalid response from OpenRouter API")

        predictions = [
            choice.message.content.strip()
            for choice in response.choices
            if choice.message.content.strip()
        ]

        # print(response.choices[0])
        # for record in response.choices[0].logprobs.content:
        #     print("Selected: ", record.token)
        #     print("Top logprobs:")
        #     for lt in record.top_logprobs:
        #         print(f"  {lt.token}: {lt.logprob:.4f}")
        #     print()

        if len(predictions) == 0:
            raise OpenRouterError("Invalid response from OpenRouter API")

        with open("openrouter_io_log.txt", "w", encoding="utf-8") as f:
            f.write("=== New Interaction ===\n")
            f.write("Input:\n")

            content = messages[-1]["content"]
            if isinstance(content, list):
                # Pretty-print multimodal content
                for part in content:
                    if part["type"] == "text":
                        f.write(part["text"] + "\n")
                    else:
                        f.write(
                            f"[{part['type']}: {part.get('image_url', part.get('data', ''))}]\n"
                        )
            else:
                f.write(content + "\n")

            f.write("Output:\n")
            f.write(predictions[0] + "\n")

        if "<|start|>" in predictions[0]:
            raise OpenRouterError("Invalid response from OpenRouter API")

        top_logprobs = [
            choice.logprobs.content
            for choice in response.choices
            if hasattr(choice, "logprobs")
            and choice.logprobs
            and hasattr(choice.logprobs, "content")
        ]

        # if len(top_logprobs) > 0:
        #     # print("Top logprobs for the first choice:")
        #     for i, tl in enumerate(top_logprobs):
        #         if len(tl) == 0:
        #             continue
        #         print(f"Top logprobs for choice {i}:")
        #         for record in tl:
        #             print("Selected: ", record.token)
        #             print("Top logprobs:")
        #             for lt in record.top_logprobs:
        #                 print(f"  {lt.token}: {lt.logprob:.4f}")
        #             print()
        # print("Top logprobs length: ", len(top_logprobs[0]))

        # save input-output pairs to a txt file

        return predictions[0], top_logprobs[0] if len(top_logprobs) > 0 else None
