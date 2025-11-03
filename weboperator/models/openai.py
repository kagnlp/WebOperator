from abc import ABC, abstractmethod
import random
import time
import openai
from openai import OpenAI
from typing import Any
from .base import BaseModel
import os


class OpenAIModel(BaseModel):
    def __init__(self, model_name: str, **kwargs):
        super().__init__(model_name=model_name, **kwargs)
        # Set up OpenAI client
        self.client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"), base_url=kwargs.get("base_url", None)
        )
        # Store base_url for recreation after unpickling
        self._base_url = kwargs.get("base_url", None)

    def __getstate__(self):
        """Custom pickling to exclude the client object."""
        state = self.__dict__.copy()
        # Remove the unpicklable client
        state["client"] = None
        return state

    def __setstate__(self, state):
        """Custom unpickling to recreate the client object."""
        self.__dict__.update(state)
        # Recreate the client
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"), base_url=self._base_url)

    def retry_with_exponential_backoff(  # type: ignore
        func,
        initial_delay: float = 1,
        exponential_base: float = 2,
        jitter: bool = True,
        max_retries: int = 3,
        errors: tuple[Any] = (openai.RateLimitError,),
    ):
        """Retry a function with exponential backoff."""

        def wrapper(*args, **kwargs):  # type: ignore
            # Initialize variables
            num_retries = 0
            delay = initial_delay

            # Loop until a successful response or max_retries is hit or an exception is raised
            while True:
                try:
                    return func(*args, **kwargs)
                # Retry on specified errors
                except errors as e:
                    # Increment retries
                    num_retries += 1

                    # Check if max retries has been reached
                    if num_retries > max_retries:
                        raise Exception(f"Maximum number of retries ({max_retries}) exceeded.")

                    # Increment the delay
                    delay *= exponential_base * (1 + jitter * random.random())
                    print(f"Retrying in {delay} seconds.")
                    # Sleep for the delay
                    time.sleep(delay)

                # Raise exceptions for any errors not specified
                except Exception as e:
                    raise e

        return wrapper

    @retry_with_exponential_backoff
    def chat(self, messages: list[dict]) -> str | list[str]:
        """
        Chat completion using the chat/completions endpoint.
        Supports multi-modal inputs (text + images) for vision models.
        """
        response = self.client.chat.completions.create(
            model=self.name,
            messages=messages,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            n=self.n,
        )

        return [choice.message.content.strip() for choice in response.choices]


# self.llm = ChatOpenAI(
#         model=self.agent_config["model_name"],
#         base_url=self.agent_config["base_url"],
#         api_key=self.agent_config["api_key"],
#         temperature=self.agent_config["temperature"],
#         timeout=300,
#         logprobs=True,
#         top_logprobs=10
#     )
# response = client.chat.completions.create(
#     model=model,
#     messages=messages,
#     max_tokens=256,
#     top_p=top_p,
#     n=n // len(models)
# )
