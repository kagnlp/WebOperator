from abc import ABC, abstractmethod
import random
import time
import openai
from openai import OpenAI, AzureOpenAI
from typing import Any
from .base import BaseModel
import os

AZURE_ENDPOINT = os.getenv("AZURE_ENDPOINT", "")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY", "")
AZURE_API_VERSION = os.getenv("AZURE_API_VERSION", "")

AzureClient = AzureOpenAI(
    azure_endpoint=AZURE_ENDPOINT,
    api_key=AZURE_OPENAI_API_KEY,
    api_version=AZURE_API_VERSION,
)


class AzureOpenAIModel(BaseModel):
    def __init__(self, model_name: str, **kwargs):
        super().__init__(model_name=model_name, **kwargs)

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
                except Exception as e:
                    print(f"Error {type(e)} -> {e} ")
                    # Increment retries
                    num_retries += 1

                    # # Check if max retries has been reached
                    # if num_retries > max_retries:
                    #     raise Exception(
                    #         f"Maximum number of retries ({max_retries}) exceeded.")

                    # Increment the delay
                    delay *= exponential_base * (1 + jitter * random.random())
                    print(f"Retrying in {delay} seconds.")
                    # Sleep for the delay
                    time.sleep(delay)

        return wrapper

    @retry_with_exponential_backoff
    def chat(self, messages: list[dict], **kwargs) -> str | list[str]:
        """
        Chat completion using the chat/completions endpoint.
        Supports multi-modal inputs (text + images) for vision models.
        """
        try:
            if "gpt-5" in self.name:
                # For gpt-5 models, we might want to set different parameters
                response = AzureClient.chat.completions.create(
                    model=self.name,
                    messages=messages,
                    # max_completion_tokens=self.max_tokens,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    n=kwargs.get("n", self.n),
                )
            elif "o4" in self.name:
                response = AzureClient.chat.completions.create(
                    model=self.name,
                    messages=messages,
                    max_tokens=self.max_tokens,
                    reasoning_effort="medium",
                    stream=False,
                )
            else:
                if kwargs.get("logprobs", False):   
                    response = AzureClient.chat.completions.create(
                        model=self.name,
                        messages=messages,
                        # max_tokens=self.max_tokens,
                        temperature=self.temperature,
                        top_p=self.top_p,
                        # reasoning_effort=self.reasoning_effort,
                        n=kwargs.get("n", self.n),
                        logprobs=True,
                        top_logprobs=10,
                    )
                else:
                    response = AzureClient.chat.completions.create(
                        model=self.name,
                        messages=messages,
                        # max_tokens=self.max_tokens,
                        temperature=self.temperature,
                        top_p=self.top_p,
                        # reasoning_effort=self.reasoning_effort,
                        n=kwargs.get("n", self.n),
                    )
        except openai.BadRequestError as e:
            print(f"BadRequestError: {e}")
            return "", []
        except Exception as e:
            raise e

        if len(response.choices) == 0:
            raise ValueError("No choices returned from the model.")
        
        predictions = [
            choice.message.content.strip()
            for choice in response.choices
            if choice.message.content.strip()
        ]
        
        if len(predictions) == 0:
            raise ValueError("No valid predictions returned from the model.")

        top_logprobs = [
            choice.logprobs.content
            for choice in response.choices
            if hasattr(choice, "logprobs") and choice.logprobs
        ]
        if kwargs.get("logprobs", False):
            return predictions[0], top_logprobs[0]
        return predictions[0], []
