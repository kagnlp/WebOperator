from openai import OpenAI, AzureOpenAI
from .base import BaseModel
import random
import os
import time
import openai
from typing import Any
from dotenv import load_dotenv

load_dotenv(override=True)


class OpenHFError(Exception):
    """Custom exception for OpenHF API errors"""

    pass


OpenHFClient = OpenAI(
    base_url=os.environ["HUGGING_FACE_API_SERVER"] + "/api/v1",
    api_key="your-api-key-here",
)


class OpenHFModel(BaseModel):
    def __init__(self, model_name: str, **kwargs):
        super().__init__(model_name=model_name, **kwargs)

    def retry_with_exponential_backoff(  # type: ignore
        func,
        initial_delay: float = 1,
        exponential_base: float = 2,
        jitter: bool = True,
        max_retries: int = 5,
        errors: tuple[Any] = (openai.RateLimitError, OpenHFError),
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
                # except errors as e:
                except Exception as e:
                    # Increment retries
                    num_retries += 1

                    # Check if max retries has been reached
                    # if num_retries > max_retries:
                    #     raise Exception(
                    #         f"Maximum number of retries ({max_retries}) exceeded.")

                    # Increment the delay
                    if delay < 1000:
                        delay *= exponential_base * (1 + jitter * random.random())
                    else:
                        delay = 1000

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
        response = OpenHFClient.chat.completions.create(
            model=self.name,
            messages=messages,
            # max_tokens=self.max_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            n=kwargs.get("n", self.n),
            logprobs=True,
            top_logprobs=10,
        )

        # Raise OpenHFError if we get invalid response to trigger retry
        if not response or not hasattr(response, "choices") or not response.choices:
            raise OpenHFError("Invalid response from OpenHF API")

        predictions = [
            choice.message.content.strip()
            for choice in response.choices
            if choice.message.content.strip()
        ]
        if len(predictions) == 0:
            raise OpenHFError("Invalid response from OpenHF API")
        top_logprobs = [
            choice.logprobs.content
            for choice in response.choices
            if hasattr(choice, "logprobs") and choice.logprobs
        ]

        with open("openhf_io_log.txt", "w", encoding="utf-8") as f:
            f.write("=== New Interaction ===\n")
            f.write("Input:\n")
            f.write(messages[-1]["content"] + "\n")
            f.write("Output:\n")
            f.write(predictions[0] + "\n")

        return predictions[0], top_logprobs[0]
