from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
from typing import List, Dict
from dotenv import load_dotenv

load_dotenv(override=True)


class HuggingFaceError(Exception):
    """Custom exception for Hugging Face model errors"""

    pass


class HuggingFaceModel:
    def __init__(self, model_name: str, **kwargs):
        self.name = model_name
        self.max_tokens = kwargs.get("max_tokens", 512)
        self.temperature = kwargs.get("temperature", 0.7)
        self.top_p = kwargs.get("top_p", 1.0)
        self.reasoning_effort = kwargs.get("reasoning_effort", None)
        self.n = kwargs.get("n", 1)
        self.input_tokens = 0
        self.output_tokens = 0
        self.tokenizer = None
        self.model = None

    def initialize(self):
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.name, trust_remote_code=True, use_fast=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.name,
            device_map="cuda",
            torch_dtype=torch.float16,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            load_in_8bit=True,
            max_memory={0: "20GB"},
        )

        self.pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
        )

        self.generation_args = {
            "max_new_tokens": 512,
            # "return_full_text": False,
            "temperature": self.temperature,
            "do_sample": True,
            "eos_token_id": self.pipe.tokenizer.convert_tokens_to_ids("<|eot_id|>")
            or self.pipe.tokenizer.eos_token_id,
            "output_scores": True,
            "return_dict_in_generate": True,
            "use_cache": True,
            # "top_k": 10,  # increase from default (usually 10)
            # "top_p": 0.95,  # optional, adds nucleus sampling
        }

    def predict(self, prompt: str, **kwargs) -> List[str]:
        """Simple text completion using tokenizer and model.generate."""
        if not self.tokenizer or not self.model:
            self.initialize()

        try:
            messages = [{"role": "user", "content": prompt}]
            output = self.pipe(messages, **self.generation_args)
            return output[0]["generated_text"][-1]["content"], output[0]["scores"]

        except Exception as e:
            raise HuggingFaceError(f"Error during text generation: {str(e)}")

    def chat(self, messages: List[Dict[str, str]], **kwargs) -> List[str]:
        """Chat completion using tokenizer.apply_chat_template."""

        if not self.tokenizer or not self.model:
            self.initialize()

        try:
            output = self.pipe(messages, **self.generation_args)
            return output[0]["generated_text"][-1]["content"], output[0]["scores"]

        except Exception as e:
            raise HuggingFaceError(f"Error during chat generation: {str(e)}")

    def from_text_to_tokens(self, text: str) -> List[str]:
        if not self.tokenizer:
            self.initialize()

        full_tokens = self.pipe.tokenizer(text, return_tensors="pt").input_ids[0]
        return full_tokens

    def _convert_to_top_logprobs(self, generated_text, scores):
        full_tokens = self.from_text_to_tokens(generated_text)
        result = []
        for token_id, logits in zip(full_tokens, scores):
            token_text = self.model.from_token_to_text(token_id)
            probs = torch.tensor(logits)
            top_probs, top_indices = torch.topk(probs, 10)
            top_logprobs = []
            for log_prob, idx in zip(top_probs, top_indices):
                alt_text = self.model.from_token_to_text(idx)
                top_logprobs.append({"token": alt_text, "log_prob": log_prob.item()})
            result.append({"token": token_text, "top_logprobs": top_logprobs})
        return result

    def from_token_to_text(self, token_id: int) -> str:
        """Convert a list of token IDs back to text."""
        if not self.tokenizer:
            self.initialize()

        return self.pipe.tokenizer.decode(token_id)

    def get_token_logprobs(self, prompt: str, target_tokens: list[str]) -> dict[str, float]:
        """Get log probabilities for specific target tokens (similar to ChatOpenAI logprobs)."""
        if not self.tokenizer or not self.model:
            self.initialize()

        try:
            inputs = self.tokenizer(prompt, return_tensors="pt")
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model(**inputs)
                # Get logits for the last token position
                logits = outputs.logits[0, -1, :]
                log_probs = torch.log_softmax(logits, dim=-1)

            # Get probabilities for specific tokens
            token_probs = {}
            for token_text in target_tokens:
                # Handle multi-token strings
                token_ids = self.tokenizer.encode(token_text, add_special_tokens=False)
                if len(token_ids) == 1:
                    token_id = token_ids[0]
                    prob = log_probs[token_id].item()
                    token_probs[token_text] = prob
                else:
                    # For multi-token strings, take the average or use a different strategy
                    total_prob = 0.0
                    for token_id in token_ids:
                        total_prob += log_probs[token_id].item()
                    token_probs[token_text] = total_prob / len(token_ids)

            return token_probs

        except Exception as e:
            raise HuggingFaceError(f"Error getting token probabilities: {str(e)}")

    def get_generation_logprobs(self) -> list[torch.Tensor]:
        """Get the log probabilities from the last generation."""
        if hasattr(self, "last_scores") and self.last_scores:
            log_probs = []
            for score_tensor in self.last_scores:
                log_prob = torch.log_softmax(score_tensor, dim=-1)
                log_probs.append(log_prob)
            return log_probs
        else:
            return []

    def calculate_logprobs_for_tokens(
        self, prompt: str, target_judge: dict[str, list[str]]
    ) -> dict[str, float]:
        """Calculate log probabilities for judge categories (WebPRM style)."""
        all_tokens = []
        for tokens in target_judge.values():
            all_tokens.extend(tokens)

        token_probs = self.get_token_logprobs(prompt, all_tokens)

        # Aggregate probabilities for each category using log-sum-exp for numerical stability
        judge_probs = {}
        for category, tokens in target_judge.items():
            log_probs = [token_probs.get(token, float("-inf")) for token in tokens]
            # Use log-sum-exp to combine probabilities
            if all(p == float("-inf") for p in log_probs):
                judge_probs[category] = float("-inf")
            else:
                max_log_prob = max(p for p in log_probs if p != float("-inf"))
                sum_exp = sum(
                    torch.exp(torch.tensor(p - max_log_prob)).item()
                    for p in log_probs
                    if p != float("-inf")
                )
                judge_probs[category] = max_log_prob + torch.log(torch.tensor(sum_exp)).item()

        return judge_probs