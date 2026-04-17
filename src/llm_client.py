import time
from typing import List, Dict, Optional

import ollama

from .schemas import ModelMetadata


class LLMClient:
    def __init__(self, model_metadata: Optional[ModelMetadata] = None):
        self.model_metadata = model_metadata or ModelMetadata()

    def generate(
        self,
        user_prompt: str,
        system_prompt: Optional[str] = None,
        history: Optional[List[Dict]] = None,
    ) -> tuple[str, int, Dict]:
        """
        Generate a response from the local Ollama model.

        Returns:
            (response_text, latency_ms, token_usage)
        """
        start_time = time.time()

        messages: List[Dict] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        if history:
            messages.extend(history)

        messages.append({"role": "user", "content": user_prompt})

        try:
            response = ollama.chat(
                model=self.model_metadata.model_name,
                messages=messages,
                options={
                    "temperature": self.model_metadata.temperature,
                    "num_predict": self.model_metadata.max_tokens,
                },
            )

            latency_ms = int((time.time() - start_time) * 1000)

            # ollama SDK >=0.4 returns an object; older versions return a dict.
            # Use attribute access (works for both via __getattr__ on newer SDK).
            prompt_tokens = getattr(response, "prompt_eval_count", 0) or 0
            completion_tokens = getattr(response, "eval_count", 0) or 0
            token_usage = {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            }

            return response.message.content, latency_ms, token_usage

        except Exception as e:
            latency_ms = int((time.time() - start_time) * 1000)
            return f"Error generating response: {e}", latency_ms, {}
