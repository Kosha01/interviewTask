import os
import json
from dotenv import load_dotenv
import requests

load_dotenv()

class LLMProvider:
    def __init__(self):
        self.provider = os.getenv("LLM_PROVIDER", "ollama")
        self.model = os.getenv("LLM_MODEL", "llama3:8b")

    def generate(self, prompt: str) -> str:
        if self.provider == "ollama":
            return self._ollama_generate(prompt)
        elif self.provider == "openai":
            return self._openai_generate(prompt)
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")

    def _ollama_generate(self, prompt: str) -> str:
        """
        Handle Ollama API streaming response properly.
        """
        url = "http://localhost:11434/api/generate"
        payload = {"model": self.model, "prompt": prompt}

        response = requests.post(url, json=payload, stream=True)
        response.raise_for_status()

        full_reply = ""
        for line in response.iter_lines():
            if line:
                try:
                    data = json.loads(line.decode("utf-8"))
                    if "response" in data:
                        full_reply += data["response"]
                except json.JSONDecodeError:
                    continue

        return full_reply.strip()

    def _openai_generate(self, prompt: str) -> str:
        from openai import OpenAI
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        completion = client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
        )
        return completion.choices[0].message.content
