from typing import Callable

import anthropic
import openai
import torch
import vertexai
from google.auth import default, transport
from server.config import Settings
from singleton import Singleton
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline


class PHI(metaclass=Singleton):
    def __init__(self) -> None:
        model_path = "./ml/huggingface/models--microsoft--Phi-3-mini-4k-instruct/snapshots/c1358f8a35e6d2af81890deffbbfa575b978c62f"
        torch.random.manual_seed(0)
        model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype="auto", device_map="cuda", trust_remote_code=True
        )

        tokenizer = AutoTokenizer.from_pretrained(model_path)

        self.pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
        )


class LLM:
    models = {
        "phi",
        "claude-3-haiku-20240307",
        "claude-3-sonnet-20240229",
        "claude-3-opus-20240229",
        "claude-3-5-sonnet-20240620",
        "google/gemini-1.5-flash-001",
        "google/gemini-1.5-pro-001",
        "gpt-4o",
        "gpt-3.5-turbo",
        "gpt-4-turbo",
        "gpt-4",
        "gpt-4o-mini",
    }

    def __init__(self, model: str) -> None:
        if not model in self.models:
            raise ValueError(f"Model {model} not found")
        self.model = model
        self.separate_system_prompt: bool = self.model.startswith("claude")

        if self.model == "phi":
            self.setup_phi()

        if self.model.startswith("google"):
            self.setup_google()

    def setup_phi(self):
        self.pipe = PHI().pipe

    def setup_google(self):
        self.project_id = "scribble-ai-training-ali"
        self.location = "us-west1"

        vertexai.init(project=self.project_id, location=self.location)

        # Programmatically get an access token
        self.credentials, _ = default(
            scopes=["https://www.googleapis.com/auth/cloud-platform"]
        )
        auth_request = transport.requests.Request()
        self.credentials.refresh(auth_request)

    def get_messages_dict(
        self, text: str, system_prompt: str | None = None
    ) -> dict[str, str | list[dict[str, str | list[dict[str, str]]]]]:
        if self.separate_system_prompt:
            system_dict = {} if system_prompt is None else dict(system=system_prompt)
            return system_dict | dict(
                messages=[
                    {"role": "user", "content": [{"type": "text", "text": text}]}
                ],
            )

        messages = []
        if system_prompt is not None:
            messages += [{"role": "system", "content": system_prompt}]
        messages += [{"role": "user", "content": text}]
        return dict(messages=messages)

    def get_model(self, temperature: float = 0) -> Callable:
        if self.model == "phi":
            return self.pipe, dict(
                max_new_tokens=1024,
                return_full_text=False,
                **(
                    dict(temperature=temperature, do_sample=True)
                    if temperature != 0
                    else dict(do_sample=False)
                ),
            )
        if self.model.startswith("google"):
            client = openai.OpenAI(
                base_url=f"https://{self.location}-aiplatform.googleapis.com/v1beta1/projects/{self.project_id}/locations/{self.location}/endpoints/openapi",
                api_key=self.credentials.token,
            )
            return client.chat.completions.create, dict(
                model=self.model,
                max_tokens=1024,
                temperature=temperature,
            )
        if self.model.startswith("claude"):
            client = anthropic.Anthropic(api_key=Settings.ANTHROPIC_API_KEY)
            return client.messages.create, dict(
                model=self.model,
                max_tokens=1024,
                temperature=temperature,
            )
        if self.model.startswith("gpt"):
            client = openai.OpenAI(api_key=Settings.OPENAI_API_KEY)
            return client.chat.completions.create, dict(
                model=self.model,
                max_tokens=1024,
                temperature=temperature,
            )

    def prompt(
        self, text: str, system_prompt: str | None = None, temperature: float = 0
    ) -> str:
        print(f"Prompting {self.model}")
        model, params = self.get_model(temperature=temperature)
        messages = self.get_messages_dict(text=text, system_prompt=system_prompt)
        if self.model == "phi":
            return model(messages.get("messages", []), **params)
        return model(**messages, **params)
