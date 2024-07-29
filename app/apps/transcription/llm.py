from typing import Callable

import anthropic
import groq
import openai
import torch
import vertexai
from google import generativeai as genai
from google.auth import default, transport
from server.config import Settings
from singleton import Singleton


class PHI(metaclass=Singleton):
    def __init__(self) -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

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
        "google/gemini-1.5-flash",
        "google/gemini-1.5-pro-001",
        "gpt-4o",
        "gpt-3.5-turbo",
        "gpt-4-turbo",
        "gpt-4",
        "gpt-4o-mini",
        "groq/llama-3.1-70b-versatile",
        "groq/llama-3.1-8b-instant",
        "groq/gemma2-9b-it",
        "groq/gemma-7b-it",
        "groq/llama3-70b-8192",
        "groq/llama3-8b-8192",
        "groq/llama3-groq-70b-8192-tool-use-preview",
        "groq/llama3-groq-8b-8192-tool-use-preview",
        "groq/mixtral-8x7b-32768",
    }

    def __init__(
        self,
        model: str = "claude-3-haiku-20240307",
        api_key: str | None = None,
        max_tokens: int = 1024,
    ) -> None:
        if not model in self.models:
            raise ValueError(f"Model {model} not found")
        self.model = model
        self.separate_system_prompt: bool = self.model.startswith(
            "claude"
        ) or self.model.startswith("google")
        self.api_key = api_key
        self.max_tokens = max_tokens
        self.setup_api_key()
        self.setup_model()

    def setup_model(self):
        if self.model == "phi":
            self.setup_phi()
        if self.model.startswith("claude"):
            return
        if self.model.startswith("gpt"):
            return
        if self.model.startswith("google"):
            self.setup_google()
            return
        if self.model.startswith("groq"):
            return

        raise ValueError(f"Model {self.model} not found")

    def setup_api_key(self):
        if self.model == "phi":
            return
        if self.model.startswith("claude"):
            self.api_key = Settings.ANTHROPIC_API_KEY
            return
        if self.model.startswith("gpt"):
            self.api_key = Settings.OPENAI_API_KEY
            return
        if self.model.startswith("google"):
            self.api_key = Settings.GEMINI_API_KEY
            return
        if self.model.startswith("groq"):
            self.api_key = Settings.GROQ_API_KEY
            return

        raise ValueError(f"Model {self.model} not found")

    def setup_phi(self):
        self.pipe = PHI().pipe

    def setup_google(self):
        genai.configure(api_key=self.api_key)
        return

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

    def get_model(self, temperature: float = 0, **kwargs) -> Callable:
        if self.model == "phi":
            return self.pipe, dict(
                max_new_tokens=self.max_tokens,
                return_full_text=False,
                **(
                    dict(temperature=temperature, do_sample=True)
                    if temperature != 0
                    else dict(do_sample=False)
                ),
            )
        if self.model.startswith("google"):
            model_name = self.model.split("/")[-1]
            model = genai.GenerativeModel(
                model_name=model_name,
                generation_config=genai.GenerationConfig(
                    max_output_tokens=self.max_tokens,
                    temperature=temperature,
                ),
                system_instruction=kwargs.get("system_prompt"),
            )
            return model.generate_content, dict()

            client = openai.OpenAI(
                base_url=f"https://{self.location}-aiplatform.googleapis.com/v1beta1/projects/{self.project_id}/locations/{self.location}/endpoints/openapi",
                api_key=self.credentials.token,
            )
            return client.chat.completions.create, dict(
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=temperature,
            )
        if self.model.startswith("claude"):
            client = anthropic.Anthropic(api_key=self.api_key)
            return client.messages.create, dict(
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=temperature,
            )
        if self.model.startswith("gpt"):
            client = openai.OpenAI(api_key=self.api_key)
            return client.chat.completions.create, dict(
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=temperature,
            )

        if self.model.startswith("groq"):
            client = groq.Groq(api_key=self.api_key)
            model = self.model.split("/")[-1]
            return client.chat.completions.create, dict(
                model=model,
                max_tokens=self.max_tokens,
                temperature=temperature,
            )

    def prompt(
        self, text: str, system_prompt: str | None = None, temperature: float = 0
    ) -> str:
        print(f"Prompting {self.model}")
        model, params = self.get_model(
            temperature=temperature, system_prompt=system_prompt
        )
        messages = self.get_messages_dict(text=text, system_prompt=system_prompt)
        if self.model == "phi":
            return model(messages.get("messages", []), **params)
        if self.model.startswith("google"):
            return model(text).text
        if self.model.startswith("claude"):
            return model(**messages, **params).content[0].text
        if self.model.startswith("gpt"):
            return model(**messages, **params).choices[0].message.content
        if self.model.startswith("groq"):
            return model(**messages, **params).choices[0].message.content


class LLMService(metaclass=Singleton):
    def __init__(self) -> None:
        self.llm = LLM()  # model=Settings.llm_model)

    def prompt(
        self, text: str, system_prompt: str | None = None, temperature: float = 0
    ) -> str:
        return self.llm.prompt(
            text=text, system_prompt=system_prompt, temperature=temperature
        )

    def summerize(self, text: str) -> str:
        return self.prompt(
            text,
            system_prompt="Summarize and extract bullet points from the following text, which is a transcription of a meeting voice.",
        )


if __name__ == "__main__":
    system_prompt = "you are a c++ developer"
    prompt = "write a program that prints fibonacci series"

    system_prompt = (
        "Summarize and extract bullet points from the following text, which is a transcription of a meeting voice.",
    )
    prompt = "I wanted to share a few things, but I'm gonna not share as much as I wanted to share because we are starting late. I'd like to get this thing going so we all get home at a decent hour. This election is very important to us."

    llm = LLM(model="google/gemini-1.5-flash-001")
    # llm = LLM()
    res = llm.prompt(prompt, system_prompt=system_prompt, temperature=0)
    print(res)
