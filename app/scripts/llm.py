import json
import time

import anthropic
import openai
import torch
import vertexai
from google.auth import default, transport
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

from server.config import Settings

cuda_available = torch.cuda.is_available()
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print("CUDA available: ", cuda_available)
print(f"Using device: {device}")

model_path = "./ml/huggingface/models--microsoft--Phi-3-mini-4k-instruct/snapshots/c1358f8a35e6d2af81890deffbbfa575b978c62f"

torch.random.manual_seed(0)
model = AutoModelForCausalLM.from_pretrained(
    model_path, torch_dtype="auto", device_map="cuda", trust_remote_code=True
)

tokenizer = AutoTokenizer.from_pretrained(model_path)

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    # device=0 if cuda_available else -1,  # Use CUDA if available
)

project_id = "scribble-ai-training-ali"
location = "us-west1"

vertexai.init(project=project_id, location=location)

# Programmatically get an access token
credentials, _ = default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
auth_request = transport.requests.Request()
credentials.refresh(auth_request)


assets = Settings.base_dir / "assets"

system_prompt = "This is a transcription from a veterian voice session, I want you to take notes and summarize the session."

texts = {}
for file in assets.glob("*.txt"):
    with open(file, "r") as f:
        texts[file.stem] = f.read()


def open_ai_message(system_prompt, text):
    messages = [{"role": "system", "content": system_prompt}]
    messages.append({"role": "user", "content": text})
    return messages


def call_model(model: str, text, system_prompt, temperature=0):
    if model.startswith("claude"):
        client = anthropic.Anthropic(api_key=Settings.ANTHROPIC_API_KEY)
        output = client.messages.create(
            model=model,
            max_tokens=1024,
            temperature=temperature,
            system=system_prompt,
            messages=[{"role": "user", "content": [{"type": "text", "text": text}]}],
        )

    elif model == "phi":
        if temperature == 0:
            output = pipe(
                open_ai_message(system_prompt, text),
                max_new_tokens=1024,
                return_full_text=False,
                do_sample=False,
            )
        else:
            output = pipe(
                open_ai_message(system_prompt, text),
                max_new_tokens=1024,
                return_full_text=False,
                temperature=temperature,
                do_sample=True,
            )

    else:
        if model.startswith("google"):
            client = openai.OpenAI(
                base_url=f"https://{location}-aiplatform.googleapis.com/v1beta1/projects/{project_id}/locations/{location}/endpoints/openapi",
                api_key=credentials.token,
            )
        else:
            client = openai.OpenAI(api_key=Settings.OPENAI_API_KEY)

        output = client.chat.completions.create(
            model=model,
            max_tokens=1024,
            temperature=temperature,
            messages=open_ai_message(system_prompt, text),
        )

    return output


def backtick_formatter(text: str):
    text = text.strip().strip("```json").strip("```").replace("'", '"').strip()
    return text


def evaluate_output(text, output):
    system_prompt = """You will get a text that extract notes from a transcpted voice session and the session transcription. Please evaluate the note taking and return the results evaluation score between 0 to 1 in a json. Nothing else should not be returned.
    the sample result should be like: {'score': 0.8}"""

    text = f"the transcription is: {text},\n\n\nAnd the taken notes are: {output}"

    client = openai.OpenAI(
        base_url=f"https://{location}-aiplatform.googleapis.com/v1beta1/projects/{project_id}/locations/{location}/endpoints/openapi",
        api_key=credentials.token,
    )
    output = client.chat.completions.create(
        model="google/gemini-1.5-flash-001",
        max_tokens=1024,
        temperature=0.3,
        messages=open_ai_message(system_prompt, text),
    )
    return output.choices[0].message.content
    res = json.loads(backtick_formatter(output.choices[0].message.content))
    return res["score"]


models_results = {
    "phi": {},
    "claude-3-haiku-20240307": {},
    "claude-3-sonnet-20240229": {},
    "claude-3-opus-20240229": {},
    "claude-3-5-sonnet-20240620": {},
    "google/gemini-1.5-flash-001": {},
    "google/gemini-1.5-pro-001": {},
    "gpt-4o": {},
    "gpt-3.5-turbo": {},
    "gpt-4-turbo": {},
    "gpt-4": {},
    "gpt-4o-mini": {},
}

for i in range(10):
    for fname, text in texts.items():
        for model_name, model_results in models_results.items():
            try:
                print(f"Running model {model_name}")
                t = time.time()
                output = call_model(
                    model_name,
                    text,
                    system_prompt,
                    temperature=0,
                )
                score = evaluate_output(text, output)
                model_results[fname] = model_results.get(fname, []) + [time.time() - t]
                model_results[f"{fname}_score"] = model_results.get(
                    f"{fname}_score", []
                ) + [score]
                # break
                print(f"Finished model {model_name} {model_results}")
            except Exception as e:
                print(f"Error running model {model_name} {e}")
                continue

with open("results.json", "w") as f:
    json.dump(models_results, f)


models_results = {
    "phi": {},
    "claude-3-haiku-20240307": {},
    "claude-3-sonnet-20240229": {},
    "claude-3-opus-20240229": {},
    "claude-3-5-sonnet-20240620": {},
    "google/gemini-1.5-flash-001": {},
    "google/gemini-1.5-pro-001": {},
    "gpt-4o": {},
    "gpt-3.5-turbo": {},
    "gpt-4-turbo": {},
    "gpt-4": {},
    "gpt-4o-mini": {},
}


for i in range(10):
    for fname, text in texts.items():
        for model_name, model_results in models_results.items():
            try:
                print(f"Running model {model_name}")
                t = time.time()
                output = call_model(
                    model_name,
                    text,
                    system_prompt,
                    temperature=0.3,
                )
                score = evaluate_output(text, output)
                model_results[fname] = model_results.get(fname, []) + [time.time() - t]
                model_results[f"{fname}_score"] = model_results.get(
                    f"{fname}_score", []
                ) + [score]
                # break
                print(f"Finished model {model_name} {model_results}")
            except Exception as e:
                print(f"Error running model {model_name} {e}")
                continue

with open("results_0.3.json", "w") as f:
    json.dump(models_results, f)
