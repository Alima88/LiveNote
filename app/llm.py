import logging
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

logging.getLogger("transformers").setLevel(logging.WARNING)
logging.getLogger("torch").setLevel(logging.WARNING)

# Check if CUDA is available
cuda_available = torch.cuda.is_available()
# Check if CUDA is available
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print("CUDA available: ", cuda_available)
print(f"Using device: {device}")

model_path = "./ml/huggingface/models--microsoft--Phi-3-mini-4k-instruct/snapshots/c1358f8a35e6d2af81890deffbbfa575b978c62f"

# "microsoft/Phi-3-mini-4k-instruct",
torch.random.manual_seed(0)
model = AutoModelForCausalLM.from_pretrained(
    model_path, torch_dtype="auto", device_map="cuda", trust_remote_code=True
)

for param in model.parameters():
    print("before:", param.device)
    break

model.to(device)

for param in model.parameters():
    print("after:", param.device)
    break

tokenizer = AutoTokenizer.from_pretrained(model_path)

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    # device=0 if cuda_available else -1,  # Use CUDA if available
)

msgs = [
    [
        {"role": "user", "content": "What about solving an 2x + 3 = 7 equation?"},
    ],
    [
        # {"role": "system", "content": "You are a helpful AI assistant."},
        {
            "role": "user",
            "content": "Can you provide ways to eat combinations of bananas and dragonfruits?",
        },
    ],
]


generation_args = {
    "max_new_tokens": 500,
    "return_full_text": False,
    "temperature": 0.0,
    "do_sample": False,
}

import time

import anthropic
import dotenv
import os

import openai

dotenv.load_dotenv()

print("********* phi")

for messages in msgs:
    t = time.time()
    output = pipe(messages, **generation_args)
    print(output[0]["generated_text"])
    print(time.time() - t)
    print()


client = anthropic.Anthropic(
    # defaults to os.environ.get("ANTHROPIC_API_KEY")
    api_key=os.environ.get("ANTHROPIC_API_KEY"),
)

print("********* haiku")
for messages in msgs:
    t = time.time()
    output = client.messages.create(
        model="claude-3-haiku-20240307",
        max_tokens=1024,
        temperature=0.3,
        messages=messages,
    )
    print(output.content)
    print(time.time() - t)
    print()


print("********* sonnet")
for messages in msgs:
    t = time.time()
    output = client.messages.create(
        model="claude-3-5-sonnet-20240620",
        max_tokens=1024,
        temperature=0.3,
        messages=messages,
    )
    print(output.content)
    print(time.time() - t)
    print()

openai.api_key = os.environ.get("OPENAI_API_KEY")

print("********* openai gpt-4o")
for messages in msgs:
    t = time.time()
    output = openai.chat.completions.create(
        model="gpt-4o",
        max_tokens=1024,
        temperature=0.3,
        messages=messages,
    )
    print(output.choices[0].message.content)
    print(time.time() - t)
    print()

print("********* openai gpt-4o-mini")

for messages in msgs:
    t = time.time()
    output = openai.chat.completions.create(
        model="gpt-4o-mini",
        max_tokens=1024,
        temperature=0.3,
        messages=messages,
    )
    print(output.choices[0].message.content)
    print(time.time() - t)
    print()
