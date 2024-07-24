import json


def backtick_formatter(text: str):
    text = text.strip().strip("```json").strip("```").replace("'", '"').strip()
    return text


with open("results.json") as f:
    data = json.load(f)


for k, v in data.items():
    for k2, v2 in v.items():
        if k2.endswith("score"):
            for i, el in enumerate(v2):
                v2[i] = json.loads(backtick_formatter(el)).get("score")


data2 = {}

for model_name, model_results in data.items():
    for fname, values in model_results.items():
        break
