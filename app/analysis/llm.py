import json
import pickle
import time

import numpy as np
import pandas as pd

from apps.llm.base_llm import LLM
from server.config import Settings

assets = Settings.base_dir / "assets"

system_prompt = "This is a transcription from a veterian voice session, I want you to take notes and summarize the session."


def backtick_formatter(text: str):
    text = text.strip().strip("```json").strip("```").replace("'", '"').strip()
    return text


def run(temperature):
    outputs = {}
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

    texts = {}
    for file in assets.glob("*.txt"):
        with open(file, "r") as f:
            texts[file.stem] = f.read()

    for i in range(10):
        for fname, text in texts.items():
            for model_name, model_results in models_results.items():
                try:
                    t = time.time()
                    outputs[f"{fname}_{model_name}_{i}"] = LLM(model_name).prompt(
                        text, system_prompt, temperature=temperature
                    )
                    model_results[fname] = model_results.get(fname, []) + [
                        time.time() - t
                    ]
                except Exception as e:
                    print(f"Error running model {model_name} {e}")
                    continue

    for model_name, model_results in models_results.items():
        for fname, text in texts.items():
            res_arr = np.array(model_results.get(fname, []))
            model_results[f"{fname}_mean"] = res_arr.mean()
            model_results[f"{fname}_std"] = res_arr.std()
            model_results[f"{fname}_num"] = len(res_arr)

    models_results = {k: models_results[k] for k in sorted(models_results)}
    with open(f"analysis/result_{temperature}.json", "w") as f:
        json.dump(models_results, f, indent=4)
    with open(f"analysis/outputs_{temperature}.pkl", "w") as f:
        pickle.dump(outputs, f)
    with open(f"analysis/outputs_{temperature}.json", "w") as f:
        json.dump(outputs, f, indent=4)

    df = pd.DataFrame.from_dict(models_results, orient="index")
    df.to_csv(f"analysis/results_{temperature}.csv")

    print(f"Results saved {temperature}")


# run(0.0)
run(0.3)
