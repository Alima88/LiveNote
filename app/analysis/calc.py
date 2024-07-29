# %%
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Load the JSON data
base_dir = Path(__file__).parent
for temp in [0.0, 0.3]:
    with open(base_dir / f"result_{temp}.json", "r") as file:
        data = json.load(file)

    # Create a list to store the data for the DataFrame
    # Create a list to store the data for the DataFrame
    df_data = []

    # Iterate through the data and extract relevant information
    for model, model_data in data.items():
        for text, times in model_data.items():
            if isinstance(times, list):
                for time in times:
                    df_data.append({"Model": model, "Text": text, "Time": time})

    # Create the DataFrame
    df = pd.DataFrame(df_data)

    # Create the plot
    plt.figure(figsize=(12, 8))
    sns.boxplot(x="Text", y="Time", hue="Model", data=df)
    plt.xticks(rotation=45, ha="right")
    plt.title("Model Performance Comparison")
    plt.xlabel("Text")
    plt.ylabel("Time (seconds)")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", borderaxespad=0)
    plt.tight_layout()
    plt.savefig(base_dir / f"plot_{temp}_whole.jpg")

    # Calculate and print the average time for each model
    average_times = df.groupby("Model")["Time"].mean().sort_values()
    print("Average processing time for each model:")
    print(average_times)

    # Identify the best model (lowest average time)
    best_model = average_times.index[0]
    print(f"\nThe best performing model is: {best_model}")

    df_data = []

    # Iterate through the data and extract relevant information
    for model, model_data in data.items():
        for text, times in model_data.items():
            if isinstance(times, list):
                avg_time = sum(times) / len(times)
                df_data.append({"Model": model, "Text": text, "Average Time": avg_time})

    # Create the DataFrame
    df = pd.DataFrame(df_data)

    # Pivot the DataFrame to have texts as columns and models as rows
    df_pivot = df.pivot(index="Model", columns="Text", values="Average Time")

    # Create the plot
    plt.figure(figsize=(15, 10))
    ax = df_pivot.plot(kind="bar", width=0.8)

    plt.title("Model Performance Comparison", fontsize=16)
    plt.xlabel("Model", fontsize=12)
    plt.ylabel("Average Time (seconds)", fontsize=12)
    plt.legend(
        title="Text", bbox_to_anchor=(1.05, 1), loc="upper left", borderaxespad=0
    )
    plt.xticks(rotation=45, ha="right")

    # Add value labels on the bars
    for container in ax.containers:
        ax.bar_label(
            container, fmt="%.2f", label_type="edge", fontsize=8, rotation=90, padding=2
        )

    plt.tight_layout()
    plt.savefig(base_dir / f"plot_{temp}.jpg")

    # Calculate and print the average time for each model
    average_times = df.groupby("Model")["Average Time"].mean().sort_values()
    print("Average processing time for each model:")
    print(average_times)

    # Identify the best model (lowest average time)
    best_model = average_times.index[0]
    print(f"\nThe best performing model is: {best_model}")

    print(temp, "finished")
