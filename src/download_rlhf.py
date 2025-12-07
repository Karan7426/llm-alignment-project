from datasets import load_dataset
import json
import os

os.makedirs("data", exist_ok=True)

# Load the HH-RLHF dataset
ds = load_dataset("Anthropic/hh-rlhf", split="train")

# Save to preferences.jsonl
with open("data/preferences.jsonl", "w", encoding="utf-8") as f:
    for example in ds:
        f.write(json.dumps(example) + "\n")

print("RLHF dataset saved to data/preferences.jsonl")
