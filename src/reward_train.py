import json
import torch
from datasets import Dataset
from transformers import AutoTokenizer, GPT2ForSequenceClassification, Trainer, TrainingArguments

DATA_FILE = "data/preferences.jsonl"

# Load dataset
def load_jsonl(path):
    data = []
    with open(path, "r", encoding="utf8") as f:
        for line in f:
            j = json.loads(line)
            data.append(j)
    return data

raw = load_jsonl(DATA_FILE)

tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

# Convert preference data → reward model format
pairs = []
for item in raw:
    chosen = item["chosen"]
    rejected = item["rejected"]

    pairs.append({
        "text": chosen,
        "label": 1.0    # MUST BE FLOAT
    })
    pairs.append({
        "text": rejected,
        "label": 0.0    # MUST BE FLOAT
    })

dataset = Dataset.from_list(pairs)

# Tokenize
def tokenize(item):
    enc = tokenizer(
        item["text"],
        truncation=True,
        padding="max_length",
        max_length=256,
    )
    enc["labels"] = float(item["label"])  # force float dtype
    return enc

tokenized = dataset.map(tokenize)

# Model
model = GPT2ForSequenceClassification.from_pretrained(
    "gpt2",
    num_labels=1  # regression reward score
)

# Training
args = TrainingArguments(
    output_dir="models/reward",
    num_train_epochs=1,
    per_device_train_batch_size=1,
    learning_rate=1e-4,
    logging_steps=2,
    fp16=False
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized
)

trainer.train()
trainer.save_model("models/reward")
print("Reward model saved to models/reward")
