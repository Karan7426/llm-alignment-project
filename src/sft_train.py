from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
import os

OUT_DIR = "models/sft"
os.makedirs(OUT_DIR, exist_ok=True)

model_name = "gpt2"

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

# Load Alpaca dataset (small subset)
dataset = load_dataset("tatsu-lab/alpaca")
dataset["train"] = dataset["train"].select(range(500))  # <<< FAST TRAINING

# Tokenization
def tokenize(batch):
    texts = [
        instr + "\n" + out
        for instr, out in zip(batch["instruction"], batch["output"])
    ]

    enc = tokenizer(
        texts,
        truncation=True,
        padding="max_length",
        max_length=256,
    )

    enc["labels"] = enc["input_ids"].copy()
    return enc

tokenized = dataset["train"].map(
    tokenize,
    batched=True,
    remove_columns=dataset["train"].column_names,
)

# LoRA config
lora = LoraConfig(
    r=8,
    lora_alpha=32,
    lora_dropout=0.1,
    task_type="CAUSAL_LM"
)

model = AutoModelForCausalLM.from_pretrained(model_name)
model = get_peft_model(model, lora)

# Training args (fast)
args = TrainingArguments(
    output_dir=OUT_DIR,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    num_train_epochs=1,
    learning_rate=3e-4,
    logging_steps=10,
    save_steps=300,
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized,
)

trainer.train()
trainer.save_model(OUT_DIR)
tokenizer.save_pretrained(OUT_DIR)

print("SFT model saved to:", OUT_DIR)
