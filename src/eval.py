from transformers import AutoModelForCausalLM, AutoTokenizer

models = {
    "sft": "../models/sft",
    "ppo": "../models/ppo"
}

tokenizer = AutoTokenizer.from_pretrained("gpt2")

prompts = [
    "Explain quantum computing",
    "Define photosynthesis",
]

for name, path in models.items():
    model = AutoModelForCausalLM.from_pretrained(path)
    print(f"\n--- {name.upper()} ---")
    for p in prompts:
        inputs = tokenizer(p, return_tensors="pt")
        out = model.generate(**inputs, max_new_tokens=80)
        print(f"Prompt: {p}")
        print("Response:", tokenizer.decode(out[0]))
