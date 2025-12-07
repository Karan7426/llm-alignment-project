# src/run_ppo.py
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification
from peft import PeftModel

# -----------------------------------------------------
# CONFIG
# -----------------------------------------------------
BASE_MODEL = "gpt2"
LORA_DIR = "models/sft"        # final SFT LoRA folder
REWARD_DIR = "models/reward"
DEVICE = torch.device("cpu")

print("Device:", DEVICE)

# -----------------------------------------------------
# TOKENIZER
# -----------------------------------------------------
# Use the tokenizer you saved during SFT training
tokenizer = AutoTokenizer.from_pretrained(LORA_DIR)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

# -----------------------------------------------------
# LOAD POLICY MODEL = base GPT2 + LoRA adapter (SFT)
# -----------------------------------------------------
print("Loading SFT policy model (GPT2 + LoRA)...")
base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL)
policy_model = PeftModel.from_pretrained(base_model, LORA_DIR)
policy_model = policy_model.to(DEVICE)
policy_model.eval()

# -----------------------------------------------------
# LOAD REWARD MODEL
# -----------------------------------------------------
print("Loading reward model...")
reward_model = AutoModelForSequenceClassification.from_pretrained(
    REWARD_DIR,
    num_labels=1
).to(DEVICE)
reward_model.eval()

# -----------------------------------------------------
# SMALL QUERY LIST FOR DEMO
# -----------------------------------------------------
queries = [
    "Tell me about AI.",
    "What is a transformer model?",
    "Explain neural networks simply.",
    "What is reinforcement learning?",
]

# -----------------------------------------------------
# HELPER: GENERATE N CANDIDATES FOR A PROMPT
# -----------------------------------------------------
def generate_candidates(prompt: str, num_candidates: int = 3, max_new_tokens: int = 64):
    """Generate multiple candidate responses from the SFT model."""
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)

    with torch.no_grad():
        outputs = policy_model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            top_p=0.9,
            temperature=0.7,
            num_return_sequences=num_candidates,
        )

    texts = []
    for out in outputs:
        text = tokenizer.decode(out, skip_special_tokens=True)
        texts.append(text)

    return texts

# -----------------------------------------------------
# HELPER: SCORE A RESPONSE WITH THE REWARD MODEL
# -----------------------------------------------------
def score_response(response: str) -> float:
    """Return scalar reward score for a response."""
    enc = tokenizer(
        response,
        return_tensors="pt",
        truncation=True,
        max_length=256,
        padding=True,
    ).to(DEVICE)

    with torch.no_grad():
        logits = reward_model(**enc).logits
        score = logits.squeeze().item()
    return float(score)

# -----------------------------------------------------
# RLHF-STYLE LOOP: SAMPLE & PICK BEST BY REWARD
# -----------------------------------------------------
print("\nStarting RLHF-style scoring (no TRL PPO)…\n")

for qi, query in enumerate(queries, start=1):
    print(f"=== Query {qi}: {query}")
    # You can also prepend something like "Instruction: ..." if you want
    prompt = query

    # 1) Generate multiple candidates
    candidates = generate_candidates(prompt, num_candidates=3, max_new_tokens=64)

    # 2) Score each candidate with the reward model
    scored = []
    for idx, cand in enumerate(candidates, start=1):
        score = score_response(cand)
        scored.append((score, cand))
        print(f"\nCandidate {idx} | Reward: {score:+.4f}")
        print(cand)
        print("-" * 40)

    # 3) Select best candidate
    best_score, best_cand = max(scored, key=lambda x: x[0])
    print("\n>>> BEST RESPONSE SELECTED BY REWARD:")
    print(f"Reward: {best_score:+.4f}")
    print(best_cand)
    print("=" * 80, "\n")

# -----------------------------------------------------
# SAVE FINAL "PPO" MODEL DIR FOR YOUR API
# (weights are same as SFT; alignment is done via reward reranking)
# -----------------------------------------------------
print("Saving final model to models/ppo ...")
policy_model.save_pretrained("models/ppo")
tokenizer.save_pretrained("models/ppo")
print("Done! You can now load models/ppo in your api.py")
