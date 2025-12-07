from fastapi import FastAPI
from transformers import AutoTokenizer, AutoModelForCausalLM

app = FastAPI()

tokenizer = AutoTokenizer.from_pretrained("../models/ppo")
model = AutoModelForCausalLM.from_pretrained("../models/ppo")

@app.get("/generate")
def generate(prompt: str):
    inputs = tokenizer(prompt, return_tensors='pt')
    output = model.generate(**inputs, max_new_tokens=100)
    text = tokenizer.decode(output[0])
    return {"response": text}
