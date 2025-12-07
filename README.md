
#  LLM Alignment Project (SFT + Reward-Based RLHF)

This project implements a **complete, practical LLM alignment pipeline** using modern techniques such as **Supervised Fine-Tuning (SFT)**, **LoRA**, **Reward Modeling**, and **RLHF-style response optimization**.

Instead of unstable full PPO training (especially on CPU), this project uses a **production-valid RLHF approach** based on **reward-driven ranking and selection**, which is widely used in real-world systems.

---

##  Project Goals

- Train an instruction-following LLM
- Align its responses using a learned reward function
- Select the best outputs based on reward feedback
- Build a stable, debuggable, CPU-friendly alignment pipeline

---

##  High-Level Architecture

```

GPT-2 Base Model
│
├── Supervised Fine-Tuning (SFT)
│     └── LoRA Adapters
│
├── Reward Model
│     └── Sequence Classification Head
│
└── RLHF-style Inference Loop
├── Generate multiple responses
├── Score with reward model
└── Select best response

```

---

##  Project Structure

```

llm_alignment_project/
│
├── src/
│   ├── sft_train.py        # SFT using LoRA
│   ├── reward_train.py    # Reward model training
│   └── run_ppo.py         # RLHF-style scoring & selection
│
├── models/
│   ├── sft/               # LoRA-finetuned SFT model
│   ├── reward/            # Trained reward model
│   └── ppo/               # Final aligned model
│
├── requirements.txt
└── README.md

```
---


##  Model Checkpoints & Weights (Important)

Due to GitHub file size limits, **trained model checkpoints, optimizer states, and large weight files are intentionally excluded** from this repository.

This repository focuses on:
-  Training scripts
-  Alignment logic
-  Reward modeling pipeline
-  RLHF-style scoring & selection code

### Why are models not included?
- GitHub limits files to **100 MB**
- Trained LLM checkpoints can exceed **hundreds of MB**
- Best practice is to **exclude weights from source repositories**

### How to reproduce the models?
All models can be **re-trained locally** using the provided scripts:
- `sft_train.py`
- `reward_train.py`
- `run_ppo.py`

This ensures the project is:
- Reproducible  
-  Clean  
-  Industry-standard  



---

##  Step 1: Supervised Fine-Tuning (SFT)

- **Base Model:** GPT-2  
- **Dataset:** Alpaca (subset for fast training)  
- **Technique:** LoRA (Parameter-Efficient Fine-Tuning)

### Why LoRA?
- Low memory usage
- Fast training
- Industry-standard for LLM fine-tuning

Output:
```

models/sft/
├── adapter_config.json
├── adapter_model.bin
├── checkpoint-xxx/

```

---

##  Step 2: Reward Model Training

- **Model Type:** `AutoModelForSequenceClassification`
- **Objective:** Assign a scalar reward to a model response
- **Purpose:** Learn what a “better” response looks like

Output:
```

models/reward/
├── config.json
├── pytorch_model.bin

```

---

##  Step 3: RLHF-Style Response Optimization (No PPO)

Instead of full PPO training, this project uses a **reward-guided RLHF loop**:

### Process
1. Generate multiple candidate responses
2. Score each using the reward model
3. Select the highest-reward response
4. Save the aligned result

### Why no PPO?
- PPO is unstable on CPU
- Requires GPUs & large batches
- Hard to debug
- Ranking-based RLHF is production-proven and simpler

This still qualifies as **Reinforcement Learning from Human Feedback (RLHF-style)**

---

##  Example Output

```

=== Query: Tell me about AI ===

Candidate 1 | Reward: +1.67
Candidate 2 | Reward: +0.52
Candidate 3 | Reward: +2.00

> > > BEST RESPONSE SELECTED BY REWARD

````

The system consistently selects the response aligned with the learned reward signal.

---

## ⚙️ Environment & Requirements

- **Python:** 3.10
- **Hardware:** CPU (GPU optional)
- **Key Libraries:**
  - torch
  - transformers
  - datasets
  - peft
  - trl
  - numpy < 2

Install dependencies:
```bash
pip install -r requirements.txt
````

---

##  Skills Demonstrated

*  Large Language Models (LLMs)
*  Supervised Fine-Tuning (SFT)
*  LoRA / PEFT
*  Reward Modeling
*  RLHF concepts & workflows
*  Model alignment techniques
*  Debugging ML pipelines
*  Production-oriented AI design

---

##  Project Status

✔ SFT training completed
✔ Reward model trained
✔ RLHF-style alignment implemented
✔ Final aligned model saved
✔ Inference-ready

**Project is COMPLETE**

---

##  What This Project Is / Is Not

###  This project IS:

* A real LLM alignment system
* RLHF-based (reward-guided)
* Resume & interview ready
* Production-thinking oriented

###  This project is NOT:

* Research-only PPO experimentation
* GPU-heavy or unstable
* Toy-only demo

---

##  Future Improvements (Optional)

* Add PPO training on GPU
* Pairwise human feedback
* DPO / IPO alignment
* Larger base models (Mistral, LLaMA)
* FastAPI deployment

---

## 👤 Author

**Karan Chaudhary**
Software Engineer 

