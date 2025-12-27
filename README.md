# LLM-Fine-Tuning-Custom-Dataset

**"Pakhair Raghlay!"** (Welcome!)

This project demonstrates how to fine-tune **Gemma-3** to adopt a real **Peshawari persona** — hospitality, slang, and cultural tone — running **100% locally**.

Built for a Lightning Talk on **Local AI** — models that understand *our people, our language, our culture*.

---

## Project Overview

| Item | Value |
|------|------|
| Base Model | `google/gemma-3-4b-it` |
| Fine-Tuning | Unsloth + LoRA |
| Persona | Peshawari (Rora, Jana, Zabardast vibes) |
| Deployment | Ollama (GGUF) |
| Mode | Fully Offline |

---

## 🛠️ Setup

### 1️⃣ Requirements

- Python **3.10+**
- NVIDIA GPU (**8GB+ VRAM recommended**)
- CUDA **12.1+**

---

### 2️⃣ Virtual Environment

```bash
python -m venv venv

# Windows
.\venv\Scripts\activate

# Linux / Mac
source venv/bin/activate
```
### 3️⃣ Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```
### Training
Ensure your dataset is present (data.jsonl)

Then run the Fine-Tuning notebook in sequence 

### Export to Ollama (GGUF)
Script creates
```bash
actual_ai_gguf/unsloth.Q4_K_M.gguf
```
### Create Modelfile
```bash
FROM ./actual_ai_gguf/unsloth.Q4_K_M.gguf

TEMPLATE """<start_of_turn>user
{{ .Prompt }}<end_of_turn>
<start_of_turn>model
"""

PARAMETER temperature 0.8
PARAMETER top_p 0.9
PARAMETER stop "<end_of_turn>"
PARAMETER stop "<start_of_turn>"
```

### Register with Ollama
```bash
ollama create actual_ai -f Modelfile
ollama run actual_ai

```
