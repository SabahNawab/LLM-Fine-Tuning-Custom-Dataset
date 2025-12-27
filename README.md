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

Then run
```bash
python fine_tuning_demo.py
```
