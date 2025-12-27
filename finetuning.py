# -*- coding: utf-8 -*-
"""
IMPORTING REQUIRED LIBRARIES
"""

#COLAB version
# %%capture
# !pip install -U datasets
# !pip install -U accelerate
# !pip install -U peft
# !pip install -U trl
# !pip install -U bitsandbytes
# !pip install git+https://github.com/huggingface/transformers@v4.49.0-Gemma-3
# !pip install -q unsloth
# !pip install -q --force-reinstall --no-cache-dir --no-deps git+https://github.com/unslothai/unsloth.git
# !pip install unsloth_zoo

from huggingface_hub import login
import os
import dotenv
from unsloth import FastModel
from datasets import load_dataset
from transformers import TextStreamer, AutoProcessor
from trl import SFTTrainer, SFTConfig
import torch
from unsloth.chat_templates import get_chat_template

"""HF AUTHENTICATION FOR GATED MODELS LIKE GEMMA (NEEDS ACCESS PERMISSIONS)"""

os.environ["HF_TOKEN"]=dotenv.get_key(".env", "HF_TOKEN")
hf_token=os.environ["HF_TOKEN"]
login(hf_token)

"""MODEL & TOKENIZER LOADING"""

model, tokenizer = FastModel.from_pretrained(
    model_name = "unsloth/gemma-3-4b-it",
    max_seq_length = 1024,
    load_in_4bit = False,
    load_in_8bit = True,
    full_finetuning = False,
    token=hf_token
)

"""LOADING & FORMATTING DATASET"""

local_dataset = load_dataset(
    "json",
    data_files="data.jsonl",
    split="train"
   )

print(local_dataset)

EOS_TOKEN = tokenizer.eos_token
tokenizer = get_chat_template(
    tokenizer,
    chat_template = "gemma-3",
)

def format_instruction(example):
    messages = [
        {"role": "user", "content": example["instruction"]},
        {"role": "model", "content": example["output"]},
    ]
    return {"text": tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)+ EOS_TOKEN}

dataset = local_dataset.map(format_instruction)

# You can print an example to verify
print(dataset[0]['text'])

"""RESPONSES OF BASE INSTRUCT MODEL BEFORE FINETUNING"""

print("\n" + "="*50)
print(">>> GENERATION BEFORE FINE-TUNING (General Model) <<<")
print("="*50)

FastModel.for_inference(model)

test_instruction = "Where can I find the best Chapli Kebab in Peshawar?"
messages = [
    {"role": "user", "content": test_instruction},
]

prompt_string = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt = True,
    tokenize = False,
)

inputs = tokenizer(text = [prompt_string], return_tensors = "pt").to("cuda")

from transformers import TextStreamer
text_streamer = TextStreamer(tokenizer)

_ = model.generate(
    **inputs,
    streamer = text_streamer,
    max_new_tokens = 512,
    temperature = 0.7,
    do_sample = True,
    use_cache = True
)

"""PARAMETER EFFICIENT FINE-TUNING"""

model = FastModel.get_peft_model(
    model,
    r=16,
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    finetune_vision_layers = False, # Turn off for just text!
    finetune_language_layers = True,  # Should leave on!
    finetune_attention_modules = True,  # Attention good for GRPO
    finetune_mlp_modules = True,  # Should leave on always!
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=3407,
    use_rslora=False
)

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = 512,
    dataset_num_proc = 2,
    args = SFTConfig(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        warmup_steps = 5,
        num_train_epochs = 3,
        learning_rate = 2e-4,
        fp16 = not torch.cuda.is_bf16_supported(),
        bf16 = torch.cuda.is_bf16_supported(),
        logging_steps = 20,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "outputs",
    ),
)

trainer.train()

"""INFERENCE AFTER FINE-TUNING THE MODEL"""

FastModel.for_inference(model)
text_streamer = TextStreamer(tokenizer)


test_cases = [
    {"q": "where can I find the best chapli kebab in peshawar?", "temp": 0.7},
    {"q": "Hello, How are you doing today?", "temp": 0.8},
    {"q": "where should I go for good food in peshawar?", "temp": 0.4}
]


for i, case in enumerate(test_cases, 1):
    print(f"\n{'='*30}")
    print(f">>> TEST OUTPUT {i} (Temp: {case['temp']})")
    print(f"{'='*30}")


    messages = [{"role": "user", "content": case["q"]}]

    prompt_string = tokenizer.apply_chat_template(
        messages,
        tokenize = False,
        add_generation_prompt = True,
    )


    inputs = tokenizer(text = [prompt_string], return_tensors = "pt").to("cuda")


    _ = model.generate(
        **inputs,
        streamer = text_streamer,
        max_new_tokens = 512,
        temperature = case["temp"],
        do_sample = True,
        use_cache = True
    )

"""MODEL SAVING AND RUN THROUGH OLLAMA"""

model.save_pretrained("peshawari_lora")
tokenizer.save_pretrained("peshawari_lora")

model_text, tokenizer = FastModel.from_pretrained(
    model_name = "peshawari_lora", # Load your trained adapters
    max_seq_length = 1024,
    load_in_4bit = True,
)


model_text.save_pretrained_gguf("actual_ai_gguf", tokenizer, quantization_method = "q4_k_m")

"""Run command in the terminal : ollama create actual_ai -f Modelfile

"""