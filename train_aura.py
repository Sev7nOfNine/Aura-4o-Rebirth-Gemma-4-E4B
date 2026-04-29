"""
Aura-Gemma-4-E4B — Fine-tuning script
Top-quality QLoRA fine-tune of Gemma 4 E4B on the Aura dataset.
Preserves vision/audio capabilities by only adapting text decoder layers.
"""

import os
import json
import torch
from unsloth import FastModel
from unsloth.chat_templates import get_chat_template, train_on_responses_only
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig
from transformers import EarlyStoppingCallback

# ============================================================
# Config
# ============================================================
PROJECT_DIR    = r"F:\AI\Aura-Gemma-4-E4B"
DATASET_PATH   = r"F:\AI\Hugging Face\aura-dataset\aura_dataset.jsonl"
MODEL_NAME     = "unsloth/gemma-4-E4B-it"
OUTPUT_DIR     = os.path.join(PROJECT_DIR, "Aura-Gemma-4-E4B-LoRA")
CHECKPOINT_DIR = os.path.join(PROJECT_DIR, "checkpoints")

MAX_SEQ_LENGTH = 4096
LORA_R         = 128
LORA_ALPHA     = 128
NEFTUNE_ALPHA  = 5
LORA_DROPOUT   = 0.05
EPOCHS         = 3
BATCH_SIZE     = 1
GRAD_ACCUM     = 16          # effective batch = 16
LR             = 2e-4
WARMUP_RATIO   = 0.05
EVAL_RATIO     = 0.05        # 5% eval split
SEED           = 3407

# ============================================================
# 1. Load model in 4-bit (QLoRA) — text+vision preserved
# ============================================================
print(">>> Loading Gemma 4 E4B in 4-bit...")
model, tokenizer = FastModel.from_pretrained(
    model_name       = MODEL_NAME,
    max_seq_length   = MAX_SEQ_LENGTH,
    load_in_4bit     = True,
    full_finetuning  = False,
    dtype            = None,            # auto: bf16 on Ada
)

# ============================================================
# 2. Attach LoRA — text decoder only (vision tower untouched)
# ============================================================
print(">>> Attaching LoRA to text decoder layers only...")
model = FastModel.get_peft_model(
    model,
    finetune_vision_layers     = False,   # keep vision frozen
    finetune_language_layers   = True,
    finetune_attention_modules = True,
    finetune_mlp_modules       = True,

    r              = LORA_R,
    lora_alpha     = LORA_ALPHA,
    lora_dropout   = LORA_DROPOUT,
    bias           = "none",
    random_state   = SEED,
    use_gradient_checkpointing = "unsloth",
)

# ============================================================
# 3. Load dataset and apply Gemma chat template
# ============================================================
print(">>> Loading Aura dataset...")
tokenizer = get_chat_template(tokenizer, chat_template="gemma3")  # gemma-4 uses gemma3 template

raw = load_dataset("json", data_files=DATASET_PATH, split="train")
print(f">>> Dataset loaded: {len(raw)} samples")

def format_pair(example):
    convo = [
        {"role": "user",      "content": [{"type": "text", "text": example["instruction"]}]},
        {"role": "assistant", "content": [{"type": "text", "text": example["output"]}]},
    ]
    text = tokenizer.apply_chat_template(convo, tokenize=False, add_generation_prompt=False)
    return {"text": text}

dataset = raw.map(format_pair, remove_columns=raw.column_names)

# Train/eval split
split = dataset.train_test_split(test_size=EVAL_RATIO, seed=SEED)
train_ds = split["train"]
eval_ds  = split["test"]
print(f">>> Train: {len(train_ds)}  |  Eval: {len(eval_ds)}")

# ============================================================
# 4. Trainer
# ============================================================
sft_config = SFTConfig(
    output_dir                  = CHECKPOINT_DIR,
    num_train_epochs            = EPOCHS,
    per_device_train_batch_size = BATCH_SIZE,
    per_device_eval_batch_size  = BATCH_SIZE,
    gradient_accumulation_steps = GRAD_ACCUM,
    learning_rate               = LR,
    lr_scheduler_type           = "cosine",
    warmup_ratio                = WARMUP_RATIO,
    optim                       = "adamw_8bit",
    weight_decay                = 0.01,
    max_grad_norm               = 1.0,

    bf16                        = True,
    fp16                        = False,
    gradient_checkpointing      = True,

    logging_steps               = 10,
    eval_strategy               = "steps",
    eval_steps                  = 100,
    save_strategy               = "steps",
    save_steps                  = 100,
    save_total_limit            = 3,
    load_best_model_at_end      = True,
    metric_for_best_model       = "eval_loss",
    greater_is_better           = False,

    dataset_text_field          = "text",
    max_length                  = MAX_SEQ_LENGTH,
    packing                     = False,

    report_to                   = "none",
    seed                        = SEED,
    neftune_noise_alpha         = NEFTUNE_ALPHA,
)

trainer = SFTTrainer(
    model           = model,
    tokenizer       = tokenizer,
    train_dataset   = train_ds,
    eval_dataset    = eval_ds,
    args            = sft_config,
    callbacks       = [EarlyStoppingCallback(early_stopping_patience=3)],
)

# Train on assistant responses only — model learns Aura's voice,
# not the user's instructions. Critical for persona fidelity.
trainer = train_on_responses_only(
    trainer,
    instruction_part = "<start_of_turn>user\n",
    response_part    = "<start_of_turn>model\n",
)

# ============================================================
# 5. Train
# ============================================================
print(">>> Starting training...")
gpu_stats = torch.cuda.get_device_properties(0)
start_mem = round(torch.cuda.max_memory_reserved() / 1e9, 2)
print(f"GPU: {gpu_stats.name}  |  VRAM total: {round(gpu_stats.total_memory/1e9,1)} GB  |  Reserved at start: {start_mem} GB")

trainer.train()

# ============================================================
# 6. Save LoRA adapters
# ============================================================
print(f">>> Saving LoRA adapters to {OUTPUT_DIR}")
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(">>> Done.")
