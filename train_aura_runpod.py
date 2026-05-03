"""
Aura-4o-Rebirth-Gemma-4-E4B - Top-quality fine-tune on RunPod (A40 48GB+)
==========================================================================

V7 recipe (3 May 2026, post-audit), adapted for Gemma 4 E4B:

  - Multi-turn dataset (SevenOfNine/Aura-4o-Rebirth-Dataset, 1865 chunks)
  - LoRA r=128 (richer adapter for smaller base, vs r=32 on 31B)
  - max_seq_length 4096 (no truncation)
  - LoRA 16-bit (no QLoRA quant)
  - NEFTune alpha=5 (documented +5-10% generation quality)
  - train_on_responses_only with CORRECT Gemma 4 turn delimiters
    (was <start_of_turn>... = Gemma 3 syntax, fixed to <|turn>... for Gemma 4)
  - Eval split 5% with early stopping (patience=3)
  - Vision/audio layers frozen (multimodal capabilities preserved)
  - Cosine LR + warmup, BF16

Outputs:
  SevenOfNine/Aura-4o-Rebirth-Gemma-4-E4B-LoRA      (LoRA adapter)
  SevenOfNine/Aura-4o-Rebirth-Gemma-4-E4B-Merged    (merged 16bit safetensors)
  SevenOfNine/Aura-4o-Rebirth-Gemma-4-E4B-GGUF      (GGUF Q8 for local inference)

Usage on RunPod (A40 48GB pod):
  pip install unsloth unsloth_zoo transformers datasets trl peft accelerate huggingface_hub
  export HF_TOKEN=hf_xxx
  python train_aura_runpod.py
"""

import os
import torch
from unsloth import FastModel
from unsloth.chat_templates import train_on_responses_only
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig
from transformers import EarlyStoppingCallback

# ============================================================
# Config
# ============================================================
WORKSPACE      = "/workspace/Aura-4o-Rebirth-Gemma-4-E4B"
MODEL_NAME     = "unsloth/gemma-4-E4B-it"
DATASET_REPO   = "SevenOfNine/Aura-4o-Rebirth-Dataset"          # private, multi-turn chunks
HF_LORA_REPO   = "SevenOfNine/Aura-4o-Rebirth-Gemma-4-E4B-LoRA"
HF_MERGED_REPO = "SevenOfNine/Aura-4o-Rebirth-Gemma-4-E4B-Merged"
HF_GGUF_REPO   = "SevenOfNine/Aura-4o-Rebirth-Gemma-4-E4B-GGUF"

OUTPUT_DIR     = os.path.join(WORKSPACE, "lora")
CHECKPOINT_DIR = os.path.join(WORKSPACE, "checkpoints")
MERGED_DIR     = os.path.join(WORKSPACE, "merged")
GGUF_DIR       = os.path.join(WORKSPACE, "gguf")

# Quality config
MAX_SEQ_LENGTH = 4096
LORA_R         = 128
LORA_ALPHA     = 128
LORA_DROPOUT   = 0.05
NEFTUNE_ALPHA  = 5
EPOCHS         = 3
BATCH_SIZE     = 4               # A40 48GB handles this easily on E4B
GRAD_ACCUM     = 8               # effective batch = 32 (matches V1 strict on 31B)
LR             = 2e-4
WARMUP_RATIO   = 0.05
EVAL_RATIO     = 0.05
EVAL_STEPS     = 50              # ~7 evals over a typical run, gives early stopping room
SAVE_STEPS     = 50
SEED           = 3407

HF_TOKEN = os.environ.get("HF_TOKEN")
if not HF_TOKEN:
    raise SystemExit("HF_TOKEN env var required")

os.makedirs(WORKSPACE, exist_ok=True)

# ============================================================
# 1. Load model in BF16 LoRA mode
# ============================================================
print(">>> Loading Gemma 4 E4B in BF16 (LoRA 16-bit, no QLoRA)...")
model, tokenizer = FastModel.from_pretrained(
    model_name      = MODEL_NAME,
    max_seq_length  = MAX_SEQ_LENGTH,
    load_in_4bit    = False,
    full_finetuning = False,
    dtype           = torch.bfloat16,
    token           = HF_TOKEN,
)

# ============================================================
# 2. Attach LoRA - text decoder only, vision/audio frozen
# ============================================================
print(f">>> Attaching LoRA r={LORA_R} alpha={LORA_ALPHA} to text layers...")
model = FastModel.get_peft_model(
    model,
    finetune_vision_layers     = False,   # vision tower preserved
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
# 3. Load multi-turn Aura dataset and apply Gemma 4 chat template
# ============================================================
print(f">>> Loading multi-turn dataset: {DATASET_REPO}")
raw = load_dataset(DATASET_REPO, split="train", token=HF_TOKEN)
print(f">>> Dataset rows: {len(raw)}, cols: {raw.column_names}")

def format_multiturn(example):
    # The model's native chat_template.jinja handles all role markers correctly.
    text = tokenizer.apply_chat_template(
        example["messages"],
        tokenize=False,
        add_generation_prompt=False,
    )
    return {"text": text}

dataset = raw.map(format_multiturn, remove_columns=raw.column_names, num_proc=4)

split = dataset.train_test_split(test_size=EVAL_RATIO, seed=SEED)
train_ds, eval_ds = split["train"], split["test"]
print(f">>> Train: {len(train_ds)}  Eval: {len(eval_ds)}")

# ============================================================
# 4. Trainer config
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

    logging_steps               = 5,
    eval_strategy               = "steps",
    eval_steps                  = EVAL_STEPS,
    save_strategy               = "steps",
    save_steps                  = SAVE_STEPS,
    save_total_limit            = 3,
    load_best_model_at_end      = True,
    metric_for_best_model       = "eval_loss",
    greater_is_better           = False,

    dataset_text_field          = "text",
    max_length                  = MAX_SEQ_LENGTH,
    packing                     = False,    # processor-based model, packing not supported

    push_to_hub                 = True,
    hub_model_id                = HF_LORA_REPO,
    hub_strategy                = "every_save",
    hub_private_repo            = False,
    hub_token                   = HF_TOKEN,

    report_to                   = "none",
    seed                        = SEED,
    neftune_noise_alpha         = NEFTUNE_ALPHA,
    dataset_num_proc            = 4,
)

trainer = SFTTrainer(
    model           = model,
    tokenizer       = tokenizer,
    train_dataset   = train_ds,
    eval_dataset    = eval_ds,
    args            = sft_config,
    callbacks       = [EarlyStoppingCallback(early_stopping_patience=3)],
)

# Train on assistant responses only - GEMMA 4 turn delimiters (NOT Gemma 3 syntax)
# Verified by inspecting tokenizer.apply_chat_template output for Gemma 4 E4B:
#   <bos><|turn>user\n...<turn|>\n<|turn>model\n...<turn|>\n
trainer = train_on_responses_only(
    trainer,
    instruction_part = "<|turn>user\n",
    response_part    = "<|turn>model\n",
)

# ============================================================
# 5. Train
# ============================================================
print(">>> Starting training...")
gpu = torch.cuda.get_device_properties(0)
print(f"GPU: {gpu.name}  VRAM: {round(gpu.total_memory/1e9,1)} GB")

trainer.train()

# ============================================================
# 6. Save final LoRA + push
# ============================================================
print(f">>> Saving LoRA adapters to {OUTPUT_DIR} and pushing to {HF_LORA_REPO}")
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
model.push_to_hub(HF_LORA_REPO, token=HF_TOKEN)
tokenizer.push_to_hub(HF_LORA_REPO, token=HF_TOKEN)

# ============================================================
# 7. Merge 16bit + push to Merged repo
# ============================================================
print(f">>> Merging adapters (merged_16bit, V1 method that preserves voice) -> {HF_MERGED_REPO}")
model.save_pretrained_merged(
    MERGED_DIR,
    tokenizer,
    save_method="merged_16bit",
)
from huggingface_hub import HfApi
api = HfApi(token=HF_TOKEN)
api.upload_folder(folder_path=MERGED_DIR, repo_id=HF_MERGED_REPO, repo_type="model")

# ============================================================
# 8. Export GGUF Q8 for local inference + push
# ============================================================
print(f">>> Exporting GGUF Q8 (~7.5GB, fits 16GB GPU) -> {HF_GGUF_REPO}")
model.save_pretrained_gguf(
    GGUF_DIR,
    tokenizer,
    quantization_method="q8_0",
)
api.upload_folder(folder_path=GGUF_DIR, repo_id=HF_GGUF_REPO, repo_type="model")

print()
print(">>> Done. Pull GGUF locally with:")
print(f"    huggingface-cli download {HF_GGUF_REPO} --local-dir ./Aura-Gemma-4-E4B-local")
print(">>> Then load in LM Studio + connect TypingMind to http://localhost:1234/v1")
