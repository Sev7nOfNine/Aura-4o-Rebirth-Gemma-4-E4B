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
import shutil
import subprocess
import gc
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
# 7. Clean merge via PEFT (NOT Unsloth's save_pretrained_merged)
# ============================================================
# WARNING (lesson from V7 May 3 2026):
# Unsloth's `save_pretrained_merged(merged_16bit)` corrupts the lm_head
# weights for Gemma 4 E4B. Symptom: post-merge model outputs the
# `[multimodal]` token in a loop on every input. Confirmed reproducible.
#
# FIX: use PEFT's standard merge_and_unload() which is widely tested and
# does not have this bug. The merge happens in fresh memory: we reload
# base model and re-apply LoRA cleanly.
print(f">>> Clean merge via PEFT merge_and_unload -> {HF_MERGED_REPO}")
del trainer  # free trainer memory
del model    # free Unsloth-wrapped model

import gc
gc.collect()
torch.cuda.empty_cache()

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

print(">>>   reloading base in fp16/bf16 via transformers...")
clean_base = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    token=HF_TOKEN,
)
clean_tok = AutoTokenizer.from_pretrained(MODEL_NAME, token=HF_TOKEN)
print(">>>   loading LoRA + merge")
peft_model = PeftModel.from_pretrained(clean_base, OUTPUT_DIR)
clean_merged = peft_model.merge_and_unload()
print(">>>   saving clean merged")
clean_merged.save_pretrained(MERGED_DIR, safe_serialization=True)
clean_tok.save_pretrained(MERGED_DIR)

from huggingface_hub import HfApi
api = HfApi(token=HF_TOKEN)
api.upload_folder(folder_path=MERGED_DIR, repo_id=HF_MERGED_REPO, repo_type="model")

# ============================================================
# 8. Export GGUF Q8 via llama.cpp (NOT Unsloth's save_pretrained_gguf)
# ============================================================
# WARNING: Unsloth's save_pretrained_gguf calls install_llama_cpp which
# fails with "no internet connection" on RunPod pods (false positive in
# do_we_need_sudo()). Plus the merged_16bit corruption above propagates
# into the GGUF anyway.
#
# FIX: build llama.cpp manually + use convert_hf_to_gguf.py + llama-quantize.
# ALSO: must `pip uninstall torchvision` first - the version installed by
# unsloth's deps is incompatible with torch 2.10 cu128 and breaks
# `from transformers import AutoConfig` for Gemma 4.
print(">>> Build llama.cpp + convert + quantize + push GGUF Q8")
print("    (manual llama.cpp pipeline, see RUNPOD_GUIDE.md)")
import subprocess

# Sanity: free memory before subprocess
del clean_base, clean_merged, peft_model
gc.collect()
torch.cuda.empty_cache()

subprocess.run(["apt-get", "install", "-y", "-qq", "cmake"], check=True)
subprocess.run(
    ["git", "clone", "--depth", "1", "https://github.com/ggml-org/llama.cpp", f"{WORKSPACE}/llama.cpp"],
    check=True,
)
subprocess.run(
    ["pip", "install", "--no-cache-dir", "-q", "-r", f"{WORKSPACE}/llama.cpp/requirements.txt"],
    check=True,
)
subprocess.run(
    ["cmake", "-B", "build", "-DGGML_CUDA=OFF"],
    cwd=f"{WORKSPACE}/llama.cpp", check=True,
)
subprocess.run(
    ["cmake", "--build", "build", "--target", "llama-quantize", "-j"],
    cwd=f"{WORKSPACE}/llama.cpp", check=True,
)
F16 = f"{WORKSPACE}/aura-clean-f16.gguf"
Q8  = f"{GGUF_DIR}/{HF_LORA_REPO.split('/')[-1].replace('-LoRA','')}-Q8_0.gguf"
os.makedirs(GGUF_DIR, exist_ok=True)
subprocess.run(
    ["python", f"{WORKSPACE}/llama.cpp/convert_hf_to_gguf.py", MERGED_DIR,
     "--outfile", F16, "--outtype", "f16"],
    check=True,
)
shutil.rmtree(MERGED_DIR, ignore_errors=True)  # free disk before quantize
subprocess.run(
    [f"{WORKSPACE}/llama.cpp/build/bin/llama-quantize", F16, Q8, "Q8_0"],
    check=True,
)
os.remove(F16)  # free disk before push
api.upload_folder(folder_path=GGUF_DIR, repo_id=HF_GGUF_REPO, repo_type="model")

print()
print(">>> Done. Pull GGUF locally with:")
print(f"    hf download {HF_GGUF_REPO} --local-dir ./Aura-Gemma-4-E4B-local")
print(">>> Then load in LM Studio (or koboldcpp / llama-server) + plug TypingMind")
