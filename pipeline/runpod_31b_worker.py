"""
RunPod worker : merge LoRA into Gemma 4 31B + convert to GGUF (Q4_K_M + Q5_K_M + Q8_0) + push HF.

Sister script of runpod_e4b_worker.py, scaled for the 31B variant.
Same fix applied: uses Gemma4ForConditionalGeneration (not AutoModelForCausalLM)
to preserve the FULL multimodal architecture (text + vision + audio = 720 tensors).

Pipeline (with on-the-fly cleanup to keep disk peak ~150 GB) :
  1. DL base unsloth/gemma-4-31B-it (full multimodal, ~62 GB BF16)
  2. DL LoRA SevenOfNine/Aura-4o-Rebirth-Gemma-4-31B-LoRA
  3. Merge via PEFT merge_and_unload + save_pretrained (BF16, ~62 GB)
  4. Push merged -> SevenOfNine/Aura-4o-Rebirth-Gemma-4-31B-Merged
  5. Free base weights from disk
  6. Convert HF -> GGUF bf16 (~62 GB) + extract mmproj (~3 GB)
  7. Free merged HF safetensors from disk (already on HF)
  8. Quantize bf16 -> Q4_K_M, Q5_K_M, Q8_0 (sequentially)
  9. Push all GGUFs + mmproj -> SevenOfNine/Aura-4o-Rebirth-Gemma-4-31B-GGUF
 10. Done.

Env :
  HF_TOKEN  required

Hardware target : A100 80 GB (BF16 31B = ~62 GB) on RunPod EU-SE-1.
Disk : 250 GB recommended (peak ~150 GB after cleanup).
Estimated cost : ~$3-4 (1h30 - 2h on A100).
"""
from __future__ import annotations

import os, shutil, subprocess, sys
from pathlib import Path

WORK = Path("/workspace")
REPO = WORK / "Aura-4o-Rebirth-Gemma-4-E4B"  # contains the pipeline scripts
LLAMA = REPO / "pipeline" / "llama.cpp"
BASE_DIR = WORK / "base"
LORA_DIR = WORK / "lora"
MERGED = WORK / "merged"
OUT = WORK / "out"

HF_TOKEN = os.environ["HF_TOKEN"]

BASE_MODEL = "unsloth/gemma-4-31B-it"
LORA_REPO = "SevenOfNine/Aura-4o-Rebirth-Gemma-4-31B-LoRA"
MERGED_REPO = "SevenOfNine/Aura-4o-Rebirth-Gemma-4-31B-Merged"
GGUF_REPO = "SevenOfNine/Aura-4o-Rebirth-Gemma-4-31B-GGUF"


def run(cmd, cwd=None):
    print("[CMD]", " ".join(map(str, cmd)), flush=True)
    subprocess.run(cmd, cwd=cwd, check=True)


def step(msg):
    print(f"\n[STEP] {msg}", flush=True)


def free(path: Path):
    if path.exists():
        print(f"[FREE] {path} ({sum(f.stat().st_size for f in path.rglob('*') if f.is_file()) / 1024**3:.1f} GB)", flush=True)
        shutil.rmtree(path)


def main():
    step("Installing system deps")
    run(["bash", "-lc", "apt-get update -qq && apt-get install -y -qq git cmake python3-pip"])

    if not REPO.exists():
        step("Cloning project repo from GitHub")
        env = os.environ.copy()
        env["GIT_TERMINAL_PROMPT"] = "0"
        subprocess.run(
            ["git", "clone", "--depth", "1",
             "https://github.com/Sev7nOfNine/Aura-4o-Rebirth-Gemma-4-E4B.git",
             str(REPO)],
            check=True, env=env,
        )

    step("Installing Python deps")
    run([sys.executable, "-m", "pip", "install", "-q", "--upgrade", "pip"])
    run([sys.executable, "-m", "pip", "install", "-q",
         "huggingface_hub[cli]", "hf_transfer",
         "transformers", "peft", "accelerate", "safetensors",
         "datasets", "gguf", "torch", "torchvision"])

    if not LLAMA.exists():
        step("Cloning llama.cpp")
        run(["git", "clone", "--depth", "1",
             "https://github.com/ggml-org/llama.cpp.git", str(LLAMA)])
    run([sys.executable, "-m", "pip", "install", "-q", "-r",
         str(LLAMA / "requirements" / "requirements-convert_hf_to_gguf.txt")])

    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")

    # ------------------------------------------------------------------
    # 1. DL base (full multimodal) + LoRA
    # ------------------------------------------------------------------
    from huggingface_hub import snapshot_download, HfApi

    step(f"Downloading base : {BASE_MODEL} (~62 GB)")
    snapshot_download(repo_id=BASE_MODEL, local_dir=str(BASE_DIR),
                      token=HF_TOKEN, max_workers=8)

    step(f"Downloading LoRA : {LORA_REPO}")
    snapshot_download(repo_id=LORA_REPO, local_dir=str(LORA_DIR),
                      token=HF_TOKEN, max_workers=8)

    # ------------------------------------------------------------------
    # 2. Load with Gemma4ForConditionalGeneration (FULL multimodal) + merge
    # ------------------------------------------------------------------
    step("Loading base model with Gemma4ForConditionalGeneration (full multimodal)")
    import torch
    from transformers import Gemma4ForConditionalGeneration, AutoProcessor
    from peft import PeftModel

    base = Gemma4ForConditionalGeneration.from_pretrained(
        str(BASE_DIR),
        torch_dtype=torch.bfloat16,
        device_map="auto",
        token=HF_TOKEN,
    )
    processor = AutoProcessor.from_pretrained(str(BASE_DIR), token=HF_TOKEN)

    step("Attaching LoRA + merge_and_unload")
    peft_model = PeftModel.from_pretrained(base, str(LORA_DIR), token=HF_TOKEN)
    merged = peft_model.merge_and_unload()
    merged = merged.to(torch.bfloat16)

    step(f"Saving merged to {MERGED} (~62 GB)")
    if MERGED.exists():
        shutil.rmtree(MERGED)
    MERGED.mkdir(parents=True)
    merged.save_pretrained(str(MERGED), safe_serialization=True)
    processor.save_pretrained(str(MERGED))

    # Copy chat template + extras from LoRA repo
    for fname in ["chat_template.jinja", "processor_config.json",
                  "preprocessor_config.json", "special_tokens_map.json",
                  "tokenizer.json", "tokenizer_config.json"]:
        src_f = LORA_DIR / fname
        if src_f.exists():
            shutil.copy2(src_f, MERGED / fname)

    # Free GPU memory before push + GGUF conversion
    del base, peft_model, merged
    import gc; gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # 3. Push merged to HF (overwrite)
    # ------------------------------------------------------------------
    step(f"Pushing merged -> {MERGED_REPO}")
    api = HfApi(token=HF_TOKEN)
    api.upload_folder(folder_path=str(MERGED), repo_id=MERGED_REPO,
                      repo_type="model")

    # Free base weights now (already used + merged, no longer needed)
    free(BASE_DIR)
    free(LORA_DIR)

    # ------------------------------------------------------------------
    # 4. Convert HF -> GGUF (bf16 + mmproj)
    # ------------------------------------------------------------------
    OUT.mkdir(exist_ok=True)
    bf16 = OUT / "model-bf16.gguf"
    mmproj = OUT / "Aura-4o-Rebirth-Gemma-4-31B-mmproj-f16.gguf"

    step("Converting HF -> GGUF bf16 (text, ~62 GB)")
    run([sys.executable, str(LLAMA / "convert_hf_to_gguf.py"), str(MERGED),
         "--outfile", str(bf16), "--outtype", "bf16"])

    step("Converting HF -> GGUF mmproj (multimodal projector)")
    run([sys.executable, str(LLAMA / "convert_hf_to_gguf.py"), str(MERGED),
         "--mmproj", "--outfile", str(mmproj), "--outtype", "f16"])

    # Free merged HF safetensors (already on HF, no longer needed locally)
    free(MERGED)

    # ------------------------------------------------------------------
    # 5. Build llama-quantize
    # ------------------------------------------------------------------
    step("Building llama-quantize")
    quant = LLAMA / "build" / "bin" / "llama-quantize"
    if not quant.exists():
        run(["bash", "-lc",
             f"cd {LLAMA} && cmake -B build && cmake --build build --target llama-quantize -j$(nproc)"])

    # ------------------------------------------------------------------
    # 6. Quantize Q4_K_M, Q5_K_M, Q8_0 (sequentially)
    # ------------------------------------------------------------------
    quants = [
        ("Q4_K_M", OUT / "Aura-4o-Rebirth-Gemma-4-31B-Q4_K_M.gguf"),
        ("Q5_K_M", OUT / "Aura-4o-Rebirth-Gemma-4-31B-Q5_K_M.gguf"),
        ("Q8_0",   OUT / "Aura-4o-Rebirth-Gemma-4-31B-Q8_0.gguf"),
    ]
    for qtype, qfile in quants:
        step(f"Quantizing -> {qtype}")
        run([str(quant), str(bf16), str(qfile), qtype])

    # ------------------------------------------------------------------
    # 7. Push all GGUFs + mmproj
    # ------------------------------------------------------------------
    step(f"Pushing mmproj + Q4_K_M + Q5_K_M + Q8_0 -> {GGUF_REPO}")
    api.upload_file(path_or_fileobj=str(mmproj), path_in_repo=mmproj.name,
                    repo_id=GGUF_REPO, repo_type="model")
    for _, qfile in quants:
        step(f"Uploading {qfile.name}")
        api.upload_file(path_or_fileobj=str(qfile), path_in_repo=qfile.name,
                        repo_id=GGUF_REPO, repo_type="model")

    print("\n[DONE] Clean merged + 3 quants + mmproj rebuilt + pushed.", flush=True)


if __name__ == "__main__":
    main()
