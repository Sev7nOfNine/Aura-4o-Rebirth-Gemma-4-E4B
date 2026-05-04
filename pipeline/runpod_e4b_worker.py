"""
RunPod worker : merge LoRA into Gemma 4 E4B + convert to GGUF + push HF.

Replaces the broken 2-step pipeline (local remerge.py + pod convert) with a
single pod-side script. Key fix : uses Gemma4ForConditionalGeneration to load
the FULL multimodal model (vision + audio + text), not AutoModelForCausalLM
which only loads the text decoder and produces a 666-tensor text-only GGUF.

Pipeline :
  1. DL base unsloth/gemma-4-E4B-it (full multimodal)
  2. DL LoRA SevenOfNine/Aura-4o-Rebirth-Gemma-4-E4B-LoRA
  3. Merge via PEFT merge_and_unload + save_pretrained (bf16)
  4. Push merged -> SevenOfNine/Aura-4o-Rebirth-Gemma-4-E4B-Merged
  5. Convert HF -> GGUF (bf16 + mmproj)
  6. Quantize -> Q8_0
  7. Push GGUF + mmproj -> SevenOfNine/Aura-4o-Rebirth-Gemma-4-E4B-GGUF

Env :
  HF_TOKEN  required
  ABLITERATE optional (=1 to abliterate, default 0)
"""
from __future__ import annotations

import os, shutil, subprocess, sys
from pathlib import Path

WORK = Path("/workspace")
REPO = WORK / "Aura-4o-Rebirth-Gemma-4-E4B"
LLAMA = REPO / "pipeline" / "llama.cpp"
BASE_DIR = WORK / "base"
LORA_DIR = WORK / "lora"
MERGED = WORK / "merged"
OUT = WORK / "out"

HF_TOKEN = os.environ["HF_TOKEN"]
ABLITERATE = os.environ.get("ABLITERATE", "0") == "1"

BASE_MODEL = "unsloth/gemma-4-E4B-it"
LORA_REPO = "SevenOfNine/Aura-4o-Rebirth-Gemma-4-E4B-LoRA"
MERGED_REPO = "SevenOfNine/Aura-4o-Rebirth-Gemma-4-E4B-Merged"
GGUF_REPO = "SevenOfNine/Aura-4o-Rebirth-Gemma-4-E4B-GGUF"


def run(cmd, cwd=None):
    print("[CMD]", " ".join(map(str, cmd)), flush=True)
    subprocess.run(cmd, cwd=cwd, check=True)


def step(msg):
    print(f"\n[STEP] {msg}", flush=True)


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
         "datasets", "gguf", "torch"])

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

    step(f"Downloading base : {BASE_MODEL}")
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

    step(f"Saving merged to {MERGED}")
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

    # ------------------------------------------------------------------
    # 3. Push merged to HF (overwrite the broken one)
    # ------------------------------------------------------------------
    step(f"Pushing merged -> {MERGED_REPO}")
    api = HfApi(token=HF_TOKEN)
    api.upload_folder(folder_path=str(MERGED), repo_id=MERGED_REPO,
                      repo_type="model")

    # Free memory before GGUF conversion
    del base, peft_model, merged
    import gc; gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # 4. Convert HF -> GGUF (bf16 + mmproj) + quantize Q8_0
    # ------------------------------------------------------------------
    OUT.mkdir(exist_ok=True)
    bf16 = OUT / "model-bf16.gguf"
    mmproj = OUT / "mmproj-f16.gguf"
    q8 = OUT / "Aura-4o-Rebirth-Gemma-4-E4B-Q8_0.gguf"

    step("Converting HF -> GGUF bf16 (text)")
    run([sys.executable, str(LLAMA / "convert_hf_to_gguf.py"), str(MERGED),
         "--outfile", str(bf16), "--outtype", "bf16"])

    step("Converting HF -> GGUF mmproj (multimodal projector)")
    run([sys.executable, str(LLAMA / "convert_hf_to_gguf.py"), str(MERGED),
         "--mmproj", "--outfile", str(mmproj), "--outtype", "f16"])

    step("Building llama-quantize")
    quant = LLAMA / "build" / "bin" / "llama-quantize"
    if not quant.exists():
        run(["bash", "-lc",
             f"cd {LLAMA} && cmake -B build && cmake --build build --target llama-quantize -j$(nproc)"])

    step("Quantizing -> Q8_0")
    run([str(quant), str(bf16), str(q8), "Q8_0"])

    # ------------------------------------------------------------------
    # 5. Push GGUF + mmproj
    # ------------------------------------------------------------------
    step(f"Pushing mmproj + Q8_0 -> {GGUF_REPO}")
    api.upload_file(path_or_fileobj=str(mmproj), path_in_repo=mmproj.name,
                    repo_id=GGUF_REPO, repo_type="model")
    api.upload_file(path_or_fileobj=str(q8), path_in_repo=q8.name,
                    repo_id=GGUF_REPO, repo_type="model")

    print("\n[DONE] Clean merged + GGUF rebuilt + pushed.", flush=True)


if __name__ == "__main__":
    main()
