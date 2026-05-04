"""
Re-merge Aura LoRA into base Gemma 4 E4B with PEFT (clean), overwriting the
corrupted Merged repo on HF.

Background : the Merged repo on HF was built with Unsloth's broken
save_pretrained_merged before the lm_head fix (commit 20e4d56). Symptom :
LM Studio outputs <unused*> / [multimodal] tokens.

This script reproduces the post-fix logic from train_aura_runpod.py:215-252
locally on Mel's PC (CPU, ~30 min, 32 GB RAM enough).

Usage :
  venv/Scripts/activate
  python pipeline/remerge.py
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent

_env_file = PROJECT / ".env"
if _env_file.exists():
    for _line in _env_file.read_text(encoding="utf-8").splitlines():
        _line = _line.strip()
        if not _line or _line.startswith("#") or "=" not in _line:
            continue
        _k, _v = _line.split("=", 1)
        os.environ.setdefault(_k.strip(), _v.strip())

os.environ.setdefault("HF_HOME", str(PROJECT / ".hf_cache"))
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
os.environ.setdefault("PIP_CACHE_DIR", str(PROJECT / ".pip_cache"))
os.environ.setdefault("TMPDIR", str(PROJECT / ".tmp"))
os.environ.setdefault("TEMP", str(PROJECT / ".tmp"))
os.environ.setdefault("TMP", str(PROJECT / ".tmp"))

PIPELINE = PROJECT / "pipeline"
MERGED_DIR = PIPELINE / "_merged_tmp"

MODEL_NAME = "unsloth/gemma-4-E4B-it"
HF_LORA_REPO = "SevenOfNine/Aura-4o-Rebirth-Gemma-4-E4B-LoRA"
HF_MERGED_REPO = "SevenOfNine/Aura-4o-Rebirth-Gemma-4-E4B-Merged"


def step(msg): print(f"\n[STEP] {msg}", flush=True)


def main():
    token = os.environ.get("HF_TOKEN")
    if not token:
        print("HF_TOKEN missing.", file=sys.stderr); sys.exit(1)

    step("Installing required deps (transformers, peft, torch, accelerate)")
    subprocess.run([
        sys.executable, "-m", "pip", "install", "--upgrade",
        "transformers", "peft", "torch", "accelerate", "safetensors",
    ], check=True)

    step("Wiping old broken merged at _merged_tmp/")
    if MERGED_DIR.exists():
        shutil.rmtree(MERGED_DIR)
    MERGED_DIR.mkdir(parents=True)

    step(f"Loading base model (bf16, CPU) : {MODEL_NAME}")
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    base = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
        token=token,
    )
    tok = AutoTokenizer.from_pretrained(MODEL_NAME, token=token)

    step(f"Pulling LoRA adapters : {HF_LORA_REPO}")
    peft_model = PeftModel.from_pretrained(base, HF_LORA_REPO, token=token)

    step("Merging LoRA into base (PEFT merge_and_unload, no Unsloth)")
    merged = peft_model.merge_and_unload()

    step("Forcing bf16 dtype on all tensors")
    merged = merged.to(torch.bfloat16)
    # Untie lm_head if tied to embed_tokens (avoids dtype issues + ensures clean lm_head)
    if hasattr(merged, "tie_weights"):
        try:
            if hasattr(merged, "config"):
                merged.config.tie_word_embeddings = False
            for n, p in merged.named_parameters():
                if p.dtype != torch.bfloat16 and p.is_floating_point():
                    p.data = p.data.to(torch.bfloat16)
        except Exception as e:
            print(f"[WARN] tie/cast cleanup: {e}")

    step(f"Saving clean merged to {MERGED_DIR} (manual sharded torch.save, bypass Windows ctypes)")
    # Manual save: state_dict -> manually sharded .bin files + index.json
    # Avoids transformers' forced safetensors backend that hits Windows int32 ctypes overflow.
    import json
    state = merged.state_dict()
    # Force bf16 on every floating tensor (including possibly-still-fp32 lm_head)
    state = {k: (v.to(torch.bfloat16) if v.is_floating_point() else v) for k, v in state.items()}

    # Shard by ~2GB
    SHARD_LIMIT = 2 * 1024**3
    shards: list[dict] = [{}]
    sizes = [0]
    for k, v in state.items():
        sz = v.element_size() * v.numel()
        if sizes[-1] + sz > SHARD_LIMIT and shards[-1]:
            shards.append({})
            sizes.append(0)
        shards[-1][k] = v
        sizes[-1] += sz

    n = len(shards)
    weight_map: dict[str, str] = {}
    total_size = sum(sizes)
    for i, shard in enumerate(shards):
        fname = f"pytorch_model-{i + 1:05d}-of-{n:05d}.bin" if n > 1 else "pytorch_model.bin"
        out_path = MERGED_DIR / fname
        print(f"  writing {fname} ({sizes[i] / 1024**3:.2f} GB, {len(shard)} tensors)")
        torch.save(shard, str(out_path))
        for k in shard:
            weight_map[k] = fname

    if n > 1:
        index = {"metadata": {"total_size": total_size}, "weight_map": weight_map}
        (MERGED_DIR / "pytorch_model.bin.index.json").write_text(json.dumps(index, indent=2))

    # Save config + generation_config so transformers can reload it
    merged.config.save_pretrained(str(MERGED_DIR))
    if hasattr(merged, "generation_config") and merged.generation_config is not None:
        merged.generation_config.save_pretrained(str(MERGED_DIR))
    tok.save_pretrained(str(MERGED_DIR))

    # Also save processor / chat template files from LoRA repo if present
    step(f"Pulling processor/chat-template files from {HF_LORA_REPO}")
    from huggingface_hub import snapshot_download
    aux = snapshot_download(
        repo_id=HF_LORA_REPO,
        token=token,
        allow_patterns=["chat_template.jinja", "processor_config.json", "preprocessor_config.json", "special_tokens_map.json"],
    )
    for f in Path(aux).iterdir():
        if f.is_file():
            shutil.copy2(f, MERGED_DIR / f.name)

    step(f"Pushing clean merged -> {HF_MERGED_REPO} (overwrite)")
    from huggingface_hub import HfApi
    api = HfApi(token=token)
    api.upload_folder(
        folder_path=str(MERGED_DIR),
        repo_id=HF_MERGED_REPO,
        repo_type="model",
    )

    print("\n[DONE] Clean merged rebuilt + pushed.")
    print("Next : python pipeline/extract_mmproj.py && python pipeline/convert_q8.py")


if __name__ == "__main__":
    main()
