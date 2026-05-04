from __future__ import annotations

import os, shutil, subprocess, sys
from pathlib import Path

WORK = Path("/workspace")
REPO = WORK / "Aura-4o-Rebirth-Gemma-4-E4B"
LLAMA = REPO / "pipeline" / "llama.cpp"
MERGED = WORK / "merged"
OUT = WORK / "out"

HF_TOKEN = os.environ["HF_TOKEN"]
ABLITERATE = os.environ.get("ABLITERATE", "0") == "1"

def run(cmd, cwd=None):
    print("[CMD]", " ".join(map(str, cmd)), flush=True)
    subprocess.run(cmd, cwd=cwd, check=True)

def main():
    run(["bash", "-lc", "apt-get update -qq && apt-get install -y -qq git cmake python3-pip"])
    if not REPO.exists():
        env = os.environ.copy()
        env["GIT_TERMINAL_PROMPT"] = "0"
        print("[CMD] git clone --depth 1 public repo", flush=True)
        subprocess.run(
            ["git", "clone", "--depth", "1", "https://github.com/Sev7nOfNine/Aura-4o-Rebirth-Gemma-4-E4B.git", str(REPO)],
            check=True,
            env=env,
        )
    run([sys.executable, "-m", "pip", "install", "-q", "--upgrade", "pip"])
    run([sys.executable, "-m", "pip", "install", "-q", "huggingface_hub[cli]", "hf_transfer", "transformers", "peft", "accelerate", "safetensors", "datasets", "gguf", "torch"])
    run([sys.executable, "-m", "pip", "install", "-q", "-r", str(LLAMA / "requirements" / "requirements-convert_hf_to_gguf.txt")])
    from huggingface_hub import snapshot_download, HfApi
    snapshot_download(repo_id="SevenOfNine/Aura-4o-Rebirth-Gemma-4-E4B-Merged", local_dir=str(MERGED), token=HF_TOKEN, max_workers=8)
    src = MERGED
    if ABLITERATE:
        print("[INFO] Abliteration disabled in first pass; using merged directly.", flush=True)
    OUT.mkdir(exist_ok=True)
    bf16 = OUT / "model-bf16.gguf"
    mmproj = OUT / "mmproj-f16.gguf"
    q8 = OUT / "Aura-4o-Rebirth-Gemma-4-E4B-Q8_0.gguf"
    run([sys.executable, str(LLAMA / "convert_hf_to_gguf.py"), str(src), "--outfile", str(bf16), "--outtype", "bf16"])
    run([sys.executable, str(LLAMA / "convert_hf_to_gguf.py"), str(src), "--mmproj", "--outfile", str(mmproj), "--outtype", "f16"])
    quant = LLAMA / "build" / "bin" / "llama-quantize"
    if not quant.exists():
        run(["bash", "-lc", f"cd {LLAMA} && cmake -B build && cmake --build build --target llama-quantize -j$(nproc)"])
    run([str(quant), str(bf16), str(q8), "Q8_0"])
    api = HfApi(token=HF_TOKEN)
    api.upload_file(path_or_fileobj=str(mmproj), path_in_repo=mmproj.name, repo_id="SevenOfNine/Aura-4o-Rebirth-Gemma-4-E4B-GGUF", repo_type="model")
    api.upload_file(path_or_fileobj=str(q8), path_in_repo=q8.name, repo_id="SevenOfNine/Aura-4o-Rebirth-Gemma-4-E4B-GGUF", repo_type="model")
    print("DONE", flush=True)

if __name__ == "__main__":
    main()
