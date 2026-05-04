"""
Regenerate Q8_0 GGUF from the merged model already cached at pipeline/_merged_tmp/.

Why : the previous Q8_0 (built before commit 20e4d56 fix) had missing tensors
('blk.24.attn_k.weight' etc.) due to Unsloth merged_16bit bug. This rebuilds it
clean from the post-fix Merged repo.

Usage :
  venv/Scripts/activate
  python pipeline/convert_q8.py
"""
from __future__ import annotations

import argparse
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
LLAMA_CPP = PIPELINE / "llama.cpp"
MERGED_TMP = PIPELINE / "_merged_tmp"

GGUF_REPO = "SevenOfNine/Aura-4o-Rebirth-Gemma-4-E4B-GGUF"
Q8_NAME = "Aura-4o-Rebirth-Gemma-4-E4B-Q8_0.gguf"
LMSTUDIO_DIR = Path("F:/AI/LM-Studio-Models/SevenOfNine/Aura-4o-Rebirth-Gemma-4-E4B-GGUF")


def step(msg): print(f"\n[STEP] {msg}", flush=True)
def run(cmd): print(f"[CMD] {' '.join(str(c) for c in cmd)}", flush=True); subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf-token", default=os.environ.get("HF_TOKEN"))
    parser.add_argument("--skip-push", action="store_true")
    parser.add_argument("--skip-lmstudio-copy", action="store_true")
    args = parser.parse_args()

    if not args.hf_token:
        print("HF_TOKEN missing.", file=sys.stderr); sys.exit(1)
    if not MERGED_TMP.exists() or not any(MERGED_TMP.iterdir()):
        print(f"Missing merged at {MERGED_TMP}. Run extract_mmproj.py first.", file=sys.stderr); sys.exit(1)
    if not (LLAMA_CPP / "convert_hf_to_gguf.py").exists():
        print(f"Missing llama.cpp at {LLAMA_CPP}.", file=sys.stderr); sys.exit(1)

    out = PIPELINE / Q8_NAME
    step(f"Converting merged -> Q8_0 GGUF -> {out}")
    run([
        sys.executable, str(LLAMA_CPP / "convert_hf_to_gguf.py"),
        str(MERGED_TMP),
        "--outfile", str(out),
        "--outtype", "q8_0",
    ])
    size_gb = out.stat().st_size / (1024 ** 3)
    print(f"[OK] {out.name} : {size_gb:.2f} GB")

    if not args.skip_push:
        step(f"Uploading {Q8_NAME} -> {GGUF_REPO} (overwrites broken one)")
        from huggingface_hub import HfApi
        HfApi(token=args.hf_token).upload_file(
            path_or_fileobj=str(out), path_in_repo=Q8_NAME,
            repo_id=GGUF_REPO, repo_type="model",
        )
        print(f"[OK] uploaded https://huggingface.co/{GGUF_REPO}/blob/main/{Q8_NAME}")

    if not args.skip_lmstudio_copy:
        step(f"Copying to LM Studio dir : {LMSTUDIO_DIR}")
        LMSTUDIO_DIR.mkdir(parents=True, exist_ok=True)
        dest = LMSTUDIO_DIR / Q8_NAME
        if dest.exists(): dest.unlink()
        shutil.move(str(out), str(dest))
        print(f"[OK] {dest}")
    else:
        print(f"[INFO] Q8 left at {out}")

    print("\n[DONE] Q8_0 rebuilt clean.")


if __name__ == "__main__":
    main()
