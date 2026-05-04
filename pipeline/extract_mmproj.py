"""
Extract mmproj (vision tower) from Aura-4o-Rebirth-Gemma-4-E4B-Merged
and push to the GGUF repo on HuggingFace.

Local-only pipeline (no RunPod). Everything stays on F:/ drive.

Usage :
  # 1. (one-time) create venv inside the project
  python -m venv venv
  venv/Scripts/activate

  # 2. HF_TOKEN comes from .env at the project root (auto-loaded).
  #    Or pass --hf-token / set env var manually.

  # 3. run
  python pipeline/extract_mmproj.py

Steps :
  1. Clone llama.cpp into pipeline/llama.cpp/  (skip if already present)
  2. pip install convert requirements
  3. snapshot_download Merged repo -> pipeline/_merged_tmp/
  4. convert_hf_to_gguf.py --mmproj -> gguf/Aura-4o-Rebirth-Gemma-4-E4B-mmproj-f16.gguf
  5. Upload mmproj to SevenOfNine/Aura-4o-Rebirth-Gemma-4-E4B-GGUF
  6. (optional) cleanup _merged_tmp/

Disk usage peak : ~25 GB during step 3-4 (merged safetensors).
RAM peak        : ~16-20 GB during conversion.
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

# Force every cache to stay on F:/ (NEVER on C:).
PROJECT = Path(__file__).resolve().parent.parent

# Load .env at project root so HF_TOKEN is auto-available.
_env_file = PROJECT / ".env"
if _env_file.exists():
    for _line in _env_file.read_text(encoding="utf-8").splitlines():
        _line = _line.strip()
        if not _line or _line.startswith("#") or "=" not in _line:
            continue
        _k, _v = _line.split("=", 1)
        os.environ.setdefault(_k.strip(), _v.strip())

os.environ.setdefault("HF_HOME", str(PROJECT / ".hf_cache"))
os.environ.setdefault("HF_HUB_CACHE", str(PROJECT / ".hf_cache" / "hub"))
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
os.environ.setdefault("PIP_CACHE_DIR", str(PROJECT / ".pip_cache"))
os.environ.setdefault("TMPDIR", str(PROJECT / ".tmp"))
os.environ.setdefault("TEMP", str(PROJECT / ".tmp"))
os.environ.setdefault("TMP", str(PROJECT / ".tmp"))

PIPELINE = PROJECT / "pipeline"
LLAMA_CPP = PIPELINE / "llama.cpp"
MERGED_TMP = PIPELINE / "_merged_tmp"
GGUF_DIR = PROJECT / "gguf"

MERGED_REPO = "SevenOfNine/Aura-4o-Rebirth-Gemma-4-E4B-Merged"
GGUF_REPO = "SevenOfNine/Aura-4o-Rebirth-Gemma-4-E4B-GGUF"
MMPROJ_NAME = "Aura-4o-Rebirth-Gemma-4-E4B-mmproj-f16.gguf"


def step(msg: str) -> None:
    print(f"\n[STEP] {msg}", flush=True)


def run(cmd: list[str], cwd: Path | None = None, check: bool = True) -> subprocess.CompletedProcess:
    print(f"[CMD] {' '.join(str(c) for c in cmd)}", flush=True)
    return subprocess.run(cmd, cwd=cwd, check=check)


def ensure_llama_cpp() -> None:
    if LLAMA_CPP.exists() and (LLAMA_CPP / "convert_hf_to_gguf.py").exists():
        step("llama.cpp already present, skipping clone")
        return
    step("Cloning llama.cpp")
    LLAMA_CPP.parent.mkdir(parents=True, exist_ok=True)
    run(["git", "clone", "--depth", "1", "https://github.com/ggml-org/llama.cpp", str(LLAMA_CPP)])


def install_requirements() -> None:
    step("Installing convert_hf_to_gguf requirements")
    req = LLAMA_CPP / "requirements" / "requirements-convert_hf_to_gguf.txt"
    run([sys.executable, "-m", "pip", "install", "-r", str(req)])
    run([sys.executable, "-m", "pip", "install", "huggingface_hub[cli]", "hf_transfer"])


def download_merged(token: str) -> None:
    if MERGED_TMP.exists() and any(MERGED_TMP.iterdir()):
        step(f"Merged already downloaded at {MERGED_TMP}, skipping")
        return
    step(f"Downloading {MERGED_REPO} -> {MERGED_TMP}")
    from huggingface_hub import snapshot_download
    snapshot_download(
        repo_id=MERGED_REPO,
        local_dir=str(MERGED_TMP),
        token=token,
        max_workers=8,
    )


def extract_mmproj() -> Path:
    out = GGUF_DIR / MMPROJ_NAME
    GGUF_DIR.mkdir(exist_ok=True)
    step(f"Extracting mmproj -> {out}")
    convert = LLAMA_CPP / "convert_hf_to_gguf.py"
    run([
        sys.executable, str(convert),
        str(MERGED_TMP),
        "--mmproj",
        "--outfile", str(out),
        "--outtype", "f16",
    ])
    size_mb = out.stat().st_size / (1024 ** 2)
    print(f"[OK] mmproj written : {out.name} ({size_mb:.1f} MB)")
    return out


def push_to_hf(mmproj_path: Path, token: str) -> None:
    step(f"Uploading {mmproj_path.name} -> {GGUF_REPO}")
    from huggingface_hub import HfApi
    api = HfApi(token=token)
    api.upload_file(
        path_or_fileobj=str(mmproj_path),
        path_in_repo=mmproj_path.name,
        repo_id=GGUF_REPO,
        repo_type="model",
    )
    print(f"[OK] uploaded to https://huggingface.co/{GGUF_REPO}/blob/main/{mmproj_path.name}")


def cleanup() -> None:
    if not MERGED_TMP.exists():
        return
    ans = input(f"\nDelete temp folder {MERGED_TMP} (~16 GB) ? [y/N] ").strip().lower()
    if ans == "y":
        shutil.rmtree(MERGED_TMP)
        print("[OK] cleaned")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf-token", default=os.environ.get("HF_TOKEN"))
    parser.add_argument("--skip-push", action="store_true", help="Skip HF upload (local extraction only).")
    parser.add_argument("--skip-cleanup", action="store_true")
    args = parser.parse_args()

    if not args.hf_token:
        print("HF_TOKEN missing. Set env var or pass --hf-token.", file=sys.stderr)
        sys.exit(1)

    ensure_llama_cpp()
    install_requirements()
    download_merged(args.hf_token)
    mmproj = extract_mmproj()
    if not args.skip_push:
        push_to_hf(mmproj, args.hf_token)
    if not args.skip_cleanup:
        cleanup()
    print("\n[DONE] Aura E4B mmproj extraction complete.")


if __name__ == "__main__":
    main()
