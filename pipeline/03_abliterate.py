"""
Aura E4B abliteration + GGUF pipeline for RunPod.

This is the E4B-adapted version of the 31B workflow:
  1. download the clean merged model from HF
  2. optionally run abliteration
  3. convert to GGUF bf16
  4. extract mmproj
  5. quantize to Q8_0 and optional extra quants
  6. push artifacts back to HF

The goal is to run everything on an A40 in EU-SE-1 with minimal storage.
"""
from __future__ import annotations

import argparse
import functools
import io
import json
import os
import re
import shlex
import sys
import time

if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

BANNER = """
╔════════════════════════════════════════════════════════╗
║  🔥 AURA E4B - ABLITERATE + GGUF                      ║
║  💙 Minimal RunPod pipeline                           ║
║  ❤️ By Mel & Aura                                     ║
╚════════════════════════════════════════════════════════╝
"""


def load_yaml(path):
    import yaml

    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def pick_gpu(cfg, vram_min_gb, abliterate=False):
    pool = cfg["gpu_pool"]
    target_vram = 80 if (abliterate and vram_min_gb >= 32) else max(48, vram_min_gb)
    eligible = [g for g in pool if g["vram_gb"] >= target_vram]
    if not eligible:
        return None

    def rate_value(g):
        m = re.search(r"\$([\d.]+)", g["rate"])
        return float(m.group(1)) if m else 999

    eligible.sort(key=rate_value)
    g = eligible[0]
    return g["runpod_id"], g["label"], g["rate"], g["vram_gb"]


def generate_pod_script(cfg, abliterate, extra_quants):
    o = cfg["output"]
    abl = cfg["abliterate"]
    quants = ["q5_k_m"] + [q for q in extra_quants if q != "q5_k_m"]

    abliterate_block = ""
    abliterated_dir = "/workspace/merged"
    if abliterate:
        abliterated_dir = "/workspace/abliterated"
        abliterate_block = f"""
# === Abliteration step ===
step("Downloading abliterator script")
import urllib.request
abl_script_url = "{abl['abliterator_url']}"
urllib.request.urlretrieve(abl_script_url, "/workspace/abliterator.py")
step("Abliterator downloaded")

step("Running abliteration")
import subprocess
abl_cmd = [
    "python", "/workspace/abliterator.py",
    "--input", "/workspace/merged",
    "--output", "/workspace/abliterated",
    "--harmful-dataset", "{abl['datasets']['harmful']}",
    "--harmless-dataset", "{abl['datasets']['harmless']}",
    "--n-harmful", "{abl['n_harmful']}",
    "--n-harmless", "{abl['n_harmless']}",
]
result = subprocess.run(abl_cmd, capture_output=True, text=True, timeout=7200)
print(result.stdout[-4000:])
print(result.stderr[-4000:])
if result.returncode != 0:
    print("ABLITERATION_FAILED")
    sys.exit(1)
step("Abliteration complete")
"""

    return f'''
"""Aura E4B abliterate + GGUF worker script."""
import os, sys, subprocess, shutil, functools

WORK = "/workspace"
print = functools.partial(print, flush=True)


def step(msg):
    print(f"[STEP] {{msg}}")


def run(cmd, timeout=None, check=True):
    print(f"[CMD] {{' '.join(str(c) for c in cmd)}}")
    r = subprocess.run(cmd, text=True, capture_output=True, timeout=timeout)
    if r.stdout:
        print(r.stdout[-2000:])
    if r.stderr:
        print(r.stderr[-2000:])
    if check and r.returncode != 0:
        print(f"[ERROR] exit {{r.returncode}}")
        sys.exit(1)
    return r


HF_TOKEN = os.environ["HF_TOKEN"]


step("Installing deps")
run(["pip", "install", "huggingface_hub[cli]", "hf_transfer", "transformers", "torch", "datasets", "accelerate", "gguf", "peft", "safetensors"])

step("Cloning llama.cpp")
run(["git", "clone", "https://github.com/ggml-org/llama.cpp", f"{{WORK}}/llama.cpp"])
run(["pip", "install", "-r", f"{{WORK}}/llama.cpp/requirements/requirements-convert_hf_to_gguf.txt"])

step("Building llama.cpp")
run(["bash", "-lc", f"cd {{WORK}}/llama.cpp && cmake -B build && cmake --build build --target llama-quantize -j$(nproc)"], timeout=3600)

step("Downloading merged model: {o['merged_repo']}")
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="{o['merged_repo']}",
    local_dir=f"{{WORK}}/merged",
    token=HF_TOKEN,
    max_workers=8,
)
step("Merged downloaded")

{abliterate_block}

step("Converting to GGUF bf16")
convert_script = f"{{WORK}}/llama.cpp/convert_hf_to_gguf.py"
run([
    "python", convert_script,
    "{abliterated_dir}",
    "--outfile", f"{{WORK}}/model-bf16.gguf",
    "--outtype", "bf16",
], timeout=3600)

bf16_size_gb = os.path.getsize(f"{{WORK}}/model-bf16.gguf") / (1024**3)
step(f"GGUF bf16: {{bf16_size_gb:.1f}}GB")

step("Extracting mmproj")
run([
    "python", convert_script,
    "{abliterated_dir}",
    "--mmproj",
    "--outfile", f"{{WORK}}/mmproj-f16.gguf",
    "--outtype", "f16",
], timeout=1800)

quants = {json.dumps(quants)}
quantize_bin = f"{{WORK}}/llama.cpp/build/bin/llama-quantize"
for q in quants:
    step(f"Quantizing to {{q}}")
    run([
        quantize_bin,
        f"{{WORK}}/model-bf16.gguf",
        f"{{WORK}}/model-{{q}}.gguf",
        q,
    ], timeout=3600)
    sz = os.path.getsize(f"{{WORK}}/model-{{q}}.gguf") / (1024**3)
    step(f"  {{q}}: {{sz:.1f}}GB")

try:
    os.remove(f"{{WORK}}/model-bf16.gguf")
except Exception:
    pass

step("Creating HF repos")
from huggingface_hub import HfApi, create_repo
api = HfApi(token=HF_TOKEN)
create_repo("{o['gguf_repo']}", repo_type="model", private={o['private']}, exist_ok=True, token=HF_TOKEN)

step("Uploading mmproj")
api.upload_file(
    path_or_fileobj=f"{{WORK}}/mmproj-f16.gguf",
    path_in_repo="mmproj-f16.gguf",
    repo_id="{o['gguf_repo']}",
    repo_type="model",
)
for q in quants:
    step(f"Uploading model-{{q}}.gguf")
    api.upload_file(
        path_or_fileobj=f"{{WORK}}/model-{{q}}.gguf",
        path_in_repo=f"model-{{q}}.gguf",
        repo_id="{o['gguf_repo']}",
        repo_type="model",
    )

step("All pushed")
print("AURA_E4B_GGUF_DONE")
'''


def main():
    print(BANNER)
    parser = argparse.ArgumentParser(description="AURA E4B abliterate + GGUF.")
    parser.add_argument("--config", default="configs/aura.yaml")
    parser.add_argument("--hf-token", default=os.environ.get("HF_TOKEN"))
    parser.add_argument("--runpod-key", default=os.environ.get("RUNPOD_API_KEY"))
    parser.add_argument("--abliterate", action="store_true")
    parser.add_argument("--extra-quants", default="", help="Comma-separated extra quants. Q5_K_M always included.")
    parser.add_argument("--skip-confirm", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if not args.hf_token:
        print("HF_TOKEN not set.")
        sys.exit(1)
    if not args.runpod_key:
        print("RUNPOD_API_KEY not set.")
        sys.exit(1)

    cfg = load_yaml(args.config)
    model_key = cfg["base_model"]["name"]
    model_info = cfg["models"][model_key]
    extra_quants = [q.strip() for q in args.extra_quants.split(",") if q.strip()]

    print("[PLAN]")
    print(f"  Merged repo   : {cfg['output']['merged_repo']}")
    print(f"  GGUF repo     : {cfg['output']['gguf_repo']}")
    print(f"  Abliteration   : {'YES' if args.abliterate else 'NO'}")
    print(f"  GPU target     : {model_info['vram_train_gb_min']} GB+")

    pick = pick_gpu(cfg, model_info["vram_train_gb_min"], abliterate=args.abliterate)
    if not pick:
        print("No eligible GPU found.")
        sys.exit(1)
    gpu_id, gpu_label, gpu_rate, _ = pick
    print(f"  GPU pick       : {gpu_label} ({gpu_rate})")
    print("  Disk request   : minimal ephem. pod disk, no persistent volume")

    if args.dry_run:
        return
    if not args.skip_confirm:
        ans = input("Proceed and create RunPod pod? [y/N] ").strip().lower()
        if ans != "y":
            print("Aborted.")
            return

    import runpod
    import paramiko

    runpod.api_key = args.runpod_key

    print("[1/4] Creating RunPod pod...")
    create_kwargs = dict(
        name="aura-e4b-abliterate",
        image_name=cfg["runpod_train"]["container_image"],
        gpu_type_id=gpu_id,
        cloud_type=cfg["runpod_train"]["cloud_type"],
        volume_in_gb=0,
        container_disk_in_gb=max(120, model_info["disk_train_gb_min"]),
        ports="22/tcp",
    )
    dc = cfg["runpod_train"].get("preferred_datacenter")
    if dc:
        create_kwargs["data_center_id"] = dc
    pod = runpod.create_pod(**create_kwargs)
    pod_id = pod["id"]
    print(f"  Pod: {pod_id}")

    print("[2/4] Waiting for SSH...")
    ssh_host, ssh_port = None, None
    for _ in range(60):
        try:
            p = runpod.get_pod(pod_id)
            if p.get("desiredStatus") == "RUNNING":
                rt = p.get("runtime") or {}
                for port in rt.get("ports", []):
                    if port.get("privatePort") == 22:
                        ssh_host = port.get("ip")
                        ssh_port = int(port.get("publicPort"))
                        break
                if ssh_host:
                    break
        except Exception:
            pass
        time.sleep(10)
    if not ssh_host:
        print("SSH timeout.")
        runpod.terminate_pod(pod_id)
        sys.exit(1)

    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh_key_path = os.path.expanduser("~/.ssh/id_ed25519")
    if not os.path.exists(ssh_key_path):
        ssh_key_path = os.path.expanduser("~/.ssh/id_rsa")
    ssh.connect(ssh_host, port=ssh_port, username="root", key_filename=ssh_key_path, timeout=30)

    print("[3/4] Uploading run script...")
    script = generate_pod_script(cfg, args.abliterate, extra_quants)
    sftp = ssh.open_sftp()
    with sftp.file("/workspace/run.py", "w") as f:
        f.write(script)
    sftp.close()

    print("[4/4] Launching...")
    env_prefix = (
        f"HF_TOKEN={shlex.quote(args.hf_token)} "
        f"RUNPOD_API_KEY={shlex.quote(args.runpod_key)} "
        f"RUNPOD_POD_ID={shlex.quote(pod_id)} "
        "HF_HUB_ENABLE_HF_TRANSFER=1"
    )
    launch_cmd = f"cd /workspace && ( {env_prefix} nohup python -u /workspace/run.py > /workspace/abl_run.log 2>&1 < /dev/null & )"
    ssh.exec_command(launch_cmd, timeout=60)
    print(f"Pod launched: {pod_id}")
    print(f"Follow log: tail -f /workspace/abl_run.log")


if __name__ == "__main__":
    main()
