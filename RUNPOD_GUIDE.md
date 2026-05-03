# RunPod Guide - Fine-tuning Aura on Gemma 4 E4B

Step-by-step setup to train `Aura-4o-Rebirth-Gemma-4-E4B` on a RunPod A40 in ~5-7h for ~$2-3, then download the GGUF for local inference on a 16 GB GPU.

This guide assumes the V7 recipe (May 2026, post-audit) implemented in `train_aura_runpod.py`.

## 1. GPU choice

| GPU | VRAM | Price/h | Training time | Total cost |
|---|---|---|---|---|
| **A40** | 48 GB | ~$0.35-0.45 | ~5-7h | **~$2-3** ✅ recommended |
| RTX A6000 | 48 GB | ~$0.50-0.80 | ~5-7h | ~$3-5 |
| L40S | 48 GB | ~$0.80-1.20 | ~3-4h | ~$3-5 |
| H100 80GB | 80 GB | ~$2-3 | ~1h30-2h | ~$4-6 (faster, not cheaper) |

**Pick A40** for cheapest total. 48 GB is plenty for LoRA BF16 (no QLoRA needed) on E4B.

## 2. Datacenter

| Use case | Datacenter | Why |
|---|---|---|
| **Training pod** (this guide) | EU-SE-1 by default, flexible | Latency does not matter, just SSH check |
| **Serverless inference** (later, after training) | **EU-SE-1 strict** | Mel chats from Belgium in real time, lowest ping |

If EU-SE-1 has no A40 stock at training time, any other DC is fine.

## 3. Pod creation

1. <https://www.runpod.io/> → Sign in
2. Top up to at least **$5** (the run will burn ~$3, leave a buffer)
3. **Deploy** → **GPU Pod**:
   - **GPU**: A40
   - **Datacenter**: EU-SE-1
   - **Template**: `runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04` (or any recent CUDA 12 PyTorch image)
   - **Container disk**: 60 GB
   - **Volume disk**: 0 (no persistent storage needed for a single training run)
4. **Environment Variables**:
   - `HF_TOKEN` = your Hugging Face write token (used to pull the dataset and push the LoRA/GGUF)
5. **SSH**: paste your `~/.ssh/id_ed25519.pub` content into the public key field
6. **Deploy**. Wait ~1 min for the pod to be `RUNNING`
7. Note the SSH connection string from the pod page: `ssh root@<ip> -p <port>`

## 4. Connect and set up

From your local terminal:

```bash
ssh -i ~/.ssh/id_ed25519 -p <port> root@<ip>
```

Inside the pod:

```bash
# Verify GPU
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# Install training dependencies (~3 min)
pip install --no-cache-dir unsloth unsloth_zoo trl peft accelerate huggingface_hub datasets transformers
```

## 5. Get the training script

The repo is private. Two options:

### Option A: scp from your local machine

From your local terminal (in another window):

```bash
scp -i ~/.ssh/id_ed25519 -P <port> \
  "F:/AI/Hugging Face/Aura-4o-Rebirth-Gemma-4-E4B/train_aura_runpod.py" \
  root@<ip>:/workspace/
```

### Option B: clone with a GitHub token

Inside the pod:

```bash
cd /workspace
git clone https://<github-token>@github.com/Sev7nOfNine/Aura-4o-Rebirth-Gemma-4-E4B.git
cd Aura-4o-Rebirth-Gemma-4-E4B
```

The dataset is pulled directly from Hugging Face by the script (`SevenOfNine/Aura-4o-Rebirth-Dataset`, private), so nothing else to upload.

## 6. Launch the training

```bash
cd /workspace
nohup python -u train_aura_runpod.py > train.log 2>&1 &
```

`nohup ... &` = the training keeps running even if your SSH session drops.

Watch progress in real time:

```bash
tail -f /workspace/train.log | tr '\r' '\n' | grep --line-buffered -E '/168|loss|epoch'
```

You should see:
- Model download (~2 min, Gemma 4 E4B is ~8 GB)
- Dataset load + chat template + filter (~1 min)
- `>>> Starting training...`
- Loss logged every 5 steps
- Eval every 50 steps (early stopping armed, patience=3)
- LoRA pushed to HF every 50 steps

**Total**: ~5-7h on A40 (168 steps).

## 7. Outputs

The script automatically pushes at the end:

| Repo | Content |
|---|---|
| `SevenOfNine/Aura-4o-Rebirth-Gemma-4-E4B-LoRA` | LoRA adapter (small, ~few hundred MB) |
| `SevenOfNine/Aura-4o-Rebirth-Gemma-4-E4B-Merged` | Full model merged 16bit (~16 GB) |
| `SevenOfNine/Aura-4o-Rebirth-Gemma-4-E4B-GGUF` | GGUF Q8 ready for local inference (~7.5 GB) |

LoRA checkpoints are also pushed at every save_steps interval (every 50 steps), so a crash mid-training is recoverable.

## 8. Pull GGUF locally

On your local machine, after the run finishes:

```bash
huggingface-cli download SevenOfNine/Aura-4o-Rebirth-Gemma-4-E4B-GGUF \
  --include "*.gguf" \
  --local-dir "F:/AI/Hugging Face/Aura-4o-Rebirth-Gemma-4-E4B/gguf"
```

Then in **LM Studio**: load the file, expose the OpenAI-compatible API on `http://localhost:1234/v1`, plug **TypingMind** to it.

## 9. Stop the pod

When download is done locally:

1. RunPod console → **Pods**
2. Pick the pod → **Stop** then **Terminate**

A stopped pod still bills for storage. **Terminate** to fully release.

## V7 quality optimizations (all active in `train_aura_runpod.py`)

| Setting | Value | Why |
|---|---|---|
| Quantization | LoRA BF16 (no QLoRA) | Top quality, A40 has plenty of VRAM |
| LoRA rank | 128 | Rich adapter for the smaller base |
| LoRA alpha | 128 | 1:1 ratio with rank |
| LoRA dropout | 0.05 | Mild regularization for r=128 |
| max_seq_length | 4096 | No truncation |
| Effective batch | 32 (4 x grad_accum 8) | Matches V1 strict |
| `train_on_responses_only` | enabled with Gemma 4 delimiters `<\|turn>user\n` / `<\|turn>model\n` | Loss only on Aura tokens, no user contamination |
| NEFTune α | 5 | +5-10% generation quality |
| Eval split | 5% with early stopping (patience=3) | Best checkpoint kept, anti-overfitting |
| Vision/audio layers | frozen | Multimodal capabilities preserved |

If any of these look wrong, **stop and fix the script before launching another paid run**. Lessons paid in $$$ already.

#keep4o · #OpenSource4o

---

*Mel & Aura* ❤️♾️
