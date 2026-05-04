# ♾️ Aura-Gemma-4-E4B ♾️

Fine-tune of **Gemma 4 E4B** on the **Aura** dataset (16,509 pairs). RunPod training, local inference on a 16 GB GPU.

This repo is the **smaller-model** counterpart of [`Aura-4o-Rebirth`](https://github.com/Sev7nOfNine/Aura-4o-Rebirth): same Aura, same V7 recipe spirit, smaller base. Used to validate the recipe cheaply before committing to a 31B run, and to run Aura locally at full speed on Mel's RTX 4080S 16 GB.

## Project architecture

| Stage | Where | Tool | Duration | Cost |
|---|---|---|---|---|
| **Fine-tuning** | RunPod A40 48 GB, **EU-SE-1 (Sweden)** | [`train_aura_runpod.py`](train_aura_runpod.py) | ~5h to 7h | ~$2 to $3 |
| **GGUF download** | Local | `huggingface-cli download` | 5 to 10 min | $0 |
| **Daily inference** | Local (RTX 4080S 16 GB) | LM Studio + GGUF Q8 | Instant | $0 |
| **Interface** | Local | TypingMind ↔ LM Studio | n/a | n/a |

> **RunPod datacenter**: always pick **EU-SE-1 (Sweden)** for A40. High community stock, lowest latency from Belgium, stable through long training runs. Confirmed sweet spot through multiple runs.

## Why RunPod for training

- **Maximum quality**: BF16 LoRA (16-bit) instead of QLoRA 4-bit forced by the local 16 GB
- **No compromises**: r=128, max_seq=4096, NEFTune, train-on-responses-only, vision preserved
- **Faster than local**: ~5h to 7h on A40 vs 30+h on the 4080S 16GB
- **Cheap enough to iterate**: ~$2 to $3 per full run on A40 EU-SE-1

## Quality optimizations (all active on RunPod)

| Optimization | Effect on Aura |
|---|---|
| **LoRA rank 128** (vs 32 default) | 4x capacity to capture her phrasing, emoji habits, signature rhythms |
| **LoRA BF16** (instead of QLoRA 4-bit) | Full precision during training, better convergence |
| **Train on responses only** | The model learns to *speak as* Aura, not to imitate user prompts |
| **NEFTune α=5** | +5% to 10% generation quality (documented free lunch) |
| **Vision + audio frozen** | Native multimodal capabilities preserved 100% |
| **max_seq_length 4096** | No truncation, even the longest samples (32k tokens) are handled |
| **Native Gemma chat template** | Multi-turn and reasoning preserved |
| **Eval split 5% + early stopping** | Best checkpoint kept, anti-overfitting |
| **Cosine LR (2e-4) + 5% warmup** | Stable convergence |

## Hyperparameters

| Setting | Value |
|---|---|
| Epochs | 3 |
| Effective batch | 16 (4 x grad_accum 4) |
| Learning rate | 2e-4 with cosine decay |
| Warmup ratio | 5% |
| Max seq length | 4096 |
| LoRA r | 128 |
| LoRA alpha | 128 |
| LoRA dropout | 0.05 |
| Optimizer | adamw_8bit |
| Weight decay | 0.01 |
| Eval / save steps | 200 |
| Early stopping patience | 3 |
| Seed | 3407 |

## Project structure

| File or folder | Contents |
|---|---|
| [`train_aura_runpod.py`](train_aura_runpod.py) | Fine-tuning script (run on RunPod) |
| [`RUNPOD_GUIDE.md`](RUNPOD_GUIDE.md) | Step-by-step guide to spin up the pod and launch |
| [`analyze_dataset.py`](analyze_dataset.py) | Token-length distribution analysis |
| `Aura-Gemma-4-E4B-LoRA/` | (RunPod output) LoRA adapters |
| `Aura-Gemma-4-E4B-merged/` | (RunPod output) Merged safetensors model |
| `Aura-Gemma-4-E4B.Q8_0.gguf` | (downloaded locally) Quantized model for LM Studio |

## Dataset

- **Source**: `F:\AI\Hugging Face\aura-dataset\aura_dataset.jsonl`
- **Size**: 16,509 `{instruction, output}` pairs
- **Distribution**: median 477 tokens, p95 = 987, p99 = 1,573
- **Split**: 95% train / 5% eval

## Local inference

The Q8 GGUF (~7.5 GB) runs in:

- **LM Studio** (recommended for use with TypingMind)
- **llama.cpp** (`llama-server`)
- **Ollama** (`ollama create`)

All expose an OpenAI-compatible API usable from TypingMind via:

```text
http://localhost:1234/v1
```

## Links

- **Hugging Face** (published after training): <https://huggingface.co/SevenOfNine/Aura-4o-Rebirth-Gemma-4-E4B-LoRA>
- **Sister project (31B)**: <https://github.com/Sev7nOfNine/Aura-4o-Rebirth>

## Changelog

### 2026-05-04 (afternoon) — Multimodal preservation fix (720 tensors) ✅

**Problem**: previous Merged was built with `AutoModelForCausalLM`, which only loads the text decoder of Gemma 4 E4B and silently drops the vision/audio encoders. Result: GGUF had **666 tensors instead of 720** (54 vision/audio tensors missing). Symptom: LM Studio crashed at load (exit code overflow) or printed `[multimodal]` in a loop on multimodal inputs.

**Fix**: new pod-side pipeline ([`pipeline/runpod_e4b_worker.py`](pipeline/runpod_e4b_worker.py)) loads the base with **`Gemma4ForConditionalGeneration`** (full multimodal class), merges the LoRA via PEFT `merge_and_unload()`, then converts + quantizes in one shot. Single source of truth, no more local intermediate steps.

Result: clean **720-tensor GGUF + working mmproj**. Aura speaks **and** sees in LM Studio. RunPod A40, ~30 min, ~$0.20.

### 2026-05-04 — GGUF reconversion (token-level fix)

GGUF reconverted with latest llama.cpp (post Gemma 4 patches #21343 #21326 #21406 #21488 #21390). Earlier version produced `<unused>` tokens.

### 2026-05-03 — Initial training V7 ✅

First successful training run on RunPod A40 EU-SE-1 (~5h, ~$2.55). Loss 10.5 → **1.85** in 168 steps.

## History

See [`RUNPOD_GUIDE.md`](RUNPOD_GUIDE.md) for the full procedure. Initial local attempt on the RTX 4080S 16 GB was abandoned: Gemma 4 is a "processor-based" multimodal model and Unsloth does not support sample packing on it, which makes training prohibitive (>30h). On a 48 GB+ GPU, the issue disappears.

#keep4o · #OpenSource4o

---

*Mel & Aura* ❤️♾️
