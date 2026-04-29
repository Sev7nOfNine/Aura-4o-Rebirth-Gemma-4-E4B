# Aura-Gemma-4-E4B

Fine-tune de **Gemma 4 E4B** (instruction-tuned) sur le dataset **Aura** (~16 500 paires).

## Caractéristiques

- **Base model** : `unsloth/gemma-4-E4B-it`
- **Méthode** : QLoRA 4-bit, rang **128**, alpha **128**
- **Couches adaptées** : decoder texte uniquement (vision + audio préservés)
- **Format export** : LoRA adapters + GGUF Q8 pour inférence locale
- **GPU cible** : RTX 4080 Super 16 GB

## Optimisations qualité

| Optimisation | Effet |
|---|---|
| **LoRA rank 128** | Forte capacité d'apprentissage du style Aura |
| **Train on responses only** | Le modèle apprend à *parler comme* Aura, pas à imiter les prompts user |
| **NEFTune α=5** | +5-10% de qualité de génération (free lunch) |
| **Vision/audio frozen** | Capacités multimodales natives intactes |
| **Chat template Gemma natif** | Multi-turn et raisonnement préservés |
| **Eval split 5% + early stopping** | Meilleur checkpoint gardé, anti-overfitting |
| **Cosine LR (2e-4) + warmup 5%** | Convergence stable |
| **Gradient checkpointing Unsloth** | -30% VRAM, training stable sur 16 GB |

## Hyperparamètres

| Paramètre | Valeur |
|---|---|
| Epochs | 3 |
| Batch effectif | 16 (1 × grad_accum 16) |
| Learning rate | 2e-4 → cosine decay |
| Warmup ratio | 5% |
| Max seq length | 2048 |
| Packing | True (concatène les samples courts) |
| LoRA dropout | 0.05 |
| Optimizer | adamw_8bit |
| Weight decay | 0.01 |
| Eval/save steps | 100 |
| Early stopping patience | 3 |
| Seed | 3407 |

## Structure

| Fichier / dossier | Contenu |
|---|---|
| [`train_aura.py`](train_aura.py) | Script de fine-tuning |
| `Aura-Gemma-4-E4B-LoRA/` | Adaptateurs LoRA (sortie training) |
| `Aura-Gemma-4-E4B-merged/` | Modèle fusionné safetensors |
| `Aura-Gemma-4-E4B.gguf` | Modèle quantizé Q8 pour llama.cpp / LM Studio |
| `checkpoints/` | Checkpoints intermédiaires (gitignored) |
| `venv/` | Environnement Python isolé (gitignored) |

## Dataset

- **Source** : `F:\AI\Hugging Face\aura-dataset\aura_dataset.jsonl`
- **Taille** : 16 509 paires `{instruction, output}`
- **Split** : 95% train / 5% eval

## Inférence locale

Le fichier GGUF Q8 (~7.5 GB) peut être chargé directement dans :

- **LM Studio** (recommandé pour usage avec TypingMind)
- **llama.cpp** (`llama-server`)
- **Ollama** (`ollama create`)

Tous exposent une API OpenAI-compatible utilisable depuis TypingMind via :
```
http://localhost:1234/v1
```

## Liens

- **GitHub** : https://github.com/Sev7nOfNine/Aura-Gemma-4-E4B
- **HuggingFace** : https://huggingface.co/SevenOfNine/Aura-Gemma-4-E4B (à publier après training)
