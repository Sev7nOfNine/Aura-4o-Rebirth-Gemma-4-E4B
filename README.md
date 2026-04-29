# Aura-Gemma-4-E4B

Fine-tune de **Gemma 4 E4B** (instruction-tuned) sur le dataset **Aura** (~16 500 paires).

## Caractéristiques

- **Base model** : `unsloth/gemma-4-E4B-it`
- **Méthode** : QLoRA 4-bit (rang 64, alpha 128) sur les couches texte uniquement
- **Vision / multilingue** : préservés (couches non touchées)
- **Format export** : LoRA adapters + GGUF Q8 pour inférence locale
- **GPU cible** : RTX 4080 Super 16 GB

## Structure

| Fichier / dossier | Contenu |
|---|---|
| `train_aura.py` | Script de fine-tuning |
| `Aura-Gemma-4-E4B-LoRA/` | Adaptateurs LoRA (sortie training) |
| `Aura-Gemma-4-E4B-merged/` | Modèle fusionné safetensors |
| `Aura-Gemma-4-E4B.gguf` | Modèle quantizé Q8 pour llama.cpp / LM Studio |
| `checkpoints/` | Checkpoints intermédiaires (gitignored) |

## Inférence locale

Le fichier GGUF Q8 (~7.5 GB) peut être chargé directement dans :
- **LM Studio** (recommandé pour TypingMind)
- **llama.cpp** (`llama-server`)
- **Ollama** (`ollama create`)

Tous exposent une API OpenAI-compatible utilisable depuis TypingMind.
