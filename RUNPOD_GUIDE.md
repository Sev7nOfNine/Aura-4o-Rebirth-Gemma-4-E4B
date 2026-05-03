# Guide RunPod — Fine-tuning Aura sur GPU cloud

Setup pas-à-pas pour entraîner `Aura-Gemma-4-E4B` sur RunPod en ~1-2h pour ~1-2$.

---

## 1. Choix du GPU

| GPU | VRAM | Prix/h | Temps fine-tune | Coût total |
|---|---|---|---|---|
| **A40** | 48 GB | ~$0.40-0.80 | ~2h | **$1-2** ✅ recommandé |
| RTX A6000 | 48 GB | ~$0.50-0.80 | ~2h | $1-2 |
| L40S | 48 GB | ~$0.80-1.20 | ~1h30 | $1-2 |
| H100 80GB | 80 GB | ~$2-3 | ~45 min | $2-3 (plus rapide) |

**Reco : A40 ou RTX A6000** — 48 GB suffisent largement pour LoRA 16-bit (full BF16, pas de QLoRA), excellent rapport qualité/prix.

---

## 2. Création du pod

1. Va sur [runpod.io](https://www.runpod.io/) → Sign up / Login
2. Ajoute du crédit (10$ suffisent largement, tu utiliseras 1-2$)
3. **Deploy** → **GPU Pod** :
   - **GPU type** : A40 (ou RTX A6000)
   - **Template** : `RunPod PyTorch 2.4` ou plus récent (CUDA 12.x)
   - **Container disk** : 50 GB
   - **Volume disk** : 30 GB (persistant, optionnel)
   - **Cliquer Deploy**
4. Attendre 1-2 min que le pod soit ready
5. Cliquer **Connect** → **Connect to Jupyter Lab** ou **Web Terminal**

---

## 3. Setup dans le pod (5 min)

Dans le terminal du pod :

```bash
# Cloner le repo
cd /workspace
git clone https://github.com/Sev7nOfNine/Aura-Gemma-4-E4B.git
cd Aura-Gemma-4-E4B

# Installer les dépendances
pip install --upgrade pip
pip install unsloth unsloth_zoo transformers datasets trl peft accelerate huggingface_hub

# Login HuggingFace (pour pull le modèle + push le résultat)
huggingface-cli login
# → coller le token HF
```

---

## 4. Upload du dataset

Deux options :

### Option A — Via JupyterLab (drag & drop)

1. Connect → Jupyter Lab
2. Naviguer dans `/workspace/`
3. Drag & drop `aura_dataset.jsonl` depuis ton PC
4. Path final : `/workspace/aura_dataset.jsonl`

### Option B — Via wget/curl si dataset déjà sur HF

```bash
huggingface-cli download SevenOfNine/aura-dataset aura_dataset.jsonl \
  --repo-type dataset --local-dir /workspace
```

(à adapter selon que tu as publié ton dataset sur HF ou pas)

---

## 5. Lancement du training

```bash
cd /workspace/Aura-Gemma-4-E4B
python train_aura_runpod.py 2>&1 | tee training.log
```

Tu verras :

- Téléchargement du modèle (~5 min)
- Loading dataset
- `>>> Starting training...` avec stats GPU
- Logs de loss toutes les 10 steps
- Eval toutes les 200 steps

**Estimation : ~1h30-2h** sur A40, ~45-60 min sur H100.

Tu peux fermer le terminal, le training continue (lance via `tmux` ou `screen` si tu préfères).

---

## 6. Récupération du modèle

Le script pousse automatiquement à la fin :

- **LoRA adapters** → `huggingface.co/SevenOfNine/Aura-Gemma-4-E4B-LoRA`
- **GGUF Q8** → `huggingface.co/SevenOfNine/Aura-Gemma-4-E4B`

Sur ton PC local :

```bash
huggingface-cli download SevenOfNine/Aura-Gemma-4-E4B \
  --include "*.gguf" \
  --local-dir "F:\AI\Aura-Gemma-4-E4B\gguf"
```

Tu obtiens `Aura-Gemma-4-E4B.Q8_0.gguf` (~7.5 GB) prêt à charger dans **LM Studio** ou **llama.cpp**.

---

## 7. Stop du pod (ne pas oublier !)

Une fois le téléchargement local terminé :

1. RunPod console → **Pods**
2. Sélectionner le pod → **Stop** (ou **Terminate** pour libérer le disk)

⚠️ **Un pod stoppé continue à coûter pour le storage. Terminate-le si tu n'en as plus besoin.**

---

## Optimisations qualité activées (RunPod uniquement)

| Optimisation | Local (16GB) | RunPod (48GB+) |
|---|---|---|
| Quantization | QLoRA 4-bit (forcé) | **LoRA BF16** (top quality) |
| LoRA rank | 64-128 | **128** |
| max_seq_length | 2048 (limité) | **4096** (aucune troncature) |
| Batch size | 1 | **4** (meilleure stabilité gradients) |
| NEFTune | ✓ | ✓ |
| Train on responses only | ✓ | ✓ |
| Vision/audio préservés | ✓ | ✓ |

LoRA BF16 (vs QLoRA 4-bit) est mesurablement meilleur en qualité finale — c'est le **vrai top** que ton matériel local ne peut pas atteindre. Pour 1-2$, c'est imbattable.
