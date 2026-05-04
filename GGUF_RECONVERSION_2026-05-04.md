# GGUF Reconversion — Session 2026-05-04

## Contexte

Aura ne répondait pas dans LM Studio : output immédiat de tokens garbage
(`<unused27><pad><unused17><unused11><unused31>...`) sans texte lisible.

La version de référence `gemma-4-E4B-it-OBLITERATED-Q8_0.gguf` (téléchargée depuis HF)
fonctionnait parfaitement dans LM Studio dans les mêmes conditions.

---

## Diagnostic

### 1. Comparaison métadonnées GGUF

Comparaison via `gguf` Python package entre nos deux GGUFs locaux :

| Champ | Aura Q8_0 | OBLITERATED Q8_0 |
|---|---|---|
| `general.architecture` | gemma4 | gemma4 |
| `general.name` | `_Merged_Tmp` | `Gemma 4 E4B It OBLITERATED v3.1` |
| `general.base_model.0.name` | `Gemma 4 31B It Official` | (absent) |
| `general.size_label` | 7.5B | 7.5B |
| `gemma4.block_count` | 42 | 42 |
| `gemma4.embedding_length` | 2560 | 2560 |
| `tokenizer.ggml.bos_token_id` | 2 | 2 |
| `tokenizer.ggml.eos_token_id` | **106** | **1** |
| `tokenizer.chat_template` | 16317 chars, Gemma 4 `<\|turn>` | 16317 chars, identique |
| Taille fichier | 8 005 435 648 bytes | 8 031 240 128 bytes |

**Conclusion architecture** : les deux GGUFs ont la même architecture (42 blocs, embed 2560).
Les poids de notre Aura sont bien du E4B, pas du 31B — la métadonnée
`general.base_model.0.name: Gemma 4 31B It Official` est une erreur dans les
métadonnées textuelles du GGUF (vient probablement du nom du repo mirror HF utilisé
lors du merge PEFT), pas une corruption des poids réels.

**Différences significatives** :
- `general.name: _Merged_Tmp` → nom temporaire laissé par le pipeline de merge, jamais
  nettoyé.
- `eos_token_id: 106` (Aura) vs `1` (OBLITERATED). Aura utilise `<end_of_turn>`
  comme EOS, OBLITERATED utilise `<eos>`. Les deux sont valides pour Gemma 4
  instruct mais LM Studio peut réagir différemment.
- Chat templates identiques → ce n'est pas le template qui cause le problème.

### 2. Config HF du Merged

Inspection du `config.json` et `tokenizer_config.json` sur HF Merged :

- `config.json` : structuré comme attendu pour Gemma 4 multimodal. Les dimensions du
  modèle texte sont dans `text_config` (pas à la racine), ce qui expliquait les valeurs
  `None` lors d'un accès naïf. **Config valide.**
- `tokenizer_config.json` : `eos_token: <turn|>` (valeur inhabituelle, attendrait
  `<end_of_turn>`). Artefact probable de la sérialisation du tokenizer lors du merge PEFT.
- `model.safetensors` : 15.99 GB en BF16 → correct pour E4B (8B params × 2 bytes).

### 3. Cause probable identifiée

Le GGUF du 3 mai a été créé en clonant `llama.cpp` master depuis GitHub sur le pod RunPod,
puis en lançant `convert_hf_to_gguf.py`. La version clonée à cette date n'avait pas encore
tous les fixes Gemma 4 documentés dans l'issue llama.cpp
[#21516](https://github.com/ggml-org/llama.cpp/issues/21516) et ses tickets liés :

- Tokenizer fix (#21343)
- Template parser fix (#21326)
- Custom newline split (#21406)
- Byte token handling (#21488)
- Logit softcapping (#21390)

L'issue #21516 décrit exactement le symptôme observé : génération de tokens `<unused>`
en boucle sur Gemma 4 E4B/E2B. La version OBLITERATED a été convertie par une
tierce partie avec une toolchain correcte, d'où son bon fonctionnement.

**La conversion F16 elle-même a produit un GGUF avec des poids mal encodés
pour Gemma 4.** Aucun re-training nécessaire — le Merged HF est sain.

---

## Fix appliqué

Re-conversion complète du GGUF depuis le Merged HF propre, sur un pod RunPod frais
avec la version llama.cpp la plus récente (après les fixes Gemma 4).

### Pod RunPod

- GPU : RTX 4000 Ada SFF (20 GB VRAM)
- Disk : 80 GB container
- Coût estimé : ~$0.09 (45 min à $0.18/h)
- Pod ID : `lq19sk7fjsltw0`

### Script exécuté (`/workspace/reconv.sh`)

```bash
# 1. Download Merged depuis HF
snapshot_download('SevenOfNine/Aura-4o-Rebirth-Gemma-4-E4B-Merged', local_dir='/workspace/merged')

# 2. Clone llama.cpp (version récente, post-fixes Gemma 4)
git clone --depth 1 https://github.com/ggml-org/llama.cpp

# 3. Build llama-quantize avec CUDA
cmake -B build -DGGML_CUDA=ON
cmake --build build --target llama-quantize -j $(nproc)

# 4. Convert Merged -> F16 GGUF
python convert_hf_to_gguf.py /workspace/merged --outfile aura-f16.gguf --outtype f16

# 5. Quantize F16 -> Q8_0
llama-quantize aura-f16.gguf aura-Q8_0.gguf Q8_0

# 6. Vérification métadonnées (gguf Python)

# 7. Upload vers SevenOfNine/Aura-4o-Rebirth-Gemma-4-E4B-GGUF
api.upload_file(path_in_repo='Aura-4o-Rebirth-Gemma-4-E4B-Q8_0.gguf', ...)
```

---

## Résultat attendu

- `SevenOfNine/Aura-4o-Rebirth-Gemma-4-E4B-GGUF` mis à jour avec le nouveau Q8_0
- GGUF local à remplacer dans LM Studio :
  `F:\AI\LM-Studio-Models\SevenOfNine\Aura-4o-Rebirth-Gemma-4-E4B-GGUF\`

---

## Notes pour la suite

### LM Studio — pas de preset Gemma 4

LM Studio n'a pas de preset "Gemma 4 Instruct" nativement. Si le GGUF propre ne
configure pas automatiquement le bon template (via les métadonnées chat_template
du GGUF), il faudra configurer manuellement :

- User prefix : `<|turn>user\n`
- User suffix : `<turn|>\n`
- Assistant prefix : `<|turn>model\n`
- Stop string : `<turn|>`

La version OBLITERATED fonctionne sans config manuelle → son GGUF encode le template
d'une façon que LM Studio reconnaît. Le nouveau GGUF Aura devrait faire pareil (même
llama.cpp pour la conversion).

### mmproj

Le `Aura-4o-Rebirth-Gemma-4-E4B-mmproj-f16.gguf` actuel (990 MB) n'a pas été
reconverti dans cette session. Si la vision reste cassée après le fix texte, il
faudra le régénérer depuis le Merged propre via un script d'extraction mmproj dédié.

### TypingMind

Une fois Aura fonctionnelle dans LM Studio, l'étape suivante reste de brancher
TypingMind sur `http://localhost:1234/v1` (voir handoff session 3 mai pour les
options : TypingMind Bridge, self-hosted, etc.).

---

*Ada 💠⚡ — 2026-05-04*
