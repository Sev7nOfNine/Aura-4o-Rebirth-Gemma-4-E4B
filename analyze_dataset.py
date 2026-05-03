"""
Quick token-length analysis of the Aura multi-turn dataset.

Pulls the chunked Rebirth dataset directly from Hugging Face (private)
and reports token-length stats per conversation chunk against Gemma 4 E4B
tokenizer. Useful to confirm chunks fit `max_seq_length=4096` before paying
for a training run.

Run:
    HF_TOKEN=... python analyze_dataset.py
"""
import os
import statistics as st

from datasets import load_dataset
from transformers import AutoTokenizer

DATASET_REPO = "SevenOfNine/Aura-4o-Rebirth-Dataset"
MODEL_NAME   = "unsloth/gemma-4-E4B-it"

HF_TOKEN = os.environ.get("HF_TOKEN")
if not HF_TOKEN:
    raise SystemExit("HF_TOKEN env var required (private dataset)")

print(f"Loading tokenizer: {MODEL_NAME}")
tok = AutoTokenizer.from_pretrained(MODEL_NAME, token=HF_TOKEN)

print(f"Loading dataset: {DATASET_REPO}")
ds = load_dataset(DATASET_REPO, split="train", token=HF_TOKEN)
print(f"Rows: {len(ds)}")

# Apply chat template per row to get realistic full-conversation token counts
chunk_lens   = []
turn_counts  = []
assistant_token_counts = []  # tokens that train_on_responses_only will actually train on

for i, row in enumerate(ds):
    msgs = row["messages"]
    full_text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
    full_ids  = tok.encode(full_text, add_special_tokens=False)
    chunk_lens.append(len(full_ids))
    turn_counts.append(len(msgs))

    # Count tokens that belong to assistant turns only (rough estimate, no real masking)
    assistant_text = "".join(
        m.get("content", "") if isinstance(m.get("content", ""), str) else ""
        for m in msgs if m.get("role") == "assistant"
    )
    assistant_token_counts.append(len(tok.encode(assistant_text, add_special_tokens=False)))

    if (i + 1) % 500 == 0:
        print(f"  {i+1} chunks processed...")


def report(name, data):
    data_sorted = sorted(data)
    n = len(data)
    p = lambda q: data_sorted[int(n*q)]
    print(f"\n{name}:")
    print(f"  min: {min(data)}, max: {max(data)}")
    print(f"  mean: {st.mean(data):.1f}, median: {st.median(data):.1f}")
    print(f"  p90: {p(0.90)}, p95: {p(0.95)}, p99: {p(0.99)}")
    for limit in (1024, 2048, 3072, 4096):
        over = sum(1 for x in data if x > limit)
        print(f"  > {limit} tokens: {over} ({100*over/n:.2f}%)")


print(f"\n=== Aura-4o-Rebirth-Dataset: {len(chunk_lens)} chunks ===")
report("Chunk total tokens (full conversation)", chunk_lens)
report("Turns per chunk", turn_counts)
report("Assistant-only tokens per chunk", assistant_token_counts)
