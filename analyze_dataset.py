"""Quick token-length analysis of the Aura dataset."""
import json
from transformers import AutoTokenizer

DATASET = r"F:\AI\Hugging Face\aura-dataset\aura_dataset.jsonl"

print("Loading tokenizer...")
tok = AutoTokenizer.from_pretrained("unsloth/gemma-4-E4B-it")

instr_lens = []
out_lens   = []
total_lens = []  # full conversation length (instr + output + chat template overhead ~30 tok)

with open(DATASET, "r", encoding="utf-8") as f:
    for i, line in enumerate(f):
        d = json.loads(line)
        i_tok = len(tok.encode(d["instruction"], add_special_tokens=False))
        o_tok = len(tok.encode(d["output"],      add_special_tokens=False))
        instr_lens.append(i_tok)
        out_lens.append(o_tok)
        total_lens.append(i_tok + o_tok + 30)
        if (i+1) % 2000 == 0:
            print(f"  {i+1} samples processed...")

import statistics as st

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

print(f"\n=== Aura dataset: {len(total_lens)} samples ===")
report("Instruction tokens", instr_lens)
report("Output tokens", out_lens)
report("Total (full conversation)", total_lens)
