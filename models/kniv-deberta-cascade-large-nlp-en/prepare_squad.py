"""Download SQuAD 2.0 and convert to our training format.

Usage:
    uv run python models/kniv-deberta-cascade-large-nlp-en/prepare_squad.py
"""
import json, os
from datasets import load_dataset

OUTPUT_DIR = "data/prepared/kniv-deberta-cascade"

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    ds = load_dataset("rajpurkar/squad_v2")

    for split_name, out_name in [("train", "squad_train"), ("validation", "squad_dev")]:
        examples = []
        n_answerable, n_impossible = 0, 0
        for ex in ds[split_name]:
            is_impossible = len(ex["answers"]["text"]) == 0
            if is_impossible:
                n_impossible += 1
            else:
                n_answerable += 1
            examples.append({
                "id": ex["id"],
                "question": ex["question"],
                "context": ex["context"],
                "answers": ex["answers"],
                "is_impossible": is_impossible,
            })
        out_path = f"{OUTPUT_DIR}/{out_name}.json"
        with open(out_path, "w") as f:
            json.dump(examples, f)
        size_mb = os.path.getsize(out_path) / 1e6
        print(f"{out_name}: {len(examples):,} examples ({n_answerable:,} answerable, {n_impossible:,} unanswerable), {size_mb:.1f} MB")

if __name__ == "__main__":
    main()
