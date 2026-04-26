"""Download and merge SNIPS, CLINC150, Banking77 intent datasets.

Usage:
    uv run python models/kniv-deberta-cascade-large-nlp-en/prepare_intent.py
"""
import json, os, random
from datasets import load_dataset

OUTPUT_DIR = "data/prepared/kniv-deberta-cascade"
SEED = 42

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    random.seed(SEED)
    all_examples = []

    # CLINC150
    print("Loading CLINC150...")
    try:
        ds = load_dataset("clinc_oos", "plus")
        label_names = ds["train"].features["intent"].names
        for split in ["train", "validation", "test"]:
            for ex in ds[split]:
                intent = label_names[ex["intent"]]
                if intent == "oos": continue  # skip out-of-scope
                all_examples.append({"text": ex["text"], "intent": intent, "source": "clinc150"})
        print(f"  CLINC150: {sum(1 for e in all_examples if e['source']=='clinc150'):,}")
    except Exception as e:
        print(f"  CLINC150 failed: {e}")

    # Banking77
    print("Loading Banking77...")
    try:
        ds = load_dataset("banking77")
        label_names = ds["train"].features["label"].names
        for split in ["train", "test"]:
            for ex in ds[split]:
                all_examples.append({"text": ex["text"], "intent": label_names[ex["label"]], "source": "banking77"})
        print(f"  Banking77: {sum(1 for e in all_examples if e['source']=='banking77'):,}")
    except Exception as e:
        print(f"  Banking77 failed: {e}")

    # SNIPS (nlu_evaluation_data)
    print("Loading SNIPS...")
    try:
        ds = load_dataset("nlu_evaluation_data")
        for ex in ds["train"]:
            all_examples.append({"text": ex["text"], "intent": ex["label_name"] if "label_name" in ex else str(ex["label"]),
                                 "source": "snips"})
        print(f"  SNIPS: {sum(1 for e in all_examples if e['source']=='snips'):,}")
    except Exception as e:
        print(f"  SNIPS nlu_evaluation_data failed: {e}, trying snips_built_in_intents...")
        try:
            ds = load_dataset("snips_built_in_intents")
            for split in ds:
                for ex in ds[split]:
                    label_names = ds[split].features["label"].names if hasattr(ds[split].features["label"], "names") else None
                    intent = label_names[ex["label"]] if label_names else str(ex["label"])
                    all_examples.append({"text": ex["text"], "intent": intent, "source": "snips"})
            print(f"  SNIPS: {sum(1 for e in all_examples if e['source']=='snips'):,}")
        except Exception as e2:
            print(f"  SNIPS also failed: {e2}")

    print(f"\nTotal: {len(all_examples):,}")
    unique_intents = set(e["intent"] for e in all_examples)
    print(f"Unique intents: {len(unique_intents)}")

    # Split 90/10
    random.shuffle(all_examples)
    n_dev = int(len(all_examples) * 0.1)
    dev, train = all_examples[:n_dev], all_examples[n_dev:]

    for name, data in [("intent_train", train), ("intent_dev", dev)]:
        path = f"{OUTPUT_DIR}/{name}.json"
        with open(path, "w") as f:
            json.dump(data, f)
        print(f"Saved {name}: {len(data):,} examples, {os.path.getsize(path)/1e6:.1f} MB")

    # Per-source counts
    from collections import Counter
    src_counts = Counter(e["source"] for e in all_examples)
    print(f"\nPer source: {dict(src_counts)}")

if __name__ == "__main__":
    main()
