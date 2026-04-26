"""Download DocRED and convert to our format.

Usage:
    uv run python models/kniv-deberta-cascade-large-nlp-en/prepare_docred.py
"""
import json, os
from datasets import load_dataset

OUTPUT_DIR = "data/prepared/kniv-deberta-cascade"

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Loading DocRED...")
    try:
        ds = load_dataset("docred", trust_remote_code=True)
    except Exception:
        try:
            ds = load_dataset("thunlp/docred", trust_remote_code=True)
        except Exception:
            # Fallback: download JSON directly from HuggingFace
            from huggingface_hub import hf_hub_download
            import zipfile, io
            print("  Falling back to direct download...")
            ds = {}
            for split, fname in [("train_annotated", "train_annotated.json"), ("validation", "dev.json")]:
                path = hf_hub_download("thunlp/docred", fname, repo_type="dataset")
                with open(path) as f:
                    ds[split] = json.load(f)
            # Convert to list-of-dicts format matching expected structure
            class ListWrapper:
                def __init__(self, data): self.data = data
                def __iter__(self): return iter(self.data)
                def __len__(self): return len(self.data)
            ds = {k: ListWrapper(v) for k, v in ds.items()}

    for split_name, out_name in [("train_annotated", "docred_train"), ("validation", "docred_dev")]:
        if split_name not in ds:
            # Try alternative split names
            alt = {"train_annotated": "train", "validation": "dev"}
            split_name = alt.get(split_name, split_name)
            if split_name not in ds:
                print(f"  Split {split_name} not found, available: {list(ds.keys())}")
                continue

        examples = []
        n_relations = 0
        relation_types = set()

        for doc in ds[split_name]:
            # DocRED has: sents (list of list of tokens), vertexSet (entities), labels (relations)
            sents = doc.get("sents", [])
            vertex_set = doc.get("vertexSet", [])
            labels = doc.get("labels", [])

            # Flatten sentences to get token offsets
            all_tokens = []
            sent_offsets = []
            for sent in sents:
                sent_offsets.append(len(all_tokens))
                all_tokens.extend(sent)

            # Build entity list
            entities = []
            for ent_idx, mentions in enumerate(vertex_set):
                # Take first mention as representative
                m = mentions[0]
                sent_id = m["sent_id"]
                start = sent_offsets[sent_id] + m["pos"][0]
                end = sent_offsets[sent_id] + m["pos"][1] - 1
                ent_type = m.get("type", "UNKNOWN")
                text = " ".join(all_tokens[start:end+1])
                entities.append({"start": start, "end": end, "type": ent_type, "text": text})

            # Build relations
            relations = []
            for rel in labels:
                head_idx = rel["head"]
                tail_idx = rel["tail"]
                rel_type = rel["relation_id"]
                relation_types.add(rel_type)
                relations.append({"head": head_idx, "tail": tail_idx, "relation": rel_type})
                n_relations += 1

            examples.append({
                "words": all_tokens,
                "entities": entities,
                "relations": relations,
            })

        path = f"{OUTPUT_DIR}/{out_name}.json"
        with open(path, "w") as f:
            json.dump(examples, f)
        size_mb = os.path.getsize(path) / 1e6
        print(f"Saved {out_name}: {len(examples):,} docs, {n_relations:,} relations, "
              f"{len(relation_types)} types, {size_mb:.1f} MB")

if __name__ == "__main__":
    main()
