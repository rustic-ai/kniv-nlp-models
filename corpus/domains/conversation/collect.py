"""Collect agentic conversation text preserving multi-turn structure.

Each utterance is saved with conv_id, turn_idx, and speaker so that
the preprocessor can link prev_text for conversational context.

Sources (all commercially licensed):
  1. Taskmaster 1/2/3 (CC BY-4.0) — task completion dialogs
  2. OASST1 (Apache 2.0) — human-assistant conversations
  3. MultiWOZ 2.2 (Apache 2.0) — goal-driven multi-domain
  4. Glaive Function Calling (Apache 2.0) — tool use
  5. Discord Dialogues (Apache 2.0) — casual conversations

Usage:
    python -m corpus.domains.conversation.collect
    python -m corpus.domains.conversation.collect --source taskmaster
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import yaml

DOMAIN_DIR = Path(__file__).parent
OUTPUT_DIR = Path(__file__).parent.parent.parent / "output" / "raw" / "conversation"


def load_config() -> dict:
    with open(DOMAIN_DIR / "config.yaml") as f:
        return yaml.safe_load(f)


def _save_utterances(utterances: list[dict], output_file: Path):
    with open(output_file, "w") as f:
        for u in utterances:
            f.write(json.dumps(u) + "\n")
    print(f"  Saved {len(utterances)} utterances → {output_file}", flush=True)


# ── Taskmaster 1/2/3 ─────────────────────────────────────────────

def collect_taskmaster(config: dict):
    cfg = config["sources"]["taskmaster"]
    out_dir = OUTPUT_DIR / "taskmaster"
    out_dir.mkdir(parents=True, exist_ok=True)

    output_file = out_dir / "utterances.jsonl"
    if output_file.exists():
        print("Taskmaster: already collected.", flush=True)
        return

    from datasets import load_dataset

    max_utt = cfg.get("max_utterances", 30000)
    utterances = []

    for ds_name in cfg["datasets"]:
        print(f"  Loading {ds_name}...", flush=True)
        try:
            dataset = load_dataset(ds_name, split="train", trust_remote_code=True)
        except Exception:
            try:
                dataset = load_dataset(ds_name, revision="refs/convert/parquet", split="train")
            except Exception as e:
                print(f"  ⚠ Failed to load {ds_name}: {e}", flush=True)
                continue

        for item in dataset:
            conv_id = item.get("conversation_id", "")
            turns = item.get("utterances", [])
            if isinstance(turns, list):
                for turn in turns:
                    text = turn.get("text", "") if isinstance(turn, dict) else str(turn)
                    # Keep every turn. A length filter here deletes turns from the MIDDLE
                    # of a conversation, leaving gaps in turn_idx (69% of
                    # Taskmaster conversations) so a window built over them
                    # splices non-adjacent turns. It also removes exactly the
                    # short acknowledgements CLS needs: 'ok', 'thanks', 'yes'
                    # are Feedback and Social. Filtering belongs downstream,
                    # where config.filtering already defines it.
                    if text and text.strip():
                        utterances.append({
                            "text": text.strip(),
                            "source": f"taskmaster/{ds_name}",
                            "domain": "conversation",
                            "conv_id": f"tm-{conv_id}",
                            "turn_idx": turn.get("index", 0) if isinstance(turn, dict) else 0,
                            "speaker": (turn.get("speaker", "").lower() if isinstance(turn, dict) else ""),
                        })
            if len(utterances) >= max_utt:
                break
        if len(utterances) >= max_utt:
            break

    utterances = utterances[:max_utt]
    _save_utterances(utterances, output_file)
    print(f"Taskmaster: {len(utterances)} utterances", flush=True)


# ── OASST1 ────────────────────────────────────────────────────────

def collect_oasst(config: dict):
    cfg = config["sources"]["oasst"]
    out_dir = OUTPUT_DIR / "oasst"
    out_dir.mkdir(parents=True, exist_ok=True)

    output_file = out_dir / "utterances.jsonl"
    if output_file.exists():
        print("OASST: already collected.", flush=True)
        return

    from datasets import load_dataset

    print("  Loading OASST1...", flush=True)
    dataset = load_dataset(cfg["dataset"], split="train")
    max_utt = cfg.get("max_utterances", 30000)

    # OASST1 is a TREE, not a linear dialogue: a message_tree_id contains
    # several alternative replies to the same parent, all at the same depth.
    # Indexing by depth makes sibling branches look like consecutive turns,
    # so a window over them would splice competing answers together as if
    # they were a conversation — worse than a gap, because it looks valid.
    #
    # One root-to-leaf path is one conversation. Only the LONGEST path per
    # tree is kept: emitting every path would repeat each shared prefix
    # many times and fill the corpus with near-duplicate windows.
    messages = {}
    for item in dataset:
        if item.get("lang") != "en":
            continue
        messages[item["message_id"]] = item

    children = {}
    roots = []
    for mid, msg in messages.items():
        parent = msg.get("parent_id")
        if parent and parent in messages:
            children.setdefault(parent, []).append(mid)
        elif not parent:
            roots.append(mid)

    def longest_path(mid):
        kids = children.get(mid, [])
        if not kids:
            return [mid]
        return [mid] + max((longest_path(k) for k in kids), key=len)

    utterances = []
    for root in roots:
        path = longest_path(root)
        tree_id = messages[root].get("message_tree_id", root)
        turn = 0
        for mid in path:
            msg = messages[mid]
            text = (msg.get("text") or "").strip()
            if not text:
                continue
            utterances.append({
                "text": text,
                "source": "oasst",
                "domain": "conversation",
                "conv_id": f"oasst-{tree_id}",
                "turn_idx": turn,
                "speaker": "user" if msg.get("role") == "prompter" else "assistant",
            })
            turn += 1
        if len(utterances) >= max_utt:
            break

    _save_utterances(utterances, output_file)
    print(f"OASST: {len(utterances)} utterances", flush=True)


# ── MultiWOZ 2.2 ─────────────────────────────────────────────────

def collect_multiwoz(config: dict):
    cfg = config["sources"]["multiwoz"]
    out_dir = OUTPUT_DIR / "multiwoz"
    out_dir.mkdir(parents=True, exist_ok=True)

    output_file = out_dir / "utterances.jsonl"
    if output_file.exists():
        print("MultiWOZ: already collected.", flush=True)
        return

    from datasets import load_dataset

    print("  Loading MultiWOZ 2.2...", flush=True)
    try:
        dataset = load_dataset(cfg["dataset"], split="train", trust_remote_code=True)
    except Exception:
        dataset = load_dataset(cfg["dataset"], revision="refs/convert/parquet", split="train")

    max_utt = cfg.get("max_utterances", 20000)
    utterances = []

    for item in dataset:
        conv_id = item.get("dialogue_id", "")
        turns = item.get("turns", {})
        turn_ids = turns.get("turn_id", [])
        speakers = turns.get("speaker", [])
        texts = turns.get("utterance", [])

        for tid, spk, text in zip(turn_ids, speakers, texts):
            # see note above: no per-turn length filter in raw collection
            if text and text.strip():
                utterances.append({
                    "text": text.strip(),
                    "source": "multiwoz",
                    "domain": "conversation",
                    "conv_id": f"mwoz-{conv_id}",
                    "turn_idx": int(tid),
                    "speaker": "user" if spk == 0 else "system",
                })
            if len(utterances) >= max_utt:
                break
        if len(utterances) >= max_utt:
            break

    utterances = utterances[:max_utt]
    _save_utterances(utterances, output_file)
    print(f"MultiWOZ: {len(utterances)} utterances", flush=True)


# ── Glaive Function Calling ───────────────────────────────────────

def collect_glaive(config: dict):
    cfg = config["sources"]["glaive"]
    out_dir = OUTPUT_DIR / "glaive"
    out_dir.mkdir(parents=True, exist_ok=True)

    output_file = out_dir / "utterances.jsonl"
    if output_file.exists():
        print("Glaive: already collected.", flush=True)
        return

    from datasets import load_dataset

    print("  Loading Glaive Function Calling...", flush=True)
    dataset = load_dataset(cfg["dataset"], split="train")
    max_utt = cfg.get("max_utterances", 15000)

    utterances = []
    for item_idx, item in enumerate(dataset):
        chat = item.get("chat", "")
        if not isinstance(chat, str):
            continue

        turn_idx = 0
        for line in chat.split("\n"):
            line = line.strip()
            if line.startswith(("USER:", "ASSISTANT:")):
                speaker = "user" if line.startswith("USER:") else "assistant"
                text = line.split(":", 1)[1].strip()
                # length filter dropped; the JSON guard stays (tool-call payloads)
                if text and text.strip() and not text.startswith(("{", "[")):
                    utterances.append({
                        "text": text,
                        "source": "glaive",
                        "domain": "conversation",
                        "conv_id": f"glaive-{item_idx}",
                        "turn_idx": turn_idx,
                        "speaker": speaker,
                    })
                    turn_idx += 1
            if len(utterances) >= max_utt:
                break
        if len(utterances) >= max_utt:
            break

    utterances = utterances[:max_utt]
    _save_utterances(utterances, output_file)
    print(f"Glaive: {len(utterances)} utterances", flush=True)


# ── Discord Dialogues ──────────────────────────────────────────────

def collect_discord(config: dict):
    cfg = config["sources"]["discord"]
    out_dir = OUTPUT_DIR / "discord"
    out_dir.mkdir(parents=True, exist_ok=True)

    output_file = out_dir / "utterances.jsonl"
    if output_file.exists():
        print("Discord: already collected.", flush=True)
        return

    from datasets import load_dataset

    print("  Loading Discord Dialogues...", flush=True)
    dataset = load_dataset(cfg["dataset"], split="train", streaming=True)
    max_utt = cfg.get("max_utterances", 30000)

    utterances = []
    for item_idx, item in enumerate(dataset):
        text = item.get("text", "")
        turns = re.split(r"<\|im_start\|>(user|assistant)\n?", text)

        # turns alternates: [preamble, role, text, role, text, ...]
        turn_idx = 0
        i = 1  # skip preamble
        while i + 1 < len(turns):
            speaker = turns[i]
            content = re.sub(r"<\|im_end\|>", "", turns[i + 1]).strip()
            # see note above: no per-turn length filter in raw collection
            if content and content.strip():
                utterances.append({
                    "text": content,
                    "source": "discord",
                    "domain": "conversation",
                    "conv_id": f"discord-{item_idx}",
                    "turn_idx": turn_idx,
                    "speaker": speaker,
                })
                turn_idx += 1
            i += 2
            if len(utterances) >= max_utt:
                break
        if len(utterances) >= max_utt:
            break

    utterances = utterances[:max_utt]
    _save_utterances(utterances, output_file)
    print(f"Discord: {len(utterances)} utterances", flush=True)


# ── Main ──────────────────────────────────────────────────────────

COLLECTORS = {
    "taskmaster": collect_taskmaster,
    "oasst": collect_oasst,
    "multiwoz": collect_multiwoz,
    "glaive": collect_glaive,
    "discord": collect_discord,
}


def main():
    parser = argparse.ArgumentParser(description="Collect conversation corpus")
    parser.add_argument("--source", choices=list(COLLECTORS.keys()),
                        help="Collect a specific source only")
    args = parser.parse_args()

    config = load_config()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.source:
        sources = {args.source: COLLECTORS[args.source]}
    else:
        sources = COLLECTORS

    for name, collector in sources.items():
        print(f"\n{'=' * 50}", flush=True)
        print(f"Collecting: {name}", flush=True)
        print(f"{'=' * 50}", flush=True)
        try:
            collector(config)
        except Exception as e:
            print(f"⚠ {name} failed: {e}", flush=True)
            import traceback
            traceback.print_exc()

    print(f"\nCollection complete. Raw data in {OUTPUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
