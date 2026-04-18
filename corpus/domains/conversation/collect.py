"""Collect agentic conversation text from multiple open datasets.

Sources (all commercially licensed):
  1. Taskmaster 1/2/3 (CC BY-4.0) — task completion dialogs
  2. OASST1 (Apache 2.0) — human-assistant conversations
  3. MultiWOZ 2.2 (Apache 2.0) — goal-driven multi-domain
  4. ABCD (MIT) — customer service with actions
  5. AirDialogue (Apache 2.0) — flight booking
  6. Glaive Function Calling (Apache 2.0) — tool use
  7. Agent-FLAN (Apache 2.0) — agent traces

Usage:
    python -m corpus.domains.conversation.collect
    python -m corpus.domains.conversation.collect --source taskmaster
"""

from __future__ import annotations

import argparse
import json
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
    """Load Taskmaster dialog datasets."""
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
            turns = item.get("utterances", item.get("dialog", []))
            if isinstance(turns, list):
                for turn in turns:
                    text = turn.get("text", turn.get("utterance", "")) if isinstance(turn, dict) else str(turn)
                    if text and len(text) > 10:
                        utterances.append({
                            "text": text.strip(),
                            "source": f"taskmaster/{ds_name}",
                            "domain": "conversation",
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
    """Load OpenAssistant conversations."""
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

    utterances = []
    for item in dataset:
        text = item.get("text", "")
        lang = item.get("lang", "en")
        if lang != "en" or not text or len(text) < 15:
            continue
        utterances.append({
            "text": text.strip(),
            "source": "oasst",
            "domain": "conversation",
        })
        if len(utterances) >= max_utt:
            break

    _save_utterances(utterances, output_file)
    print(f"OASST: {len(utterances)} utterances", flush=True)


# ── MultiWOZ 2.2 ─────────────────────────────────────────────────

def collect_multiwoz(config: dict):
    """Load MultiWOZ goal-driven dialogs."""
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
        turns = item.get("turns", {})
        turn_texts = turns.get("utterance", []) if isinstance(turns, dict) else []
        for text in turn_texts:
            if text and len(text) > 10:
                utterances.append({
                    "text": text.strip(),
                    "source": "multiwoz",
                    "domain": "conversation",
                })
            if len(utterances) >= max_utt:
                break
        if len(utterances) >= max_utt:
            break

    utterances = utterances[:max_utt]
    _save_utterances(utterances, output_file)
    print(f"MultiWOZ: {len(utterances)} utterances", flush=True)


# ── ABCD ──────────────────────────────────────────────────────────

def collect_abcd(config: dict):
    """Load ABCD customer service dialogs."""
    cfg = config["sources"]["abcd"]
    out_dir = OUTPUT_DIR / "abcd"
    out_dir.mkdir(parents=True, exist_ok=True)

    output_file = out_dir / "utterances.jsonl"
    if output_file.exists():
        print("ABCD: already collected.", flush=True)
        return

    from datasets import load_dataset

    print("  Loading ABCD...", flush=True)
    try:
        dataset = load_dataset(cfg["dataset"], split="train", trust_remote_code=True)
    except Exception:
        dataset = load_dataset(cfg["dataset"], revision="refs/convert/parquet", split="train")

    max_utt = cfg.get("max_utterances", 15000)
    utterances = []

    for item in dataset:
        convo = item.get("convo", item.get("dialog", []))
        if isinstance(convo, list):
            for turn in convo:
                text = turn.get("original", turn.get("text", "")) if isinstance(turn, dict) else str(turn)
                if text and len(text) > 10:
                    utterances.append({
                        "text": text.strip(),
                        "source": "abcd",
                        "domain": "conversation",
                    })
                if len(utterances) >= max_utt:
                    break
        if len(utterances) >= max_utt:
            break

    utterances = utterances[:max_utt]
    _save_utterances(utterances, output_file)
    print(f"ABCD: {len(utterances)} utterances", flush=True)


# ── AirDialogue ───────────────────────────────────────────────────

def collect_airdialogue(config: dict):
    """Load AirDialogue flight booking conversations."""
    cfg = config["sources"]["airdialogue"]
    out_dir = OUTPUT_DIR / "airdialogue"
    out_dir.mkdir(parents=True, exist_ok=True)

    output_file = out_dir / "utterances.jsonl"
    if output_file.exists():
        print("AirDialogue: already collected.", flush=True)
        return

    from datasets import load_dataset

    print("  Loading AirDialogue...", flush=True)
    try:
        dataset = load_dataset(cfg["dataset"], split="train", trust_remote_code=True)
    except Exception:
        dataset = load_dataset(cfg["dataset"], revision="refs/convert/parquet", split="train")

    max_utt = cfg.get("max_utterances", 15000)
    utterances = []

    for item in dataset:
        dialog = item.get("dialogue", item.get("dialog", ""))
        if isinstance(dialog, str):
            for line in dialog.split("\n"):
                line = line.strip()
                # Strip speaker prefix (e.g., "customer: " or "agent: ")
                if ": " in line:
                    line = line.split(": ", 1)[1]
                if line and len(line) > 10:
                    utterances.append({
                        "text": line,
                        "source": "airdialogue",
                        "domain": "conversation",
                    })
                if len(utterances) >= max_utt:
                    break
        if len(utterances) >= max_utt:
            break

    utterances = utterances[:max_utt]
    _save_utterances(utterances, output_file)
    print(f"AirDialogue: {len(utterances)} utterances", flush=True)


# ── Glaive Function Calling ───────────────────────────────────────

def collect_glaive(config: dict):
    """Load Glaive function calling conversations."""
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
    for item in dataset:
        # Glaive has chat format with system/user/assistant turns
        chat = item.get("chat", item.get("conversations", ""))
        if isinstance(chat, str):
            for line in chat.split("\n"):
                line = line.strip()
                if line.startswith(("USER:", "ASSISTANT:")):
                    text = line.split(":", 1)[1].strip()
                    # Skip function call JSON
                    if text and len(text) > 10 and not text.startswith(("{", "[")):
                        utterances.append({
                            "text": text,
                            "source": "glaive",
                            "domain": "conversation",
                        })
                if len(utterances) >= max_utt:
                    break
        if len(utterances) >= max_utt:
            break

    utterances = utterances[:max_utt]
    _save_utterances(utterances, output_file)
    print(f"Glaive: {len(utterances)} utterances", flush=True)


# ── Agent-FLAN ────────────────────────────────────────────────────

def collect_agent_flan(config: dict):
    """Load Agent-FLAN interaction traces."""
    cfg = config["sources"]["agent_flan"]
    out_dir = OUTPUT_DIR / "agent_flan"
    out_dir.mkdir(parents=True, exist_ok=True)

    output_file = out_dir / "utterances.jsonl"
    if output_file.exists():
        print("Agent-FLAN: already collected.", flush=True)
        return

    from datasets import load_dataset

    print("  Loading Agent-FLAN...", flush=True)
    dataset = load_dataset(cfg["dataset"], split="train")
    max_utt = cfg.get("max_utterances", 10000)

    utterances = []
    for item in dataset:
        conversations = item.get("conversations", [])
        if isinstance(conversations, list):
            for turn in conversations:
                text = turn.get("value", turn.get("content", "")) if isinstance(turn, dict) else str(turn)
                if text and len(text) > 15 and len(text) < 500 and not text.startswith(("{", "[")):
                    utterances.append({
                        "text": text.strip(),
                        "source": "agent_flan",
                        "domain": "conversation",
                    })
                if len(utterances) >= max_utt:
                    break
        if len(utterances) >= max_utt:
            break

    utterances = utterances[:max_utt]
    _save_utterances(utterances, output_file)
    print(f"Agent-FLAN: {len(utterances)} utterances", flush=True)


# ── Discord Dialogues ──────────────────────────────────────────────

def collect_discord(config: dict):
    """Load Discord casual conversations."""
    cfg = config["sources"]["discord"]
    out_dir = OUTPUT_DIR / "discord"
    out_dir.mkdir(parents=True, exist_ok=True)

    output_file = out_dir / "utterances.jsonl"
    if output_file.exists():
        print("Discord: already collected.", flush=True)
        return

    from datasets import load_dataset
    import re

    print("  Loading Discord Dialogues...", flush=True)
    dataset = load_dataset(cfg["dataset"], split="train", streaming=True)
    max_utt = cfg.get("max_utterances", 30000)

    utterances = []
    for item in dataset:
        text = item.get("text", "")
        # Parse ChatML format — extract user/assistant turns
        turns = re.split(r"<\|im_start\|>(?:user|assistant)\n?", text)
        for turn in turns:
            turn = re.sub(r"<\|im_end\|>", "", turn).strip()
            if turn and len(turn) > 10 and len(turn) < 500:
                utterances.append({
                    "text": turn,
                    "source": "discord",
                    "domain": "conversation",
                })
            if len(utterances) >= max_utt:
                break
        if len(utterances) >= max_utt:
            break

    _save_utterances(utterances, output_file)
    print(f"Discord: {len(utterances)} utterances", flush=True)


# ── Main ──────────────────────────────────────────────────────────

COLLECTORS = {
    "taskmaster": collect_taskmaster,
    "oasst": collect_oasst,
    "multiwoz": collect_multiwoz,
    "abcd": collect_abcd,
    "airdialogue": collect_airdialogue,
    "glaive": collect_glaive,
    "agent_flan": collect_agent_flan,
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
