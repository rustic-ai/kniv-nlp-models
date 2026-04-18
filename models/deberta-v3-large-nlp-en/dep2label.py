"""dep2label: Convert dependency trees to token classification labels.

Encoding scheme (rel-pos): each token gets a label encoding the
relative position and POS of its head, plus the dependency relation.

Format: {signed_offset}@{relation}@{head_UPOS}

Examples:
  +1@nsubj@VERB   → head is the 1st VERB to the right, relation is nsubj
  -2@det@NOUN     → head is the 2nd NOUN to the left, relation is det
  0@root@ROOT     → this token is the root of the sentence

Reference:
  Strzyz et al. (2019). "Viable Dependency Parsing as Sequence Labeling."
  https://aclanthology.org/N19-1077/
"""

from __future__ import annotations


def encode_sentence(
    words: list[str],
    heads: list[int],
    deprels: list[str],
    upos_tags: list[str],
) -> list[str]:
    """Convert a dependency-annotated sentence to dep2label tags.

    Args:
        words: Token strings (1-indexed in CoNLL-U, 0-indexed here).
        heads: Head indices (0 = root, 1-indexed in CoNLL-U; 0-indexed here after conversion).
        deprels: Dependency relation labels.
        upos_tags: Universal POS tags for each token.

    Returns:
        List of dep2label tags, one per token.
    """
    labels = []
    n = len(words)

    for i in range(n):
        head_idx = heads[i]  # 0-indexed: -1 means root
        deprel = deprels[i]

        if head_idx == -1:
            # Root token
            labels.append("0@root@ROOT")
            continue

        head_upos = upos_tags[head_idx]
        offset = _compute_offset(i, head_idx, upos_tags, head_upos)
        sign = "+" if offset >= 0 else ""
        labels.append(f"{sign}{offset}@{deprel}@{head_upos}")

    return labels


def decode_sentence(
    labels: list[str],
    upos_tags: list[str],
) -> tuple[list[int], list[str]]:
    """Convert dep2label tags back to a dependency tree.

    Args:
        labels: dep2label tags, one per token.
        upos_tags: Universal POS tags (needed to resolve relative positions).

    Returns:
        (heads, deprels) where heads[i] is the 0-indexed head of token i
        (-1 for root), and deprels[i] is the dependency relation.
    """
    n = len(labels)
    heads = [-1] * n
    deprels = ["_"] * n

    for i, label in enumerate(labels):
        parts = label.split("@")
        if len(parts) != 3:
            continue

        offset_str, deprel, head_upos = parts
        offset = int(offset_str)
        deprels[i] = deprel

        if deprel == "root":
            heads[i] = -1
            continue

        head_idx = _resolve_offset(i, offset, upos_tags, head_upos)
        if head_idx is not None:
            heads[i] = head_idx

    return heads, deprels


def _compute_offset(
    token_idx: int,
    head_idx: int,
    upos_tags: list[str],
    head_upos: str,
) -> int:
    """Compute the signed offset: how many tokens of the same POS
    are between the current token and its head.

    +1 means the head is the 1st token of that POS to the right.
    -2 means the head is the 2nd token of that POS to the left.
    """
    if head_idx > token_idx:
        # Head is to the right
        count = 0
        for j in range(token_idx + 1, len(upos_tags)):
            if upos_tags[j] == head_upos:
                count += 1
            if j == head_idx:
                return count
    else:
        # Head is to the left
        count = 0
        for j in range(token_idx - 1, -1, -1):
            if upos_tags[j] == head_upos:
                count -= 1
            if j == head_idx:
                return count

    return 0  # Fallback (shouldn't happen with valid data)


def _resolve_offset(
    token_idx: int,
    offset: int,
    upos_tags: list[str],
    head_upos: str,
) -> int | None:
    """Resolve a signed offset back to an absolute head index."""
    n = len(upos_tags)

    if offset > 0:
        # Scan right
        count = 0
        for j in range(token_idx + 1, n):
            if upos_tags[j] == head_upos:
                count += 1
            if count == offset:
                return j
    elif offset < 0:
        # Scan left
        count = 0
        for j in range(token_idx - 1, -1, -1):
            if upos_tags[j] == head_upos:
                count -= 1
            if count == offset:
                return j
    else:
        # offset == 0 with non-root: shouldn't happen, but handle gracefully
        return None

    return None


def collect_label_vocabulary(conllu_path: str) -> set[str]:
    """Scan a CoNLL-U file and collect all unique dep2label tags."""
    import conllu

    labels = set()
    with open(conllu_path) as f:
        for sentence in conllu.parse(f.read()):
            # Filter to real tokens only (skip multi-word tokens and empty nodes)
            tokens = [t for t in sentence if isinstance(t["id"], int)]
            if not tokens:
                continue
            words = [t["form"] for t in tokens]
            heads = [t["head"] - 1 if t["head"] and t["head"] > 0 else -1 for t in tokens]
            deprels = [t["deprel"] or "_" for t in tokens]
            upos = [t["upos"] or "X" for t in tokens]
            encoded = encode_sentence(words, heads, deprels, upos)
            labels.update(encoded)

    return labels
