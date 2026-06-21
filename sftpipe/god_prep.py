"""Data-prep helpers vendored & generalized from gradients-ai/G.O.D
`validator/tasks/task_prep.py`.

GOD operates on its own task pydantic types; here the same logic is decoupled
to operate on our CanonicalRecord / config so this repo stands alone. Behaviour
(seed=42 split, chat-row cleaning, empty-row guard) is preserved so output is
identical to what the trainer already expects.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from sftpipe.config import ChatCfg

# ---------------------------------------------------------------------------
# chat-row cleaning — verbatim logic from GOD process_chat_row
# ---------------------------------------------------------------------------

def process_chat_row(value, role_field: str, content_field: str):
    if isinstance(value, str) and value.strip().startswith("[") and value.strip().endswith("]"):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            pass

    if isinstance(value, list):
        cleaned = []
        for msg in value:
            if isinstance(msg, dict) and msg.get(content_field) is not None and msg.get(role_field) is not None:
                cleaned.append(msg)
        return cleaned if len(cleaned) > 1 else []
    return value if value is not None else ""


# ---------------------------------------------------------------------------
# empty-row guard — from GOD change_to_json_format (chat branch)
# ---------------------------------------------------------------------------

def guard_not_mostly_empty(conversations_rows: list[dict], chat: ChatCfg, max_empty_frac: float = 0.8) -> None:
    total = len(conversations_rows)
    if total == 0:
        raise ValueError("No rows produced")
    empty = sum(1 for r in conversations_rows if not r.get(chat.column))
    if empty / total > max_empty_frac:
        raise ValueError(f"More than {max_empty_frac:.0%} of rows are empty ({empty}/{total})")


# ---------------------------------------------------------------------------
# seeded train/test split — matches GOD train_test_split (seed=42)
# ---------------------------------------------------------------------------

def train_test_split(rows: list[dict], test_size: float, seed: int = 42) -> tuple[list[dict], list[dict]]:
    """Dedup-aware split. P7 upsamples (repeats records) to hit token budgets, so
    a naive index split leaks identical rows across train/test. We key on full row
    content: every copy of a record goes to the SAME side, test holds one copy of
    each selected record, and train keeps the upsampled copies. Diverges from GOD's
    naive split deliberately — a leak-free held-out set matters more than parity."""
    import random

    if not rows:
        return [], []
    keys = [json.dumps(r, sort_keys=True, ensure_ascii=False) for r in rows]
    unique = list(dict.fromkeys(keys))  # order-preserving dedup
    rng = random.Random(seed)
    rng.shuffle(unique)
    n_test = max(1, int(len(unique) * test_size))
    test_keys = set(unique[:n_test])

    train, test, seen_test = [], [], set()
    for row, key in zip(rows, keys):
        if key in test_keys:
            if key not in seen_test:  # one copy per record in test
                test.append(row)
                seen_test.add(key)
        else:
            train.append(row)  # keep upsampled duplicates in train
    return train, test


# ---------------------------------------------------------------------------
# jsonl io
# ---------------------------------------------------------------------------

def write_jsonl(rows: Iterable[dict], path: str | Path) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    return n


def read_jsonl(path: str | Path) -> list[dict]:
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows
