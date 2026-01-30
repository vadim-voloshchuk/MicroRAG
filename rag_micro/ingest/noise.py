from __future__ import annotations
import os
import hashlib
import random
from typing import Dict, List, Tuple

from .chunker_v2 import iter_pages_from_txt


_CONFUSION_MAP = {
    "0": "O",
    "1": "l",
    "2": "Z",
    "5": "S",
    "6": "G",
    "8": "B",
    "O": "0",
    "I": "1",
    "l": "1",
    "S": "5",
    "Z": "2",
    "B": "8",
    "G": "6",
}


def _stable_seed(seed: int, key: str) -> int:
    digest = hashlib.md5(key.encode("utf-8")).hexdigest()
    return seed ^ int(digest[:8], 16)


def _apply_noise(text: str, noise_level: float, rng: random.Random) -> str:
    if not text:
        return text

    chars = list(text)
    n_ops = max(1, int(len(chars) * noise_level))

    alphabet = list({c for c in text if c.isalnum()}) or list("abcdefghijklmnopqrstuvwxyz")
    ops = ["replace", "delete", "insert", "swap"]
    weights = [0.6, 0.2, 0.1, 0.1]

    for _ in range(n_ops):
        if not chars:
            break
        op = rng.choices(ops, weights=weights, k=1)[0]
        idx = rng.randrange(0, len(chars))

        if op == "replace":
            original = chars[idx]
            replacement = _CONFUSION_MAP.get(original)
            if replacement is None:
                replacement = rng.choice(alphabet)
            chars[idx] = replacement
        elif op == "delete":
            chars.pop(idx)
        elif op == "insert":
            insert_char = rng.choice(alphabet)
            chars.insert(idx, insert_char)
        elif op == "swap" and idx + 1 < len(chars):
            chars[idx], chars[idx + 1] = chars[idx + 1], chars[idx]

    return "".join(chars)


def _parse_pages(text: str) -> Tuple[List[Dict], bool]:
    pages = list(iter_pages_from_txt(text))
    if pages:
        return pages, True
    return [{"page": None, "text": text}], False


def _render_pages(pages: List[Dict], use_markers: bool) -> str:
    if not use_markers:
        return pages[0]["text"]

    parts = []
    for page in pages:
        page_num = page.get("page", 0) or 0
        parts.append(f"[[[PAGE {page_num}]]]\n{page.get('text', '').rstrip()}\n")
    return "\n".join(parts).strip() + "\n"


def generate_noisy_text(text: str, noise_level: float, seed: int, doc_id: str) -> str:
    pages, use_markers = _parse_pages(text)
    rng = random.Random(_stable_seed(seed, doc_id))

    noisy_pages = []
    for page in pages:
        noisy_text = _apply_noise(page.get("text", ""), noise_level, rng)
        noisy_pages.append({"page": page.get("page"), "text": noisy_text})

    return _render_pages(noisy_pages, use_markers)


def generate_noisy_corpus(
    clean_dir: str,
    noisy_dir: str,
    noise_level: float,
    seed: int = 42,
    overwrite: bool = False,
) -> int:
    count = 0
    clean_dir = os.path.abspath(clean_dir)
    noisy_dir = os.path.abspath(noisy_dir)
    for root, _, files in os.walk(clean_dir):
        for name in files:
            if not name.endswith(".txt"):
                continue
            rel_path = os.path.relpath(os.path.join(root, name), clean_dir)
            clean_path = os.path.join(clean_dir, rel_path)
            noisy_path = os.path.join(noisy_dir, rel_path)

            if not overwrite and os.path.exists(noisy_path):
                continue

            os.makedirs(os.path.dirname(noisy_path), exist_ok=True)
            with open(clean_path, "r", encoding="utf-8", errors="ignore") as f:
                clean_text = f.read()

            noisy_text = generate_noisy_text(clean_text, noise_level, seed, rel_path)
            with open(noisy_path, "w", encoding="utf-8") as f:
                f.write(noisy_text)
            count += 1

    return count
