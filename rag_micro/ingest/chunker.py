from __future__ import annotations
from typing import Iterator, Dict, List
import re

def iter_pages_from_txt(txt: str) -> Iterator[Dict]:
    """
    Разбивает txt (формат из save_pages_to_txt) на страницы.
    Возвращает {"page": int, "text": str}
    """
    parts = re.split(r"\[\[\[PAGE (\d+)\]\]\]", txt)
    # parts: ['', '1', 'text...', '2', 'text...', ...]
    it = iter(parts)
    head = next(it, None)
    while True:
        num = next(it, None)
        if num is None:
            break
        body = next(it, "")
        yield {"page": int(num), "text": body.strip()}

def chunk_text(text: str, chunk_chars: int = 1000, overlap: int = 200) -> List[str]:
    text = re.sub(r"\s+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    chunks = []
    start = 0
    n = len(text)
    while start < n:
        end = min(n, start + chunk_chars)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == n:
            break
        start = max(0, end - overlap)
    return chunks
