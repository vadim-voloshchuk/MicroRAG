from __future__ import annotations
import os, json, glob
from typing import List, Dict, Optional
from tqdm import tqdm

from .pdf_to_text import extract_text_from_pdf, save_pages_to_txt
from .chunker import iter_pages_from_txt, chunk_text

def ingest(from_pdf: Optional[str], from_txt: Optional[str], out_root: str,
           chunk_chars: int = 1000, overlap: int = 200):
    """
    Строит JSONL корпус из PDF и/или TXT.
    JSONL элементы:
      { "id": str, "doc": str, "page": int|None, "chunk_index": int, "text": str, "meta": {...} }
    """
    os.makedirs(out_root, exist_ok=True)
    out_jsonl = os.path.join(out_root, "corpus.jsonl")

    # Собираем все .txt (из папки txt и из распарсенных pdf)
    txt_paths: List[str] = []

    # 1) PDF → TXT временно рядом
    if from_pdf and os.path.isdir(from_pdf):
        pdfs = sorted(glob.glob(os.path.join(from_pdf, "*.pdf")))
        for pp in tqdm(pdfs, desc="PDF->TXT"):
            base = os.path.splitext(os.path.basename(pp))[0]
            out_txt = os.path.join(out_root, "txt_from_pdf", base + ".txt")
            if not os.path.exists(out_txt):
                pages = extract_text_from_pdf(pp, ocr=False)
                save_pages_to_txt(pages, out_txt)
            txt_paths.append(out_txt)

    # 2) Имеющиеся TXT
    if from_txt and os.path.isdir(from_txt):
        txts = sorted(glob.glob(os.path.join(from_txt, "*.txt")))
        txt_paths.extend(txts)

    # 3) Чанкинг и запись JSONL
    counter = 0
    with open(out_jsonl, "w", encoding="utf-8") as fout:
        for txt_path in tqdm(txt_paths, desc="Chunking"):
            doc_name = os.path.splitext(os.path.basename(txt_path))[0]
            with open(txt_path, "r", encoding="utf-8", errors="ignore") as f:
                raw = f.read()
            # попытка страниц
            pages = list(iter_pages_from_txt(raw))
            if not pages:
                pages = [{"page": None, "text": raw}]

            for p in pages:
                chs = chunk_text(p["text"], chunk_chars, overlap)
                for ci, ch in enumerate(chs):
                    rec = {
                        "id": f"{doc_name}:{p['page'] or 0}:{ci}",
                        "doc": doc_name,
                        "page": p["page"],
                        "chunk_index": ci,
                        "text": ch,
                        "meta": {
                            "source": txt_path,
                        }
                    }
                    fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    counter += 1
    return out_jsonl, counter
