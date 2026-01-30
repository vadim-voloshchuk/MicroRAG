from __future__ import annotations
import os
from typing import List, Dict
import fitz  # PyMuPDF
import pdfplumber
from PIL import Image
import pytesseract
from tqdm import tqdm

def extract_text_from_pdf(pdf_path: str, ocr: bool = False) -> List[Dict]:
    """
    Извлекает текст и страницы из PDF.
    Возвращает список словарей: {"page": int, "text": str}
    """
    pages = []
    # 1) базовый текст слоями
    with fitz.open(pdf_path) as doc:
        for i, page in enumerate(doc):
            text = page.get_text("text")
            pages.append({"page": i + 1, "text": text or ""})

    # 2) таблицы отдельным проходом (pdfplumber), добавляем к тексту страницы
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for i, page in enumerate(pdf.pages):
                try:
                    tables = page.extract_tables()
                    if tables:
                        tbl_texts = []
                        for t in tables:
                            rows = ["\t".join([c if c is not None else "" for c in row]) for row in t]
                            tbl_texts.append("\n".join(rows))
                        pages[i]["text"] += "\n\n[TABLE]\n" + "\n\n".join(tbl_texts)
                except Exception:
                    pass
    except Exception:
        pass

    # 3) OCR (по желанию) — только для пустых/подозрительных страниц
    if ocr:
        with fitz.open(pdf_path) as doc:
            for i, page in enumerate(doc):
                if pages[i]["text"].strip():
                    continue
                # рендер страницы в изображение и OCR
                pix = page.get_pixmap(dpi=200)
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                ocr_text = pytesseract.image_to_string(img, lang="eng")
                pages[i]["text"] = (pages[i]["text"] + "\n" + ocr_text).strip()

    return pages

def save_pages_to_txt(pages: List[Dict], out_txt_path: str):
    """
    Сохраняет страницы в один .txt с маркерами [[[PAGE N]]]
    """
    os.makedirs(os.path.dirname(out_txt_path), exist_ok=True)
    with open(out_txt_path, "w", encoding="utf-8") as f:
        for p in pages:
            f.write(f"[[[PAGE {p['page']}]]]\n{p['text'].rstrip()}\n\n")
