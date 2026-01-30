"""
Build Corpus v2 - Построение корпуса с поддержкой:
- Разных стратегий чанкинга
- Извлечения таблиц с контекстом
- Метаданных source_type (text/table)
"""
from __future__ import annotations
import os
import json
import glob
from typing import List, Dict, Optional, Literal
from pathlib import Path
from tqdm import tqdm
import logging

from .pdf_to_text import extract_text_from_pdf, save_pages_to_txt
from .chunker_v2 import (
    get_chunking_strategy,
    iter_pages_from_txt,
    TableAwareChunker,
    Chunk,
)

logger = logging.getLogger(__name__)


def ingest_v2(
    from_pdf: Optional[str],
    from_txt: Optional[str],
    out_root: str,
    chunk_strategy: Literal["fixed_tokens", "sentence_aware", "section_aware"] = "sentence_aware",
    chunk_size: int = 512,
    chunk_overlap: int = 128,
    include_tables: bool = True,
    table_format: Literal["markdown", "key_value", "raw"] = "markdown",
    table_context_sentences: int = 2,
) -> tuple[str, int]:
    """
    Строит JSONL корпус из PDF и/или TXT.

    Args:
        from_pdf: Директория с PDF файлами
        from_txt: Директория с TXT файлами
        out_root: Корневая директория для выходных данных
        chunk_strategy: Стратегия чанкинга
        chunk_size: Размер чанка
        chunk_overlap: Перекрытие
        include_tables: Включать ли таблицы как отдельные чанки
        table_format: Формат таблиц
        table_context_sentences: Сколько предложений контекста для таблиц

    Returns:
        Tuple[str, int]: Путь к corpus.jsonl и количество записей
    """
    os.makedirs(out_root, exist_ok=True)
    out_jsonl = os.path.join(out_root, "corpus.jsonl")

    # Инициализируем стратегию чанкинга с правильными параметрами для каждой стратегии
    if chunk_strategy == "fixed_tokens":
        chunker = get_chunking_strategy(
            chunk_strategy,
            chunk_size=chunk_size,
            overlap=chunk_overlap,
            min_chunk_size=min_chunk_size if 'min_chunk_size' in dir() else 100,
        )
    elif chunk_strategy == "sentence_aware":
        chunker = get_chunking_strategy(
            chunk_strategy,
            target_size=chunk_size,
            overlap_sentences=max(1, chunk_overlap // 50),
            min_chunk_size=100,
        )
    elif chunk_strategy == "section_aware":
        chunker = get_chunking_strategy(
            chunk_strategy,
            max_section_size=chunk_size,
        )
    else:
        raise ValueError(f"Unknown chunk strategy: {chunk_strategy}")

    # Инициализируем обработчик таблиц
    table_chunker = TableAwareChunker(
        context_sentences=table_context_sentences,
        table_format=table_format,
    ) if include_tables else None

    # Собираем все .txt пути
    txt_paths: List[str] = []

    # 1) PDF → TXT
    if from_pdf and os.path.isdir(from_pdf):
        pdfs = sorted(glob.glob(os.path.join(from_pdf, "*.pdf")))
        txt_dir = os.path.join(out_root, "txt_from_pdf")
        os.makedirs(txt_dir, exist_ok=True)

        for pp in tqdm(pdfs, desc="PDF->TXT"):
            base = os.path.splitext(os.path.basename(pp))[0]
            out_txt = os.path.join(txt_dir, base + ".txt")

            if not os.path.exists(out_txt):
                try:
                    pages = extract_text_from_pdf(pp, ocr=False)
                    save_pages_to_txt(pages, out_txt)
                except Exception as e:
                    logger.error(f"Failed to process {pp}: {e}")
                    continue

            txt_paths.append(out_txt)

    # 2) Существующие TXT
    if from_txt and os.path.isdir(from_txt):
        txts = sorted(glob.glob(os.path.join(from_txt, "*.txt")))
        txt_paths.extend(txts)

    # Убираем дубликаты
    txt_paths = list(dict.fromkeys(txt_paths))

    # 3) Чанкинг и запись JSONL
    total_chunks = 0
    total_tables = 0

    with open(out_jsonl, "w", encoding="utf-8") as fout:
        for txt_path in tqdm(txt_paths, desc="Chunking"):
            doc_name = os.path.splitext(os.path.basename(txt_path))[0]

            with open(txt_path, "r", encoding="utf-8", errors="ignore") as f:
                raw = f.read()

            # Парсим страницы
            pages = list(iter_pages_from_txt(raw))
            if not pages:
                pages = [{"page": None, "text": raw}]

            for page_info in pages:
                page_num = page_info["page"]
                page_text = page_info["text"]

                # 1) Чанки текста
                text_chunks = chunker.chunk(page_text, doc_id=doc_name, page=page_num)

                for chunk in text_chunks:
                    rec = chunk.to_dict()
                    rec["meta"]["chunk_strategy"] = chunk_strategy
                    rec["meta"]["chunk_size"] = chunk_size
                    rec["meta"]["source"] = txt_path
                    fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    total_chunks += 1

                # 2) Чанки таблиц (если включены)
                if table_chunker and "[TABLE]" in page_text:
                    table_chunks = table_chunker.extract_tables_with_context(
                        page_text, doc_id=doc_name, page=page_num
                    )

                    for chunk in table_chunks:
                        # Переназначаем ID для таблиц
                        chunk.chunk_index = total_chunks + total_tables
                        rec = chunk.to_dict()
                        rec["id"] = f"{doc_name}:{page_num or 0}:table_{total_tables}"
                        rec["meta"]["table_format"] = table_format
                        rec["meta"]["source"] = txt_path
                        fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                        total_tables += 1

    logger.info(f"Corpus built: {total_chunks} text chunks, {total_tables} table chunks")
    return out_jsonl, total_chunks + total_tables


def build_split_corpus(
    from_pdf: Optional[str],
    from_txt: Optional[str],
    out_root: str,
    include_tables: bool = True,
    **kwargs
) -> Dict[str, str]:
    """
    Строит раздельные корпуса для text-only и text+tables.

    Полезно для абляции табличного вклада (пункт 16).

    Returns:
        Dict с путями: {"text_only": path, "text_tables": path}
    """
    paths = {}

    # Text-only corpus
    text_only_root = os.path.join(out_root, "text_only")
    path, _ = ingest_v2(
        from_pdf, from_txt, text_only_root,
        include_tables=False, **kwargs
    )
    paths["text_only"] = path

    # Text + tables corpus
    if include_tables:
        text_tables_root = os.path.join(out_root, "text_tables")
        path, _ = ingest_v2(
            from_pdf, from_txt, text_tables_root,
            include_tables=True, **kwargs
        )
        paths["text_tables"] = path

    return paths


# =============================================================================
# CORPUS STATISTICS
# =============================================================================

def get_corpus_stats(corpus_jsonl: str) -> Dict:
    """Возвращает статистику корпуса."""
    stats = {
        "total_chunks": 0,
        "text_chunks": 0,
        "table_chunks": 0,
        "documents": set(),
        "pages": set(),
        "avg_chunk_length": 0,
        "min_chunk_length": float("inf"),
        "max_chunk_length": 0,
    }

    chunk_lengths = []

    with open(corpus_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            stats["total_chunks"] += 1
            stats["documents"].add(rec.get("doc", ""))
            stats["pages"].add((rec.get("doc"), rec.get("page")))

            source_type = rec.get("source_type", "text")
            if source_type == "table":
                stats["table_chunks"] += 1
            else:
                stats["text_chunks"] += 1

            text_len = len(rec.get("text", ""))
            chunk_lengths.append(text_len)
            stats["min_chunk_length"] = min(stats["min_chunk_length"], text_len)
            stats["max_chunk_length"] = max(stats["max_chunk_length"], text_len)

    stats["documents"] = len(stats["documents"])
    stats["pages"] = len(stats["pages"])
    stats["avg_chunk_length"] = sum(chunk_lengths) / len(chunk_lengths) if chunk_lengths else 0

    return stats


# =============================================================================
# LEGACY COMPATIBILITY
# =============================================================================

def ingest(from_pdf, from_txt, out_root, chunk_chars=1000, overlap=200):
    """Legacy функция для обратной совместимости."""
    return ingest_v2(
        from_pdf, from_txt, out_root,
        chunk_strategy="fixed_tokens",
        chunk_size=chunk_chars,
        chunk_overlap=overlap,
        include_tables=True,
    )
