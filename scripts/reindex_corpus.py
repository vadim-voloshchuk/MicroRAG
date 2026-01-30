#!/usr/bin/env python3
"""
Скрипт для полной переиндексации корпуса с новыми параметрами.
Использует sentence_aware chunking и мультиязычные эмбеддинги.
"""
import os
import sys
import json
import logging

# Добавляем корень проекта в путь
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    from rag_micro.ingest.build_corpus_v2 import ingest_v2, get_corpus_stats
    from rag_micro.retrievers.bm25_whoosh import build_whoosh_index
    from rag_micro.retrievers.faiss_dense import build_faiss_index
    from rag_micro.retrievers.embedders import get_embedder

    # Пути
    raw_pdf_dir = "data/raw"
    processed_txt_dir = "data/processed/text"
    index_root = "data/index"

    # Параметры чанкинга из config.yaml
    chunk_strategy = "sentence_aware"
    chunk_size = 512
    chunk_overlap = 128
    include_tables = True
    table_format = "markdown"
    table_context_sentences = 2

    # Мультиязычный эмбеддер для RU вопросов + EN даташитов
    # Используем быструю модель для демонстрации (MiniLM, 384d)
    # Для production лучше multilingual-e5-base или bge-m3
    embed_model = "multilingual-minilm"  # sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2

    logger.info("=" * 60)
    logger.info("REINDEXING CORPUS WITH NEW PARAMETERS")
    logger.info("=" * 60)
    logger.info(f"Strategy: {chunk_strategy}")
    logger.info(f"Chunk size: {chunk_size}")
    logger.info(f"Overlap: {chunk_overlap}")
    logger.info(f"Include tables: {include_tables}")
    logger.info(f"Embed model: {embed_model}")
    logger.info("=" * 60)

    # 1. Build corpus
    logger.info("\n[1/4] Building corpus from PDF and TXT...")
    corpus_path, total_chunks = ingest_v2(
        from_pdf=raw_pdf_dir,
        from_txt=processed_txt_dir,
        out_root=index_root,
        chunk_strategy=chunk_strategy,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        include_tables=include_tables,
        table_format=table_format,
        table_context_sentences=table_context_sentences,
    )
    logger.info(f"Corpus built: {corpus_path} ({total_chunks} chunks)")

    # 2. Get corpus stats
    logger.info("\n[2/4] Corpus statistics:")
    stats = get_corpus_stats(corpus_path)
    for key, value in stats.items():
        logger.info(f"  {key}: {value}")

    # 3. Build BM25 index
    logger.info("\n[3/4] Building BM25 (Whoosh) index...")
    bm25_dir = os.path.join(index_root, "bm25")
    build_whoosh_index(corpus_path, bm25_dir)
    logger.info(f"BM25 index built: {bm25_dir}")

    # 4. Build FAISS dense index
    logger.info("\n[4/4] Building FAISS dense index with multilingual embeddings...")
    faiss_dir = os.path.join(index_root, "faiss")
    embedder = get_embedder(embed_model)
    logger.info(f"Using embedder: {embedder.info.model_id} (dim={embedder.info.dim})")

    # Получаем full model_id для FAISS
    build_faiss_index(corpus_path, faiss_dir, embedder.info.model_id)
    logger.info(f"FAISS index built: {faiss_dir}")

    # Save index metadata
    metadata = {
        "chunk_strategy": chunk_strategy,
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "include_tables": include_tables,
        "embed_model": embed_model,
        "embed_model_id": embedder.info.model_id,
        "embed_dim": embedder.info.dim,
        "total_chunks": total_chunks,
        "corpus_stats": stats,
    }
    metadata_path = os.path.join(index_root, "index_metadata.json")
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    logger.info(f"\nIndex metadata saved: {metadata_path}")

    logger.info("\n" + "=" * 60)
    logger.info("REINDEXING COMPLETE!")
    logger.info("=" * 60)

    return corpus_path, total_chunks


if __name__ == "__main__":
    main()
