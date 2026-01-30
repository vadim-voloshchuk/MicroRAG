#!/usr/bin/env bash
set -e
rag-cli ingest --from-pdf data/raw --from-txt data/processed/text --out data/index
rag-cli index --build --index-root data/index --faiss --bm25
echo "Ready. Ask a question:"
rag-cli ask "What is the minimum VDD of STM32F103?" --mode hybrid --k 5 --index-root data/index
