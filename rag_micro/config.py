from pydantic import BaseModel
from typing import Optional
from dotenv import load_dotenv
import os

load_dotenv()

class Settings(BaseModel):
    index_root: str = "data/index"
    chunk_chars: int = 1000
    chunk_overlap: int = 200
    embed_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    bm25_index_dir: str = "data/index/bm25"
    faiss_index_dir: str = "data/index/faiss"
    store_jsonl: str = "data/index/corpus.jsonl"

    # LLM
    llm_backend: str = os.getenv("LLM_BACKEND", "stub")  # stub | llama_cpp | openai
    # llama.cpp
    llama_model_path: Optional[str] = os.getenv("LLAMA_MODEL_PATH")
    llama_ctx: int = int(os.getenv("LLAMA_CTX", "4096"))
    llama_threads: int = int(os.getenv("LLAMA_THREADS", "8"))

    # OpenAI-compatible
    openai_api_key: Optional[str] = os.getenv("OPENAI_API_KEY")
    openai_base_url: Optional[str] = os.getenv("OPENAI_BASE_URL")
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    ollama_host: str = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")
    ollama_model: str = os.getenv("OLLAMA_MODEL", "llama3.1:8b-instruct")

settings = Settings()
