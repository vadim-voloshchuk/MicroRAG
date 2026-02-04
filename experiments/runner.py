#!/usr/bin/env python3
"""
LLM RAG Benchmark Runner
========================

Этот скрипт реализует систему сравнения ответов LLM на базе различных методов поиска (RAG).
Модели работают на удаленном сервере.
"""

import os
import sys
import json
import time
import argparse
import datetime
import signal
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import shutil

# Подключение к ollama
import ollama

# Импорт существующих функций поиска из проекта
from rag_micro.retrievers import bm25_whoosh, faiss_dense, hybrid, hybrid_v2, splade_sparse, unicoil_sparse
from rag_micro.retrievers.reranker import get_reranker
from rag_micro.retrievers.embedders import get_embedder

# Конфигурация логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('experiment.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Константы
OLLAMA_SERVER = "http://192.168.1.102:11434"
MODELS = [
    "gpt-oss:20b",      # самая трудоёмкая
    "nemotron:70b",
    "deepseek-r1:70b",
    "llama3.3:70b",
    "command-r:35b",
    "qwen3:32b",
    "qwen3:14b",
    "mistral-nemo",
    "qwen3:4b"          # самая менее трудоёмкая
]

# Методы поиска (все доступные в проекте)
RETRIEVERS = {
    "bm25": bm25_whoosh.search_whoosh,
    "dense": faiss_dense.search_faiss,
    "hybrid": hybrid.search_hybrid,
    "hybrid_v2": hybrid_v2.search_hybrid_v2,
    "splade": splade_sparse.SpladeRetriever,
    "unicoil": unicoil_sparse.UniCOILRetriever
}

# Вопросы для тестирования
QUESTIONS = []
ANSWERS = []

# Загрузка вопросов и ответов из датасета
try:
    import pandas as pd
    df = pd.read_csv('data/benchmark/qa.csv')
    QUESTIONS = df['question'].tolist()
    ANSWERS = df['answer'].tolist()
except Exception as e:
    logger.error(f"Failed to load questions from dataset: {e}")
    # Используем заглушку, если не удалось загрузить датасет
    QUESTIONS = ["Какая максимальная частота CPU у ESP32-WROOM-32?"]
    ANSWERS = ["До 240 МГц (dual-core Xtensa LX6)"]

# Инициализация клиента ollama
client = ollama.Client(host=OLLAMA_SERVER)

@dataclass
class ExperimentConfig:
    """Конфигурация эксперимента"""
    session_dir: str
    resume: bool = False
    model: str = None
    method: str = None
    question_id: int = None

class ExperimentRunner:
    """Основной класс для запуска экспериментов"""
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.session_dir = Path(config.session_dir)
        self.retrieval_cache_path = self.session_dir / "retrieval_cache.jsonl"
        self.results_path = self.session_dir / "results.jsonl"
        self.completed_path = self.session_dir / ".completed"
        
        # Создаем директорию сессии
        self.session_dir.mkdir(parents=True, exist_ok=True)
        
        # Логика для обработки сигналов
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        # Состояние выполнения
        self.current_model = None
        self.current_method = None
        self.current_question_id = None
        self.running = True
        
        # Загружаем прогресс если есть
        self._load_progress()
    
    def _signal_handler(self, signum, frame):
        """Обработчик сигналов для безопасного завершения"""
        logger.info("Received interrupt signal, saving progress...")
        self.running = False
        sys.exit(0)
    
    def _load_progress(self):
        """Загрузка состояния выполнения из results.jsonl"""
        if self.results_path.exists():
            try:
                with open(self.results_path, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                    if lines:
                        last_line = lines[-1]
                        last_entry = json.loads(last_line)
                        self.current_model = last_entry.get('model')
                        self.current_method = last_entry.get('method')
                        self.current_question_id = last_entry.get('q_id')
                        logger.info(f"Resuming from model={self.current_model}, method={self.current_method}, question_id={self.current_question_id}")
            except Exception as e:
                logger.warning(f"Could not load progress: {e}")
    
    def _save_progress(self, model: str, method: str, q_id: int):
        """Сохранение текущего прогресса"""
        self.current_model = model
        self.current_method = method
        self.current_question_id = q_id
    
    def _is_session_completed(self) -> bool:
        """Проверяет, завершен ли эксперимент"""
        return self.completed_path.exists()
    
    def _mark_session_completed(self):
        """Отмечает сессию как завершенную"""
        self.completed_path.touch()
    
    def _get_existing_questions(self) -> set:
        """Получает список вопросов, по которым уже есть результаты"""
        existing_questions = set()
        if self.results_path.exists():
            with open(self.results_path, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        entry = json.loads(line)
                        existing_questions.add(entry['q_id'])
                    except:
                        continue
        return existing_questions
    
    def _get_existing_methods(self) -> set:
        """Получает список методов, по которым уже есть результаты"""
        existing_methods = set()
        if self.results_path.exists():
            with open(self.results_path, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        entry = json.loads(line)
                        existing_methods.add(entry['method'])
                    except:
                        continue
        return existing_methods
    
    def _get_existing_models(self) -> set:
        """Получает список моделей, по которым уже есть результаты"""
        existing_models = set()
        if self.results_path.exists():
            with open(self.results_path, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        entry = json.loads(line)
                        existing_models.add(entry['model'])
                    except:
                        continue
        return existing_models
    
    def _get_existing_retrieval_entries(self) -> dict:
        """Получает список уже существующих записей в кэше поиска"""
        existing_entries = {}
        if self.retrieval_cache_path.exists():
            with open(self.retrieval_cache_path, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        entry = json.loads(line)
                        key = (entry['q_id'], entry['method'])
                        existing_entries[key] = entry
                    except:
                        continue
        return existing_entries
    
    def run_precomputation(self):
        """Этап 1: Матрица поиска (Pre-computation)"""
        logger.info("Starting pre-computation stage...")
        
        # # Создаем директорию для индексов
        # index_dir = self.session_dir / "indexes"
        # index_dir.mkdir(exist_ok=True)
        
        # Получаем существующие записи кэша
        existing_cache_entries = self._get_existing_retrieval_entries()
        
        # Создаем список вопросов для обработки
        questions_to_process = []
        for q_id, question in enumerate(QUESTIONS):
            for method_name in RETRIEVERS.keys():
                key = (q_id, method_name)
                if key not in existing_cache_entries:
                    questions_to_process.append((q_id, question, method_name))
        
        if not questions_to_process:
            logger.info("All retrieval results already exist. Skipping pre-computation.")
            return
        
        logger.info(f"Processing {len(questions_to_process)} questions/methods combinations...")
        
        # Обработка каждого вопроса/метода
        for q_id, question, method_name in questions_to_process:
            if not self.running:
                logger.info("Pre-computation interrupted by user.")
                return
                
            logger.info(f"Processing: q_id={q_id}, method={method_name}")
            
            try:
                start_time = time.time()
                
                # Выполняем поиск в зависимости от метода
                if method_name == "bm25":
                    # Для BM25 нужно построить индекс
                    index_path = "data/index/bm25"
                    # В данном случае используем существующий индекс
                    results = bm25_whoosh.search_whoosh(index_path, question, k=5)
                elif method_name == "dense":
                    # Для dense используем существующий индекс
                    index_path = "data/index/faiss"
                    results = faiss_dense.search_faiss(index_path, question, k=5, embed_model="sentence-transformers/all-MiniLM-L6-v2")
                elif method_name == "hybrid":
                    # Для hybrid используем существующий индекс
                    results = hybrid.search_hybrid("data/index", question, k=5, embed_model="sentence-transformers/all-MiniLM-L6-v2")
                elif method_name == "hybrid_v2":
                    # Для hybrid_v2 используем существующий индекс
                    results = hybrid_v2.search_hybrid_v2("data/index", question, k=5, embed_model="sentence-transformers/all-MiniLM-L6-v2")
                elif method_name == "splade":
                    # Для splade используем существующий индекс
                    splade_retriever = splade_sparse.SpladeRetriever("data/index/splade", "naver/splade-v3")
                    results = splade_retriever.search(question, k=5)
                elif method_name == "unicoil":
                    # Для unicoil используем существующий индекс
                    unicoil_retriever = unicoil_sparse.UniCOILRetriever("data/index/unicoil", "castorini/unicoil-msmarco-passage")
                    results = unicoil_retriever.search(question, k=5)
                else:
                    logger.warning(f"Unknown method: {method_name}")
                    continue
                
                # Сохраняем результаты
                duration = time.time() - start_time
                
                for result in results:
                    # Добавляем информацию о методе и времени
                    result["timestamp"] = datetime.datetime.now().isoformat()
                    result["method"] = method_name
                    result["q_id"] = q_id
                    result["question"] = question
                    result["duration_sec"] = duration
                    
                    # Записываем в файл
                    with open(self.retrieval_cache_path, 'a', encoding='utf-8') as f:
                        f.write(json.dumps(result, ensure_ascii=False) + '\n')
                
                logger.info(f"Completed: q_id={q_id}, method={method_name}, duration={duration:.2f}s")
                
            except Exception as e:
                logger.error(f"Error processing q_id={q_id}, method={method_name}: {e}")
                continue
        
        logger.info("Pre-computation stage completed.")
    
    def run_inference(self):
        """Этап 2: Массовая генерация (Inference)"""
        logger.info("Starting inference stage...")
        
        # Проверяем, что кэш поиска готов
        if not self.retrieval_cache_path.exists():
            logger.error("Retrieval cache not found. Please run pre-computation first.")
            return
        
        # Получаем уже выполненные модели/методы/вопросы
        existing_models = self._get_existing_models()
        existing_methods = self._get_existing_methods()
        existing_questions = self._get_existing_questions()
        
        # Определяем, с какой модели начать
        start_model_idx = 0
        if self.config.model:
            try:
                start_model_idx = MODELS.index(self.config.model)
            except ValueError:
                logger.warning(f"Model {self.config.model} not found in list, starting from beginning")
        
        # Основной цикл по моделям
        for model_idx, model_name in enumerate(MODELS[start_model_idx:], start_model_idx):
            if not self.running:
                logger.info("Inference interrupted by user.")
                return
                
            # Устанавливаем keep_alive для модели
            logger.info(f"Loading model: {model_name}")
            try:
                # Установка keep_alive для модели
                client.generate(model=model_name, prompt="test", keep_alive=600)
            except Exception as e:
                logger.error(f"Failed to load model {model_name}: {e}")
                continue
            
            logger.info(f"Model {model_name} loaded successfully")
            
            # Внутренний цикл по методам
            for method_name in RETRIEVERS.keys():
                if not self.running:
                    break
                    
                # Пропускаем уже выполненные методы, если не в режиме resume
                if model_name in existing_models and method_name in existing_methods:
                    logger.info(f"Skipping already completed: model={model_name}, method={method_name}")
                    continue
                
                logger.info(f"Processing: model={model_name}, method={method_name}")
                
                # Внутренний цикл по вопросам
                for q_id, question in enumerate(QUESTIONS):
                    if not self.running:
                        break
                    
                    # Пропускаем уже выполненные вопросы, если не в режиме resume
                    if q_id in existing_questions:
                        logger.info(f"Skipping already completed question: q_id={q_id}")
                        continue
                    
                    # Получаем контекст из кэша
                    context = self._get_context_for_question_method(q_id, method_name)
                    if not context:
                        logger.warning(f"No context found for q_id={q_id}, method={method_name}")
                        continue
                    
                    # Генерация ответа
                    try:
                        start_time = time.time()
                        
                        # Формируем промт
                        prompt = f"Question: {question}\nContext: {context}\nAnswer:"
                        
                        # Запрашиваем ответ у модели
                        response = client.generate(
                            model=model_name,
                            prompt=prompt,
                            keep_alive=-1,
                            stream=False,
                            options={
                                "temperature": 0.0,
                                "top_p": 0.0,
                                "repeat_penalty": 1.0,
                                # "stop": ["\n", ".", "?", "!"]
                            }
                        )
                        
                        duration = time.time() - start_time
                        
                        # Обрабатываем ответ
                        answer = response.get('response', '').strip()
                        
                        # Сохраняем результат
                        result = {
                            "timestamp": datetime.datetime.now().isoformat(),
                            "model": model_name,
                            "method": method_name,
                            "q_id": q_id,
                            "question": question,
                            "context": context,
                            "answer": answer,
                            "duration_sec": duration
                        }
                        
                        # Записываем в файл
                        with open(self.results_path, 'a', encoding='utf-8') as f:
                            f.write(json.dumps(result, ensure_ascii=False) + '\n')
                        
                        logger.info(f"Completed: model={model_name}, method={method_name}, q_id={q_id}, duration={duration:.2f}s")
                        
                        # Сохраняем прогресс
                        self._save_progress(model_name, method_name, q_id)
                        
                    except Exception as e:
                        logger.error(f"Error generating answer for q_id={q_id}, model={model_name}, method={method_name}: {e}")
                        continue
                
                # После завершения работы с методом - сбрасываем keep_alive
                try:
                    client.generate(model=model_name, prompt="test", keep_alive="0")
                except Exception as e:
                    logger.warning(f"Failed to reset keep_alive for model {model_name}: {e}")
            
            # После завершения работы с моделью - сбрасываем keep_alive
            try:
                client.generate(model=model_name, prompt="test", keep_alive="0")
            except Exception as e:
                logger.warning(f"Failed to reset keep_alive for model {model_name}: {e}")
        
        # Отмечаем сессию как завершенную
        self._mark_session_completed()
        logger.info("Inference stage completed.")
    
    def _get_context_for_question_method(self, q_id: int, method_name: str) -> str:
        """Получает контекст для вопроса и метода из кэша"""
        if not self.retrieval_cache_path.exists():
            return ""
        
        try:
            with open(self.retrieval_cache_path, 'r', encoding='utf-8') as f:
                for line in f:
                    entry = json.loads(line)
                    if entry.get('q_id') == q_id and entry.get('method') == method_name:
                        return entry.get('text', '')
        except Exception as e:
            logger.error(f"Error reading retrieval cache: {e}")
        
        return ""
    
    def run(self):
        """Основной метод запуска эксперимента"""
        logger.info("Starting experiment runner...")
        
        # Проверяем, завершена ли сессия
        if self._is_session_completed():
            logger.info("Session is already completed.")
            return
        
        # Проверяем, что файл с вопросами существует
        if not QUESTIONS:
            logger.error("No questions found.")
            return
        
        # Запускаем этапы
        self.run_precomputation()
        self.run_inference()
        
        logger.info("Experiment completed successfully.")

def list_sessions() -> List[str]:
    """Список существующих сессий"""
    sessions = []
    for item in Path("experiments").iterdir():
        if item.is_dir() and item.name.startswith("session_"):
            sessions.append(item.name)
    return sorted(sessions, reverse=True)

def main():
    """Основная функция CLI"""
    parser = argparse.ArgumentParser(description="LLM RAG Benchmark Runner")
    parser.add_argument("--resume", action="store_true", help="Resume existing session")
    parser.add_argument("--new", action="store_true", help="Start new session")
    parser.add_argument("--model", help="Specific model to run (skip others)")
    parser.add_argument("--method", help="Specific method to run (skip others)")
    parser.add_argument("--question", type=int, help="Specific question ID to run (skip others)")
    
    args = parser.parse_args()
    
    # Создаем директорию для экспериментов
    os.makedirs("experiments", exist_ok=True)
    
    if args.resume:
        # Список существующих сессий
        sessions = list_sessions()
        if not sessions:
            logger.info("No existing sessions found. Starting new session.")
            args.new = True
        else:
            logger.info("Available sessions:")
            for i, session in enumerate(sessions, 1):
                print(f"{i}. {session}")
            
            try:
                choice = int(input("Choose session to resume (number) or 0 to start new: "))
                if choice == 0:
                    args.new = True
                elif 1 <= choice <= len(sessions):
                    session_name = sessions[choice - 1]
                    session_dir = f"experiments/{session_name}"
                    config = ExperimentConfig(
                        session_dir=session_dir,
                        resume=True,
                        model=args.model,
                        method=args.method,
                        question_id=args.question
                    )
                    runner = ExperimentRunner(config)
                    runner.run()
                    return
                else:
                    logger.error("Invalid choice")
                    return
            except ValueError:
                logger.error("Invalid input")
                return
    
    if args.new or not args.resume:
        # Создаем новую сессию
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        session_name = f"session_{timestamp}"
        session_dir = f"experiments/{session_name}"
        
        config = ExperimentConfig(
            session_dir=session_dir,
            resume=False,
            model=args.model,
            method=args.method,
            question_id=args.question
        )
        
        runner = ExperimentRunner(config)
        runner.run()

if __name__ == "__main__":
    main()