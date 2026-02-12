import os
import json
import time
import argparse
import datetime
import pandas as pd
from pathlib import Path
from ollama import Client
from tqdm import tqdm

# --- КОНФИГУРАЦИЯ ---
OLLAMA_URL = "http://192.168.1.102:11434"
# OLLAMA_URL = "http://localhost:11434"
QUESTIONS_FILE = Path("data/benchmark/qa.csv")
RETRIEVAL_LOG = Path("data/retrieval_cache.jsonl") 
BASE_EXP_DIR = Path("experiments")

# MODELS = ["qwen3:32b", "qwen3:14b", "mistral-nemo", "qwen3:4b"]
# MODELS = [
#     "llama3.3:70b",    # 1. Точка отсчета (Baseline)
#     "deepseek-r1:70b", # 2. Современный Reasoning (Логика)
#     "gpt-oss:20b",     # 3. Твой интерес (Главный челленджер среднего веса)
#     "command-r:35b",   # 4. Профильный RAG-специалист
#     "qwen3:4b",        # 5. Нижний порог (для контраста)
#     "mistral-nemo",    # 6. Оптимальный баланс (12b)
#     "nemotron:70b",    # 7. Альтернативный топ от NVIDIA
#     "qwen3:32b",       # 8. Мощный Qwen среднего размера
#     "qwen3:14b"        # 9. Замыкающая модель
# ]
MODELS = ["qwen3:4b"]
METHODS = ["bm25", "dense", "hybrid", "hybrid_v2", "splade", "unicoil"]

def load_search_data(filepath):
    """Группирует кандидатов поиска (5 штук) в единый контекст."""
    if not filepath.exists():
        return {}
    data = {}
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                item = json.loads(line)
                q_id, method, text = str(item['q_id']), item['method'], item['text'].strip()
                if q_id not in data: data[q_id] = {}
                if method not in data[q_id]: data[q_id][method] = []
                data[q_id][method].append(text)
            except: continue
    
    final_contexts = {}
    for q_id, methods in data.items():
        final_contexts[q_id] = {}
        for method, chunks in methods.items():
            combined = "".join([f"--- Фрагмент {i+1} ---\n{c}\n\n" for i, c in enumerate(chunks)])
            final_contexts[q_id][method] = combined.strip()
    return final_contexts

class ExperimentManager:
    def __init__(self, session_path: Path):
        self.session_path = session_path
        self.results_file = session_path / "final_results.jsonl"
        self.config_file = session_path / "config.json"
        self.complete_marker = session_path / "COMPLETED"
        self.session_path.mkdir(parents=True, exist_ok=True)

    def is_completed(self):
        return self.complete_marker.exists()

    def mark_as_completed(self):
        self.complete_marker.touch()

    def save_config(self, config_data):
        with open(self.config_file, 'w', encoding='utf-8') as f:
            json.dump(config_data, f, ensure_ascii=False, indent=2)

    def get_progress_info(self, total_expected):
        done_count = 0
        if self.results_file.exists():
            with open(self.results_file, 'r', encoding='utf-8') as f:
                done_count = sum(1 for _ in f)
        return done_count, total_expected

    def get_done_keys(self):
        done = set()
        if self.results_file.exists():
            with open(self.results_file, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        d = json.loads(line)
                        done.add(f"{d['model']}_{d['method']}_{d['q_id']}")
                    except: continue
        return done

def run_experiment(manager: ExperimentManager):
    df = pd.read_csv(QUESTIONS_FILE)
    questions = df['question'].tolist()
    search_data = load_search_data(RETRIEVAL_LOG)
    client = Client(host=OLLAMA_URL)
    
    done_keys = manager.get_done_keys()
    
    try:
        pbar_models = tqdm(MODELS, desc="📊 Модели")
        for model in pbar_models:
            pbar_models.set_description(f"📊 Модель: {model}")
            
            pbar_methods = tqdm(METHODS, desc="🔍 Методы", leave=False)
            for method in pbar_methods:
                pbar_methods.set_description(f"🔍 Метод: {method}")
                
                pbar_questions = tqdm(range(len(questions)), desc="❓ Вопросы", leave=False)
                for q_id in pbar_questions:
                    key = f"{model}_{method}_{q_id}"
                    if key in done_keys:
                        continue

                    context = search_data.get(str(q_id), {}).get(method, "")
                    if not context: continue

                    prompt = f"Контекст:\n{context}\n\nВопрос: {questions[q_id]}\n\nОтвет:"
                    
                    try:
                        start_t = time.time()
                        response = client.generate(model=model, prompt=prompt, options={"temperature": 0.0}, keep_alive="15m")
                        duration = time.time() - start_t
                        
                        result = {
                            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "model": model, "method": method, "q_id": q_id,
                            "question": questions[q_id], "answer": response['response'].strip(),
                            "duration_sec": round(duration, 2)
                        }
                        
                        with open(manager.results_file, 'a', encoding='utf-8') as f:
                            f.write(json.dumps(result, ensure_ascii=False) + "\n")
                            
                    except Exception as e:
                        tqdm.write(f"❌ Ошибка {model} | Q{q_id}: {e}")
                        time.sleep(5)
        
        # Если все циклы завершены без прерывания
        manager.mark_as_completed()
        print(f"\n✅ Эксперимент полностью завершен!")

    except KeyboardInterrupt:
        print("\n🛑 Эксперимент прерван пользователем. Прогресс сохранен.")

def main():
    parser = argparse.ArgumentParser(description="RAG Benchmark Inference Runner")
    parser.add_argument("--new", action="store_true", help="Начать новую сессию")
    parser.add_argument("--resume", action="store_true", help="Продолжить существующую сессию")
    args = parser.parse_args()

    BASE_EXP_DIR.mkdir(exist_ok=True)
    df = pd.read_csv(QUESTIONS_FILE)
    total_expected = len(MODELS) * len(METHODS) * len(df)

    if args.resume:
        sessions = sorted([d for d in BASE_EXP_DIR.iterdir() if d.is_dir()], key=lambda x: x.stat().st_mtime, reverse=True)
        if not sessions:
            print("❌ Нет доступных сессий.")
            return
        
        print(f"\n{'№':<3} | {'Статус':<12} | {'Прогресс':<10} | {'Имя сессии'}")
        print("-" * 60)
        
        session_objects = []
        for i, s_path in enumerate(sessions):
            m = ExperimentManager(s_path)
            done, total = m.get_progress_info(total_expected)
            status = "✅ Готово" if m.is_completed() else "⏳ В процессе"
            prog_str = f"{done}/{total}"
            print(f"{i+1:<3} | {status:<12} | {prog_str:<10} | {s_path.name}")
            session_objects.append(m)
        
        try:
            idx = int(input("\nВыберите номер сессии для продолжения (или 0 для отмены): ")) - 1
            if idx == -1: return
            manager = session_objects[idx]
        except (ValueError, IndexError):
            print("❌ Неверный выбор.")
            return
    else:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        manager = ExperimentManager(BASE_EXP_DIR / f"session_{timestamp}")
        manager.save_config({
            "models": MODELS, "methods": METHODS, "ollama_url": OLLAMA_URL,
            "questions_file": str(QUESTIONS_FILE), "retrieval_log": str(RETRIEVAL_LOG)
        })
        print(f"🆕 Новая сессия: {manager.session_path.name}")

    run_experiment(manager)

if __name__ == "__main__":
    main()