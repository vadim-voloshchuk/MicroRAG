#!/usr/bin/env python3
"""
Анализ результатов эксперимента: конвертация results.jsonl -> .csv
"""

import json
import pandas as pd
import sys
import os
from pathlib import Path

def convert_results_to_csv(input_file: str, output_file: str):
    """
    Конвертирует results.jsonl в CSV файл
    
    Args:
        input_file (str): Путь к входному файлу results.jsonl
        output_file (str): Путь к выходному файлу results.csv
    """
    results = []
    
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    try:
                        entry = json.loads(line)
                        results.append(entry)
                    except json.JSONDecodeError as e:
                        print(f"Ошибка при чтении строки: {e}")
                        continue
        
        # Создаем DataFrame
        df = pd.DataFrame(results)
        
        # Сохраняем в CSV
        df.to_csv(output_file, index=False, encoding='utf-8')
        print(f"Успешно конвертировано {len(results)} записей в {output_file}")
        
    except FileNotFoundError:
        print(f"Файл {input_file} не найден")
        sys.exit(1)
    except Exception as e:
        print(f"Ошибка при конвертации: {e}")
        sys.exit(1)

def main():
    """Основная функция"""
    # Проверяем аргументы
    if len(sys.argv) < 2:
        print("Использование: python analysis.py <results.jsonl> [output.csv]")
        print("Пример: python analysis.py experiments/session_20260202_123456/results.jsonl")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else input_file.replace('.jsonl', '.csv')
    
    # Проверяем существование входного файла
    if not os.path.exists(input_file):
        print(f"Файл {input_file} не существует")
        sys.exit(1)
    
    convert_results_to_csv(input_file, output_file)

if __name__ == "__main__":
    main()