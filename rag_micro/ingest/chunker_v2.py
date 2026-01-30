"""
Chunker v2 - Модульная система чанкинга с поддержкой разных стратегий.

Стратегии:
- fixed_tokens: Фиксированное окно по токенам/символам
- sentence_aware: Чанки по границам предложений
- section_aware: Чанки по границам секций документа
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Iterator, Optional, Literal
import re


@dataclass
class Chunk:
    """Структура чанка с метаданными."""
    text: str
    doc_id: str
    page: Optional[int]
    section: Optional[str]
    chunk_index: int
    source_type: Literal["text", "table"] = "text"
    context_before: str = ""  # Контекст для таблиц
    start_char: int = 0
    end_char: int = 0

    def to_dict(self) -> Dict:
        return {
            "id": f"{self.doc_id}:{self.page or 0}:{self.chunk_index}",
            "doc": self.doc_id,
            "page": self.page,
            "section": self.section,
            "chunk_index": self.chunk_index,
            "text": self.text,
            "source_type": self.source_type,
            "context_before": self.context_before,
            "meta": {
                "start_char": self.start_char,
                "end_char": self.end_char,
            }
        }


class ChunkingStrategy(ABC):
    """Базовый класс для стратегий чанкинга."""

    @abstractmethod
    def chunk(self, text: str, doc_id: str, page: Optional[int] = None) -> List[Chunk]:
        """Разбивает текст на чанки."""
        pass


class FixedTokensStrategy(ChunkingStrategy):
    """
    Стратегия фиксированного окна.
    Простое скользящее окно с перекрытием.
    """

    def __init__(self, chunk_size: int = 1000, overlap: int = 200, min_chunk_size: int = 100):
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.min_chunk_size = min_chunk_size

    def chunk(self, text: str, doc_id: str, page: Optional[int] = None) -> List[Chunk]:
        # Нормализация текста
        text = re.sub(r"\s+\n", "\n", text)
        text = re.sub(r"\n{3,}", "\n\n", text)

        chunks = []
        start = 0
        n = len(text)
        chunk_idx = 0

        while start < n:
            end = min(n, start + self.chunk_size)
            chunk_text = text[start:end].strip()

            if len(chunk_text) >= self.min_chunk_size:
                chunks.append(Chunk(
                    text=chunk_text,
                    doc_id=doc_id,
                    page=page,
                    section=None,
                    chunk_index=chunk_idx,
                    start_char=start,
                    end_char=end,
                ))
                chunk_idx += 1

            if end == n:
                break
            start = max(0, end - self.overlap)

        return chunks


class SentenceAwareStrategy(ChunkingStrategy):
    """
    Стратегия чанкинга по границам предложений.
    Группирует предложения до достижения target_size, не разрывая предложения.
    """

    # Паттерны для разбиения на предложения
    SENTENCE_ENDINGS = re.compile(r'(?<=[.!?])\s+(?=[A-ZА-ЯЁ])|(?<=[.!?])$')

    def __init__(self, target_size: int = 512, overlap_sentences: int = 2, min_chunk_size: int = 100):
        self.target_size = target_size
        self.overlap_sentences = overlap_sentences
        self.min_chunk_size = min_chunk_size

    def _split_sentences(self, text: str) -> List[str]:
        """Разбивает текст на предложения."""
        # Сначала разбиваем по очевидным границам
        sentences = self.SENTENCE_ENDINGS.split(text)
        # Фильтруем пустые и слишком короткие
        sentences = [s.strip() for s in sentences if s.strip()]
        return sentences

    def chunk(self, text: str, doc_id: str, page: Optional[int] = None) -> List[Chunk]:
        sentences = self._split_sentences(text)

        if not sentences:
            return []

        chunks = []
        current_sentences = []
        current_length = 0
        chunk_idx = 0
        start_char = 0

        for i, sentence in enumerate(sentences):
            sentence_len = len(sentence)

            # Если добавление предложения превысит лимит и уже есть контент
            if current_length + sentence_len > self.target_size and current_sentences:
                chunk_text = " ".join(current_sentences)

                if len(chunk_text) >= self.min_chunk_size:
                    chunks.append(Chunk(
                        text=chunk_text,
                        doc_id=doc_id,
                        page=page,
                        section=None,
                        chunk_index=chunk_idx,
                        start_char=start_char,
                        end_char=start_char + len(chunk_text),
                    ))
                    chunk_idx += 1

                # Overlap: оставляем последние N предложений
                overlap_sents = current_sentences[-self.overlap_sentences:] if self.overlap_sentences > 0 else []
                current_sentences = overlap_sents
                current_length = sum(len(s) for s in current_sentences)
                start_char = text.find(current_sentences[0]) if current_sentences else start_char + len(chunk_text)

            current_sentences.append(sentence)
            current_length += sentence_len

        # Последний чанк
        if current_sentences:
            chunk_text = " ".join(current_sentences)
            if len(chunk_text) >= self.min_chunk_size:
                chunks.append(Chunk(
                    text=chunk_text,
                    doc_id=doc_id,
                    page=page,
                    section=None,
                    chunk_index=chunk_idx,
                    start_char=start_char,
                    end_char=start_char + len(chunk_text),
                ))

        return chunks


class SectionAwareStrategy(ChunkingStrategy):
    """
    Стратегия чанкинга по границам секций документа.
    Определяет заголовки секций и группирует контент под ними.
    """

    DEFAULT_SECTION_PATTERNS = [
        r"^(\d+\.)+\s+[A-Z]",           # 1.2.3 Section
        r"^[A-Z][A-Z\s]{5,}$",          # ALL CAPS HEADER
        r"^Chapter\s+\d+",              # Chapter N
        r"^Section\s+\d+",              # Section N
        r"^Appendix\s+[A-Z]",           # Appendix A
        r"^\d+\.\s+[A-Z][a-z]",         # 1. Something
    ]

    def __init__(
        self,
        max_section_size: int = 1024,
        section_patterns: Optional[List[str]] = None,
        fallback_strategy: Optional[ChunkingStrategy] = None
    ):
        self.max_section_size = max_section_size
        self.section_patterns = [re.compile(p, re.MULTILINE) for p in (section_patterns or self.DEFAULT_SECTION_PATTERNS)]
        self.fallback_strategy = fallback_strategy or SentenceAwareStrategy(target_size=512)

    def _is_section_header(self, line: str) -> bool:
        """Проверяет, является ли строка заголовком секции."""
        line = line.strip()
        for pattern in self.section_patterns:
            if pattern.match(line):
                return True
        return False

    def _extract_sections(self, text: str) -> List[Dict]:
        """Извлекает секции с заголовками."""
        lines = text.split('\n')
        sections = []
        current_section = {"header": None, "lines": []}

        for line in lines:
            if self._is_section_header(line):
                # Сохраняем предыдущую секцию
                if current_section["lines"]:
                    sections.append(current_section)
                current_section = {"header": line.strip(), "lines": []}
            else:
                current_section["lines"].append(line)

        # Последняя секция
        if current_section["lines"]:
            sections.append(current_section)

        return sections

    def chunk(self, text: str, doc_id: str, page: Optional[int] = None) -> List[Chunk]:
        sections = self._extract_sections(text)
        chunks = []
        chunk_idx = 0

        for section in sections:
            section_text = "\n".join(section["lines"]).strip()
            section_header = section["header"]

            if not section_text:
                continue

            # Если секция слишком большая, используем fallback
            if len(section_text) > self.max_section_size:
                sub_chunks = self.fallback_strategy.chunk(section_text, doc_id, page)
                for sub_chunk in sub_chunks:
                    sub_chunk.section = section_header
                    sub_chunk.chunk_index = chunk_idx
                    chunk_idx += 1
                    chunks.append(sub_chunk)
            else:
                chunks.append(Chunk(
                    text=section_text,
                    doc_id=doc_id,
                    page=page,
                    section=section_header,
                    chunk_index=chunk_idx,
                ))
                chunk_idx += 1

        return chunks


class TableAwareChunker:
    """
    Обработчик таблиц с добавлением контекста.
    Отвечает на вопросы 3, 16 из переписки.
    """

    TABLE_MARKER = re.compile(r'\[TABLE\]')

    def __init__(
        self,
        context_sentences: int = 2,
        max_table_size: int = 1024,
        table_format: Literal["markdown", "key_value", "raw"] = "markdown"
    ):
        self.context_sentences = context_sentences
        self.max_table_size = max_table_size
        self.table_format = table_format

    def _get_context_before(self, text: str, table_start: int) -> str:
        """Извлекает N предложений перед таблицей."""
        before_text = text[:table_start]
        sentences = re.split(r'(?<=[.!?])\s+', before_text)
        context_sentences = sentences[-self.context_sentences:] if sentences else []
        return " ".join(context_sentences)

    def _format_table_markdown(self, table_text: str) -> str:
        """Форматирует таблицу в markdown."""
        rows = table_text.strip().split('\n')
        if not rows:
            return table_text

        # Первая строка - заголовки
        formatted = []
        for i, row in enumerate(rows):
            cells = row.split('\t')
            formatted.append("| " + " | ".join(cells) + " |")
            if i == 0:
                formatted.append("|" + "|".join(["---"] * len(cells)) + "|")

        return "\n".join(formatted)

    def _format_table_key_value(self, table_text: str) -> str:
        """Форматирует таблицу как key: value пары."""
        rows = table_text.strip().split('\n')
        if len(rows) < 2:
            return table_text

        headers = rows[0].split('\t')
        formatted = []

        for row in rows[1:]:
            cells = row.split('\t')
            pairs = []
            for h, c in zip(headers, cells):
                if c.strip():
                    pairs.append(f"{h}: {c}")
            if pairs:
                formatted.append("; ".join(pairs))

        return "\n".join(formatted)

    def extract_tables_with_context(self, text: str, doc_id: str, page: Optional[int] = None) -> List[Chunk]:
        """
        Извлекает таблицы как отдельные чанки с контекстом.
        Сохраняет семантическую связь с текстом до таблицы.
        """
        chunks = []
        table_idx = 0

        # Находим все таблицы
        parts = self.TABLE_MARKER.split(text)

        if len(parts) <= 1:
            return []  # Нет таблиц

        current_pos = 0
        for i, part in enumerate(parts):
            if i == 0:
                current_pos = len(part)
                continue

            # Это содержимое после [TABLE]
            table_end = part.find('\n\n')
            if table_end == -1:
                table_text = part.strip()
            else:
                table_text = part[:table_end].strip()

            if not table_text:
                current_pos += len(part) + 7  # len('[TABLE]')
                continue

            # Получаем контекст до таблицы
            context = self._get_context_before(text, current_pos)

            # Форматируем таблицу
            if self.table_format == "markdown":
                formatted_table = self._format_table_markdown(table_text)
            elif self.table_format == "key_value":
                formatted_table = self._format_table_key_value(table_text)
            else:
                formatted_table = table_text

            # Ограничиваем размер
            if len(formatted_table) > self.max_table_size:
                formatted_table = formatted_table[:self.max_table_size] + "..."

            # Создаём чанк с контекстом
            full_text = f"{context}\n\n{formatted_table}" if context else formatted_table

            chunks.append(Chunk(
                text=full_text,
                doc_id=doc_id,
                page=page,
                section=None,
                chunk_index=table_idx,
                source_type="table",
                context_before=context,
            ))
            table_idx += 1
            current_pos += len(part) + 7

        return chunks


def get_chunking_strategy(
    strategy: Literal["fixed_tokens", "sentence_aware", "section_aware"],
    **kwargs
) -> ChunkingStrategy:
    """Фабрика для создания стратегий чанкинга."""
    strategies = {
        "fixed_tokens": FixedTokensStrategy,
        "sentence_aware": SentenceAwareStrategy,
        "section_aware": SectionAwareStrategy,
    }

    if strategy not in strategies:
        raise ValueError(f"Unknown strategy: {strategy}. Available: {list(strategies.keys())}")

    return strategies[strategy](**kwargs)


# =============================================================================
# LEGACY COMPATIBILITY
# =============================================================================

def iter_pages_from_txt(txt: str) -> Iterator[Dict]:
    """
    Разбивает txt (формат из save_pages_to_txt) на страницы.
    Возвращает {"page": int, "text": str}
    """
    parts = re.split(r"\[\[\[PAGE (\d+)\]\]\]", txt)
    it = iter(parts)
    next(it, None)  # skip head
    while True:
        num = next(it, None)
        if num is None:
            break
        body = next(it, "")
        yield {"page": int(num), "text": body.strip()}


def chunk_text(text: str, chunk_chars: int = 1000, overlap: int = 200) -> List[str]:
    """Legacy function для обратной совместимости."""
    strategy = FixedTokensStrategy(chunk_size=chunk_chars, overlap=overlap)
    chunks = strategy.chunk(text, doc_id="", page=None)
    return [c.text for c in chunks]
