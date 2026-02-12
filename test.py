from rag_micro.retrievers.bm25_whoosh import search_whoosh, build_whoosh_index

asnwer = "Какой объём встроенной флеш-памяти у ESP32-WROOM-32?"
index = "index_test/bm25"

print(search_whoosh(index, asnwer))

# # build_whoosh_index("data/index/corpus.jsonl", "index_test/bm25")


# from whoosh import index

# ix = index.open_dir("index_test/bm25")
# with ix.searcher() as s:
#     print(f"Всего документов в индексе: {s.doc_count_all()}")
#     # Выведем пример одного документа, чтобы увидеть, что там внутри
#     for doc in s.all_stored_fields():
#         print(doc)
#         break # печатаем только один для теста