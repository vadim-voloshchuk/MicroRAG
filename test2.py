from rag_micro.retrievers.splade_sparse import SpladeRetriever, build_splade_index
from rag_micro.retrievers.unicoil_sparse import UniCOILRetriever, build_unicoil_index


# build_splade_index("data/index/corpus.jsonl", "index_test/splade", device="cuda")
# build_unicoil_index("data/index/corpus.jsonl", "index_test/unicoil", device="cuda")


retriever_splade = SpladeRetriever("index_test/splade", "naver/splade-v3", device="cuda")
retriever_unicoil = UniCOILRetriever("index_test/unicoil", "castorini/unicoil-msmarco-passage", device="cuda")

print(retriever_splade.search("Какой объём встроенной флеш-памяти у ESP32-WROOM-32?"))
print(retriever_unicoil.search("Какой объём встроенной флеш-памяти у ESP32-WROOM-32?"))
