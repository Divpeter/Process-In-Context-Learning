from .base_retriever import BaseRetriever


def __getattr__(name):
    if name == 'EmbeddingRetriever':
        from .embedding_retriever import EmbeddingRetriever
        return EmbeddingRetriever
    if name == 'PromptingRetriever':
        from .prompting_retriever import PromptingRetriever
        return PromptingRetriever
    if name == 'BM25Retriever':
        from .bm25_retriever import BM25Retriever
        return BM25Retriever
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
