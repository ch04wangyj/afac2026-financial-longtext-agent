from agent.retrieve.doc_first import retrieve_doc_first

from .conftest import KEYWORD_BUNDLES


def test_doc_first_retrieval_returns_ranked_results(bm25_index):
    results = retrieve_doc_first(bm25_index, keyword_bundles=KEYWORD_BUNDLES, top_docs=12, top_k=10)

    assert results
    assert all(result.source == "doc_first_chunk_rerank" for result in results)
    assert results == sorted(results, key=lambda item: item.score, reverse=True)
