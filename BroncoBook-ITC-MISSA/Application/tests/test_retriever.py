from app.retriever import CorpusRetriever


def test_retriever_search_returns_results():
    retriever = CorpusRetriever("data/corpus")
    results = retriever.search("financial aid deadline", top_k=3)
    assert results
    assert any("financial" in r.title.lower() for r in results)


def test_get_page():
    retriever = CorpusRetriever("data/corpus")
    page = retriever.get_page("https://www.cpp.edu/admissions/index.shtml")
    assert page is not None
    assert page.markdown_file == "admissions_index.md"
