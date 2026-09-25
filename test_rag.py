#!/usr/bin/env python3
"""Runnable checks for rag.py — python3 test_rag.py. No pytest, no network."""
import os

os.environ["RAG_LLM_BASE"] = "http://127.0.0.1:1/v1"  # dead port: force the fallback paths

from rag import Rag, recommend, _embed_inputs  # noqa: E402

DOCS = {
    "pw-reset": "How to reset my password. Click forgot password on the login page.",
    "invoice": "Invoice #12345 was paid on 2026-01-02. Refunds take 5 days.",
    "atlas": "Atlas batch jobs run through the data processing pipeline. Use atlas submit.",
    "pandas": "pandas read_csv loads a CSV file. merge dataframes with merge(). matplotlib plot.",
}


def _rag(**kw):
    r = Rag(use_llm=False, **kw)
    for doc_id, text in DOCS.items():
        r.add(doc_id, text)
    return r


def test_recipe1_bm25():
    r = _rag()
    hits = r.search("invoice #12345")
    assert hits and hits[0]["id"] == "invoice", hits
    assert r.search("") == [] and r.search('"quoted" (paren) [bracket]?') is not None  # parser can't be broken by input


def test_recipe2_rewrite():
    r = _rag()
    q = r.rewrite("How do I use Atlas for batch jobs?")
    assert "Atlas" in q, q  # proprietary term preserved exactly
    assert "how" not in q.lower()  # stopwords stripped
    assert r.search(q)[0]["id"] == "atlas"
    assert r.agentic_search("How do I use Atlas for batch jobs?")[0]["id"] == "atlas"


def test_recipe3_hybrid():
    r = _rag()
    hits = r.hybrid("pandas merge dataframes", k=3)
    assert hits and hits[0]["id"] == "pandas", hits
    assert len(hits) <= 3


def test_recipe6_preembed():
    r = _rag()
    assert r.preembed() == len(DOCS)
    hits = r.search_preembedded("reset password", k=2)
    assert hits and hits[0]["id"] == "pw-reset", hits


def test_recipe5_hot_cold():
    r = _rag(hot_threshold=2)
    for _ in range(3):
        r.search("invoice #12345")
    assert "invoice" in r.hot_tier()
    assert r.refresh_hot() == 1  # only the newly-hot doc gets embedded
    hits = r.search_hot_cold("invoice paid", k=3)
    assert hits and hits[0]["id"] == "invoice", hits
    assert len({h["id"] for h in hits}) == len(hits)  # merge doesn't duplicate


def test_multi_intent():
    r = _rag()
    assert len(r.decompose("read a CSV file, clean missing data, and plot the results")) >= 2
    hits = r.search_multi_intent("read a CSV file, clean missing data, and plot the results", k=3)
    ids = [h["id"] for h in hits]
    assert ids[0] == "pandas", ids
    assert len(ids) == len(set(ids))  # deduped across intents
    assert all(h["sub_query"] for h in hits)


def test_multi_intent_concurrency():
    # the parallel fan-out shares one sqlite connection: without the lock this
    # raises "bad parameter or other API misuse"
    r = Rag(use_llm=False)
    for i in range(30):
        r.add(f"d{i}", f"alpha beta gamma topic {i} pipeline data processing")
    hits = r.search_multi_intent("alpha topic, beta pipeline, gamma data, delta processing", k=5)
    ids = [h["id"] for h in hits]
    assert hits and len(ids) == len(set(ids))


def test_embed_truncation():
    # a 2048-token embedder 400s on long docs; one bad text must not sink the batch
    assert _embed_inputs(["x" * 9000], cap=4000) == ["x" * 4000]
    assert _embed_inputs(["short", "y" * 9000], cap=4000)[1] == "y" * 4000


def test_llm_down_falls_back():
    r = Rag(use_llm=True)  # RAG_LLM_BASE is a dead port
    r.add("x", "atlas batch jobs")
    assert r.rewrite("How do I use Atlas?") == "use Atlas"
    assert r.decompose("a and b") == ["a", "b"]


def test_decision_tree():
    assert recommend(has_search=False)["recipe"] == 1
    assert recommend(has_search=True, satisfied=True)["recipe"] is None
    assert recommend(has_search=True, complaint="can't find docs that exist")["recipe"] == 2
    assert recommend(has_search=True, complaint="results are okay but not great", qpd=20_000, corpus=1_000_000, ml_team=True)["recipe"] == 6
    assert recommend(has_search=True, complaint="not great", churn_pct_day=25)["recipe"] == 4
    assert recommend(has_search=True, complaint="not great", hot_patterns=True)["recipe"] == 5
    assert recommend(has_search=True, qpd=500)["recipe"] == 1
    assert recommend(has_search=True, complaint="not great", latency_ok=False)["recipe"] == 2


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print("ok", t.__name__)
    print(f"{len(tests)} checks passed")
