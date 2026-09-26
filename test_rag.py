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


# ------------------------------------------------------------------- new: config, ingest, answers, UI


def _tmpdir():
    import tempfile

    return tempfile.mkdtemp(prefix="ragtest-")


def _stub_llm(reply):
    """Rewrite calls get keywords; answer calls get `reply`."""

    def f(messages, *a, **kw):
        return "invoice refunds" if "rewrite" in messages[0]["content"].lower() else reply

    return f


def test_config_resolution():
    import json
    import ragcli

    d = _tmpdir()
    cfg = os.path.join(d, "rag.json")
    with open(cfg, "w") as f:
        json.dump({"llm_base": "https://cfg.example", "llm_model": "cfg-model"}, f)
    c = ragcli.load_config(cfg)
    assert c["llm_base"] == "https://cfg.example" and c["llm_model"] == "cfg-model"
    ragcli.CFG.clear()
    ragcli.CFG.update(c)
    try:
        assert ragcli.resolve("llm_base", None, None, "http://d") == "https://cfg.example"
        assert ragcli.resolve("llm_base", "http://flag", None, "http://d") == "http://flag"
        os.environ["RAG_TEST_BASE"] = "http://env"
        assert ragcli.resolve("llm_base", None, "RAG_TEST_BASE", "http://d") == "http://env"
        del os.environ["RAG_TEST_BASE"]
        assert ragcli.resolve("nope", None, None, "fallback") == "fallback"
    finally:
        ragcli.CFG.clear()
    with open(cfg, "w") as f:
        f.write("{not json")
    assert ragcli.load_config(cfg) == {}  # malformed -> empty config, defaults apply, no crash


def test_cloud_auth_header():
    import rag

    assert rag._headers(None) == {"Content-Type": "application/json"}
    assert rag._headers("sk-abc")["Authorization"] == "Bearer sk-abc"


def test_pdf_missing_dep_message():
    import sys
    import ragcli

    p = os.path.join(_tmpdir(), "contract.pdf")
    with open(p, "wb") as f:
        f.write(b"%PDF-1.4\n")
    saved = sys.modules.pop("pypdf", None)
    sys.modules["pypdf"] = None  # block the import: deterministic whether or not pypdf is installed
    try:
        try:
            ragcli.extract(p)
            assert False, "should have raised"
        except ragcli.ExtractError as e:
            assert "uv add pypdf" in str(e), e
    finally:
        sys.modules.pop("pypdf", None)
        if saved is not None:
            sys.modules["pypdf"] = saved


def test_corrupt_pdf_message():
    # scanned/encrypted PDFs are normal in legal work; they must not kill the batch
    import sys
    import ragcli

    if sys.modules.get("pypdf") is None or not __import__("importlib").util.find_spec("pypdf"):
        return  # extractor not installed; the missing-dep path is covered above
    p = os.path.join(_tmpdir(), "broken.pdf")
    with open(p, "wb") as f:
        f.write(b"%PDF-1.4\nnot really a pdf")
    try:
        ragcli.extract(p)
        assert False, "should have raised"
    except ragcli.ExtractError as e:
        assert "could not read" in str(e), e


def test_add_incremental():
    r = _rag()
    assert r.add("d", "unchanged text") is True
    assert r.add("d", "unchanged text") is False  # same hash -> skipped
    assert r.add("d", "changed text") is True
    assert r.doc("d") == "changed text"


def test_walk_skips_noise():
    import ragcli

    d = _tmpdir()  # lives under a dot-directory (.hermes cache): must still index
    os.makedirs(os.path.join(d, ".git"))
    os.makedirs(os.path.join(d, "sub"))
    for p in ("a.md", ".git/x.md", "sub/b.md", "rag.db", "rag.json"):
        with open(os.path.join(d, p), "w") as f:
            f.write("x")
    got = {ragcli.doc_id(p, d) for p in ragcli.walk([d], d)}
    assert got == {"a.md", "sub/b.md"}, got


def test_answer_no_model():
    out = _rag().answer("invoice refunds", k=3)
    assert out["answer"] is None
    assert out["status"] == "no_model", out
    assert out["citations"] and out["citations"][0]["id"] == "invoice", out
    assert "no model" in out["note"].lower()  # it says so rather than pretending


def test_answer_not_found():
    import rag

    r = _rag()
    r.use_llm = True
    orig = rag.llm
    try:
        rag.llm = _stub_llm("NOT_FOUND_IN_CORPUS")
        out = r.answer("what is the capital of Peru?", k=3)
        assert out["status"] == "not_found", out
        assert out["answer"] is None  # never invents one
    finally:
        rag.llm = orig


def test_answer_ok():
    import rag

    r = _rag()
    r.use_llm = True
    orig = rag.llm
    try:
        rag.llm = _stub_llm("Refunds take 5 days [1].")
        out = r.answer("how long do refunds take?", k=3)
        assert out["status"] == "ok", out
        assert "[1]" in out["answer"]
        assert out["citations"][0]["id"] == "invoice"
    finally:
        rag.llm = orig


def test_retrieve_survives_a_mangled_rewrite():
    # FTS5 has no stemming: the rewrite can turn "refunds" into "refund", which
    # appears in no document. The raw query must still reach the exact document.
    import rag

    r = _rag()
    r.use_llm = True
    orig = rag.llm
    try:
        rag.llm = lambda *a, **kw: "refund processing time"
        ids = [h["id"] for h in r.retrieve("how long do refunds take?", k=5)]
        assert "invoice" in ids, ids
    finally:
        rag.llm = orig


def test_serve_ask():
    import json
    import threading
    import urllib.error
    import urllib.request

    import ragcli

    srv = ragcli.make_server(_rag(), "127.0.0.1", 0)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    try:
        host, port = srv.server_address[:2]
        base = f"http://{host}:{port}"
        with urllib.request.urlopen(base + "/api/ask?q=invoice+refunds&k=3", timeout=5) as resp:
            out = json.load(resp)
        assert out["status"] == "no_model", out
        assert out["citations"][0]["id"] == "invoice"
        with urllib.request.urlopen(base + "/", timeout=5) as resp:
            assert b"<html" in resp.read().lower()
        try:
            urllib.request.urlopen(base + "/api/ask", timeout=5)  # no q
            assert False, "missing q should be 400"
        except urllib.error.HTTPError as e:
            assert e.code == 400
    finally:
        srv.shutdown()
        t.join(timeout=5)


def test_cli_add_search_json():
    import ragcli

    d = _tmpdir()
    docs = os.path.join(d, "docs")
    os.makedirs(docs)
    with open(os.path.join(docs, "pw.md"), "w") as f:
        f.write("How to reset my password. Click forgot password.")
    cfg = os.path.join(d, "rag.json")
    db = os.path.join(d, "t.db")
    assert ragcli.main(["--config", cfg, "--db", db, "add", docs]) == 0
    assert ragcli.main(["--config", cfg, "--db", db, "add", docs]) == 0  # unchanged -> skipped
    assert ragcli.main(["--config", cfg, "--db", db, "--json", "search", "reset password"]) == 0


def test_cli_empty_corpus_hint():
    import ragcli

    d = _tmpdir()
    assert ragcli.main(["--db", os.path.join(d, "empty.db"), "--config", os.path.join(d, "c.json"), "search", "x"]) == 3


def test_cli_doctor_dead_endpoint():
    import contextlib
    import io

    import ragcli

    cfg = os.path.join(_tmpdir(), "rag.json")
    with open(cfg, "w") as f:
        f.write('{"llm_base": "http://127.0.0.1:1"}')
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        code = ragcli.main(["--config", cfg, "doctor"])
    assert code != 0  # reports unreachable, no traceback
    assert "unreachable" in buf.getvalue().lower()


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print("ok", t.__name__)
    print(f"{len(tests)} checks passed")
