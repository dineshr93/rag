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


def test_endpoint_failure_warns_once():
    import contextlib
    import io
    import rag

    rag._warned = False
    err = io.StringIO()
    try:
        with contextlib.redirect_stderr(err):
            assert rag.llm([{"role": "user", "content": "hi"}]) is None  # dead port
            assert rag.llm([{"role": "user", "content": "hi"}]) is None
    finally:
        rag._warned = False
    lines = [l for l in err.getvalue().splitlines() if "warning:" in l]
    assert len(lines) == 1 and "falling back" in lines[0], err.getvalue()


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


def test_answer_sends_the_matching_window_not_the_head():
    """A whole document is longer than a model can take, and the first screenful of a
    contract is its cover page. The answer must be built from the region that matches
    the question, and the cut must be visible rather than silent."""
    import rag

    r = _rag()
    filler = "VERMIETERVEREIN BOILERPLATE PARA. "
    r.add("lease", filler * 300 + "Die Miete betraegt 900 EUR netto pro Monat. " + filler * 300)
    assert r.doc("lease").index("900 EUR") > 2000  # deep in the doc, past any head cut
    r.use_llm = True
    seen = {}

    def spy(messages, *a, **kw):
        seen["sources"] = messages[1]["content"]
        return "Die Miete betraegt 900 EUR netto pro Monat [1]."

    orig = rag.llm
    rag.CFG["answer_max_chars"] = "2000"  # the default cap is now dynamic; windowing is the override path
    try:
        rag.llm = spy
        out = r.answer("wie hoch ist die Miete 900 EUR", k=1)
    finally:
        rag.llm = orig
        rag.CFG.pop("answer_max_chars", None)
    assert out["status"] == "ok", out
    assert "900 EUR" in seen["sources"], seen["sources"][:400]
    assert len(seen["sources"]) < 2400  # a window, not the whole ~21k-char document
    c = out["citations"][0]
    assert c["chars"] == len(r.doc("lease"))
    assert c["sent"] <= rag._answer_budget() < c["chars"], c
    assert [t["id"] for t in out["truncated"]] == ["lease"], out["truncated"]


def test_answer_budget_default_grows_with_the_corpus():
    """No config key to hunt for: the biggest document fits whole by default."""
    import rag

    r = _rag()
    rag.CFG.pop("answer_max_chars", None)
    big = "x" * 5000
    r.add("big", big)
    assert rag._answer_budget(r.db) == 5000
    assert rag._answer_budget() == 2000  # no connection: the floor


def test_answer_short_doc_is_not_truncated():
    """A document that fits is sent whole: nothing is reported as cut, and the
    not_found note can then honestly say the corpus holds no answer."""
    import rag

    r = _rag()
    r.use_llm = True
    orig = rag.llm
    try:
        rag.llm = _stub_llm("NOT_FOUND_IN_CORPUS")
        out = r.answer("invoice refund policy", k=1)
    finally:
        rag.llm = orig
    assert out["status"] == "not_found", out
    assert out["truncated"] == [], out["truncated"]
    assert out["citations"][0]["sent"] == out["citations"][0]["chars"]
    assert out["note"] == "the corpus does not contain an answer", out["note"]


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


def test_image_added_with_vision():
    """A vision-capable model turns a photo into indexed, searchable text."""
    import rag
    import ragcli

    d = _tmpdir()
    photos = os.path.join(d, "photos")
    os.makedirs(photos)
    with open(os.path.join(photos, "living-room.jpg"), "wb") as f:
        f.write(b"\xff\xd8\xff\x00")  # JPEG SOI header; the bytes content is
                                      # irrelevant — the LLM (stubbed) reads it
    cfg = os.path.join(d, "rag.json")
    db = os.path.join(d, "t.db")
    orig = rag.llm
    try:
        rag.llm = lambda *a, **kw: "Living room with a large window and a garden view."
        assert ragcli.main(["--corpus", d, "--config", cfg, "--db", db,
                            "add", photos]) == 0
    finally:
        rag.llm = orig
    r = Rag(db=db)
    ids = r.doc_ids()
    assert "photos/living-room.jpg" in ids, ids
    assert "garden" in r.doc("photos/living-room.jpg"), \
        r.doc("photos/living-room.jpg")


def test_image_skipped_when_llm_down():
    """With no reachable model (the test env's dead port), images are skipped with
    a clear message and never silently stored — text docs in the same folder still
    get indexed (one bad file must not kill the batch)."""
    import contextlib
    import io
    import ragcli

    d = _tmpdir()
    photos = os.path.join(d, "photos")
    os.makedirs(photos)
    with open(os.path.join(photos, "a.jpg"), "wb") as f:
        f.write(b"\xff\xd8\xff\x00")
    with open(os.path.join(photos, "notes.txt"), "w") as f:
        f.write("invoice 12345")
    cfg = os.path.join(d, "rag.json")
    db = os.path.join(d, "t.db")
    buf = io.StringIO()
    with contextlib.redirect_stderr(buf):
        code = ragcli.main(["--corpus", d, "--config", cfg, "--db", db,
                            "add", photos])
    assert code == 0, code
    out = buf.getvalue()
    assert "a.jpg" in out and "skip:" in out, out
    r = Rag(db=db)
    ids = r.doc_ids()
    assert "photos/a.jpg" not in ids, ids
    assert "photos/notes.txt" in ids, ids


def test_embed_max_chars_from_config_file():
    # the rag.json key must work too, not only the env var (README promises both)
    import rag

    saved = dict(rag.CFG)
    try:
        rag.CFG["embed_max_chars"] = 123
        assert _embed_inputs(["x" * 500]) == ["x" * 123]
        os.environ["RAG_EMBED_MAX_CHARS"] = "50"
        assert _embed_inputs(["x" * 500]) == ["x" * 50]  # env still wins
        del os.environ["RAG_EMBED_MAX_CHARS"]
        rag.CFG.clear()
        assert _embed_inputs(["x" * 5000]) == ["x" * 4000]  # built-in default
    finally:
        rag.CFG.clear()
        rag.CFG.update(saved)


def test_min_score_prefers_semantic_over_bm25():
    # docs say the threshold is cosine; a hybrid-shape hit carries both a big
    # BM25 score and a small sem — min_score must judge the sem, not the BM25
    r = _rag()
    hits = [{"id": "invoice", "text": DOCS["invoice"], "score": 5.0, "sem": 0.1}]
    r.retrieve = lambda q, k=10: list(hits)
    out = r.answer("anything", k=1, min_score=0.3)
    assert out["status"] == "empty", out  # 0.1 sem < 0.3 -> dropped
    hits[0]["sem"] = 0.9
    out = r.answer("anything", k=1, min_score=0.3)
    assert out["status"] == "no_model" and out["citations"][0]["score"] == 0.9, out


def test_llm_sends_model_only_when_configured():
    # default is "the endpoint's own model": a stale hardcoded name must not
    # ride along and 404 every call
    import io
    import json as _json
    import rag

    captured = []

    class FakeResp(io.BytesIO):
        def __enter__(self):
            return self
        def __exit__(self, *a):
            return False

    def fake_urlopen(req, timeout=None):
        captured.append(_json.loads(req.data))
        return FakeResp(_json.dumps(
            {"choices": [{"message": {"content": "hi"}, "finish_reason": "stop"}]}).encode())

    real = rag.urllib.request.urlopen
    saved = dict(rag.CFG)
    try:
        rag.urllib.request.urlopen = fake_urlopen
        rag.CFG.clear()
        assert rag.llm([{"role": "user", "content": "q"}]) == "hi"
        assert "model" not in captured[-1], captured[-1]
        rag.CFG["llm_model"] = "cfg-model"
        rag.llm([{"role": "user", "content": "q"}])
        assert captured[-1]["model"] == "cfg-model", captured[-1]
    finally:
        rag.urllib.request.urlopen = real
        rag.CFG.clear()
        rag.CFG.update(saved)


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print("ok", t.__name__)
    print(f"{len(tests)} checks passed")
