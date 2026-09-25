#!/usr/bin/env python3
"""rag.py — the recipe book from rag.md, implemented ladder-first.

Recipes: 1 BM25 full-text · 2 LLM query rewriting · 3 hybrid (BM25 candidates ->
embedding rerank) · 4 on-the-fly embedding · 5 hot/cold tiers · 6 full
pre-embedding · plus multi-intent decomposition.

Ladder: SQLite FTS5 is the search engine (stdlib, native BM25), urllib is the
LLM/embedding client (stdlib), numpy is the vector math (already installed).
No Elasticsearch, no vector DB, no framework, no chunking — documents are
stored whole, which is recipe 1's whole point.

Run the recipes:   python3 rag.py index docs/*.md && python3 rag.py ask "..."
Decision tree:     python3 rag.py recommend --qpd 500 --churn 2
"""
import hashlib
import json
import os
import re
import sqlite3
import threading
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import numpy as np

EMBED_DIM = 512

STOP = frozenset(
    "a an and are as at be by do does for from how i in is it my of on or the to what when where which with you your".split()
)
# ponytail: seed synonyms for the no-LLM fallback; the LLM rewriter is the real
# expansion path (recipe 2), this just keeps search sane without a model.
SYN = {"car": ("automobile", "vehicle"), "bug": ("error", "defect"), "fix": ("debug", "repair")}

SCHEMA = """
CREATE VIRTUAL TABLE IF NOT EXISTS docs USING fts5(id UNINDEXED, text);
CREATE TABLE IF NOT EXISTS vectors(id TEXT PRIMARY KEY, vec BLOB, model TEXT);
CREATE TABLE IF NOT EXISTS access(id TEXT PRIMARY KEY, n INTEGER NOT NULL DEFAULT 0);
"""


# --------------------------------------------------------------------------- helpers
def _toks(s):
    return re.findall(r"[A-Za-z0-9]+", s)


def fts_match(query):
    """Quote every token and OR them: arbitrary user input can't break the MATCH
    parser, and BM25's idf does the ranking."""
    toks = _toks(query)
    return " OR ".join('"%s"' % t for t in toks) if toks else None


def _api_base(url):
    url = url.rstrip("/")
    return url if url.endswith("/v1") else url + "/v1"


def llm(messages, model=None, max_tokens=1024, timeout=120):
    """OpenAI-compatible chat call, stdlib only. None when unreachable -> callers
    fall back to deterministic rewriting."""
    base = _api_base(os.environ.get("RAG_LLM_BASE") or os.environ.get("LLAMA_CPP_BASE_URL") or "http://127.0.0.1:8888")
    url = base + "/chat/completions"
    for effort, budget in ((None, max_tokens), ("none", max_tokens * 2)):
        body = {
            "model": model or os.environ.get("RAG_LLM_MODEL") or "deepseek-v4.1-flash",
            "messages": messages,
            "temperature": 0,
            "max_tokens": budget,
        }
        if effort:
            body["reasoning_effort"] = effort  # sent only on retry: strict endpoints never see it
        req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                out = json.load(r)
        except Exception:
            return None
        choice = out["choices"][0]
        content = (choice["message"].get("content") or "").strip()
        if content:
            return content
        if choice.get("finish_reason") != "length":
            return None
        # ponytail: a reasoning model burns the whole budget on reasoning_content and
        # returns empty content. Query rewriting is a trivial task, so the retry
        # turns reasoning off (llama.cpp-style servers) and doubles the room.
    return None


def _embed_inputs(texts, cap=None):
    """ponytail: truncate to the model's context window — a 2048-token embedder
    returns HTTP 400 on a long document, and one bad text sank the whole batch to
    the hashed fallback. Chunk only if whole-doc truncation measurably hurts ranking."""
    cap = int(cap if cap is not None else os.environ.get("RAG_EMBED_MAX_CHARS", "4000"))
    return [t[:cap] for t in texts]


def _hash_vec(text):
    """ponytail: lexical hashed bag-of-words, NOT semantic. It stands in for a
    real embedding model so recipes 3-6 are runnable with zero setup; point
    RAG_EMBED_BASE at an /embeddings endpoint for actual semantics."""
    v = np.zeros(EMBED_DIM, dtype=np.float32)
    for t in _toks(text.lower()):
        h = int.from_bytes(hashlib.blake2b(t.encode(), digest_size=4).digest(), "big") % EMBED_DIM
        v[h] += 1.0
    n = float(np.linalg.norm(v))
    return v / n if n else v


def _glossary():
    path = os.environ.get("RAG_GLOSSARY", "glossary.txt")
    if os.path.exists(path):
        return open(path, encoding="utf-8").read().strip()
    return ""


def _rewrite_messages(query, glossary, feedback=None):
    sys = (
        "You rewrite a user's question into a compact keyword search string.\n"
        "Drop stopwords. Expand synonyms. Decompose nothing — one string only.\n"
        "Domain-specific terms must be preserved EXACTLY as written:\n"
        f"{glossary or '(none configured)'}\n"
        "Reply with the keywords and nothing else."
    )
    user = query if not feedback else f"{query}\n\nThe last search missed these: {feedback}\nTry again."
    return [{"role": "system", "content": sys}, {"role": "user", "content": user}]


def _decompose_messages(query):
    return [
        {"role": "system", "content": 'Split the user request into focused sub-queries. Reply as JSON: {"sub_queries": ["..."]}'},
        {"role": "user", "content": query},
    ]


def _feedback(results, quality):
    top = " ".join(r["text"][:200] for r in results[:3])
    return f"quality={quality:.2f}; top hits contain: {top}"


def _rec(recipe, why):
    return {"recipe": recipe, "why": why}


# --------------------------------------------------------------------------- engine
class Rag:
    def __init__(self, db=":memory:", glossary=None, use_llm=True, hot_threshold=3):
        # ponytail: one shared connection; sqlite3.threadsafety==3 (serialized) makes
        # it safe for the parallel sub-query fan-out. Per-thread conns if that changes.
        self.db = sqlite3.connect(db, check_same_thread=False)
        self.db.executescript(SCHEMA)
        self.use_llm = use_llm
        self.hot_threshold = hot_threshold
        self.glossary = glossary if glossary is not None else _glossary()
        self.embed_model = os.environ.get("RAG_EMBED_MODEL", "hash-bow-512")
        self.lock = threading.Lock()

    def _q(self, sql, params=()):
        """ponytail: one lock around every statement — a sqlite connection is not safe
        for concurrent use, and the parallel fan-out's real win is the API calls, not
        the DB. Per-thread connections if DB time ever dominates."""
        with self.lock:
            rows = self.db.execute(sql, params).fetchall()
            self.db.commit()
            return rows

    # -- store: whole documents, no chunking -------------------------------------
    def add(self, doc_id, text):
        self._q("DELETE FROM docs WHERE id=?", (doc_id,))
        self._q("INSERT INTO docs(id, text) VALUES(?,?)", (doc_id, text))
        self._q("DELETE FROM vectors WHERE id=?", (doc_id,))

    def doc(self, doc_id):
        rows = self._q("SELECT text FROM docs WHERE id=?", (doc_id,))
        return rows[0][0] if rows else ""

    def doc_ids(self):
        return [r[0] for r in self._q("SELECT id FROM docs")]

    # -- recipe 1: BM25 ----------------------------------------------------------
    def candidates(self, query, n):
        m = fts_match(query)
        if not m:
            return []
        rows = self._q(
            "SELECT id, text, bm25(docs) AS s, snippet(docs, 1, '[', ']', ' … ', 10) FROM docs WHERE docs MATCH ? ORDER BY rank LIMIT ?",
            (m, n),
        )
        # bm25 is negative; flip it. snip = window around the match, not the doc's first line.
        return [{"id": r[0], "text": r[1], "score": -r[2], "snip": r[3]} for r in rows]

    def search(self, query, k=10):
        """Recipe 1: full-text search only. Zero cost, ~ms, debuggable, no chunking."""
        out = self.candidates(query, k)
        self._bump(out)
        return out

    def _bump(self, results):
        for r in results:
            self._q("INSERT INTO access(id,n) VALUES(?,1) ON CONFLICT(id) DO UPDATE SET n=n+1", (r["id"],))

    # -- recipe 2: query rewriting ----------------------------------------------
    def rewrite(self, query, feedback=None):
        """Recipe 2: the LLM turns messy queries into keyword searches. Results bad?
        Edit the prompt, not the corpus. Falls back to stopword/synonym stripping."""
        if self.use_llm:
            out = llm(_rewrite_messages(query, self.glossary, feedback))
            if out:
                return out.strip().strip('"')
        return self._rewrite_fallback(query)

    def _rewrite_fallback(self, query):
        terms = []
        for t in _toks(query):
            lt = t.lower()
            if lt in STOP:
                continue
            terms.append(t)  # proprietary terms like "Atlas" survive untouched
            terms.extend(SYN.get(lt, ()))
        return " ".join(dict.fromkeys(terms)) or query

    def agentic_search(self, query, k=10, max_iterations=3, threshold=0.6):
        """Recipe 2's loop: rewrite -> search -> evaluate -> refine. No re-indexing."""
        q, results = query, []
        for _ in range(max_iterations):
            results = self.search(self.rewrite(q), k=k)
            quality = self.evaluate(results, query)
            if quality >= threshold or not results:
                break
            nxt = self.rewrite(query, feedback=_feedback(results, quality))
            if nxt == q:
                break  # deterministic fallback can't change -> iterating is pointless
            q = nxt
        return results

    def evaluate(self, results, query):
        """ponytail: lexical-coverage proxy — share of query terms present in the top
        hits. Swap in LLM-as-judge when the budget is there."""
        want = {t.lower() for t in _toks(query) if t.lower() not in STOP}
        if not want or not results:
            return 0.0
        have = set()
        for r in results[:3]:
            have.update(t.lower() for t in _toks(r["text"]))
        return len(want & have) / len(want)

    # -- recipes 3 & 4: hybrid / on-the-fly --------------------------------------
    def embed(self, texts):
        vecs = self._embed_api(texts) if os.environ.get("RAG_EMBED_BASE") else None
        return vecs if vecs is not None else [_hash_vec(t) for t in texts]

    def _embed_api(self, texts):
        base = _api_base(os.environ["RAG_EMBED_BASE"])
        body = {"model": self.embed_model, "input": _embed_inputs(texts)}
        req = urllib.request.Request(
            base + "/embeddings", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}
        )
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                data = json.load(r)["data"]
            return [np.asarray(d["embedding"], dtype=np.float32) for d in data]
        except Exception:
            return None  # falls back to hashed vectors rather than dying mid-query

    def hybrid(self, query, k=10, candidates=50):
        """Recipes 3 & 4: BM25 top-N candidates, embedding rerank, top-k. Docs are
        embedded per query (on-the-fly) — no vector index to maintain, always fresh.
        Cost: ~$0.0005/query. Latency: +200-500ms. That's the trade, not the money."""
        cands = self.candidates(query, candidates)
        if not cands:
            return []
        vecs = self.embed([query] + [c["text"] for c in cands])
        qv = np.asarray(vecs[0], dtype=np.float32)
        for c, dv in zip(cands, vecs[1:]):
            dv = np.asarray(dv, dtype=np.float32)
            c["sem"] = float(dv @ qv) if dv.shape == qv.shape else 0.0
        cands.sort(key=lambda c: c["sem"], reverse=True)
        out = cands[:k]
        self._bump(out)
        return out

    # -- recipe 5: hot/cold tiers -------------------------------------------------
    def hot_tier(self):
        return {r[0] for r in self._q("SELECT id FROM access WHERE n >= ?", (self.hot_threshold,))}

    def _has_vec(self, doc_id):
        return bool(self._q("SELECT 1 FROM vectors WHERE id=?", (doc_id,)))

    def refresh_hot(self):
        """Recipe 5's weekly job: embed only the newly-hot docs."""
        need = [i for i in self.hot_tier() if not self._has_vec(i)]
        if not need:
            return 0
        for i, v in zip(need, self.embed([self.doc(i) for i in need])):
            self._store_vec(i, v)
        return len(need)

    def _store_vec(self, doc_id, vec):
        self._q(
            "INSERT OR REPLACE INTO vectors(id,vec,model) VALUES(?,?,?)",
            (doc_id, np.asarray(vec, dtype=np.float32).tobytes(), self.embed_model),
        )

    def search_hot_cold(self, query, k=10, candidates=100):
        """Recipe 5: hot docs scored from stored vectors, cold docs embedded on the
        fly, merged. Fast for the 80% of traffic that hits the hot tier."""
        cands = self.candidates(query, candidates)
        if not cands:
            return []
        hot_ids = self.hot_tier()
        hot = [c for c in cands if c["id"] in hot_ids and self._has_vec(c["id"])]
        cold = [c for c in cands if c["id"] not in {h["id"] for h in hot}]
        qv = np.asarray(self.embed([query])[0], dtype=np.float32)
        for c in hot:
            blob = self._q("SELECT vec FROM vectors WHERE id=?", (c["id"],))[0][0]
            dv = np.frombuffer(blob, dtype=np.float32)
            c["sem"] = float(dv @ qv) if dv.shape == qv.shape else 0.0  # model switched -> ignore stale vec
        for c, dv in zip(cold, self.embed([c["text"] for c in cold]) if cold else []):
            dv = np.asarray(dv, dtype=np.float32)
            c["sem"] = float(dv @ qv) if dv.shape == qv.shape else 0.0
        merged = sorted(hot + cold, key=lambda c: c["sem"], reverse=True)[:k]
        self._bump(merged)
        return merged

    # -- recipe 6: full pre-embedding ---------------------------------------------
    def preembed(self):
        """Recipe 6: embed everything once, store vectors. Only sane for a stable
        corpus — every model switch means re-embedding all of it."""
        ids = self.doc_ids()
        if not ids:
            return 0
        for i, v in zip(ids, self.embed([self.doc(i) for i in ids])):
            self._store_vec(i, v)
        return len(ids)

    def search_preembedded(self, query, k=10):
        """Recipe 6's query path: cosine over stored vectors, no per-query doc
        embedding. ponytail: brute-force O(n) scan; swap in hnsw/faiss past ~100K docs."""
        qv = np.asarray(self.embed([query])[0], dtype=np.float32)
        scored = []
        for doc_id, blob in self._q("SELECT id, vec FROM vectors"):
            dv = np.frombuffer(blob, dtype=np.float32)
            if dv.shape == qv.shape:  # skip vectors from a deprecated model
                scored.append((float(dv @ qv), doc_id))
        scored.sort(reverse=True)
        out = [{"id": d, "text": self.doc(d), "score": s} for s, d in scored[:k]]
        self._bump(out)
        return out

    # -- multi-intent -------------------------------------------------------------
    def decompose(self, query):
        """'read CSV, clean data, plot results' is three intents, not one query."""
        if self.use_llm:
            out = llm(_decompose_messages(query))
            if out:
                try:
                    data = json.loads(out[out.find("{") : out.rfind("}") + 1])
                    subs = [s for s in data.get("sub_queries", []) if s]
                    if subs:
                        return subs
                except ValueError:
                    pass
        # ponytail: crude splitter; the LLM is what adds dependencies (read > clean > plot)
        parts = re.split(r"\s+and\s+|,\s*|\s+then\s+", query)
        parts = [re.sub(r"^(and|then)\s+", "", p).strip() for p in parts]
        return [p for p in parts if p] or [query]

    def search_multi_intent(self, query, k=5):
        """Decompose, run each sub-query in parallel (latency = max, not sum), then
        interleave so every intent shows up in the answer."""
        subs = self.decompose(query)
        with ThreadPoolExecutor(max_workers=min(8, len(subs))) as pool:
            lists = list(pool.map(lambda s: self.hybrid(s, k=k, candidates=25), subs))
        out, seen = [], set()
        for i in range(max((len(lst) for lst in lists), default=0)):
            for sub, lst in zip(subs, lists):
                if i < len(lst) and lst[i]["id"] not in seen:
                    seen.add(lst[i]["id"])
                    out.append({**lst[i], "sub_query": sub})
        return out


# --------------------------------------------------------------------------- decision tree
def recommend(
    has_search=False,
    satisfied=None,
    complaint="",
    qpd=0,
    churn_pct_day=0.0,
    corpus=0,
    ml_team=False,
    hot_patterns=False,
    latency_ok=True,
):
    """rag.md's decision tree, encoded. Returns the recipe to build (None = stop)."""
    if not has_search:
        return _rec(1, "no search at all — build BM25 first, stop reading and build it")
    if satisfied:
        return _rec(None, "users are happy with the baseline — stop and ship features")
    c = complaint.lower()
    if "can't find" in c or "cannot find" in c or "exist" in c:
        return _rec(2, "vocabulary mismatch, not retrieval — LLM rewriting is ~$0.001/query and zero re-indexing")
    if "not great" in c or "okay" in c or "semantic" in c:
        if not latency_ok:
            return _rec(2, "100-500ms rerank isn't worth it here — push query rewriting further instead")
        if churn_pct_day > 10:
            return _rec(4, "high churn (>10%/day) — embed candidates on-the-fly, nothing to re-index")
        if hot_patterns:
            return _rec(5, "Pareto access pattern — pre-embed the hot 20%, embed cold on the fly")
        if corpus > 100_000 and qpd > 10_000 and ml_team:
            return _rec(6, "stable corpus + >10K q/day + ML team — full pre-embedding with ANN")
        return _rec(3, "hybrid: BM25 candidates, embedding rerank, top-k")
    if qpd > 10_000 and corpus > 100_000 and ml_team and churn_pct_day < 5:
        return _rec(6, "very high volume, stable corpus, ML team to run the index")
    if churn_pct_day > 10:
        return _rec(4, "high churn — on-the-fly embedding keeps freshness perfect")
    if qpd < 1000:
        return _rec(1, "under 1K q/day — full-text is sufficient; add rewriting only if users miss terms")
    return _rec(3, "1K-10K q/day — selective optimization: hybrid rerank")


# --------------------------------------------------------------------------- CLI
def main(argv=None):
    import argparse

    p = argparse.ArgumentParser(prog="rag", description="rag.md recipes, ladder-first")
    p.add_argument("--db", default=os.environ.get("RAG_DB", "rag.db"))
    p.add_argument("--no-llm", action="store_true", help="force deterministic rewriting")
    sub = p.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("index", help="store whole files as documents")
    a.add_argument("paths", nargs="+")
    for name in ("search", "ask", "hybrid", "multi"):
        s = sub.add_parser(name)
        s.add_argument("query")
        s.add_argument("-k", type=int, default=5)
    pre = sub.add_parser("pre", help="pre-embed (recipe 6); with a query, search the stored vectors")
    pre.add_argument("query", nargs="?")
    pre.add_argument("-k", type=int, default=5)
    hot = sub.add_parser("hot", help="search with hot/cold tiers (recipe 5)")
    hot.add_argument("query")
    hot.add_argument("-k", type=int, default=5)
    r = sub.add_parser("recommend", help="run the decision tree")
    r.add_argument("--qpd", type=int, default=0)
    r.add_argument("--churn", type=float, default=0.0, help="%% of docs changed per day")
    r.add_argument("--corpus", type=int, default=0)
    r.add_argument("--complaint", default="")
    r.add_argument("--ml", action="store_true")
    r.add_argument("--hot-patterns", action="store_true")
    r.add_argument("--no-search", action="store_true")
    r.add_argument("--latency-ok", action=argparse.BooleanOptionalAction, default=True)
    args = p.parse_args(argv)

    if args.cmd == "recommend":
        rec = recommend(
            has_search=not args.no_search,
            complaint=args.complaint,
            qpd=args.qpd,
            churn_pct_day=args.churn,
            corpus=args.corpus,
            ml_team=args.ml,
            hot_patterns=args.hot_patterns,
            latency_ok=args.latency_ok,
        )
        print(f"recipe {rec['recipe']}: {rec['why']}")
        return 0

    rag = Rag(db=args.db, use_llm=not args.no_llm)

    if args.cmd == "index":
        n = 0
        for path in args.paths:
            rag.add(path, open(path, encoding="utf-8").read())
            n += 1
        print(f"indexed {n} documents (whole, unchunked) -> {args.db}")
        return 0

    q = args.query
    hits = []
    if args.cmd == "search":
        hits = rag.search(q, k=args.k)
    elif args.cmd == "ask":
        hits = rag.search(rag.rewrite(q), k=args.k)
    elif args.cmd == "hybrid":
        hits = rag.hybrid(q, k=args.k)
    elif args.cmd == "multi":
        hits = rag.search_multi_intent(q, k=args.k)
    elif args.cmd == "hot":
        rag.refresh_hot()
        hits = rag.search_hot_cold(q, k=args.k)
    elif args.cmd == "pre":
        if args.query:
            hits = rag.search_preembedded(q, k=args.k)
        else:
            print(f"pre-embedded {rag.preembed()} documents")
            return 0
    for h in hits:
        extra = f" [{h['sub_query']}]" if "sub_query" in h else ""
        score = h.get("sem", h.get("score", 0.0))
        print(f"{score:+.4f}  {h['id']}{extra}  {(h.get('snip') or h['text'][:70]).replace(chr(10), ' ')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
