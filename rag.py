#!/usr/bin/env python3
"""rag.py — the recipe book from recipes.md, implemented ladder-first.

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
import sys
import threading
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from importlib import metadata

import numpy as np

EMBED_DIM = 512


def _get_version():
    """Return the installed package version, falling back to pyproject next to this file."""
    try:
        return metadata.version("rag-ladder")
    except metadata.PackageNotFoundError:
        pass
    try:  # never a second hardcoded version: it rots the first time you bump
        m = re.search(r'^version = "([^"]+)"',
                      open(os.path.join(os.path.dirname(__file__), "pyproject.toml"),
                           encoding="utf-8").read(), re.M)
        return m[1] if m else "0.0.0"
    except OSError:
        return "0.0.0"


VERSION = _get_version()

CFG = {}  # rag.json contents, loaded by ragcli at startup; env vars still win


def _cfg(env, key, default=None):
    """Resolution order: env var > rag.json > built-in default."""
    v = os.environ.get(env)
    if v:
        return v
    v = CFG.get(key)
    return v if v not in (None, "") else default


def _headers(key=None):
    h = {"Content-Type": "application/json"}
    if key:
        h["Authorization"] = "Bearer " + key
    return h


# A dead endpoint degrades retrieval to BM25 + hashed vectors silently; one
# stderr line per process names the real cause (401 != "no model configured").
_warned = False


def _warn_once(e):
    global _warned
    if not _warned:
        _warned = True
        code = getattr(e, "code", None)
        print(f"warning: endpoint call failed ({'HTTP ' + str(code) if code else e}) "
              "— falling back to BM25/hashed retrieval; run: rag doctor", file=sys.stderr)

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
CREATE TABLE IF NOT EXISTS meta(id TEXT PRIMARY KEY, sha TEXT, mtime REAL, chars INTEGER);
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
    base = _api_base(_cfg("RAG_LLM_BASE", "llm_base", os.environ.get("LLAMA_CPP_BASE_URL") or "http://127.0.0.1:8888"))
    key = _cfg("RAG_API_KEY", "llm_api_key", None)
    url = base + "/chat/completions"
    # ponytail: "auto" keeps attempt 1 field-free, so strict OpenAI-compatible
    # endpoints never see reasoning_effort. "none" is for endpoints running a
    # reasoning model, where a trivial rewrite otherwise burns ~20s thinking.
    reasoning = _cfg("RAG_LLM_REASONING", "llm_reasoning", "auto")
    for effort, budget in ((reasoning if reasoning != "auto" else None, max_tokens), ("none", max_tokens * 2)):
        body = {
            "messages": messages,
            "temperature": 0,
            "max_tokens": budget,
        }
        # README default is "the endpoint's own model": send the field only when a
        # name is configured, so a stale hardcoded default can't 404 every call.
        name = model or _cfg("RAG_LLM_MODEL", "llm_model", None)
        if name:
            body["model"] = name
        if effort:
            body["reasoning_effort"] = effort  # sent only on retry: strict endpoints never see it
        req = urllib.request.Request(url, data=json.dumps(body).encode(), headers=_headers(key))
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                out = json.load(r)
        except Exception as e:
            _warn_once(e)
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
    cap = int(cap if cap is not None else _cfg("RAG_EMBED_MAX_CHARS", "embed_max_chars", 4000))
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
    path = _cfg("RAG_GLOSSARY", "glossary", "glossary.txt")
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


# The honesty requirement: a tool that invents a clause is worse than one that
# says it does not know. Both modes are instructed to admit absence explicitly.
ANSWER_PROMPTS = {
    "general": (
        "Answer the question using ONLY the numbered sources below.\n"
        "Quote the relevant text verbatim and cite the source number like [1].\n"
        "If the sources do not contain the answer, reply exactly NOT_FOUND_IN_CORPUS.\n"
        "Never use outside knowledge."
    ),
    "legal": (
        "You are a document retrieval assistant for legal and contract review.\n"
        "Answer the question using ONLY the numbered sources below.\n"
        "Quote the exact clause text verbatim and cite the source number like [1].\n"
        "Never paraphrase a clause, never infer, never apply outside law or knowledge.\n"
        "If the sources do not contain the answer, reply exactly NOT_FOUND_IN_CORPUS."
    ),
}
NOT_FOUND = "NOT_FOUND_IN_CORPUS"


def _answer_messages(query, sources, mode):
    sys = ANSWER_PROMPTS.get(mode, ANSWER_PROMPTS["general"])
    return [
        {"role": "system", "content": sys},
        {"role": "user", "content": f"Sources:\n\n{sources}\n\nQuestion: {query}"},
    ]


def _feedback(results, quality):
    top = " ".join(r["text"][:200] for r in results[:3])
    return f"quality={quality:.2f}; top hits contain: {top}"


def _rec(recipe, why):
    return {"recipe": recipe, "why": why}


# ---------------------------------------------------------------- answer windows
# Documents are stored whole, so a retrieved document is usually far longer than
# what a model can take. Cutting off the head sends the cover page of a 60-page PDF
# and then reports NOT_FOUND_IN_CORPUS about an answer the corpus does contain —
# measured: a 62,698-char contract, rent clause at char 6,451, first 2,000 sent.
WIN_CTX = 150  # chars of context kept around each matched term


def _answer_budget(con=None):
    """How much of each retrieved document a cited answer is built from. This is a
    budget, not a head-cut: what gets sent is the regions that match the question
    (see _spans). Default grows with the corpus — the biggest document fits whole —
    so nobody hunts for a config key just to read page 12 of a contract.
    answer_max_chars (or RAG_ANSWER_MAX_CHARS) stays as an override. The prompt
    costs k documents x this."""
    v = _cfg("RAG_ANSWER_MAX_CHARS", "answer_max_chars", None)
    if v:
        return int(v)
    # ponytail: no ceiling from the model's context window; a corpus bigger than the
    # endpoint's context overflows the prompt — set answer_max_chars when it does.
    try:
        row = con.execute("SELECT max(chars) FROM meta").fetchone()
    except (sqlite3.Error, AttributeError):
        return 2000
    return max(2000, row[0] or 0)


def _word(term):
    """Match a whole word, Unicode-aware. ASCII-only _toks splits German words
    (Kühler -> h, ler), and the windowing has to follow the corpus's own spelling.
    Whole-word so 'rent' never matches inside 'Miete' or 'Vermieterverein'."""
    return re.compile(r"(?<!\w)" + re.escape(term) + r"(?!\w)", re.IGNORECASE)


def _spans(text, query, cap, ctx=WIN_CTX):
    """The part of a document to send to a model, as one string.

    Rank windows of +/-ctx chars around each query term by how distinctive they are
    -- a term appearing once in the document is worth more than one appearing 200
    times, which is what keeps 'die'/'ist' boilerplate from pushing out the clause
    you asked about -- take the best up to cap, and return them in reading order.
    A document that fits is returned whole. With no lexical match (a vector-only
    hit, or a query of pure stopwords) the head is all there is, and the caller
    reports the difference as truncation rather than hiding it.
    """
    if len(text) <= cap:
        return text
    terms = {t.lower() for t in re.findall(r"\w+", query) if len(t) > 2} - STOP
    if not terms:
        return text[:cap]
    freq = {t: max(1, len(_word(t).findall(text))) for t in terms}
    wins = []
    for t in sorted(terms):
        for m in _word(t).finditer(text):
            wins.append((max(0, m.start() - ctx), min(len(text), m.end() + ctx)))
    if not wins:
        return text[:cap]
    merged = []
    for s, e in sorted(wins):
        if merged and s <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    ranked = []
    for s, e in merged:
        piece = text[s:e]
        # inverse document frequency inside the document itself: distinctive terms
        # choose the window, connectives don't. No corpus stats needed.
        weight = sum(1.0 / freq[t] for t in terms if _word(t).search(piece))
        ranked.append((-weight, s, e, piece))
    ranked.sort()
    chosen, used = [], 0
    for _, s, _, piece in ranked:
        if used + len(piece) > cap:
            continue  # no half-windows: a fragment of a clause is not evidence
        chosen.append((s, piece))
        used += len(piece)
    if not chosen:
        return text[:cap]
    return "\n…\n".join(p for _, p in sorted(chosen))


def _answer_sources(query, hits, cap):
    """Number the documents for the model, each represented by its matching regions.
    Returns (sources, chars_sent_per_hit). FTS's own snippet words join the query for
    windowing: they are the terms the search actually matched, usually closer to the
    corpus's vocabulary than the user's wording."""
    parts, sent = [], []
    for i, h in enumerate(hits):
        window = _spans(h["text"], f"{query} {h.get('snip', '')}", cap)
        sent.append(len(window))
        parts.append(f"[{i + 1}] {h['id']}\n{window}")
    return "\n\n".join(parts), sent


def _not_found_note(truncated):
    """Say what was actually looked at. 'the corpus has no answer' is false when only
    part of a document was sent; the two must be tellable apart."""
    if not truncated:
        return "the corpus does not contain an answer"
    return (
        f"the parts sent do not contain the answer, and {len(truncated)} document(s) "
        "were truncated — the answer may be in the part that was not sent "
        "(see 'truncated'; raise answer_max_chars or search the term directly)"
    )


# --------------------------------------------------------------------------- engine
class Rag:
    def __init__(self, db=":memory:", glossary=None, use_llm=True, hot_threshold=3):
        # ponytail: one shared connection; sqlite3.threadsafety==3 (serialized) makes
        # it safe for the parallel sub-query fan-out. Per-thread conns if that changes.
        self.db = sqlite3.connect(db, check_same_thread=False)
        self.db.executescript(SCHEMA)
        self.db_path = db
        self.use_llm = use_llm
        self.hot_threshold = hot_threshold
        self.glossary = glossary if glossary is not None else _glossary()
        self.embed_model = _cfg("RAG_EMBED_MODEL", "embed_model", "hash-bow-512")
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
    def add(self, doc_id, text, mtime=None):
        """Whole-document store. Returns False when the content hash is unchanged,
        so re-indexing a folder only pays for the files that actually changed."""
        sha = hashlib.sha256(text.encode("utf-8")).hexdigest()
        row = self._q("SELECT sha FROM meta WHERE id=?", (doc_id,))
        if row and row[0][0] == sha:
            return False
        self._q("DELETE FROM docs WHERE id=?", (doc_id,))
        self._q("INSERT INTO docs(id, text) VALUES(?,?)", (doc_id, text))
        self._q("DELETE FROM vectors WHERE id=?", (doc_id,))
        self._q(
            "INSERT OR REPLACE INTO meta(id,sha,mtime,chars) VALUES(?,?,?,?)",
            (doc_id, sha, mtime if mtime is not None else 0.0, len(text)),
        )
        return True

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

    # -- cited answers -------------------------------------------------------------
    def retrieve(self, query, k=10):
        """Raw query AND its rewrite, unioned. The rewrite fixes vocabulary mismatch
        but FTS5 has no stemming, so it can also mutate a term the corpus spells
        differently ('refunds' -> 'refund') and miss the exact document. One extra
        BM25 pass buys the exact-match path back."""
        raw = self.search(query, k=k)
        if not self.use_llm:
            return raw
        out, seen = [], set()
        for h in raw + self.search(self.rewrite(query), k=k):
            if h["id"] not in seen:
                seen.add(h["id"])
                out.append(h)
        return out[:k]

    def answer(self, query, k=6, mode=None, pre=False, min_score=0.0):
        """Retrieval + synthesis with citations. Never invents an answer: with no
        model it returns the passages and says so, and a NOT_FOUND reply from the
        model is surfaced as status='not_found', not smoothed over.

        Each document is sent as its matching regions, not its first screenful (see
        _spans). Every citation says how much of itself was sent, and a truncated
        document is listed under 'truncated' — so an answer that misses something
        deep in a file is a visible tradeoff instead of a mystery.

        pre=True: retrieve from stored vectors (recipe 6) instead of BM25. Requires
        a prior `preembed()` call; falls back to BM25 if no vectors exist.
        min_score: drop hits scoring below this before sending to the LLM.
        0.0 (default) sends all top-k; raise it to avoid burning tokens on
        irrelevant docs (e.g. 0.3 for cosine similarity)."""
        mode = mode or CFG.get("answer_mode", "general")
        if pre:
            hits = self.search_preembedded(query, k=k)
            if not hits:
                hits = self.retrieve(query, k=k)
        else:
            hits = self.retrieve(query, k=k)
        if min_score > 0:
            # cosine is the documented scale; prefer sem when a rerank provided it
            hits = [h for h in hits if h.get("sem", h.get("score", 0.0)) >= min_score]
        cites = [
            {
                "n": i + 1,
                "id": h["id"],
                "snip": h.get("snip") or h["text"][:160],
                "score": h.get("sem", h.get("score", 0.0)),
                "chars": len(h["text"]),
                "sent": 0,  # filled below, once the window is chosen
            }
            for i, h in enumerate(hits)
        ]
        if not hits:
            return {"answer": None, "status": "empty", "citations": [],
                    "note": "no matching passages in the corpus"}
        sources, sent = _answer_sources(query, hits, _answer_budget(self.db))
        for c, n in zip(cites, sent):
            c["sent"] = n
        truncated = [
            {"n": c["n"], "id": c["id"], "chars": c["chars"], "sent": c["sent"]}
            for c in cites if c["sent"] < c["chars"]
        ]
        if self.use_llm:
            out = llm(_answer_messages(query, sources, mode))
            if out:
                if NOT_FOUND in out:
                    return {"answer": None, "status": "not_found", "citations": cites,
                            "note": _not_found_note(truncated), "truncated": truncated}
                return {"answer": out, "status": "ok", "citations": cites, "note": None,
                        "truncated": truncated}
        return {"answer": None, "status": "no_model", "citations": cites,
                "note": "no model configured — showing matching passages only",
                "truncated": truncated}

    # -- recipes 3 & 4: hybrid / on-the-fly --------------------------------------
    def embed(self, texts):
        base = _cfg("RAG_EMBED_BASE", "embed_base", None)
        vecs = self._embed_api(texts, base) if base else None
        return vecs if vecs is not None else [_hash_vec(t) for t in texts]

    def _embed_api(self, texts, base):
        base = _api_base(base)
        key = _cfg("RAG_API_KEY", "llm_api_key", None)
        body = {"model": self.embed_model, "input": _embed_inputs(texts)}
        req = urllib.request.Request(
            base + "/embeddings", data=json.dumps(body).encode(), headers=_headers(key)
        )
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                data = json.load(r)["data"]
            return [np.asarray(d["embedding"], dtype=np.float32) for d in data]
        except Exception as e:
            _warn_once(e)
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
    """recipes.md's decision tree, encoded. Returns the recipe to build (None = stop)."""
    if not has_search:
        return _rec(1, "no search at all — build BM25 first, stop reading and build it")
    if satisfied:
        return _rec(None, "users are happy with the baseline — stop and ship features")
    c = complaint.lower().replace("'", "")
    if "cant find" in c or "cannot find" in c or "exist" in c:
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
    """Kept so `python3 rag.py ...` still works. The real CLI — config, ingestion,
    the browser UI — lives in ragcli; one CLI, so the two can't drift apart."""
    import ragcli

    return ragcli.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
