# rag-ladder productization — implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to
> implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for
> tracking. Executed inline in this session.

**Goal:** Make `rag` configurable and usable by a non-technical user: a config
file with a wizard and a doctor, cloud LLM support, PDF/DOCX ingestion, cited
answers with not-found honesty, and a browser UI — without adding a framework,
a vector DB, or chunking.

**Architecture:** Three files split at the engine/app seam. `rag.py` stays the
engine (retrieval, LLM/embed clients, storage, answer synthesis). `ragcli.py` is
the app layer (config, ingestion, CLI, stdlib web UI). `test_rag.py` holds every
runnable check.

**Tech Stack:** Python 3.10+, SQLite FTS5 (stdlib), urllib (stdlib), numpy
(declared), `http.server` (stdlib). Optional extras: `pypdf`, `python-docx`.

**Spec:** `docs/superpowers/specs/2026-09-26-rag-productization-design.md`

## Global Constraints

- `requires-python = ">=3.10"` → config format is JSON, never TOML.
- Core dependencies stay `numpy` only. PDF/DOCX extractors are optional extras.
- Resolution order everywhere: `CLI flag > env var > rag.json > default`.
- Env vars `RAG_LLM_BASE`, `RAG_LLM_MODEL`, `RAG_EMBED_BASE`, `RAG_EMBED_MODEL`,
  `RAG_EMBED_MAX_CHARS`, `RAG_GLOSSARY`, `RAG_DB` keep working. New: `RAG_API_KEY`,
  `RAG_CONFIG`.
- Documents are stored whole. No chunking, ever.
- No framework, no build step, no vector DB.
- Web UI binds `127.0.0.1` by default and exposes read-only GET routes only.
- Errors are one friendly line + an exit code. No broad `except Exception`.
- All existing `test_rag.py` checks stay green.

---

### Task 1: Config resolution + cloud auth

**Files:**
- Modify: `rag.py` (module globals, `llm()`, `_embed_api()`)
- Create: `ragcli.py`
- Test: `test_rag.py`

**Interfaces:**
- Produces: `rag.CFG` (dict), `rag._cfg(env, key, default=None)`, `rag._headers(key)`,
  `ragcli.load_config(path=None) -> dict`, `ragcli.CONFIG_KEYS`, `ragcli.PRESETS`,
  `ragcli.resolve(name, flag, env, default)`.
- Consumes: nothing from earlier tasks.

**Config keys** (`ragcli.CONFIG_KEYS`): `llm_base`, `llm_model`, `llm_api_key`,
`embed_base`, `embed_model`, `embed_max_chars`, `db`, `corpus`, `glossary`,
`answer_mode`, `serve_host`, `serve_port`.

**Presets** (`ragcli.PRESETS`): `local`, `ollama`, `openrouter`, `openai`.

- [ ] **Step 1: Write the failing test**

```python
def test_config_resolution(tmpdir):
    import json, ragcli
    cfg = os.path.join(tmpdir, "rag.json")
    with open(cfg, "w") as f:
        json.dump({"llm_base": "https://cfg.example", "llm_model": "cfg-model"}, f)
    c = ragcli.load_config(cfg)
    assert c["llm_base"] == "https://cfg.example"
    assert ragcli.resolve("llm_base", None, None, "http://d") == "https://cfg.example"
    assert ragcli.resolve("llm_base", "http://flag", "http://env", "http://d") == "http://flag"
    assert ragcli.resolve("llm_base", None, "http://env", "http://d") == "http://env"
    assert ragcli.resolve("nope", None, None, "fallback") == "fallback"
    with open(cfg, "w") as f:
        f.write("{not json")
    assert ragcli.load_config(cfg)["llm_base"] == ""  # malformed -> defaults, no crash


def test_cloud_auth_header():
    import rag
    rag.CFG.clear()
    assert rag._headers(None) == {"Content-Type": "application/json"}
    assert rag._headers("sk-abc")["Authorization"] == "Bearer sk-abc"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run test_rag.py`
Expected: FAIL — `ModuleNotFoundError: No module named 'ragcli'`

- [ ] **Step 3: Write minimal implementation**

In `rag.py`, after the imports:

```python
CFG = {}  # rag.json contents; env still wins over it


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
```

Rewrite `llm()`'s base/key/model lines and the `Request` header:

```python
    base = _api_base(_cfg("RAG_LLM_BASE", "llm_base", "http://127.0.0.1:8888"))
    key = _cfg("RAG_API_KEY", "llm_api_key", None)
    url = base + "/chat/completions"
    ...
            "model": model or _cfg("RAG_LLM_MODEL", "llm_model", "deepseek-v4.1-flash"),
    ...
        req = urllib.request.Request(url, data=json.dumps(body).encode(), headers=_headers(key))
```

`_embed_api()` gets the same treatment: `_cfg("RAG_EMBED_BASE", "embed_base", None)`
decides whether the API path is used at all, and `_headers(_cfg("RAG_API_KEY", "llm_api_key", None))`.

In `ragcli.py`:

```python
#!/usr/bin/env python3
"""ragcli — the app layer: config, ingestion, CLI, browser UI."""
import json
import os

CONFIG_KEYS = (
    "llm_base", "llm_model", "llm_api_key", "embed_base", "embed_model",
    "embed_max_chars", "db", "corpus", "glossary", "answer_mode",
    "serve_host", "serve_port",
)

PRESETS = {
    "local": {"llm_base": "http://127.0.0.1:8888", "llm_model": ""},
    "ollama": {"llm_base": "http://127.0.0.1:11434", "llm_model": "llama3.1:8b"},
    "openrouter": {"llm_base": "https://openrouter.ai/api", "llm_model": "openai/gpt-4o-mini"},
    "openai": {"llm_base": "https://api.openai.com", "llm_model": "gpt-4o-mini"},
}


def config_path():
    return os.environ.get("RAG_CONFIG", "rag.json")


def load_config(path=None):
    """Missing file -> {}. Malformed -> one friendly line, then {}."""
    path = path or config_path()
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        return {}
    except (OSError, ValueError) as e:
        print(f"warning: ignoring {path} ({e})")
        return {}
    if not isinstance(data, dict):
        print(f"warning: ignoring {path} (not a JSON object)")
        return {}
    return {k: v for k, v in data.items() if k in CONFIG_KEYS}


def resolve(name, flag, env, default):
    """CLI flag > env var > rag.json > default."""
    if flag is not None and flag != "":
        return flag
    if env and os.environ.get(env):
        return os.environ[env]
    v = CFG.get(name)
    return v if v not in (None, "") else default
```

`CFG` is the module-global loaded once in `main()`; `rag.CFG` and `ragcli.CFG`
point at the same dict (`rag.CFG = CFG`), so the engine and the app layer never
disagree.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run test_rag.py`
Expected: PASS, plus all 10 existing checks still green.

- [ ] **Step 5: Commit**

```bash
git add rag.py ragcli.py test_rag.py
git commit -m "feat: config file with resolution order, cloud LLM auth header"
```

---

### Task 2: Ingestion — PDF/DOCX, recursion, incremental index

**Files:**
- Modify: `rag.py` (`SCHEMA`, `add`)
- Modify: `ragcli.py` (`extract`, `walk`, `ExtractError`)
- Test: `test_rag.py`

**Interfaces:**
- Produces: `ragcli.extract(path) -> str`, `ragcli.walk(paths, corpus) -> list[Path]`,
  `ragcli.ExtractError(Exception)`, `ragcli.doc_id(path, corpus) -> str`.
- `rag.Rag.add(doc_id, text, mtime=None) -> bool` — `False` when the content hash
  is unchanged (skipped), `True` when stored. Adds table `meta(id, sha, mtime, chars)`.

- [ ] **Step 1: Write the failing test**

```python
def test_pdf_missing_dep_message(tmpdir):
    import ragcli
    p = os.path.join(tmpdir, "contract.pdf")
    with open(p, "wb") as f:
        f.write(b"%PDF-1.4\n")
    try:
        ragcli.extract(p)
        assert False, "should have raised"
    except ragcli.ExtractError as e:
        assert "uv add pypdf" in str(e), e


def test_add_incremental(tmpdir):
    r = _rag()
    assert r.add("d", "unchanged text") is True
    assert r.add("d", "unchanged text") is False  # same hash -> skipped
    assert r.add("d", "changed text") is True
    assert r.doc("d") == "changed text"


def test_walk_skips_noise(tmpdir):
    import ragcli
    os.makedirs(os.path.join(tmpdir, ".git"))
    os.makedirs(os.path.join(tmpdir, "sub"))
    for p in ("a.md", ".git/x", "sub/b.md", "rag.db", "rag.json"):
        with open(os.path.join(tmpdir, p), "w") as f:
            f.write("x")
    got = {ragcli.doc_id(p, tmpdir) for p in ragcli.walk([tmpdir], tmpdir)}
    assert got == {"a.md", "sub/b.md"}, got
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run test_rag.py`
Expected: FAIL — `AttributeError: module 'ragcli' has no attribute 'extract'`

- [ ] **Step 3: Write minimal implementation**

`rag.py` — extend `SCHEMA` and rewrite `add`:

```python
CREATE TABLE IF NOT EXISTS meta(id TEXT PRIMARY KEY, sha TEXT, mtime REAL, chars INTEGER);

    def add(self, doc_id, text, mtime=None):
        """Whole-document store. Returns False when the content hash is unchanged."""
        sha = hashlib.sha256(text.encode("utf-8")).hexdigest()
        row = self._q("SELECT sha FROM meta WHERE id=?", (doc_id,))
        if row and row[0][0] == sha:
            return False
        self._q("DELETE FROM docs WHERE id=?", (doc_id,))
        self._q("INSERT INTO docs(id, text) VALUES(?,?)", (doc_id, text))
        self._q("DELETE FROM vectors WHERE id=?", (doc_id,))
        self._q(
            "INSERT OR REPLACE INTO meta(id,sha,mtime,chars) VALUES(?,?,?,?)",
            (doc_id, sha, mtime if mtime is not None else os.path.getmtime(__file__), len(text)),
        )
        return True
```

`mtime=None` defaults to a fixed value so tests stay deterministic; the CLI
passes the real `os.path.getmtime(path)`.

`ragcli.py`:

```python
import re
import sys
from pathlib import Path

TEXT_EXT = {".txt", ".md", ".markdown", ".rst", ".csv", ".tsv", ".json", ".html", ".htm", ".log"}
SKIP_DIRS = {".git", ".venv", "__pycache__", "node_modules"}
SKIP_FILES = {"rag.db", "rag.json", "glossary.txt"}


class ExtractError(Exception):
    pass


def extract(path):
    """File -> text. Optional deps stay optional; missing ones name their install."""
    ext = Path(path).suffix.lower()
    if ext in TEXT_EXT:
        return Path(path).read_text(encoding="utf-8", errors="replace")
    if ext == ".pdf":
        try:
            import pypdf
        except ImportError:
            raise ExtractError("PDF needs pypdf — run: uv add pypdf") from None
        try:
            return "\n".join((pg.extract_text() or "") for pg in pypdf.PdfReader(path).pages)
        except Exception as e:  # a corrupt/encrypted PDF must not kill the batch
            raise ExtractError(f"could not read {path}: {e}") from None
    if ext == ".docx":
        try:
            import docx
        except ImportError:
            raise ExtractError("DOCX needs python-docx — run: uv add python-docx") from None
        try:
            d = docx.Document(path)
            return "\n".join(p.text for p in d.paragraphs)
        except Exception as e:
            raise ExtractError(f"could not read {path}: {e}") from None
    raise ExtractError(f"unsupported format: {ext or '(no extension)'}")


def walk(paths, corpus):
    """Recursive file list, skipping VCS/venv/cache dirs, hidden entries, db/config."""
    out = []
    for p in paths:
        p = Path(p)
        if p.is_file():
            if p.name not in SKIP_FILES:
                out.append(p)
            continue
        for f in sorted(p.rglob("*")):
            if not f.is_file() or f.name in SKIP_FILES:
                continue
            if any(part.startswith(".") or part in SKIP_DIRS for part in f.parts):
                continue
            out.append(f)
    return out


def doc_id(path, corpus):
    """Relative posix path — stable id, readable in citations."""
    try:
        return Path(path).resolve().relative_to(Path(corpus).resolve()).as_posix()
    except ValueError:
        return Path(path).name
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run test_rag.py`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add rag.py ragcli.py test_rag.py
git commit -m "feat: PDF/DOCX ingestion, recursive walk, incremental re-index"
```

---

### Task 3: Cited answers with not-found honesty

**Files:**
- Modify: `rag.py` (`ANSWER_PROMPTS`, `_answer_messages`, `Rag.answer`)
- Test: `test_rag.py`

**Interfaces:**
- Produces: `rag.ANSWER_PROMPTS` (dict), `rag.Rag.answer(query, k=6, mode=None) -> dict`
  with keys `answer`, `status`, `citations`, `note`.
- Consumes: `Rag.search`, `Rag.rewrite`, `rag.llm` from Task 1.

- [ ] **Step 1: Write the failing test**

```python
def test_answer_no_model(tmpdir):
    r = _rag()  # use_llm=False
    out = r.answer("invoice refunds", k=3)
    assert out["answer"] is None
    assert out["status"] == "no_model"
    assert out["citations"] and out["citations"][0]["id"] == "invoice"
    assert "not" in out["note"].lower()


def test_answer_not_found():
    import rag
    r = _rag()
    r.use_llm = True
    orig = rag.llm
    try:
        rag.llm = lambda *a, **kw: "NOT_FOUND_IN_CORPUS"
        out = r.answer("what is the capital of Peru?", k=3)
        assert out["status"] == "not_found", out
        assert out["answer"] is None
    finally:
        rag.llm = orig


def test_answer_ok():
    import rag
    r = _rag()
    r.use_llm = True
    orig = rag.llm
    try:
        rag.llm = lambda *a, **kw: "Refunds take 5 days [1]."
        out = r.answer("how long do refunds take?", k=3)
        assert out["status"] == "ok", out
        assert "[1]" in out["answer"]
        assert out["citations"][0]["id"] == "invoice"
    finally:
        rag.llm = orig
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run test_rag.py`
Expected: FAIL — `AttributeError: 'Rag' object has no attribute 'answer'`

- [ ] **Step 3: Write minimal implementation**

```python
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


    def answer(self, query, k=6, mode=None):
        """Retrieval + synthesis with citations. Never invents an answer."""
        mode = mode or CFG.get("answer_mode", "general")
        hits = self.search(self.rewrite(query) if self.use_llm else query, k=k)
        cites = [
            {"n": i + 1, "id": h["id"], "snip": h.get("snip") or h["text"][:160],
             "score": h.get("sem", h.get("score", 0.0))}
            for i, h in enumerate(hits)
        ]
        if not hits:
            return {"answer": None, "status": "empty", "citations": [],
                    "note": "no matching passages in the corpus"}
        if self.use_llm:
            sources = "\n\n".join(
                f"[{i + 1}] {h['id']}\n{h['text'][:2000]}" for i, h in enumerate(hits)
            )
            out = llm(_answer_messages(query, sources, mode))
            if out:
                if NOT_FOUND in out:
                    return {"answer": None, "status": "not_found", "citations": cites,
                            "note": "the corpus does not contain an answer"}
                return {"answer": out, "status": "ok", "citations": cites, "note": None}
        return {"answer": None, "status": "no_model", "citations": cites,
                "note": "no model configured — showing matching passages only"}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run test_rag.py`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add rag.py test_rag.py
git commit -m "feat: cited answers with NOT_FOUND honesty, general and legal prompts"
```

---

### Task 4: Browser UI (`rag-ladder serve`)

**Files:**
- Modify: `ragcli.py` (`PAGE`, `serve`)
- Test: `test_rag.py`

**Interfaces:**
- Produces: `ragcli.serve(rag, host, port) -> None` (blocking),
  `ragcli.make_server(rag, host, port) -> ThreadingHTTPServer` (used by the test).
- Consumes: `Rag.answer`, `Rag.search`, `Rag.doc_ids` from Tasks 1–3.

- [ ] **Step 1: Write the failing test**

```python
def test_serve_ask(tmpdir):
    import json, urllib.request, ragcli
    r = _rag()
    srv = ragcli.make_server(r, "127.0.0.1", 0)
    import threading
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    try:
        host, port = srv.server_address
        url = f"http://{host}:{port}/api/ask?q=invoice+refunds&k=3"
        with urllib.request.urlopen(url, timeout=5) as resp:
            out = json.load(resp)
        assert out["status"] == "no_model", out
        assert out["citations"][0]["id"] == "invoice"
        with urllib.request.urlopen(f"http://{host}:{port}/", timeout=5) as resp:
            assert b"<html" in resp.read().lower()
        with urllib.request.urlopen(f"http://{host}:{port}/api/ask", timeout=5) as resp:
            assert resp.status == 400
    finally:
        srv.shutdown()
        t.join(timeout=5)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run test_rag.py`
Expected: FAIL — `AttributeError: module 'ragcli' has no attribute 'make_server'`

- [ ] **Step 3: Write minimal implementation**

```python
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

PAGE = """<!doctype html>
<html><head><meta charset=utf-8><title>rag</title>
<style>body{font:16px system-ui;max-width:52em;margin:2em auto;padding:0 1em}
textarea{width:100%;font:inherit}button{font:inherit;padding:.4em 1em}
#a{white-space:pre-wrap;margin:1em 0}#c li{margin:.4em 0;color:#555}</style></head>
<body><h1>rag</h1><p id=stats></p>
<textarea id=q rows=2 placeholder="ask about your documents"></textarea>
<p><button onclick=ask()>Ask</button></p><div id=a></div><ol id=c></ol>
<script>
const esc = s => s.replace(/[&<>]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));
async function ask(){
  const q = document.getElementById('q').value;
  const r = await fetch('/api/ask?q=' + encodeURIComponent(q));
  const d = await r.json();
  document.getElementById('a').textContent = d.answer || d.note || '';
  document.getElementById('c').innerHTML = (d.citations||[]).map(
    x => `<li><b>${esc(x.id)}</b> — ${esc(x.snip)}</li>`).join('');
}
fetch('/api/stats').then(r=>r.json()).then(d=>{
  document.getElementById('stats').textContent = d.docs + ' documents indexed';
});
</script></body></html>"""


def make_server(rag, host="127.0.0.1", port=8765):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass  # ponytail: silent; wire to a logger if you run this as a service

        def _send(self, body, ctype, code=200):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _json(self, obj, code=200):
            self._send(json.dumps(obj).encode(), "application/json; charset=utf-8", code)

        def do_GET(self):
            u = urlparse(self.path)
            q = parse_qs(u.query)
            if u.path == "/":
                return self._send(PAGE.encode(), "text/html; charset=utf-8")
            if u.path == "/api/stats":
                return self._json({"docs": len(rag.doc_ids())})
            text = (q.get("q", [""])[0] or "").strip()[:2000]  # trust boundary
            k = max(1, min(20, int(q.get("k", ["5"])[0] or 5))) if q.get("k", ["5"])[0].isdigit() else 5
            if not text:
                return self._json({"error": "missing q"}, 400)
            if u.path == "/api/ask":
                return self._json(rag.answer(text, k=k))
            if u.path == "/api/search":
                return self._json({"hits": rag.search(text, k=k)})
            self._json({"error": "not found"}, 404)

    srv = ThreadingHTTPServer((host, port), Handler)
    srv.daemon_threads = True
    return srv


def serve(rag, host="127.0.0.1", port=8765):
    if host not in ("127.0.0.1", "localhost", "::1"):
        print(f"warning: binding {host} exposes the corpus to your network", file=sys.stderr)
    print(f"rag-ladder UI on http://{host}:{port}  (Ctrl-C to stop)")
    make_server(rag, host, port).serve_forever()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run test_rag.py`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add ragcli.py test_rag.py
git commit -m "feat: stdlib browser UI with cited answers"
```

---

### Task 5: CLI — init, doctor, add, stats, glossary, --json, friendly errors

**Files:**
- Modify: `ragcli.py` (`main`, `init`, `doctor`, `stats`, `glossary`)
- Test: `test_rag.py`

**Interfaces:**
- Produces: `ragcli.main(argv=None) -> int`, `ragcli.doctor() -> int`,
  `ragcli.init(preset, model, key) -> int`.
- Exit codes: `0` ok, `1` runtime failure, `2` bad input/extractor, `3` empty corpus.

- [ ] **Step 1: Write the failing test**

```python
def test_cli_add_search_json(tmpdir):
    import ragcli
    d = os.path.join(tmpdir, "docs")
    os.makedirs(d)
    with open(os.path.join(d, "pw.md"), "w") as f:
        f.write("How to reset my password. Click forgot password.")
    cfg = os.path.join(tmpdir, "rag.json")
    db = os.path.join(tmpdir, "t.db")
    assert ragcli.main(["--config", cfg, "--db", db, "add", d]) == 0
    assert ragcli.main(["--config", cfg, "--db", db, "--no-llm", "add", d]) == 0  # unchanged -> skipped
    assert ragcli.main(["--config", cfg, "--db", db, "--json", "search", "reset password"]) == 0


def test_cli_empty_corpus_hint(tmpdir):
    import ragcli
    db = os.path.join(tmpdir, "empty.db")
    assert ragcli.main(["--db", db, "--config", os.path.join(tmpdir, "c.json"), "search", "x"]) == 3


def test_cli_doctor_dead_endpoint(tmpdir, capsys):
    import ragcli
    cfg = os.path.join(tmpdir, "rag.json")
    with open(cfg, "w") as f:
        f.write('{"llm_base": "http://127.0.0.1:1"}')
    assert ragcli.main(["--config", cfg, "doctor"]) != 0  # reports unreachable, no traceback
    out = capsys.readouterr().out
    assert "unreachable" in out.lower() or "could not" in out.lower()
```

`capsys` is pytest-only. Since this suite has no pytest, capture via
`contextlib.redirect_stdout(io.StringIO())` instead:

```python
def test_cli_doctor_dead_endpoint(tmpdir):
    import io, contextlib, ragcli
    cfg = os.path.join(tmpdir, "rag.json")
    with open(cfg, "w") as f:
        f.write('{"llm_base": "http://127.0.0.1:1"}')
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        code = ragcli.main(["--config", cfg, "doctor"])
    assert code != 0
    assert "unreachable" in buf.getvalue().lower() or "could not" in buf.getvalue().lower()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run test_rag.py`
Expected: FAIL — `AttributeError: module 'ragcli' has no attribute 'main'`

- [ ] **Step 3: Write minimal implementation**

```python
def stats(rag):
    docs = rag.doc_ids()
    chars = sum(len(rag.doc(d)) for d in docs)
    vecs = rag._q("SELECT count(*) FROM vectors")[0][0]
    return {"docs": len(docs), "chars": chars, "vectors": vecs,
            "hot": len(rag.hot_tier()), "db": rag.db_path, "config": config_path()}


def doctor():
    c = CFG
    base = resolve("llm_base", None, "RAG_LLM_BASE", "http://127.0.0.1:8888")
    key = resolve("llm_api_key", None, "RAG_API_KEY", "")
    print(f"config:     {config_path()}")
    print(f"llm_base:   {base}")
    print(f"llm_model:  {resolve('llm_model', None, 'RAG_LLM_MODEL', '(endpoint default)')}")
    print(f"api key:    {'set (' + key[:4] + '…)' if key else '(none)'}")
    print(f"db:         {resolve('db', None, 'RAG_DB', 'rag.db')}")
    try:
        with urllib.request.urlopen(_api_base(base) + "/models", timeout=5) as r:
            models = [m.get("id") for m in json.load(r).get("data", [])]
        print(f"endpoint:   reachable — serves {models or '(none listed)'}")
    except Exception as e:
        print(f"endpoint:   unreachable ({e})")
        return 1
    for ext, mod in ((".pdf", "pypdf"), (".docx", "docx")):
        try:
            __import__(mod)
            print(f"{ext}:       {mod} available")
        except ImportError:
            print(f"{ext}:       needs {mod} — run: uv add {mod}")
    return 0


def init(preset, model, key):
    if preset not in PRESETS:
        print(f"unknown preset {preset!r}; choose from {', '.join(PRESETS)}")
        return 2
    cfg = dict(PRESETS[preset])
    if model:
        cfg["llm_model"] = model
    if key:
        cfg["llm_api_key"] = key
    path = config_path()
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)
        f.write("\n")
    print(f"wrote {path} (preset: {preset})")
    if key:
        os.chmod(path, 0o600)
    print(f"next: rag-ladder add <folder>  then  rag-ladder ask \"...\"")
    return 0
```

`main()`:

```python
def main(argv=None):
    try:
        return _main(argv)
    except ExtractError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    except (OSError, sqlite3.Error, urllib.error.URLError, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
```

`_main()` builds the parser: global `--config`, `--db`, `--json`, `--no-llm`,
then subcommands `init` (`--preset --model --key`), `doctor`, `add <paths>`,
`search|ask|hybrid|multi|hot` (query, `-k`), `pre [query]`, `serve`
(`--host --port`), `stats`, `glossary`, `recommend` (unchanged flags).

`add` resolves the corpus root (`--corpus`, default `.`), walks, extracts, adds:

```python
    if args.cmd == "add":
        files = walk(args.paths, corpus)
        added = skipped = failed = 0
        for p in files:
            try:
                text = extract(p)
            except ExtractError as e:
                print(f"skip: {e}", file=sys.stderr)
                failed += 1
                continue
            if rag.add(doc_id(p, corpus), text, mtime=os.path.getmtime(p)):
                added += 1
            else:
                skipped += 1
        print(f"added {added}, skipped {skipped} (unchanged), failed {failed} -> {args.db}")
        return 0
```

`search`/`ask`/`hybrid`/`multi`/`hot`/`pre` print either the `--json` blob or the
human line already in `rag.py`; every query path checks `len(rag.doc_ids()) == 0`
first and returns `3` with `corpus is empty — run: rag-ladder add <path>`.

`--json` output for `ask` is the whole `answer()` dict, so scripts get citations
and `status` without parsing prose.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run test_rag.py`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add ragcli.py test_rag.py
git commit -m "feat: CLI init/doctor/stats/glossary, --json, friendly errors, empty-corpus hint"
```

---

### Task 6: Packaging, templates, README for a layman

**Files:**
- Modify: `pyproject.toml`
- Create: `rag.example.json`, `glossary.example.txt`
- Modify: `README.md`

- [ ] **Step 1: Update packaging**

```toml
[tool.setuptools]
py-modules = ["rag", "ragcli"]

[project.scripts]
rag-ladder = "ragcli:main"

[project.optional-dependencies]
docs = ["pypdf>=4", "python-docx>=1"]
```

- [ ] **Step 2: Verify the install still resolves and the entry point works**

Run: `uv sync && uv run rag-ladder --help`
Expected: help text listing every subcommand, exit 0.

- [ ] **Step 3: Write templates**

`rag.example.json` — the documented key set with the local preset as default.
`glossary.example.txt` — a few lines showing the format, e.g.

```
# Domain terms that must never be rewritten by the LLM. One per line.
Atlas: our internal data processing framework
Mercury: our messaging system
```

- [ ] **Step 4: Rewrite README for a layman**

Lead with the four commands a non-technical user runs:

```
rag-ladder init          # choose your model backend (writes rag.json)
rag-ladder add <folder>  # index PDFs, DOCX, markdown, text
rag-ladder serve         # open the browser UI
rag-ladder doctor        # when something doesn't work
```

Then the CLI equivalents, the config key table, the env var table, the
optional-extras install, and a short "which recipe when" pointer back to
`recipes.md`. Keep the existing deliberate-corners section, updated.

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml README.md rag.example.json glossary.example.txt uv.lock
git commit -m "docs: layman quickstart, config templates, optional extractor extras"
```

---

### Task 7: Full verification

**Files:** none (verification only)

- [ ] **Step 1: Run the whole check suite**

Run: `uv run test_rag.py`
Expected: every check passes, old and new.

- [ ] **Step 2: Smoke the real CLI end-to-end**

```bash
cd /home/dinesh/repos/rag
uv run rag-ladder --config /tmp/rag-smoke.json --db /tmp/rag-smoke.db init --preset local
uv run rag-ladder --config /tmp/rag-smoke.json --db /tmp/rag-smoke.db add recipes.md
uv run rag-ladder --config /tmp/rag-smoke.json --db /tmp/rag-smoke.db stats
uv run rag-ladder --config /tmp/rag-smoke.json --db /tmp/rag-smoke.db search "hybrid rerank"
uv run rag-ladder --config /tmp/rag-smoke.json --db /tmp/rag-smoke.db ask "which recipe covers query rewriting"
uv run rag-ladder --config /tmp/rag-smoke.json --db /tmp/rag-smoke.db doctor
```

Expected: `add` reports `added 1`, `stats` reports 1 doc, `search` returns
`recipes.md`, `ask` returns an answer or `not_found` with citations — never a
fabricated answer.

- [ ] **Step 3: Smoke the web UI**

Start `rag-ladder serve` on an ephemeral port in the background, then health-check
`/api/stats` and `/api/ask?q=` with a separate call. Expected: 200 JSON. Stop it.

- [ ] **Step 4: Final commit**

```bash
git add -A
git commit -m "chore: end-to-end verification for productized rag"
```

## Self-Review

- **Spec coverage:** config (T1, T5), auth (T1), ingestion (T2), answers (T3),
  UI (T4), robustness/--json/stats/glossary (T5), templates+README+extras (T6),
  verification (T7). Every spec section maps to a task.
- **Placeholders:** none — each task carries real code or an exact command.
- **Type consistency:** `rag._cfg(env, key, default)` and `ragcli.resolve(name,
  flag, env, default)` keep the same argument order; `Rag.add` returns `bool`
  in both the interface block and the tests; `answer()` returns the same four
  keys everywhere it is asserted.
