# rag-ladder productization — design

## Problem

`rag.py` implements recipes.md's recipes as a single-file CLI. It works (10 checks pass)
but is not operable by a non-technical user:

1. Config is 8 undocumented env vars. No file, no validation, no discovery.
   `glossary.txt` is read from CWD and never created.
2. `llm()` and `_embed_api()` send no `Authorization` header, so OpenAI /
   OpenRouter / Anthropic-compatible cloud endpoints are unusable. Only
   anonymous localhost endpoints work.
3. Ingestion is plain text only (`rag-ladder index docs/*.md`). No PDF, no DOCX, no
   directories. Legal work is PDF-heavy, so the stated target field is dead on
   arrival.
4. Output is `score  id  snippet`. No answer, no citation, no page/section, no
   "this is not in the corpus" honesty.
5. Errors surface as tracebacks. An empty corpus silently returns `[]`.

## Grounded findings (2026-09-26)

- A local OpenAI-compatible server is live at `127.0.0.1:8888`, serving exactly
  one model, `deepseek-v4.1-flash`.
- That model is a reasoning model. A 16-token chat call returned empty
  `content`, `finish_reason=length`, with the whole budget spent on
  `reasoning_content`. `rag.py` already carries a retry
  (`reasoning_effort="none"`) to survive this.
- `rag.db` contains one document, `recipes.md`. It is a working demo, not a product.

## Goal

A layman indexes a folder of legal PDFs, opens a browser, asks a question, and
gets an answer quoted from named documents — and can choose which LLM is used
without reading source code.

## Constraints

- `requires-python = ">=3.10"` → config format is JSON. `tomllib` is 3.11+.
- Core stays stdlib + numpy. PDF/DOCX extractors are optional extras.
- Env vars keep working and take precedence over the config file (back-compat:
  `test_rag.py` sets `RAG_LLM_BASE` to a dead port at import).
- No framework, no build step, no vector DB, no chunking.
- Documents stay whole — recipe 1's point.

## Config resolution order

`CLI flag  >  env var  >  rag.json  >  built-in default`

`RAG_CONFIG` selects the config file path; default `rag.json` in CWD.

## Architecture

Three files, split at the engine/app seam:

- `rag.py` — engine. `Rag`, LLM/embed clients, whole-document storage, answer
  synthesis, `recommend()`. Public surface is preserved; new methods added.
- `ragcli.py` — app layer. Config load/wizard/doctor, ingestion (file → text),
  CLI, stdlib web UI.
- `test_rag.py` — extended checks. No network, no pytest.

Rationale for the split: the engine is cohesive and already tested; the app
layer (config + ingest + CLI + HTTP) is what a layman touches. One ~1000-line
file is where 3am debugging starts.

## 1. Config

`rag.json`, stdlib `json`:

```json
{
  "llm_base": "https://openrouter.ai/api",
  "llm_model": "openai/gpt-4o-mini",
  "llm_api_key": "sk-or-...",
  "embed_base": "",
  "embed_model": "hash-bow-512",
  "embed_max_chars": 4000,
  "db": "rag.db",
  "corpus": ".",
  "glossary": "glossary.txt",
  "answer_mode": "general",
  "serve_host": "127.0.0.1",
  "serve_port": 8765
}
```

- Missing file → `{}`, defaults apply. Malformed → one friendly line naming the
  path and the JSON error. Never a traceback.
- `rag-ladder init` writes the file. Interactive menu of presets; also scriptable:
  `rag-ladder init --preset openai --model gpt-4o-mini --key sk-...`.
- Presets:

  | preset | base | default model |
  |---|---|---|
  | local | `http://127.0.0.1:8888` | (endpoint's own) |
  | ollama | `http://127.0.0.1:11434` | `llama3.1:8b` |
  | openrouter | `https://openrouter.ai/api` | `openai/gpt-4o-mini` |
  | openai | `https://api.openai.com` | `gpt-4o-mini` |

- `rag-ladder doctor` prints the resolved settings with the key masked, pings
  `<base>/models`, lists what the endpoint serves, and warns when the endpoint
  is a reasoning model or unreachable. It also reports corpus size, optional
  extractor availability, and the glossary file.
- Env vars `RAG_LLM_BASE`, `RAG_LLM_MODEL`, `RAG_EMBED_BASE`, `RAG_EMBED_MODEL`,
  `RAG_EMBED_MAX_CHARS`, `RAG_GLOSSARY`, `RAG_DB` keep working; new
  `RAG_API_KEY`.

## 2. Model access

`Authorization: Bearer <key>` added to `llm()` and `_embed_api()` when a key is
configured. Module-level `CFG` dict is populated by `ragcli` at startup;
`_cfg(env, key, default)` resolves env > CFG > default.

Default model is left unchanged so the working local setup keeps working. The
reasoning-model trap is *detected and reported* by `rag-ladder doctor` rather than
silently patched; presets supply a cheap non-reasoning default.

## 3. Ingestion

- `rag-ladder add <file|dir>...` walks directories recursively.
- Text formats read directly: `.txt .md .markdown .rst .csv .tsv .json .html
  .htm .log`.
- `.pdf` via optional `pypdf`; `.docx` via optional `python-docx`. Missing
  extractor raises `ExtractError("PDF needs pypdf — run: uv add pypdf")`, not a
  traceback.
- Document id = path relative to the corpus root, posix separators. Stable
  across machines.
- New table `meta(id, sha, mtime, chars)`. `add` skips a document whose content
  hash is unchanged and reports `added N, skipped M (unchanged)`.
- Walk skips `.git`, `.venv`, `__pycache__`, hidden entries, the DB file, and
  the config file.

## 4. Cited answers

`Rag.answer(query, k=6, mode=None)`:

- Retrieve with `rewrite` + `search` when an LLM is configured, plain `search`
  otherwise.
- Build a numbered source block from the top-k whole documents.
- Prompt (strict, one of two presets):
  - `general`: answer only from the sources, quote verbatim, cite `[n]`.
  - `legal`: same, plus never infer, never paraphrase a clause, and reply
    `NOT_FOUND_IN_CORPUS` when the sources do not contain the answer.
- Returns
  `{"answer": str|None, "status": "ok"|"not_found"|"no_model"|"empty",
    "citations": [{"n","id","snip","score"}], "note": str|None}`.
- No model → `status="no_model"` with citations, never a fabricated answer.
- Model output containing `NOT_FOUND_IN_CORPUS` → `status="not_found"`.

Honesty is the requirement: a legal tool that invents a clause is worse than
one that says it does not know.

## 5. Web UI

`rag-ladder serve` — stdlib `http.server` `ThreadingHTTPServer`, one inline HTML page,
no framework, no build step.

- `GET /` → page: query box, Ask button, answer pane, citations, corpus count.
- `GET /api/ask?q=&k=` → the `answer()` dict as JSON.
- `GET /api/search?q=&k=` → raw hits.
- `GET /api/stats` → corpus size.
- Read-only GET surface. No write endpoint → no CSRF surface.
- Binds `127.0.0.1` by default; a non-local `--host` prints a warning.
- Trust boundary validation: query capped at 2000 chars, `k` clamped to
  `1..20`, missing `q` → 400.

## 6. Robustness

- `main()` catches `ExtractError`, `OSError`, `sqlite3.Error`,
  `urllib.error.URLError`, `ValueError` → one friendly line on stderr plus an
  exit code. No broad `except Exception` swallowing real bugs.
- Empty corpus → `exit 3` with `corpus is empty — run: rag-ladder add <path>`.
- `--json` on `search`, `ask`, `hybrid`, `multi`, `hot`, `pre`, `stats`.
- `rag-ladder stats` — docs, chars, vectors, hot tier, db path, config path.
- `rag-ladder glossary` writes `glossary.example.txt` / the configured glossary path.
- Shipped templates: `rag.example.json`, `glossary.example.txt`.

## Non-goals (YAGNI)

Multi-corpus, web auth, multi-user, rerankers, vector DB, chunking, ANN index.
Add multi-corpus when two unrelated domains collide; add auth when the UI leaves
localhost; add ANN past ~100K docs.

## Testing

`test_rag.py` extended, stdlib only, no network, no pytest. New checks:

- config resolution order (flag > env > file > default); malformed config → no
  crash.
- `.pdf` without `pypdf` → `ExtractError` naming the install command.
- incremental `add`: second identical add is skipped.
- `answer()` with no model → citations present, `status="no_model"`, no answer.
- `answer()` with a stubbed LLM returning `NOT_FOUND_IN_CORPUS` →
  `status="not_found"`.
- `serve` on an ephemeral port in a thread → `GET /api/ask` returns JSON, then
  shutdown.
- CLI `add` + `search` against a temp dir and temp DB → exit 0.
- CLI `search` on an empty corpus → exit 3 with the hint.
- `doctor` against a dead endpoint → non-zero exit, no traceback.

All existing checks stay green.
