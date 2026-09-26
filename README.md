# rag — the rag.md recipe book, ladder-first

rag.md's thesis: most stacks jump to embeddings + vector DB + rerankers while
users just want the doc that says "how to reset my password". So this code
climbs the ladder and stops where the problem is already solved.

**Start here. Four commands do the whole job:**

    rag init          # choose your model backend — writes rag.json
    rag add <folder>  # index PDFs, DOCX, markdown, text, images (vision), video
    rag serve         # open the browser UI, ask questions, get cited answers
    rag doctor        # when something doesn't work

`rag ask "..."` answers from your documents with citations, and says
`NOT_FOUND_IN_CORPUS` when your documents don't contain the answer. It never
invents one — that is the whole point for legal and contract work.

## Install

    uv sync                    # core: numpy only, stdlib search
    uv sync --extra docs       # adds pypdf + python-docx, for PDF/DOCX folders
    uv tool install -e .       # then `rag` is a standalone command

No new pip dependency for images: each photo is read through your LLM's vision
capability. Video needs the `ffmpeg` system binary (no pip) — see "Image and video".

Without `--extra docs`, `rag add` still works for `.txt .md .csv .json .html`
and tells you the exact install command when it meets a PDF.

## Image and video

Photos in a listing folder are usually the actual content. `rag add` sends each
image through `llm()` as an OpenAI `image_url` content block and stores the
transcription plus a scene description for search — the citation is the file path,
so an answer quoting a photo traces back to it. No extra pip dependency; your
`llm_model` just needs vision.

- **Images** (`.jpg .jpeg .png .webp .gif .bmp .tif .tiff`) — need a **vision-capable**
  `llm_model` (e.g. `gpt-4o-mini`). A text-only model is skipped with a clear
  message and never filled with guessed text.
- **Video** (`.mp4 .mov .avi .mkv .webm .flv .m4v`) — needs `ffmpeg`. Up to
  `MAX_VIDEO_FRAMES` (12) evenly spaced frames are extracted, each described via
  vision and joined. This is the most expensive part of `rag add`: one vision
  call per frame.
- **Size** — a 20 MB per-image cap (`MAX_IMAGE_BYTES`); resize oversized files
  before indexing.

`rag doctor` reports whether `ffmpeg` is present and which `llm_model` is resolved
— confirm that model has vision before adding photo folders.

## Which model to use

`rag init` offers four backends:

| preset | endpoint | model |
|---|---|---|
| `local` | `http://127.0.0.1:8888` | whatever the endpoint serves |
| `ollama` | `http://127.0.0.1:11434` | `llama3.1:8b` |
| `openrouter` | `https://openrouter.ai/api` | `openai/gpt-4o-mini` |
| `openai` | `https://api.openai.com` | `gpt-4o-mini` |

Anything OpenAI-compatible works: set `llm_base` and `llm_model` in `rag.json`.
Cloud providers need a key — `rag init --preset openrouter --key sk-or-...`
writes it into `rag.json` with mode `0600`, or export `RAG_API_KEY`.

`rag doctor` tells you what your endpoint actually serves. If it reports a
reasoning model, recipe 2 (query rewriting) costs far more than the article's
$0.001/query — point `llm_model` at a small non-reasoning model for the real
price. `rag ask` works with no model at all: it returns matching passages and
says so instead of guessing.

## CLI reference

    rag add docs/ contracts.pdf       # index files or folders (recursive)
    rag add docs/                     # re-run: unchanged files are skipped
    rag ask "how long do refunds take?"
    rag ask --mode legal "what is the notice period?"   # quotes clauses, never infers
    rag search "invoice #12345"       # recipe 1, raw BM25 hits
    rag hybrid "alternatives to X"    # recipe 3/4
    rag multi "read csv, clean, plot"
    rag hot "invoice paid"            # recipe 5
    rag pre                           # recipe 6: build vectors
    rag pre "reset password"          # recipe 6: query them
    rag stats                         # corpus and index sizes
    rag glossary                      # write the glossary template
    rag recommend --qpd 500 --churn 2 --complaint "can't find docs"

Add `--json` to `search ask hybrid multi hot pre stats` for scripting.
Exit codes: `0` ok, `1` runtime failure, `2` bad input or missing extractor,
`3` empty corpus (the message tells you to run `rag add`).

## Config

`rag init` writes `rag.json` (see `rag.example.json`). Every key is optional.

| key | default | what it does |
|---|---|---|
| `llm_base` | `http://127.0.0.1:8888` | OpenAI-compatible endpoint |
| `llm_model` | endpoint's own | model name |
| `llm_api_key` | — | sent as `Authorization: Bearer` |
| `llm_reasoning` | `auto` | `none` sends `reasoning_effort=none` — for endpoints running a reasoning model, where a trivial rewrite otherwise burns ~20s thinking |
| `embed_base` | — | `/embeddings` endpoint; blank = hashed fallback |
| `embed_model` | `hash-bow-512` | embedding model name |
| `embed_max_chars` | `4000` | truncation cap for long documents |
| `db` | `rag.db` | SQLite file |
| `corpus` | `.` | root folder document ids are relative to |
| `glossary` | `glossary.txt` | domain terms never rewritten |
| `answer_mode` | `general` | `general` or `legal` (stricter, quotes clauses) |
| `serve_host` / `serve_port` | `127.0.0.1` / `8765` | browser UI bind |

Resolution order everywhere: **CLI flag > env var > `rag.json` > default.**
Env vars: `RAG_CONFIG`, `RAG_LLM_BASE`, `RAG_LLM_MODEL`, `RAG_API_KEY`,
`RAG_LLM_REASONING`, `RAG_EMBED_BASE`, `RAG_EMBED_MODEL`, `RAG_EMBED_MAX_CHARS`,
`RAG_DB`, `RAG_CORPUS`, `RAG_GLOSSARY`, `RAG_SERVE_HOST`, `RAG_SERVE_PORT`.
`--no-llm` forces deterministic rewriting — useful when there's no model around.

## Browser UI

`rag serve` binds `127.0.0.1:8765` and serves a read-only page: ask box, answer
pane, citations, corpus count. Routes: `GET /`, `GET /api/ask?q=&k=`,
`GET /api/search?q=&k=`, `GET /api/stats`. No write endpoint, so there's no CSRF
surface. Binding a non-local `--host` prints a warning — that exposes the corpus
to your network.

## Architecture

Two modules split at the engine/app seam:

- `rag.py` — engine: BM25 retrieval, LLM/embed clients, whole-document storage,
  answer synthesis, `recommend()`.
- `ragcli.py` — app layer: config, ingestion, CLI, browser UI.
- `test_rag.py` — 26 runnable checks, no pytest, no network. `uv run test_rag.py`.

One SQLite file. Documents are stored **whole** — no chunking, no chunk-size or
overlap decisions, no eval harness for chunks (recipe 1's whole point). A `meta`
table holds the content hash, so re-indexing skips unchanged files.

| Piece | Rung | Why |
|---|---|---|
| BM25 index | SQLite FTS5 (`bm25()`, `ORDER BY rank`) | stdlib, native, zero ops |
| LLM client | `urllib` → OpenAI-compatible `/chat/completions` | no SDK dependency |
| Embeddings | `urllib` → `/embeddings`, else hashed bag-of-words | recipes 3-6 runnable with zero setup |
| Vector math | numpy>=1.26 (declared dep) | rerank + brute-force cosine |
| ANN | brute-force scan over BLOBs | ponytail: O(n); faiss/hnsw past ~100K docs |
| UI | `http.server` + one inline page | stdlib, no framework, no build step |

Tables: `docs` (FTS5, id + full text), `vectors` (id, vec BLOB, model),
`access` (id, hit count — feeds the hot tier), `meta` (id, sha, mtime, chars).

## Recipe → code

| Recipe | Entry point | When |
|---|---|---|
| 1 BM25 | `Rag.search` | start here, always |
| 2 query rewriting | `Rag.rewrite`, `Rag.agentic_search` | vocabulary mismatch; results bad → edit the prompt, not the corpus |
| 3 hybrid | `Rag.hybrid` | "okay but not great"; 100-500ms is acceptable |
| 4 on-the-fly | `Rag.hybrid` (docs embedded per query) | churn >10%/day, or you're swapping embedding models |
| 5 hot/cold | `Rag.search_hot_cold`, `Rag.refresh_hot` | Pareto access pattern, 100K+ docs |
| 6 pre-embedding | `Rag.preembed`, `Rag.search_preembedded` | >10K q/day, <5% churn/month, ML team |
| multi-intent | `Rag.decompose`, `Rag.search_multi_intent` | one query carrying several intents; parallel → latency is max, not sum |
| cited answers | `Rag.answer` | every user-facing question; `answer_mode: legal` for contracts |

The decision tree from the article is `recommend()`; the CLI runs it.

## Deliberate corners

- Fallback embeddings are lexical (hashed BoW), **not semantic**. Point
  `embed_base` at a real model for recipes 3-6 to mean what the article says.
- Documents are embedded whole but truncated to `embed_max_chars` — a 2048-token
  model can't see a long document's tail. That's the chunking cost recipe 1
  avoids; pay it only if truncation measurably hurts ranking.
- `agentic_search`'s quality signal is lexical coverage, not an LLM judge.
- ANN is a linear scan; hot tier is recomputed on read, not on a cron.
- `decompose`'s fallback splits on "and"/commas and invents no dependencies.
- `answer()` cites the whole document it retrieved, not a page number — page
  offsets aren't tracked, because documents are stored whole.
- A reasoning model spends its budget on `reasoning_content` before emitting the
  rewrite, so recipe 2 costs far more than the article's $0.001/query. `llm()`
  retries such truncations with `reasoning_effort="none"` (llama.cpp-style
  servers); `rag doctor` warns, and presets supply a cheap model.

Bottom line from the article, encoded as the default: don't build the 5%
solution for a 60% problem. 60% of systems stop at recipe 1 + recipe 2.
