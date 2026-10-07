# rag-ladder — the 6 RAG recipe book, ladder-first

The article's thesis: most stacks jump to embeddings + vector DB + rerankers while
users just want the doc that says "how to reset my password". So this code
climbs the ladder and stops where the problem is already solved.

**Inspiration.** This code implements the recipe book from Rafael Pierre's
article ["RAG Is Simpler Than You Think"](https://www.lighthousenewsletter.com/p/rag-is-simpler-than-you-think)
(Lighthouse Newsletter, Jun 10, 2026). The six recipes — BM25, agentic query
rewriting, hybrid, on-the-fly, hot/cold, and full pre-embedding — and the
decision tree that picks between them come directly from that article. The
guiding principle: *don't build the 5% solution for a 60% problem.*

**Start here. Four commands do the whole job:**

    rag-ladder init          # choose your model backend — writes rag.json
    rag-ladder add <folder>  # index PDFs, DOCX, markdown, text, images (vision), video
    rag-ladder serve         # open the browser UI, ask questions, get cited answers
    rag-ladder doctor        # when something doesn't work

`rag-ladder ask "..."` answers from your documents with citations, and says
`NOT_FOUND_IN_CORPUS` when your documents don't contain the answer. It never
invents one — that is the whole point for legal and contract work.

## Install

    uv sync                    # core: numpy only, stdlib search
    uv sync --extra docs       # adds pypdf + python-docx, for PDF/DOCX folders
    uv tool install -e .       # then `rag-ladder` is a standalone command

No new pip dependency for images: each photo is read through your LLM's vision
capability. Video needs the `ffmpeg` system binary (no pip) — see "Image and video".

Without `--extra docs`, `rag-ladder add` still works for `.txt .md .csv .json .html`
and tells you the exact install command when it meets a PDF.

## Image and video

Photos in a listing folder are usually the actual content. `rag-ladder add` sends each
image through `llm()` as an OpenAI `image_url` content block and stores the
transcription plus a scene description for search — the citation is the file path,
so an answer quoting a photo traces back to it. No extra pip dependency; your
`llm_model` just needs vision.

- **Images** (`.jpg .jpeg .png .webp .gif .bmp .tif .tiff`) — need a **vision-capable**
  `llm_model` (e.g. `gpt-4o-mini`). A text-only model is skipped with a clear
  message and never filled with guessed text.
- **Video** (`.mp4 .mov .avi .mkv .webm .flv .m4v`) — needs `ffmpeg`. Up to
  `MAX_VIDEO_FRAMES` (12) evenly spaced frames are extracted, each described via
  vision and joined. This is the most expensive part of `rag-ladder add`: one vision
  call per frame.
- **Size** — a 20 MB per-image cap (`MAX_IMAGE_BYTES`); resize oversized files
  before indexing.

`rag-ladder doctor` reports whether `ffmpeg` is present and which `llm_model` is resolved
— confirm that model has vision before adding photo folders.

## Which model to use

`rag-ladder init` offers four backends:

| preset | endpoint | model |
|---|---|---|
| `local` | `http://127.0.0.1:8888` | whatever the endpoint serves |
| `ollama` | `http://127.0.0.1:11434` | `llama3.1:8b` |
| `openrouter` | `https://openrouter.ai/api` | `openai/gpt-4o-mini` |
| `openai` | `https://api.openai.com` | `gpt-4o-mini` |

Anything OpenAI-compatible works: set `llm_base` and `llm_model` in `rag.json`.
Cloud providers need a key — `rag-ladder init --preset openrouter --key sk-or-...`
writes it into `rag.json` with mode `0600`, or export `RAG_API_KEY`.

`rag-ladder doctor` tells you what your endpoint actually serves. If it reports a
reasoning model, recipe 2 (query rewriting) costs far more than the article's
$0.001/query — point `llm_model` at a small non-reasoning model for the real
price. `rag-ladder ask` works with no model at all: it returns matching passages and
says so instead of guessing.

## CLI reference

    rag-ladder add docs/ contracts.pdf       # index files or folders (recursive)
    rag-ladder add docs/                     # re-run: unchanged files are skipped
    rag-ladder ask "how long do refunds take?"
    rag-ladder ask --mode legal "what is the notice period?"   # quotes clauses, never infers
    rag-ladder ask --pre "when was the termination notified?"  # vector retrieval + cited answer
    rag-ladder ask --pre --min-score 0.3 "query"   # skip docs scoring below 0.3
    rag-ladder search "invoice #12345"       # recipe 1, raw BM25 hits
    rag-ladder hybrid "alternatives to X"    # recipe 3/4
    rag-ladder multi "read csv, clean, plot"
    rag-ladder hot "invoice paid"            # recipe 5
    rag-ladder pre                           # recipe 6: build vectors
    rag-ladder pre "reset password"          # recipe 6: query them
    rag-ladder stats                         # corpus and index sizes
    rag-ladder glossary                      # write the glossary template
    rag-ladder recommend --qpd 500 --churn 2 --complaint "can't find docs"

Add `--json` to `search ask hybrid multi hot pre stats` for scripting.
Exit codes: `0` ok, `1` runtime failure, `2` bad input or missing extractor,
`3` empty corpus (the message tells you to run `rag-ladder add`).

## Decision tree

`rag-ladder recommend` picks the next recipe for you. It needs two numbers and a
complaint:

    rag-ladder recommend --qpd 500 --churn 2 --complaint "can't find docs"

| Flag | Meaning | How to estimate |
|---|---|---|
| `--qpd` | **Queries per day** — how many searches your users make | Count `rag-ladder ask`/`rag-ladder search` calls in a day, or estimate from traffic |
| `--churn` | **Corpus churn %/day** — how many documents change daily | Adding 50 docs to a 500-doc corpus = `--churn 10` |
| `--complaint` | What's wrong with the BM25 results | "can't find", "not great", "okay" |

The tree's answer tells you the next rung to build:

| Complaint | Condition | Next recipe |
|---|---|---|
| "can't find" | vocabulary mismatch | **2** — LLM query rewriting |
| "not great" / "okay" | low latency OK, low churn | **3** — hybrid BM25 + embedding rerank |
| "not great" | churn >10%/day | **4** — embed candidates on the fly |
| "not great" | Pareto access (hot 20%) | **5** — hot/cold tiers |
| "not great" | >100K docs, >10K qpd, ML team | **6** — full pre-embedding |
| (satisfied) | users happy | **stop** — ship features |

Run it after `rag-ladder add`, before you write any code. It returns `None` when
recipe 1 + 2 are enough — that's the target for ~60% of systems.

## Config

`rag-ladder init` writes `rag.json` (see `rag.example.json`). Every key is optional.

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
| `answer_max_chars` | auto | optional cap: how much of each retrieved document the answer is built from — the regions matching the question, not the head; costs `k ×` this in the prompt. Default is the biggest document in the corpus, so nothing truncates until you cap it |
| `serve_host` / `serve_port` | `127.0.0.1` / `8765` | browser UI bind |

Resolution order everywhere: **CLI flag > env var > `rag.json` > default.**
Env vars: `RAG_CONFIG`, `RAG_LLM_BASE`, `RAG_LLM_MODEL`, `RAG_API_KEY`,
`RAG_LLM_REASONING`, `RAG_EMBED_BASE`, `RAG_EMBED_MODEL`, `RAG_EMBED_MAX_CHARS`,
`RAG_ANSWER_MAX_CHARS`, `RAG_DB`, `RAG_CORPUS`, `RAG_GLOSSARY`, `RAG_SERVE_HOST`,
`RAG_SERVE_PORT`.
`--no-llm` forces deterministic rewriting — useful when there's no model around.

## Browser UI

`rag-ladder serve` binds `127.0.0.1:8765` and serves a read-only page: ask box, answer
pane, citations, corpus count. Routes: `GET /`, `GET /api/ask?q=&k=`,
`GET /api/search?q=&k=`, `GET /api/stats`. No write endpoint, so there's no CSRF
surface. Binding a non-local `--host` prints a warning — that exposes the corpus
to your network.

## Architecture

Two modules split at the engine/app seam:

- `rag.py` — engine: BM25 retrieval, LLM/embed clients, whole-document storage,
  answer synthesis, `recommend()`.
- `ragcli.py` — app layer: config, ingestion, CLI, browser UI.
- `test_rag.py` — runnable checks, no pytest, no network. `uv run test_rag.py`.

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
| 6 pre-embedding | `Rag.preembed`, `Rag.search_preembedded`, `Rag.answer(pre=True)` | >10K q/day, <5% churn/month, ML team |
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
- `answer()` sends the regions of a long document that match the question, up to
  `answer_max_chars` per document — never its first screenful, which for a 60-page
  PDF is its cover page. Each citation reports `sent`/`chars` and truncated documents
  are listed under `truncated`. Windows are chosen lexically: a query whose words the
  corpus never uses still finds nothing, which is recipe 1's advice (`search` the
  exact term) and recipe 2's job (rewriting), not windowing's.
- `answer()` cites the whole document it retrieved, not a page number — page
  offsets aren't tracked, because documents are stored whole.
- A reasoning model spends its budget on `reasoning_content` before emitting the
  rewrite, so recipe 2 costs far more than the article's $0.001/query. `llm()`
  retries such truncations with `reasoning_effort="none"` (llama.cpp-style
  servers); `rag-ladder doctor` warns, and presets supply a cheap model.

Bottom line from the article, encoded as the default: don't build the 5%
solution for a 60% problem. 60% of systems stop at recipe 1 + recipe 2.
