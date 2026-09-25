# rag — the rag.md recipe book, implemented ladder-first

rag.md's thesis: most stacks jump to embeddings + vector DB + rerankers while
users just want the doc that says "how to reset my password". So the code here
climbs the ladder and stops where the problem is already solved.

## Architecture

One SQLite file. Documents are stored **whole** — no chunking, no chunk-size or
overlap decisions, no eval harness for chunks (recipe 1's whole point).

| Piece | Rung | Why |
|---|---|---|
| BM25 index | SQLite FTS5 (`bm25()`, `ORDER BY rank`) | stdlib, native, zero ops |
| LLM client | `urllib` → OpenAI-compatible `/chat/completions` | no SDK dependency |
| Embeddings | `urllib` → `/embeddings`, else hashed bag-of-words | recipes 3-6 runnable with zero setup |
| Vector math | numpy>=1.26 (declared dep) | rerank + brute-force cosine |
| ANN | brute-force scan over BLOBs | ponytail: O(n); faiss/hnsw past ~100K docs |

Tables: `docs` (FTS5, id + full text), `vectors` (id, vec BLOB, model),
`access` (id, hit count — feeds the hot tier).

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

The decision tree from the article is `recommend()`; the CLI runs it.

## Use

Install once, then `rag` is a standalone command (editable, so edits to
`rag.py` apply without reinstalling):

    uv tool install -e .

    rag index docs/*.md
    rag search "invoice #12345"       # recipe 1
    rag ask "How do I use Atlas?"     # recipe 2
    rag hybrid "alternatives to X"    # recipe 3/4
    rag pre                           # recipe 6: build vectors
    rag pre "reset password"          # recipe 6: query them
    rag hot "invoice paid"            # recipe 5
    rag multi "read csv, clean, plot"
    rag recommend --qpd 500 --churn 2 --complaint "can't find docs"
    uv run test_rag.py                # runnable checks (dev venv, no pytest)

Env: `RAG_LLM_BASE` + `RAG_LLM_MODEL` (defaults to the local OpenAI-compatible
endpoint), `RAG_EMBED_BASE` + `RAG_EMBED_MODEL` + `RAG_EMBED_MAX_CHARS` (4000) for
real embeddings, `RAG_GLOSSARY` (file of domain terms that must never be
rewritten), `RAG_DB`.
`--no-llm` forces deterministic rewriting — useful when there's no model around.

## Deliberate corners

- Fallback embeddings are lexical (hashed BoW), **not semantic**. Point
  `RAG_EMBED_BASE` at a real model for recipes 3-6 to mean what the article says.
- Documents are embedded whole but truncated to `RAG_EMBED_MAX_CHARS` — a
  2048-token model can't see a long document's tail. That's the chunking cost
  recipe 1 avoids; pay it only if truncation measurably hurts ranking.
- `agentic_search`'s quality signal is lexical coverage, not an LLM judge.
- ANN is a linear scan; hot tier is recomputed on read, not on a cron.
- `decompose`'s fallback splits on "and"/commas and invents no dependencies.
- A reasoning model spends its budget on `reasoning_content` before emitting the
  rewrite, so recipe 2 costs far more than the article's $0.001/query. `llm()`
  retries such truncations with `reasoning_effort="none"` (llama.cpp-style
  servers); point `RAG_LLM_MODEL` at a small non-reasoning model for the real price.

Bottom line from the article, encoded as the default: don't build the 5%
solution for a 60% problem. 60% of systems stop at recipe 1 + recipe 2.
