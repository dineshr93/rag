<!--
  Attribution: This is a recipe book adapted from Rafael Pierre's article
  "RAG Is Simpler Than You Think" (Lighthouse Newsletter, Jun 10, 2026).
  It is NOT the original article — it is the six recipes extracted and
  reorganized for this project. The original article remains at:
  https://www.lighthousenewsletter.com/p/rag-is-simpler-than-you-think
-->

* * *

Nowadays, most people seem to over-engineer their Retrieval Augmented Generation (RAG) stack. They jump straight to embeddings, vector databases, and reranking pipelines. Meanwhile, their users just want to find the doc that says _“How to reset my password.”_

In engineering, there’s always the right tool for the right problem. In AI Retrieval Systems it’s not different.

### Decision Factors

Before we dive into recipes, let’s establish when you should use each approach. The key factors are:

**1\. Data Freshness Requirements** \- Real-time updates (news, social media) favor approaches with easy re-indexing. Daily or weekly updates work well with hybrid approaches. A stable corpus (monthly or quarterly updates) makes pre-embedding sensible.

**2\. Corpus Characteristics** \- High churn (more than 10% changes daily) means you should avoid full pre-embedding. Stable documents work fine with pre-embedding. Long-tail distribution (90% never accessed) means on-the-fly wins.

**3\. Query Patterns** \- Keyword-heavy queries should start with full-text search. Semantic or conversational queries benefit from embeddings. Mixed patterns need hybrid approaches.

**4\. Scale & Performance** \- Less than 1000 queries per day means simple approaches are sufficient. 1K to 10K queries per day requires selective optimization. More than 10K queries per day justifies full optimization.

**5\. Team Capabilities** \- No ML expertise means stay with full-text plus query rewriting. Some ML experience makes hybrid search manageable. Having an ML team available makes advanced approaches viable.

Now, let’s look at the recipe book.

* * *

### Recipe 1: The MVP – Full-Text Search Only

#### What it is

Good old BM25. Elasticsearch. Postgres full-text search. The stuff that existed before _“embedding”_ became a verb.

#### When to use

You’re just starting out. Your users write keyword-style queries _(”pandas merge dataframe”)_. Exact matches matter _(”invoice #12345”)_. You want zero ML complexity. Your corpus has proprietary terminology _(more on this later)_.

#### Pros

Zero API costs. Fast (under 10ms). Easy to debug (you can see exactly why a document matched). Surprisingly effective (handles many use cases). **No chunking strategy needed** – works with full documents. **No evaluation complexity** – easy to test and validate. No model deprecation risk (BM25 doesn’t change).

#### Cons

Misses synonyms (”car” vs “automobile”). Fails on semantic queries (”How do I...?”). Can’t understand intent beyond keywords.

In my experience, this handles a significant portion of use cases. Don’t skip this step. You might be surprised how far you can get.

When you jump straight to embeddings, you immediately face questions like: What chunk size? (512 tokens? 1024?) What overlap? (50 tokens? 100?) Semantic chunking or fixed-size? How do I evaluate if my chunking is good?

With full-text search, you skip all of this. Your documents are your documents. Search just works.

* * *

_Thanks for reading Lighthouse AI! This post is public so feel free to share it._

[Share](https://www.lighthousenewsletter.com/p/rag-is-simpler-than-you-think?utm_source=substack&utm_medium=email&utm_content=share&action=share)

### Recipe 2: Agentic Query Rewriting

#### What it is

Use an LLM to transform messy user queries into clean keyword searches.

#### The insight

Most “semantic search” problems are actually _query formulation_ problems.

[![Flow diagram](https://substackcdn.com/image/fetch/$s_!TXCo!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F035b8f95-ee55-4356-a00e-d6e5f1e76f4e_1400x400.avif)](https://substackcdn.com/image/fetch/$s_!TXCo!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F035b8f95-ee55-4356-a00e-d6e5f1e76f4e_1400x400.avif)

#### When to use

Users ask questions conversationally. Vocabulary mismatch (users say “fix bugs”, docs say “debugging”). You have internal jargon (your framework called “Atlas”). You want flexibility to iterate quickly on query strategies.

#### Cost

_~$0.001_ per query (using GPT-4o-mini for query rewriting)

#### The magic

An LLM can remove stopwords (” _how do I_” becomes nothing). It can add synonyms (” _car_” becomes “ _car automobile vehicle_”). It can translate domain terms (” _speed up code_” becomes “ _optimize performan_ ce”). It can decompose complex queries (” _read CSV and plo_ t” becomes \[” _read CSV”, “plot data”_\]). It can learn from your glossary (via system prompt).

#### Why this is more flexible than embeddings

With embeddings, if results aren’t good, you need to adjust chunking strategy, re-embed entire corpus, run regression tests on your eval set, and hope it improved.

With query rewriting, if results aren’t good, you adjust the system prompt. That’s it. Test immediately.

#### Multi-turn agentic rewriting

Even better, you can create a loop:

```
def agentic_search(query, max_iterations=3):
    for i in range(max_iterations):
        # Rewrite query
        optimized = query_rewriter.rewrite(query, iteration=i)

        # Search
        results = bm25_search(optimized)

        # Evaluate quality
        quality = evaluate_results(results, query)

        if quality > threshold:
            return results

        # Agent learns and tries again
        query = refine_based_on_feedback(query, results, quality)

    return results
```

The agent can iterate, learn, and adapt – all without re-embedding anything.

#### Example: The Proprietary Terminology Problem

Say your company has a Python framework called “Atlas.” If you use general-purpose embeddings:

```
General embedding model (trained on internet):
“Atlas” = [vectors pointing toward: Greek mythology, maps, geography]
Your actual Atlas docs = [vectors about data processing]
Similarity score: 0.15 (terrible!)
```

The model has no idea your “ _Atlas_” exists. It falls back to what it learned in training. But with query rewriting:

```
system_prompt = “”“
  Domain-specific terms (NEVER modify these, use as exact keywords):
    - Atlas: our internal data processing framework
    - Mercury: our messaging system
    - Zeus: our auth service

    Preserve these terms exactly and optimize the rest of the query.
“”“

# User: “How do I use Atlas for batch jobs?”
# Agent: “Atlas batch jobs data processing pipeline”
# BM25: Perfect match on “Atlas” ✓
```

For proprietary terms, exact keyword matching beats semantic understanding.

Thanks for reading Lighthouse AI! Subscribe for free to receive new posts and support my work.

### Recipe 3: Hybrid Search (Sparse + Dense Reranking)

#### What it is

Use BM25 to get candidates (top 50-100), then rerank with embeddings (top 10).

#### Why this works

BM25 is fast and great at keyword matching. Embeddings are good at semantic understanding. Together, they cover each other’s weaknesses.

#### When to use

Users ask semantic questions (” _find alternatives to X”_). BM25 plus query rewriting alone isn’t cutting it (you have data proving this). You can tolerate 100-500ms latency. Your corpus is relatively stable (not changing every minute).

#### The pipeline

[![Flow diagram](https://substackcdn.com/image/fetch/$s_!DbiV!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F2237f1d5-80c6-4894-8a29-bac5bb4d2e3d_1400x400.avif)](https://substackcdn.com/image/fetch/$s_!DbiV!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F2237f1d5-80c6-4894-8a29-bac5bb4d2e3d_1400x400.avif)

#### Cost considerations

Let’s do the math with current pricing (OpenAI text-embedding-3-small at $0.02 per 1M tokens):

- Embedding 50 docs per query (avg 500 tokens each) means 50 docs × 500 tokens = 25,000 tokens

- Cost: 25,000 × $0.00002 = ~$0.0005 per query. At 1,000 queries per day × 30 days = ~$15 per month.


Actually pretty reasonable. But there’s a catch: **latency**.

Embedding 50 documents on-the-fly adds 200-500ms per query. For user-facing search, that’s noticeable. This is where the real trade-off lives – not cost, but speed.

#### Important consideration: The chunking problem returns

When you introduce embeddings, you need to decide how to chunk your documents (fixed-size? semantic? by section?). You need to determine what chunk size and overlap to use. You need to handle chunks that span important context.

This adds complexity that pure full-text search avoids.

### Recipe 4: On-The-Fly Embedding (The Fresh Data Play)

#### The insight

If your data changes frequently, why pay to re-embed everything?

#### What it is

[![Flow diagram](https://substackcdn.com/image/fetch/$s_!Yl1E!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fb8df52ca-f2e6-46be-b95d-9e3b6aa6621b_1400x400.png)](https://substackcdn.com/image/fetch/$s_!Yl1E!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fb8df52ca-f2e6-46be-b95d-9e3b6aa6621b_1400x400.png)

#### When to use

**High document churn** (more than 10% of docs updated daily). **Real-time content** (news, social media, live updates). You’re **experimenting with embedding models** (no re-indexing needed). **Data freshness is critical** (documents must be up-to-date). Small K for reranking (20-50 docs).

#### Math time

```
On-the-fly / online (1000 queries/day, 50 docs/query):
- Embedding cost: ~$15/month (ongoing)
- Storage: $0 (just store text)
- Latency: 200-500ms per query
- Freshness: Perfect (always current)
- Model switching: Easy (just change the API call)
```

#### Model deprecation

**Embedding models get deprecated.**

OpenAI deprecated text-embedding-ada-002 in favor of text-embedding-3. If you pre-embedded 10 million documents with the old model, you now need to re-embed all 10 million documents with the new model, update your vector database, run regression tests on your evaluation set, validate that quality didn’t degrade, handle the cutover period, and deal with any API changes.

#### With on-the-fly / online embedding

You literally just change one line of code. Done.

#### Downside

Latency. You’re embedding documents on every query. This is only viable if you’re okay with 200-500ms latency, K is small (reranking 20-50 docs, not 500), and your use case favors freshness over speed.

### Recipe 5: Pre-Embedding with Hot/Cold Tiers

#### What it is

Pre-embed frequently accessed documents (”hot tier”), embed rarely-accessed documents on-the-fly (”cold tier”).

#### The insight

Access patterns follow Pareto distribution. 20% of docs get 80% of traffic.

```
# Track access patterns
access_counts = Counter()

def adaptive_search(query):
    # BM25 to get candidates
    candidates = bm25_search(query, top_k=100)

    # Separate hot and cold
    hot = [d for d in candidates if d.id in hot_tier]
    cold = [d for d in candidates if d.id not in hot_tier]

    # Hot docs: use pre-computed embeddings (fast)
    hot_scores = vector_db.similarity_search(query_emb, hot)

    # Cold docs: embed on-the-fly (slower, but rare)
    cold_scores = embed_and_score(cold, query_emb)

    return merge_and_rank(hot_scores, cold_scores)

## Periodically promote frequently accessed docs to hot tier
def update_tiers_weekly():
    frequently_accessed = [doc_id for doc_id, count\
                          in access_counts.items()\
                          if count > threshold]

    # Only re-embed the new hot docs
    newly_hot = set(frequently_accessed) - set(hot_tier)
    embed_and_index(newly_hot)
```

#### When to use

Clear access patterns (some docs are accessed way more than others). Medium-to-large corpus (more than 100K documents). Mix of stable and changing content. Need good latency for common queries. Want to minimize re-embedding on model updates.

#### Benefits

Fast for 80% of queries (hit pre-embedded cache). Fresh for rarely-accessed docs. Only re-embed hot tier when switching models (20% of corpus). Adapts to changing access patterns. Best latency/cost/flexibility trade-off.

#### The model update story

When your embedding model gets deprecated:

```
Full pre-embedding: Re-embed 1M docs × $0.01 = $10,000 + downtime
Hot/cold tiers: Re-embed 200K docs × $0.01 = $2,000 + minimal downtime
On-the-fly: Change one line of code = $0 + zero downtime
```

### Recipe 6: Full Pre-Embedding (The Scale Play)

#### What it is

Embed everything upfront. Store in vector database. Search with ANN (approximate nearest neighbors).

#### When to use

Very high query volume (more than 10K queries per day). Need under 50ms latency. **Very stable corpus** (under 5% churn per month). Access pattern is broad (no long tail). You have ML team to manage infrastructure.

### Cost breakdown

```
Pre-embedding (1M docs):
- One-time embedding: 1M docs × 500 tokens × $0.00002 = $10
- Storage: 1M × 1536 dims × 4 bytes = 6GB (~$10-30/month)
- Search latency: under 50ms (blazing fast!)
- Freshness: Only as fresh as last re-index
```

#### When NOT to use

Documents change frequently (more than 10% per week). You’re experimenting with embedding models. Low query volume (under 1K queries per day). You haven’t tried simpler approaches first.

#### The model deprecation nightmare

This is where full pre-embedding hurts the most. When you need to switch models, you face **downtime** (your search is degraded while re-embedding), **compute cost** (re-embedding millions of documents), **testing burden** (full regression test suite on new embeddings), **chunking reevaluation** (maybe new model works better with different chunk sizes?), and **risk** (what if the new model is worse for your domain?).

> _This is overkill for most systems. I’ve seen teams spend months optimizing their vector database setup when query rewriting would have solved 90% of their problems._

But if you’re Pinterest, Shopify, or handling massive scale with a stable corpus, this is where you end up.

## The Multi-Intent Query Problem

We’ve been discussing single-intent queries: _“How do I merge dataframes?”_

But real users ask stuff like: **“How do I read a CSV file, clean missing data, and plot the results?”**

That’s three separate intents. Searching for this as one query is like trying to find a restaurant that serves pizza, sushi, and tacos. Good luck.

#### The Perplexity Playbook

Modern agentic RAG systems (Perplexity, ChatGPT search) handle this elegantly:

#### Query Understanding Agent

Break down the query.

```
# Input: “read CSV, clean data, plot results”

# Agent output:

{
  “query_type”: “complex”,
  “sub_queries”: [\
    “pandas read csv file”,\
    “pandas clean missing data”,\
    “matplotlib plot dataframe”\
  ],
  “dependencies”: [”read > clean > plot”]
}
```

#### Parallel Adaptive Processing

Route each sub-query optimally

```
Sub-query 1 (simple):
  “pandas read csv”
  Stopwords + lemma, then BM25
  Cost: $0, Latency: 15ms

Sub-query 2 (moderate):
  “pandas clean missing data”
  Synonym expansion, then BM25
  Cost: $0, Latency: 20ms

Sub-query 3 (complex):
  “matplotlib plot dataframe”
  LLM rewrite, then Multi-search
  Cost: $0.001, Latency: 250ms

Total (parallel): $0.002, 250ms (not 285ms!)
```

#### Synthesize

Combine results into coherent answer

```
Here’s a complete workflow:

1. Reading CSV Files
   [relevant docs from sub-query 1]

2. Cleaning Missing Data
   [relevant docs from sub-query 2]

3. Plotting Results
   [relevant docs from sub-query 3]

[Code example combining all three steps]
```

#### Why this works

Each sub-query is focused and precise, leading to better retrieval. Parallel execution means lower latency (max, not sum). Adaptive routing results in lower cost (only complex queries pay for LLM). Structured output provides better UX.

#### Cost comparison

**Without decomposition**

- LLM rewriting entire complex query: $0.005

- Embedding 50 docs: $0.025

- Total: $0.03


**With decomposition**

- Decompose: $0.001

- Sub-query 1 (simple): $0

- Sub-query 2 (simple): $0

- Sub-query 3 (complex): $0.001

- Total: $0.002


15x cheaper, better quality.

This is where agentic retrieval really shines. The agent can intelligently decide which sub-queries need expensive processing (embeddings) and which can be handled with cheap methods (simple preprocessing + BM25).

## The Decision Tree (Or: When to Use What)

Okay, you’ve read this far. You just want to know: “What should I build?”

**Start here: Do you have search at all?** If not, build BM25 first. Seriously. Stop reading and build it. If you do have search, continue.

**Measure your baseline.** Run your current search for 2-4 weeks and collect user feedback. Are users happy with the results? If yes, stop. You’re done. Go ship features. If no, continue.

**What’s the main complaint?**

If users say “Can’t find docs that clearly exist,” try query rewriting first. At $0.001 per query with zero re-indexing, it’s worth testing. Run an A/B test for 2 weeks. If you see good improvement, keep it and you’re done. If it’s not enough, continue.

If users say “Results are okay but not great,” A/B test hybrid search (sparse plus embedding rerank). Is the added latency worth it? If yes, decide on implementation. If your data changes frequently, use on-the-fly embedding. If you have clear hot docs, use hot/cold tiers. If you have a stable corpus and high scale, use full pre-embedding. If the latency isn’t worth it, optimize query rewriting further instead.

If users say “Need better semantic understanding,” use hybrid search and choose your approach based on your situation. High churn (more than 10% per day) means on-the-fly. Medium scale with clear patterns means hot/cold tiers. Massive scale with stable data means full pre-embedding.

**Key decision factors:**

Full-text with query rewriting offers perfect data freshness with low setup complexity and query latency under 50ms. Model switching is trivial, no chunking is needed, and it works for most use cases.

On-the-fly embedding provides perfect data freshness with low setup complexity but higher query latency of 200-500ms. Model switching is trivial, chunking is needed, and it’s best for high churn scenarios.

Hot/cold tiers provide mixed data freshness with medium setup complexity and query latency of 50-100ms. Model switching is easy, chunking is needed, and it offers balanced performance for varied needs.

Full pre-embedding has stale data until reindex with high setup complexity but query latency under 50ms. Model switching is painful, chunking is needed, and it’s designed for massive scale operations.

**The 80/20 rule:** 60% of systems should stop at full-text plus query rewriting. 25% need hybrid with on-the-fly or hot/cold. 10% need full pre-embedding. 5% need custom solutions.

**Bottomline: Don’t be the person who builds the 5% solution for a 60% problem.**

Thanks for reading Lighthouse Newsletter! Subscribe for free to receive new posts and support my work.

* * *