# L1-M1.1 — What is AutoRAG? How It Works

**Level:** Essentials
**Duration:** 30 min

## Overview

Learn what AutoRAG is and how it applies AutoML-style automation to find the optimal RAG pipeline configuration for your data. By the end of this lesson you will understand the full pipeline, the catalog of available modules, the evaluation metrics AutoRAG uses, and its greedy optimization strategy.

## Prerequisites

- Python 3.10+
- AutoRAG installed (`uv sync` in this directory)

## Concepts

Building a RAG (Retrieval-Augmented Generation) system means making dozens of interconnected decisions: how to chunk documents, which embedding model to use, whether to apply BM25 or vector search or a hybrid, which reranker to add, how to format the prompt, and which LLM to call. Each of these choices has multiple viable options, and the best combination depends entirely on your data and use case. Tuning a RAG pipeline by hand is slow, error-prone, and rarely reaches a global optimum because the search space is too large to explore manually.

AutoRAG treats this problem the same way AutoML frameworks (AutoSklearn, FLAML, Optuna) treat model selection and hyperparameter tuning for classical ML. You supply three inputs — a corpus of documents, a set of evaluation questions, and ground-truth answers — along with a YAML configuration file that lists which modules to try at each pipeline stage. AutoRAG then systematically runs every candidate module, measures quality with retrieval and generation metrics, and outputs the best-performing pipeline configuration.

The optimization follows a greedy, node-by-node strategy. AutoRAG evaluates all candidate modules at the first node (for example, query expansion), picks the winner, locks it in, and moves to the next node (retrieval). This continues through every stage of the pipeline. The greedy approach is far cheaper than exhaustive search — which would require evaluating every possible end-to-end combination — while still producing strong results in practice.

AutoRAG separates data preparation from pipeline optimization. **Parsing** (raw files to text) and **chunking** (text to passages) are handled by separate YAML-driven pipelines that run before optimization begins. They produce the corpus of chunked passages. The optimization pipeline then operates on that corpus through eight node types: query expansion, retrieval (lexical, semantic, or hybrid), passage augmentation, reranking, filtering, compression, prompt making, and generation. Each node corresponds to a slot in AutoRAG's configuration, and each slot can host one or more candidate "modules" that AutoRAG will evaluate.

A distinctive feature is the "pass module" concept. Every optional node has a pass variant (e.g., `pass_reranker`, `pass_compressor`) that does nothing -- it forwards input unchanged. AutoRAG includes these automatically so it can test whether skipping a node entirely produces better results than any active module at that node. This is important because adding pipeline stages does not always help; sometimes a simpler pipeline wins.

## Step-by-Step

### Step 1: Understanding the Architecture

AutoRAG separates data preparation from pipeline optimization:

**Separate pipelines (run before optimization):**
- **Parsing** -- Converts raw files (PDF, DOCX, HTML) into plain text. Configured in its own YAML.
- **Chunking** -- Splits text into passages (token-based, sentence-based, or semantic). Also configured in its own YAML.

These are NOT optimization nodes. They produce the corpus of chunked passages that the optimization pipeline operates on.

**Optimization pipeline -- 8 node types:**

| # | Node Type | What it does |
|---|-----------|-------------|
| 1 | Query Expansion | Expand or rephrase the query (HyDE, multi-query, decomposition) |
| 2 | Retrieval | Find relevant passages -- lexical (BM25), semantic (vector), or hybrid |
| 3 | Passage Augmenter | Add surrounding context (e.g., adjacent chunks from the same document) |
| 4 | Passage Reranker | Reorder passages by relevance using cross-encoders, ColBERT, FlashRank |
| 5 | Passage Filter | Remove low-quality passages via similarity thresholds or recency |
| 6 | Passage Compressor | Compress or summarize passages to reduce token usage |
| 7 | Prompt Maker | Format query + passages into a prompt template |
| 8 | Generator | Send the prompt to an LLM and generate the answer |

### Step 2: Pass Modules -- Testing Whether to Skip a Node

Every optional node has a "pass" variant that does nothing -- it forwards input unchanged. AutoRAG includes these so it can test whether skipping a node produces better results than any active module:

| Pass Module | Effect |
|-------------|--------|
| `pass_query_expansion` | Skip query expansion -- use the original query |
| `pass_reranker` | Skip reranking -- keep the retriever's ordering |
| `pass_passage_augmenter` | Skip augmentation -- use passages as-is |
| `pass_passage_filter` | Skip filtering -- keep all retrieved passages |
| `pass_compressor` | Skip compression -- send full passages to the LLM |

If `pass_reranker` wins at the reranker node, it means your retriever is already returning well-ordered results and adding a reranker only hurts (or does not help enough to justify the cost).

### Step 3: Available Modules

Each node type has multiple interchangeable modules. The full catalog is printed by `main.py` and listed below:

- **Query Expansion:** `pass_query_expansion`, `query_decompose`, `hyde`, `multi_query_expansion`
- **Retrieval:** `bm25`, `vectordb`, `hybrid_rrf`, `hybrid_cc`
- **Passage Augmenters:** `pass_passage_augmenter`, `prev_next_augmenter`
- **Passage Rerankers (17+):** `pass_reranker`, `monot5`, `tart`, `upr`, `koreranker`, `cohere_reranker`, `rankgpt`, `jina_reranker`, `colbert_reranker`, `sentence_transformer_reranker`, `flag_embedding_reranker`, `flag_embedding_llm_reranker`, `time_reranker`, `openvino_reranker`, `voyageai_reranker`, `mixedbreadai_reranker`, `flashrank_reranker`
- **Passage Filters:** `pass_passage_filter`, `similarity_threshold_cutoff`, `similarity_percentile_cutoff`, `recency_filter`, `threshold_cutoff`, `percentile_cutoff`
- **Passage Compressors:** `pass_compressor`, `tree_summarize`, `refine`, `longllmlingua`
- **Prompt Makers:** `fstring`, `chat_fstring`, `long_context_reorder`, `window_replacement`
- **Generators:** `llama_index_llm`, `openai_llm`, `vllm`, `vllm_api`, `minimax_llm`

### Step 4: Evaluation Metrics

AutoRAG measures pipeline quality at multiple levels: retrieval quality (did we find the right chunks?), generation quality (did the LLM produce a good answer?), and specialized metrics for compression and context relevance.

**Retrieval metrics** evaluate the set of retrieved passages against known relevant passages:

| Metric | What it measures |
|--------|-----------------|
| `retrieval_precision` | Fraction of retrieved documents that are relevant (@k) |
| `retrieval_recall` | Fraction of relevant documents that were retrieved (@k) |
| `retrieval_f1` | Harmonic mean of precision and recall |
| `retrieval_mrr` | Mean Reciprocal Rank -- how early the first relevant doc appears |
| `retrieval_ndcg` | Normalized Discounted Cumulative Gain -- rewards relevant docs ranked higher |
| `retrieval_map` | Mean Average Precision -- average precision at each relevant doc |

**Generation metrics** compare the generated answer against a reference (ground-truth) answer:

| Metric | What it measures |
|--------|-----------------|
| `bleu` | N-gram overlap between generated and reference text |
| `rouge` | Recall-oriented n-gram overlap (ROUGE-1, ROUGE-2, ROUGE-L) |
| `meteor` | Alignment-based metric that accounts for synonyms and stemming |
| `generation_f1` | Token-level F1 between generated and reference text |
| `sem_score` | Semantic similarity via embeddings (SemScore) |
| `bert_score` | Contextual embedding similarity using BERT-family models (BERTScore) |
| `g_eval` | LLM-as-judge evaluation -- coherence, consistency, fluency, and relevance |
| `faithfulness` | Whether the answer is grounded in the retrieved passages |

**Passage compressor metrics** evaluate compression quality:

| Metric | What it measures |
|--------|-----------------|
| `retrieval_token_f1` | Token-level F1 after compression |
| `retrieval_token_recall` | Token-level recall after compression |
| `retrieval_token_precision` | Token-level precision after compression |

**RAGAS metrics** for context evaluation:

| Metric | What it measures |
|--------|-----------------|
| `ragas_context_precision` | Relevance of retrieved context (RAGAS framework) |

You specify which metrics to use in your YAML config. AutoRAG uses these scores to compare modules and pick winners at each node.

### Step 5: Greedy Optimization

AutoRAG's optimization algorithm is greedy: it processes the pipeline one node at a time, evaluates all candidate modules for that node, selects the best, and moves on.

```
Node 1 (Query Expansion):
  Try: pass, decompose, hyde     -> Winner: hyde
  
Node 2 (Retrieval):
  Try: bm25, vectordb, hybrid    -> Winner: hybrid_rrf

Node 3 (Reranking):
  Try: pass, colbert, flashrank  -> Winner: colbert_reranker

Node 4 (Prompt Making):
  Try: fstring, reorder          -> Winner: fstring

Node 5 (Generation):
  Try: openai_llm, vllm          -> Winner: openai_llm
```

At the end, AutoRAG produces an optimized YAML configuration that chains the winning modules together. You can deploy this configuration directly in production without any further tuning.

The greedy approach has a key tradeoff: it is far faster than exhaustive search (which would need to evaluate every possible combination across all nodes), but it might miss cases where a suboptimal choice at one node would have led to a better overall pipeline when paired with a specific module at a later node. In practice, greedy optimization works well because pipeline stages are relatively independent.

## Running the Lesson

```bash
cd tutorial/level_1/M1_fundamentals/1_what_is_autorag
uv sync
uv run python main.py
```

## Expected Output

```
============================================================
  AutoRAG Version
============================================================

AutoRAG is installed — version: 0.3.x

============================================================
  What is AutoRAG?
============================================================

AutoRAG is an AutoML-style framework for Retrieval-Augmented
Generation (RAG). ...

============================================================
  Pipeline Architecture
============================================================

AutoRAG separates data preparation from pipeline optimization:

  SEPARATE YAML-DRIVEN PIPELINES (run before optimization):
  ----------------------------------------------------------
  Parsing   — Converts raw documents (PDF, DOCX, HTML, etc.)
              into plain text. Configured in its own YAML.
  Chunking  — Splits extracted text into smaller pieces ...

  OPTIMIZATION PIPELINE — 8 NODE TYPES:
  ----------------------------------------------------------
  1. Query Expansion ...
  2. Retrieval ...
  ...

============================================================
  Pass Modules — Testing Whether to Skip a Node
============================================================

Every optional node has a 'pass' variant that does nothing ...

  Pass modules:
    - pass_query_expansion  ...
    - pass_reranker         ...
    - pass_passage_augmenter ...
    - pass_passage_filter   ...
    - pass_compressor       ...

============================================================
  Available Modules per Node Type
============================================================

  Query Expansion:
    - pass_query_expansion
    - query_decompose
    - hyde
    - multi_query_expansion

  Retrieval:
    - bm25
    - vectordb
    - hybrid_rrf
    - hybrid_cc

  Passage Rerankers (17+):
    - pass_reranker
    - monot5
    - colbert_reranker
    - flashrank_reranker
    ...

============================================================
  Evaluation Metrics
============================================================

  Retrieval Metrics:
    - retrieval_precision ...
    - retrieval_recall    ...
    ...
  Generation Metrics:
    - bleu ...
    - g_eval (coherence) ...
    - faithfulness ...
  Passage Compressor Metrics:
    - retrieval_token_f1 ...
  RAGAS Metrics:
    - ragas_context_precision ...

============================================================
  Greedy Optimization Strategy
============================================================

AutoRAG uses a greedy, node-by-node optimization strategy:

  1. Start at the first pipeline node (e.g., query expansion)
  2. Try every module configured for that node
  3. Evaluate each using the specified metrics
  4. Keep the BEST module for that node
  5. Move to the next node and repeat

...

============================================================
  Done
============================================================

You now have a high-level understanding of AutoRAG.
Next lesson: L1-M1.2 — Setting up an AutoRAG project.
```

## Key Takeaways

- AutoRAG automates RAG pipeline optimization using a greedy search strategy, eliminating manual trial-and-error tuning
- Parsing and chunking are separate YAML-driven pipelines -- they produce the corpus but are NOT optimization nodes
- The optimization pipeline has 8 node types: query expansion, retrieval, passage augmenter, reranker, filter, compressor, prompt maker, and generator
- Every optional node has a "pass" module that tests whether skipping the node produces better results
- Each node offers multiple interchangeable modules (17+ rerankers alone), giving AutoRAG a large search space
- Metrics span retrieval (precision@k, recall@k, MRR, NDCG, MAP), generation (BLEU, ROUGE, METEOR, BERTScore, G-Eval, faithfulness), compression, and RAGAS
- The greedy strategy is fast -- it evaluates modules node by node rather than testing every possible end-to-end combination

## Next Steps

In the next lesson (L1-M1.2), you will set up an AutoRAG project and learn about data format requirements — how to prepare your corpus, evaluation questions, and ground-truth answers so AutoRAG can optimize your pipeline.
