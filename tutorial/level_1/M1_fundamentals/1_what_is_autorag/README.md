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

The pipeline itself is a linear sequence of stages that mirror how a production RAG system works: parse documents into text, chunk that text, retrieve relevant chunks for a query, optionally rerank and filter them, assemble a prompt, and generate an answer. Each stage corresponds to a "node" in AutoRAG's configuration, and each node can host one or more candidate "modules" that AutoRAG will evaluate.

## Step-by-Step

### Step 1: Understanding the Pipeline

AutoRAG's pipeline has six stages. Each stage transforms its input into an output that feeds the next stage:

```python
stages = [
    {"name": "1. Parsing",      "desc": "Document -> text extraction"},
    {"name": "2. Chunking",     "desc": "Text -> chunks (token, sentence, semantic)"},
    {"name": "3. Retrieval",    "desc": "Query -> relevant chunks (BM25, vector, hybrid)"},
    {"name": "4. Reranking",    "desc": "Re-score retrieved chunks (cross-encoder, etc.)"},
    {"name": "5. Prompt Making", "desc": "Query + passages -> formatted prompt"},
    {"name": "6. Generation",   "desc": "Prompt -> answer (any LLM via LlamaIndex)"},
]
```

**Parsing** handles raw file formats (PDF, DOCX, HTML) and extracts plain text. **Chunking** splits that text into manageable pieces — you can chunk by a fixed token count, by sentence boundaries, or by semantic similarity. **Retrieval** finds the chunks most relevant to a user query using sparse methods (BM25), dense methods (vector embeddings), or a combination. **Reranking** takes the initial retrieval results and re-scores them with a more powerful model (cross-encoder, ColBERT, FlashRank) so the top-k list is more accurate. **Prompt Making** formats the query and selected passages into a prompt template. **Generation** sends that prompt to an LLM and returns the answer.

### Step 2: Available Modules

AutoRAG organizes pipeline components into node types and modules. A node type is a slot in the pipeline (e.g., "passage_reranker"), and modules are the concrete implementations you can plug into that slot.

The main node types are:

| Node Type | Purpose |
|-----------|---------|
| `query_expansion` | Expand or rephrase the user query |
| `lexical_retrieval` (sparse) | Keyword-based retrieval (BM25) |
| `semantic_retrieval` (dense) | Embedding-based vector search |
| `hybrid_retrieval` | Combine sparse + dense scores |
| `passage_augmenter` | Add surrounding context to passages |
| `passage_reranker` | Reorder passages by relevance |
| `passage_filter` | Drop low-quality passages |
| `passage_compressor` | Summarize or compress passages |
| `prompt_maker` | Assemble the final prompt |
| `generator` | Call the LLM to produce an answer |

Each node type has multiple modules. For example, retrieval offers `bm25`, `vectordb`, `hybrid_rrf`, and `hybrid_cc`. Reranking offers twelve options ranging from a no-op passthrough (`pass_reranker`) to specialized models like `colbert_reranker` and `flashrank_reranker`. The full catalog is printed by `main.py` and listed below:

- **Query Expansion:** `pass_query_expansion`, `query_decompose`, `hyde`, `multi_query_expansion`
- **Retrieval:** `bm25`, `vectordb`, `hybrid_rrf`, `hybrid_cc`
- **Rerankers:** `pass_reranker`, `monot5`, `tart`, `upr`, `cohere_reranker`, `rankgpt`, `jina_reranker`, `colbert_reranker`, `sentence_transformer_reranker`, `flag_embedding_reranker`, `flashrank_reranker`, `voyageai_reranker`
- **Passage Filters:** `pass_passage_filter`, `similarity_threshold_cutoff`, `similarity_percentile_cutoff`, `recency_filter`
- **Passage Compressors:** `pass_compressor`, `tree_summarize`, `refine`, `longllmlingua`
- **Prompt Makers:** `fstring`, `long_context_reorder`, `window_replacement`
- **Generators:** `llama_index_llm`, `openai_llm`, `vllm`, `vllm_api`

### Step 3: Evaluation Metrics

AutoRAG measures pipeline quality at two levels: retrieval quality (did we find the right chunks?) and generation quality (did the LLM produce a good answer?).

**Retrieval metrics** evaluate the set of retrieved passages against known relevant passages:

| Metric | What it measures |
|--------|-----------------|
| `retrieval_f1` | Harmonic mean of precision and recall |
| `retrieval_recall` | Fraction of relevant documents that were retrieved |
| `retrieval_precision` | Fraction of retrieved documents that are relevant |
| `retrieval_ndcg` | Normalized Discounted Cumulative Gain — rewards relevant docs ranked higher |
| `retrieval_mrr` | Mean Reciprocal Rank — how early the first relevant doc appears |
| `retrieval_map` | Mean Average Precision — average precision at each relevant doc |

**Generation metrics** compare the generated answer against a reference (ground-truth) answer:

| Metric | What it measures |
|--------|-----------------|
| `bleu` | N-gram overlap between generated and reference text |
| `meteor` | Alignment-based metric that accounts for synonyms and stemming |
| `rouge` | Recall-oriented n-gram overlap (ROUGE-1, ROUGE-2, ROUGE-L) |
| `sem_score` | Semantic similarity between embeddings of generated and reference text |
| `g_eval` | LLM-as-judge evaluation — uses GPT to score quality on multiple dimensions |
| `bert_score` | Contextual embedding similarity using BERT-family models |

You specify which metrics to use in your YAML config. AutoRAG uses these scores to compare modules and pick winners at each node.

### Step 4: Greedy Optimization

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
Generation (RAG). Instead of manually tuning every knob in your
RAG pipeline — chunking strategy, embedding model, retrieval
method, reranker, prompt template, LLM — you provide evaluation
data (questions + ground-truth answers + corpus) and AutoRAG
systematically explores combinations to find the best config.

Core idea:
  1. You define a YAML config listing which modules to try
  2. AutoRAG runs every combination against your eval data
  3. It measures retrieval and generation quality with metrics
  4. It returns the optimal pipeline configuration

...

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

  Rerankers:
    - pass_reranker
    - monot5
    - colbert_reranker
    - flashrank_reranker
    ...

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
- The pipeline has 6 stages: parsing, chunking, retrieval, reranking, prompt making, and generation
- Each stage offers multiple interchangeable modules (12 rerankers alone), giving AutoRAG a large search space to explore
- Both retrieval metrics (F1, recall, NDCG, MRR, MAP) and generation metrics (BLEU, ROUGE, BERTScore, G-Eval) are used to evaluate configurations
- The greedy strategy is fast — it evaluates modules node by node rather than testing every possible end-to-end combination

## Next Steps

In the next lesson (L1-M1.2), you will set up an AutoRAG project and learn about data format requirements — how to prepare your corpus, evaluation questions, and ground-truth answers so AutoRAG can optimize your pipeline.
