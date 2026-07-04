# L2-M1.1 — Intermediate Pipeline Nodes

**Level:** Practitioner
**Duration:** 45 min

## Overview

A RAG pipeline is more than just retrieval and generation. Between those two stages sit several intermediate processing nodes that can significantly improve quality: query expansion (before retrieval), passage augmentation (after retrieval), passage filtering (after reranking), passage compression (before generation), and prompt construction (before generation). This lesson explores each node type, explains when it helps versus when it adds unnecessary latency, and uses AutoRAG to evaluate pass-through baselines against active implementations.

## Prerequisites

- Completed: L1-M3.1 (Configuration YAML), L1-M3.2 (Running Evaluations)
- Ollama running with `gemma4:e2b` model
- Python 3.10+, `uv` installed

## Concepts

### The Full Pipeline

In Level 1 you worked with a simplified pipeline: retrieval then generation. The full AutoRAG pipeline has eight node types arranged in order:

```
query_expansion -> retrieval -> passage_augmenter -> passage_reranker
-> passage_filter -> passage_compressor -> prompt_maker -> generator
```

Each intermediate node has a `pass_*` module that does nothing (baseline) and one or more active modules. AutoRAG evaluates pass vs active at each node and keeps the winner.

### Query Expansion (Pre-Retrieval)

Query expansion transforms the user's query before it reaches the retriever. The goal is to improve recall by generating better search terms.

- **pass_query_expansion**: baseline, no transformation.
- **HyDE (Hypothetical Document Embeddings)**: generates a hypothetical answer with an LLM, then embeds that answer for retrieval. The hypothesis is closer to relevant documents in embedding space than the original short query.
- **query_decompose**: breaks a complex query into sub-queries and retrieves for each.
- **multi_query_expansion**: generates multiple paraphrased variants and retrieves for each.

Simple factoid queries ("What is X?") rarely benefit from expansion. Complex, multi-hop, or ambiguous queries see the biggest gains.

### Passage Augmenter (Post-Retrieval)

After retrieval, the augmenter can fetch adjacent chunks to restore context lost during chunking.

- **pass_passage_augmenter**: baseline, no augmentation.
- **prev_next_augmenter**: fetches previous and/or next chunks from the corpus using `prev_id`/`next_id` metadata fields. Configurable window size and direction (`prev`, `next`, or `both`).

Augmentation helps when answers span chunk boundaries. It adds noise when adjacent chunks cover unrelated topics.

### Passage Filter (Post-Reranking)

Filters remove low-quality passages using quality thresholds. Unlike rerankers that always return a fixed `top_k`, filters remove a variable number of passages.

- **pass_passage_filter**: baseline, no filtering.
- **similarity_threshold_cutoff**: removes passages below an absolute score threshold.
- **similarity_percentile_cutoff**: removes passages below a relative percentile.
- **recency_filter**: prefers recently modified documents using `last_modified_datetime` metadata.

Filtering is valuable when returning no context is better than returning irrelevant context.

### Passage Compressor (Pre-Generation)

Compressors reduce the volume of text sent to the generator, saving tokens and reducing latency.

- **pass_compressor**: baseline, no compression.
- **tree_summarize**: hierarchical summarization of passages into a single summary.
- **refine**: iterative refinement, processing one passage at a time.
- **LongLLMLingua**: token-level compression using perplexity-based selection (2-5x compression).

With expensive LLMs, the token savings from compression often outweigh the compression cost.

### Prompt Maker (Pre-Generation)

The prompt maker assembles the final prompt from the query and retrieved passages.

- **fstring**: template-based prompt with `{query}` and `{retrieved_contents}` placeholders.
- **chat_fstring**: same but formatted as chat messages with system/user roles.
- **long_context_reorder**: reorders passages to address the "lost in the middle" problem -- places the most relevant passages at the start and end.
- **window_replacement**: used after LongLLMLingua to replace compressed tokens with original text.

## Step-by-Step

### Step 1: Prepare evaluation data

The lesson generates a 10-document Python-concepts corpus and 20 QA pairs, the same dataset used across Level 2 lessons.

```python
create_sample_data()
# Creates data/corpus.parquet (10 docs) and data/qa.parquet (20 QA pairs)
```

### Step 2: Understand the configuration

The `config.yaml` defines all eight node types with pass-through baselines and active alternatives:

- **query_expansion**: `pass_query_expansion` vs `hyde`
- **retrieval**: `bm25` (top_k=3)
- **passage_augmenter**: `pass_passage_augmenter` vs `prev_next_augmenter`
- **passage_reranker**: `pass_reranker` vs `flashrank_reranker`
- **passage_filter**: `pass_passage_filter` vs `similarity_threshold_cutoff`
- **passage_compressor**: `pass_compressor` vs `tree_summarize`
- **prompt_maker**: `fstring` vs `long_context_reorder`
- **generator**: `llama_index_llm` with Ollama gemma4:e2b

AutoRAG's greedy algorithm evaluates each node independently, selecting the best module at each stage.

### Step 3: Run the evaluation

With Ollama running, the evaluation processes all module combinations:

```python
from autorag.evaluator import Evaluator

evaluator = Evaluator(
    qa_data_path="data/qa.parquet",
    corpus_data_path="data/corpus.parquet",
    project_dir="./results",
)
evaluator.start_trial("config.yaml")
```

### Step 4: Analyze results

After evaluation, `results/0/summary.csv` shows which nodes selected pass-through vs active modules. Nodes where pass-through wins add latency without improving quality and should be removed from the production pipeline.

```python
compare_results()
# Prints winning modules with pass-through vs active labels
```

## Running the Lesson

```bash
cd tutorial/level_2/M1_advanced_optimization/1_intermediate_pipeline_nodes
uv sync
uv run python main.py
```

If Ollama is not running, the lesson prints explanations of all node types but skips the evaluation. To run the full evaluation:

```bash
ollama serve            # in one terminal
ollama pull gemma4:e2b  # pull the model
uv run python main.py   # in another terminal
```

## Expected Output

```
============================================================
L2-M1.1 — Intermediate Pipeline Nodes
============================================================

============================================================
Step 1: Create sample evaluation data
============================================================
Created corpus with 10 documents
  Columns: ['doc_id', 'contents', 'metadata']
  Sample doc_id: doc_001
  Sample preview: Variables in Python are names that refer to objects stored in memory. Unlike ...

Created 20 QA pairs
  Columns: ['qid', 'query', 'retrieval_gt', 'generation_gt']
  Sample query: What naming rules must Python variables follow?
  Sample retrieval_gt: [['doc_001']]

Saved to data/corpus.parquet and data/qa.parquet

============================================================
Step 2: Check Ollama availability
============================================================
Ollama is running with 1 model(s): ['gemma4:e2b']

============================================================
Step 3: Query expansion explained
============================================================
Query Expansion (Pre-Retrieval)
  ...

============================================================
Step 4: Passage augmenter explained
============================================================
Passage Augmenter (Post-Retrieval)
  ...

============================================================
Step 5: Passage filter explained
============================================================
Passage Filter (Post-Reranking)
  ...

============================================================
Step 6: Passage compressor explained
============================================================
Passage Compressor (Pre-Generation)
  ...

============================================================
Step 7: Prompt maker explained
============================================================
Prompt Maker (Pre-Generation)
  ...

============================================================
Step 8: Run AutoRAG evaluation
============================================================
Starting evaluation (this may take several minutes)...
Evaluation complete. Results saved to ./results/

============================================================
Step 9: Compare results
============================================================
Results loaded from results/0/summary.csv
  ...

============================================================
Done!
============================================================
```

## Key Takeaways

- The full AutoRAG pipeline has eight node types; intermediate nodes between retrieval and generation can significantly improve quality.
- Every intermediate node has a pass-through baseline -- if it wins, that processing step is unnecessary for your data.
- Query expansion (HyDE, decompose, multi-query) improves recall for complex queries but adds LLM latency for simple ones.
- Passage augmentation restores context lost during chunking; passage filtering removes low-quality results with variable thresholds.
- Passage compression saves generator tokens and can improve quality by removing noise from the context.
- The prompt maker's `long_context_reorder` strategy addresses the "lost in the middle" problem where LLMs underweight information in the center of their context window.
- AutoRAG evaluates pass vs active at each node so you do not have to guess which intermediate steps add value.

## Next Steps

In **L2-M1.2 -- Advanced Retrieval Strategies**, you will explore hybrid retrieval (BM25 + vector search), fusion methods (RRF vs Convex Combination), and neural reranking pipelines to optimize the retrieval stage itself.
