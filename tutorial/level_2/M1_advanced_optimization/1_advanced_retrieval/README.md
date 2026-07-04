# L2-M1.1 — Advanced Retrieval Strategies

**Level:** Practitioner
**Duration:** 45 min

## Overview

This lesson explores hybrid retrieval strategies that combine lexical (BM25) and semantic (vector) search, fusion methods for merging their results, and neural reranking to refine the final ranking. You will configure AutoRAG to systematically evaluate these combinations and identify the optimal multi-stage retrieval pipeline for a Python-concepts corpus.

## Prerequisites

- Completed: L1-M3.1 (Configuration YAML), L1-M3.2 (Running Evaluations)
- Ollama running with `gemma4:e2b` model
- Python 3.10+, `uv` installed

## Concepts

### Hybrid Retrieval

Lexical retrieval (BM25) and semantic retrieval (vector search) have complementary strengths. BM25 ranks documents by term frequency and inverse document frequency — it excels when the query and document share exact keywords but fails on synonyms and paraphrases. Vector search encodes text as dense embeddings and retrieves by cosine similarity, capturing semantic meaning regardless of word overlap, but it can miss exact keyword matches that BM25 would catch trivially.

Hybrid retrieval combines both approaches. The retriever runs BM25 and vector search in parallel, producing two ranked lists. A fusion step merges these lists into a single ranking. This consistently outperforms either method alone because it covers both lexical precision and semantic breadth. The question is *how* to fuse the results — and that is what AutoRAG evaluates.

### Reciprocal Rank Fusion (RRF) vs Convex Combination (CC)

**Reciprocal Rank Fusion (RRF)** merges rankings without looking at raw scores. For each document, it computes:

```
score(d) = sum( 1 / (k + rank_i(d)) )  for each retriever i
```

The parameter `k` controls how much top ranks are favored. RRF is robust because it only depends on rank positions, not on the scale or distribution of raw scores. It works well out of the box with minimal tuning.

**Convex Combination (CC)** blends the actual scores from each retriever:

```
score(d) = w * bm25_score(d) + (1 - w) * vector_score(d)
```

Because BM25 and vector scores are on different scales, they must be normalized first. AutoRAG supports three normalization methods:
- **mm (min-max)**: scales scores to [0, 1]
- **tmm (trimmed min-max)**: clips outlier scores before min-max scaling
- **z (z-score)**: standardizes to mean=0, std=1

CC gives finer control over the lexical-vs-semantic balance through the weight `w`, but is more sensitive to the choice of normalization.

### Reranking

Coarse retrieval (BM25, vector, hybrid) is fast enough to search the entire corpus but may not rank the most relevant documents at the very top. Neural rerankers take the top-k candidates from retrieval and re-score each one with a more powerful model, pushing the best matches to the top.

This lesson evaluates two rerankers against a no-op baseline:
- **Pass Reranker**: keeps the original order (baseline — measures whether reranking adds value)
- **FlashRank**: a lightweight distilled cross-encoder optimized for speed while retaining strong accuracy

Other rerankers available in AutoRAG include sentence-transformer cross-encoders (highest accuracy, slower) and ColBERT (late-interaction model, good speed-quality balance).

## Step-by-Step

### Step 1: Prepare evaluation data

The lesson generates a corpus of 10 Python-concept documents and 20 QA pairs (2 per document). This is the same dataset used in L1-M2.1, saved as Parquet files in `data/`.

```python
create_sample_data()
# Creates data/corpus.parquet (10 docs) and data/qa.parquet (20 QA pairs)
```

### Step 2: Understand the configuration

The `config.yaml` defines the evaluation search space. Key sections:

**Vector DB** — Chroma with BGE-small embeddings for the vector retrieval side of hybrid search.

**Hybrid Retrieval Node** — Runs BM25 and vector search, then fuses with either RRF or CC (with three normalization variants). AutoRAG evaluates all combinations and picks the winner by retrieval F1, recall, precision, and NDCG.

**Passage Reranker Node** — Takes the top-5 from retrieval and re-scores with either pass_reranker (baseline) or flashrank_reranker, keeping the top-3.

**Prompt Maker + Generator** — Uses the retrieved passages to generate answers via Ollama (gemma4:e2b), evaluated with BLEU and ROUGE.

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

AutoRAG's greedy algorithm evaluates each node independently — it finds the best retrieval strategy first, then the best reranker on top of that, then the best generator.

### Step 4: Analyze results

After evaluation, `results/0/summary.csv` contains the winning configuration at each node. The lesson loads this file and prints which retrieval fusion (RRF vs CC) and which reranker (pass vs FlashRank) produced the best metrics.

```python
compare_results()
# Prints the winning modules and their metric scores
```

## Running the Lesson

```bash
cd tutorial/level_2/M1_advanced_optimization/1_advanced_retrieval
uv sync
uv run python main.py
```

If Ollama is not running, the lesson will print explanations of all strategies but skip the evaluation. To run the full evaluation:

```bash
ollama serve            # in one terminal
ollama pull gemma4:e2b  # pull the model
uv run python main.py   # in another terminal
```

## Expected Output

```
============================================================
L2-M1.1 — Advanced Retrieval Strategies
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
Step 3: Retrieval strategies explained
============================================================
1. BM25 (Lexical Retrieval)
  ...

============================================================
Step 4: Reranking strategies explained
============================================================
1. Pass Reranker (No-Op Baseline)
  ...

============================================================
Step 5: Multi-stage retrieval pipeline
============================================================
Multi-Stage Retrieval Pipeline
  ...

============================================================
Step 6: Run AutoRAG evaluation
============================================================
Starting evaluation (this may take several minutes)...
Evaluation complete. Results saved to ./results/

============================================================
Step 7: Compare results
============================================================
Results loaded from results/0/summary.csv
  ...

============================================================
Done!
============================================================
```

## Key Takeaways

- Hybrid retrieval (BM25 + vector) consistently outperforms either method alone by combining lexical precision with semantic understanding.
- RRF is a robust, parameter-light fusion method that works well out of the box; CC offers finer control but requires careful normalization.
- Neural rerankers (FlashRank, cross-encoders) improve precision by re-scoring a small candidate set with a more powerful model.
- The two-stage pipeline (fast coarse retrieval then slow precise reranking) is the standard production pattern for high-quality RAG.
- AutoRAG systematically evaluates all retriever-reranker combinations so you do not have to guess which one works best for your data.

## Next Steps

In **L2-M1.2 — Embedding Model Comparison**, you will evaluate multiple embedding models (BGE, E5, GTE, nomic-embed-text) on the same corpus to understand how the embedding model choice affects retrieval quality, and analyze the tradeoffs between model size, speed, and accuracy.
