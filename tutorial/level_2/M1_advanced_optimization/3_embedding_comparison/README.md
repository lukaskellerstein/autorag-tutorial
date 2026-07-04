# L2-M1.3 — Embedding Model Comparison

**Level:** Practitioner
**Duration:** 45 min

## Overview

The embedding model is the foundation of any semantic search pipeline -- it determines how well your system understands the meaning of queries and documents. In this lesson you will use AutoRAG to systematically compare two embedding models (BAAI/bge-small-en-v1.5 and all-mpnet-base-v2) on the same corpus and learn how to reason about the tradeoffs between dimensionality, model size, and retrieval quality.

## Prerequisites

- Completed: L2-M1.2 (Advanced Retrieval Strategies)
- Ollama running with `gemma4:e2b` model pulled (`ollama pull gemma4:e2b`)
- Python 3.10+, `uv` installed

## Concepts

### Why Embedding Model Choice Matters

Every RAG pipeline begins with encoding text into dense vectors. The embedding model controls which documents are retrieved for a given query, so a poor embedding model means the generator never sees the right context -- no matter how powerful the LLM is. Two documents that are semantically similar should land close together in vector space, and the embedding model's architecture and training data determine how well it achieves this.

Choosing an embedding model is not a one-size-fits-all decision. Smaller models are faster and cheaper to run but may miss subtle semantic distinctions. Larger models capture more nuance but cost more to store and query. Domain-specific fine-tuned models can dramatically outperform general-purpose ones on specialized corpora, but they require additional effort to train and maintain.

### Dimensions vs Quality

Embedding dimensionality is the number of floating-point values in each vector. A 384-dimensional model encodes each text chunk as 384 numbers; a 768-dimensional model uses twice as many. Higher dimensions give the model more room to represent fine-grained semantic differences, which can improve retrieval precision. However, higher dimensions also mean:

- **More storage**: each vector is larger on disk and in memory.
- **Slower search**: distance calculations take longer as dimensionality grows.
- **Diminishing returns**: beyond a certain point, extra dimensions add noise rather than signal.

The right dimensionality depends on your corpus size, query complexity, and infrastructure budget. AutoRAG lets you measure the actual impact on your data instead of guessing.

### Models Compared in This Lesson

| Model | Dimensions | Size | Characteristics |
|---|---|---|---|
| BAAI/bge-small-en-v1.5 | 384 | ~130 MB | Lightweight, fast indexing, English-optimized. Part of the BGE family trained with contrastive learning on large English corpora. Good baseline for prototyping. |
| all-mpnet-base-v2 | 768 | ~420 MB | Based on Microsoft MPNet, fine-tuned by sentence-transformers on over 1 billion training pairs. Higher quality embeddings at the cost of larger vectors and slower encoding. |

### vLLM as Embedding Backend

For large embedding models or GPU-accelerated scenarios, vLLM can serve embeddings through its OpenAI-compatible API. This is useful when you need to run models with over 1 billion parameters (e.g., e5-mistral-7b-instruct) or need high-throughput indexing of large corpora. For small models (<500M params) like bge-small or MiniLM, local sentence-transformers is sufficient.

### VectorDB Configuration

AutoRAG supports multiple vector databases through the `vectordb` section in `config.yaml`. Each entry specifies a database type (Chroma, Milvus, Weaviate, Pinecone, Couchbase, Qdrant), an embedding model, and connection parameters. Different vectordb entries can use different embedding models, allowing AutoRAG to compare embeddings by creating separate indices.

## Step-by-Step

### Step 1: Prepare evaluation data

The lesson generates a 10-document corpus about Python programming concepts and 20 QA pairs (two per document). This is the same corpus format used throughout the tutorial -- parquet files with `doc_id`, `contents`, and `metadata` columns for the corpus, and `qid`, `query`, `retrieval_gt`, `generation_gt` columns for the QA set.

```python
create_sample_data()
# -> data/corpus.parquet (10 documents)
# -> data/qa.parquet (20 QA pairs)
```

### Step 2: Understand the configuration

The `config.yaml` defines two vector databases, each using a different embedding model:

```yaml
vectordb:
  - name: bge_small
    embedding_model: huggingface_baai_bge_small
  - name: mpnet
    embedding_model: huggingface_all_mpnet_base_v2
```

AutoRAG evaluates both embedding models by indexing the same corpus into each vector database and then running the same queries against both. The retrieval node tests each vectordb module and reports metrics (F1, recall, precision, NDCG, MRR) for each.

### Step 3: Run the evaluation

When Ollama is available, AutoRAG runs the full trial -- embedding the corpus with both models, retrieving against both indices, generating answers, and computing metrics. The greedy evaluation algorithm selects the best embedding at the retrieval node and carries the winner forward to the generation stage.

```python
from autorag.evaluator import Evaluator

evaluator = Evaluator(
    qa_data_path="data/qa.parquet",
    corpus_data_path="data/corpus.parquet",
    project_dir="./results",
)
evaluator.start_trial("config.yaml")
```

### Step 4: Compare results

After the evaluation completes, `results/0/summary.csv` contains per-module metrics. The `compare_embeddings()` function loads this file and shows which embedding model achieved higher retrieval scores. Look at `retrieval_f1` and `retrieval_ndcg` as the primary indicators -- they balance precision and recall and account for ranking quality.

## Running the Lesson

```bash
cd tutorial/level_2/M1_advanced_optimization/3_embedding_comparison
uv sync
uv run python main.py
```

## Expected Output

```
============================================================
L2-M1.3 — Embedding Model Comparison
============================================================

============================================================
Step 1: Prepare evaluation data
============================================================
Created corpus with 10 documents
Created 20 QA pairs (2 per document)
Saved to: data/corpus.parquet (4,521 bytes)
Saved to: data/qa.parquet (6,138 bytes)

============================================================
Step 2: Check Ollama availability
============================================================
Ollama is running.
Available models: ['gemma4:e2b']

============================================================
Step 3: Embedding models under comparison
============================================================
Embedding models compared in this lesson:

| Model                   | Dimensions  | Size    | Notes                                 |
|------------------------|-----------|--------|--------------------------------------|
| BAAI/bge-small-en-v1.5 | 384       | ~130MB | Lightweight, good for English        |
| all-mpnet-base-v2      | 768       | ~420MB | Higher quality, larger dimensions    |

...

============================================================
Step 9: Recommendations
============================================================
Recommendations:

For prototyping and development:
  Use bge-small-en-v1.5 (384 dims, ~130MB)
  ...

============================================================
Done! Review the results to decide which embedding model
works best for your use case.
============================================================
```

## Key Takeaways

- The embedding model is the single most impactful component in a RAG retrieval pipeline -- it determines which documents the generator sees.
- Higher embedding dimensions (768 vs 384) generally improve retrieval quality but increase storage and compute costs.
- AutoRAG makes embedding comparison systematic: same corpus, same queries, same metrics, different models.
- Always benchmark on your own data -- MTEB leaderboard rankings do not always predict performance on domain-specific corpora.
- Start with a lightweight model for prototyping, then upgrade to a larger model and verify the improvement with AutoRAG before committing to production.

## Next Steps

In **L2-M1.4 -- Custom Evaluation Metrics**, you will learn how to define domain-specific metrics beyond the built-in BLEU, ROUGE, and retrieval scores, and register them with AutoRAG to optimize for what matters most in your application.
