---
globs: ["tutorial/**/*.py", "tutorial/**/*.yaml"]
---

# AutoRAG Patterns and APIs

## Core Concepts by Level

### Level 1 — Essentials

#### Fundamentals (L1-M1)
- AutoRAG = AutoML for RAG pipeline optimization
- Pipeline nodes: parsing → chunking → embedding → retrieval → reranking → generation
- Greedy algorithm: selects best configuration at each node
- Data format: `qa.parquet` (QA pairs) + `corpus.parquet` (document corpus)
- CLI: `autorag evaluate`, `autorag dashboard`, `autorag deploy`

#### Evaluation Data (L1-M2)
- QA dataset columns: `qid`, `query`, `retrieval_gt`, `generation_gt`
- Corpus columns: `doc_id`, `contents`, `metadata`
- QA generation: `autorag generate_qa` from corpus documents
- Minimum ~50 QA pairs, recommended ~200+ for reliable optimization
- Train/test split: optimization set vs validation set

#### Running Experiments (L1-M3)
- Configuration YAML: defines nodes, modules, and parameters to evaluate
- Node types: `chunker`, `retriever`, `generator` (and `reranker`)
- Evaluation: `autorag evaluate --config config.yaml --qa qa.parquet --corpus corpus.parquet`
- Dashboard: `autorag dashboard --trial_dir results/`
- Deployment: `autorag deploy --trial_dir results/` (FastAPI server)
- Metrics: retrieval (precision, recall, MRR, NDCG), generation (BLEU, ROUGE, F1, semantic similarity, faithfulness)

### Level 2 — Practitioner

#### Advanced Retrieval (L2-M1.1)
- Hybrid retrieval: BM25 + vector search
- Reranking: cross-encoder, ColBERT, FlashRank
- Multi-stage: coarse retrieval → fine reranking
- Query expansion and transformation

#### Embedding Comparison (L2-M1.2)
- Multiple embedding models: nomic-embed-text, BGE, E5, GTE, OpenAI
- Dimensions, speed, and quality trade-offs
- Cost analysis

#### Custom Metrics (L2-M1.3)
- Custom retrieval/generation metrics
- LLM-as-judge metrics
- Registering custom metrics with AutoRAG

#### Custom Modules (L2-M2.1)
- Custom chunkers, retrievers, generators
- Module interface and registration

#### OpenShift AI Integration (L2-M2.2)
- Translating AutoRAG results to production deployment
- Optimal chunk size → ingestion pipeline
- Optimal embedding model → deployment config
- Optimal retrieval strategy → vector DB configuration

## Configuration YAML Pattern

```yaml
node_lines:
  - node_type: chunker
    modules:
      - module_type: token
        chunk_size: [128, 256, 512]
        chunk_overlap: [0, 32, 64]
      - module_type: sentence
  - node_type: retriever
    modules:
      - module_type: bm25
        top_k: [3, 5, 10]
      - module_type: vector
        embedding_model: ["nomic-embed-text", "bge-small-en"]
        top_k: [3, 5, 10]
  - node_type: generator
    modules:
      - module_type: llm
        llm: "ollama/gemma4:e2b"
```

## Python API Pattern

```python
from autorag.evaluator import Evaluator

evaluator = Evaluator(
    qa_data_path="qa.parquet",
    corpus_data_path="corpus.parquet",
    project_dir="./results",
)
evaluator.start_trial("config.yaml")
```

## Data Preparation Pattern

```python
import pandas as pd

# QA dataset
qa_df = pd.DataFrame({
    "qid": ["q1", "q2"],
    "query": ["What is RAG?", "How does chunking work?"],
    "retrieval_gt": [["doc1_chunk3"], ["doc2_chunk1"]],
    "generation_gt": [["RAG is..."], ["Chunking splits..."]],
})
qa_df.to_parquet("qa.parquet", index=False)

# Corpus
corpus_df = pd.DataFrame({
    "doc_id": ["doc1", "doc2"],
    "contents": ["RAG combines retrieval...", "Chunking divides documents..."],
    "metadata": [{"source": "wiki"}, {"source": "textbook"}],
})
corpus_df.to_parquet("corpus.parquet", index=False)
```

## Always Check the AutoRAG Documentation
When implementing AutoRAG features, verify the API exists and its signature by consulting:
- Documentation: https://marker-inc-korea.github.io/AutoRAG/
- Source code: https://github.com/Marker-Inc-Korea/AutoRAG
- Local source: /Users/lkellers/Projects/github/marker-inc-korea/AutoRAG

Do NOT guess at API names or parameters. Read the docs or source code if unsure.
