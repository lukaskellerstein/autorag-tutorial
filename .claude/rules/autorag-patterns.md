---
globs: ["tutorial/**/*.py", "tutorial/**/*.yaml"]
---

# AutoRAG Patterns and APIs

## Core Concepts by Level

### Level 1 — Essentials

#### Fundamentals (L1-M1)
- AutoRAG = AutoML for RAG pipeline optimization
- 8 optimization node types: query_expansion, lexical_retrieval, semantic_retrieval, hybrid_retrieval, passage_augmenter, passage_reranker, passage_filter, passage_compressor, prompt_maker, generator
- Parsing and chunking are separate YAML-driven pipelines (not optimization nodes)
- Pass modules: every node has a `pass_*` variant to test whether skipping the node is better
- Greedy algorithm: selects best configuration at each node
- Data format: `qa.parquet` (QA pairs) + `corpus.parquet` (document corpus)
- CLI: `autorag evaluate`, `autorag dashboard`, `autorag run_api`, `autorag run_web`

#### Evaluation Data (L1-M2)
- L1-M2.1: Parsing & Corpus Creation — YAML-driven parsing (Parser class) and chunking (Chunker class) pipelines
- L1-M2.2: Creating QA Datasets — fluent API: corpus.sample().batch_apply().batch_filter().to_parquet()
- QA dataset columns: `qid`, `query`, `retrieval_gt`, `generation_gt`
- Corpus columns: `doc_id`, `contents`, `metadata`
- Minimum ~50 QA pairs, recommended ~200+ for reliable optimization
- Train/test split: optimization set vs validation set

#### Running Experiments (L1-M3)
- Configuration YAML: top-level `vectordb:` section + `node_lines:` with nodes and modules
- Node types: `query_expansion`, `lexical_retrieval`, `semantic_retrieval`, `hybrid_retrieval`, `passage_augmenter`, `passage_reranker`, `passage_filter`, `passage_compressor`, `prompt_maker`, `generator`
- Evaluation: `autorag evaluate --config config.yaml --qa qa.parquet --corpus corpus.parquet`
- Dashboard: `autorag dashboard --trial_dir results/`
- Deployment: `autorag run_api --trial_dir results/` (FastAPI) or `autorag run_web --trial_path results/0` (Gradio)
- Metrics: retrieval (precision, recall, MRR, NDCG, MAP), generation (BLEU, ROUGE, METEOR, F1, semantic similarity, faithfulness, G-Eval, BERTScore, SemScore)

### Level 2 — Practitioner

#### Intermediate Pipeline Nodes (L2-M1.1)
- Deep dive into nodes between retrieval and generation
- Query Expansion: HyDE, query_decompose, multi_query_expansion
- Passage Augmenter: prev_next_augmenter
- Passage Filter: similarity_threshold_cutoff, similarity_percentile_cutoff, recency_filter
- Passage Compressor: tree_summarize, refine, LongLLMLingua
- Prompt Maker: fstring, chat_fstring, long_context_reorder, window_replacement

#### Advanced Retrieval (L2-M1.2)
- Hybrid retrieval: hybrid_rrf (Reciprocal Rank Fusion), hybrid_cc (Convex Combination)
- 17+ reranker modules: flashrank, flag_embedding, colbert, cohere, jina, rankgpt, etc.
- Multi-stage: coarse retrieval → fine reranking → filtering

#### Embedding Comparison (L2-M1.3)
- Multiple embedding models: nomic-embed-text, BGE, E5, GTE, OpenAI
- VectorDB configuration: Chroma, Milvus, Weaviate, Pinecone, Couchbase, Qdrant
- Dimensions, speed, and quality trade-offs

#### Custom Metrics (L2-M1.4)
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
vectordb:
  - name: default
    db_type: chroma
    client_type: persistent
    path: ./chroma_db
    embedding_model: openai
    collection_name: openai

node_lines:
- node_line_name: retrieve_node_line
  nodes:
    - node_type: query_expansion
      strategy:
        metrics: [retrieval_f1, retrieval_recall]
        retrieval_modules:
          - module_type: bm25
      modules:
        - module_type: pass_query_expansion
        - module_type: hyde
          llm: ollama
          model: gemma4:e2b
    - node_type: lexical_retrieval
      strategy:
        metrics: [retrieval_f1, retrieval_recall]
      top_k: 3
      modules:
        - module_type: bm25
    - node_type: passage_reranker
      strategy:
        metrics: [retrieval_f1, retrieval_recall]
      top_k: 3
      modules:
        - module_type: pass_reranker
        - module_type: flashrank_reranker
- node_line_name: post_retrieve_node_line
  nodes:
    - node_type: prompt_maker
      strategy:
        metrics: [bleu, rouge]
        generator_modules:
          - module_type: llama_index_llm
            llm: ollama
            model: gemma4:e2b
      modules:
        - module_type: fstring
          prompt: "Read the passages and answer the question.\nQuestion: {query}\nPassage: {retrieved_contents}\nAnswer:"
    - node_type: generator
      strategy:
        metrics:
          - metric_name: bleu
          - metric_name: rouge
      modules:
        - module_type: llama_index_llm
          llm: ollama
          model: [gemma4:e2b]
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
