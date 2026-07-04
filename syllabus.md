# AutoRAG Tutorial: Syllabus

## Purpose

Learn AutoRAG — the open-source framework for automatically finding the optimal RAG pipeline for your data. AutoRAG applies AutoML-style automation to RAG optimization: you provide evaluation data, it explores combinations of chunking strategies, embedding models, retrievers, and generators, then identifies the best configuration.

## Why a Separate Tutorial?

OpenShift AI provides the infrastructure to run RAG (vLLM + pgvector/Milvus + OGX), but it doesn't tell you _which_ RAG configuration is best for your data. AutoRAG fills that gap — it systematically evaluates RAG pipeline variations and selects the optimal one. Learning AutoRAG independently lets you optimize your RAG setup before deploying the winning configuration on OpenShift AI.

> **Standalone vs Dashboard AutoRAG:** This sub-tutorial covers the **standalone AutoRAG Python tool** (`pip install autorag`) — a Tier 3 library you run locally or in a workbench. OpenShift AI 3.4+ also offers a **Tier 2 dashboard AutoRAG feature** (Gen AI Studio > AutoRAG) that runs optimization as KFP pipelines using OGX/Llama Stack. The dashboard feature requires `dashboard` + `aipipelines` + `kserve` + `llamastackoperator` components to be `Managed` in the DataScienceCluster CR. The standalone tool taught here gives you deeper understanding and full control; the dashboard feature (covered in the main syllabus L2-M1.6) provides a streamlined integrated experience.

## Target Audience

- Developers building RAG applications who want to optimize retrieval quality
- Anyone who completed OpenShift AI L2-M1 (RAG) and wants to improve their pipeline
- ML engineers who want systematic RAG evaluation rather than manual trial-and-error

## Technical Stack

- **AutoRAG**: Latest (`autorag` Python package)
- **Python**: 3.10+
- **Package Manager**: `uv`
- **LLM**: Ollama or vLLM (local development)
- **Embedding Models**: sentence-transformers, OpenAI, any HuggingFace model
- **Vector DBs**: Chroma, Milvus, Weaviate, Pinecone, Couchbase, Qdrant (configured via YAML `vectordb:` section)
- **Data Format**: Parquet (QA datasets and corpus)

## Reference Sources

- **AutoRAG GitHub**: https://github.com/Marker-Inc-Korea/AutoRAG
- **Research Paper**: https://arxiv.org/html/2410.20878v1

- **Documentation**:
https://marker-inc-korea.github.io/AutoRAG/

- **Source Code**:
https://github.com/Marker-Inc-Korea/AutoRAG

- **Local Source Code**:
/Users/lkellers/Projects/github/marker-inc-korea/AutoRAG

---

# LEVEL 1 — ESSENTIALS

*Goal: Understand AutoRAG, create evaluation data, run experiments, find the optimal RAG pipeline.*
*Estimated time: ~5-6 hours*

---

## L1-M1: AutoRAG Fundamentals

### L1-1.1 — What is AutoRAG? How It Works
**Duration:** 30 min
**Topics:**
- The problem: RAG pipelines have many knobs (chunk size, embedding model, retriever, reranker, top-k, generator) — manual tuning is slow and incomplete
- AutoRAG's approach: AutoML for RAG
  - Define which modules and configurations to evaluate
  - Provide evaluation data (questions + ground truth answers + corpus)
  - AutoRAG runs all combinations and measures quality
  - Greedy algorithm selects the best configuration at each node
- Pipeline nodes (8 node types, evaluated in order):
  1. **Query Expansion**: transform query for better retrieval (HyDE, query decompose, multi-query expansion)
  2. **Retrieval**: query → relevant chunks — three separate node types:
     - Lexical retrieval (BM25)
     - Semantic retrieval (vector search)
     - Hybrid retrieval (hybrid_rrf — Reciprocal Rank Fusion, hybrid_cc — Convex Combination)
  3. **Passage Augmenter**: add surrounding context (prev_next_augmenter)
  4. **Passage Reranker**: re-score and re-order retrieved chunks
  5. **Passage Filter**: remove irrelevant passages (similarity threshold, percentile cutoff, recency filter)
  6. **Passage Compressor**: compress passage content before generation (tree_summarize, refine, LongLLMLingua)
  7. **Prompt Maker**: construct the prompt (fstring, long_context_reorder, window_replacement)
  8. **Generator**: context + query → answer (via LlamaIndex LLM backends)
- **Pass modules**: every node has a `pass_*` variant (e.g., `pass_query_expansion`, `pass_reranker`) that tests whether skipping the node entirely performs better — fundamental to AutoRAG's optimization philosophy
- Evaluation metrics:
  - Retrieval: precision@k, recall@k, MRR, NDCG, MAP
  - Generation: BLEU, ROUGE, METEOR, F1, semantic similarity, faithfulness, G-Eval (coherence, consistency, fluency, relevance), BERTScore, SemScore
  - Passage compressor: retrieval_token_f1, retrieval_token_recall, retrieval_token_precision
  - RAGAS: context precision (no ground truth needed)
- AutoRAG vs manual RAG tuning: systematic exploration vs intuition

**Deliverables:**
- Architecture diagram: AutoRAG evaluation pipeline
- Understanding of the node-based pipeline structure

---

### L1-1.2 — Installing and Project Setup
**Duration:** 30 min
**Topics:**
- Installation: `pip install autorag`
- Project structure:
  ```
  project/
    config.yaml         # Experiment configuration
    qa.parquet           # Evaluation QA pairs
    corpus.parquet       # Document corpus
    results/             # AutoRAG outputs (auto-generated)
  ```
- Configuration YAML: defining nodes, modules, and parameters to evaluate
- Top-level `vectordb:` section: configuring the vector database backend (Chroma, Milvus, Weaviate, Pinecone, Couchbase, Qdrant)
- Data format requirements:
  - `qa.parquet`: columns `qid`, `query`, `retrieval_gt` (ground truth chunk IDs), `generation_gt` (ground truth answers)
  - `corpus.parquet`: columns `doc_id`, `contents`, `metadata`
- CLI overview: `autorag evaluate`, `autorag dashboard`, `autorag run_web` (Gradio), `autorag run_api` (FastAPI)
- LLM backends: LlamaIndex-based — OpenAI, OpenAILike, Ollama, AWS Bedrock, HuggingFace, vLLM, NVIDIA NIM

**Deliverables:**
- AutoRAG project initialized
- Understanding of data format requirements

---

## L1-M2: Evaluation Data

### L1-2.1 — Parsing and Corpus Creation
**Duration:** 45 min
**Topics:**
- Corpus format: `doc_id`, `contents`, `metadata`
- **Parsing pipeline** (YAML-driven, separate from optimization):
  - Multiple parser backends: LangChain (pdfminer, unstructured, Upstage), LlamaParse, Clova, table_hybrid_parse
  - Configure parsing via YAML: choose parser per document type
- **Chunking pipeline** (YAML-driven, separate from optimization):
  - LangChain chunkers: Token, Sentence
  - LlamaIndex chunkers: Token, Sentence, Semantic, SemanticDoubling, SentenceWindow
  - Configure chunking via YAML: chunk method, size, overlap
- Raw → Corpus flow: parse documents → chunk → produce `corpus.parquet`
- Document sources: PDF, HTML, Markdown, plain text
- Pre-processing: cleaning, deduplication, metadata extraction
- Corpus size considerations: too small → unreliable results, too large → slow evaluation

**Deliverables:**
- Parsing and chunking YAML configurations
- Corpus parquet file from sample documents with metadata

---

### L1-2.2 — Creating QA Evaluation Datasets
**Duration:** 45 min
**Topics:**
- The most critical step: evaluation data quality determines optimization quality
- Approaches to creating QA data:
  1. **Manual curation**: domain experts write Q&A pairs (highest quality, most effort)
  2. **LLM-generated**: use an LLM to generate questions from documents (fast, needs review)
  3. **AutoRAG's built-in generator**: programmatic API + CLI
- AutoRAG QA generation pipeline:
  - **Query generation types**: `factoid`, `concept_completion`, `two_hop_incremental`, custom prompts
  - **Query evolving**: `reasoning_evolve_ragas`, `conditional_evolve_ragas`, `compress_ragas`, custom evolving functions
  - **Filtering**: rule-based `dont_know_filter`, LLM-based `dont_know_filter`, `passage_dependency_filter`
  - **Answer generation**: `make_basic_gen_gt`, `make_concise_gen_gt`
  - **Sampling methods**: `random_single_hop`, `range_single_hop`
- Quality review: filtering generated QA pairs
- Dataset size: minimum ~50 QA pairs, recommended ~200+ for reliable optimization
- Splitting: train (optimization) vs test (final validation)

**Deliverables:**
- QA evaluation dataset (100+ pairs) using multiple query generation types
- Quality-reviewed and split into optimization/validation sets

---

## L1-M3: Running Experiments

### L1-3.1 — Configuration YAML
**Duration:** 1 hour
**Topics:**
- YAML structure: defining what to evaluate at each pipeline node
- Top-level `vectordb:` section — configure the vector database backend:
  ```yaml
  vectordb:
    - name: default
      db_type: chroma
      client_type: persistent
      path: ./chroma_db
  ```
- Node configuration with all 8 node types:
  ```yaml
  node_lines:
    - node_type: query_expansion
      modules:
        - module_type: pass_query_expansion
        - module_type: hyde
        - module_type: query_decompose
        - module_type: multi_query_expansion
    - node_type: retriever
      modules:
        - module_type: bm25
          top_k: [3, 5, 10]
        - module_type: vector
          embedding_model: ["nomic-embed-text", "bge-small-en"]
          top_k: [3, 5, 10]
        - module_type: hybrid_rrf
          top_k: [3, 5, 10]
    - node_type: passage_augmenter
      modules:
        - module_type: pass_passage_augmenter
        - module_type: prev_next_augmenter
    - node_type: passage_reranker
      modules:
        - module_type: pass_reranker
        - module_type: flag_embedding_reranker
        - module_type: flashrank
    - node_type: passage_filter
      modules:
        - module_type: pass_passage_filter
        - module_type: similarity_threshold_cutoff
        - module_type: recency_filter
    - node_type: passage_compressor
      modules:
        - module_type: pass_compressor
        - module_type: tree_summarize
    - node_type: prompt_maker
      modules:
        - module_type: fstring
        - module_type: long_context_reorder
    - node_type: generator
      modules:
        - module_type: llama_index_llm
          llm: "ollama/gemma4:e2b"
  ```
- **Pass modules**: every node has a `pass_*` variant — AutoRAG tests whether using the node helps or hurts
- Available modules per node:
  - Query Expansion: hyde, query_decompose, multi_query_expansion, pass
  - Retrieval: BM25, vector, hybrid_rrf, hybrid_cc
  - Passage Augmenter: prev_next_augmenter, pass
  - Passage Reranker: 17+ modules (cross-encoder, ColBERT, FlashRank, Flag Embedding, Cohere, RankGPT, Jina, MonoT5, UPR, Tart, NVIDIA, OpenVINO, VoyageAI, MixedBreadAI, etc.)
  - Passage Filter: similarity_threshold_cutoff, similarity_percentile_cutoff, recency_filter, pass
  - Passage Compressor: tree_summarize, refine, LongLLMLingua, pass
  - Prompt Maker: fstring, chat_fstring, long_context_reorder, window_replacement
  - Generator: llama_index_llm, openai_llm, vllm, vllm_api
- **Optimization strategies**: `mean` (default), `rank` (reciprocal rank), `normalize_mean`
- **Speed threshold**: `speed_threshold` parameter per node for latency-constrained optimization
- Metrics configuration: which metrics to optimize for
- Resource management: limiting concurrent evaluations

**Deliverables:**
- Configuration YAML exploring all 8 node types with pass modules
- Understanding of combinatorial explosion and how the greedy algorithm manages it

---

### L1-3.2 — Running and Monitoring Evaluations
**Duration:** 45 min
**Topics:**
- Running: `autorag evaluate --config config.yaml --qa qa.parquet --corpus corpus.parquet`
- Evaluation process: node-by-node greedy optimization
- Monitoring progress: logs, intermediate results
- Understanding results:
  - Per-node best configuration
  - Per-metric scores
  - Pipeline summary: best end-to-end configuration
- AutoRAG dashboard: `autorag dashboard --trial_dir results/`
  - Visual comparison of configurations
  - Metric distributions
  - Best pipeline visualization

**Deliverables:**
- Completed evaluation run
- Dashboard showing results
- Identified optimal RAG configuration

---

### L1-3.3 — Analyzing Results and Deploying
**Duration:** 45 min
**Topics:**
- Reading the trial report: which configuration won and why
- Metric interpretation:
  - Retrieval metrics: precision@k, recall@k, MRR, NDCG, MAP
  - Generation metrics: BLEU, ROUGE, METEOR, F1, semantic similarity, faithfulness, G-Eval (coherence, consistency, fluency, relevance), BERTScore, SemScore
  - Passage compressor metrics: retrieval_token_f1, retrieval_token_recall, retrieval_token_precision
  - RAGAS context precision (does not require retrieval ground truth)
- Understanding pass module results: when skipping a node is optimal
- **Deployment options**:
  - **FastAPI server** (`autorag run_api`): production-ready REST API
    - `/v1/run` — full query → retrieve → generate → answer
    - `/v1/retrieve` — retrieval only (no generation)
    - `/v1/stream` — streaming response with SSE
    - `/version` — version endpoint
    - NGrok tunnel support for public access
  - **Gradio web interface** (`autorag run_web`): interactive UI for testing the pipeline, with shareable links
  - **Streamlit dashboard** (`autorag dashboard`): visualization of evaluation results (not the same as the API)
- Exporting the configuration for manual implementation
- Validating on the held-out test set (not used during optimization)
- Translating results to OpenShift AI: applying the optimal chunk size, embedding model, retrieval strategy to your OGX/LangChain RAG deployment

**Deliverables:**
- Optimal pipeline deployed via both FastAPI API and Gradio web interface
- Configuration exported for OpenShift AI RAG deployment
- Validation results on test set

---

### Level 1 Summary

| Module | Lessons | Estimated Time |
|--------|---------|---------------|
| M1: Fundamentals | 2 lessons | ~1 hour |
| M2: Evaluation Data | 2 lessons | ~1.5 hours |
| M3: Running Experiments | 3 lessons | ~2.5 hours |
| **Total** | **7 lessons** | **~5-6 hours** |

---

# LEVEL 2 — PRACTITIONER

*Goal: Advanced optimization, custom modules, integration with OpenShift AI.*
*Estimated time: ~4.25-5 hours*

---

## L2-M1: Advanced Optimization

### L2-1.1 — Intermediate Pipeline Nodes
**Duration:** 45 min
**Topics:**
- Deep dive into the 4 pipeline nodes between retrieval and generation:
- **Query Expansion** (pre-retrieval):
  - HyDE (Hypothetical Document Embeddings): generate a hypothetical answer, embed it, retrieve similar chunks
  - Query decompose: break complex queries into sub-queries
  - Multi-query expansion: generate multiple query variants for broader retrieval
  - When to use vs pass: simple factoid queries rarely benefit from expansion
- **Passage Augmenter** (post-retrieval):
  - `prev_next_augmenter`: fetch adjacent chunks from the corpus for context windows
  - Configuring window size
  - When surrounding context helps vs adds noise
- **Passage Filter** (post-reranking):
  - `similarity_threshold_cutoff`: remove passages below an absolute similarity score
  - `similarity_percentile_cutoff`: remove passages below a relative percentile
  - `recency_filter`: prefer recent documents
  - Key difference from reranker: filter removes variable number of passages, reranker returns fixed top_k
- **Passage Compressor** (pre-generation):
  - `tree_summarize`: hierarchical summarization of passages
  - `refine`: iterative refinement
  - `LongLLMLingua`: token-level compression using small models
  - Token savings and latency reduction
- **Prompt Maker** (pre-generation):
  - `fstring` / `chat_fstring`: template-based prompt construction
  - `long_context_reorder`: reorder passages to put most relevant in the middle (lost-in-the-middle problem)
  - `window_replacement`: replace compressed tokens with original text in a window
  - `token_threshold` strategy parameter
- Running evaluations that include all 8 nodes

**Deliverables:**
- Configuration YAML using all 8 node types with pass modules
- Comparison: full pipeline vs simplified pipeline (pass on intermediate nodes)

---

### L2-1.2 — Advanced Retrieval Strategies
**Duration:** 45 min
**Topics:**
- Hybrid retrieval deep dive:
  - `hybrid_rrf` (Reciprocal Rank Fusion): rank-based merging, no tuning needed
  - `hybrid_cc` (Convex Combination): weighted score merging, tunable alpha parameter
  - When to use RRF vs CC
- Three separate retrieval node types: lexical, semantic, hybrid
- Reranking strategies — 17+ modules:
  - Cross-encoder, ColBERT, FlashRank (basic)
  - Flag Embedding, Sentence Transformer, Cohere, Jina, NVIDIA, VoyageAI (cloud/API-based)
  - RankGPT (LLM-based), MonoT5, UPR, Tart (specialized)
  - Ko-reranker (Korean), OpenVINO (optimized inference), MixedBreadAI
  - Time reranker (recency-weighted)
- Multi-stage retrieval: coarse retrieval → fine reranking → filtering
- Evaluating retrieval independently from generation

**Deliverables:**
- Evaluation comparing hybrid retrieval modes (RRF vs CC)
- Reranker comparison across 5+ models

---

### L2-1.3 — Embedding Model Comparison
**Duration:** 45 min
**Topics:**
- Evaluating multiple embedding models:
  - Open-source: nomic-embed-text, BGE, E5, GTE
  - Proprietary: OpenAI text-embedding-3, Cohere embed
  - vLLM embedding backend: serving embedding models via vLLM API
  - Domain-specific: fine-tuned embeddings
- VectorDB configuration: top-level `vectordb:` YAML section
  - Configuring Chroma, Milvus, Weaviate, Pinecone, Couchbase, Qdrant
  - Comparing vector DB performance for retrieval quality
- Dimensions, speed, and quality trade-offs
- Embedding model impact on retrieval quality
- Cost analysis: larger embeddings = more storage and compute

**Deliverables:**
- Comparison of 4+ embedding models on the same corpus
- VectorDB configuration tested with 2+ backends
- Cost-quality analysis

---

### L2-1.4 — Custom Evaluation Metrics
**Duration:** 30 min
**Topics:**
- Built-in metrics limitations for domain-specific tasks
- Creating custom metrics:
  - Custom retrieval metrics (domain-specific relevance)
  - Custom generation metrics (domain-specific correctness)
  - LLM-as-judge metrics
- Registering custom metrics with AutoRAG
- Optimizing for custom metrics

**Deliverables:**
- Custom domain-specific metric integrated into AutoRAG evaluation

---

## L2-M2: Integration and Production

### L2-2.1 — Custom Modules
**Duration:** 45 min
**Topics:**
- Extending AutoRAG with custom modules:
  - Custom chunkers (document-aware, semantic)
  - Custom retrievers (API-based, graph-based)
  - Custom generators (custom prompts, post-processing)
- Module interface: what to implement
- Registering custom modules in the configuration

**Deliverables:**
- Custom chunker or retriever module integrated into AutoRAG

---

### L2-2.2 — From AutoRAG to OpenShift AI
**Duration:** 45 min
**Topics:**
- Workflow: AutoRAG (local optimization) → OpenShift AI (production deployment)
- Translating AutoRAG results to OpenShift AI configuration:
  - Optimal chunk size → document ingestion pipeline (KFP)
  - Optimal embedding model → vLLM embedding deployment
  - Optimal retrieval strategy → pgvector/Milvus configuration
  - Optimal top-k/reranking → OGX or LangChain retriever config
- Running AutoRAG in a workbench on OpenShift AI (using cluster GPUs)
- Continuous optimization: re-running AutoRAG when data changes
- A/B testing: old vs new RAG configuration on OpenShift AI

**Deliverables:**
- OpenShift AI RAG deployment configured with AutoRAG-optimized parameters
- Documentation of the optimization → deployment workflow

---

### Level 2 Summary

| Module | Lessons | Estimated Time |
|--------|---------|---------------|
| M1: Advanced Optimization | 4 lessons | ~2.75 hours |
| M2: Integration & Production | 2 lessons | ~1.5 hours |
| **Total** | **6 lessons** | **~4.25-5 hours** |

---

# Complete Course Summary

| Level | Focus | Lessons | Time |
|-------|-------|---------|------|
| **Level 1 — Essentials** | QA data, experiments, optimal pipeline | 7 lessons | ~5-6 hours |
| **Level 2 — Practitioner** | Advanced strategies, custom modules, OpenShift AI | 6 lessons | ~4.25-5 hours |
| **Total** | | **13 lessons** | **~9.25-11 hours** |
