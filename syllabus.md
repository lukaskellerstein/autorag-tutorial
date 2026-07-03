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
- **Vector DBs**: FAISS, Chroma, Qdrant (AutoRAG supports many)
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
*Estimated time: ~6-8 hours*

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
- Pipeline nodes:
  1. **Parsing**: document → text extraction
  2. **Chunking**: text → chunks (fixed, semantic, recursive, sentence)
  3. **Embedding**: chunks → vectors
  4. **Retrieval**: query → relevant chunks (BM25, vector search, hybrid)
  5. **Reranking**: re-score retrieved chunks
  6. **Generation**: context + query → answer
- Evaluation metrics: retrieval (precision, recall, MRR, NDCG), generation (BLEU, ROUGE, F1, semantic similarity, faithfulness)
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
- Data format requirements:
  - `qa.parquet`: columns `qid`, `query`, `retrieval_gt` (ground truth chunk IDs), `generation_gt` (ground truth answers)
  - `corpus.parquet`: columns `doc_id`, `contents`, `metadata`
- CLI overview: `autorag evaluate`, `autorag dashboard`, `autorag deploy`

**Deliverables:**
- AutoRAG project initialized
- Understanding of data format requirements

---

## L1-M2: Evaluation Data

### L1-2.1 — Creating QA Evaluation Datasets
**Duration:** 1 hour
**Topics:**
- The most critical step: evaluation data quality determines optimization quality
- Approaches to creating QA data:
  1. **Manual curation**: domain experts write Q&A pairs (highest quality, most effort)
  2. **LLM-generated**: use an LLM to generate questions from documents (fast, needs review)
  3. **AutoRAG's built-in generator**: `autorag generate_qa` from your corpus
- AutoRAG QA generation:
  - Provide corpus documents
  - LLM generates diverse questions with ground truth answers
  - Automatic retrieval ground truth linking (which chunks answer which question)
- Quality review: filtering generated QA pairs
- Dataset size: minimum ~50 QA pairs, recommended ~200+ for reliable optimization
- Splitting: train (optimization) vs test (final validation)

**Deliverables:**
- QA evaluation dataset (100+ pairs) from a sample document corpus
- Quality-reviewed and split into optimization/validation sets

---

### L1-2.2 — Preparing the Corpus
**Duration:** 30 min
**Topics:**
- Corpus format: `doc_id`, `contents`, `metadata`
- Document sources: PDF, HTML, Markdown, plain text
- Pre-processing: cleaning, deduplication, metadata extraction
- Chunking happens during evaluation — corpus contains raw documents
- Corpus size considerations: too small → unreliable results, too large → slow evaluation
- Multiple corpus versions for different experiments

**Deliverables:**
- Corpus parquet file from sample documents
- Metadata properly structured

---

## L1-M3: Running Experiments

### L1-3.1 — Configuration YAML
**Duration:** 45 min
**Topics:**
- YAML structure: defining what to evaluate at each pipeline node
- Node configuration:
  ```yaml
  node_lines:
    - node_type: chunker
      modules:
        - module_type: token
          chunk_size: [128, 256, 512]
          chunk_overlap: [0, 32, 64]
        - module_type: sentence
          ...
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
- Available modules per node:
  - Chunker: token, sentence, semantic, recursive
  - Retrieval: BM25, vector, hybrid
  - Reranker: cross-encoder, colbert, flashrank
  - Generator: any LLM via litellm
- Metrics configuration: which metrics to optimize for
- Resource management: limiting concurrent evaluations

**Deliverables:**
- Configuration YAML exploring 3+ chunking strategies, 2+ retrievers, multiple top-k values
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
**Duration:** 30 min
**Topics:**
- Reading the trial report: which configuration won and why
- Metric interpretation:
  - Retrieval metrics: precision@k, recall@k, MRR, NDCG
  - Generation metrics: BLEU, ROUGE, F1, answer similarity, faithfulness
- Deploying the optimal pipeline: `autorag deploy --trial_dir results/`
  - Creates a FastAPI server with the optimal configuration
  - API: query → retrieve → generate → answer
- Exporting the configuration for manual implementation
- Validating on the held-out test set (not used during optimization)
- Translating results to OpenShift AI: applying the optimal chunk size, embedding model, retrieval strategy to your OGX/LangChain RAG deployment

**Deliverables:**
- Optimal pipeline deployed locally
- Configuration exported for OpenShift AI RAG deployment
- Validation results on test set

---

### Level 1 Summary

| Module | Lessons | Estimated Time |
|--------|---------|---------------|
| M1: Fundamentals | 2 lessons | ~1 hour |
| M2: Evaluation Data | 2 lessons | ~1.5 hours |
| M3: Running Experiments | 3 lessons | ~2 hours |
| **Total** | **7 lessons** | **~4.5-5.5 hours** |

---

# LEVEL 2 — PRACTITIONER

*Goal: Advanced optimization, custom modules, integration with OpenShift AI.*
*Estimated time: ~4-5 hours*

---

## L2-M1: Advanced Optimization

### L2-1.1 — Advanced Retrieval Strategies
**Duration:** 45 min
**Topics:**
- Hybrid retrieval: combining BM25 + vector search
- Ensemble retrievers: weighted combinations
- Reranking strategies: cross-encoder, ColBERT, FlashRank
- Multi-stage retrieval: coarse retrieval → fine reranking
- Evaluating retrieval independently from generation
- Query expansion and transformation

**Deliverables:**
- Evaluation comparing hybrid retrieval + reranking combinations
- Optimal multi-stage retrieval configuration

---

### L2-1.2 — Embedding Model Comparison
**Duration:** 45 min
**Topics:**
- Evaluating multiple embedding models:
  - Open-source: nomic-embed-text, BGE, E5, GTE
  - Proprietary: OpenAI text-embedding-3, Cohere embed
  - Domain-specific: fine-tuned embeddings
- Dimensions, speed, and quality trade-offs
- Embedding model impact on retrieval quality
- Cost analysis: larger embeddings = more storage and compute

**Deliverables:**
- Comparison of 4+ embedding models on the same corpus
- Cost-quality analysis

---

### L2-1.3 — Custom Evaluation Metrics
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
| M1: Advanced Optimization | 3 lessons | ~2 hours |
| M2: Integration & Production | 2 lessons | ~1.5 hours |
| **Total** | **5 lessons** | **~3.5-4 hours** |

---

# Complete Course Summary

| Level | Focus | Lessons | Time |
|-------|-------|---------|------|
| **Level 1 — Essentials** | QA data, experiments, optimal pipeline | 7 lessons | ~4.5-5.5 hours |
| **Level 2 — Practitioner** | Advanced strategies, custom modules, OpenShift AI | 5 lessons | ~3.5-4 hours |
| **Total** | | **12 lessons** | **~8-10 hours** |
