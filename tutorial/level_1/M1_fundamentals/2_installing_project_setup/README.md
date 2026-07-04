# L1-M1.2 — Installing and Project Setup

**Level:** Essentials
**Duration:** 30 min

## Overview

Set up your first AutoRAG project and understand the data format requirements for QA evaluation datasets and corpus documents. By the end of this lesson you will know how to structure a project directory, create valid input data, validate it with AutoRAG's built-in utilities, and navigate the CLI.

## Prerequisites

- Python 3.10+
- Completed L1-M1.1 (What is AutoRAG?)
- `uv` installed (see https://docs.astral.sh/uv/)

## Concepts

### Project Structure

Every AutoRAG experiment lives in a project directory with three required inputs and one auto-generated output folder. The **config.yaml** file describes the pipeline you want to evaluate: which chunking strategies, retrievers, rerankers, and generators to try, along with the parameter ranges for each. The **qa.parquet** file holds your evaluation data -- questions paired with ground-truth retrieval targets and expected answers. The **corpus.parquet** file contains the raw documents that AutoRAG will chunk, embed, and retrieve from during evaluation. When you run `autorag evaluate`, AutoRAG creates the **results/** directory with per-node scores, the greedy-selected best configuration, and data for the dashboard.

### QA Dataset Format

The QA dataset is the heart of AutoRAG's evaluation loop. Each row represents one evaluation question. The `qid` column is a unique string identifier. The `query` column holds the question text. The `retrieval_gt` column contains ground-truth document IDs -- structured as `List[List[str]]` to support multiple valid retrieval sets per question. The `generation_gt` column holds one or more reference answers as `List[str]`. AutoRAG measures retrieval quality by comparing retrieved documents against `retrieval_gt`, and generation quality by comparing the model's answer against `generation_gt`.

### Corpus Format

The corpus dataset stores the knowledge base that AutoRAG retrieves from. Each row has a `doc_id` (unique string), a `contents` field with the document text, and a `metadata` dictionary. The metadata must include `last_modified_datetime` (for temporal ordering), plus `prev_id` and `next_id` (for document sequence tracking -- set to `None` if not applicable). AutoRAG handles chunking internally during evaluation, so the corpus contains full or pre-segmented documents, not pre-chunked text.

### Validation Utilities

Before running a potentially long evaluation, you should validate that your data is well-formed. AutoRAG provides three utilities in `autorag.utils.preprocess`: `cast_qa_dataset()` normalizes QA column types and checks for required fields; `cast_corpus_dataset()` validates metadata keys and drops rows with empty content; `validate_qa_from_corpus_dataset()` cross-checks that every `doc_id` referenced in `retrieval_gt` actually exists in the corpus. Running these before evaluation saves debugging time.

## Step-by-Step

### Step 1: Project Structure

A minimal AutoRAG project looks like this:

```
my_project/
├── config.yaml         # Pipeline configuration
├── qa.parquet           # Evaluation QA pairs
├── corpus.parquet       # Document corpus
└── results/             # Auto-generated output
```

- **config.yaml** -- defines what to evaluate (modules, parameters, metrics).
- **qa.parquet** -- defines how to measure quality (queries + ground truth).
- **corpus.parquet** -- provides the documents to retrieve from.
- **results/** -- created by `autorag evaluate`; holds scores and the best config.

### Step 2: QA Dataset Format

Create a QA DataFrame with the four required columns:

```python
import pandas as pd

qa_data = {
    "qid": ["q_001", "q_002", "q_003"],
    "query": [
        "What is a decorator in Python?",
        "How do list comprehensions work in Python?",
        "What is the GIL in Python?",
    ],
    "retrieval_gt": [
        [["doc_001"]],
        [["doc_002"]],
        [["doc_003"]],
    ],
    "generation_gt": [
        ["A decorator is a function that takes another function..."],
        ["List comprehensions provide a concise way to create lists..."],
        ["The GIL is a mutex in CPython that allows only one thread..."],
    ],
}

qa_df = pd.DataFrame(qa_data)
```

Notice that `retrieval_gt` is `List[List[str]]` -- the outer list allows multiple valid retrieval sets, and the inner list holds the doc IDs for each set. For most use cases a single inner list is sufficient: `[["doc_001"]]`.

### Step 3: Corpus Format

Create a corpus DataFrame with the three required columns:

```python
from datetime import datetime

corpus_data = {
    "doc_id": ["doc_001", "doc_002", "doc_003"],
    "contents": [
        "Decorators in Python are a powerful tool for modifying...",
        "List comprehensions provide a concise and readable way...",
        "The Global Interpreter Lock (GIL) is a mutex that...",
    ],
    "metadata": [
        {"last_modified_datetime": datetime.now(), "prev_id": None, "next_id": "doc_002"},
        {"last_modified_datetime": datetime.now(), "prev_id": "doc_001", "next_id": "doc_003"},
        {"last_modified_datetime": datetime.now(), "prev_id": "doc_002", "next_id": None},
    ],
}

corpus_df = pd.DataFrame(corpus_data)
```

Every metadata dict must include `last_modified_datetime`, `prev_id`, and `next_id`. If your documents are unordered, set `prev_id` and `next_id` to `None`.

### Step 4: Data Validation

AutoRAG provides three validation functions in `autorag.utils.preprocess`:

```python
from autorag.utils.preprocess import (
    cast_qa_dataset,
    cast_corpus_dataset,
    validate_qa_from_corpus_dataset,
)

qa_validated = cast_qa_dataset(qa_df)
corpus_validated = cast_corpus_dataset(corpus_df)
validate_qa_from_corpus_dataset(qa_validated, corpus_validated)
```

- **`cast_qa_dataset()`** -- checks that `qid` and `query` are strings, normalizes `retrieval_gt` to `List[List[str]]`, normalizes `generation_gt` to `List[str]`, and preprocesses text (Unicode normalization).
- **`cast_corpus_dataset()`** -- drops rows with empty `contents`, ensures metadata has `last_modified_datetime`, `prev_id`, and `next_id` keys, and normalizes text.
- **`validate_qa_from_corpus_dataset()`** -- asserts that every doc ID in `retrieval_gt` exists in the corpus. This catches typos and missing documents before you start a long evaluation run.

### Step 5: CLI Commands

AutoRAG provides several CLI commands:

| Command | Purpose |
|---------|---------|
| `autorag evaluate` | Run a full evaluation experiment |
| `autorag dashboard` | Launch the Streamlit results dashboard |
| `autorag run_api` | Deploy the best pipeline as a FastAPI server |
| `autorag run_web` | Launch a Gradio web UI for interactive testing |
| `autorag validate` | Validate a config YAML without running evaluation |
| `autorag extract_best_config` | Extract the optimal configuration from results |
| `autorag restart_evaluate` | Resume a previously interrupted evaluation run |

The most common workflow is: `validate` your config, `evaluate` the experiment, view results with `dashboard`, then `extract_best_config` or `run_api`/`run_web` to use the winning pipeline.

### Step 6: Vector Database Configuration

AutoRAG requires a top-level `vectordb` section in the config YAML that defines where embeddings are stored:

```yaml
vectordb:
  - name: default
    db_type: chroma
    client_type: persistent
    path: ./chroma_db
```

Supported vector databases: Chroma, Qdrant, Milvus, Weaviate, Pinecone, and Couchbase. The `vectordb` section is separate from `node_lines` and applies to all retrieval modules that use vector search.

### Step 7: LLM Backends

AutoRAG uses LlamaIndex for LLM access. The generator config specifies `llm` (the backend) and `model` (the model name) as separate fields:

| Backend | Description |
|---------|-------------|
| `ollama` | Local models via Ollama (default for this tutorial) |
| `openai` | OpenAI API (GPT-4o, GPT-4, GPT-3.5) |
| `openailike` | OpenAI-compatible APIs (e.g., local servers) |
| `huggingfacellm` | HuggingFace Inference API or local models |
| `bedrock` | AWS Bedrock (Claude, Titan, etc.) |
| `vllm` | vLLM for high-throughput local serving |

Example generator config for Ollama:

```yaml
- module_type: llama_index_llm
  llm: ollama
  model: gemma4:e2b
```

Note: AutoRAG does NOT use LiteLLM format (e.g., `ollama/gemma4:e2b`). The `llm` and `model` fields are always separate.

### Step 8: Configuration YAML

The config YAML defines the evaluation search space. Here is a simplified example:

```yaml
vectordb:
  - name: default
    db_type: chroma
    client_type: persistent
    path: ./chroma_db

node_lines:
  - node_line_name: retrieve_and_generate
    nodes:
      - node_type: retrieval
        strategy:
          metrics: [retrieval_f1, retrieval_recall]
        modules:
          - module_type: bm25
            top_k: [3, 5, 10]
          - module_type: vectordb
            embedding_model:
              - huggingface/BAAI/bge-small-en-v1.5
            top_k: [3, 5, 10]

      - node_type: generator
        strategy:
          metrics: [generation_f1, rouge]
        modules:
          - module_type: llama_index_llm
            llm: ollama
            model: gemma4:e2b
```

Key concepts:
- **vectordb** configures the vector store for embedding storage.
- **node_lines** group nodes into a sequential pipeline.
- Each **node** lists candidate **modules** with parameter ranges.
- **strategy.metrics** controls which metric drives greedy optimization.
- Generator uses `llama_index_llm` with separate `llm` and `model` fields.
- AutoRAG evaluates all combinations and selects the best module at each node.

## Running the Lesson

```bash
cd tutorial/level_1/M1_fundamentals/2_installing_project_setup
uv sync
uv run python main.py
```

## Expected Output

```
============================================================
  L1-M1.2 — Installing and Project Setup
============================================================

This lesson walks through AutoRAG project structure, data
format requirements, validation utilities, and CLI commands.

============================================================
  Step 1: AutoRAG Project Structure
============================================================

An AutoRAG project requires three inputs and produces one output directory:

    my_project/
    ├── config.yaml       # Pipeline configuration — which modules and
    │                      #   parameters to evaluate at each node
    ├── qa.parquet         # Evaluation QA pairs — queries with ground-truth
    │                      #   retrieval doc IDs and expected answers
    ├── corpus.parquet     # Document corpus — the knowledge base to
    │                      #   retrieve from (doc_id, contents, metadata)
    └── results/           # Auto-generated after running evaluation —
                           #   contains trial results, metrics, and the
                           #   best configuration found

============================================================
  Step 2: QA Dataset Format
============================================================

The QA dataset has four required columns:
  - qid:            unique question ID (str)
  - query:          the question text (str)
  - retrieval_gt:   ground-truth doc IDs, List[List[str]]
  - generation_gt:  expected answers, List[str]

QA DataFrame:
------------------------------------------------------------
  qid                                        query
q_001             What is a decorator in Python?
q_002  How do list comprehensions work in Python?
q_003              What is the GIL in Python?

Column dtypes:
qid               object
query             object
retrieval_gt      object
generation_gt     object

============================================================
  Step 3: Corpus Dataset Format
============================================================

The corpus dataset has three required columns:
  - doc_id:    unique document ID (str)
  - contents:  the document text (str)
  - metadata:  dict with last_modified_datetime, prev_id, next_id

Corpus DataFrame:
------------------------------------------------------------
  doc_id:   doc_001
  contents: Decorators in Python are a powerful tool for modifying the behav...
  metadata: {'last_modified_datetime': ..., 'prev_id': None, 'next_id': 'doc_002'}

  doc_id:   doc_002
  contents: List comprehensions provide a concise and readable way to creat...
  metadata: {'last_modified_datetime': ..., 'prev_id': 'doc_001', 'next_id': 'doc_003'}

  doc_id:   doc_003
  contents: The Global Interpreter Lock (GIL) is a mutex that protects acce...
  metadata: {'last_modified_datetime': ..., 'prev_id': 'doc_002', 'next_id': None}

============================================================
  Step 4: Data Validation
============================================================

[1/3] Validating QA dataset...
      OK — QA dataset passed validation.
      Rows: 3, Columns: ['qid', 'query', 'retrieval_gt', 'generation_gt']

[2/3] Validating corpus dataset...
      OK — Corpus dataset passed validation.
      Rows: 3, Columns: ['doc_id', 'contents', 'metadata']

[3/3] Cross-validating QA retrieval_gt against corpus doc_ids...
      OK — All retrieval_gt doc_ids exist in the corpus.

All validations passed successfully.

============================================================
  Step 5: AutoRAG CLI Commands
============================================================

  autorag evaluate ...
  autorag dashboard ...
  autorag run_api ...
  autorag run_web ...
  autorag validate ...
  autorag extract_best_config ...
  autorag restart_evaluate ...

============================================================
  Step 6: Vector Database Configuration
============================================================

    vectordb:
      - name: default
        db_type: chroma
        ...

============================================================
  Step 7: LLM Backends
============================================================

  ollama               Local models via Ollama ...
  openai               OpenAI API ...
  ...

  Example generator config for Ollama:
    - module_type: llama_index_llm
      llm: ollama
      model: gemma4:e2b

============================================================
  Step 8: Sample Configuration YAML
============================================================

    vectordb:
      - name: default
        db_type: chroma
        ...
    node_lines:
    - node_line_name: retrieve_node_line
      nodes:
        - node_type: lexical_retrieval
          ...
        - node_type: passage_reranker
          ...
    - node_line_name: post_retrieve_node_line
      nodes:
        - node_type: prompt_maker
          ...
        - node_type: generator
          ...
            - module_type: llama_index_llm
              llm: ollama
              model: [gemma4:e2b]

============================================================
  Lesson Complete
============================================================

You now understand the AutoRAG project layout, data formats,
and validation workflow. In the next module (L1-M2.1), you
will learn how to parse documents and create a corpus.
```

## Key Takeaways

- An AutoRAG project needs three files: `config.yaml`, `qa.parquet`, and `corpus.parquet`.
- The QA dataset contains queries with ground-truth retrieval document IDs and expected answers.
- The corpus dataset contains documents with unique IDs, content text, and metadata (including `last_modified_datetime`, `prev_id`, `next_id`).
- AutoRAG provides validation utilities (`cast_qa_dataset`, `cast_corpus_dataset`, `validate_qa_from_corpus_dataset`) to check data consistency before running evaluation.
- The config YAML requires a top-level `vectordb` section to configure the vector store for embeddings.
- AutoRAG uses LlamaIndex for LLM access — generator config uses `llm: ollama` + `model: gemma4:e2b` (NOT LiteLLM format).
- The CLI provides commands for evaluation, dashboard, API serving (`run_api`), web UI (`run_web`), validation, config extraction, and resuming interrupted runs (`restart_evaluate`).
- The config YAML defines the search space of modules and parameters that AutoRAG will explore at each pipeline node.

## Next Steps

In the next lesson (L1-M2.1), you will learn how to parse documents and create a corpus — the foundation for all AutoRAG evaluations.
