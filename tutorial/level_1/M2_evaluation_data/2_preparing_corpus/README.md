# L1-M2.2 — Preparing the Corpus

**Level:** Essentials
**Duration:** 30 min

## Overview

Learn how to prepare a document corpus in AutoRAG's required format. This lesson covers text preprocessing, metadata enrichment, validation with AutoRAG's built-in utilities, and saving the corpus to Parquet for use in optimization pipelines.

## Prerequisites

- Python 3.10+
- Completed L1-M1.2 (Installing and Project Setup)

## Concepts

The corpus is the knowledge base that feeds every stage of an AutoRAG RAG pipeline. During optimization, AutoRAG chunks the corpus into passages, generates embeddings for each passage, indexes them in a vector store, and retrieves relevant passages in response to queries. The quality of your corpus directly determines the ceiling of your retrieval quality --- no amount of tuning can compensate for noisy, duplicated, or poorly structured source documents.

AutoRAG expects the corpus as a pandas DataFrame (persisted as Parquet) with three columns: `doc_id`, `contents`, and `metadata`. The `doc_id` column must contain unique string identifiers for each document. The `contents` column holds the raw text. The `metadata` column stores a Python dictionary per row with arbitrary keys, though several keys have special meaning inside AutoRAG's pipeline.

Two metadata fields unlock specific AutoRAG features. The `last_modified_datetime` field (an ISO 8601 string) enables the `recency_filter` retrieval node, which can boost recently updated documents during retrieval. The `prev_id` and `next_id` fields enable the `window_replacement` prompt-making strategy, which expands retrieved passages by pulling in content from neighboring documents to provide broader context to the language model.

Before constructing the DataFrame, you should preprocess your raw text. Common steps include stripping leading and trailing whitespace, collapsing multiple spaces and newlines into a single space, removing duplicate documents, and verifying that every document contains meaningful content. These steps reduce noise during chunking and embedding, leading to higher-quality retrieval results.

## Step-by-Step

### Step 1: Raw Document Collection

The first step is gathering your source documents. In a real project these might come from web scraping, a CMS export, PDF extraction, or a database dump. For this lesson we define 10 short educational texts about Python programming concepts directly in the script.

Each raw document is stored as a dictionary with a `topic` key (for later metadata) and a `text` key holding the content:

```python
RAW_DOCUMENTS: list[dict[str, str]] = [
    {
        "topic": "variables",
        "text": """
            Variables in Python are names that refer to objects stored in memory.
            Python is dynamically typed, so you do not need to declare ...
        """,
    },
    # ... 9 more documents
]
```

### Step 2: Text Preprocessing

Raw text often contains inconsistent whitespace from copy-paste, indentation artifacts, or extraction tools. We normalize each document in two steps: strip outer whitespace, then collapse all internal whitespace runs into single spaces.

```python
def normalize_whitespace(text: str) -> str:
    """Strip leading/trailing whitespace and collapse internal whitespace."""
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    return text
```

We also deduplicate by tracking the cleaned content of every document seen so far and skipping exact matches:

```python
def preprocess_documents(docs: list[dict[str, str]]) -> list[dict[str, str]]:
    cleaned: list[dict[str, str]] = []
    seen_contents: set[str] = set()

    for doc in docs:
        content = normalize_whitespace(doc["text"])
        if content in seen_contents:
            print(f"  [SKIP] Duplicate detected for topic '{doc['topic']}'")
            continue
        seen_contents.add(content)
        cleaned.append({"topic": doc["topic"], "text": content})

    return cleaned
```

The script prints a before/after comparison for the first document so you can see the effect of normalization.

### Step 3: Building the Corpus DataFrame

With clean documents in hand, we construct the DataFrame that AutoRAG expects. Each row gets a zero-padded `doc_id` string (`doc_001` through `doc_010`), the cleaned text in `contents`, and a metadata dictionary:

```python
def build_corpus_dataframe(docs: list[dict[str, str]]) -> pd.DataFrame:
    rows: list[dict] = []
    total = len(docs)

    for idx, doc in enumerate(docs):
        doc_id = f"doc_{idx + 1:03d}"
        prev_id = f"doc_{idx:03d}" if idx > 0 else None
        next_id = f"doc_{idx + 2:03d}" if idx < total - 1 else None

        metadata = {
            "last_modified_datetime": f"2024-01-{15 + idx:02d}T10:00:00",
            "prev_id": prev_id,
            "next_id": next_id,
            "source": "python_tutorial",
            "topic": doc["topic"],
        }

        rows.append({
            "doc_id": doc_id,
            "contents": doc["text"],
            "metadata": metadata,
        })

    return pd.DataFrame(rows)
```

### Step 4: Metadata Design

Each metadata field serves a purpose in the AutoRAG pipeline:

| Field | Purpose |
|-------|---------|
| `last_modified_datetime` | ISO 8601 timestamp. Enables the `recency_filter` retrieval node, which boosts or filters documents based on their freshness. |
| `prev_id` | ID of the preceding document (or `None`). Used by the `window_replacement` prompt maker to expand context by pulling in the previous document's content. |
| `next_id` | ID of the following document (or `None`). Also used by `window_replacement` to expand context forward. |
| `source` | Custom field. Useful for filtering or tagging documents by origin (e.g., "python_tutorial", "api_docs"). |
| `topic` | Custom field. Enables per-topic analysis of retrieval performance. |

The `prev_id` / `next_id` chain is especially important: when a retriever finds a relevant passage, the window replacement strategy can automatically include surrounding documents to give the LLM more context --- similar to how a reader would look at the paragraphs before and after a highlighted passage.

### Step 5: Validation

AutoRAG provides `cast_corpus_dataset` to verify your DataFrame has the correct schema. It checks that `doc_id`, `contents`, and `metadata` columns exist and have the right types. Always validate before running an optimization pipeline --- schema errors surface early here rather than deep inside a long-running experiment.

```python
from autorag.utils.preprocess import cast_corpus_dataset

validated_df = cast_corpus_dataset(corpus_df)
print("cast_corpus_dataset() passed successfully!")
```

### Step 6: Saving and Loading

AutoRAG uses Parquet as its data format. Parquet preserves complex data types (including nested dictionaries in the metadata column) and compresses well. Save with pandas and verify by loading it back:

```python
os.makedirs("data", exist_ok=True)
corpus_df.to_parquet("data/corpus.parquet")

loaded_df = pd.read_parquet("data/corpus.parquet")
print(f"Loaded DataFrame shape: {loaded_df.shape}")
```

## Running the Lesson

```bash
cd tutorial/level_1/M2_evaluation_data/2_preparing_corpus
uv sync
uv run python main.py
```

## Expected Output

```
============================================================
Step 1: Raw Document Collection
============================================================
  Loaded 10 raw documents
  Topics: ['variables', 'functions', 'classes', 'decorators', 'generators', 'list_comprehensions', 'error_handling', 'modules', 'virtual_environments', 'type_hints']

============================================================
Step 2: Text Preprocessing
============================================================

  --- Before (first 120 chars) ---
  '\n            Variables in Python are names that refer to objects stored in memory.\n            Python is dynamically ty'

  --- After (first 120 chars) ---
  'Variables in Python are names that refer to objects stored in memory. Python is dynamically typed, so you do not need t'

  Documents after preprocessing: 10

============================================================
Step 3: Building the Corpus DataFrame
============================================================
  DataFrame shape: (10, 3)
  Columns: ['doc_id', 'contents', 'metadata']

  First 3 rows (doc_id and content preview):
    doc_001: Variables in Python are names that refer to objects stored in me...
    doc_002: Functions let you encapsulate reusable logic behind a descriptiv...
    doc_003: Classes are blueprints for creating objects that bundle data and...

============================================================
Step 4: Metadata Inspection
============================================================
  Sample metadata (doc_001):
    last_modified_datetime: 2024-01-15T10:00:00
    prev_id: None
    next_id: doc_002
    source: python_tutorial
    topic: variables

============================================================
Step 5: Validation with AutoRAG
============================================================
  cast_corpus_dataset() passed successfully!
  Validated DataFrame shape: (10, 3)
  Validated columns: ['doc_id', 'contents', 'metadata']

============================================================
Step 6: Corpus Statistics
============================================================
  Total documents    : 10
  Avg content length : 572 characters
  Min content length : 498 characters
  Max content length : 645 characters
  Metadata keys      : ['last_modified_datetime', 'next_id', 'prev_id', 'source', 'topic']
  Unique topics      : ['classes', 'decorators', 'error_handling', 'functions', 'generators', 'list_comprehensions', 'modules', 'type_hints', 'variables', 'virtual_environments']

============================================================
Step 7: Saving to Parquet
============================================================
  Saved corpus to data/corpus.parquet

============================================================
Step 8: Loading and Verification
============================================================
  Loaded DataFrame shape: (10, 3)
  Columns: ['doc_id', 'contents', 'metadata']

  First 5 rows:
    doc_001: Variables in Python are names that refer to objects sto...
    doc_002: Functions let you encapsulate reusable logic behind a ...
    doc_003: Classes are blueprints for creating objects that bundl...
    doc_004: Decorators are a powerful pattern for modifying or ext...
    doc_005: Generators provide a memory-efficient way to produce s...

  Corpus saved and reloaded successfully!
```

Note: Exact character counts will vary slightly depending on whitespace normalization.

## Key Takeaways

- The corpus DataFrame requires three columns: `doc_id` (unique string), `contents` (text), and `metadata` (dict).
- Text preprocessing --- whitespace normalization and deduplication --- improves downstream chunking and retrieval quality.
- Metadata fields `prev_id` and `next_id` enable the `window_replacement` prompt-making strategy for expanded context.
- The `last_modified_datetime` metadata field enables `recency_filter` for time-aware retrieval.
- Always validate your corpus with `cast_corpus_dataset` before running an optimization pipeline.
- Parquet format preserves complex data types like nested dictionaries and compresses efficiently.

## Next Steps

In the next lesson (L1-M2.3), you will learn how to create QA evaluation datasets from your corpus, either manually or using LLM-based generation. These question-answer pairs are what AutoRAG uses to measure retrieval and generation quality during optimization.
