# L1-M2.1 — Parsing and Corpus Creation

**Level:** Essentials
**Duration:** 45 min

## Overview

Learn how AutoRAG transforms raw documents into a corpus ready for optimization. This lesson covers the YAML-driven parsing and chunking pipelines, then walks through building a corpus manually so you understand the required format --- `doc_id`, `contents`, and `metadata` columns stored as Parquet.

## Prerequisites

- Completed: L1-M1.2 (Installing and Project Setup)
- Python 3.10+ with `uv` installed

## Concepts

### The Raw-to-Corpus Pipeline

Before AutoRAG can optimize a RAG pipeline, it needs a corpus --- a collection of text passages stored as a Parquet file. Getting from raw documents (PDFs, HTML, Markdown) to that Parquet file involves two stages: **parsing** (extracting text from files) and **chunking** (splitting text into passages). AutoRAG provides dedicated classes for both, each driven by a YAML configuration file.

### Parsing Pipeline

The `Parser` class reads raw files and extracts their text content. You point it at a glob pattern (e.g., `./data/*.pdf`) and a YAML config that specifies which parse module to use for each file type.

```python
from autorag.parser import Parser

parser = Parser(
    data_path_glob="./data/*.pdf",
    project_dir="./project"
)
parser.start_parsing("parse_config.yaml")
```

The parsing YAML lists one or more modules, each targeting a file type:

```yaml
modules:
  - module_type: langchain_parse
    file_type: pdf
    parse_method: pdfminer
```

Available parse modules include:

| Module | Backends |
|--------|----------|
| `langchain_parse` | pdfminer (PDF), csv, unstructuredmarkdown (Markdown), bshtml (HTML) |
| `llamaparse` | LlamaParse cloud service |
| `clova` | Clova OCR-based parsing |
| `table_hybrid_parse` | Specialized table extraction |

The parser outputs a `parsed_result.parquet` file containing the extracted text from each source document.

### Chunking Pipeline

The `Chunker` class takes parsed text and splits it into smaller passages suitable for retrieval. It reads the parsed Parquet file and applies a chunking strategy defined in YAML.

```python
from autorag.chunker import Chunker

chunker = Chunker.from_parquet(
    "parsed_result.parquet",
    project_dir="./project"
)
chunker.start_chunking("chunk_config.yaml")
```

The chunking YAML specifies the method and parameters:

```yaml
modules:
  - module_type: llama_index_chunk
    chunk_method: Token
    chunk_size: 512
    chunk_overlap: 24
```

Available chunking methods:

| Module | Methods |
|--------|---------|
| `llama_index_chunk` | Token, Sentence, SentenceWindow, Semantic, SemanticDoubleMerging |
| `langchain_chunk` | recursivecharacter, sentencetransformerstoken |

The chunker produces the final `corpus.parquet` file with `doc_id`, `contents`, and `metadata` columns.

### The Complete Flow

```
Raw Documents (PDF, HTML, Markdown, CSV)
        |
        v
Parser (parse_config.yaml)
        -> parsed_result.parquet
        |
        v
Chunker (chunk_config.yaml)
        -> corpus.parquet (doc_id, contents, metadata)
        |
        v
Ready for AutoRAG evaluation!
```

### Corpus Format

AutoRAG expects the corpus as a pandas DataFrame (persisted as Parquet) with three columns:

| Column | Type | Description |
|--------|------|-------------|
| `doc_id` | str | Unique identifier for each passage |
| `contents` | str | The text content |
| `metadata` | dict | Arbitrary metadata per passage |

Two metadata fields unlock specific AutoRAG features. The `last_modified_datetime` field (ISO 8601 string) enables the `recency_filter` retrieval node. The `prev_id` and `next_id` fields enable the `window_replacement` prompt-making strategy, which expands retrieved passages by pulling in content from neighboring documents.

### Text Preprocessing

Before constructing the DataFrame, you should preprocess your raw text. Common steps include stripping leading and trailing whitespace, collapsing multiple spaces and newlines into a single space, removing duplicate documents, and verifying that every document contains meaningful content. These steps reduce noise during chunking and embedding, leading to higher-quality retrieval results.

## Step-by-Step

### Steps 1-5: Understanding the Pipelines

The first five steps of `main.py` are informational --- they print explanations of the parsing pipeline, parsing YAML format, chunking pipeline, chunking YAML format, and the end-to-end flow. These steps teach you how the automated pipeline works before we build a corpus manually.

In a real project with PDF or HTML files, you would use the `Parser` and `Chunker` classes. For this lesson, we work with in-memory text to focus on the corpus format.

### Step 6: Raw Document Collection

We define 10 short educational texts about Python programming concepts directly in the script:

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

### Step 7: Text Preprocessing

Raw text often contains inconsistent whitespace from copy-paste, indentation artifacts, or extraction tools. We normalize each document in two steps: strip outer whitespace, then collapse all internal whitespace runs into single spaces. We also deduplicate by tracking cleaned content:

```python
def normalize_whitespace(text: str) -> str:
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    return text
```

### Step 8: Building the Corpus DataFrame

With clean documents in hand, we construct the DataFrame that AutoRAG expects. Each row gets a zero-padded `doc_id` string, the cleaned text in `contents`, and a metadata dictionary with `last_modified_datetime`, `prev_id`, `next_id`, `source`, and `topic`.

### Step 9: Metadata Design

Each metadata field serves a purpose:

| Field | Purpose |
|-------|---------|
| `last_modified_datetime` | ISO 8601 timestamp. Enables `recency_filter` retrieval node. |
| `prev_id` / `next_id` | Used by `window_replacement` prompt maker to expand context. |
| `source` | Custom field for filtering or tagging documents by origin. |
| `topic` | Custom field for per-topic analysis. |

### Step 10: Validation

AutoRAG provides `cast_corpus_dataset` to verify your DataFrame has the correct schema:

```python
from autorag.utils.preprocess import cast_corpus_dataset

validated_df = cast_corpus_dataset(corpus_df)
```

### Steps 11-13: Statistics, Saving, and Verification

We print corpus statistics, save to Parquet, and reload to verify the round-trip.

## Running the Lesson

```bash
cd tutorial/level_1/M2_evaluation_data/1_parsing_corpus_creation
uv sync
uv run python main.py
```

## Expected Output

```
============================================================
L1-M2.1 — Parsing and Corpus Creation
============================================================

============================================================
Step 1: AutoRAG Parsing Pipeline
============================================================
  AutoRAG provides a Parser class that converts raw documents
  (PDF, HTML, Markdown, CSV, etc.) into a parsed DataFrame.
  ...

============================================================
Step 2: Parsing YAML Configuration
============================================================
  Sample parse_config.yaml:

  modules:
    - module_type: langchain_parse
      file_type: pdf
      parse_method: pdfminer
  ...

============================================================
Step 3: AutoRAG Chunking Pipeline
============================================================
  After parsing, AutoRAG's Chunker class splits the parsed
  text into smaller passages suitable for retrieval.
  ...

============================================================
Step 4: Chunking YAML Configuration
============================================================
  Sample chunk_config.yaml:

  modules:
    - module_type: llama_index_chunk
      chunk_method: Token
      chunk_size: 512
      chunk_overlap: 24
  ...

============================================================
Step 5: Raw-to-Corpus Flow
============================================================
  The full pipeline from raw files to corpus.parquet:

  1. Raw Documents (PDF, HTML, Markdown, CSV, ...)
         |
         v
  2. Parser (parse_config.yaml)
         -> parsed_result.parquet
         |
         v
  3. Chunker (chunk_config.yaml)
         -> corpus.parquet (doc_id, contents, metadata)
         |
         v
  4. Ready for AutoRAG evaluation!
  ...

============================================================
Step 6: Raw Document Collection (manual approach)
============================================================
  Loaded 10 raw documents
  Topics: ['variables', 'functions', 'classes', ...]

============================================================
Step 7: Text Preprocessing
============================================================

  --- Before (first 120 chars) ---
  '\n            Variables in Python are names that refer to objects...'

  --- After (first 120 chars) ---
  'Variables in Python are names that refer to objects stored in memory...'

  Documents after preprocessing: 10

============================================================
Step 8: Building the Corpus DataFrame
============================================================
  DataFrame shape: (10, 3)
  Columns: ['doc_id', 'contents', 'metadata']
  ...

============================================================
Step 9: Metadata Inspection
============================================================
  Sample metadata (doc_001):
    last_modified_datetime: 2024-01-15T10:00:00
    prev_id: None
    next_id: doc_002
    source: python_tutorial
    topic: variables

============================================================
Step 10: Validation with AutoRAG
============================================================
  cast_corpus_dataset() passed successfully!
  ...

============================================================
Step 11: Corpus Statistics
============================================================
  Total documents    : 10
  ...

============================================================
Step 12: Saving to Parquet
============================================================
  Saved corpus to data/corpus.parquet

============================================================
Step 13: Loading and Verification
============================================================
  Loaded DataFrame shape: (10, 3)
  ...
  Corpus saved and reloaded successfully!

============================================================
Done!
============================================================
  Corpus is ready. Next: L1-M2.2 Creating QA Datasets
```

## Key Takeaways

- AutoRAG provides a **Parser** class for extracting text from raw documents (PDF, HTML, CSV, Markdown) via YAML configuration.
- AutoRAG provides a **Chunker** class for splitting parsed text into retrieval-ready passages via YAML configuration.
- The full flow is: Raw files -> Parser -> parsed_result.parquet -> Chunker -> corpus.parquet.
- The corpus DataFrame requires three columns: `doc_id` (unique string), `contents` (text), and `metadata` (dict).
- Metadata fields `prev_id`/`next_id` enable `window_replacement`; `last_modified_datetime` enables `recency_filter`.
- Always validate your corpus with `cast_corpus_dataset` before running an optimization pipeline.

## Next Steps

In **L1-M2.2 --- Creating QA Evaluation Datasets**, you will learn how to create question-answer pairs from your corpus using both manual curation and AutoRAG's fluent QA generation API. These QA pairs are what AutoRAG uses to measure retrieval and generation quality during optimization.
