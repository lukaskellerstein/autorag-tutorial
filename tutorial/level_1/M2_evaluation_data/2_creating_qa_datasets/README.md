# L1-M2.2 — Creating QA Evaluation Datasets

**Level:** Essentials
**Duration:** 45 min

## Overview

Evaluation data is the foundation of AutoRAG's optimization pipeline. Without high-quality question-answer pairs linked to source documents, AutoRAG cannot measure retrieval accuracy or generation quality. This lesson teaches you how to create, validate, and save QA evaluation datasets in the format AutoRAG expects, and introduces AutoRAG's fluent API for generating QA pairs at scale.

## Prerequisites

- Completed: L1-M2.1 (Parsing and Corpus Creation)
- Python 3.10+ with `uv` installed
- No infrastructure required --- this lesson works offline

## Concepts

### Why Evaluation Data Quality Matters

AutoRAG runs experiments by testing different retrieval and generation configurations against your evaluation data. If your QA pairs are low quality --- vague questions, incorrect ground truth answers, or broken document links --- the experiment results will be misleading. The optimizer might select a poor configuration because the evaluation data did not accurately represent real-world usage.

### Three Approaches to Creating QA Data

1. **Manual creation** --- Write questions and answers by hand. Highest quality but time-consuming. Best for small, domain-specific datasets where precision matters.

2. **LLM-generated** --- Use a language model to generate questions from your documents. Scales well but requires review for accuracy. Good for bootstrapping large datasets.

3. **AutoRAG's fluent API** --- AutoRAG provides a chainable API that automates sampling, question generation, filtering, and answer generation. Best for production use with large document collections.

This lesson uses manual creation (approach 1) so you understand the data format, then explains the fluent API (approach 3) so you can scale up.

### Data Format Requirements

AutoRAG expects two datasets in parquet format:

**Corpus DataFrame:**
| Column   | Type   | Description                        |
|----------|--------|------------------------------------|
| doc_id   | str    | Unique document identifier         |
| contents | str    | Document text content              |
| metadata | dict   | Additional metadata (dates, links) |

**QA DataFrame:**
| Column        | Type             | Description                              |
|---------------|------------------|------------------------------------------|
| qid           | str              | Unique question identifier               |
| query         | str              | The question text                        |
| retrieval_gt  | List[List[str]]  | Ground truth doc_ids for retrieval       |
| generation_gt | List[str]        | Expected answer(s) for generation eval   |

The `retrieval_gt` format uses nested lists: `[["doc_001"]]` means document "doc_001" is the relevant passage. Multiple inner lists represent multiple valid retrieval sets.

### AutoRAG Fluent QA Generation API

AutoRAG provides a fluent (chainable) API for generating QA pairs programmatically. The pipeline has four stages:

**Sampling** --- Select passages from the corpus:
- `random_single_hop(corpus_df, n=100)` --- randomly sample n passages
- `range_single_hop(corpus_df, idx_range)` --- sample a specific index range

**Query generation** --- Create questions from sampled passages:
- `factoid_query_gen` --- factual questions with specific answers
- `concept_completion_query_gen` --- conceptual/definitional questions
- `two_hop_incremental` --- multi-hop questions requiring two passages

**Query evolving** --- Increase question difficulty:
- `reasoning_evolve_ragas` --- add reasoning steps
- `conditional_evolve_ragas` --- add conditional constraints
- `compress_ragas` --- compress queries while preserving meaning

**Filtering** --- Remove low-quality pairs:
- `dontknow_filter_rule_based` --- rule-based filter for unanswerable questions
- `dontknow_filter_openai` / `dontknow_filter_llama_index` --- LLM-based filtering
- `passage_dependency_filter` --- filter questions that don't depend on the passage

**Answer generation** --- Create reference answers:
- `make_basic_gen_gt` --- generate detailed reference answers
- `make_concise_gen_gt` --- generate concise reference answers

The complete pipeline chains these stages together:

```python
from openai import AsyncOpenAI
client = AsyncOpenAI()

corpus.sample(random_single_hop, n=100)
    .batch_apply(factoid_query_gen, client=client)
    .batch_apply(make_basic_gen_gt, client=client)
    .batch_filter(dontknow_filter_openai, client=client)
    .to_parquet("qa.parquet", "corpus.parquet")
```

### Train/Test Splitting

Splitting your QA data into train and test sets ensures you can validate that AutoRAG's optimized pipeline generalizes beyond the data it was tuned on. An 80/20 split is standard for evaluation datasets of this size.

## Step-by-Step

### Step 1: Create a Sample Corpus

We create 10 documents about Python programming concepts. Each document has a unique `doc_id`, text `contents`, and `metadata` dictionary.

```python
documents = [
    {
        "doc_id": "doc_001",
        "contents": "Variables in Python are names that refer to objects...",
        "metadata": {
            "last_modified_datetime": "2024-01-15T10:00:00",
            "prev_id": None,
            "next_id": None
        },
    },
    # ... 9 more documents
]
corpus_df = pd.DataFrame(documents)
```

### Step 2: Create QA Pairs

For each document, we create two questions --- one factual and one conceptual. Each QA pair links back to its source document via `retrieval_gt`.

```python
qa_pairs = [
    {
        "qid": "q_001",
        "query": "What naming rules must Python variables follow?",
        "retrieval_gt": [["doc_001"]],
        "generation_gt": ["Variable names must start with a letter or underscore..."],
    },
    # ... 19 more pairs
]
qa_df = pd.DataFrame(qa_pairs)
```

### Step 3: Review Quality

We validate the dataset by checking for duplicate queries, verifying all `retrieval_gt` references point to real documents, and ensuring no `generation_gt` values are empty.

### Step 4: Split Train/Test

An 80/20 split with a fixed random seed ensures reproducibility:

```python
test_df = qa_df.sample(frac=0.2, random_state=42)
train_df = qa_df.drop(test_df.index)
```

### Step 5: Understand AutoRAG's Fluent QA Generation API

AutoRAG provides a fluent API for generating QA pairs at scale. The pipeline chains sampling, query generation (factoid, concept completion, multi-hop), query evolving, filtering (rule-based and LLM-based), and answer generation into a single chainable expression.

### Step 6: Save to Parquet

Both datasets are saved in parquet format, which AutoRAG reads natively:

```python
qa_df.to_parquet("data/qa.parquet", index=False)
corpus_df.to_parquet("data/corpus.parquet", index=False)
```

## Running the Lesson

```bash
cd tutorial/level_1/M2_evaluation_data/2_creating_qa_datasets
uv sync
uv run python main.py
```

## Expected Output

```
============================================================
L1-M2.2 — Creating QA Evaluation Datasets
============================================================

============================================================
Step 1: Create sample corpus
============================================================
Created corpus with 10 documents
Columns: ['doc_id', 'contents', 'metadata']

Sample document (doc_001):
  Content preview: Variables in Python are names that refer to objects stored in memory. Unlike...
  Metadata: {'last_modified_datetime': '2024-01-15T10:00:00', 'prev_id': None, 'next_id': None}

============================================================
Step 2: Create QA evaluation dataset
============================================================
Created 20 QA pairs
Columns: ['qid', 'query', 'retrieval_gt', 'generation_gt']

Sample QA pair (q_001):
  Query: What naming rules must Python variables follow?
  Retrieval GT: [['doc_001']]
  Generation GT: Variable names must start with a letter or underscore, followed by letters,...

============================================================
Step 3: Review QA quality
============================================================
Total QA pairs: 20
Average query length: 9.5 words
  Min: 7 words, Max: 13 words
Unique documents referenced: 10 / 10
Duplicate queries: 0
  OK — no duplicates found
Retrieval GT validation: OK — all doc_ids exist in corpus
Generation GT validation: OK — no empty answers

============================================================
Step 4: Split into train/test sets
============================================================
Total QA pairs: 20
Train set: 16 pairs (80%)
Test set:  4 pairs (20%)

============================================================
Step 5: AutoRAG Fluent QA Generation API (explanation)
============================================================
AutoRAG provides a fluent API for generating QA pairs at scale.
The pipeline chains sampling, query generation, filtering, and
answer generation into a single expression.

--- Sampling Methods ---
  random_single_hop(corpus_df, n=100)
  range_single_hop(corpus_df, idx_range)

--- Query Generation Types ---
  factoid_query_gen
  concept_completion_query_gen
  two_hop_incremental

--- Query Evolving (increase difficulty) ---
  reasoning_evolve_ragas
  conditional_evolve_ragas
  compress_ragas

--- Filtering (remove low-quality pairs) ---
  dontknow_filter_rule_based
  dontknow_filter_openai / dontknow_filter_llama_index
  passage_dependency_filter

--- Answer Generation ---
  make_basic_gen_gt
  make_concise_gen_gt

--- Fluent API Example ---
  corpus.sample(random_single_hop, n=100)
      .batch_apply(factoid_query_gen, client=client)
      .batch_apply(make_basic_gen_gt, client=client)
      .batch_filter(dontknow_filter_openai, client=client)
      .to_parquet("qa.parquet", "corpus.parquet")
...

============================================================
Step 6: Save datasets to parquet
============================================================
Saved QA dataset:     data/qa.parquet (X,XXX bytes)
Saved corpus dataset: data/corpus.parquet (X,XXX bytes)

============================================================
Done! Next: L1-M3.1 Configuration YAML
============================================================
```

## Key Takeaways

- AutoRAG requires two parquet files: a **corpus** (documents) and a **QA dataset** (questions with ground truth).
- Each QA pair must include `retrieval_gt` (which documents are relevant) and `generation_gt` (expected answers).
- AutoRAG's fluent API chains sampling (`random_single_hop`), query generation (`factoid_query_gen`, `concept_completion_query_gen`, `two_hop_incremental`), filtering (`dontknow_filter`), and answer generation (`make_basic_gen_gt`) into a single pipeline.
- Query evolving methods (`reasoning_evolve_ragas`, `conditional_evolve_ragas`, `compress_ragas`) increase question difficulty for more robust evaluation.
- Always validate your dataset: check for duplicates, missing references, and empty answers.
- Split data into train/test sets to verify your optimized pipeline generalizes.

## Next Steps

In **L1-M3.1 --- Configuration YAML**, you will learn how to write the YAML configuration file that tells AutoRAG which modules, parameters, and metrics to evaluate. This configuration drives the optimization pipeline that finds your best RAG setup.
