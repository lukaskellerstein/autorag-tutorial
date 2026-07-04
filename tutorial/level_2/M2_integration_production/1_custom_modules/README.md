# L2-M2.1 -- Custom Modules

**Level:** Practitioner
**Duration:** 45 min

## Overview

AutoRAG's power comes from its modular architecture -- every step in the RAG pipeline (retrieval, reranking, prompt construction, generation) is a swappable module. This lesson teaches you how the module system works, how to build your own custom modules, and how to register them so AutoRAG can evaluate them alongside built-in options.

## Prerequisites

- Completed: L1-M3.2 (Evaluation Basics)
- Completed: L2-M1.3 (Custom Evaluation Metrics)
- Python 3.10+ with `uv` installed

## Concepts

### Module Hierarchy

Every AutoRAG module inherits from `BaseModule`, which defines the contract:

```
BaseModule (abstract base class)
  pure(previous_result, *args, **kwargs) -- public entry point
  _pure(*args, **kwargs)                -- internal implementation
  cast_to_run(previous_result)          -- extracts typed inputs from DataFrame
```

Specialized base classes add domain-specific behavior:

| Base Class | Purpose | Input Columns |
|------------|---------|---------------|
| `BaseRetrieval` | Retrieval modules | queries |
| `BasePassageReranker` | Reranking modules | queries, contents, scores, ids |
| `BaseGenerator` | Generation modules | prompts |

### Data Flow

Modules communicate through pandas DataFrames. Each module:
1. Receives a DataFrame from the previous node via `pure()`
2. Extracts its required columns via `cast_to_run()`
3. Processes the data in `_pure()`
4. Returns a new DataFrame with its results

This uniform interface is what allows AutoRAG to swap modules freely during evaluation.

### The `@result_to_dataframe` Decorator

Built-in modules use this decorator to automatically convert their tuple outputs into a DataFrame with named columns. This keeps the `_pure()` method focused on logic, not data wrangling.

## Step-by-Step

### Step 1: Understand the Module Architecture

The lesson begins by explaining the module hierarchy and data flow. Understanding this architecture is essential before building your own modules.

### Step 2: Build a Custom Reranker

We implement `KeywordBoostReranker` -- a reranker that boosts passage scores based on keyword overlap with the query. While simple, it demonstrates the full module interface:

```python
class KeywordBoostReranker:
    def __init__(self, project_dir: str, boost_factor: float = 2.0):
        self.project_dir = project_dir
        self.boost_factor = boost_factor

    def rerank(self, queries, contents_list, scores_list,
               ids_list, top_k=3):
        # Boost scores for passages containing query keywords
        # Sort by boosted score, return top_k
        ...
```

### Step 3: Run the Custom Reranker

We create mock retrieval data and run the `KeywordBoostReranker` to see how it changes the ranking. The before/after comparison shows the effect of keyword boosting.

### Step 4: Study a Built-in Module

We examine `FlashRankReranker` to understand how production modules use `BasePassageReranker`, the `@result_to_dataframe` decorator, and the `pure()` / `_pure()` split.

### Step 5: Compare Boost Factors

We run the custom reranker with different `boost_factor` values to observe how the hyperparameter affects ranking. This mirrors what AutoRAG does during evaluation -- testing multiple configurations.

### Step 6: Learn Registration

The lesson explains how to register custom modules so AutoRAG can discover and evaluate them.

## Running the Lesson

```bash
cd tutorial/level_2/M2_integration_production/1_custom_modules
uv sync
uv run python main.py
```

## Expected Output

```
============================================================
AutoRAG Module Architecture
============================================================
...
============================================================
Running KeywordBoostReranker
============================================================

Before reranking:
  [doc_001] score=0.80  Decorators modify functions in Python ...
  [doc_002] score=0.70  Python has many features including ...
  [doc_003] score=0.60  The @ symbol is used for decorators ...
  [doc_004] score=0.55  Java annotations also use the @ symbol ...

After reranking (boost_factor=2.0, top_k=3):
  [doc_001] score=4.80  Decorators modify functions in Python ...
  [doc_003] score=4.60  The @ symbol is used for decorators ...
  [doc_002] score=2.70  Python has many features including ...
...
============================================================
Lesson complete!
============================================================
```

## Key Takeaways

- AutoRAG modules follow a uniform interface: `pure()` receives a DataFrame, `_pure()` contains the logic, `cast_to_run()` extracts typed inputs.
- Specialized base classes (`BaseRetrieval`, `BasePassageReranker`, `BaseGenerator`) handle domain-specific plumbing.
- Custom modules can be as simple as a keyword heuristic or as complex as a neural model -- the interface is the same.
- The `@result_to_dataframe` decorator converts tuple outputs to DataFrames automatically.
- Registration happens at runtime before calling `autorag.evaluate()`.

## Next Steps

Continue to L2-M2.2 (From AutoRAG to OpenShift AI) to learn how to take AutoRAG's optimized configuration and deploy the resulting RAG pipeline on OpenShift AI.
