# L2-M1.3 -- Custom Evaluation Metrics

**Level:** Practitioner
**Duration:** 30 min

## Overview

AutoRAG ships with standard metrics (BLEU, ROUGE, F1, etc.), but real-world RAG systems often need domain-specific quality signals. This lesson teaches you how AutoRAG's metric system works under the hood and how to create, test, and register your own custom evaluation metrics.

## Prerequisites

- Completed: L1-M3.2 (Evaluation Basics)
- Python 3.10+ with `uv` installed
- Familiarity with AutoRAG's evaluation pipeline

## Concepts

### The Metric System

AutoRAG's evaluation pipeline is built on pluggable metric functions. Every metric is a Python function wrapped by one of two decorators:

- **`@autorag_metric(fields_to_check=[...])`** -- wraps a function that takes a single `MetricInput` and returns a `float` score.
- **`@autorag_metric_loop(fields_to_check=[...])`** -- wraps a function that takes a list of `MetricInput` objects and returns a list of `float` scores (batch mode).

The `fields_to_check` parameter tells AutoRAG which fields must be present in the `MetricInput` before running the metric. If a required field is missing, the framework raises an error before the function executes.

### MetricInput

`MetricInput` is a dataclass that carries all the data a metric might need:

| Field | Description |
|-------|-------------|
| `query` | The user's query |
| `queries` | Batch of queries |
| `retrieval_gt_contents` | Ground-truth retrieved contents |
| `retrieved_contents` | Actually retrieved contents |
| `retrieval_gt` | Ground-truth retrieved IDs |
| `retrieved_ids` | Actually retrieved IDs |
| `prompt` | The prompt sent to the generator |
| `generated_texts` | The generated answer |
| `generation_gt` | Ground-truth answer(s) |
| `generated_log_probs` | Log probabilities from the generator |

### Metric Registration

Custom metrics must be registered in one of two dictionaries before running an evaluation:

- `RETRIEVAL_METRIC_FUNC_DICT` -- for retrieval quality metrics
- `GENERATION_METRIC_FUNC_DICT` -- for generation quality metrics

Once registered, they can be referenced by name in `config.yaml`.

## Step-by-Step

### Step 1: Understand Built-in Metrics

The lesson begins by listing AutoRAG's built-in metrics and their field requirements. This helps you understand the interface your custom metrics must follow.

### Step 2: Create a Keyword Match Metric

We define `keyword_match_score` -- a metric that checks how many keywords from the ground-truth answer appear in the generated text. It filters out common stop words to focus on meaningful terms.

```python
@autorag_metric(fields_to_check=["generated_texts", "generation_gt"])
def keyword_match_score(metric_input: MetricInput) -> float:
    generated = metric_input.generated_texts.lower()
    gt_words = set(metric_input.generation_gt[0].lower().split())
    keywords = gt_words - stop_words
    if not keywords:
        return 1.0
    matches = sum(1 for kw in keywords if kw in generated)
    return matches / len(keywords)
```

### Step 3: Create a Length Ratio Metric

We define `length_ratio_score` -- a metric that scores how close the generated text's word count is to the ground truth's word count. A perfect score (1.0) means equal length; the score decreases as the ratio deviates from 1.0.

### Step 4: Test the Custom Metrics

We create sample `MetricInput` objects and run both metrics against them to verify they produce sensible scores.

### Step 5: Learn Registration

The lesson explains how to register custom metrics so they can be referenced in AutoRAG's YAML configuration.

## Running the Lesson

```bash
cd tutorial/level_2/M1_advanced_optimization/3_custom_metrics
uv sync
uv run python main.py
```

## Expected Output

```
============================================================
How AutoRAG's Metric System Works
============================================================
...
============================================================
Built-in Metrics
============================================================
Metric               Required Fields                                Category
-------------------------------------------------------------------------------------
retrieval_f1         retrieved_ids, retrieval_gt                    Retrieval
...
============================================================
Testing Custom Metrics
============================================================

Test case 1:
  Generated : Python decorators modify function behavior using the @ syntax
  Ground truth: Decorators in Python are functions that modify ...
  keyword_match_score : 0.571
  length_ratio_score  : 0.643
...
============================================================
Lesson complete!
============================================================
```

## Key Takeaways

- AutoRAG metrics are plain Python functions wrapped by the `@autorag_metric` decorator.
- `MetricInput` is a dataclass that carries query, retrieval, and generation data to the metric.
- The `fields_to_check` parameter ensures the required data is present before the metric runs.
- Custom metrics must be registered in the appropriate dictionary (`GENERATION_METRIC_FUNC_DICT` or `RETRIEVAL_METRIC_FUNC_DICT`) before evaluation.
- You can create metrics that capture domain-specific quality signals not covered by standard NLP metrics.

## Next Steps

Continue to L2-M2.1 (Custom Modules) to learn how to build custom pipeline components -- retrievers, rerankers, and generators -- that plug into AutoRAG's evaluation framework.
