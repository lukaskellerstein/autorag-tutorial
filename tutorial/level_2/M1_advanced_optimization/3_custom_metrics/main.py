"""
L2-M1.3 — Custom Evaluation Metrics

Learn how AutoRAG's metric system works and create your own custom
evaluation metrics using the @autorag_metric decorator.
"""

from dataclasses import dataclass, field
from typing import Callable, List, Optional


# ---------------------------------------------------------------------------
# Lightweight stand-in for AutoRAG's MetricInput so the lesson runs without
# a full AutoRAG installation.  The real class lives in
# autorag.schema.metricinput and carries the same fields.
# ---------------------------------------------------------------------------

@dataclass
class MetricInput:
    """Mirrors autorag.schema.metricinput.MetricInput."""

    query: Optional[str] = None
    queries: Optional[List[str]] = None
    retrieval_gt_contents: Optional[List[List[str]]] = None
    retrieved_contents: Optional[List[str]] = None
    retrieval_gt: Optional[List[List[str]]] = None
    retrieved_ids: Optional[List[str]] = None
    prompt: Optional[str] = None
    generated_texts: Optional[str] = None
    generation_gt: Optional[List[str]] = field(default_factory=list)
    generated_log_probs: Optional[List[float]] = None


def autorag_metric(fields_to_check: List[str]) -> Callable:
    """Decorator that mirrors autorag.evaluation.metric.util.autorag_metric."""

    def decorator(func: Callable) -> Callable:
        def wrapper(metric_input: MetricInput) -> float:
            for f in fields_to_check:
                val = getattr(metric_input, f, None)
                if val is None:
                    raise ValueError(f"MetricInput is missing required field: {f}")
            return func(metric_input)
        wrapper.__name__ = func.__name__
        wrapper.__doc__ = func.__doc__
        wrapper._fields_to_check = fields_to_check
        return wrapper
    return decorator


# ---------------------------------------------------------------------------
# 1. Explain the metric system
# ---------------------------------------------------------------------------

def explain_metric_system() -> None:
    print("=" * 60)
    print("How AutoRAG's Metric System Works")
    print("=" * 60)
    print("""
AutoRAG evaluates RAG pipelines using pluggable metric functions.

Decorator variants:
  @autorag_metric(fields_to_check=[...])
      Wraps a function that receives a single MetricInput and
      returns a float score.

  @autorag_metric_loop(fields_to_check=[...])
      Wraps a function that receives a list of MetricInputs and
      returns a list of floats (batch evaluation).

MetricInput is a dataclass whose fields carry everything a metric
might need:
  query, queries, retrieval_gt_contents, retrieved_contents,
  retrieval_gt, retrieved_ids, prompt, generated_texts,
  generation_gt, generated_log_probs

The decorator validates that the required fields are present
before the metric function runs.

Metrics are registered in one of two dictionaries:
  - RETRIEVAL_METRIC_FUNC_DICT   (retrieval quality)
  - GENERATION_METRIC_FUNC_DICT  (generation quality)
""")


# ---------------------------------------------------------------------------
# 2. Show built-in metrics
# ---------------------------------------------------------------------------

def show_built_in_metrics() -> None:
    print("=" * 60)
    print("Built-in Metrics")
    print("=" * 60)

    metrics = [
        ("retrieval_f1",     "retrieved_ids, retrieval_gt",     "Retrieval"),
        ("retrieval_recall", "retrieved_ids, retrieval_gt",     "Retrieval"),
        ("bleu",             "generated_texts, generation_gt",  "Generation"),
        ("rouge",            "generated_texts, generation_gt",  "Generation"),
        ("sem_score",        "generated_texts, generation_gt (+ embedding model)", "Generation"),
        ("g_eval",           "generated_texts, generation_gt (+ LLM judge)",       "Generation"),
    ]

    print(f"\n{'Metric':<20} {'Required Fields':<45} {'Category'}")
    print("-" * 85)
    for name, fields, cat in metrics:
        print(f"{name:<20} {fields:<45} {cat}")
    print()


# ---------------------------------------------------------------------------
# 3. Custom metric: keyword match score
# ---------------------------------------------------------------------------

@autorag_metric(fields_to_check=["generated_texts", "generation_gt"])
def keyword_match_score(metric_input: MetricInput) -> float:
    """Check how many keywords from the ground truth appear in the generated text."""
    generated = metric_input.generated_texts.lower()
    gt_words = set(metric_input.generation_gt[0].lower().split())

    stop_words = {
        "the", "a", "an", "is", "are", "was", "were", "in", "on", "at",
        "to", "for", "of", "and", "or", "but", "not", "with", "by",
        "from", "as", "it", "this", "that",
    }
    keywords = gt_words - stop_words
    if not keywords:
        return 1.0
    matches = sum(1 for kw in keywords if kw in generated)
    return matches / len(keywords)


def create_keyword_match_metric() -> None:
    print("=" * 60)
    print("Custom Metric 1: keyword_match_score")
    print("=" * 60)
    print(f"\nFunction : {keyword_match_score.__name__}")
    print(f"Fields   : {keyword_match_score._fields_to_check}")
    print(f"Purpose  : {keyword_match_score.__doc__}")
    print()


# ---------------------------------------------------------------------------
# 4. Custom metric: length ratio score
# ---------------------------------------------------------------------------

@autorag_metric(fields_to_check=["generated_texts", "generation_gt"])
def length_ratio_score(metric_input: MetricInput) -> float:
    """Score how close the generated text length is to the ground truth length."""
    gen_len = len(metric_input.generated_texts.split())
    gt_len = len(metric_input.generation_gt[0].split())
    if gt_len == 0:
        return 0.0
    ratio = gen_len / gt_len
    return max(0.0, 1.0 - abs(1.0 - ratio))


def create_length_ratio_metric() -> None:
    print("=" * 60)
    print("Custom Metric 2: length_ratio_score")
    print("=" * 60)
    print(f"\nFunction : {length_ratio_score.__name__}")
    print(f"Fields   : {length_ratio_score._fields_to_check}")
    print(f"Purpose  : {length_ratio_score.__doc__}")
    print()


# ---------------------------------------------------------------------------
# 5. Test both custom metrics
# ---------------------------------------------------------------------------

def test_custom_metrics() -> None:
    print("=" * 60)
    print("Testing Custom Metrics")
    print("=" * 60)

    test_cases = [
        MetricInput(
            generated_texts="Python decorators modify function behavior using the @ syntax",
            generation_gt=["Decorators in Python are functions that modify the behavior of other functions using the @ symbol"],
        ),
        MetricInput(
            generated_texts="Machine learning is a subset of AI that learns from data",
            generation_gt=["Machine learning is a branch of artificial intelligence focused on building systems that learn from data"],
        ),
        MetricInput(
            generated_texts="The quick brown fox",
            generation_gt=["The quick brown fox jumps over the lazy dog near the river bank on a sunny afternoon"],
        ),
    ]

    for i, tc in enumerate(test_cases, 1):
        kw_score = keyword_match_score(tc)
        lr_score = length_ratio_score(tc)
        print(f"\nTest case {i}:")
        print(f"  Generated : {tc.generated_texts}")
        print(f"  Ground truth: {tc.generation_gt[0]}")
        print(f"  keyword_match_score : {kw_score:.3f}")
        print(f"  length_ratio_score  : {lr_score:.3f}")
    print()


# ---------------------------------------------------------------------------
# 6. Explain registration
# ---------------------------------------------------------------------------

def explain_registration() -> None:
    print("=" * 60)
    print("Registering Custom Metrics in AutoRAG")
    print("=" * 60)
    print("""
To use custom metrics in an AutoRAG evaluation:

1. Define your metric function with the @autorag_metric decorator:

   @autorag_metric(fields_to_check=["generated_texts", "generation_gt"])
   def keyword_match_score(metric_input: MetricInput) -> float:
       ...

2. Register it in the metric function dictionary:

   from autorag.evaluation.generation import GENERATION_METRIC_FUNC_DICT
   GENERATION_METRIC_FUNC_DICT["keyword_match_score"] = keyword_match_score

3. Reference it in your config.yaml:

   strategy:
     metrics:
       - metric_name: keyword_match_score

Note: Registration must happen at the code level before running
the evaluation. Import your custom metrics module early in the
script that calls autorag.evaluate().
""")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    explain_metric_system()
    show_built_in_metrics()
    create_keyword_match_metric()
    create_length_ratio_metric()
    test_custom_metrics()
    explain_registration()

    print("=" * 60)
    print("Lesson complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
