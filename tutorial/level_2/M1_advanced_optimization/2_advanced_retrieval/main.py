"""
L2-M1.2 — Advanced Retrieval Strategies

This lesson explores hybrid retrieval (BM25 + vector search), fusion
strategies (RRF vs Convex Combination), and reranking pipelines. We
configure AutoRAG to evaluate these combinations and identify the
optimal multi-stage retrieval strategy for our Python-concepts corpus.
"""

import os
from pathlib import Path
from typing import Any

import httpx
import pandas as pd


def create_sample_data() -> None:
    """Generate a 10-document Python-concepts corpus and 20 QA pairs."""
    documents: list[dict[str, Any]] = [
        {
            "doc_id": "doc_001",
            "contents": (
                "Variables in Python are names that refer to objects stored in memory. "
                "Unlike statically typed languages, Python variables do not require explicit "
                "type declarations. A variable is created the moment you assign a value to it "
                "using the assignment operator (=). Python supports multiple assignment, "
                "allowing you to assign values to several variables in a single line. "
                "Variable names must start with a letter or underscore, followed by letters, "
                "digits, or underscores. Python is dynamically typed, meaning a variable can "
                "be reassigned to a value of a different type at any time. Common built-in "
                "types include int, float, str, bool, list, dict, tuple, and set."
            ),
            "metadata": {"last_modified_datetime": "2024-01-15T10:00:00", "prev_id": None, "next_id": None},
        },
        {
            "doc_id": "doc_002",
            "contents": (
                "Functions in Python are reusable blocks of code defined with the def keyword. "
                "They accept parameters, perform operations, and optionally return values using "
                "the return statement. Functions support default parameter values, keyword "
                "arguments, and variable-length argument lists using *args and **kwargs. "
                "Python also supports lambda expressions for creating small anonymous functions "
                "inline. Functions are first-class objects in Python, meaning they can be passed "
                "as arguments to other functions, returned from functions, and assigned to "
                "variables. Docstrings placed immediately after the function definition provide "
                "documentation accessible via the help() function. Functions help organize code, "
                "reduce repetition, and improve readability."
            ),
            "metadata": {"last_modified_datetime": "2024-01-15T10:00:00", "prev_id": None, "next_id": None},
        },
        {
            "doc_id": "doc_003",
            "contents": (
                "Classes in Python provide a means of bundling data and functionality together. "
                "A class is defined using the class keyword and serves as a blueprint for "
                "creating objects (instances). Each class can have an __init__ method that "
                "initializes instance attributes when an object is created. Methods are functions "
                "defined within a class that operate on instance data via the self parameter. "
                "Python supports inheritance, allowing a class to derive from one or more parent "
                "classes. The super() function enables calling methods from a parent class. "
                "Python also supports class variables shared across all instances, instance "
                "variables unique to each object, and special dunder methods like __str__ and "
                "__repr__ for customizing object behavior."
            ),
            "metadata": {"last_modified_datetime": "2024-01-15T10:00:00", "prev_id": None, "next_id": None},
        },
        {
            "doc_id": "doc_004",
            "contents": (
                "Decorators in Python are a powerful pattern for modifying or extending the "
                "behavior of functions and methods without changing their source code. A "
                "decorator is a callable that takes a function as input and returns a new "
                "function. They are applied using the @decorator_name syntax placed above the "
                "function definition. Common built-in decorators include @staticmethod, "
                "@classmethod, and @property. Decorators can accept arguments by using nested "
                "functions: an outer function receives the arguments, a middle function receives "
                "the decorated function, and an inner function wraps the execution. The "
                "functools.wraps decorator preserves the original function's metadata when "
                "creating wrapper functions. Decorators are widely used for logging, "
                "authentication, caching, and input validation."
            ),
            "metadata": {"last_modified_datetime": "2024-01-15T10:00:00", "prev_id": None, "next_id": None},
        },
        {
            "doc_id": "doc_005",
            "contents": (
                "Generators in Python are special functions that use the yield keyword to "
                "produce a sequence of values lazily, one at a time. Unlike regular functions "
                "that return a single value and terminate, generators pause execution after each "
                "yield and resume where they left off when the next value is requested. This "
                "makes them memory-efficient for processing large datasets or infinite sequences. "
                "Generator expressions provide a concise syntax similar to list comprehensions "
                "but enclosed in parentheses. The itertools module provides many useful generator "
                "functions like chain, islice, and count. Generators implement the iterator "
                "protocol, supporting next() calls and for-loop iteration. The send() method "
                "allows sending values back into a generator, enabling coroutine-like behavior."
            ),
            "metadata": {"last_modified_datetime": "2024-01-15T10:00:00", "prev_id": None, "next_id": None},
        },
        {
            "doc_id": "doc_006",
            "contents": (
                "List comprehensions in Python provide a concise way to create lists based on "
                "existing iterables. The basic syntax is [expression for item in iterable], with "
                "optional conditional filtering using an if clause. They replace multi-line "
                "for-loop patterns with a single readable expression. Python also supports "
                "dictionary comprehensions ({key: value for item in iterable}), set "
                "comprehensions ({expression for item in iterable}), and nested comprehensions "
                "for working with multi-dimensional data. Comprehensions are generally faster "
                "than equivalent for loops because they are optimized at the bytecode level. "
                "However, overly complex comprehensions can reduce readability and should be "
                "refactored into regular loops or helper functions when they span multiple "
                "conditions or transformations."
            ),
            "metadata": {"last_modified_datetime": "2024-01-15T10:00:00", "prev_id": None, "next_id": None},
        },
        {
            "doc_id": "doc_007",
            "contents": (
                "Error handling in Python uses try-except blocks to catch and respond to "
                "exceptions that occur during program execution. The try block contains code "
                "that might raise an exception, while except blocks specify handlers for "
                "specific exception types. The else clause runs when no exception occurs, and "
                "the finally clause executes regardless of whether an exception was raised. "
                "Python has a rich hierarchy of built-in exceptions, with BaseException at the "
                "root and Exception as the base for most user-catchable errors. Custom "
                "exceptions are created by subclassing Exception. The raise statement allows "
                "explicitly raising exceptions, and the assert statement raises AssertionError "
                "when a condition is False. Context managers (with statement) provide a clean "
                "pattern for resource management and error handling."
            ),
            "metadata": {"last_modified_datetime": "2024-01-15T10:00:00", "prev_id": None, "next_id": None},
        },
        {
            "doc_id": "doc_008",
            "contents": (
                "Modules and packages in Python organize code into reusable units. A module is "
                "a single Python file containing definitions and statements. Packages are "
                "directories containing an __init__.py file and one or more modules. The import "
                "statement loads modules, with variations like import module, from module import "
                "name, and from module import *. Python searches for modules in sys.path, which "
                "includes the script directory, installed packages, and standard library paths. "
                "The __name__ variable equals '__main__' when a module is run directly and the "
                "module name when imported. Relative imports use dots to reference modules "
                "within the same package. The standard library provides hundreds of modules for "
                "file I/O, networking, math, data processing, and more."
            ),
            "metadata": {"last_modified_datetime": "2024-01-15T10:00:00", "prev_id": None, "next_id": None},
        },
        {
            "doc_id": "doc_009",
            "contents": (
                "Virtual environments in Python create isolated spaces for project dependencies, "
                "preventing conflicts between different projects that require different package "
                "versions. The venv module (built into Python 3.3+) creates virtual environments "
                "with their own Python interpreter and pip installation. Activating a virtual "
                "environment modifies the PATH so that python and pip commands use the "
                "environment's versions. Tools like uv, pipenv, and poetry provide enhanced "
                "dependency management with lock files for reproducible builds. The "
                "requirements.txt file lists project dependencies and can be generated with pip "
                "freeze. Virtual environments are essential for maintaining clean, reproducible "
                "development setups and avoiding the problems of global package installation."
            ),
            "metadata": {"last_modified_datetime": "2024-01-15T10:00:00", "prev_id": None, "next_id": None},
        },
        {
            "doc_id": "doc_010",
            "contents": (
                "Type hints in Python provide optional static type annotations that improve code "
                "readability and enable static analysis tools. Introduced in PEP 484 and "
                "expanded in subsequent PEPs, type hints use the colon syntax for variables "
                "(name: str) and arrow syntax for function return types (def func() -> int). "
                "The typing module provides complex types like List, Dict, Optional, Union, "
                "Tuple, and Callable. Generic types allow creating type-parameterized classes "
                "and functions. Type checkers like mypy and pyright analyze code without "
                "executing it, catching type-related bugs early. Python 3.10+ introduced the "
                "union operator (X | Y) as a cleaner alternative to Union[X, Y]. Type hints "
                "are not enforced at runtime by default but can be checked using libraries like "
                "pydantic and beartype."
            ),
            "metadata": {"last_modified_datetime": "2024-01-15T10:00:00", "prev_id": None, "next_id": None},
        },
    ]

    qa_pairs: list[dict[str, Any]] = [
        {
            "qid": "q_001",
            "query": "What naming rules must Python variables follow?",
            "retrieval_gt": [["doc_001"]],
            "generation_gt": [
                "Variable names must start with a letter or underscore, followed by "
                "letters, digits, or underscores."
            ],
        },
        {
            "qid": "q_002",
            "query": "Why is Python considered a dynamically typed language?",
            "retrieval_gt": [["doc_001"]],
            "generation_gt": [
                "Python is dynamically typed because a variable can be reassigned to "
                "a value of a different type at any time without explicit type declarations."
            ],
        },
        {
            "qid": "q_003",
            "query": "What are *args and **kwargs used for in Python functions?",
            "retrieval_gt": [["doc_002"]],
            "generation_gt": [
                "The *args and **kwargs syntax allows functions to accept variable-length "
                "argument lists — *args for positional arguments and **kwargs for keyword arguments."
            ],
        },
        {
            "qid": "q_004",
            "query": "In what sense are Python functions first-class objects?",
            "retrieval_gt": [["doc_002"]],
            "generation_gt": [
                "Functions are first-class objects because they can be passed as arguments "
                "to other functions, returned from functions, and assigned to variables."
            ],
        },
        {
            "qid": "q_005",
            "query": "What is the purpose of the __init__ method in a Python class?",
            "retrieval_gt": [["doc_003"]],
            "generation_gt": [
                "The __init__ method initializes instance attributes when an object is "
                "created from the class."
            ],
        },
        {
            "qid": "q_006",
            "query": "How does Python support inheritance in classes?",
            "retrieval_gt": [["doc_003"]],
            "generation_gt": [
                "Python supports inheritance by allowing a class to derive from one or "
                "more parent classes. The super() function enables calling methods from "
                "a parent class."
            ],
        },
        {
            "qid": "q_007",
            "query": "What are some common built-in decorators in Python?",
            "retrieval_gt": [["doc_004"]],
            "generation_gt": [
                "Common built-in decorators include @staticmethod, @classmethod, and @property."
            ],
        },
        {
            "qid": "q_008",
            "query": "How do decorators modify function behavior without changing source code?",
            "retrieval_gt": [["doc_004"]],
            "generation_gt": [
                "A decorator is a callable that takes a function as input and returns a "
                "new function. Applied with the @decorator_name syntax, it wraps the "
                "original function to add behavior like logging or caching."
            ],
        },
        {
            "qid": "q_009",
            "query": "What keyword distinguishes a generator function from a regular function?",
            "retrieval_gt": [["doc_005"]],
            "generation_gt": [
                "The yield keyword distinguishes a generator function from a regular function. "
                "Generators use yield to produce values lazily, one at a time."
            ],
        },
        {
            "qid": "q_010",
            "query": "Why are generators considered memory-efficient?",
            "retrieval_gt": [["doc_005"]],
            "generation_gt": [
                "Generators are memory-efficient because they produce values lazily, one at "
                "a time, pausing execution after each yield rather than storing the entire "
                "sequence in memory."
            ],
        },
        {
            "qid": "q_011",
            "query": "What is the basic syntax of a Python list comprehension?",
            "retrieval_gt": [["doc_006"]],
            "generation_gt": [
                "The basic syntax is [expression for item in iterable], with optional "
                "conditional filtering using an if clause."
            ],
        },
        {
            "qid": "q_012",
            "query": "When should you avoid using list comprehensions in Python?",
            "retrieval_gt": [["doc_006"]],
            "generation_gt": [
                "Overly complex comprehensions that span multiple conditions or "
                "transformations should be refactored into regular loops or helper "
                "functions to maintain readability."
            ],
        },
        {
            "qid": "q_013",
            "query": "What are the four clauses of a try-except block in Python?",
            "retrieval_gt": [["doc_007"]],
            "generation_gt": [
                "The four clauses are: try (code that might raise an exception), except "
                "(handlers for specific exception types), else (runs when no exception "
                "occurs), and finally (executes regardless of whether an exception was raised)."
            ],
        },
        {
            "qid": "q_014",
            "query": "How do context managers relate to error handling in Python?",
            "retrieval_gt": [["doc_007"]],
            "generation_gt": [
                "Context managers (using the with statement) provide a clean pattern for "
                "resource management and error handling, ensuring resources are properly "
                "released even if exceptions occur."
            ],
        },
        {
            "qid": "q_015",
            "query": "What is the difference between a module and a package in Python?",
            "retrieval_gt": [["doc_008"]],
            "generation_gt": [
                "A module is a single Python file containing definitions and statements. "
                "A package is a directory containing an __init__.py file and one or more modules."
            ],
        },
        {
            "qid": "q_016",
            "query": "How does Python determine where to find modules when importing?",
            "retrieval_gt": [["doc_008"]],
            "generation_gt": [
                "Python searches for modules in sys.path, which includes the script directory, "
                "installed packages, and standard library paths."
            ],
        },
        {
            "qid": "q_017",
            "query": "What Python module is used to create virtual environments?",
            "retrieval_gt": [["doc_009"]],
            "generation_gt": [
                "The venv module, built into Python 3.3+, creates virtual environments with "
                "their own Python interpreter and pip installation."
            ],
        },
        {
            "qid": "q_018",
            "query": "Why are virtual environments essential for Python development?",
            "retrieval_gt": [["doc_009"]],
            "generation_gt": [
                "Virtual environments create isolated spaces for project dependencies, "
                "preventing conflicts between different projects that require different "
                "package versions and maintaining reproducible development setups."
            ],
        },
        {
            "qid": "q_019",
            "query": "What syntax does Python use for type hints on variables and return types?",
            "retrieval_gt": [["doc_010"]],
            "generation_gt": [
                "Python uses the colon syntax for variables (name: str) and arrow syntax "
                "for function return types (def func() -> int)."
            ],
        },
        {
            "qid": "q_020",
            "query": "How do type hints improve Python code quality without runtime enforcement?",
            "retrieval_gt": [["doc_010"]],
            "generation_gt": [
                "Type hints enable static analysis tools like mypy and pyright to analyze "
                "code without executing it, catching type-related bugs early. They also "
                "improve code readability for developers."
            ],
        },
    ]

    os.makedirs("data", exist_ok=True)

    corpus_df = pd.DataFrame(documents)
    qa_df = pd.DataFrame(qa_pairs)

    corpus_df.to_parquet("data/corpus.parquet", index=False)
    qa_df.to_parquet("data/qa.parquet", index=False)

    print(f"Created corpus with {len(corpus_df)} documents")
    print(f"  Columns: {list(corpus_df.columns)}")
    print(f"  Sample doc_id: {corpus_df.iloc[0]['doc_id']}")
    print(f"  Sample preview: {corpus_df.iloc[0]['contents'][:80]}...")
    print(f"\nCreated {len(qa_df)} QA pairs")
    print(f"  Columns: {list(qa_df.columns)}")
    print(f"  Sample query: {qa_df.iloc[0]['query']}")
    print(f"  Sample retrieval_gt: {qa_df.iloc[0]['retrieval_gt']}")
    print(f"\nSaved to data/corpus.parquet and data/qa.parquet")


def check_ollama() -> bool:
    """Check if Ollama is running at localhost:11434."""
    try:
        response = httpx.get("http://localhost:11434/api/tags", timeout=5.0)
        if response.status_code == 200:
            models = response.json().get("models", [])
            model_names = [m.get("name", "") for m in models]
            print(f"Ollama is running with {len(models)} model(s): {model_names}")
            return True
        print(f"Ollama responded with status {response.status_code}")
        return False
    except httpx.ConnectError:
        print("Ollama is NOT running at http://localhost:11434")
        print("Start it with: ollama serve")
        print("Then pull the model: ollama pull gemma4:e2b")
        return False
    except httpx.TimeoutException:
        print("Ollama connection timed out")
        return False


def explain_retrieval_strategies() -> None:
    """Explain the four retrieval strategies evaluated in this lesson."""
    print("1. BM25 (Lexical Retrieval)")
    print("-" * 40)
    print("  BM25 is a TF-IDF-based ranking function that scores documents by")
    print("  term frequency and inverse document frequency. It excels at exact")
    print("  keyword matching — if the query contains the same words as the")
    print("  document, BM25 will find it. No embeddings or GPU are needed,")
    print("  making it fast and lightweight. However, BM25 misses semantic")
    print("  similarity: 'car' and 'automobile' are unrelated to BM25.")
    print()
    print("2. Vector Search (Semantic Retrieval)")
    print("-" * 40)
    print("  Vector search encodes queries and documents as dense embeddings")
    print("  in a high-dimensional space. Documents close to the query in this")
    print("  space are retrieved, capturing meaning rather than exact words.")
    print("  This requires an embedding model (e.g., BGE-small) and a vector")
    print("  database (e.g., Chroma, Qdrant). It handles synonyms and")
    print("  paraphrases well but can miss exact keyword matches.")
    print()
    print("3. Hybrid RRF (Reciprocal Rank Fusion)")
    print("-" * 40)
    print("  RRF combines BM25 and vector rankings without needing score")
    print("  normalization. For each document, RRF computes:")
    print("    score = sum(1 / (k + rank_i)) across retrievers")
    print("  The parameter k (weight_range) controls how much top ranks are")
    print("  favored over lower ranks. Higher k values flatten the ranking")
    print("  differences. RRF is robust and parameter-light.")
    print()
    print("4. Hybrid CC (Convex Combination)")
    print("-" * 40)
    print("  CC blends the normalized scores from BM25 and vector search:")
    print("    score = w * bm25_score + (1-w) * vector_score")
    print("  Scores must be normalized first. Normalization methods:")
    print("    - mm  (min-max):         scale to [0, 1]")
    print("    - tmm (trimmed min-max): clip outliers, then min-max")
    print("    - z   (z-score):         standardize to mean=0, std=1")
    print("  The weight w (weight_range) controls the BM25-vs-vector balance.")
    print("  CC gives fine-grained control but is sensitive to normalization.")


def explain_reranking() -> None:
    """Explain the reranking strategies evaluated in this lesson."""
    print("1. Pass Reranker (No-Op Baseline)")
    print("-" * 40)
    print("  The pass reranker keeps the original retrieval order unchanged.")
    print("  It serves as a baseline: if a neural reranker does not improve")
    print("  metrics over pass_reranker, the reranking stage adds latency")
    print("  without benefit and should be removed.")
    print()
    print("2. FlashRank Reranker")
    print("-" * 40)
    print("  FlashRank is a lightweight neural reranker based on distilled")
    print("  cross-encoder models. It scores each (query, passage) pair")
    print("  independently and re-sorts by relevance score. FlashRank is")
    print("  optimized for speed — typically 5-10x faster than full")
    print("  cross-encoders — while retaining most of the accuracy gains.")
    print("  Good default choice for production pipelines.")
    print()
    print("3. Sentence Transformer Reranker (Cross-Encoder)")
    print("-" * 40)
    print("  Cross-encoder models process the query and passage together")
    print("  through a transformer, producing a single relevance score.")
    print("  This joint encoding is more accurate than bi-encoder retrieval")
    print("  but much slower (O(n) forward passes for n candidates).")
    print("  Models like ms-marco-MiniLM are popular choices. Best for")
    print("  small candidate sets (top-10 to top-50).")
    print()
    print("4. ColBERT Reranker")
    print("-" * 40)
    print("  ColBERT uses late interaction: it encodes query and passage")
    print("  tokens independently, then computes fine-grained token-level")
    print("  similarity via MaxSim. This gives near-cross-encoder accuracy")
    print("  with much better efficiency, since passage embeddings can be")
    print("  precomputed. A strong middle ground between speed and quality.")
    print()
    print("All Available Reranker Modules in AutoRAG:")
    print("-" * 40)
    rerankers = [
        "monot5", "tart", "upr", "koreranker", "pass_reranker",
        "cohere_reranker", "rankgpt", "jina_reranker", "colbert_reranker",
        "sentence_transformer_reranker", "flag_embedding_reranker",
        "flag_embedding_llm_reranker", "time_reranker", "openvino_reranker",
        "voyageai_reranker", "mixedbreadai_reranker", "flashrank_reranker",
    ]
    for i, name in enumerate(rerankers, 1):
        print(f"  {i:2d}. {name}")


def explain_multi_stage() -> None:
    """Explain the multi-stage retrieval pipeline concept."""
    print("Multi-Stage Retrieval Pipeline")
    print("-" * 40)
    print()
    print("  The key insight: retrieval and ranking have different tradeoffs.")
    print()
    print("  Stage 1 — Coarse Retrieval (Fast, High Recall)")
    print("    Retrieve a broad set of candidates (e.g., top-50) using fast")
    print("    methods like BM25, vector search, or hybrid fusion. The goal")
    print("    is high recall: ensure the relevant documents are somewhere")
    print("    in the candidate set, even if irrelevant ones are mixed in.")
    print("    Speed matters here because we search the entire corpus.")
    print()
    print("  Stage 2 — Reranking (Slower, High Precision)")
    print("    Re-score the small candidate set (e.g., 50 passages) with a")
    print("    more powerful model. Neural rerankers like FlashRank or")
    print("    cross-encoders examine each (query, passage) pair in detail.")
    print("    The goal is high precision: push the most relevant documents")
    print("    to the top. Speed is less critical because the set is small.")
    print()
    print("  Why This Works")
    print("    - Coarse retrieval is fast but imprecise (may rank a mediocre")
    print("      match above a great one).")
    print("    - Neural rerankers are precise but too slow to run over the")
    print("      entire corpus.")
    print("    - Combining them gives you the best of both: fast corpus-wide")
    print("      search followed by accurate fine-grained ranking.")
    print("    - AutoRAG evaluates which combination of retriever + reranker")
    print("      produces the best end-to-end metrics for YOUR data.")


def run_evaluation() -> None:
    """Run the AutoRAG evaluation with the hybrid retrieval config."""
    from autorag.evaluator import Evaluator

    evaluator = Evaluator(
        qa_data_path="data/qa.parquet",
        corpus_data_path="data/corpus.parquet",
        project_dir="./results",
    )
    evaluator.start_trial("config.yaml")
    print("Evaluation complete. Results saved to ./results/")


def compare_results() -> None:
    """Load and display the evaluation results summary."""
    summary_path = Path("results/0/summary.csv")

    if not summary_path.exists():
        print(f"Results file not found: {summary_path}")
        print("Run the full evaluation first (this requires Ollama with gemma4:e2b).")
        print("The evaluation will produce a summary.csv with the best configuration")
        print("at each pipeline node and the overall winning strategy.")
        return

    summary_df = pd.read_csv(summary_path)

    print(f"Results loaded from {summary_path}")
    print(f"Columns: {list(summary_df.columns)}")
    print()

    print("Full Summary Table:")
    print("-" * 40)
    print(summary_df.to_string(index=False))
    print()

    # Extract winning configurations if columns are present
    if "best_module_name" in summary_df.columns:
        print("Winning Configurations by Node:")
        print("-" * 40)
        for _, row in summary_df.iterrows():
            node = row.get("node_type", "unknown")
            module = row.get("best_module_name", "unknown")
            print(f"  {node}: {module}")
    elif "module_name" in summary_df.columns:
        print("Module Results:")
        print("-" * 40)
        for _, row in summary_df.iterrows():
            module = row.get("module_name", "unknown")
            print(f"  Module: {module}")
            for col in summary_df.columns:
                if col not in ("module_name", "node_type", "module_type"):
                    print(f"    {col}: {row[col]}")

    print()
    print("Interpretation Guide:")
    print("  - Higher retrieval_f1 / retrieval_recall = better retrieval")
    print("  - Compare hybrid_rrf vs hybrid_cc to see which fusion wins")
    print("  - Compare pass_reranker vs flashrank_reranker for reranking value")
    print("  - The best pipeline balances retrieval quality and latency")


def main() -> None:
    """Run all steps of the advanced retrieval strategies lesson."""
    print("=" * 60)
    print("L2-M1.2 — Advanced Retrieval Strategies")
    print("=" * 60)

    print("\n" + "=" * 60)
    print("Step 1: Create sample evaluation data")
    print("=" * 60)
    create_sample_data()

    print("\n" + "=" * 60)
    print("Step 2: Check Ollama availability")
    print("=" * 60)
    ollama_ok = check_ollama()

    print("\n" + "=" * 60)
    print("Step 3: Retrieval strategies explained")
    print("=" * 60)
    explain_retrieval_strategies()

    print("\n" + "=" * 60)
    print("Step 4: Reranking strategies explained")
    print("=" * 60)
    explain_reranking()

    print("\n" + "=" * 60)
    print("Step 5: Multi-stage retrieval pipeline")
    print("=" * 60)
    explain_multi_stage()

    print("\n" + "=" * 60)
    print("Step 6: Run AutoRAG evaluation")
    print("=" * 60)
    if ollama_ok:
        print("Starting evaluation (this may take several minutes)...")
        run_evaluation()
    else:
        print("Skipping evaluation — Ollama is not running.")
        print("To run the full evaluation:")
        print("  1. Start Ollama:  ollama serve")
        print("  2. Pull model:    ollama pull gemma4:e2b")
        print("  3. Re-run:        uv run python main.py")

    print("\n" + "=" * 60)
    print("Step 7: Compare results")
    print("=" * 60)
    compare_results()

    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)


if __name__ == "__main__":
    main()
