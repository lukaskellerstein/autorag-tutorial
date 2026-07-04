"""
L2-M1.2 — Embedding Model Comparison

This lesson compares different embedding models using AutoRAG to
systematically evaluate how embedding choice affects retrieval quality.
We test BAAI/bge-small-en-v1.5 (384 dimensions) against
all-mpnet-base-v2 (768 dimensions) on the same Python-concepts corpus
and analyze the tradeoffs between model size, dimensionality, and quality.
"""

import os
from typing import Any

import httpx
import pandas as pd


def create_sample_data() -> None:
    """Generate a 10-document Python corpus and 20 QA pairs, saved as parquet."""
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
    print(f"Created {len(qa_df)} QA pairs (2 per document)")
    print(f"Saved to: data/corpus.parquet ({os.path.getsize('data/corpus.parquet'):,} bytes)")
    print(f"Saved to: data/qa.parquet ({os.path.getsize('data/qa.parquet'):,} bytes)")


def check_ollama() -> bool:
    """Check if Ollama is running at localhost:11434."""
    try:
        response = httpx.get("http://localhost:11434/api/tags", timeout=5.0)
        if response.status_code == 200:
            models = response.json().get("models", [])
            model_names = [m.get("name", "") for m in models]
            print("Ollama is running.")
            print(f"Available models: {model_names}")
            return True
        print(f"Ollama responded with status {response.status_code}")
        return False
    except httpx.ConnectError:
        print("Ollama is NOT running at http://localhost:11434")
        print("Start it with: ollama serve")
        return False
    except httpx.TimeoutException:
        print("Ollama connection timed out.")
        return False


def explain_embedding_models() -> None:
    """Print a comparison table of the two embedding models under test."""
    print("Embedding models compared in this lesson:\n")
    print(f"{'| Model':<25} {'| Dimensions':<13} {'| Size':<9} {'| Notes':<39}|")
    print(f"|{'-' * 24}|{'-' * 12}|{'-' * 8}|{'-' * 38}|")
    print(f"{'| BAAI/bge-small-en-v1.5':<25} {'| 384':<13} {'| ~130MB':<9} {'| Lightweight, good for English':<39}|")
    print(f"{'| all-mpnet-base-v2':<25} {'| 768':<13} {'| ~420MB':<9} {'| Higher quality, larger dimensions':<39}|")
    print()
    print("Both models are from the sentence-transformers ecosystem and")
    print("produce normalized embeddings suitable for cosine similarity search.")


def explain_tradeoffs() -> None:
    """Print analysis of embedding model tradeoffs."""
    print("Key tradeoffs when choosing an embedding model:\n")

    print("1. Dimensions (384 vs 768)")
    print("   Higher dimensions capture more semantic nuance, allowing the")
    print("   model to distinguish between subtly different meanings. However,")
    print("   higher dimensions require more storage per vector and increase")
    print("   the computational cost of similarity search.\n")

    print("2. Model Size (~130MB vs ~420MB)")
    print("   Larger models generally produce higher-quality embeddings because")
    print("   they have more parameters to encode semantic relationships. The")
    print("   tradeoff is slower indexing time and higher memory usage during")
    print("   embedding generation.\n")

    print("3. Language Coverage")
    print("   bge-small-en-v1.5 is optimized for English text. For multilingual")
    print("   corpora, consider models like paraphrase-multilingual-MiniLM-L12-v2")
    print("   or multilingual-e5-large. English-only models tend to outperform")
    print("   multilingual models on English benchmarks.\n")

    print("4. Domain Specificity")
    print("   General-purpose models work well for broad content. For specialized")
    print("   domains (legal, medical, scientific), fine-tuned embeddings can")
    print("   significantly improve retrieval quality. Consider fine-tuning on")
    print("   your own data if general models underperform.")


def run_evaluation() -> None:
    """Run the AutoRAG evaluation comparing both embedding models."""
    from autorag.evaluator import Evaluator

    evaluator = Evaluator(
        qa_data_path="data/qa.parquet",
        corpus_data_path="data/corpus.parquet",
        project_dir="./results",
    )
    evaluator.start_trial("config.yaml")
    print("Evaluation complete. Results saved to ./results/")


def compare_embeddings() -> None:
    """Load evaluation results and compare embedding model performance."""
    summary_path = "results/0/summary.csv"

    if not os.path.exists(summary_path):
        print(f"Results file not found: {summary_path}")
        print("Run the evaluation first (Step 4) to generate results.")
        print("\nShowing what to expect when results are available:")
        print("  - summary.csv contains per-module retrieval metrics")
        print("  - Compare retrieval_f1, retrieval_recall, retrieval_precision")
        print("  - The module with higher scores used the better embedding model")
        return

    summary_df = pd.read_csv(summary_path)
    print("Evaluation Results Summary")
    print("-" * 50)
    print(summary_df.to_string(index=False))

    print("\n" + "-" * 50)
    print("Embedding Model Comparison:")

    retrieval_cols = [c for c in summary_df.columns if c.startswith("retrieval_")]
    if retrieval_cols:
        print(f"\nRetrieval metrics found: {retrieval_cols}")
        for col in retrieval_cols:
            best_idx = summary_df[col].idxmax()
            best_val = summary_df[col].max()
            print(f"  {col}: best = {best_val:.4f} (row {best_idx})")
    else:
        print("No retrieval metric columns found in summary.")


def print_recommendations() -> None:
    """Print practical recommendations for embedding model selection."""
    print("Recommendations:\n")

    print("For prototyping and development:")
    print("  Use bge-small-en-v1.5 (384 dims, ~130MB)")
    print("  - Fast to download and index")
    print("  - Good enough to validate your RAG pipeline logic")
    print("  - Low memory footprint for local development\n")

    print("For production deployments:")
    print("  Use all-mpnet-base-v2 (768 dims, ~420MB) or larger models")
    print("  - Higher retrieval quality justifies the extra resources")
    print("  - Consider all-MiniLM-L12-v2 as a middle ground (384 dims, better quality)\n")

    print("For domain-specific applications:")
    print("  Consider fine-tuned embedding models")
    print("  - Medical: PubMedBERT-based embeddings")
    print("  - Legal: legal-bert-based embeddings")
    print("  - Use AutoRAG to verify fine-tuned models actually outperform general ones\n")

    print("Cost considerations:")
    print("  - 768-dim vectors use 2x the storage of 384-dim vectors")
    print("  - Larger vector indices require more RAM for fast search")
    print("  - For millions of documents, storage and compute costs scale linearly")
    print("  - Always benchmark on YOUR data — the best model depends on your domain")


def main() -> None:
    """Run all steps of the embedding model comparison lesson."""
    print("=" * 60)
    print("L2-M1.2 — Embedding Model Comparison")
    print("=" * 60)

    print("\n" + "=" * 60)
    print("Step 1: Prepare evaluation data")
    print("=" * 60)
    create_sample_data()

    print("\n" + "=" * 60)
    print("Step 2: Check Ollama availability")
    print("=" * 60)
    ollama_ok = check_ollama()

    print("\n" + "=" * 60)
    print("Step 3: Embedding models under comparison")
    print("=" * 60)
    explain_embedding_models()

    print("\n" + "=" * 60)
    print("Step 4: Embedding model tradeoffs")
    print("=" * 60)
    explain_tradeoffs()

    print("\n" + "=" * 60)
    print("Step 5: Run AutoRAG evaluation")
    print("=" * 60)
    if ollama_ok:
        run_evaluation()
    else:
        print("Skipping evaluation — Ollama is not running.")
        print("Start Ollama and re-run this lesson to execute the evaluation.")

    print("\n" + "=" * 60)
    print("Step 6: Compare embedding results")
    print("=" * 60)
    compare_embeddings()

    print("\n" + "=" * 60)
    print("Step 7: Recommendations")
    print("=" * 60)
    print_recommendations()

    print("\n" + "=" * 60)
    print("Done! Review the results to decide which embedding model")
    print("works best for your use case.")
    print("=" * 60)


if __name__ == "__main__":
    main()
