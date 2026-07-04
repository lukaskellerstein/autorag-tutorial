"""
L1-M2.2 — Creating QA Evaluation Datasets

This lesson demonstrates how to create high-quality QA evaluation datasets
for AutoRAG. We build a corpus of Python programming documents and craft
QA pairs that link questions to their source documents, then explore
AutoRAG's fluent API for generating QA pairs at scale.
"""

import os
from typing import Any

import pandas as pd


def create_sample_corpus() -> pd.DataFrame:
    """Create 10 documents about Python programming concepts."""
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
            "metadata": {"last_modified_datetime": "2024-01-16T11:00:00", "prev_id": None, "next_id": None},
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
            "metadata": {"last_modified_datetime": "2024-01-17T09:30:00", "prev_id": None, "next_id": None},
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
            "metadata": {"last_modified_datetime": "2024-01-18T14:00:00", "prev_id": None, "next_id": None},
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
            "metadata": {"last_modified_datetime": "2024-01-19T08:45:00", "prev_id": None, "next_id": None},
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
            "metadata": {"last_modified_datetime": "2024-01-20T13:15:00", "prev_id": None, "next_id": None},
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
            "metadata": {"last_modified_datetime": "2024-01-21T10:30:00", "prev_id": None, "next_id": None},
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
            "metadata": {"last_modified_datetime": "2024-01-22T15:00:00", "prev_id": None, "next_id": None},
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
            "metadata": {"last_modified_datetime": "2024-01-23T11:45:00", "prev_id": None, "next_id": None},
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
            "metadata": {"last_modified_datetime": "2024-01-24T09:00:00", "prev_id": None, "next_id": None},
        },
    ]

    corpus_df = pd.DataFrame(documents)
    print(f"Created corpus with {len(corpus_df)} documents")
    print(f"Columns: {list(corpus_df.columns)}")
    print(f"\nSample document (doc_001):")
    print(f"  Content preview: {corpus_df.iloc[0]['contents'][:80]}...")
    print(f"  Metadata: {corpus_df.iloc[0]['metadata']}")
    return corpus_df


def create_qa_dataset(corpus_df: pd.DataFrame) -> pd.DataFrame:
    """Create 20 QA pairs — 2 per document (one factual, one conceptual)."""
    qa_pairs: list[dict[str, Any]] = [
        # doc_001 — Variables
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
        # doc_002 — Functions
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
        # doc_003 — Classes
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
        # doc_004 — Decorators
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
        # doc_005 — Generators
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
        # doc_006 — List Comprehensions
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
        # doc_007 — Error Handling
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
        # doc_008 — Modules and Packages
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
        # doc_009 — Virtual Environments
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
        # doc_010 — Type Hints
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

    qa_df = pd.DataFrame(qa_pairs)
    print(f"Created {len(qa_df)} QA pairs")
    print(f"Columns: {list(qa_df.columns)}")
    print(f"\nSample QA pair (q_001):")
    print(f"  Query: {qa_df.iloc[0]['query']}")
    print(f"  Retrieval GT: {qa_df.iloc[0]['retrieval_gt']}")
    print(f"  Generation GT: {qa_df.iloc[0]['generation_gt'][0][:80]}...")
    return qa_df


def review_qa_quality(qa_df: pd.DataFrame, corpus_df: pd.DataFrame) -> None:
    """Print quality statistics and validate the QA dataset."""
    print(f"Total QA pairs: {len(qa_df)}")

    # Average query length
    word_counts = qa_df["query"].apply(lambda q: len(q.split()))
    print(f"Average query length: {word_counts.mean():.1f} words")
    print(f"  Min: {word_counts.min()} words, Max: {word_counts.max()} words")

    # Unique documents referenced
    all_doc_ids: set[str] = set()
    for gt_list in qa_df["retrieval_gt"]:
        for doc_ids in gt_list:
            all_doc_ids.update(doc_ids)
    print(f"Unique documents referenced: {len(all_doc_ids)} / {len(corpus_df)}")

    # Check for duplicate queries
    duplicates = qa_df["query"].duplicated().sum()
    print(f"Duplicate queries: {duplicates}")
    if duplicates == 0:
        print("  OK — no duplicates found")
    else:
        print("  WARNING — duplicate queries detected!")

    # Validate retrieval_gt doc_ids exist in corpus
    corpus_doc_ids = set(corpus_df["doc_id"].tolist())
    missing_refs: list[str] = []
    for _, row in qa_df.iterrows():
        for doc_ids in row["retrieval_gt"]:
            for doc_id in doc_ids:
                if doc_id not in corpus_doc_ids:
                    missing_refs.append(doc_id)
    if not missing_refs:
        print("Retrieval GT validation: OK — all doc_ids exist in corpus")
    else:
        print(f"Retrieval GT validation: FAILED — missing doc_ids: {missing_refs}")

    # Check for empty generation_gt
    empty_gt = qa_df["generation_gt"].apply(
        lambda gt: len(gt) == 0 or all(len(s.strip()) == 0 for s in gt)
    ).sum()
    if empty_gt == 0:
        print("Generation GT validation: OK — no empty answers")
    else:
        print(f"Generation GT validation: WARNING — {empty_gt} empty answers found")


def split_train_test(qa_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split QA dataset 80/20 into train and test sets."""
    test_df = qa_df.sample(frac=0.2, random_state=42)
    train_df = qa_df.drop(test_df.index)

    print(f"Total QA pairs: {len(qa_df)}")
    print(f"Train set: {len(train_df)} pairs ({len(train_df)/len(qa_df)*100:.0f}%)")
    print(f"Test set:  {len(test_df)} pairs ({len(test_df)/len(qa_df)*100:.0f}%)")
    print(f"\nTrain QIDs: {sorted(train_df['qid'].tolist())}")
    print(f"Test QIDs:  {sorted(test_df['qid'].tolist())}")

    return train_df, test_df


def explain_llm_generation() -> None:
    """Explain AutoRAG's fluent API for QA generation."""
    print("AutoRAG provides a fluent API for generating QA pairs at scale.")
    print("The pipeline chains sampling, query generation, filtering, and")
    print("answer generation into a single expression.")
    print()
    print("--- Sampling Methods ---")
    print("  random_single_hop(corpus_df, n=100)")
    print("    -> randomly sample n passages for question generation")
    print("  range_single_hop(corpus_df, idx_range)")
    print("    -> sample a specific index range of passages")
    print()
    print("--- Query Generation Types ---")
    print("  factoid_query_gen")
    print("    -> factual questions with specific answers")
    print("  concept_completion_query_gen")
    print("    -> questions about concepts and definitions")
    print("  two_hop_incremental")
    print("    -> multi-hop questions requiring two passages")
    print()
    print("--- Query Evolving (increase difficulty) ---")
    print("  reasoning_evolve_ragas")
    print("    -> add reasoning steps to existing queries")
    print("  conditional_evolve_ragas")
    print("    -> add conditional constraints to queries")
    print("  compress_ragas")
    print("    -> compress queries while preserving meaning")
    print()
    print("--- Filtering (remove low-quality pairs) ---")
    print("  dontknow_filter_rule_based")
    print("    -> rule-based filter for unanswerable questions")
    print("  dontknow_filter_openai / dontknow_filter_llama_index")
    print("    -> LLM-based filter for unanswerable questions")
    print("  passage_dependency_filter")
    print("    -> filter questions that don't depend on the passage")
    print()
    print("--- Answer Generation ---")
    print("  make_basic_gen_gt")
    print("    -> generate detailed reference answers")
    print("  make_concise_gen_gt")
    print("    -> generate concise reference answers")
    print()
    print("--- Fluent API Example ---")
    print("  from openai import AsyncOpenAI")
    print("  client = AsyncOpenAI()")
    print()
    print("  corpus.sample(random_single_hop, n=100)")
    print("      .batch_apply(factoid_query_gen, client=client)")
    print("      .batch_apply(make_basic_gen_gt, client=client)")
    print("      .batch_filter(dontknow_filter_openai, client=client)")
    print('      .to_parquet("qa.parquet", "corpus.parquet")')
    print()
    print("This approach scales to large corpora and produces diverse,")
    print("natural-sounding questions. For this tutorial, we created QA")
    print("pairs manually to understand the data format requirements.")


def save_data(
    qa_df: pd.DataFrame,
    corpus_df: pd.DataFrame,
) -> None:
    """Save QA and corpus datasets as parquet files."""
    os.makedirs("data", exist_ok=True)

    qa_path = os.path.join("data", "qa.parquet")
    corpus_path = os.path.join("data", "corpus.parquet")

    qa_df.to_parquet(qa_path, index=False)
    corpus_df.to_parquet(corpus_path, index=False)

    qa_size = os.path.getsize(qa_path)
    corpus_size = os.path.getsize(corpus_path)

    print(f"Saved QA dataset:     {qa_path} ({qa_size:,} bytes)")
    print(f"Saved corpus dataset: {corpus_path} ({corpus_size:,} bytes)")
    print(f"\nFiles can be loaded with:")
    print(f"  qa_df = pd.read_parquet('{qa_path}')")
    print(f"  corpus_df = pd.read_parquet('{corpus_path}')")


def main() -> None:
    """Run all steps of the QA dataset creation lesson."""
    print("=" * 60)
    print("L1-M2.2 — Creating QA Evaluation Datasets")
    print("=" * 60)

    print("\n" + "=" * 60)
    print("Step 1: Create sample corpus")
    print("=" * 60)
    corpus_df = create_sample_corpus()

    print("\n" + "=" * 60)
    print("Step 2: Create QA evaluation dataset")
    print("=" * 60)
    qa_df = create_qa_dataset(corpus_df)

    print("\n" + "=" * 60)
    print("Step 3: Review QA quality")
    print("=" * 60)
    review_qa_quality(qa_df, corpus_df)

    print("\n" + "=" * 60)
    print("Step 4: Split into train/test sets")
    print("=" * 60)
    train_df, test_df = split_train_test(qa_df)

    print("\n" + "=" * 60)
    print("Step 5: AutoRAG Fluent QA Generation API (explanation)")
    print("=" * 60)
    explain_llm_generation()

    print("\n" + "=" * 60)
    print("Step 6: Save datasets to parquet")
    print("=" * 60)
    save_data(qa_df, corpus_df)

    print("\n" + "=" * 60)
    print("Done! Next: L1-M3.1 Configuration YAML")
    print("=" * 60)


if __name__ == "__main__":
    main()
