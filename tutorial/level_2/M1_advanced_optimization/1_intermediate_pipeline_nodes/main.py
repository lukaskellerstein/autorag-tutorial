"""
L2-M1.1 — Intermediate Pipeline Nodes

Deep dive into the pipeline nodes between retrieval and generation:
query expansion (pre-retrieval), passage augmenter (post-retrieval),
passage filter (post-reranking), passage compressor (pre-generation),
and prompt maker (pre-generation). We configure AutoRAG to evaluate
pass-through vs active variants for each node and identify which
intermediate processing steps improve end-to-end quality.
"""

import os
from pathlib import Path
from typing import Any

import httpx
import pandas as pd


def print_header(title: str) -> None:
    """Print a section header."""
    print("=" * 60)
    print(title)
    print("=" * 60)


def explain_query_expansion() -> None:
    """Explain pre-retrieval query expansion strategies."""
    print("Query Expansion (Pre-Retrieval)")
    print("-" * 40)
    print()
    print("Query expansion transforms the user's query before retrieval")
    print("to improve recall. AutoRAG supports several strategies:")
    print()
    print("1. pass_query_expansion (No-Op Baseline)")
    print("   Passes the original query through unchanged. Serves as a")
    print("   baseline to measure whether expansion adds value.")
    print()
    print("2. HyDE (Hypothetical Document Embeddings)")
    print("   Uses an LLM to generate a hypothetical answer, then embeds")
    print("   that answer (instead of the query) for vector retrieval.")
    print("   The hypothesis is closer in embedding space to relevant")
    print("   documents than the short query would be. Works well for")
    print("   complex queries where the answer's vocabulary differs from")
    print("   the question's vocabulary.")
    print()
    print("3. query_decompose")
    print("   Breaks a complex multi-part query into simpler sub-queries,")
    print("   retrieves for each, and merges the results. Useful when a")
    print("   single query asks about multiple concepts.")
    print()
    print("4. multi_query_expansion")
    print("   Generates multiple paraphrased variants of the query and")
    print("   retrieves for each variant. The union of results has higher")
    print("   recall because different phrasings match different documents.")
    print()
    print("When to use pass vs active:")
    print("   Simple factoid queries ('What is X?') rarely benefit from")
    print("   expansion. Complex, multi-hop, or ambiguous queries see the")
    print("   biggest gains. HyDE adds LLM latency, so measure whether")
    print("   the retrieval improvement justifies the cost.")


def explain_passage_augmenter() -> None:
    """Explain post-retrieval passage augmentation."""
    print("Passage Augmenter (Post-Retrieval)")
    print("-" * 40)
    print()
    print("After retrieval returns a set of passages, the augmenter can")
    print("fetch additional context around each passage.")
    print()
    print("1. pass_passage_augmenter (No-Op Baseline)")
    print("   Returns retrieved passages unchanged.")
    print()
    print("2. prev_next_augmenter")
    print("   For each retrieved chunk, fetches the previous and/or next")
    print("   chunks from the original document. This restores context")
    print("   that was lost during chunking.")
    print()
    print("   Configuration:")
    print("     mode: 'prev', 'next', or 'both'")
    print("     num_passages: how many adjacent chunks to fetch (1-3)")
    print()
    print("   The corpus metadata must include 'prev_id' and 'next_id'")
    print("   fields linking chunks to their neighbors.")
    print()
    print("When context helps vs adds noise:")
    print("   Adjacent chunks help when answers span chunk boundaries")
    print("   or when surrounding context disambiguates the passage.")
    print("   They add noise when the adjacent content is on a different")
    print("   topic (e.g., at section boundaries). Let AutoRAG decide")
    print("   by comparing pass_passage_augmenter against prev_next.")


def explain_passage_filter() -> None:
    """Explain post-reranking passage filtering."""
    print("Passage Filter (Post-Reranking)")
    print("-" * 40)
    print()
    print("Filters remove low-quality passages from the candidate set.")
    print("Unlike rerankers (which return a fixed top_k), filters remove")
    print("a variable number of passages based on quality thresholds.")
    print()
    print("1. pass_passage_filter (No-Op Baseline)")
    print("   Returns all passages unchanged.")
    print()
    print("2. similarity_threshold_cutoff")
    print("   Removes passages whose similarity score falls below an")
    print("   absolute threshold. For example, threshold=0.5 keeps only")
    print("   passages with score >= 0.5. Good when you have a clear")
    print("   notion of what constitutes a relevant score.")
    print()
    print("3. similarity_percentile_cutoff")
    print("   Removes passages below a relative percentile of the score")
    print("   distribution. For example, percentile=0.6 keeps only the")
    print("   top 40% of passages. Adapts to different queries where")
    print("   absolute score ranges vary.")
    print()
    print("4. recency_filter")
    print("   Prefers more recently modified documents. Uses the")
    print("   'last_modified_datetime' field in corpus metadata to")
    print("   filter or boost recent content.")
    print()
    print("Key distinction: filter vs reranker")
    print("   A reranker always returns exactly top_k results.")
    print("   A filter may return fewer results (or even zero) if no")
    print("   passages meet the threshold. This is useful when it is")
    print("   better to return nothing than to return irrelevant content.")


def explain_passage_compressor() -> None:
    """Explain pre-generation passage compression."""
    print("Passage Compressor (Pre-Generation)")
    print("-" * 40)
    print()
    print("Compressors reduce the volume of retrieved text before it is")
    print("sent to the generator. This saves tokens, reduces latency,")
    print("and can improve generation quality by removing noise.")
    print()
    print("1. pass_compressor (No-Op Baseline)")
    print("   Sends all retrieved passages to the generator unchanged.")
    print()
    print("2. tree_summarize")
    print("   Hierarchical summarization: recursively summarizes groups")
    print("   of passages into shorter summaries until a single summary")
    print("   remains. Preserves key information across many passages.")
    print("   Requires an LLM call, adding latency but reducing the")
    print("   generator's input size.")
    print()
    print("3. refine")
    print("   Iterative refinement: processes passages one at a time,")
    print("   each time refining the running summary with new information.")
    print("   Produces more coherent summaries than tree_summarize but")
    print("   is sequential (slower for many passages).")
    print()
    print("4. LongLLMLingua")
    print("   Token-level compression using perplexity-based selection.")
    print("   Keeps the most informative tokens and drops filler words.")
    print("   Can achieve 2-5x compression with minimal information loss.")
    print("   Does not require an LLM — uses a small language model for")
    print("   perplexity scoring.")
    print()
    print("Token savings and latency:")
    print("   Compression trades pre-generation compute for reduced")
    print("   generator input. With expensive LLMs, the token savings")
    print("   often outweigh the compression cost. Measure end-to-end")
    print("   latency and quality, not just compression ratio.")


def explain_prompt_maker() -> None:
    """Explain pre-generation prompt construction."""
    print("Prompt Maker (Pre-Generation)")
    print("-" * 40)
    print()
    print("The prompt maker assembles the final prompt from the query")
    print("and retrieved passages before sending it to the generator.")
    print()
    print("1. fstring")
    print("   Template-based prompt using Python f-string syntax.")
    print("   Placeholders: {query}, {retrieved_contents}")
    print("   Example: 'Answer based on: {retrieved_contents}\\n")
    print("   Question: {query}\\nAnswer:'")
    print()
    print("2. chat_fstring")
    print("   Same as fstring but formats the prompt as a chat message")
    print("   with system/user roles. Better for chat-tuned LLMs.")
    print()
    print("3. long_context_reorder")
    print("   Reorders passages to address the 'lost in the middle'")
    print("   problem: LLMs attend more to the beginning and end of")
    print("   their context window. This strategy places the most")
    print("   relevant passages at the start and end, with less")
    print("   relevant ones in the middle.")
    print()
    print("4. window_replacement")
    print("   Used after LongLLMLingua compression: replaces the")
    print("   compressed token sequences with the original full text")
    print("   from a window around each kept token. Restores")
    print("   readability while maintaining the compression benefit.")


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


def run_evaluation() -> None:
    """Run the AutoRAG evaluation with the intermediate nodes config."""
    from autorag.evaluator import Evaluator

    evaluator = Evaluator(
        qa_data_path="data/qa.parquet",
        corpus_data_path="data/corpus.parquet",
        project_dir="./results",
    )
    evaluator.start_trial("config.yaml")
    print("Evaluation complete. Results saved to ./results/")


def compare_results() -> None:
    """Load and display results, highlighting pass vs active node choices."""
    summary_path = Path("results/0/summary.csv")

    if not summary_path.exists():
        print(f"Results file not found: {summary_path}")
        print("Run the full evaluation first (this requires Ollama with gemma4:e2b).")
        print("\nWhen results are available, this step shows which intermediate")
        print("nodes used pass-through vs active processing:")
        print("  query_expansion:    pass_query_expansion vs hyde")
        print("  passage_augmenter:  pass_passage_augmenter vs prev_next_augmenter")
        print("  passage_reranker:   pass_reranker vs flashrank_reranker")
        print("  passage_filter:     pass_passage_filter vs similarity_threshold_cutoff")
        print("  passage_compressor: pass_compressor vs tree_summarize")
        print("  prompt_maker:       fstring vs long_context_reorder")
        return

    summary_df = pd.read_csv(summary_path)

    print(f"Results loaded from {summary_path}")
    print(f"Columns: {list(summary_df.columns)}")
    print()

    print("Full Summary Table:")
    print("-" * 40)
    print(summary_df.to_string(index=False))
    print()

    pass_modules = {
        "pass_query_expansion", "pass_passage_augmenter", "pass_reranker",
        "pass_passage_filter", "pass_compressor",
    }

    if "best_module_name" in summary_df.columns:
        print("Winning Configurations by Node:")
        print("-" * 40)
        for _, row in summary_df.iterrows():
            node = row.get("node_type", "unknown")
            module = row.get("best_module_name", "unknown")
            marker = " (pass-through)" if module in pass_modules else " (active)"
            print(f"  {node}: {module}{marker}")
    elif "module_name" in summary_df.columns:
        print("Module Results:")
        print("-" * 40)
        for _, row in summary_df.iterrows():
            module = row.get("module_name", "unknown")
            marker = " (pass-through)" if module in pass_modules else ""
            print(f"  Module: {module}{marker}")
            for col in summary_df.columns:
                if col not in ("module_name", "node_type", "module_type"):
                    print(f"    {col}: {row[col]}")

    print()
    print("Interpretation Guide:")
    print("  - If a pass-through module wins, that processing step adds no value")
    print("    for this corpus and can be removed to reduce latency.")
    print("  - If an active module wins, the intermediate processing improves")
    print("    end-to-end quality enough to justify the added compute cost.")
    print("  - The optimal pipeline is the combination of active nodes that")
    print("    maximizes quality while minimizing unnecessary processing.")


def main() -> None:
    """Run all steps of the intermediate pipeline nodes lesson."""
    print_header("L2-M1.1 — Intermediate Pipeline Nodes")

    print("\n")
    print_header("Step 1: Create sample evaluation data")
    create_sample_data()

    print("\n")
    print_header("Step 2: Check Ollama availability")
    ollama_ok = check_ollama()

    print("\n")
    print_header("Step 3: Query expansion explained")
    explain_query_expansion()

    print("\n")
    print_header("Step 4: Passage augmenter explained")
    explain_passage_augmenter()

    print("\n")
    print_header("Step 5: Passage filter explained")
    explain_passage_filter()

    print("\n")
    print_header("Step 6: Passage compressor explained")
    explain_passage_compressor()

    print("\n")
    print_header("Step 7: Prompt maker explained")
    explain_prompt_maker()

    print("\n")
    print_header("Step 8: Run AutoRAG evaluation")
    if ollama_ok:
        print("Starting evaluation (this may take several minutes)...")
        run_evaluation()
    else:
        print("Skipping evaluation — Ollama is not running.")
        print("To run the full evaluation:")
        print("  1. Start Ollama:  ollama serve")
        print("  2. Pull model:    ollama pull gemma4:e2b")
        print("  3. Re-run:        uv run python main.py")

    print("\n")
    print_header("Step 9: Compare results")
    compare_results()

    print("\n")
    print_header("Done!")


if __name__ == "__main__":
    main()
