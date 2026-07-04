"""
L1-M3.2 — Running and Monitoring AutoRAG Evaluations

This lesson runs an AutoRAG evaluation trial, monitors
its progress, and inspects the results directory structure.
"""

import os
import sys
import time

import httpx
import pandas as pd


def print_header(title: str) -> None:
    """Print a formatted section header."""
    print()
    print("=" * 60)
    print(f"  {title}")
    print("=" * 60)
    print()


def create_sample_data() -> None:
    """Generate 10 corpus documents and 20 QA pairs about Python concepts."""
    os.makedirs("data", exist_ok=True)

    # 10 corpus documents about Python concepts
    documents = [
        {
            "doc_id": "doc_001",
            "contents": (
                "Variables in Python are created by assigning a value to a name "
                "using the equals sign. Python is dynamically typed, meaning you "
                "do not need to declare the type of a variable before using it. "
                "Variable names must start with a letter or underscore and can "
                "contain letters, digits, and underscores. Python supports "
                "multiple assignment, where you can assign values to several "
                "variables in one line. Variables can hold any type of data "
                "including integers, floats, strings, lists, and objects. "
                "The type of a variable can change during program execution."
            ),
            "metadata": {"last_modified_datetime": "2024-01-15T10:00:00", "prev_id": None, "next_id": None},
        },
        {
            "doc_id": "doc_002",
            "contents": (
                "Functions in Python are defined using the def keyword followed "
                "by the function name and parentheses containing optional parameters. "
                "Functions allow you to organize code into reusable blocks. They can "
                "accept positional arguments, keyword arguments, default values, and "
                "variable-length argument lists using *args and **kwargs. Functions "
                "return values using the return statement. If no return statement is "
                "present, the function returns None. Python also supports lambda "
                "functions for creating small anonymous functions in a single expression. "
                "Functions are first-class objects and can be passed as arguments to "
                "other functions or stored in data structures."
            ),
            "metadata": {"last_modified_datetime": "2024-01-15T10:00:00", "prev_id": None, "next_id": None},
        },
        {
            "doc_id": "doc_003",
            "contents": (
                "Classes in Python provide a means of bundling data and functionality "
                "together. Creating a new class creates a new type of object. Each "
                "class instance can have attributes for maintaining state and methods "
                "for modifying state. Python supports inheritance, allowing a class to "
                "inherit attributes and methods from a parent class. The __init__ method "
                "is the constructor called when creating a new instance. Python supports "
                "multiple inheritance, where a class can inherit from more than one parent. "
                "Special methods like __str__, __repr__, and __eq__ customize object behavior. "
                "Classes support encapsulation through naming conventions using underscores."
            ),
            "metadata": {"last_modified_datetime": "2024-01-15T10:00:00", "prev_id": None, "next_id": None},
        },
        {
            "doc_id": "doc_004",
            "contents": (
                "Decorators in Python are a powerful way to modify the behavior of "
                "functions or classes without changing their source code. A decorator "
                "is a function that takes another function as input and returns a new "
                "function with extended behavior. Decorators are applied using the @ "
                "symbol placed above the function definition. Common built-in decorators "
                "include @staticmethod, @classmethod, and @property. You can create "
                "custom decorators to add logging, timing, authentication, or caching "
                "to functions. Decorators can also accept arguments by using a nested "
                "function structure. Multiple decorators can be stacked on a single function."
            ),
            "metadata": {"last_modified_datetime": "2024-01-15T10:00:00", "prev_id": None, "next_id": None},
        },
        {
            "doc_id": "doc_005",
            "contents": (
                "Generators in Python are a special type of iterator that generate "
                "values lazily, one at a time, instead of storing them all in memory. "
                "They are defined using functions with the yield keyword instead of "
                "return. When a generator function is called, it returns a generator "
                "object without executing the function body. Each call to next() on "
                "the generator executes the function until the next yield statement. "
                "Generator expressions provide a concise syntax similar to list "
                "comprehensions but with parentheses. Generators are memory-efficient "
                "for processing large datasets or infinite sequences. They maintain "
                "their state between successive calls automatically."
            ),
            "metadata": {"last_modified_datetime": "2024-01-15T10:00:00", "prev_id": None, "next_id": None},
        },
        {
            "doc_id": "doc_006",
            "contents": (
                "List comprehensions in Python provide a concise way to create lists "
                "based on existing iterables. The syntax consists of an expression "
                "followed by a for clause and optional if clauses, all enclosed in "
                "square brackets. List comprehensions are generally faster than "
                "equivalent for loops because they are optimized internally. They can "
                "include nested loops and multiple conditions. Dictionary and set "
                "comprehensions follow a similar syntax using curly braces. While "
                "powerful, overly complex comprehensions can reduce readability. "
                "List comprehensions create the entire list in memory, unlike "
                "generator expressions which produce values lazily one at a time."
            ),
            "metadata": {"last_modified_datetime": "2024-01-15T10:00:00", "prev_id": None, "next_id": None},
        },
        {
            "doc_id": "doc_007",
            "contents": (
                "Error handling in Python uses try-except blocks to catch and handle "
                "exceptions gracefully. The try block contains code that might raise "
                "an exception, while except blocks handle specific exception types. "
                "The else clause runs when no exception occurs, and the finally clause "
                "always executes regardless of exceptions. Python has a hierarchy of "
                "built-in exceptions including ValueError, TypeError, KeyError, and "
                "FileNotFoundError. You can create custom exceptions by subclassing "
                "the Exception class. The raise statement throws exceptions explicitly. "
                "Context managers using the with statement provide automatic resource "
                "cleanup and are preferred for file handling and database connections."
            ),
            "metadata": {"last_modified_datetime": "2024-01-15T10:00:00", "prev_id": None, "next_id": None},
        },
        {
            "doc_id": "doc_008",
            "contents": (
                "Modules and packages in Python organize code into reusable components. "
                "A module is a single Python file containing definitions and statements. "
                "Packages are directories containing multiple modules and an __init__.py "
                "file. You import modules using the import statement or from-import syntax. "
                "Python searches for modules in sys.path, which includes the current "
                "directory, installed packages, and standard library paths. The standard "
                "library provides hundreds of modules for common tasks like file I/O, "
                "networking, and data processing. Third-party packages are installed from "
                "PyPI using pip or uv. Namespace packages allow splitting a package "
                "across multiple directories without __init__.py files."
            ),
            "metadata": {"last_modified_datetime": "2024-01-15T10:00:00", "prev_id": None, "next_id": None},
        },
        {
            "doc_id": "doc_009",
            "contents": (
                "Virtual environments in Python create isolated spaces for project "
                "dependencies, preventing conflicts between different projects. The "
                "venv module, included in the standard library, creates lightweight "
                "virtual environments. Each virtual environment has its own Python "
                "binary and independent set of installed packages. You activate a "
                "virtual environment using a shell-specific activation script. Modern "
                "tools like uv and poetry manage virtual environments automatically. "
                "Virtual environments solve the problem of different projects requiring "
                "different versions of the same library. They are essential for "
                "reproducible development and deployment workflows."
            ),
            "metadata": {"last_modified_datetime": "2024-01-15T10:00:00", "prev_id": None, "next_id": None},
        },
        {
            "doc_id": "doc_010",
            "contents": (
                "Type hints in Python add optional type annotations to function "
                "signatures and variable declarations. Introduced in PEP 484 and "
                "expanded in subsequent PEPs, type hints improve code readability "
                "and enable static analysis tools like mypy to catch type errors "
                "before runtime. Common type hints include int, str, float, list, "
                "dict, Optional, Union, and generic types from the typing module. "
                "Type hints do not affect runtime behavior — Python remains dynamically "
                "typed. They are especially valuable in large codebases and team "
                "environments where they serve as documentation. Modern Python versions "
                "support built-in generic syntax like list[int] instead of List[int]."
            ),
            "metadata": {"last_modified_datetime": "2024-01-15T10:00:00", "prev_id": None, "next_id": None},
        },
    ]

    # 20 QA pairs (2 per document)
    qa_pairs = [
        {"qid": "q_001", "query": "How are variables created in Python?", "retrieval_gt": [["doc_001"]], "generation_gt": ["Variables in Python are created by assigning a value to a name using the equals sign. Python is dynamically typed, so you do not need to declare the type beforehand."]},
        {"qid": "q_002", "query": "What types of data can Python variables hold?", "retrieval_gt": [["doc_001"]], "generation_gt": ["Python variables can hold any type of data including integers, floats, strings, lists, and objects. The type can change during program execution."]},
        {"qid": "q_003", "query": "How do you define a function in Python?", "retrieval_gt": [["doc_002"]], "generation_gt": ["Functions in Python are defined using the def keyword followed by the function name and parentheses containing optional parameters."]},
        {"qid": "q_004", "query": "What are *args and **kwargs in Python functions?", "retrieval_gt": [["doc_002"]], "generation_gt": ["*args allows a function to accept variable-length positional arguments, and **kwargs allows variable-length keyword arguments. They provide flexibility in function signatures."]},
        {"qid": "q_005", "query": "What is the purpose of classes in Python?", "retrieval_gt": [["doc_003"]], "generation_gt": ["Classes in Python provide a means of bundling data and functionality together. They create new types of objects with attributes for state and methods for behavior."]},
        {"qid": "q_006", "query": "How does inheritance work in Python classes?", "retrieval_gt": [["doc_003"]], "generation_gt": ["Python supports inheritance, allowing a class to inherit attributes and methods from a parent class. It also supports multiple inheritance where a class can inherit from more than one parent."]},
        {"qid": "q_007", "query": "What are decorators in Python?", "retrieval_gt": [["doc_004"]], "generation_gt": ["Decorators are a way to modify the behavior of functions or classes without changing their source code. A decorator takes a function as input and returns a new function with extended behavior."]},
        {"qid": "q_008", "query": "How do you apply a decorator to a function?", "retrieval_gt": [["doc_004"]], "generation_gt": ["Decorators are applied using the @ symbol placed above the function definition. Multiple decorators can be stacked on a single function."]},
        {"qid": "q_009", "query": "What are generators in Python?", "retrieval_gt": [["doc_005"]], "generation_gt": ["Generators are a special type of iterator that generate values lazily, one at a time, instead of storing them all in memory. They are defined using functions with the yield keyword."]},
        {"qid": "q_010", "query": "Why are generators memory-efficient?", "retrieval_gt": [["doc_005"]], "generation_gt": ["Generators are memory-efficient because they produce values one at a time instead of storing the entire sequence in memory. This makes them ideal for processing large datasets or infinite sequences."]},
        {"qid": "q_011", "query": "What is a list comprehension in Python?", "retrieval_gt": [["doc_006"]], "generation_gt": ["A list comprehension provides a concise way to create lists based on existing iterables. The syntax consists of an expression followed by a for clause and optional if clauses, enclosed in square brackets."]},
        {"qid": "q_012", "query": "How do list comprehensions compare to for loops?", "retrieval_gt": [["doc_006"]], "generation_gt": ["List comprehensions are generally faster than equivalent for loops because they are optimized internally by Python. However, overly complex comprehensions can reduce readability."]},
        {"qid": "q_013", "query": "How does error handling work in Python?", "retrieval_gt": [["doc_007"]], "generation_gt": ["Error handling in Python uses try-except blocks. The try block contains code that might raise an exception, while except blocks handle specific exception types. The finally clause always executes."]},
        {"qid": "q_014", "query": "How do you create custom exceptions in Python?", "retrieval_gt": [["doc_007"]], "generation_gt": ["You can create custom exceptions by subclassing the Exception class. The raise statement is used to throw exceptions explicitly in your code."]},
        {"qid": "q_015", "query": "What are modules and packages in Python?", "retrieval_gt": [["doc_008"]], "generation_gt": ["A module is a single Python file containing definitions and statements. Packages are directories containing multiple modules and an __init__.py file. They organize code into reusable components."]},
        {"qid": "q_016", "query": "How does Python find modules when you import them?", "retrieval_gt": [["doc_008"]], "generation_gt": ["Python searches for modules in sys.path, which includes the current directory, installed packages, and standard library paths."]},
        {"qid": "q_017", "query": "What is a virtual environment in Python?", "retrieval_gt": [["doc_009"]], "generation_gt": ["A virtual environment creates an isolated space for project dependencies, preventing conflicts between different projects. Each has its own Python binary and independent set of installed packages."]},
        {"qid": "q_018", "query": "Why are virtual environments important?", "retrieval_gt": [["doc_009"]], "generation_gt": ["Virtual environments solve the problem of different projects requiring different versions of the same library. They are essential for reproducible development and deployment workflows."]},
        {"qid": "q_019", "query": "What are type hints in Python?", "retrieval_gt": [["doc_010"]], "generation_gt": ["Type hints add optional type annotations to function signatures and variable declarations. They improve code readability and enable static analysis tools like mypy to catch type errors before runtime."]},
        {"qid": "q_020", "query": "Do type hints affect Python runtime behavior?", "retrieval_gt": [["doc_010"]], "generation_gt": ["No, type hints do not affect runtime behavior. Python remains dynamically typed. Type hints serve as documentation and enable static analysis tools but are not enforced at runtime."]},
    ]

    corpus_df = pd.DataFrame(documents)
    qa_df = pd.DataFrame(qa_pairs)

    corpus_df.to_parquet("data/corpus.parquet", index=False)
    qa_df.to_parquet("data/qa.parquet", index=False)

    print(f"Created data/corpus.parquet: {len(corpus_df)} documents")
    print(f"Created data/qa.parquet:     {len(qa_df)} QA pairs")


def check_ollama() -> None:
    """Check that Ollama is running and accessible."""
    print("Checking Ollama at http://localhost:11434 ...")
    try:
        response = httpx.get("http://localhost:11434/api/tags", timeout=5.0)
        response.raise_for_status()
        models = response.json().get("models", [])
        model_names = [m["name"] for m in models]
        print(f"Ollama is running. Available models: {model_names}")
        if not any("gemma4" in name for name in model_names):
            print("WARNING: gemma4:e2b not found. Pull it with:")
            print("  ollama pull gemma4:e2b")
    except (httpx.ConnectError, httpx.TimeoutException):
        print("ERROR: Cannot connect to Ollama at http://localhost:11434")
        print("Start Ollama and pull the model:")
        print("  ollama serve")
        print("  ollama pull gemma4:e2b")
        sys.exit(1)


def prepare_data() -> None:
    """Generate sample data if it does not already exist."""
    if os.path.exists("data/qa.parquet") and os.path.exists("data/corpus.parquet"):
        print("Sample data already exists in data/")
    else:
        create_sample_data()

    qa_df = pd.read_parquet("data/qa.parquet")
    corpus_df = pd.read_parquet("data/corpus.parquet")

    print(f"\nQA dataset shape:     {qa_df.shape}")
    print(f"QA columns:           {list(qa_df.columns)}")
    print(f"Corpus dataset shape: {corpus_df.shape}")
    print(f"Corpus columns:       {list(corpus_df.columns)}")


def run_evaluation() -> None:
    """Run the AutoRAG evaluation using the Evaluator API."""
    from autorag.evaluator import Evaluator

    print("Starting evaluation...")
    print("This will run BM25 retrieval + prompt making + generation.")
    print("Progress is logged to the console by AutoRAG.\n")

    start_time = time.time()

    evaluator = Evaluator(
        qa_data_path="data/qa.parquet",
        corpus_data_path="data/corpus.parquet",
        project_dir="./results",
    )
    evaluator.start_trial("config.yaml")

    elapsed = time.time() - start_time
    print(f"\nEvaluation complete! Took {elapsed:.1f} seconds.")


def inspect_results() -> None:
    """Walk the results/ directory tree and print its structure."""
    print("Results directory structure:\n")
    for root, dirs, files in os.walk("./results"):
        level = root.replace("./results", "").count(os.sep)
        indent = "  " * level
        basename = os.path.basename(root) or "results/"
        print(f"{indent}{basename}/")
        sub_indent = "  " * (level + 1)
        for f in sorted(files):
            size = os.path.getsize(os.path.join(root, f))
            if size > 1024:
                size_str = f"{size / 1024:.1f} KB"
            else:
                size_str = f"{size} B"
            print(f"{sub_indent}{f}  ({size_str})")


def print_summary() -> None:
    """Load and print the trial summary CSV."""
    # Find the trial directory (usually 0/)
    trial_dirs = [
        d for d in os.listdir("./results")
        if os.path.isdir(os.path.join("./results", d)) and d.isdigit()
    ]
    if not trial_dirs:
        print("No trial directories found in results/")
        return

    trial_dir = sorted(trial_dirs)[0]
    summary_path = os.path.join("./results", trial_dir, "summary.csv")

    if not os.path.exists(summary_path):
        print(f"summary.csv not found at {summary_path}")
        return

    print(f"Loading summary from: {summary_path}\n")
    summary_df = pd.read_csv(summary_path)

    print("Trial Summary:")
    print("-" * 60)
    for _, row in summary_df.iterrows():
        print(f"  Node: {row.get('node_type', 'N/A')}")
        print(f"  Best module: {row.get('best_module_name', 'N/A')}")
        for col in summary_df.columns:
            if col not in ("node_type", "best_module_name", "best_module_params"):
                val = row.get(col)
                if pd.notna(val):
                    print(f"    {col}: {val}")
        print()

    print("Full summary table:")
    print(summary_df.to_string(index=False))


def main() -> None:
    """Run all steps of the lesson."""
    print_header("L1-M3.2 -- Running and Monitoring Evaluations")

    print_header("Step 1: Check Ollama")
    check_ollama()

    print_header("Step 2: Prepare Evaluation Data")
    prepare_data()

    print_header("Step 3: Run AutoRAG Evaluation")
    run_evaluation()

    print_header("Step 4: Inspect Results Directory")
    inspect_results()

    print_header("Step 5: Trial Summary")
    print_summary()

    print_header("Done")
    print("You have successfully run an AutoRAG evaluation trial.")
    print("Next lesson: L1-M3.3 -- Analyzing Results and Deploying.")


if __name__ == "__main__":
    main()
