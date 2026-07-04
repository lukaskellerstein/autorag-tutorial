"""
L1-M3.3 -- Analyzing Results and Deploying the Optimal Pipeline

This lesson analyzes AutoRAG evaluation results, extracts
the optimal configuration, and deploys it using Runner.
"""

import os
import sys

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

    # 10 corpus documents about Python concepts (same data as M3.2)
    documents = [
        {"doc_id": "doc_001", "contents": "Variables in Python are created by assigning a value to a name using the equals sign. Python is dynamically typed, meaning you do not need to declare the type of a variable before using it. Variable names must start with a letter or underscore and can contain letters, digits, and underscores. Python supports multiple assignment, where you can assign values to several variables in one line. Variables can hold any type of data including integers, floats, strings, lists, and objects. The type of a variable can change during program execution.", "metadata": {"last_modified_datetime": "2024-01-15T10:00:00", "prev_id": None, "next_id": None}},
        {"doc_id": "doc_002", "contents": "Functions in Python are defined using the def keyword followed by the function name and parentheses containing optional parameters. Functions allow you to organize code into reusable blocks. They can accept positional arguments, keyword arguments, default values, and variable-length argument lists using *args and **kwargs. Functions return values using the return statement. If no return statement is present, the function returns None. Python also supports lambda functions for creating small anonymous functions in a single expression. Functions are first-class objects and can be passed as arguments to other functions or stored in data structures.", "metadata": {"last_modified_datetime": "2024-01-15T10:00:00", "prev_id": None, "next_id": None}},
        {"doc_id": "doc_003", "contents": "Classes in Python provide a means of bundling data and functionality together. Creating a new class creates a new type of object. Each class instance can have attributes for maintaining state and methods for modifying state. Python supports inheritance, allowing a class to inherit attributes and methods from a parent class. The __init__ method is the constructor called when creating a new instance. Python supports multiple inheritance, where a class can inherit from more than one parent. Special methods like __str__, __repr__, and __eq__ customize object behavior. Classes support encapsulation through naming conventions using underscores.", "metadata": {"last_modified_datetime": "2024-01-15T10:00:00", "prev_id": None, "next_id": None}},
        {"doc_id": "doc_004", "contents": "Decorators in Python are a powerful way to modify the behavior of functions or classes without changing their source code. A decorator is a function that takes another function as input and returns a new function with extended behavior. Decorators are applied using the @ symbol placed above the function definition. Common built-in decorators include @staticmethod, @classmethod, and @property. You can create custom decorators to add logging, timing, authentication, or caching to functions. Decorators can also accept arguments by using a nested function structure. Multiple decorators can be stacked on a single function.", "metadata": {"last_modified_datetime": "2024-01-15T10:00:00", "prev_id": None, "next_id": None}},
        {"doc_id": "doc_005", "contents": "Generators in Python are a special type of iterator that generate values lazily, one at a time, instead of storing them all in memory. They are defined using functions with the yield keyword instead of return. When a generator function is called, it returns a generator object without executing the function body. Each call to next() on the generator executes the function until the next yield statement. Generator expressions provide a concise syntax similar to list comprehensions but with parentheses. Generators are memory-efficient for processing large datasets or infinite sequences. They maintain their state between successive calls automatically.", "metadata": {"last_modified_datetime": "2024-01-15T10:00:00", "prev_id": None, "next_id": None}},
        {"doc_id": "doc_006", "contents": "List comprehensions in Python provide a concise way to create lists based on existing iterables. The syntax consists of an expression followed by a for clause and optional if clauses, all enclosed in square brackets. List comprehensions are generally faster than equivalent for loops because they are optimized internally. They can include nested loops and multiple conditions. Dictionary and set comprehensions follow a similar syntax using curly braces. While powerful, overly complex comprehensions can reduce readability. List comprehensions create the entire list in memory, unlike generator expressions which produce values lazily one at a time.", "metadata": {"last_modified_datetime": "2024-01-15T10:00:00", "prev_id": None, "next_id": None}},
        {"doc_id": "doc_007", "contents": "Error handling in Python uses try-except blocks to catch and handle exceptions gracefully. The try block contains code that might raise an exception, while except blocks handle specific exception types. The else clause runs when no exception occurs, and the finally clause always executes regardless of exceptions. Python has a hierarchy of built-in exceptions including ValueError, TypeError, KeyError, and FileNotFoundError. You can create custom exceptions by subclassing the Exception class. The raise statement throws exceptions explicitly. Context managers using the with statement provide automatic resource cleanup and are preferred for file handling and database connections.", "metadata": {"last_modified_datetime": "2024-01-15T10:00:00", "prev_id": None, "next_id": None}},
        {"doc_id": "doc_008", "contents": "Modules and packages in Python organize code into reusable components. A module is a single Python file containing definitions and statements. Packages are directories containing multiple modules and an __init__.py file. You import modules using the import statement or from-import syntax. Python searches for modules in sys.path, which includes the current directory, installed packages, and standard library paths. The standard library provides hundreds of modules for common tasks like file I/O, networking, and data processing. Third-party packages are installed from PyPI using pip or uv. Namespace packages allow splitting a package across multiple directories without __init__.py files.", "metadata": {"last_modified_datetime": "2024-01-15T10:00:00", "prev_id": None, "next_id": None}},
        {"doc_id": "doc_009", "contents": "Virtual environments in Python create isolated spaces for project dependencies, preventing conflicts between different projects. The venv module, included in the standard library, creates lightweight virtual environments. Each virtual environment has its own Python binary and independent set of installed packages. You activate a virtual environment using a shell-specific activation script. Modern tools like uv and poetry manage virtual environments automatically. Virtual environments solve the problem of different projects requiring different versions of the same library. They are essential for reproducible development and deployment workflows.", "metadata": {"last_modified_datetime": "2024-01-15T10:00:00", "prev_id": None, "next_id": None}},
        {"doc_id": "doc_010", "contents": "Type hints in Python add optional type annotations to function signatures and variable declarations. Introduced in PEP 484 and expanded in subsequent PEPs, type hints improve code readability and enable static analysis tools like mypy to catch type errors before runtime. Common type hints include int, str, float, list, dict, Optional, Union, and generic types from the typing module. Type hints do not affect runtime behavior -- Python remains dynamically typed. They are especially valuable in large codebases and team environments where they serve as documentation. Modern Python versions support built-in generic syntax like list[int] instead of List[int].", "metadata": {"last_modified_datetime": "2024-01-15T10:00:00", "prev_id": None, "next_id": None}},
    ]

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


def run_trial_if_needed() -> None:
    """Run an evaluation trial if results do not already exist."""
    trial_path = "./results/0"
    if os.path.exists(trial_path):
        print(f"Trial results already exist at {trial_path}")
        return

    print("No existing trial found. Running evaluation...")
    print("(This makes the lesson self-contained.)\n")

    # Prepare data first
    if not os.path.exists("data/qa.parquet"):
        create_sample_data()

    from autorag.evaluator import Evaluator

    evaluator = Evaluator(
        qa_data_path="data/qa.parquet",
        corpus_data_path="data/corpus.parquet",
        project_dir="./results",
    )
    evaluator.start_trial("config.yaml")
    print("Evaluation complete.")


def analyze_results(trial_path: str) -> None:
    """Load and analyze the trial summary."""
    summary_path = os.path.join(trial_path, "summary.csv")
    if not os.path.exists(summary_path):
        print(f"summary.csv not found at {summary_path}")
        return

    summary_df = pd.read_csv(summary_path)

    print("Pipeline Optimization Results")
    print("-" * 60)

    for _, row in summary_df.iterrows():
        node = row.get("node_type", "N/A")
        best = row.get("best_module_name", "N/A")
        print(f"\n  Node: {node}")
        print(f"  Best module: {best}")
        print(f"  Metrics:")
        for col in summary_df.columns:
            if col not in ("node_type", "best_module_name", "best_module_params"):
                val = row.get(col)
                if pd.notna(val):
                    try:
                        print(f"    {col}: {float(val):.4f}")
                    except (ValueError, TypeError):
                        print(f"    {col}: {val}")

    print("\n\nFull summary table:")
    print(summary_df.to_string(index=False))


def extract_optimal_config(trial_path: str) -> None:
    """Extract the best configuration from the trial."""
    from autorag.deploy import extract_best_config

    output_path = "best_config.yaml"
    config = extract_best_config(
        trial_path=trial_path,
        output_path=output_path,
    )

    print("Extracted optimal configuration:")
    print(f"  Saved to: {output_path}")
    print()

    if isinstance(config, dict):
        import yaml
        print(yaml.dump(config, default_flow_style=False, sort_keys=False))
    else:
        print(config)


def run_with_runner(trial_path: str) -> None:
    """Use Runner to query the optimal pipeline."""
    from autorag.deploy import Runner

    print("Loading optimal pipeline from trial results...")
    runner = Runner.from_trial_folder(trial_path)

    test_queries = [
        "What are Python decorators?",
        "How do you handle errors in Python?",
        "What is a virtual environment?",
    ]

    print("Running test queries through the optimal pipeline:\n")
    for query in test_queries:
        result = runner.run(query)
        print(f"Q: {query}")
        print(f"A: {result}")
        print()


def explain_pass_module_results() -> None:
    """Explain how to interpret when a pass module wins."""
    print("When a pass_* module wins at a node, it means skipping that")
    print("processing step produces better results on your data.\n")

    examples = [
        ("pass_query_expansion wins", "Your queries are already clear and specific.\n"
         "    Query expansion adds noise rather than helping retrieval."),
        ("pass_reranker wins", "The initial retrieval ordering is already good.\n"
         "    Reranking introduces errors or does not improve ranking quality."),
        ("pass_passage_filter wins", "All retrieved passages are relevant enough.\n"
         "    Filtering removes passages that actually contain useful context."),
        ("pass_compressor wins", "Full passage text produces better generation.\n"
         "    Compression loses important details needed for the answer."),
        ("pass_passage_augmenter wins", "Retrieved chunks are self-contained.\n"
         "    Adding surrounding context introduces irrelevant information."),
    ]

    for scenario, explanation in examples:
        print(f"  If {scenario}:")
        print(f"    {explanation}")
        print()

    print("Pass module wins simplify your production pipeline — fewer")
    print("components means less latency, lower cost, and easier maintenance.")


def explain_api_deployment() -> None:
    """Explain how to deploy the optimal pipeline as a FastAPI server."""
    print("AutoRAG can deploy the optimal pipeline as a FastAPI server.\n")
    print("CLI command:")
    print("  autorag run_api --trial_dir ./results/0 --host 0.0.0.0 --port 8000\n")
    print("Or via Python:")
    print("-" * 60)
    print("""
    from autorag.deploy import ApiRunner

    runner = ApiRunner.from_trial_folder("./results/0")
    runner.run_api_server(host="0.0.0.0", port=8000)
""")
    print("API Endpoints:")
    print(f"  {'Endpoint':<25s} {'Description'}")
    print(f"  {'-' * 25} {'-' * 40}")
    print(f"  {'POST /v1/run':<25s} Full pipeline (retrieve + generate)")
    print(f"  {'POST /v1/retrieve':<25s} Retrieval only (no generation)")
    print(f"  {'POST /v1/stream':<25s} Streaming generation via SSE")
    print(f"  {'GET /version':<25s} API version")
    print()
    print("NGrok tunnel support:")
    print("  AutoRAG supports NGrok tunnels for public access without port")
    print("  forwarding. Pass --ngrok_auth_token to enable a public URL.")
    print()
    print("This lesson does NOT start the server to keep things simple.")
    print("Try it yourself after completing this lesson.")


def explain_gradio_deployment() -> None:
    """Explain the Gradio web interface deployment option."""
    print("AutoRAG provides a Gradio web interface for interactive testing.\n")
    print("CLI command:")
    print("  autorag run_web --trial_dir ./results/0\n")
    print("Or via Python:")
    print("-" * 60)
    print("""
    from autorag.deploy import GradioRunner

    runner = GradioRunner.from_trial_folder("./results/0")
    runner.run_web()
""")
    print("Features:")
    print("  - Interactive chat-like UI for testing queries")
    print("  - Shows retrieved passages alongside generated answers")
    print("  - Shareable links for team collaboration")
    print("  - No code needed — just point at the trial directory")
    print()
    print("The Gradio interface is ideal for quick demos and user testing")
    print("before moving to the production FastAPI deployment.")


def explain_dashboard() -> None:
    """Explain the Streamlit dashboard for visualizing results."""
    print("AutoRAG provides a Streamlit dashboard for visualizing trial results.\n")
    print("CLI command:")
    print("  autorag dashboard --trial_dir ./results/0\n")
    print("The dashboard shows:")
    print("  - Per-node comparison of module performance")
    print("  - Metric distributions across QA pairs")
    print("  - Best pipeline visualization")
    print("  - Detailed score breakdowns per query")
    print()
    print("Note: The dashboard visualizes evaluation results. It is separate")
    print("from the FastAPI server and Gradio interface, which serve queries.")


def interpret_metrics() -> None:
    """Print a reference table explaining each metric."""
    metrics = [
        ("retrieval_f1", "0-1", "higher", "Harmonic mean of precision and recall for retrieved docs"),
        ("retrieval_recall", "0-1", "higher", "Fraction of relevant documents successfully retrieved"),
        ("retrieval_precision", "0-1", "higher", "Fraction of retrieved documents that are relevant"),
        ("retrieval_ndcg", "0-1", "higher", "Rewards relevant documents ranked higher"),
        ("retrieval_mrr", "0-1", "higher", "Position of the first relevant result"),
        ("retrieval_map", "0-1", "higher", "Average precision across all recall levels"),
        ("bleu", "0-1", "higher", "N-gram overlap between generated and reference answer"),
        ("rouge", "0-1", "higher", "Recall-oriented n-gram overlap (ROUGE-L)"),
        ("meteor", "0-1", "higher", "Alignment-based metric with synonyms and stemming"),
        ("sem_score", "0-1", "higher", "Cosine similarity of sentence embeddings"),
        ("bert_score", "0-1", "higher", "Token-level semantic similarity using BERT"),
        ("faithfulness", "0-1", "higher", "Whether the answer is supported by context"),
        ("g_eval", "1-5", "higher", "LLM-as-judge: coherence, consistency, fluency, relevance"),
        ("retrieval_token_f1", "0-1", "higher", "Token-level F1 of compressed passages"),
        ("retrieval_token_recall", "0-1", "higher", "Token-level recall of compressed passages"),
        ("retrieval_token_precision", "0-1", "higher", "Token-level precision of compressed passages"),
        ("context_precision", "0-1", "higher", "RAGAS: context relevance (no ground truth needed)"),
    ]

    print(f"  {'Metric':<28s} {'Range':<8s} {'Better':<10s} Description")
    print(f"  {'-' * 28} {'-' * 8} {'-' * 10} {'-' * 48}")
    for name, rng, direction, desc in metrics:
        print(f"  {name:<28s} {rng:<8s} {direction:<10s} {desc}")

    print()
    print("Interpreting your results:")
    print("  - Retrieval metrics > 0.7 indicate strong retrieval quality")
    print("  - BLEU > 0.3 is generally good for open-ended generation")
    print("  - ROUGE > 0.4 suggests good coverage of reference content")
    print("  - G-Eval scores are 1-5; above 3.5 is good quality")
    print("  - Compressor metrics measure how well compression preserves content")
    print("  - context_precision (RAGAS) is useful when you lack retrieval ground truth")
    print("  - Compare across modules to see which configuration wins")


def main() -> None:
    """Run all steps of the lesson."""
    print_header("L1-M3.3 -- Analyzing Results and Deploying")

    print_header("Step 1: Check Ollama")
    check_ollama()

    print_header("Step 2: Prepare Data and Run Trial")
    if not os.path.exists("data/qa.parquet"):
        create_sample_data()
    run_trial_if_needed()

    trial_path = "./results/0"

    print_header("Step 3: Analyze Results")
    analyze_results(trial_path)

    print_header("Step 4: Extract Optimal Configuration")
    extract_optimal_config(trial_path)

    print_header("Step 5: Run Queries with Runner")
    run_with_runner(trial_path)

    print_header("Step 6: Metric Interpretation Guide")
    interpret_metrics()

    print_header("Step 7: Pass Module Results")
    explain_pass_module_results()

    print_header("Step 8: FastAPI Deployment")
    explain_api_deployment()

    print_header("Step 9: Gradio Web Interface")
    explain_gradio_deployment()

    print_header("Step 10: Streamlit Dashboard")
    explain_dashboard()

    print_header("Done")
    print("You have analyzed AutoRAG results and explored deployment options.")
    print("You now know how to:")
    print("  - Read trial summaries to identify the best configuration")
    print("  - Extract the optimal config as a reusable YAML file")
    print("  - Run queries through the optimal pipeline using Runner")
    print("  - Interpret evaluation metrics across all categories")
    print("  - Understand pass module results and their implications")
    print("  - Deploy via FastAPI (autorag run_api) for production APIs")
    print("  - Deploy via Gradio (autorag run_web) for interactive testing")
    print("  - Visualize results via Streamlit (autorag dashboard)")
    print()
    print("This completes Level 1, Module 3: Running Experiments.")


if __name__ == "__main__":
    main()
