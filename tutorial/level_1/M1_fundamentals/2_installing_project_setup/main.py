"""
L1-M1.2 — Installing and Project Setup

Demonstrates AutoRAG project structure, data format requirements,
validation utilities, CLI commands, and configuration YAML structure.
"""

from datetime import datetime

import pandas as pd


def print_section(title: str) -> None:
    """Print a formatted section header."""
    print(f"\n{'=' * 60}\n  {title}\n{'=' * 60}\n")


def explain_project_structure() -> None:
    """Print the standard AutoRAG project directory layout."""
    print_section("Step 1: AutoRAG Project Structure")
    print("""\
    my_project/
    ├── config.yaml       # Pipeline config — modules & parameters to evaluate
    ├── qa.parquet         # QA pairs — queries + ground-truth answers & doc IDs
    ├── corpus.parquet     # Document corpus — text + metadata to retrieve from
    └── results/           # Auto-generated — trial results, metrics, best config

  config.yaml  — defines WHAT to evaluate (chunking, retrieval, generation).
  qa.parquet   — defines HOW to measure quality (ground-truth evaluation data).
  corpus.parquet — provides the documents AutoRAG chunks, embeds, and retrieves.
  results/     — created by `autorag evaluate`; holds scores and best pipeline.""")


def create_qa_dataframe() -> pd.DataFrame:
    """Create a sample QA evaluation DataFrame with 3 rows."""
    print_section("Step 2: QA Dataset Format")
    print("Required columns: qid (str), query (str),")
    print("  retrieval_gt (List[List[str]]), generation_gt (List[str])\n")

    qa_df = pd.DataFrame({
        "qid": ["q_001", "q_002", "q_003"],
        "query": [
            "What is a decorator in Python?",
            "How do list comprehensions work in Python?",
            "What is the GIL in Python?",
        ],
        "retrieval_gt": [[["doc_001"]], [["doc_002"]], [["doc_003"]]],
        "generation_gt": [
            ["A decorator is a function that takes another function as an "
             "argument and extends its behavior without modifying it."],
            ["List comprehensions provide a concise way to create lists by "
             "applying an expression to each item in an iterable."],
            ["The GIL (Global Interpreter Lock) is a mutex in CPython that "
             "allows only one thread to execute Python bytecode at a time."],
        ],
    })

    print("QA DataFrame:")
    print("-" * 60)
    print(qa_df[["qid", "query"]].to_string(index=False))
    print(f"\nColumn dtypes:\n{qa_df.dtypes.to_string()}")
    print(f"\nSample retrieval_gt (q_001): {qa_df.iloc[0]['retrieval_gt']}")
    print(f"Sample generation_gt (q_001): {qa_df.iloc[0]['generation_gt']}")
    return qa_df


def create_corpus_dataframe() -> pd.DataFrame:
    """Create a sample corpus DataFrame with 3 rows."""
    print_section("Step 3: Corpus Dataset Format")
    print("Required columns: doc_id (str), contents (str),")
    print("  metadata (dict with last_modified_datetime, prev_id, next_id)\n")

    now = datetime.now()
    corpus_df = pd.DataFrame({
        "doc_id": ["doc_001", "doc_002", "doc_003"],
        "contents": [
            "Decorators in Python are a powerful tool for modifying the "
            "behavior of functions or classes. A decorator takes another "
            "function as its argument and returns a new function that extends "
            "the original behavior. Applied using the @decorator_name syntax.",
            "List comprehensions provide a concise way to create new lists. "
            "The basic syntax is [expression for item in iterable if condition]. "
            "They replace multi-line for-loops with a single readable line. "
            "Similar syntax exists for dicts, sets, and generator expressions.",
            "The Global Interpreter Lock (GIL) is a mutex that protects access "
            "to Python objects, preventing multiple threads from executing "
            "Python bytecodes at once in CPython. For CPU-bound parallelism, "
            "use multiprocessing or alternative interpreters like PyPy.",
        ],
        "metadata": [
            {"last_modified_datetime": now, "prev_id": None, "next_id": "doc_002"},
            {"last_modified_datetime": now, "prev_id": "doc_001", "next_id": "doc_003"},
            {"last_modified_datetime": now, "prev_id": "doc_002", "next_id": None},
        ],
    })

    print("Corpus DataFrame:")
    print("-" * 60)
    for _, row in corpus_df.iterrows():
        print(f"  doc_id:   {row['doc_id']}")
        print(f"  contents: {row['contents'][:65]}...")
        print(f"  metadata: {row['metadata']}\n")
    print(f"Column dtypes:\n{corpus_df.dtypes.to_string()}")
    return corpus_df


def validate_datasets(qa_df: pd.DataFrame, corpus_df: pd.DataFrame) -> None:
    """Validate QA and corpus DataFrames using AutoRAG utilities."""
    print_section("Step 4: Data Validation")
    print("AutoRAG provides validation utilities in autorag.utils.preprocess:\n"
          "  - cast_qa_dataset()               — validate & normalize QA data\n"
          "  - cast_corpus_dataset()           — validate & normalize corpus\n"
          "  - validate_qa_from_corpus_dataset()— cross-check QA vs corpus IDs\n")

    try:
        from autorag.utils.preprocess import (
            cast_corpus_dataset,
            cast_qa_dataset,
            validate_qa_from_corpus_dataset,
        )

        print("[1/3] Validating QA dataset...")
        qa_validated = cast_qa_dataset(qa_df)
        print(f"      OK — {len(qa_validated)} rows, columns: {list(qa_validated.columns)}\n")

        print("[2/3] Validating corpus dataset...")
        corpus_validated = cast_corpus_dataset(corpus_df)
        print(f"      OK — {len(corpus_validated)} rows, columns: {list(corpus_validated.columns)}\n")

        print("[3/3] Cross-validating retrieval_gt doc_ids against corpus...")
        validate_qa_from_corpus_dataset(qa_validated, corpus_validated)
        print("      OK — All retrieval_gt doc_ids exist in the corpus.\n")
        print("All validations passed successfully.")

    except ImportError:
        print("ERROR: autorag not installed. Run: uv add autorag")
    except AssertionError as e:
        print(f"VALIDATION FAILED: {e}")
    except Exception as e:
        print(f"UNEXPECTED ERROR: {type(e).__name__}: {e}")


def show_cli_commands() -> None:
    """Print an overview of AutoRAG CLI commands."""
    print_section("Step 5: AutoRAG CLI Commands")
    commands = [
        ("autorag evaluate",
         "Run a full evaluation experiment.\n"
         "    autorag evaluate --config config.yaml "
         "--qa_data_path qa.parquet --corpus_data_path corpus.parquet"),
        ("autorag dashboard",
         "Launch the Streamlit dashboard to visualize results.\n"
         "    autorag dashboard --trial_dir results/0"),
        ("autorag run_api",
         "Deploy the best pipeline as a FastAPI server.\n"
         "    autorag run_api --trial_dir results/0 --host 0.0.0.0 --port 8000"),
        ("autorag validate",
         "Validate a config YAML without running evaluation.\n"
         "    autorag validate --config config.yaml"),
        ("autorag extract_best_config",
         "Extract the optimal configuration from completed results.\n"
         "    autorag extract_best_config --trial_dir results/0"),
    ]
    for cmd, desc in commands:
        print(f"  {cmd}\n    {desc}\n")


def show_sample_config() -> None:
    """Print a sample AutoRAG configuration YAML."""
    print_section("Step 6: Sample Configuration YAML")
    print("The config.yaml defines which nodes, modules, and parameter")
    print("combinations AutoRAG will evaluate:\n")
    print("""\
    node_lines:
      - node_line_name: retrieve_and_generate
        nodes:
          - node_type: retrieval
            strategy:
              metrics: [retrieval_f1, retrieval_recall]
            modules:
              - module_type: bm25
                top_k: [3, 5, 10]
              - module_type: vectordb
                embedding_model:
                  - huggingface/BAAI/bge-small-en-v1.5
                top_k: [3, 5, 10]
              - module_type: hybrid_rrf
                weight_range: (4, 80)
                top_k: [3, 5, 10]
          - node_type: passage_reranker
            strategy:
              metrics: [retrieval_f1, retrieval_recall]
            modules:
              - module_type: pass_reranker
              - module_type: flashrank_reranker
          - node_type: generator
            strategy:
              metrics: [generation_f1, rouge]
            modules:
              - module_type: llm
                llm: ollama/gemma4:e2b
                prompt: [default, detailed]""")
    print("\nKey points:")
    print("  - node_lines group nodes into a sequential pipeline")
    print("  - Each node lists modules with parameter ranges to explore")
    print("  - strategy.metrics controls which metric drives optimization")
    print("  - AutoRAG's greedy algorithm picks the best module at each node")


def main() -> None:
    """Run all demonstration steps for L1-M1.2."""
    print("=" * 60)
    print("  L1-M1.2 — Installing and Project Setup")
    print("=" * 60)
    print("\nThis lesson walks through AutoRAG project structure, data")
    print("format requirements, validation utilities, and CLI commands.")

    explain_project_structure()
    qa_df = create_qa_dataframe()
    corpus_df = create_corpus_dataframe()
    validate_datasets(qa_df, corpus_df)
    show_cli_commands()
    show_sample_config()

    print(f"\n{'=' * 60}\n  Lesson Complete\n{'=' * 60}")
    print("\nYou now understand the AutoRAG project layout, data formats,")
    print("and validation workflow. In the next module (L1-M2.1), you")
    print("will create a real QA evaluation dataset from your own documents.")


if __name__ == "__main__":
    main()
