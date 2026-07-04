"""
L1-M2.2 — Preparing the Corpus

Demonstrates how to prepare a document corpus in AutoRAG's required format.
Covers text preprocessing, metadata enrichment, validation, and persistence.
"""

import os
import re

import pandas as pd
from autorag.utils.preprocess import cast_corpus_dataset


# ---------------------------------------------------------------------------
# Raw documents: 10 educational texts about Python programming concepts
# ---------------------------------------------------------------------------

RAW_DOCUMENTS: list[dict[str, str]] = [
    {
        "topic": "variables",
        "text": """
            Variables in Python are names that refer to objects stored in memory.
            Python is dynamically typed, so you do not need to declare a variable's
            type before assigning a value.  Common built-in data types include int,
            float, str, bool, list, tuple, dict, and set.  Assignment uses the
            equals sign: `x = 42` binds the name x to an integer object.  Multiple
            assignment is supported: `a, b, c = 1, 2, 3`.  Variables can be
            reassigned to objects of a different type at any time, which provides
            flexibility but requires careful attention to avoid type-related bugs
            at runtime.  Use the `type()` built-in to inspect an object's type.
        """,
    },
    {
        "topic": "functions",
        "text": """
            Functions let you encapsulate reusable logic behind a descriptive name.
            Define a function with the `def` keyword, followed by the function name
            and a parenthesized list of parameters.  Python supports positional
            arguments, keyword arguments, default values, *args for variable
            positional arguments, and **kwargs for variable keyword arguments.
            Functions return a value with the `return` statement; if omitted, the
            function returns None.  First-class functions mean you can assign them
            to variables, pass them as arguments, and return them from other
            functions.  Docstrings placed as the first statement inside a function
            body provide inline documentation accessible via `help()`.
        """,
    },
    {
        "topic": "classes",
        "text": """
            Classes are blueprints for creating objects that bundle data and behavior
            together.  Define a class with the `class` keyword.  The `__init__`
            method is the constructor, called automatically when you create an
            instance.  Instance attributes are set on `self`, which refers to the
            current object.  Methods are functions defined inside a class body and
            receive `self` as their first parameter.  Python supports single and
            multiple inheritance, allowing a subclass to extend or override parent
            behavior.  Special (dunder) methods like `__str__`, `__repr__`, and
            `__eq__` let you customize how objects are printed, represented, and
            compared.  Object-oriented programming in Python encourages composition
            over deep inheritance hierarchies.
        """,
    },
    {
        "topic": "decorators",
        "text": """
            Decorators are a powerful pattern for modifying or extending functions
            and classes without changing their source code.  A decorator is a
            callable that takes another callable as input and returns a new callable.
            Apply a decorator with the `@decorator_name` syntax above a function
            definition.  Common built-in decorators include `@staticmethod`,
            `@classmethod`, and `@property`.  You can write custom decorators by
            defining a wrapper function that adds behavior before or after calling
            the original function.  Use `functools.wraps` to preserve the wrapped
            function's name, docstring, and other attributes.  Decorators are widely
            used in web frameworks, logging, access control, and caching.
        """,
    },
    {
        "topic": "generators",
        "text": """
            Generators provide a memory-efficient way to produce sequences of values
            lazily.  A generator function uses `yield` instead of `return` to produce
            values one at a time, suspending execution between each yield.  When you
            call a generator function, it returns a generator iterator without
            executing the body.  Iteration (via `next()` or a `for` loop) resumes
            execution until the next `yield`.  Generator expressions, written like
            list comprehensions but with parentheses, offer a concise inline syntax.
            Because generators produce values on demand rather than storing an entire
            sequence in memory, they are ideal for processing large datasets, reading
            files line by line, or implementing infinite sequences.
        """,
    },
    {
        "topic": "list_comprehensions",
        "text": """
            List comprehensions provide a concise syntax for creating lists from
            existing iterables.  The basic form is `[expression for item in
            iterable]`, which applies an expression to each element.  You can add a
            conditional filter: `[x for x in range(10) if x % 2 == 0]` produces
            only even numbers.  Nested comprehensions iterate over multiple
            sequences: `[x * y for x in rows for y in cols]`.  Python also supports
            dict comprehensions (`{k: v for k, v in pairs}`), set comprehensions
            (`{x for x in items}`), and generator expressions.  While comprehensions
            are Pythonic and often faster than equivalent for-loops, deeply nested
            or overly complex comprehensions hurt readability.  Prefer a regular loop
            when the logic spans more than two lines.
        """,
    },
    {
        "topic": "error_handling",
        "text": """
            Error handling in Python uses try/except blocks to catch and respond to
            exceptions.  Place risky code inside a `try` block and handle specific
            exceptions in one or more `except` clauses.  Always catch the most
            specific exception first; a bare `except:` catches everything, including
            KeyboardInterrupt, so avoid it.  The optional `else` clause runs only if
            no exception was raised, while `finally` runs unconditionally and is used
            for cleanup such as closing files.  Raise exceptions with `raise
            ValueError("message")`.  Define custom exception classes by subclassing
            `Exception`.  Context managers (`with` statement) automate resource
            cleanup and are preferred over try/finally for file and database handles.
        """,
    },
    {
        "topic": "modules",
        "text": """
            Modules and packages organize Python code into reusable, maintainable
            units.  A module is a single `.py` file that can be imported with the
            `import` statement.  A package is a directory containing an `__init__.py`
            file and zero or more submodules.  Use `from module import name` to
            import specific objects, or `import module` to access them via dotted
            notation.  The `sys.path` list controls where Python searches for
            modules.  Relative imports (`from . import sibling`) are used within
            packages.  The `__all__` variable in a module controls what is exported
            when `from module import *` is used.  Structuring code into well-named
            modules and packages improves readability, testability, and reuse across
            projects.
        """,
    },
    {
        "topic": "virtual_environments",
        "text": """
            Virtual environments isolate project dependencies so that different
            projects can use different versions of the same library without conflict.
            Create one with `python -m venv .venv`, then activate it with
            `source .venv/bin/activate` (Linux/macOS) or `.venv\\Scripts\\activate`
            (Windows).  Once active, `pip install` places packages in the
            environment's own site-packages directory rather than the system-wide
            location.  Pin dependencies with `pip freeze > requirements.txt` and
            reproduce them with `pip install -r requirements.txt`.  Modern tools
            like `uv` and `poetry` combine environment management with dependency
            resolution.  Always use a virtual environment for every project to avoid
            version collisions and ensure reproducible builds.
        """,
    },
    {
        "topic": "type_hints",
        "text": """
            Type hints annotate function signatures and variables with expected types,
            improving readability and enabling static analysis.  Use the colon syntax
            for parameters (`def greet(name: str) -> str:`) and variables
            (`count: int = 0`).  The `typing` module provides generic types like
            `List[int]`, `Dict[str, Any]`, `Optional[str]`, and `Union[int, str]`.
            From Python 3.10 onward, you can use built-in types directly in
            annotations (`list[int]`) and the `X | Y` union syntax.  Static type
            checkers such as mypy and pyright catch type errors before runtime.
            Type hints are not enforced at runtime by default; they serve as
            documentation and tooling support.  Adopting type hints in a codebase
            makes refactoring safer and helps new contributors understand interfaces.
        """,
    },
]


def normalize_whitespace(text: str) -> str:
    """Strip leading/trailing whitespace and collapse internal whitespace."""
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    return text


def preprocess_documents(docs: list[dict[str, str]]) -> list[dict[str, str]]:
    """Clean and deduplicate raw documents."""
    cleaned: list[dict[str, str]] = []
    seen_contents: set[str] = set()

    for doc in docs:
        content = normalize_whitespace(doc["text"])

        # Deduplicate by exact content match
        if content in seen_contents:
            print(f"  [SKIP] Duplicate detected for topic '{doc['topic']}'")
            continue

        seen_contents.add(content)
        cleaned.append({"topic": doc["topic"], "text": content})

    return cleaned


def build_corpus_dataframe(docs: list[dict[str, str]]) -> pd.DataFrame:
    """Build a corpus DataFrame with doc_id, contents, and metadata columns."""
    rows: list[dict] = []
    total = len(docs)

    for idx, doc in enumerate(docs):
        doc_id = f"doc_{idx + 1:03d}"
        prev_id = f"doc_{idx:03d}" if idx > 0 else None
        next_id = f"doc_{idx + 2:03d}" if idx < total - 1 else None

        metadata = {
            "last_modified_datetime": f"2024-01-{15 + idx:02d}T10:00:00",
            "prev_id": prev_id,
            "next_id": next_id,
            "source": "python_tutorial",
            "topic": doc["topic"],
        }

        rows.append({
            "doc_id": doc_id,
            "contents": doc["text"],
            "metadata": metadata,
        })

    return pd.DataFrame(rows)


def print_corpus_statistics(df: pd.DataFrame) -> None:
    """Print summary statistics about the corpus."""
    lengths = df["contents"].str.len()
    all_keys: set[str] = set()
    topics: set[str] = set()

    for meta in df["metadata"]:
        all_keys.update(meta.keys())
        topics.add(meta["topic"])

    print(f"  Total documents    : {len(df)}")
    print(f"  Avg content length : {lengths.mean():.0f} characters")
    print(f"  Min content length : {lengths.min()} characters")
    print(f"  Max content length : {lengths.max()} characters")
    print(f"  Metadata keys      : {sorted(all_keys)}")
    print(f"  Unique topics      : {sorted(topics)}")


def main() -> None:
    # ------------------------------------------------------------------
    print("=" * 60)
    print("Step 1: Raw Document Collection")
    print("=" * 60)
    print(f"  Loaded {len(RAW_DOCUMENTS)} raw documents")
    print(f"  Topics: {[d['topic'] for d in RAW_DOCUMENTS]}")

    # ------------------------------------------------------------------
    print()
    print("=" * 60)
    print("Step 2: Text Preprocessing")
    print("=" * 60)

    # Show before/after for the first document
    sample_raw = RAW_DOCUMENTS[0]["text"]
    sample_clean = normalize_whitespace(sample_raw)
    print(f"\n  --- Before (first 120 chars) ---")
    print(f"  {repr(sample_raw[:120])}")
    print(f"\n  --- After (first 120 chars) ---")
    print(f"  {repr(sample_clean[:120])}")

    cleaned_docs = preprocess_documents(RAW_DOCUMENTS)
    print(f"\n  Documents after preprocessing: {len(cleaned_docs)}")

    # ------------------------------------------------------------------
    print()
    print("=" * 60)
    print("Step 3: Building the Corpus DataFrame")
    print("=" * 60)
    corpus_df = build_corpus_dataframe(cleaned_docs)
    print(f"  DataFrame shape: {corpus_df.shape}")
    print(f"  Columns: {list(corpus_df.columns)}")
    print(f"\n  First 3 rows (doc_id and content preview):")
    for _, row in corpus_df.head(3).iterrows():
        print(f"    {row['doc_id']}: {row['contents'][:70]}...")

    # ------------------------------------------------------------------
    print()
    print("=" * 60)
    print("Step 4: Metadata Inspection")
    print("=" * 60)
    sample_meta = corpus_df.iloc[0]["metadata"]
    print(f"  Sample metadata (doc_001):")
    for key, value in sample_meta.items():
        print(f"    {key}: {value}")

    # ------------------------------------------------------------------
    print()
    print("=" * 60)
    print("Step 5: Validation with AutoRAG")
    print("=" * 60)
    validated_df = cast_corpus_dataset(corpus_df)
    print("  cast_corpus_dataset() passed successfully!")
    print(f"  Validated DataFrame shape: {validated_df.shape}")
    print(f"  Validated columns: {list(validated_df.columns)}")

    # ------------------------------------------------------------------
    print()
    print("=" * 60)
    print("Step 6: Corpus Statistics")
    print("=" * 60)
    print_corpus_statistics(corpus_df)

    # ------------------------------------------------------------------
    print()
    print("=" * 60)
    print("Step 7: Saving to Parquet")
    print("=" * 60)
    os.makedirs("data", exist_ok=True)
    corpus_df.to_parquet("data/corpus.parquet")
    print("  Saved corpus to data/corpus.parquet")

    # ------------------------------------------------------------------
    print()
    print("=" * 60)
    print("Step 8: Loading and Verification")
    print("=" * 60)
    loaded_df = pd.read_parquet("data/corpus.parquet")
    print(f"  Loaded DataFrame shape: {loaded_df.shape}")
    print(f"  Columns: {list(loaded_df.columns)}")
    print(f"\n  First 5 rows:")
    for _, row in loaded_df.head(5).iterrows():
        print(f"    {row['doc_id']}: {row['contents'][:60]}...")
    print("\n  Corpus saved and reloaded successfully!")


if __name__ == "__main__":
    main()
