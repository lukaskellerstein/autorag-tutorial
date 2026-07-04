"""
L2-M2.1 — Custom Modules

Learn how AutoRAG's module system works, build a custom reranker,
and understand how to register custom modules for use in evaluations.
"""

from typing import List, Tuple


# ---------------------------------------------------------------------------
# 1. Explain the module system
# ---------------------------------------------------------------------------

def explain_module_system() -> None:
    print("=" * 60)
    print("AutoRAG Module Architecture")
    print("=" * 60)
    print("""
AutoRAG pipelines are composed of modules — pluggable components
that handle one step in the RAG pipeline.

BaseModule (abstract base class)
  pure(previous_result, *args, **kwargs) -- main execution method
  _pure(*args, **kwargs)                -- internal implementation
  cast_to_run(previous_result)          -- extracts inputs from DataFrame

Specialized Base Classes:
  BaseRetrieval       -- for retrieval modules (loads corpus)
  BasePassageReranker -- for reranking modules
  BaseGenerator       -- for generation modules (takes llm parameter)
  (Prompt Maker base) -- for prompt construction

Data flow:
  Each module receives input as a pandas DataFrame from the
  previous node and returns a DataFrame with its results.
  This uniform interface lets AutoRAG swap modules freely
  during evaluation.
""")


# ---------------------------------------------------------------------------
# 2. Custom reranker: KeywordBoostReranker
# ---------------------------------------------------------------------------

class KeywordBoostReranker:
    """Custom reranker that boosts passages containing query keywords."""

    def __init__(self, project_dir: str, boost_factor: float = 2.0):
        self.project_dir = project_dir
        self.boost_factor = boost_factor

    def rerank(
        self,
        queries: List[str],
        contents_list: List[List[str]],
        scores_list: List[List[float]],
        ids_list: List[List[str]],
        top_k: int = 3,
    ) -> Tuple[List[List[str]], List[List[str]], List[List[float]]]:
        """Rerank passages by boosting scores when query keywords appear."""
        result_contents: List[List[str]] = []
        result_ids: List[List[str]] = []
        result_scores: List[List[float]] = []

        for query, contents, scores, ids in zip(
            queries, contents_list, scores_list, ids_list
        ):
            query_words = set(query.lower().split())
            boosted_scores: List[float] = []
            for content, score in zip(contents, scores):
                content_words = set(content.lower().split())
                overlap = len(query_words & content_words)
                boosted = score + (overlap * self.boost_factor)
                boosted_scores.append(boosted)

            ranked = sorted(
                zip(contents, ids, boosted_scores),
                key=lambda x: x[2],
                reverse=True,
            )[:top_k]

            result_contents.append([r[0] for r in ranked])
            result_ids.append([r[1] for r in ranked])
            result_scores.append([r[2] for r in ranked])

        return result_contents, result_ids, result_scores


def create_custom_reranker() -> None:
    print("=" * 60)
    print("Custom Module: KeywordBoostReranker")
    print("=" * 60)
    print("""
KeywordBoostReranker adds a score bonus for every query keyword
found in a passage.  This is a simple but effective heuristic
that can complement neural rerankers.

Constructor parameters:
  project_dir  -- path to the AutoRAG project directory
  boost_factor -- score bonus per overlapping keyword (default 2.0)

The rerank() method signature mirrors BasePassageReranker._pure():
  queries        : List[str]
  contents_list  : List[List[str]]
  scores_list    : List[List[float]]
  ids_list       : List[List[str]]
  top_k          : int
  Returns -> (contents, ids, scores) — each a list of lists
""")


# ---------------------------------------------------------------------------
# 3. Demonstrate the custom reranker
# ---------------------------------------------------------------------------

def demonstrate_custom_reranker() -> None:
    print("=" * 60)
    print("Running KeywordBoostReranker")
    print("=" * 60)

    queries = ["What are Python decorators?"]
    contents = [[
        "Decorators modify functions in Python using the @ syntax.",
        "Python has many features including list comprehensions.",
        "The @ symbol is used for decorators in Python code.",
        "Java annotations also use the @ symbol but differ from Python.",
    ]]
    scores = [[0.80, 0.70, 0.60, 0.55]]
    ids = [["doc_001", "doc_002", "doc_003", "doc_004"]]

    print("\nBefore reranking:")
    for content, score, doc_id in zip(contents[0], scores[0], ids[0]):
        print(f"  [{doc_id}] score={score:.2f}  {content}")

    reranker = KeywordBoostReranker(project_dir=".", boost_factor=2.0)
    new_contents, new_ids, new_scores = reranker.rerank(
        queries, contents, scores, ids, top_k=3
    )

    print(f"\nAfter reranking (boost_factor=2.0, top_k=3):")
    for content, doc_id, score in zip(
        new_contents[0], new_ids[0], new_scores[0]
    ):
        print(f"  [{doc_id}] score={score:.2f}  {content}")
    print()


# ---------------------------------------------------------------------------
# 4. Show built-in module structure
# ---------------------------------------------------------------------------

def show_builtin_module_structure() -> None:
    print("=" * 60)
    print("Built-in Module Example: FlashRankReranker")
    print("=" * 60)
    print("""
FlashRankReranker (from autorag.nodes.passagereranker.flashrank):

class FlashRankReranker(BasePassageReranker):
    def __init__(self, project_dir, model_name="ms-marco-MiniLM-L-12-v2"):
        super().__init__(project_dir)
        self.model = Ranker(model_name=model_name)

    @result_to_dataframe(["retrieved_contents", "retrieved_ids",
                          "retrieve_scores"])
    def pure(self, previous_result, *args, **kwargs):
        queries, contents, scores, ids = self.cast_to_run(
            previous_result
        )
        return self._pure(queries, contents, ids, scores, top_k)

    def _pure(self, queries, contents_list, ids_list,
              scores_list, top_k):
        # Neural reranking using FlashRank model ...

Key patterns:
  - Inherit from the appropriate base class
  - Use @result_to_dataframe to convert output to a DataFrame
  - Implement pure() as the public entry point
  - Implement _pure() with the core logic
  - cast_to_run() extracts typed inputs from the DataFrame
""")


# ---------------------------------------------------------------------------
# 5. Explain registration
# ---------------------------------------------------------------------------

def explain_registration() -> None:
    print("=" * 60)
    print("Registering Custom Modules in AutoRAG")
    print("=" * 60)
    print("""
To integrate a custom module into AutoRAG:

1. Create your module class following the base class interface
   (see KeywordBoostReranker above for the method signatures).

2. Register it at runtime before running evaluation:

   from autorag.support import get_support_modules
   support_modules = get_support_modules()
   support_modules["keyword_boost_reranker"] = KeywordBoostReranker

3. Reference it in your config.yaml:

   - module_type: keyword_boost_reranker
     boost_factor: 2.0

Available base classes for different node types:
  BaseRetrieval        -- custom retrievers (BM25, hybrid, etc.)
  BasePassageReranker  -- custom rerankers (keyword, neural, etc.)
  BaseGenerator        -- custom generators (wrapping any LLM)

Each base class provides cast_to_run() which extracts the
correct columns from the input DataFrame, so your _pure()
receives clean typed data.
""")


# ---------------------------------------------------------------------------
# 6. Explain custom chunkers
# ---------------------------------------------------------------------------

def explain_custom_chunker() -> None:
    print("=" * 60)
    print("Custom Chunkers")
    print("=" * 60)
    print("""
Custom chunkers are defined differently -- they extend the data
creation pipeline rather than the evaluation pipeline.

They use the Corpus / Raw fluent API:

  from autorag.data.corpus import Raw

  raw = Raw.from_parquet("raw_data.parquet")
  corpus = raw.chunk("my_custom_chunker", chunk_size=512)
  corpus.to_parquet("corpus.parquet")

Built-in chunkers include:
  - token        -- split by token count
  - sentence     -- split by sentence boundaries
  - recursive    -- LangChain RecursiveCharacterTextSplitter

To create a custom chunker, define a function that takes a list
of documents and returns chunked passages, then register it in
AutoRAG's chunker registry before calling raw.chunk().
""")


# ---------------------------------------------------------------------------
# 7. Demonstrate a second reranker for comparison
# ---------------------------------------------------------------------------

def demonstrate_boost_factor_comparison() -> None:
    print("=" * 60)
    print("Comparing Boost Factors")
    print("=" * 60)

    queries = ["machine learning algorithms"]
    contents = [[
        "Deep learning algorithms are a subset of machine learning.",
        "Sorting algorithms include quicksort and mergesort.",
        "Machine learning uses statistical methods to learn from data.",
        "Algorithms are step-by-step procedures for calculations.",
    ]]
    scores = [[0.75, 0.72, 0.70, 0.68]]
    ids = [["doc_a", "doc_b", "doc_c", "doc_d"]]

    for boost in [1.0, 2.0, 5.0]:
        reranker = KeywordBoostReranker(project_dir=".", boost_factor=boost)
        new_contents, new_ids, new_scores = reranker.rerank(
            queries, contents, scores, ids, top_k=3
        )
        print(f"\nboost_factor = {boost}:")
        for content, doc_id, score in zip(
            new_contents[0], new_ids[0], new_scores[0]
        ):
            print(f"  [{doc_id}] score={score:.2f}  {content[:60]}...")
    print()


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    explain_module_system()
    create_custom_reranker()
    demonstrate_custom_reranker()
    show_builtin_module_structure()
    explain_registration()
    explain_custom_chunker()
    demonstrate_boost_factor_comparison()

    print("=" * 60)
    print("Lesson complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
