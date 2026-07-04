"""
L1-M1.1 — What is AutoRAG? How It Works

This lesson explores AutoRAG's architecture, pipeline stages, available
modules, evaluation metrics, and greedy optimization strategy.
AutoRAG is "AutoML for RAG" — give it evaluation data and it will
systematically explore combinations to find the optimal pipeline config.
"""


def print_header(title: str) -> None:
    """Print a formatted section header."""
    print()
    print("=" * 60)
    print(f"  {title}")
    print("=" * 60)
    print()


def print_version() -> None:
    """Print the installed AutoRAG version."""
    print_header("AutoRAG Version")
    try:
        import autorag

        version = getattr(autorag, "__version__", "unknown")
        print(f"AutoRAG is installed — version: {version}")
    except ImportError:
        print("AutoRAG is not installed yet.")
        print("Run 'uv sync' in this directory to install it.")


def print_architecture_overview() -> None:
    """Explain AutoRAG as 'AutoML for RAG'."""
    print_header("What is AutoRAG?")
    print(
        "AutoRAG is an AutoML-style framework for Retrieval-Augmented\n"
        "Generation (RAG). Instead of manually tuning every knob in your\n"
        "RAG pipeline — chunking strategy, embedding model, retrieval\n"
        "method, reranker, prompt template, LLM — you provide evaluation\n"
        "data (questions + ground-truth answers + corpus) and AutoRAG\n"
        "systematically explores combinations to find the best config.\n"
    )
    print("Core idea:")
    print("  1. You define a YAML config listing which modules to try")
    print("  2. AutoRAG runs every combination against your eval data")
    print("  3. It measures retrieval and generation quality with metrics")
    print("  4. It returns the optimal pipeline configuration")
    print()
    print(
        "This is analogous to how AutoML tools (AutoSklearn, FLAML, etc.)\n"
        "search over ML model hyperparameters — but for RAG pipelines."
    )


def print_pipeline_stages() -> None:
    """Print and explain each pipeline stage."""
    print_header("Pipeline Stages")

    stages: list[dict[str, str]] = [
        {
            "name": "1. Parsing",
            "desc": (
                "Document -> text extraction. Converts raw documents\n"
                "      (PDF, DOCX, HTML, etc.) into plain text for processing."
            ),
        },
        {
            "name": "2. Chunking",
            "desc": (
                "Text -> chunks. Splits extracted text into smaller\n"
                "      pieces using token-based, sentence-based, or semantic\n"
                "      chunking strategies."
            ),
        },
        {
            "name": "3. Retrieval",
            "desc": (
                "Query -> relevant chunks. Finds the most relevant\n"
                "      chunks for a given query using BM25 (sparse), vector\n"
                "      search (dense), or hybrid approaches."
            ),
        },
        {
            "name": "4. Reranking",
            "desc": (
                "Re-score retrieved chunks. Uses cross-encoders,\n"
                "      FlashRank, ColBERT, or other models to reorder chunks\n"
                "      by true relevance to the query."
            ),
        },
        {
            "name": "5. Prompt Making",
            "desc": (
                "Query + passages -> formatted prompt. Assembles the\n"
                "      final prompt from the user query and retrieved passages\n"
                "      using templates and ordering strategies."
            ),
        },
        {
            "name": "6. Generation",
            "desc": (
                "Prompt -> answer. Sends the constructed prompt to an\n"
                "      LLM (via LlamaIndex) to generate the final response."
            ),
        },
    ]

    for stage in stages:
        print(f"  {stage['name']}")
        print(f"      {stage['desc']}")
        print()


def print_node_types() -> None:
    """Print available node types as a catalog."""
    print_header("Available Node Types")

    node_types: list[dict[str, str]] = [
        {"node": "query_expansion", "desc": "Expand or rephrase the user query"},
        {
            "node": "retrieval (lexical / sparse)",
            "desc": "Keyword-based retrieval (BM25 and variants)",
        },
        {
            "node": "retrieval (semantic / dense)",
            "desc": "Embedding-based vector similarity search",
        },
        {
            "node": "hybrid_retrieval",
            "desc": "Combine sparse and dense retrieval scores",
        },
        {
            "node": "passage_augmenter",
            "desc": "Augment retrieved passages with additional context",
        },
        {
            "node": "passage_reranker",
            "desc": "Reorder passages by relevance using a reranking model",
        },
        {
            "node": "passage_filter",
            "desc": "Remove low-quality or irrelevant passages",
        },
        {
            "node": "passage_compressor",
            "desc": "Compress or summarize passages to reduce token usage",
        },
        {"node": "prompt_maker", "desc": "Format query + passages into a prompt"},
        {"node": "generator", "desc": "Generate final answer from the prompt"},
    ]

    for nt in node_types:
        print(f"  - {nt['node']:40s} {nt['desc']}")


def print_modules_per_node() -> None:
    """Print available modules for each node type."""
    print_header("Available Modules per Node Type")

    catalog: dict[str, list[str]] = {
        "Query Expansion": [
            "pass_query_expansion",
            "query_decompose",
            "hyde",
            "multi_query_expansion",
        ],
        "Retrieval": [
            "bm25",
            "vectordb",
            "hybrid_rrf",
            "hybrid_cc",
        ],
        "Rerankers": [
            "pass_reranker",
            "monot5",
            "tart",
            "upr",
            "cohere_reranker",
            "rankgpt",
            "jina_reranker",
            "colbert_reranker",
            "sentence_transformer_reranker",
            "flag_embedding_reranker",
            "flashrank_reranker",
            "voyageai_reranker",
        ],
        "Passage Filters": [
            "pass_passage_filter",
            "similarity_threshold_cutoff",
            "similarity_percentile_cutoff",
            "recency_filter",
        ],
        "Passage Compressors": [
            "pass_compressor",
            "tree_summarize",
            "refine",
            "longllmlingua",
        ],
        "Prompt Makers": [
            "fstring",
            "long_context_reorder",
            "window_replacement",
        ],
        "Generators": [
            "llama_index_llm",
            "openai_llm",
            "vllm",
            "vllm_api",
        ],
    }

    for category, modules in catalog.items():
        print(f"  {category}:")
        for mod in modules:
            print(f"    - {mod}")
        print()


def print_evaluation_metrics() -> None:
    """Print retrieval and generation evaluation metrics."""
    print_header("Evaluation Metrics")

    print("  Retrieval Metrics:")
    retrieval_metrics: dict[str, str] = {
        "retrieval_f1": "Harmonic mean of precision and recall",
        "retrieval_recall": "Fraction of relevant docs retrieved",
        "retrieval_precision": "Fraction of retrieved docs that are relevant",
        "retrieval_ndcg": "Normalized Discounted Cumulative Gain (rank-aware)",
        "retrieval_mrr": "Mean Reciprocal Rank of first relevant result",
        "retrieval_map": "Mean Average Precision across queries",
    }
    for metric, desc in retrieval_metrics.items():
        print(f"    - {metric:25s} {desc}")

    print()
    print("  Generation Metrics:")
    generation_metrics: dict[str, str] = {
        "bleu": "N-gram overlap with reference answer",
        "meteor": "Alignment-based metric (synonyms + stemming)",
        "rouge": "Recall-oriented n-gram overlap (ROUGE-L, etc.)",
        "sem_score": "Semantic similarity via embeddings",
        "g_eval": "LLM-as-judge evaluation (GPT-based scoring)",
        "bert_score": "Contextual embedding similarity (BERTScore)",
    }
    for metric, desc in generation_metrics.items():
        print(f"    - {metric:25s} {desc}")


def print_greedy_optimization() -> None:
    """Explain the greedy optimization strategy."""
    print_header("Greedy Optimization Strategy")
    print(
        "AutoRAG uses a greedy, node-by-node optimization strategy:\n"
    )
    print("  1. Start at the first pipeline node (e.g., query expansion)")
    print("  2. Try every module configured for that node")
    print("  3. Evaluate each using the specified metrics")
    print("  4. Keep the BEST module for that node")
    print("  5. Move to the next node and repeat")
    print()
    print(
        "This is greedy because it locks in the best choice at each\n"
        "stage rather than exploring all possible end-to-end combos.\n"
        "The tradeoff: it is much faster than exhaustive search but\n"
        "may miss interactions between stages.\n"
    )
    print("Example optimization flow:")
    print("  Query Expansion:  try pass, decompose, hyde  -> best: hyde")
    print("  Retrieval:        try bm25, vectordb, hybrid -> best: hybrid_rrf")
    print("  Reranking:        try pass, colbert, flashrank -> best: colbert")
    print("  Prompt Making:    try fstring, reorder        -> best: fstring")
    print("  Generation:       try openai_llm, vllm        -> best: openai_llm")
    print()
    print("Result: an optimized YAML config you can deploy in production.")


def main() -> None:
    """Run all sections of the lesson."""
    print_version()
    print_architecture_overview()
    print_pipeline_stages()
    print_node_types()
    print_modules_per_node()
    print_evaluation_metrics()
    print_greedy_optimization()

    print_header("Done")
    print("You now have a high-level understanding of AutoRAG.")
    print("Next lesson: L1-M1.2 — Setting up an AutoRAG project.")


if __name__ == "__main__":
    main()
