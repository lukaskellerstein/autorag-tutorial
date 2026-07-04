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
    """Print and explain AutoRAG's architecture: separate pipelines + optimization nodes."""
    print_header("Pipeline Architecture")

    print("AutoRAG separates data preparation from pipeline optimization:\n")

    print("  SEPARATE YAML-DRIVEN PIPELINES (run before optimization):")
    print("  ----------------------------------------------------------")
    print("  Parsing   — Converts raw documents (PDF, DOCX, HTML, etc.)")
    print("              into plain text. Configured in its own YAML.")
    print("  Chunking  — Splits extracted text into smaller pieces using")
    print("              token-based, sentence-based, or semantic strategies.")
    print("              Also configured in its own YAML.")
    print()
    print("  These are NOT optimization nodes. They produce the corpus")
    print("  (chunked passages) that the optimization pipeline operates on.")
    print()

    print("  OPTIMIZATION PIPELINE — 8 NODE TYPES:")
    print("  ----------------------------------------------------------")
    nodes: list[dict[str, str]] = [
        {
            "name": "1. Query Expansion",
            "desc": "Expand or rephrase the user query (e.g., HyDE,\n"
                    "              multi-query, decomposition) to improve retrieval.",
        },
        {
            "name": "2. Retrieval",
            "desc": "Find relevant passages. Three variants: lexical\n"
                    "              (BM25), semantic (vector embeddings), and hybrid\n"
                    "              (combining sparse + dense scores).",
        },
        {
            "name": "3. Passage Augmenter",
            "desc": "Augment retrieved passages with surrounding context\n"
                    "              (e.g., adjacent chunks from the same document).",
        },
        {
            "name": "4. Passage Reranker",
            "desc": "Reorder passages by relevance using cross-encoders,\n"
                    "              ColBERT, FlashRank, or other reranking models.",
        },
        {
            "name": "5. Passage Filter",
            "desc": "Remove low-quality or irrelevant passages using\n"
                    "              similarity thresholds, percentile cutoffs, or recency.",
        },
        {
            "name": "6. Passage Compressor",
            "desc": "Compress or summarize passages to reduce token usage\n"
                    "              (tree_summarize, refine, LongLLMLingua).",
        },
        {
            "name": "7. Prompt Maker",
            "desc": "Format query + passages into a prompt using templates\n"
                    "              and ordering strategies (e.g., long context reorder).",
        },
        {
            "name": "8. Generator",
            "desc": "Send the prompt to an LLM (via LlamaIndex) and\n"
                    "              generate the final answer.",
        },
    ]

    for node in nodes:
        print(f"  {node['name']}")
        print(f"              {node['desc']}")
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


def print_pass_modules() -> None:
    """Explain the pass module concept."""
    print_header("Pass Modules — Testing Whether to Skip a Node")

    print(
        "Every optional node has a 'pass' variant that does nothing —\n"
        "it forwards input unchanged. AutoRAG includes these so it can\n"
        "test whether SKIPPING a node produces better results than any\n"
        "active module. This is a key insight: sometimes less is more.\n"
    )
    print("  Pass modules:")
    pass_modules: dict[str, str] = {
        "pass_query_expansion": "Skip query expansion — use the original query",
        "pass_reranker": "Skip reranking — keep the retriever's ordering",
        "pass_passage_augmenter": "Skip augmentation — use passages as-is",
        "pass_passage_filter": "Skip filtering — keep all retrieved passages",
        "pass_compressor": "Skip compression — send full passages to the LLM",
    }
    for mod, desc in pass_modules.items():
        print(f"    - {mod:30s} {desc}")
    print()
    print(
        "If pass_reranker wins at the reranker node, it means your\n"
        "retriever is already returning well-ordered results and adding\n"
        "a reranker only hurts (or doesn't help enough to justify the cost)."
    )


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
        "Passage Augmenters": [
            "pass_passage_augmenter",
            "prev_next_augmenter",
        ],
        "Passage Rerankers (17+)": [
            "pass_reranker",
            "monot5",
            "tart",
            "upr",
            "koreranker",
            "cohere_reranker",
            "rankgpt",
            "jina_reranker",
            "colbert_reranker",
            "sentence_transformer_reranker",
            "flag_embedding_reranker",
            "flag_embedding_llm_reranker",
            "time_reranker",
            "openvino_reranker",
            "voyageai_reranker",
            "mixedbreadai_reranker",
            "flashrank_reranker",
        ],
        "Passage Filters": [
            "pass_passage_filter",
            "similarity_threshold_cutoff",
            "similarity_percentile_cutoff",
            "recency_filter",
            "threshold_cutoff",
            "percentile_cutoff",
        ],
        "Passage Compressors": [
            "pass_compressor",
            "tree_summarize",
            "refine",
            "longllmlingua",
        ],
        "Prompt Makers": [
            "fstring",
            "chat_fstring",
            "long_context_reorder",
            "window_replacement",
        ],
        "Generators": [
            "llama_index_llm",
            "openai_llm",
            "vllm",
            "vllm_api",
            "minimax_llm",
        ],
    }

    for category, modules in catalog.items():
        print(f"  {category}:")
        for mod in modules:
            print(f"    - {mod}")
        print()


def print_evaluation_metrics() -> None:
    """Print retrieval, generation, and specialized evaluation metrics."""
    print_header("Evaluation Metrics")

    print("  Retrieval Metrics:")
    retrieval_metrics: dict[str, str] = {
        "retrieval_precision": "Fraction of retrieved docs that are relevant (@k)",
        "retrieval_recall": "Fraction of relevant docs retrieved (@k)",
        "retrieval_f1": "Harmonic mean of precision and recall",
        "retrieval_mrr": "Mean Reciprocal Rank of first relevant result",
        "retrieval_ndcg": "Normalized Discounted Cumulative Gain (rank-aware)",
        "retrieval_map": "Mean Average Precision across queries",
    }
    for metric, desc in retrieval_metrics.items():
        print(f"    - {metric:30s} {desc}")

    print()
    print("  Generation Metrics:")
    generation_metrics: dict[str, str] = {
        "bleu": "N-gram overlap with reference answer",
        "rouge": "Recall-oriented n-gram overlap (ROUGE-1/2/L)",
        "meteor": "Alignment-based metric (synonyms + stemming)",
        "generation_f1": "Token-level F1 between generated and reference",
        "sem_score": "Semantic similarity via embeddings (SemScore)",
        "bert_score": "Contextual embedding similarity (BERTScore)",
        "g_eval (coherence)": "LLM-as-judge: logical flow and readability",
        "g_eval (consistency)": "LLM-as-judge: factual alignment with source",
        "g_eval (fluency)": "LLM-as-judge: grammatical quality",
        "g_eval (relevance)": "LLM-as-judge: answer relevance to the query",
        "faithfulness": "Whether the answer is grounded in retrieved passages",
    }
    for metric, desc in generation_metrics.items():
        print(f"    - {metric:30s} {desc}")

    print()
    print("  Passage Compressor Metrics:")
    compressor_metrics: dict[str, str] = {
        "retrieval_token_f1": "Token-level F1 after compression",
        "retrieval_token_recall": "Token-level recall after compression",
        "retrieval_token_precision": "Token-level precision after compression",
    }
    for metric, desc in compressor_metrics.items():
        print(f"    - {metric:30s} {desc}")

    print()
    print("  RAGAS Metrics:")
    ragas_metrics: dict[str, str] = {
        "ragas_context_precision": "Relevance of retrieved context (RAGAS)",
    }
    for metric, desc in ragas_metrics.items():
        print(f"    - {metric:30s} {desc}")


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
    print_pass_modules()
    print_modules_per_node()
    print_evaluation_metrics()
    print_greedy_optimization()

    print_header("Done")
    print("You now have a high-level understanding of AutoRAG.")
    print("Next lesson: L1-M1.2 — Setting up an AutoRAG project.")


if __name__ == "__main__":
    main()
