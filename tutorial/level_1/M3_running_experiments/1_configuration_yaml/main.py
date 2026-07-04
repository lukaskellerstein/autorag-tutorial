"""
L1-M3.1 — Configuration YAML

This lesson walks through the AutoRAG configuration YAML format,
explaining how node_lines, nodes, strategies, and modules work
together to define an optimization experiment.
"""

import os
from pathlib import Path
from typing import Any

import yaml


def load_config() -> dict[str, Any]:
    """Load config.yaml and print the raw structure."""
    config_path = Path(__file__).parent / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path) as f:
        config = yaml.safe_load(f)

    print(f"Loaded config from: {config_path.name}")
    print(f"Top-level keys: {list(config.keys())}")
    if "vectordb" in config:
        print(f"Vector DB configs: {len(config['vectordb'])}")
    print(f"Number of node_lines: {len(config['node_lines'])}")
    return config


def explain_structure(config: dict[str, Any]) -> None:
    """Walk through the hierarchical config structure."""
    if "vectordb" in config:
        print("Top-level sections:")
        print("  vectordb            (vector database configuration)")
        print("  node_lines          (pipeline stages)\n")
    print("AutoRAG config follows a 4-level hierarchy:\n")
    print("  node_lines          (top level — pipeline stages)")
    print("    -> nodes          (processing steps within a stage)")
    print("      -> strategy     (how to evaluate and select the best)")
    print("      -> modules      (implementations to compare)")
    print()

    for nl_idx, node_line in enumerate(config["node_lines"]):
        nl_name = node_line["node_line_name"]
        print(f"  node_line [{nl_idx}]: {nl_name}")

        for n_idx, node in enumerate(node_line["nodes"]):
            node_type = node["node_type"]
            strategy = node.get("strategy", {})
            metrics = strategy.get("metrics", [])
            modules = node.get("modules", [])

            # Handle metrics that may be strings or dicts
            metric_names: list[str] = []
            for m in metrics:
                if isinstance(m, str):
                    metric_names.append(m)
                elif isinstance(m, dict):
                    metric_names.append(m.get("metric_name", str(m)))

            print(f"    node [{n_idx}]: {node_type}")
            print(f"      strategy metrics: {metric_names}")
            print(f"      modules ({len(modules)}):")
            for mod in modules:
                print(f"        - {mod['module_type']}")
        print()


def explain_node_lines(config: dict[str, Any]) -> None:
    """Explain how node_lines chain together sequentially."""
    print("Node lines define the high-level stages of the RAG pipeline.")
    print("They execute sequentially: the output of one feeds into the next.\n")

    for i, node_line in enumerate(config["node_lines"]):
        name = node_line["node_line_name"]
        nodes = node_line["nodes"]
        node_types = [n["node_type"] for n in nodes]

        print(f"  Stage {i + 1}: {name}")
        print(f"    Nodes: {' -> '.join(node_types)}")

        if "retrieve" in name:
            print("    Purpose: Find and rank relevant passages from the corpus")
        elif "post_retrieve" in name:
            print("    Purpose: Format prompts and generate answers from retrieved passages")
        print()

    print("Pipeline flow:")
    all_nodes: list[str] = []
    for node_line in config["node_lines"]:
        for node in node_line["nodes"]:
            all_nodes.append(node["node_type"])
    print(f"  {' -> '.join(all_nodes)}")


def count_configurations(config: dict[str, Any]) -> None:
    """Count how many module configurations each node will test."""
    total_configs = 0

    print("AutoRAG tests all module configurations at each node,")
    print("then selects the best before moving to the next node.\n")

    for node_line in config["node_lines"]:
        print(f"  {node_line['node_line_name']}:")

        for node in node_line["nodes"]:
            node_type = node["node_type"]
            modules = node.get("modules", [])
            node_configs = 0

            for mod in modules:
                mod_type = mod["module_type"]
                # Count parameter combinations for this module
                param_combos = 1
                combo_details: list[str] = []

                for key, value in mod.items():
                    if key == "module_type":
                        continue
                    if isinstance(value, list):
                        param_combos *= len(value)
                        combo_details.append(f"{key}={len(value)} values")

                node_configs += param_combos
                if combo_details:
                    print(f"    {mod_type}: {param_combos} config(s) ({', '.join(combo_details)})")
                else:
                    print(f"    {mod_type}: {param_combos} config(s)")

            print(f"    -> {node_type} total: {node_configs} configuration(s)")
            total_configs += node_configs
        print()

    print(f"  Grand total: {total_configs} module configurations to evaluate")


def explain_strategy(config: dict[str, Any]) -> None:
    """Explain the strategy section and greedy optimization."""
    print("The 'strategy' section in each node controls optimization:\n")

    print("  1. METRICS — What to measure")
    print("     Each strategy lists metrics used to score module outputs.")
    print("     AutoRAG evaluates every module configuration against these")
    print("     metrics using the ground truth from your QA dataset.\n")

    print("  2. GREEDY SELECTION — How the best is chosen")
    print("     AutoRAG uses a greedy algorithm:")
    print("       a. Run ALL module configs at the current node")
    print("       b. Score each using the strategy metrics")
    print("       c. Select the best-performing configuration")
    print("       d. Pass its output to the next node")
    print("       e. Repeat for each subsequent node\n")

    print("  3. GENERATOR_MODULES (prompt_maker only)")
    print("     The prompt_maker node needs a generator to evaluate prompt")
    print("     quality. The generator_modules field specifies which LLM")
    print("     to use for this evaluation step.\n")

    print("Strategy sections found in this config:")
    for node_line in config["node_lines"]:
        for node in node_line["nodes"]:
            strategy = node.get("strategy", {})
            metrics = strategy.get("metrics", [])

            metric_names: list[str] = []
            for m in metrics:
                if isinstance(m, str):
                    metric_names.append(m)
                elif isinstance(m, dict):
                    metric_names.append(m.get("metric_name", str(m)))

            print(f"  {node['node_type']}: {metric_names}")


def explain_metrics() -> None:
    """Print a reference table of available metrics."""
    metrics = [
        ("retrieval_f1", "Retrieval", "Harmonic mean of retrieval precision and recall"),
        ("retrieval_recall", "Retrieval", "Fraction of relevant documents that were retrieved"),
        ("retrieval_precision", "Retrieval", "Fraction of retrieved documents that are relevant"),
        ("retrieval_ndcg", "Retrieval", "Normalized discounted cumulative gain — rewards relevant docs ranked higher"),
        ("retrieval_mrr", "Retrieval", "Mean reciprocal rank — position of the first relevant result"),
        ("retrieval_map", "Retrieval", "Mean average precision — average precision across all recall levels"),
        ("bleu", "Generation", "N-gram overlap between generated and reference text"),
        ("meteor", "Generation", "Alignment-based similarity using synonyms and stemming"),
        ("rouge", "Generation", "Recall-oriented n-gram overlap with reference text"),
        ("sem_score", "Generation", "Cosine similarity of sentence embeddings (semantic similarity)"),
        ("g_eval", "Generation", "LLM-as-judge — coherence, consistency, fluency, relevance"),
        ("bert_score", "Generation", "Token-level semantic similarity using BERT embeddings"),
        ("faithfulness", "Generation", "Measures whether the answer is supported by retrieved context"),
        ("retrieval_token_f1", "Compressor", "Token-level F1 between compressed and original passages"),
        ("retrieval_token_recall", "Compressor", "Token-level recall of compressed passages"),
        ("retrieval_token_precision", "Compressor", "Token-level precision of compressed passages"),
        ("context_precision", "RAGAS", "Relevance of retrieved context (no ground truth needed)"),
    ]

    print(f"  {'Metric':<28} {'Category':<12} {'Description'}")
    print(f"  {'-' * 28} {'-' * 12} {'-' * 55}")

    for name, category, description in metrics:
        print(f"  {name:<28} {category:<12} {description}")

    print(f"\n  Total: {len(metrics)} metrics available")
    print(f"  Retrieval metrics require retrieval_gt in the QA dataset")
    print(f"  Generation metrics require generation_gt in the QA dataset")
    print(f"  Compressor metrics evaluate passage compression quality")
    print(f"  RAGAS context_precision does not require retrieval ground truth")


def explain_vectordb_section(config: dict[str, Any]) -> None:
    """Explain the top-level vectordb configuration."""
    vectordb = config.get("vectordb")
    if not vectordb:
        print("No vectordb section found in this config.")
        print("AutoRAG will use default in-memory settings.")
        return

    print("The top-level 'vectordb' section configures the vector database")
    print("backend used by retrieval nodes (vector, hybrid_rrf, hybrid_cc).\n")

    for db in vectordb:
        print(f"  Name:            {db.get('name', 'N/A')}")
        print(f"  DB type:         {db.get('db_type', 'N/A')}")
        print(f"  Client type:     {db.get('client_type', 'N/A')}")
        print(f"  Path:            {db.get('path', 'N/A')}")
        print(f"  Embedding model: {db.get('embedding_model', 'N/A')}")
        print(f"  Collection:      {db.get('collection_name', 'N/A')}")
        print()

    print("Supported vector databases:")
    print("  Chroma, Milvus, Weaviate, Pinecone, Couchbase, Qdrant")
    print()
    print("The vectordb section is separate from node_lines because it")
    print("defines shared infrastructure that multiple retrieval modules use.")


def explain_pass_modules() -> None:
    """Explain the pass_* module concept."""
    print("Every node type has a pass_* variant (except generator):\n")

    pass_modules = [
        ("pass_query_expansion", "Skips query expansion — uses the original query as-is"),
        ("pass_passage_augmenter", "Skips augmentation — uses retrieved passages without adding context"),
        ("pass_reranker", "Skips reranking — keeps the retriever's original ordering"),
        ("pass_passage_filter", "Skips filtering — keeps all retrieved passages"),
        ("pass_compressor", "Skips compression — passes full passage text to the prompt"),
    ]

    for name, desc in pass_modules:
        print(f"  {name:<28s} {desc}")

    print()
    print("Why include pass modules?")
    print("  AutoRAG tests whether each processing step actually helps.")
    print("  If pass_reranker wins over flashrank_reranker, it means")
    print("  reranking hurts performance on your data — a valuable finding.")
    print("  This is fundamental to AutoRAG's optimization philosophy:")
    print("  more processing is not always better.")


def explain_optimization_strategies() -> None:
    """Explain the optimization strategy options."""
    print("The strategy section supports different optimization methods:\n")

    strategies = [
        ("mean (default)", "Averages metric scores across all QA pairs.\n"
         "                         Best for balanced optimization."),
        ("rank", "Uses reciprocal rank fusion to combine metric rankings.\n"
         "                         Robust when metrics have different scales."),
        ("normalize_mean", "Normalizes scores to [0,1] before averaging.\n"
         "                         Useful when combining metrics with different ranges."),
    ]

    for name, desc in strategies:
        print(f"  {name:<24s} {desc}")

    print()
    print("Example usage in YAML:")
    print("  strategy:")
    print("    metrics: [retrieval_f1, retrieval_recall]")
    print("    strategy_name: rank  # default is 'mean'")


def explain_speed_threshold() -> None:
    """Explain the speed_threshold parameter."""
    print("The speed_threshold parameter adds a latency constraint to node")
    print("optimization. Modules slower than the threshold are excluded,")
    print("even if they score higher on quality metrics.\n")

    print("Example usage in YAML:")
    print("  strategy:")
    print("    metrics: [retrieval_f1, retrieval_recall]")
    print("    speed_threshold: 5.0  # seconds per query\n")

    print("When to use speed_threshold:")
    print("  - Production systems with response time SLAs")
    print("  - When a module scores slightly better but is 10x slower")
    print("  - To automatically filter out impractical configurations\n")

    print("Without speed_threshold, AutoRAG selects purely on quality.")
    print("With it, quality is optimized subject to the latency constraint.")


def validate_config(config: dict[str, Any]) -> None:
    """Validate that the config has all required keys."""
    errors: list[str] = []
    warnings: list[str] = []

    # Check top-level
    if "node_lines" not in config:
        errors.append("Missing top-level 'node_lines' key")
        print("FAILED — missing 'node_lines'")
        return

    for nl_idx, node_line in enumerate(config["node_lines"]):
        nl_prefix = f"node_lines[{nl_idx}]"

        # Check node_line_name
        if "node_line_name" not in node_line:
            errors.append(f"{nl_prefix}: missing 'node_line_name'")
        else:
            nl_prefix = f"{node_line['node_line_name']}"

        # Check nodes
        if "nodes" not in node_line:
            errors.append(f"{nl_prefix}: missing 'nodes'")
            continue

        for n_idx, node in enumerate(node_line["nodes"]):
            n_prefix = f"{nl_prefix}.nodes[{n_idx}]"

            if "node_type" not in node:
                errors.append(f"{n_prefix}: missing 'node_type'")
            else:
                n_prefix = f"{nl_prefix}.{node['node_type']}"

            if "modules" not in node:
                errors.append(f"{n_prefix}: missing 'modules'")
            else:
                for m_idx, module in enumerate(node["modules"]):
                    m_prefix = f"{n_prefix}.modules[{m_idx}]"
                    if "module_type" not in module:
                        errors.append(f"{m_prefix}: missing 'module_type'")

            if "strategy" not in node:
                warnings.append(f"{n_prefix}: no 'strategy' defined (will use defaults)")

    # Print results
    if errors:
        print(f"Validation FAILED — {len(errors)} error(s):")
        for err in errors:
            print(f"  ERROR: {err}")
    else:
        print("Validation PASSED — all required keys present")

    if warnings:
        print(f"\n  {len(warnings)} warning(s):")
        for warn in warnings:
            print(f"  WARNING: {warn}")

    # Summary
    total_nodes = sum(len(nl["nodes"]) for nl in config["node_lines"])
    total_modules = sum(
        len(node.get("modules", []))
        for nl in config["node_lines"]
        for node in nl["nodes"]
    )
    print(f"\n  Config summary:")
    print(f"    Node lines: {len(config['node_lines'])}")
    print(f"    Total nodes: {total_nodes}")
    print(f"    Total modules: {total_modules}")


def main() -> None:
    """Run all steps of the configuration YAML lesson."""
    print("=" * 60)
    print("L1-M3.1 — Configuration YAML")
    print("=" * 60)

    print("\n" + "=" * 60)
    print("Step 1: Load configuration")
    print("=" * 60)
    config = load_config()

    print("\n" + "=" * 60)
    print("Step 2: Understand the structure")
    print("=" * 60)
    explain_structure(config)

    print("\n" + "=" * 60)
    print("Step 3: Vector database configuration")
    print("=" * 60)
    explain_vectordb_section(config)

    print("\n" + "=" * 60)
    print("Step 4: Node lines — pipeline stages")
    print("=" * 60)
    explain_node_lines(config)

    print("\n" + "=" * 60)
    print("Step 5: Pass modules — testing whether nodes help")
    print("=" * 60)
    explain_pass_modules()

    print("\n" + "=" * 60)
    print("Step 6: Count configurations to test")
    print("=" * 60)
    count_configurations(config)

    print("\n" + "=" * 60)
    print("Step 7: Strategy — greedy optimization")
    print("=" * 60)
    explain_strategy(config)

    print("\n" + "=" * 60)
    print("Step 8: Optimization strategies")
    print("=" * 60)
    explain_optimization_strategies()

    print("\n" + "=" * 60)
    print("Step 9: Speed threshold")
    print("=" * 60)
    explain_speed_threshold()

    print("\n" + "=" * 60)
    print("Step 10: Metrics reference")
    print("=" * 60)
    explain_metrics()

    print("\n" + "=" * 60)
    print("Step 11: Validate configuration")
    print("=" * 60)
    validate_config(config)

    print("\n" + "=" * 60)
    print("Done! You now understand AutoRAG's configuration format.")
    print("=" * 60)


if __name__ == "__main__":
    main()
