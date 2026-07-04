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
    print(f"Number of node_lines: {len(config['node_lines'])}")
    return config


def explain_structure(config: dict[str, Any]) -> None:
    """Walk through the hierarchical config structure."""
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
        ("bleu", "Generation", "N-gram overlap between generated and reference text"),
        ("meteor", "Generation", "Alignment-based similarity using synonyms and stemming"),
        ("rouge", "Generation", "Recall-oriented n-gram overlap with reference text"),
        ("sem_score", "Generation", "Cosine similarity of sentence embeddings (semantic similarity)"),
        ("g_eval", "Generation", "LLM-as-judge — uses a language model to score output quality"),
    ]

    # Print header
    print(f"  {'Metric':<25} {'Category':<12} {'Description'}")
    print(f"  {'-' * 25} {'-' * 12} {'-' * 55}")

    for name, category, description in metrics:
        print(f"  {name:<25} {category:<12} {description}")

    print(f"\n  Total: {len(metrics)} metrics available")
    print(f"  Retrieval metrics require retrieval_gt in the QA dataset")
    print(f"  Generation metrics require generation_gt in the QA dataset")


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
    print("Step 3: Node lines — pipeline stages")
    print("=" * 60)
    explain_node_lines(config)

    print("\n" + "=" * 60)
    print("Step 4: Count configurations to test")
    print("=" * 60)
    count_configurations(config)

    print("\n" + "=" * 60)
    print("Step 5: Strategy — greedy optimization")
    print("=" * 60)
    explain_strategy(config)

    print("\n" + "=" * 60)
    print("Step 6: Metrics reference")
    print("=" * 60)
    explain_metrics()

    print("\n" + "=" * 60)
    print("Step 7: Validate configuration")
    print("=" * 60)
    validate_config(config)

    print("\n" + "=" * 60)
    print("Done! You now understand AutoRAG's configuration format.")
    print("=" * 60)


if __name__ == "__main__":
    main()
