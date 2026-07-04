"""
L2-M2.2 — From AutoRAG to OpenShift AI

Learn how to take AutoRAG's optimized configuration and deploy
the resulting RAG pipeline on OpenShift AI.
"""

import json
from typing import Any


# ---------------------------------------------------------------------------
# 1. Create a sample best config
# ---------------------------------------------------------------------------

def create_sample_best_config() -> dict[str, Any]:
    print("=" * 60)
    print("Step 1: AutoRAG Best Configuration")
    print("=" * 60)

    best_config: dict[str, Any] = {
        "node_lines": [
            {
                "node_line_name": "retrieve_node_line",
                "nodes": [
                    {
                        "node_type": "hybrid_retrieval",
                        "strategy": {"metrics": ["retrieval_f1"]},
                        "top_k": 5,
                        "modules": [{
                            "module_type": "hybrid_rrf",
                            "weight_range": "(4, 80)",
                        }],
                    },
                    {
                        "node_type": "passage_reranker",
                        "strategy": {"metrics": ["retrieval_f1"]},
                        "top_k": 3,
                        "modules": [{
                            "module_type": "flashrank_reranker",
                        }],
                    },
                ],
            },
            {
                "node_line_name": "post_retrieve_node_line",
                "nodes": [
                    {
                        "node_type": "prompt_maker",
                        "modules": [{
                            "module_type": "fstring",
                            "prompt": (
                                "Answer based on context.\n"
                                "Context: {retrieved_contents}\n"
                                "Question: {query}\n"
                                "Answer:"
                            ),
                        }],
                    },
                    {
                        "node_type": "generator",
                        "modules": [{
                            "module_type": "llama_index_llm",
                            "llm": "ollama",
                            "model": "gemma4:e2b",
                        }],
                    },
                ],
            },
        ],
    }

    print("\nThis configuration was produced by AutoRAG's optimization")
    print("pipeline.  It represents the best-performing combination")
    print("of modules, strategies, and hyperparameters.\n")
    print(json.dumps(best_config, indent=2))
    print()
    return best_config


# ---------------------------------------------------------------------------
# 2. Extract deployment parameters
# ---------------------------------------------------------------------------

def extract_deployment_params(config: dict[str, Any]) -> dict[str, Any]:
    print("=" * 60)
    print("Step 2: Extracting Deployment Parameters")
    print("=" * 60)

    params: dict[str, Any] = {}

    for node_line in config["node_lines"]:
        for node in node_line["nodes"]:
            ntype = node["node_type"]
            module = node["modules"][0]

            if ntype == "hybrid_retrieval":
                params["retrieval_strategy"] = module["module_type"]
                params["retrieval_top_k"] = node["top_k"]
            elif ntype == "passage_reranker":
                params["reranker"] = module["module_type"]
                params["reranker_top_k"] = node["top_k"]
            elif ntype == "prompt_maker":
                params["prompt_template"] = module["prompt"]
            elif ntype == "generator":
                params["generator_llm"] = module.get("llm", "unknown")
                params["generator_model"] = module.get("model", "unknown")

    params["embedding_model"] = "BAAI/bge-small-en-v1.5"

    print()
    for key, value in params.items():
        display = value if len(str(value)) < 60 else str(value)[:57] + "..."
        print(f"  {key:<25} {display}")
    print()
    return params


# ---------------------------------------------------------------------------
# 3. Map parameters to OpenShift AI config
# ---------------------------------------------------------------------------

def map_to_openshift_config(params: dict[str, Any]) -> None:
    print("=" * 60)
    print("Step 3: Mapping to OpenShift AI Components")
    print("=" * 60)
    print(f"""
AutoRAG Result              OpenShift AI Deployment
{"=" * 55}

1. RETRIEVAL STRATEGY: {params['retrieval_strategy']}
   Deploy both a BM25 index (Elasticsearch / OpenSearch)
   AND a vector store (pgvector / Milvus).
   Configure fusion in OGX or a LangChain retriever.

2. EMBEDDING MODEL: {params['embedding_model']}
   Deploy as a vLLM / HuggingFace model-serving endpoint.
   Or use the OGX embedding API with the same model.

3. TOP-K RETRIEVAL: {params['retrieval_top_k']}
   Set in the retriever service configuration.

4. RERANKER: {params['reranker']}
   Deploy FlashRank as a sidecar service.
   Or use the OGX reranking API.

5. TOP-K AFTER RERANKING: {params['reranker_top_k']}
   Configure in the reranker service.

6. PROMPT TEMPLATE:
   "{params['prompt_template'][:50]}..."
   Set in the OGX agent configuration or LangChain prompt.

7. GENERATOR: {params['generator_llm']}/{params['generator_model']}
   Deploy via vLLM on a GPU node.
   Configure OGX to route to the vLLM endpoint.
""")


# ---------------------------------------------------------------------------
# 4. Generate a deployment checklist
# ---------------------------------------------------------------------------

def generate_deployment_checklist(params: dict[str, Any]) -> None:
    print("=" * 60)
    print("Step 4: OpenShift AI Deployment Checklist")
    print("=" * 60)
    print(f"""
OpenShift AI RAG Deployment Checklist
{"=" * 42}

Infrastructure:
[ ] GPU node available for vLLM inference
[ ] Persistent storage for vector database
[ ] Persistent storage for BM25 index

Model Serving:
[ ] Deploy {params['generator_model']} via vLLM (InferenceService)
[ ] Deploy {params['embedding_model']} for embeddings

Data Pipeline (Kubeflow Pipelines):
[ ] Document ingestion pipeline
[ ] Chunking: use token chunker (validated by AutoRAG)
[ ] Embedding: {params['embedding_model']}
[ ] Index: pgvector/Milvus + Elasticsearch

Retrieval:
[ ] Strategy: {params['retrieval_strategy']} (BM25 + vector)
[ ] Top-K retrieval: {params['retrieval_top_k']}
[ ] Reranker: {params['reranker']}
[ ] Top-K after reranking: {params['reranker_top_k']}

Application:
[ ] OGX server configured with optimal parameters
[ ] Prompt template deployed
[ ] Health checks and readiness probes
[ ] Monitoring and logging enabled
""")


# ---------------------------------------------------------------------------
# 5. Explain continuous optimization
# ---------------------------------------------------------------------------

def explain_continuous_optimization() -> None:
    print("=" * 60)
    print("Step 5: Continuous Optimization Workflow")
    print("=" * 60)
    print("""
Deploying a RAG pipeline is not a one-time event.  As your
document corpus evolves, the optimal configuration may shift.

1. Data Changes -> Re-run AutoRAG
   When your document corpus changes significantly, re-run
   AutoRAG to verify the optimal configuration still holds.
   New document types or domains can favor different chunking,
   retrieval, and reranking strategies.

2. A/B Testing
   Deploy the new AutoRAG-optimized config alongside the
   existing one.  Route a percentage of traffic to each and
   compare real-world metrics (latency, user satisfaction,
   answer accuracy).

3. Automated Pipelines
   Set up a Kubeflow Pipeline that periodically:
   - Generates new QA evaluation data from updated documents
   - Runs AutoRAG optimization
   - Compares results with the current production config
   - Alerts if a significantly better config is found

4. Feedback Loop
   Collect user feedback (thumbs up/down, corrections) and
   feed it back into the evaluation dataset.  This creates
   a ground-truth signal grounded in real usage.
""")


# ---------------------------------------------------------------------------
# 6. Print architecture diagram
# ---------------------------------------------------------------------------

def print_architecture_diagram() -> None:
    print("=" * 60)
    print("Step 6: End-to-End Architecture")
    print("=" * 60)
    print("""
AutoRAG -> OpenShift AI Workflow

+---------------------------------------------+
|  LOCAL DEVELOPMENT                          |
|                                             |
|  Documents -> AutoRAG Evaluation            |
|     |           |                           |
|  qa.parquet   config.yaml                   |
|     |           |                           |
|  autorag evaluate                           |
|     |                                       |
|  Optimal Config (best.yaml)                 |
+----------------------+----------------------+
                       |
                       v
+---------------------------------------------+
|  OPENSHIFT AI                               |
|                                             |
|  +-------+  +----------+  +-------------+  |
|  | vLLM  |  | pgvector |  |Elasticsearch|  |
|  |gemma4 |  | (vectors)|  | (BM25 index)|  |
|  +---+---+  +----+-----+  +------+------+  |
|      |           |               |          |
|      +-----------+-------+-------+          |
|                          |                  |
|                    +-----v-----+            |
|                    |    OGX    |            |
|                    |  Server   |            |
|                    +-----+-----+            |
|                          |                  |
|                    +-----v-----+            |
|                    |  Your App |            |
|                    +-----------+            |
+---------------------------------------------+

Key OpenShift AI components used:
  - Model Serving (KServe) -- hosts vLLM and embedding models
  - Data Science Pipelines  -- Kubeflow-based data ingestion
  - Workbenches            -- Jupyter for experimentation
  - Model Registry         -- version and track models
""")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    config = create_sample_best_config()
    params = extract_deployment_params(config)
    map_to_openshift_config(params)
    generate_deployment_checklist(params)
    explain_continuous_optimization()
    print_architecture_diagram()

    print("=" * 60)
    print("Lesson complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
