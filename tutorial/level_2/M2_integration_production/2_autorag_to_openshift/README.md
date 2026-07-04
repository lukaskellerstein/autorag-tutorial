# L2-M2.2 -- From AutoRAG to OpenShift AI

**Level:** Practitioner
**Duration:** 45 min

## Overview

AutoRAG finds the optimal RAG configuration through automated evaluation. But an optimized config sitting on your laptop is not a production system. This lesson bridges the gap: you will learn how to read AutoRAG's output, extract the key deployment parameters, map them to OpenShift AI components, and establish a continuous optimization workflow.

## Prerequisites

- Completed: L2-M1 (all lessons -- Intermediate Pipeline Nodes, Advanced Retrieval, Embedding Comparison, Custom Metrics)
- Completed: L2-M2.1 (Custom Modules)
- Familiarity with Kubernetes / OpenShift concepts (pods, services, routes)
- Understanding of model serving (vLLM, KServe)

## Concepts

### The Optimization-to-Deployment Gap

AutoRAG evaluates hundreds of configurations and tells you which combination of retrieval strategy, reranker, prompt template, and generator performs best on your data. The challenge is translating that finding into a deployed system.

Each AutoRAG component maps to one or more infrastructure decisions:

| AutoRAG Component | OpenShift AI Equivalent |
|-------------------|------------------------|
| `hybrid_rrf` retrieval | pgvector + Elasticsearch, fused via OGX |
| Embedding model | Model serving endpoint (KServe / vLLM) |
| `flashrank_reranker` | Sidecar service or OGX reranking API |
| Prompt template | OGX agent config or LangChain prompt |
| Generator (LLM) | vLLM on GPU node via KServe |

### Continuous Optimization

A production RAG system is not static. Documents change, user needs evolve, and better models become available. The lesson covers three strategies for keeping your pipeline optimal:

1. **Re-evaluation** -- re-run AutoRAG when the corpus changes significantly
2. **A/B testing** -- deploy new configurations alongside existing ones and compare real-world metrics
3. **Automated pipelines** -- use Kubeflow Pipelines to periodically re-optimize and alert when better configurations are found

### Feedback Loops

The strongest optimization signal comes from real users. Collecting thumbs-up/down feedback and corrections creates ground-truth data grounded in actual usage, which feeds back into AutoRAG's evaluation dataset.

## Step-by-Step

### Step 1: Read AutoRAG's Best Configuration

AutoRAG produces a `best.yaml` configuration after optimization. We create a representative config to work with, showing hybrid retrieval, FlashRank reranking, an fstring prompt template, and an Ollama-based generator.

### Step 2: Extract Deployment Parameters

We parse the configuration and extract the key parameters that drive deployment decisions: retrieval strategy, embedding model, top-K values, reranker, prompt template, and generator model.

### Step 3: Map Parameters to OpenShift AI

Each extracted parameter maps to specific OpenShift AI infrastructure. The lesson walks through each mapping with concrete deployment guidance.

### Step 4: Generate a Deployment Checklist

A structured checklist covers infrastructure provisioning, model serving, data pipeline setup, and application configuration. This serves as a practical guide for the deployment process.

### Step 5: Plan Continuous Optimization

We cover strategies for keeping the pipeline optimal over time: re-evaluation triggers, A/B testing patterns, and automated Kubeflow Pipelines for periodic re-optimization.

### Step 6: Visualize the Architecture

An end-to-end architecture diagram shows the flow from local AutoRAG optimization to OpenShift AI deployment, including all the infrastructure components and their connections.

## Running the Lesson

```bash
cd tutorial/level_2/M2_integration_production/2_autorag_to_openshift
uv sync
uv run python main.py
```

## Expected Output

```
============================================================
Step 1: AutoRAG Best Configuration
============================================================
...JSON config output...

============================================================
Step 2: Extracting Deployment Parameters
============================================================
  retrieval_strategy    hybrid_rrf
  retrieval_top_k       5
  reranker              flashrank_reranker
  reranker_top_k        3
  ...

============================================================
Step 3: Mapping to OpenShift AI Components
============================================================
...mapping details...

============================================================
Step 4: OpenShift AI Deployment Checklist
============================================================
...checklist...

============================================================
Step 6: End-to-End Architecture
============================================================
...ASCII diagram...

============================================================
Lesson complete!
============================================================
```

## Key Takeaways

- AutoRAG's optimized configuration is a blueprint, not a deployment -- you must map each component to infrastructure.
- Hybrid retrieval (e.g., `hybrid_rrf`) requires both a vector store and a BM25 index in production.
- Model serving on OpenShift AI uses KServe (InferenceService) for both the generator LLM and embedding models.
- Continuous optimization through re-evaluation, A/B testing, and automated pipelines prevents configuration drift.
- User feedback creates the strongest ground-truth signal for ongoing optimization.

## Next Steps

This is the final lesson in the tutorial. You now have the knowledge to:
- Evaluate and optimize RAG pipelines with AutoRAG (Level 1)
- Build custom metrics and modules (Level 2, M1-M2.1)
- Deploy optimized pipelines to OpenShift AI (this lesson)

Consider setting up an automated optimization pipeline as described in Step 5 to keep your production RAG system performing at its best.
