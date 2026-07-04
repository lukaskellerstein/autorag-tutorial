# L1-M3.3 -- Analyzing Results and Deploying the Optimal Pipeline

**Level:** Essentials
**Duration:** 45 min

## Overview

After running an AutoRAG evaluation trial, you need to understand the results and put the winning pipeline into action. This lesson teaches you how to read trial summaries, extract the optimal configuration, query the pipeline programmatically with Runner, interpret all evaluation metrics (including pass module results), and deploy using three options: FastAPI server, Gradio web interface, and Streamlit dashboard.

## Prerequisites

- Completed L1-M3.2 (Running Your First Experiment) -- or this lesson will run its own trial automatically if no results exist
- Ollama running with the `gemma4:e2b` model pulled
- Python 3.10+

## Concepts

### Reading Trial Results

When AutoRAG completes an evaluation trial, it writes results into a numbered trial directory (e.g., `results/0/`). The most important file is `summary.csv`, which contains one row per pipeline node. Each row records the node type, the best-performing module for that node, and the metric scores that determined the winner. By reading this file, you can quickly see which retrieval strategy, prompt template, and generator configuration performed best across your QA dataset.

### Extracting the Optimal Configuration

AutoRAG provides `extract_best_config`, a utility that reads the trial results and produces a standalone YAML file containing only the winning module for each node. This extracted configuration is a deployable artifact -- you can version it, share it with teammates, or use it to recreate the optimal pipeline without rerunning the full evaluation.

### Runner: Programmatic Querying

The `Runner` class loads the optimal pipeline from a trial folder and lets you send queries to it from Python code. Under the hood, Runner reconstructs the retrieval, prompt-making, and generation stages using the best modules and parameters found during evaluation. You call `runner.run(query)` and get back the generated answer.

### Metric Interpretation

Understanding what each metric measures helps you decide whether your pipeline is good enough or needs further tuning:

**Retrieval Metrics** (require retrieval_gt in QA dataset):
- **retrieval_f1** -- Harmonic mean of precision and recall for retrieved documents.
- **retrieval_recall** -- Fraction of relevant documents that were successfully retrieved.
- **retrieval_precision** -- Fraction of retrieved documents that are actually relevant.
- **retrieval_ndcg** -- Normalized discounted cumulative gain, rewarding relevant docs ranked higher.
- **retrieval_mrr** -- Mean reciprocal rank, measuring the position of the first relevant result.
- **retrieval_map** -- Mean average precision across all recall levels.

**Generation Metrics** (require generation_gt in QA dataset):
- **bleu** -- N-gram overlap between generated and reference answer. Scores above 0.3 are good for open-ended generation.
- **rouge** -- Recall-oriented n-gram overlap (ROUGE-L). Scores above 0.4 suggest good coverage.
- **meteor** -- Alignment-based metric accounting for synonyms and stemming.
- **sem_score** -- Cosine similarity of sentence embeddings (semantic similarity).
- **bert_score** -- Token-level semantic similarity using BERT embeddings.
- **faithfulness** -- Measures whether the answer is supported by the retrieved context.
- **g_eval** -- LLM-as-judge evaluation scoring coherence, consistency, fluency, and relevance (1-5 scale, above 3.5 is good).

**Passage Compressor Metrics:**
- **retrieval_token_f1** -- Token-level F1 between compressed and original passages.
- **retrieval_token_recall** -- Token-level recall of compressed passages.
- **retrieval_token_precision** -- Token-level precision of compressed passages.

**RAGAS Metrics:**
- **context_precision** -- Relevance of retrieved context. Does not require retrieval ground truth, making it useful when ground truth is unavailable.

### Understanding Pass Module Results

When a pass_* module wins at a node, it means skipping that processing step produces better results on your data. For example:

- **pass_reranker wins**: The initial retrieval ordering is already good enough. Reranking introduces errors or does not improve quality.
- **pass_compressor wins**: Full passage text produces better generation. Compression loses important details.
- **pass_passage_filter wins**: All retrieved passages are relevant enough. Filtering removes useful context.

Pass module wins simplify your production pipeline -- fewer components means less latency, lower cost, and easier maintenance.

### Deployment Options

AutoRAG provides three deployment methods:

| Option | Command | Purpose |
|---|---|---|
| **FastAPI Server** | `autorag run_api` | Production REST API |
| **Gradio Web Interface** | `autorag run_web` | Interactive testing UI |
| **Streamlit Dashboard** | `autorag dashboard` | Results visualization |

#### FastAPI Server (`autorag run_api`)

Wraps the optimal pipeline in a production-ready REST API:

| Endpoint | Method | Description |
|---|---|---|
| `/v1/run` | POST | Full pipeline: retrieve context and generate an answer |
| `/v1/retrieve` | POST | Retrieval only: return relevant passages without generation |
| `/v1/stream` | POST | Streaming generation: returns tokens via SSE |
| `/version` | GET | Returns the API version |

Supports NGrok tunnels for public access without port forwarding.

#### Gradio Web Interface (`autorag run_web`)

Provides an interactive chat-like UI for testing the pipeline. Supports shareable links for team collaboration. Ideal for demos and user testing before moving to production.

#### Streamlit Dashboard (`autorag dashboard`)

Visualizes evaluation results -- per-node module comparisons, metric distributions, and best pipeline visualization. This is for analyzing results, not for serving queries.

## Step-by-Step

### Step 1: Check Ollama

The script verifies that Ollama is running and that the `gemma4:e2b` model is available.

### Step 2: Prepare Data and Run Trial

If no trial results exist yet, the script generates a 10-document corpus and 20 QA pairs, then runs a full evaluation trial.

```python
from autorag.evaluator import Evaluator

evaluator = Evaluator(
    qa_data_path="data/qa.parquet",
    corpus_data_path="data/corpus.parquet",
    project_dir="./results",
)
evaluator.start_trial("config.yaml")
```

### Step 3: Analyze Results

The script loads `summary.csv` from the trial directory and prints the best module and metric scores for each pipeline node.

```python
summary_df = pd.read_csv(os.path.join(trial_path, "summary.csv"))
for _, row in summary_df.iterrows():
    print(f"Node: {row['node_type']}, Best: {row['best_module_name']}")
```

### Step 4: Extract Optimal Configuration

Using `extract_best_config`, the script pulls out the winning configuration and saves it as `best_config.yaml`.

```python
from autorag.deploy import extract_best_config

config = extract_best_config(
    trial_path=trial_path,
    output_path="best_config.yaml",
)
```

### Step 5: Run Queries with Runner

The script creates a `Runner` from the trial folder and sends three test queries through the optimal pipeline.

```python
from autorag.deploy import Runner

runner = Runner.from_trial_folder(trial_path)
result = runner.run("What are Python decorators?")
```

### Step 6: Metric Interpretation Guide

A reference table is printed explaining all 17 metrics, their ranges, and what constitutes a good score.

### Step 7: Pass Module Results

Explains how to interpret pass module wins and their implications for pipeline simplification.

### Step 8: FastAPI Deployment

Shows how to deploy using `autorag run_api` or the `ApiRunner` Python class. Lists all REST endpoints and mentions NGrok tunnel support.

### Step 9: Gradio Web Interface

Shows how to deploy using `autorag run_web` or the `GradioRunner` Python class for interactive testing.

### Step 10: Streamlit Dashboard

Shows how to visualize results using `autorag dashboard` for detailed metric analysis.

## Running the Lesson

```bash
cd tutorial/level_1/M3_running_experiments/3_analyzing_deploying
uv sync
uv run python main.py
```

## Expected Output

```
============================================================
  L1-M3.3 -- Analyzing Results and Deploying
============================================================

============================================================
  Step 1: Check Ollama
============================================================

Checking Ollama at http://localhost:11434 ...
Ollama is running. Available models: ['gemma4:e2b', ...]

============================================================
  Step 2: Prepare Data and Run Trial
============================================================

Trial results already exist at ./results/0

============================================================
  Step 3: Analyze Results
============================================================

Pipeline Optimization Results
------------------------------------------------------------

  Node: lexical_retrieval
  Best module: bm25
  Metrics:
    retrieval_f1: 0.8500
    retrieval_recall: 0.9000

  Node: prompt_maker
  Best module: fstring
  Metrics:
    bleu: 0.2145
    rouge: 0.4532
  ...

============================================================
  Step 4: Extract Optimal Configuration
============================================================

Extracted optimal configuration:
  Saved to: best_config.yaml
...

============================================================
  Step 5: Run Queries with Runner
============================================================

Loading optimal pipeline from trial results...
Running test queries through the optimal pipeline:

Q: What are Python decorators?
A: Decorators are a powerful way to modify the behavior of functions ...

Q: How do you handle errors in Python?
A: Error handling in Python uses try-except blocks to catch ...

Q: What is a virtual environment?
A: A virtual environment creates an isolated space for project ...

============================================================
  Step 6: Metric Interpretation Guide
============================================================

  Metric                       Range    Better     Description
  ---------------------------- -------- ---------- ------------------------------------------------
  retrieval_f1                 0-1      higher     Harmonic mean of precision and recall ...
  retrieval_recall             0-1      higher     Fraction of relevant documents ...
  ...
  context_precision            0-1      higher     RAGAS: context relevance (no ground truth needed)

  Total: 17 metrics available

============================================================
  Step 7: Pass Module Results
============================================================

When a pass_* module wins at a node, it means skipping that
processing step produces better results on your data.

  If pass_query_expansion wins:
    Your queries are already clear and specific.
  ...

============================================================
  Step 8: FastAPI Deployment
============================================================

AutoRAG can deploy the optimal pipeline as a FastAPI server.

CLI command:
  autorag run_api --trial_dir ./results/0 --host 0.0.0.0 --port 8000

API Endpoints:
  POST /v1/run              Full pipeline (retrieve + generate)
  POST /v1/retrieve         Retrieval only (no generation)
  POST /v1/stream           Streaming generation via SSE
  GET /version              API version

============================================================
  Step 9: Gradio Web Interface
============================================================

AutoRAG provides a Gradio web interface for interactive testing.

CLI command:
  autorag run_web --trial_dir ./results/0
...

============================================================
  Step 10: Streamlit Dashboard
============================================================

AutoRAG provides a Streamlit dashboard for visualizing trial results.

CLI command:
  autorag dashboard --trial_dir ./results/0
...

============================================================
  Done
============================================================

You have analyzed AutoRAG results and explored deployment options.
You now know how to:
  - Read trial summaries to identify the best configuration
  - Extract the optimal config as a reusable YAML file
  - Run queries through the optimal pipeline using Runner
  - Interpret evaluation metrics across all categories
  - Understand pass module results and their implications
  - Deploy via FastAPI (autorag run_api) for production APIs
  - Deploy via Gradio (autorag run_web) for interactive testing
  - Visualize results via Streamlit (autorag dashboard)

This completes Level 1, Module 3: Running Experiments.
```

Note: Exact metric values and generated answers will vary depending on model output. The structure and flow will match the above.

## Key Takeaways

- **summary.csv** is the primary output of an AutoRAG trial -- it tells you which module won at each pipeline stage.
- **extract_best_config** produces a standalone YAML with only the winning modules, ready for deployment.
- **Runner** lets you query the optimal pipeline programmatically from Python.
- **17 metrics** are available: 6 retrieval, 7 generation, 3 compressor, and 1 RAGAS metric.
- **Pass module wins** indicate that skipping a node is optimal -- simplifying your production pipeline.
- **Three deployment options**: FastAPI server (`autorag run_api`) for production APIs, Gradio (`autorag run_web`) for interactive testing, and Streamlit dashboard (`autorag dashboard`) for results visualization.

## Next Steps

This completes Level 1, Module 3: Running Experiments. You have learned how to prepare data, configure and run evaluations, and analyze and deploy the results.

In Level 2, you will explore advanced optimization strategies -- intermediate pipeline nodes, hybrid retrieval, embedding model comparisons, custom metrics, and integration with OpenShift AI.
