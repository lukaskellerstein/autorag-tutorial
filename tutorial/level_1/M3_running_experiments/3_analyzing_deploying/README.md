# L1-M3.3 -- Analyzing Results and Deploying the Optimal Pipeline

**Level:** Essentials
**Duration:** 30 min

## Overview

After running an AutoRAG evaluation trial, you need to understand the results and put the winning pipeline into action. This lesson teaches you how to read trial summaries, extract the optimal configuration, query the pipeline programmatically with Runner, and deploy it as a FastAPI server with ApiRunner. You will also learn how to interpret each evaluation metric so you can judge whether your pipeline is performing well.

## Prerequisites

- Completed L1-M3.2 (Running Your First Experiment) -- or this lesson will run its own trial automatically if no results exist
- Ollama running with the `gemma4:e2b` model pulled
- Python 3.10+

## Concepts

### Reading Trial Results

When AutoRAG completes an evaluation trial, it writes results into a numbered trial directory (e.g., `results/0/`). The most important file is `summary.csv`, which contains one row per pipeline node. Each row records the node type, the best-performing module for that node, and the metric scores that determined the winner. By reading this file, you can quickly see which retrieval strategy, prompt template, and generator configuration performed best across your QA dataset.

### Extracting the Optimal Configuration

AutoRAG provides `extract_best_config`, a utility that reads the trial results and produces a standalone YAML file containing only the winning module for each node. This extracted configuration is a deployable artifact -- you can version it, share it with teammates, or use it to recreate the optimal pipeline without rerunning the full evaluation. The output YAML has the same structure as the input config but with all non-winning alternatives removed.

### Runner: Programmatic Querying

The `Runner` class loads the optimal pipeline from a trial folder and lets you send queries to it from Python code. Under the hood, Runner reconstructs the retrieval, prompt-making, and generation stages using the best modules and parameters found during evaluation. You call `runner.run(query)` and get back the generated answer. This is useful for batch processing, integration testing, or embedding the pipeline into a larger application.

### ApiRunner: Deploying as a FastAPI Server

For production use, AutoRAG provides `ApiRunner`, which wraps the optimal pipeline in a FastAPI server. Once started, the server exposes REST endpoints:

| Endpoint | Method | Description |
|---|---|---|
| `/v1/run` | POST | Full pipeline: retrieve context and generate an answer |
| `/v1/retrieve` | POST | Retrieval only: return relevant passages without generation |
| `/v1/stream` | POST | Streaming generation: returns tokens as they are produced |
| `/version` | GET | Returns the API version |

This lets any HTTP client -- a web frontend, a chatbot, another microservice -- query your optimized RAG pipeline over the network.

### Metric Interpretation

Understanding what each metric measures helps you decide whether your pipeline is good enough or needs further tuning:

- **retrieval_f1** -- Harmonic mean of precision and recall for retrieved documents. Balances finding all relevant documents against avoiding irrelevant ones.
- **retrieval_recall** -- Fraction of relevant documents that were successfully retrieved. High recall means you are not missing important context.
- **retrieval_precision** -- Fraction of retrieved documents that are actually relevant. High precision means you are not flooding the prompt with noise.
- **bleu** -- Measures n-gram overlap between the generated answer and the reference answer. Scores above 0.3 are generally good for open-ended generation.
- **rouge** -- Recall-oriented n-gram overlap (ROUGE-L uses longest common subsequence). Scores above 0.4 suggest good coverage of the reference content.
- **meteor** -- Alignment-based metric that accounts for synonyms and stemming, giving a more flexible measure of answer quality.

## Step-by-Step

### Step 1: Check Ollama

The script verifies that Ollama is running and that the `gemma4:e2b` model is available. If Ollama is not reachable, the script exits with instructions to start it.

### Step 2: Prepare Data and Run Trial

If no trial results exist yet, the script generates the same 10-document corpus and 20 QA pairs used in L1-M3.2, then runs a full evaluation trial using `config.yaml`. This makes the lesson self-contained -- you do not need to have completed M3.2 first, though having prior results will skip this step.

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

The script loads `summary.csv` from the trial directory and prints the best module and metric scores for each pipeline node. This gives you a quick overview of which configurations won.

```python
summary_df = pd.read_csv(os.path.join(trial_path, "summary.csv"))
for _, row in summary_df.iterrows():
    print(f"Node: {row['node_type']}, Best: {row['best_module_name']}")
```

### Step 4: Extract Optimal Configuration

Using `extract_best_config`, the script pulls out the winning configuration and saves it as `best_config.yaml`. The extracted YAML is printed to the console so you can inspect it.

```python
from autorag.deploy import extract_best_config

config = extract_best_config(
    trial_path=trial_path,
    output_path="best_config.yaml",
)
```

### Step 5: Run Queries with Runner

The script creates a `Runner` from the trial folder and sends three test queries through the optimal pipeline. Each query goes through retrieval, prompt construction, and generation, producing a final answer.

```python
from autorag.deploy import Runner

runner = Runner.from_trial_folder(trial_path)
result = runner.run("What are Python decorators?")
print(result)
```

### Step 6: API Deployment (Explanation)

The script explains how to deploy the pipeline as a FastAPI server using `ApiRunner`. It prints the code you would use and lists the available endpoints. The server is not actually started in this lesson to keep things simple.

### Step 7: Metric Interpretation Guide

A reference table is printed explaining each metric, its range, and what constitutes a good score. This helps you interpret the numbers from Step 3 and decide whether to iterate further.

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

  Node: generator
  Best module: llama_index_llm
  Metrics:
    bleu: 0.2145
    rouge: 0.4532

Full summary table:
...

============================================================
  Step 4: Extract Optimal Configuration
============================================================

Extracted optimal configuration:
  Saved to: best_config.yaml

node_lines:
- node_line_name: retrieve_node_line
  nodes:
  - node_type: lexical_retrieval
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
  Step 6: API Deployment (Explanation)
============================================================

AutoRAG can deploy the optimal pipeline as a FastAPI server.
...

============================================================
  Step 7: Metric Interpretation Guide
============================================================

  Metric                    Range    Better     Description
  ------------------------- -------- ---------- ---------------------------------------------
  retrieval_f1              0-1      higher     Harmonic mean of precision and recall ...
  retrieval_recall          0-1      higher     Fraction of relevant documents ...
  ...

============================================================
  Done
============================================================

You have analyzed AutoRAG results and deployed the optimal pipeline.
You now know how to:
  - Read trial summaries to identify the best configuration
  - Extract the optimal config as a reusable YAML file
  - Run queries through the optimal pipeline using Runner
  - Deploy the pipeline as a FastAPI server using ApiRunner
  - Interpret evaluation metrics to assess quality

This completes Level 1, Module 3: Running Experiments.
```

Note: Exact metric values and generated answers will vary depending on model output. The structure and flow will match the above.

## Key Takeaways

- **summary.csv** is the primary output of an AutoRAG trial -- it tells you which module won at each pipeline stage and the metric scores that determined the winner.
- **extract_best_config** produces a standalone YAML with only the winning modules, ready for deployment or version control.
- **Runner** lets you query the optimal pipeline programmatically from Python, useful for batch processing or integration into larger applications.
- **ApiRunner** wraps the pipeline in a FastAPI server with REST endpoints, making it accessible to any HTTP client.
- Understanding metric ranges and thresholds (retrieval recall > 0.7, BLEU > 0.3, ROUGE > 0.4) helps you judge pipeline quality and decide when to iterate.

## Next Steps

This completes Level 1, Module 3: Running Experiments. You have learned how to prepare data, configure and run evaluations, and analyze and deploy the results.

In Level 2, you will explore advanced optimization strategies -- comparing more retrieval and generation modules side by side, tuning hyperparameters systematically, and building production-grade pipelines with monitoring and error handling.
