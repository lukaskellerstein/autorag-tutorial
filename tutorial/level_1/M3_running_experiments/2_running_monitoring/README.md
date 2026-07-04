# L1-M3.2 -- Running and Monitoring Evaluations

**Level:** Essentials
**Duration:** 45 min

## Overview

This lesson teaches you how to run an AutoRAG evaluation trial end-to-end, monitor its progress through console logs, and understand the results directory structure that AutoRAG produces. By the end, you will know how to launch an evaluation, track what is happening at each stage, and locate the outputs that matter.

## Prerequisites

- Completed: L1-M3.1 (Configuration YAML)
- Ollama running locally with the `gemma4:e2b` model pulled
  ```bash
  ollama serve
  ollama pull gemma4:e2b
  ```

## Concepts

### The Evaluator API

AutoRAG provides an `Evaluator` class that orchestrates the entire evaluation pipeline. You initialize it with three pieces of information:

1. **qa_data_path** -- path to your QA dataset (Parquet file with queries, retrieval ground truth, and generation ground truth).
2. **corpus_data_path** -- path to your corpus (Parquet file with document IDs and contents).
3. **project_dir** -- directory where AutoRAG writes all results.

Once configured, calling `evaluator.start_trial("config.yaml")` kicks off the evaluation using the pipeline defined in your YAML configuration.

### Node-by-Node Greedy Optimization

AutoRAG evaluates each node in your pipeline sequentially. For each node, it:

1. Runs every module listed under that node with every parameter combination.
2. Scores each module using the metrics specified in the `strategy` section.
3. Selects the best-performing module based on the primary metric.
4. Passes the best module's output forward to the next node.

This is a **greedy** approach -- AutoRAG optimizes one node at a time rather than searching the full combinatorial space. This makes evaluation tractable even with many modules and parameters.

### Monitoring Progress

When you run an evaluation, AutoRAG logs progress directly to the console. You will see:

- Which node line and node are currently being evaluated.
- Which modules are being tested.
- Metric scores as they are computed.
- Timing information for each stage.

This real-time feedback lets you estimate how long the full evaluation will take and catch configuration errors early.

### The Results Directory Structure

After an evaluation completes, AutoRAG creates a structured output directory:

```
results/
  0/                            # Trial directory (numbered sequentially)
    config.yaml                 # Copy of the config used for this trial
    summary.csv                 # Per-node summary with best module and scores
    retrieve_node_line/         # One directory per node line
      lexical_retrieval/        # One directory per node
        0/                      # Module run directories
          ...                   # Detailed per-query results (Parquet files)
    post_retrieve_node_line/
      prompt_maker/
        ...
      generator/
        ...
```

Key files to look at:

- **summary.csv** -- the most important output. Shows which module won at each node and its metric scores.
- **config.yaml** -- a copy of the configuration, useful for reproducibility.
- **Per-node Parquet files** -- detailed per-query results for deeper analysis (covered in the next lesson).

## Step-by-Step

### Step 1: Check Ollama

Before running the evaluation, we verify that Ollama is running and that the `gemma4:e2b` model is available. The script queries the Ollama API at `http://localhost:11434/api/tags` and prints the list of available models. If Ollama is not reachable, the script exits with instructions.

```python
def check_ollama() -> None:
    response = httpx.get("http://localhost:11434/api/tags", timeout=5.0)
    models = response.json().get("models", [])
    model_names = [m["name"] for m in models]
    print(f"Ollama is running. Available models: {model_names}")
```

### Step 2: Prepare Evaluation Data

The script generates sample data if it does not already exist: 10 corpus documents about Python programming concepts and 20 QA pairs (2 per document). Each QA pair includes a query, retrieval ground truth (which document should be retrieved), and generation ground truth (the expected answer).

The data is saved as Parquet files in a `data/` directory:

```python
corpus_df.to_parquet("data/corpus.parquet", index=False)
qa_df.to_parquet("data/qa.parquet", index=False)
```

### Step 3: Run AutoRAG Evaluation

This is the core of the lesson. We create an `Evaluator` instance and start a trial:

```python
from autorag.evaluator import Evaluator

evaluator = Evaluator(
    qa_data_path="data/qa.parquet",
    corpus_data_path="data/corpus.parquet",
    project_dir="./results",
)
evaluator.start_trial("config.yaml")
```

AutoRAG will:
1. Build a BM25 index from the corpus.
2. Run lexical retrieval for all 20 queries.
3. Score retrieval using F1 and recall metrics.
4. Generate prompts using the fstring template.
5. Call Ollama (gemma4:e2b) to generate answers.
6. Score generation using BLEU and ROUGE metrics.
7. Select the best module at each node and write the results.

Watch the console output -- AutoRAG logs each stage as it runs.

### Step 4: Inspect Results Directory

After the evaluation completes, the script walks the `results/` directory and prints every file with its size. This helps you understand what AutoRAG produced and where to find specific outputs.

```python
for root, dirs, files in os.walk("./results"):
    level = root.replace("./results", "").count(os.sep)
    indent = "  " * level
    basename = os.path.basename(root) or "results/"
    print(f"{indent}{basename}/")
```

### Step 5: Trial Summary

Finally, the script loads and prints `summary.csv`, which contains the most important information: which module performed best at each node, and what metric scores it achieved.

```python
summary_df = pd.read_csv(summary_path)
for _, row in summary_df.iterrows():
    print(f"  Node: {row.get('node_type', 'N/A')}")
    print(f"  Best module: {row.get('best_module_name', 'N/A')}")
```

## Running the Lesson

```bash
cd tutorial/level_1/M3_running_experiments/2_running_monitoring
uv sync
uv run python main.py
```

## Expected Output

```
============================================================
  L1-M3.2 -- Running and Monitoring Evaluations
============================================================

============================================================
  Step 1: Check Ollama
============================================================

Checking Ollama at http://localhost:11434 ...
Ollama is running. Available models: ['gemma4:e2b']

============================================================
  Step 2: Prepare Evaluation Data
============================================================

Created data/corpus.parquet: 10 documents
Created data/qa.parquet:     20 QA pairs

QA dataset shape:     (20, 4)
QA columns:           ['qid', 'query', 'retrieval_gt', 'generation_gt']
Corpus dataset shape: (10, 3)
Corpus columns:       ['doc_id', 'contents', 'metadata']

============================================================
  Step 3: Run AutoRAG Evaluation
============================================================

Starting evaluation...
This will run BM25 retrieval + prompt making + generation.
Progress is logged to the console by AutoRAG.

  ... (AutoRAG progress logs appear here) ...

Evaluation complete! Took 42.3 seconds.

============================================================
  Step 4: Inspect Results Directory
============================================================

Results directory structure:

results/
  0/
    config.yaml  (1.2 KB)
    summary.csv  (456 B)
    retrieve_node_line/
      lexical_retrieval/
        0/
          ...
    post_retrieve_node_line/
      prompt_maker/
        ...
      generator/
        ...

============================================================
  Step 5: Trial Summary
============================================================

Loading summary from: ./results/0/summary.csv

Trial Summary:
------------------------------------------------------------
  Node: lexical_retrieval
  Best module: bm25
    retrieval_f1: 0.85
    retrieval_recall: 0.90

  Node: prompt_maker
  Best module: fstring
    bleu: 0.12
    rouge: 0.45

  Node: generator
  Best module: llama_index_llm
    bleu: 0.12
    rouge: 0.45

Full summary table:
  node_type  best_module_name  ...

============================================================
  Done
============================================================

You have successfully run an AutoRAG evaluation trial.
Next lesson: L1-M3.3 -- Analyzing Results and Deploying.
```

Note: Exact metric values and timing will vary depending on your hardware, model version, and Ollama performance. The structure and flow will be the same.

## Key Takeaways

- The **Evaluator** class is the main entry point for running AutoRAG evaluations. You provide it with QA data, corpus data, a project directory, and a config YAML.
- AutoRAG uses **greedy node-by-node optimization** -- it evaluates all modules at each node, picks the best, and passes its output to the next node.
- The **results directory** is structured hierarchically: trial directories contain node line directories, which contain node directories, which contain per-module results.
- The **summary.csv** file is the most important output -- it tells you which module won at each node and what scores it achieved.
- **Console logs** provide real-time monitoring of the evaluation progress, making it easy to track what is happening and estimate completion time.

## Next Steps

In **L1-M3.3 -- Analyzing Results and Deploying**, you will learn how to dig deeper into the per-query results, compare module performance in detail, and deploy the best pipeline configuration as a serving endpoint.
