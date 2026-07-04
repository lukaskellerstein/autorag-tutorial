# L1-M3.1 — Configuration YAML

**Level:** Essentials
**Duration:** 45 min

## Overview

The configuration YAML is the heart of every AutoRAG experiment. It defines which retrieval, reranking, prompting, and generation modules to test, what metrics to optimize for, and how AutoRAG selects the best configuration at each stage. This lesson walks through the YAML structure, explains the greedy optimization strategy, and teaches you to read and write AutoRAG configs.

## Prerequisites

- Completed: L1-M2.2 (Preparing Corpus Data)
- Python 3.10+ with `uv` installed
- No infrastructure required -- this lesson works offline

## Concepts

### The Configuration YAML

AutoRAG experiments are driven by a single YAML file that describes the entire RAG pipeline as a series of **node lines**, each containing **nodes**, each containing **modules**. AutoRAG reads this config, runs every module configuration, measures performance using your QA dataset, and selects the best option at each node.

### Node Lines

Node lines represent high-level stages of the pipeline. They execute sequentially -- the output of one node line feeds into the next. A typical RAG pipeline has two node lines:

1. **retrieve_node_line** -- Retrieves and reranks relevant passages from the corpus.
2. **post_retrieve_node_line** -- Formats prompts and generates answers from retrieved passages.

### Nodes

Each node line contains one or more nodes. A node represents a specific processing step (e.g., `lexical_retrieval`, `passage_reranker`, `prompt_maker`, `generator`). Nodes within a node line also execute sequentially.

### Modules

Modules are the implementations that AutoRAG compares. For example, the `passage_reranker` node might test `pass_reranker` (no reranking) against `flashrank_reranker`. AutoRAG runs both and picks the winner based on strategy metrics.

### Strategy and Greedy Optimization

Each node has a `strategy` section that defines which metrics to optimize. AutoRAG uses a **greedy algorithm**:

1. Run all module configurations at the current node.
2. Score each using the strategy metrics.
3. Select the best-performing configuration.
4. Pass its output to the next node.
5. Repeat.

This is greedy because each node is optimized independently -- it does not search for the globally optimal combination across all nodes.

### Parameter Grid Search

When a module parameter is a list, AutoRAG generates all combinations. For example, if `prompt` has 2 templates and `model` has 2 models, AutoRAG tests all 4 combinations (2 x 2).

## The Configuration File

Here is the full `config.yaml` used in this lesson:

```yaml
node_lines:
- node_line_name: retrieve_node_line
  nodes:
    - node_type: lexical_retrieval
      strategy:
        metrics: [retrieval_f1, retrieval_recall, retrieval_precision]
      top_k: 3
      modules:
        - module_type: bm25
    - node_type: passage_reranker
      strategy:
        metrics: [retrieval_f1, retrieval_recall]
      top_k: 3
      modules:
        - module_type: pass_reranker
        - module_type: flashrank_reranker

- node_line_name: post_retrieve_node_line
  nodes:
    - node_type: prompt_maker
      strategy:
        metrics: [bleu, rouge]
        generator_modules:
          - module_type: llama_index_llm
            llm: ollama
            model: gemma4:e2b
      modules:
        - module_type: fstring
          prompt:
            - "Read the passages and answer the question.\nQuestion: {query}\nPassage: {retrieved_contents}\nAnswer:"
            - "Based on the following context, answer the question.\nContext: {retrieved_contents}\nQuestion: {query}\nAnswer:"
    - node_type: generator
      strategy:
        metrics:
          - metric_name: bleu
          - metric_name: rouge
      modules:
        - module_type: llama_index_llm
          llm: ollama
          model: [gemma4:e2b]
```

### Reading the Config

- **retrieve_node_line**: First does `lexical_retrieval` with BM25, then `passage_reranker` comparing pass-through vs. FlashRank.
- **post_retrieve_node_line**: Tests 2 prompt templates in `prompt_maker`, then runs `generator` with the LLM.
- **Strategy metrics**: Retrieval nodes optimize for F1/recall/precision. Generation nodes optimize for BLEU/ROUGE.

## Step-by-Step

### Step 1: Load Configuration

Load the YAML file and inspect the top-level structure:

```python
with open("config.yaml") as f:
    config = yaml.safe_load(f)
```

### Step 2: Understand the Hierarchy

Walk the 4-level hierarchy: node_lines -> nodes -> strategy + modules.

### Step 3: Trace Node Lines

Follow how data flows through the pipeline stages.

### Step 4: Count Configurations

Calculate how many module configurations AutoRAG will test at each node. List parameters create combinatorial expansion.

### Step 5: Understand Strategy

Learn how the greedy algorithm selects the best module at each node.

### Step 6: Review Available Metrics

See all retrieval and generation metrics with descriptions.

### Step 7: Validate the Config

Check that every node_line has a name, every node has a type and modules, and every module has a type.

## Running the Lesson

```bash
cd tutorial/level_1/M3_running_experiments/1_configuration_yaml
uv sync
uv run python main.py
```

## Expected Output

```
============================================================
L1-M3.1 — Configuration YAML
============================================================

============================================================
Step 1: Load configuration
============================================================
Loaded config from: config.yaml
Top-level keys: ['node_lines']
Number of node_lines: 2

============================================================
Step 2: Understand the structure
============================================================
AutoRAG config follows a 4-level hierarchy:

  node_lines          (top level -- pipeline stages)
    -> nodes          (processing steps within a stage)
      -> strategy     (how to evaluate and select the best)
      -> modules      (implementations to compare)

  node_line [0]: retrieve_node_line
    node [0]: lexical_retrieval
      strategy metrics: ['retrieval_f1', 'retrieval_recall', 'retrieval_precision']
      modules (1):
        - bm25
    node [1]: passage_reranker
      strategy metrics: ['retrieval_f1', 'retrieval_recall']
      modules (2):
        - pass_reranker
        - flashrank_reranker

  node_line [1]: post_retrieve_node_line
    node [0]: prompt_maker
      strategy metrics: ['bleu', 'rouge']
      modules (1):
        - fstring
    node [1]: generator
      strategy metrics: ['bleu', 'rouge']
      modules (1):
        - llama_index_llm

============================================================
Step 3: Node lines -- pipeline stages
============================================================
Node lines define the high-level stages of the RAG pipeline.
They execute sequentially: the output of one feeds into the next.

  Stage 1: retrieve_node_line
    Nodes: lexical_retrieval -> passage_reranker
    Purpose: Find and rank relevant passages from the corpus

  Stage 2: post_retrieve_node_line
    Nodes: prompt_maker -> generator
    Purpose: Format prompts and generate answers from retrieved passages

Pipeline flow:
  lexical_retrieval -> passage_reranker -> prompt_maker -> generator

============================================================
Step 4: Count configurations to test
============================================================
AutoRAG tests all module configurations at each node,
then selects the best before moving to the next node.

  retrieve_node_line:
    bm25: 1 config(s)
    -> lexical_retrieval total: 1 configuration(s)
    pass_reranker: 1 config(s)
    flashrank_reranker: 1 config(s)
    -> passage_reranker total: 2 configuration(s)

  post_retrieve_node_line:
    fstring: 2 config(s) (prompt=2 values)
    -> prompt_maker total: 2 configuration(s)
    llama_index_llm: 1 config(s) (model=1 values)
    -> generator total: 1 configuration(s)

  Grand total: 6 module configurations to evaluate

============================================================
Step 5: Strategy -- greedy optimization
============================================================
The 'strategy' section in each node controls optimization:

  1. METRICS -- What to measure
     ...

  2. GREEDY SELECTION -- How the best is chosen
     ...

Strategy sections found in this config:
  lexical_retrieval: ['retrieval_f1', 'retrieval_recall', 'retrieval_precision']
  passage_reranker: ['retrieval_f1', 'retrieval_recall']
  prompt_maker: ['bleu', 'rouge']
  generator: ['bleu', 'rouge']

============================================================
Step 6: Metrics reference
============================================================
  Metric                    Category     Description
  ------------------------- ------------ -------------------------------------------------------
  retrieval_f1              Retrieval    Harmonic mean of retrieval precision and recall
  retrieval_recall          Retrieval    Fraction of relevant documents that were retrieved
  retrieval_precision       Retrieval    Fraction of retrieved documents that are relevant
  retrieval_ndcg            Retrieval    Normalized discounted cumulative gain ...
  retrieval_mrr             Retrieval    Mean reciprocal rank ...
  bleu                      Generation   N-gram overlap between generated and reference text
  meteor                    Generation   Alignment-based similarity using synonyms and stemming
  rouge                     Generation   Recall-oriented n-gram overlap with reference text
  sem_score                 Generation   Cosine similarity of sentence embeddings
  g_eval                    Generation   LLM-as-judge evaluation

  Total: 10 metrics available

============================================================
Step 7: Validate configuration
============================================================
Validation PASSED -- all required keys present

  Config summary:
    Node lines: 2
    Total nodes: 4
    Total modules: 5

============================================================
Done! You now understand AutoRAG's configuration format.
============================================================
```

## Key Takeaways

- AutoRAG configs follow a 4-level hierarchy: **node_lines -> nodes -> strategy -> modules**.
- **Node lines** are pipeline stages that execute sequentially (retrieve, then post-retrieve).
- **Nodes** are processing steps within a stage (e.g., retrieval, reranking, prompting, generation).
- **Modules** are implementations to compare (e.g., BM25 vs. vector search, different prompt templates).
- The **strategy** section defines metrics for the greedy optimizer to select the best module at each node.
- List parameters create a **parameter grid** -- AutoRAG tests all combinations.

## Next Steps

In **L1-M3.2 -- Running and Monitoring Experiments**, you will use this configuration to run an actual AutoRAG experiment, monitor progress, and interpret the results.
