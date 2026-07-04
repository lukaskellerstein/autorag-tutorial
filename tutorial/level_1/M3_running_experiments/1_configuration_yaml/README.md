# L1-M3.1 — Configuration YAML

**Level:** Essentials
**Duration:** 1 hour

## Overview

The configuration YAML is the heart of every AutoRAG experiment. It defines the vector database backend, which retrieval and generation modules to test across all 8 pipeline node types, what metrics to optimize for, and how AutoRAG selects the best configuration at each stage. This lesson walks through the YAML structure, explains pass modules, optimization strategies, and teaches you to read and write AutoRAG configs.

## Prerequisites

- Completed: L1-M2.2 (Creating QA Datasets)
- Python 3.10+ with `uv` installed
- No infrastructure required -- this lesson works offline

## Concepts

### The Configuration YAML

AutoRAG experiments are driven by a single YAML file that describes the entire RAG pipeline. The file has two top-level sections: `vectordb` (vector database configuration) and `node_lines` (pipeline stages containing nodes and modules). AutoRAG reads this config, runs every module configuration, measures performance using your QA dataset, and selects the best option at each node.

### Vector Database Configuration

The top-level `vectordb` section configures the vector database backend used by retrieval nodes. This is separate from node_lines because it defines shared infrastructure:

```yaml
vectordb:
  - name: default
    db_type: chroma
    client_type: persistent
    path: ./chroma_db
    embedding_model: openai
    collection_name: openai
```

Supported backends include Chroma, Milvus, Weaviate, Pinecone, Couchbase, and Qdrant.

### Node Lines

Node lines represent high-level stages of the pipeline. They execute sequentially -- the output of one node line feeds into the next. A typical RAG pipeline has two node lines:

1. **retrieve_node_line** -- Query expansion, retrieval, augmentation, reranking, filtering, and compression.
2. **post_retrieve_node_line** -- Prompt construction and answer generation.

### The 8 Node Types

AutoRAG evaluates 8 node types in order:

| Node Type | Purpose |
|---|---|
| **query_expansion** | Transform query for better retrieval (HyDE, decompose, multi-query) |
| **retrieval** | Find relevant chunks (BM25, vector, hybrid_rrf, hybrid_cc) |
| **passage_augmenter** | Add surrounding context to retrieved passages |
| **passage_reranker** | Re-score and re-order retrieved chunks |
| **passage_filter** | Remove irrelevant passages by threshold or percentile |
| **passage_compressor** | Compress passage content before generation |
| **prompt_maker** | Construct the prompt from context and query |
| **generator** | Generate the final answer from the prompt |

### Pass Modules

Every node type (except generator) has a `pass_*` variant -- for example, `pass_query_expansion`, `pass_reranker`, `pass_compressor`. These modules skip the processing step entirely, passing input through unchanged. AutoRAG includes them to test whether each node actually helps on your data. If `pass_reranker` outperforms `flashrank_reranker`, it means reranking hurts quality -- a valuable finding that simplifies your pipeline.

### Strategy and Greedy Optimization

Each node has a `strategy` section that defines which metrics to optimize. AutoRAG uses a **greedy algorithm**:

1. Run all module configurations at the current node.
2. Score each using the strategy metrics.
3. Select the best-performing configuration.
4. Pass its output to the next node.
5. Repeat.

This is greedy because each node is optimized independently -- it does not search for the globally optimal combination across all nodes.

### Optimization Strategies

AutoRAG supports three methods for combining metric scores:

- **mean** (default) -- Averages metric scores across all QA pairs. Best for balanced optimization.
- **rank** -- Uses reciprocal rank fusion to combine metric rankings. Robust when metrics have different scales.
- **normalize_mean** -- Normalizes scores to [0, 1] before averaging. Useful when combining metrics with different ranges.

### Speed Threshold

The `speed_threshold` parameter adds a latency constraint to node optimization. Modules slower than the threshold (in seconds per query) are excluded, even if they score higher on quality metrics. This is essential for production systems with response time SLAs.

### Parameter Grid Search

When a module parameter is a list, AutoRAG generates all combinations. For example, if `prompt` has 2 templates and `model` has 2 models, AutoRAG tests all 4 combinations (2 x 2).

## The Configuration File

Here is the full `config.yaml` used in this lesson, covering all 8 node types:

```yaml
vectordb:
  - name: default
    db_type: chroma
    client_type: persistent
    path: ./chroma_db
    embedding_model: openai
    collection_name: openai

node_lines:
- node_line_name: retrieve_node_line
  nodes:
    - node_type: query_expansion
      strategy:
        metrics: [retrieval_f1, retrieval_recall, retrieval_precision]
      modules:
        - module_type: pass_query_expansion
        - module_type: hyde
          generator_module_type: llama_index_llm
          llm: ollama
          model: gemma4:e2b
    - node_type: lexical_retrieval
      strategy:
        metrics: [retrieval_f1, retrieval_recall, retrieval_precision]
      top_k: 3
      modules:
        - module_type: bm25
    - node_type: passage_augmenter
      strategy:
        metrics: [retrieval_f1, retrieval_recall]
      modules:
        - module_type: pass_passage_augmenter
        - module_type: prev_next_augmenter
          num_prev: 1
          num_next: 1
    - node_type: passage_reranker
      strategy:
        metrics: [retrieval_f1, retrieval_recall]
      top_k: 3
      modules:
        - module_type: pass_reranker
        - module_type: flashrank_reranker
    - node_type: passage_filter
      strategy:
        metrics: [retrieval_f1, retrieval_recall]
      modules:
        - module_type: pass_passage_filter
        - module_type: similarity_threshold_cutoff
          threshold: 0.5
    - node_type: passage_compressor
      strategy:
        metrics: [retrieval_token_f1, retrieval_token_recall, retrieval_token_precision]
      modules:
        - module_type: pass_compressor
        - module_type: tree_summarize
          llm: ollama
          model: gemma4:e2b

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

- **vectordb** configures a persistent Chroma database for vector storage.
- **retrieve_node_line**: Tests query expansion (pass vs HyDE), BM25 retrieval, passage augmentation (pass vs prev/next), reranking (pass vs FlashRank), filtering (pass vs similarity cutoff), and compression (pass vs tree summarize).
- **post_retrieve_node_line**: Tests 2 prompt templates, then runs generation with the LLM.
- **Pass modules** in each node let AutoRAG determine whether the processing step helps or hurts.
- **Strategy metrics**: Retrieval nodes optimize for F1/recall/precision. Compressor nodes use token-level metrics. Generation nodes optimize for BLEU/ROUGE.

## Step-by-Step

### Step 1: Load Configuration

Load the YAML file and inspect the top-level structure:

```python
with open("config.yaml") as f:
    config = yaml.safe_load(f)
```

### Step 2: Understand the Hierarchy

Walk the config hierarchy: vectordb + node_lines -> nodes -> strategy + modules.

### Step 3: Vector Database Configuration

Examine the vectordb section and understand how it configures the storage backend.

### Step 4: Trace Node Lines

Follow how data flows through all 8 node types across the two pipeline stages.

### Step 5: Pass Modules

Understand why pass modules are included and what it means when they win.

### Step 6: Count Configurations

Calculate how many module configurations AutoRAG will test at each node.

### Step 7: Understand Strategy

Learn how the greedy algorithm selects the best module at each node.

### Step 8: Optimization Strategies

Learn about the three methods for combining metric scores: mean, rank, and normalize_mean.

### Step 9: Speed Threshold

Understand how to constrain optimization with latency requirements.

### Step 10: Review Available Metrics

See all retrieval, generation, compressor, and RAGAS metrics with descriptions.

### Step 11: Validate the Config

Check that every node_line has a name, every node has a type and modules, and every module has a type.

## Metrics Reference

| Metric | Category | Description |
|---|---|---|
| retrieval_f1 | Retrieval | Harmonic mean of retrieval precision and recall |
| retrieval_recall | Retrieval | Fraction of relevant documents retrieved |
| retrieval_precision | Retrieval | Fraction of retrieved documents that are relevant |
| retrieval_ndcg | Retrieval | Normalized discounted cumulative gain |
| retrieval_mrr | Retrieval | Mean reciprocal rank |
| retrieval_map | Retrieval | Mean average precision across recall levels |
| bleu | Generation | N-gram overlap with reference text |
| meteor | Generation | Alignment-based similarity with synonyms/stemming |
| rouge | Generation | Recall-oriented n-gram overlap (ROUGE-L) |
| sem_score | Generation | Cosine similarity of sentence embeddings |
| bert_score | Generation | Token-level semantic similarity using BERT |
| g_eval | Generation | LLM-as-judge (coherence, consistency, fluency, relevance) |
| faithfulness | Generation | Whether the answer is supported by retrieved context |
| retrieval_token_f1 | Compressor | Token-level F1 between compressed and original passages |
| retrieval_token_recall | Compressor | Token-level recall of compressed passages |
| retrieval_token_precision | Compressor | Token-level precision of compressed passages |
| context_precision | RAGAS | Relevance of retrieved context (no ground truth needed) |

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
Top-level keys: ['vectordb', 'node_lines']
Vector DB configs: 1
Number of node_lines: 2

============================================================
Step 2: Understand the structure
============================================================
Top-level sections:
  vectordb            (vector database configuration)
  node_lines          (pipeline stages)

AutoRAG config follows a 4-level hierarchy:

  node_lines          (top level -- pipeline stages)
    -> nodes          (processing steps within a stage)
      -> strategy     (how to evaluate and select the best)
      -> modules      (implementations to compare)

  node_line [0]: retrieve_node_line
    node [0]: query_expansion
      strategy metrics: ['retrieval_f1', 'retrieval_recall', 'retrieval_precision']
      modules (2):
        - pass_query_expansion
        - hyde
    node [1]: lexical_retrieval
      ...
    node [2]: passage_augmenter
      ...
    node [3]: passage_reranker
      ...
    node [4]: passage_filter
      ...
    node [5]: passage_compressor
      ...

  node_line [1]: post_retrieve_node_line
    node [0]: prompt_maker
      ...
    node [1]: generator
      ...

============================================================
Step 3: Vector database configuration
============================================================
The top-level 'vectordb' section configures the vector database
backend used by retrieval nodes (vector, hybrid_rrf, hybrid_cc).

  Name:            default
  DB type:         chroma
  Client type:     persistent
  Path:            ./chroma_db
  Embedding model: openai
  Collection:      openai

Supported vector databases:
  Chroma, Milvus, Weaviate, Pinecone, Couchbase, Qdrant

============================================================
Step 4: Node lines -- pipeline stages
============================================================
...

============================================================
Step 5: Pass modules -- testing whether nodes help
============================================================
Every node type has a pass_* variant (except generator):

  pass_query_expansion         Skips query expansion -- uses the original query as-is
  pass_passage_augmenter       Skips augmentation -- uses retrieved passages without adding context
  pass_reranker                Skips reranking -- keeps the retriever's original ordering
  pass_passage_filter          Skips filtering -- keeps all retrieved passages
  pass_compressor              Skips compression -- passes full passage text to the prompt

Why include pass modules?
  AutoRAG tests whether each processing step actually helps.
  ...

============================================================
Step 6: Count configurations to test
============================================================
...
  Grand total: 13 module configurations to evaluate

============================================================
Step 7: Strategy -- greedy optimization
============================================================
...

============================================================
Step 8: Optimization strategies
============================================================
  mean (default)               Averages metric scores across all QA pairs.
  rank                         Uses reciprocal rank fusion to combine metric rankings.
  normalize_mean               Normalizes scores to [0,1] before averaging.

============================================================
Step 9: Speed threshold
============================================================
  The speed_threshold parameter adds a latency constraint to node
  optimization. Modules slower than the threshold are excluded...

============================================================
Step 10: Metrics reference
============================================================
  Metric                       Category     Description
  ...
  Total: 17 metrics available

============================================================
Step 11: Validate configuration
============================================================
Validation PASSED -- all required keys present

  Config summary:
    Node lines: 2
    Total nodes: 8
    Total modules: 13

============================================================
Done! You now understand AutoRAG's configuration format.
============================================================
```

## Key Takeaways

- AutoRAG configs have two top-level sections: **vectordb** (database backend) and **node_lines** (pipeline stages).
- The pipeline has **8 node types** evaluated in order: query expansion, retrieval, augmentation, reranking, filtering, compression, prompt making, and generation.
- **Pass modules** test whether each processing step helps -- skipping a node is sometimes the best choice.
- The **strategy** section defines metrics and supports three optimization methods: mean, rank, and normalize_mean.
- **speed_threshold** constrains optimization to meet latency requirements.
- **17 metrics** are available across retrieval, generation, compressor, and RAGAS categories.

## Next Steps

In **L1-M3.2 -- Running and Monitoring Experiments**, you will use this configuration to run an actual AutoRAG experiment, monitor progress, and interpret the results.
