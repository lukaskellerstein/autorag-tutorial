---
globs: ["tutorial/**/*.py"]
---

# Python Coding Standards for Tutorial Code

## Style

- Target Python 3.10+. Use type hints on function signatures.
- Use `if __name__ == "__main__":` guard in every `main.py`.
- Use `asyncio.run(main())` for async lessons if needed.
- Import order: stdlib, third-party, local — separated by blank lines.
- Use f-strings for string formatting.
- Keep functions short and focused — this is tutorial code, readability is paramount.

## Complexity by Level

- **Level 1**: Simple, single-concept scripts. Minimal abstraction. One main function.
- **Level 2**: Multi-step projects. Helper functions OK. Can import from local modules.

## AutoRAG Setup

AutoRAG is used primarily via its CLI and Python API:

```python
import autorag
from autorag.evaluator import Evaluator

evaluator = Evaluator(
    qa_data_path="qa.parquet",
    corpus_data_path="corpus.parquet",
    project_dir="./results",
)
evaluator.start_trial("config.yaml")
```

## Data Preparation with Pandas/PyArrow

QA and corpus data use Parquet format:

```python
import pandas as pd

qa_df = pd.DataFrame({
    "qid": ["q1", "q2"],
    "query": ["What is RAG?", "How does chunking work?"],
    "retrieval_gt": [["doc1_chunk3"], ["doc2_chunk1"]],
    "generation_gt": [["RAG is..."], ["Chunking splits..."]],
})
qa_df.to_parquet("qa.parquet", index=False)
```

## LLM Configuration via LiteLLM

AutoRAG uses LiteLLM for model access. For local Ollama:

```yaml
# In config.yaml
- module_type: llm
  llm: "ollama/gemma4:e2b"
```

## Error Handling

- Check that required data files exist before running evaluations.
- Print clear error messages if Ollama/vLLM is not running.
- Do not silently swallow exceptions — this is educational code.

## Dependencies

- Use `uv add` to add dependencies, never `pip install`.
- Common dependencies by topic:
  - All lessons: `autorag`
  - Data preparation: `pandas`, `pyarrow`
  - QA generation: `autorag` (built-in generator)
  - Embedding models: `sentence-transformers`
  - LLM access: handled via `litellm` (bundled with autorag)
  - Vector DBs: `faiss-cpu`, `chromadb`, `qdrant-client` (as needed)

## Console Output

Print section headers and results so users can follow along:

```python
print("=" * 60)
print("Step 1: Loading evaluation data")
print("=" * 60)
```

Print key results inline — metric scores, best configurations, comparisons.
