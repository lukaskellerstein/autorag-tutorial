# Running AutoRAG Locally

This guide sets up AutoRAG for local development on macOS.

## Architecture

```
┌──────────────────────────────────────────────────┐
│  Your Python code (tutorial lessons)             │
│  Uses: autorag CLI + Python API                  │
└──────────────┬───────────────────────────────────┘
               │
┌──────────────▼───────────────────────────────────┐
│  AutoRAG (Python package)                        │
│  - Evaluates RAG pipeline configurations         │
│  - Greedy optimization across nodes              │
│  - CLI: evaluate, dashboard, deploy              │
└──────┬──────────────────────────────────────┬────┘
       │                                      │
┌──────▼──────────┐                ┌──────────▼────┐
│  Ollama (native)│                │  Vector DB    │
│  Port: 11434    │                │  (in-process) │
│  Model:         │                │  FAISS or     │
│  gemma4:e2b     │                │  Chroma       │
│  Runs on host   │                │               │
│  (Apple Silicon │                │  (AutoRAG     │
│   GPU access)   │                │   manages)    │
└─────────────────┘                └───────────────┘
```

**Why Ollama runs natively:** On macOS with Apple Silicon, Ollama runs outside containers to access the GPU. AutoRAG calls Ollama via LiteLLM for generation and evaluation.

## Prerequisites

1. **Ollama** — local LLM inference
   ```bash
   brew install ollama
   ```

2. **uv** — Python package manager
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

## Quick Start

### 1. Start Ollama and pull the model

```bash
# Start Ollama (if not already running)
ollama serve &

# Pull gemma4:e2b (~1.8 GB)
ollama pull gemma4:e2b
```

### 2. Install AutoRAG and verify

```bash
cd infra
uv sync
uv run python main.py
```

You should see all checks pass:
```
  Ollama       : OK
  Inference    : OK
  AutoRAG      : OK
  Data deps    : OK

All checks passed! AutoRAG is ready for tutorial lessons.
```

## How AutoRAG Uses Ollama

AutoRAG uses LiteLLM under the hood. In your `config.yaml`, reference Ollama models with the `ollama/` prefix:

```yaml
node_lines:
  - node_type: generator
    modules:
      - module_type: llm
        llm: "ollama/gemma4:e2b"
```

Make sure Ollama is running before starting any AutoRAG evaluation.

## Key Commands

```bash
# Run an evaluation
autorag evaluate --config config.yaml --qa qa.parquet --corpus corpus.parquet

# View results in the dashboard
autorag dashboard --trial_dir results/

# Deploy the optimal pipeline as a FastAPI server
autorag deploy --trial_dir results/

# Generate QA data from a corpus
autorag generate_qa --corpus corpus.parquet --llm "ollama/gemma4:e2b"
```

## Data Format

AutoRAG requires two Parquet files:

### qa.parquet — Evaluation QA pairs
| Column | Type | Description |
|--------|------|-------------|
| `qid` | str | Unique question ID |
| `query` | str | The question |
| `retrieval_gt` | list[list[str]] | Ground truth chunk IDs |
| `generation_gt` | list[str] | Ground truth answers |

### corpus.parquet — Document corpus
| Column | Type | Description |
|--------|------|-------------|
| `doc_id` | str | Unique document ID |
| `contents` | str | Document text |
| `metadata` | dict | Optional metadata |

## Service URLs

| Service | URL | Purpose |
|---------|-----|---------|
| Ollama | http://localhost:11434 | LLM inference backend |
| AutoRAG Dashboard | http://localhost:8501 (when running) | Results visualization |
| AutoRAG Deploy | http://localhost:8000 (when deployed) | Optimal pipeline API |

## Troubleshooting

### Ollama not running
```bash
# Start Ollama
ollama serve

# Check it's running
curl http://localhost:11434/api/tags
```

### Model not found
```bash
# List available models
ollama list

# Pull the model
ollama pull gemma4:e2b
```

### AutoRAG import errors
```bash
# Ensure you're using the right Python version
uv python install 3.12
uv python pin 3.12
uv sync
```

### Evaluation runs slowly
- Use a smaller QA dataset for initial experiments (~50 pairs)
- Reduce the number of configurations in `config.yaml`
- The `gemma4:e2b` model is small and fast — larger models will be slower

### Out of memory
- Close other GPU-intensive applications
- Use `gemma4:e2b` (2B effective params) instead of larger models
- Reduce `top_k` values in retrieval configuration

## Next Steps

Once everything is verified, start with the tutorial:
```bash
cd tutorial/level_1/M1_fundamentals/1_what_is_autorag/
uv sync
uv run python main.py
```
