# AutoRAG Tutorial Project

## Purpose

This is a two-level tutorial for AutoRAG — the open-source framework that applies AutoML-style automation to RAG pipeline optimization. You provide evaluation data, AutoRAG explores combinations of chunking strategies, embedding models, retrievers, rerankers, and generators, then identifies the best configuration.

The tutorial is structured in two progressive levels:
- **Level 1 — Essentials**: Understand AutoRAG, create evaluation data, run experiments, find the optimal RAG pipeline (~4.5-5.5 hours).
- **Level 2 — Practitioner**: Advanced retrieval strategies, custom modules, integration with OpenShift AI (~3.5-4 hours).

The full syllabus lives in `syllabus.md` — always consult it for module structure, lesson topics, deliverables, and time estimates before creating or modifying any lesson.

## Technical Stack

- **AutoRAG**: Latest (`autorag` Python package)
- **Python**: 3.10+
- **Package manager**: `uv` (every lesson is a standalone `uv` project)
- **LLM**: Ollama (primary, local dev) or vLLM
- **Model**: `ollama/gemma4:e2b` (default for generation)
- **Embedding models**: sentence-transformers, BGE, nomic-embed-text, OpenAI, any HuggingFace model
- **Vector DBs**: FAISS, Chroma, Qdrant (AutoRAG supports many)
- **Data format**: Parquet (QA datasets and corpus)

## Project Layout

```
syllabus.md                         # Master syllabus — the source of truth
infra/                              # Local setup: Ollama + AutoRAG verification
  main.py                           #   Verification script
  pyproject.toml                    #   uv project with autorag dependency
  README.md                         #   Setup guide
tutorial/
  level_1/                          # Level 1: Essentials
    M1_fundamentals/
      1_what_is_autorag/
      2_installing_project_setup/
    M2_evaluation_data/
      1_creating_qa_datasets/
      2_preparing_corpus/
    M3_running_experiments/
      1_configuration_yaml/
      2_running_monitoring/
      3_analyzing_deploying/
  level_2/                          # Level 2: Practitioner
    M1_advanced_optimization/
      1_advanced_retrieval/
      2_embedding_comparison/
      3_custom_metrics/
    M2_integration_production/
      1_custom_modules/
      2_autorag_to_openshift/
```

Each lesson is a self-contained directory:
```
N_lesson_name/
  pyproject.toml        # uv project — declares dependencies
  main.py               # Working code (the lesson implementation)
  README.md             # Lesson guide with explanation, steps, expected output
  .gitignore            # Ignore .venv, __pycache__
```

## Starting Infrastructure

```bash
# Start Ollama (if not already running)
ollama serve &

# Pull the model
ollama pull gemma4:e2b

# Verify setup
cd infra
uv sync
uv run python main.py
```

| Service | URL |
|---------|-----|
| Ollama | http://localhost:11434 |

## Running a Lesson

```bash
cd tutorial/<level>/<module>/<lesson>
uv sync
uv run python main.py
```

## Key Commands

- `uv init` — scaffold a new lesson project
- `uv add <package>` — add a dependency
- `uv run python main.py` — run the lesson code
- `autorag evaluate --config config.yaml --qa qa.parquet --corpus corpus.parquet` — run evaluation
- `autorag dashboard --trial_dir results/` — launch results dashboard
- `autorag deploy --trial_dir results/` — deploy optimal pipeline as FastAPI server

## Rules

Modular instructions are in `.claude/rules/`. Read them — they cover:
- `tutorial-structure.md` — two-level layout and file conventions
- `coding-standards.md` — Python style for tutorial code
- `autorag-patterns.md` — AutoRAG APIs, config YAML, and patterns to use
- `references.md` — where to find source code, docs, and research paper
- `lesson-content.md` — how to write README.md guides
