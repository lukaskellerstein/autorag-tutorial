---
globs: ["tutorial/**"]
---

# Tutorial Structure Rules

## Two-Level Architecture

The tutorial is organized in two progressive levels:

- **`tutorial/level_1/`** — Essentials: understand AutoRAG, create evaluation data, run experiments (~5-6 hours)
- **`tutorial/level_2/`** — Practitioner: advanced optimization, custom modules, OpenShift AI integration (~4.25-5 hours)

Always consult `syllabus.md` for the full module/lesson breakdown.

## Lesson Directory Convention

Every lesson lives in `tutorial/<level>/<module>/<lesson>/` and contains exactly:

1. **`pyproject.toml`** — standalone `uv` project. Use `[project]` with `name`, `version`, `description`, `requires-python`, and `dependencies`.
2. **`main.py`** — the working lesson code. This is the primary deliverable.
3. **`README.md`** — lesson guide (see `lesson-content.md` rule for format).
4. **`.gitignore`** — always ignore: `.venv/`, `__pycache__/`, `*.pyc`, `.python-version`, `results/`.

Some lessons may also include:
- **`config.yaml`** — AutoRAG experiment configuration
- **`data/`** — sample documents, QA parquet files, or corpus files

## Directory Structure

```
syllabus.md                         # Master syllabus — source of truth
tutorial/
  level_1/
    M1_fundamentals/
      1_what_is_autorag/
      2_installing_project_setup/
    M2_evaluation_data/
      1_parsing_corpus_creation/
      2_creating_qa_datasets/
    M3_running_experiments/
      1_configuration_yaml/
      2_running_monitoring/
      3_analyzing_deploying/
  level_2/
    M1_advanced_optimization/
      1_intermediate_pipeline_nodes/
      2_advanced_retrieval/
      3_embedding_comparison/
      4_custom_metrics/
    M2_integration_production/
      1_custom_modules/
      2_autorag_to_openshift/
```

## pyproject.toml Template

```toml
[project]
name = "autorag-tutorial-L<level>-M<module>-<lesson>"
version = "0.1.0"
description = "<Lesson title from syllabus>"
requires-python = ">=3.10"

[project.dependencies]
autorag = ">=0.3"
# Add lesson-specific deps here
```

## .gitignore Template

```
.venv/
__pycache__/
*.pyc
.python-version
results/
```

## Principles

- Each lesson must be fully self-contained — a user should be able to `cd` into it, run `uv sync && uv run python main.py`, and see results.
- AutoRAG lessons that run evaluations should include sample data (small QA/corpus parquet files) or generate them programmatically.
- Print meaningful output to the console so the user sees what's happening.
- Keep `main.py` under ~200 lines. If a lesson needs helper code, put it in a separate module within the same directory.
- Level 1 lessons should be concise and focused on one AutoRAG concept.
- Level 2 lessons can be longer and build multi-step production scenarios.
- Include `config.yaml` files for lessons that run AutoRAG evaluations.
